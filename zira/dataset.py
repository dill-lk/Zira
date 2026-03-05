"""
Data pipeline for Zira.

Phase 1 – Foundation pre-training
  Supported: wikitext-103, openwebtext (subset)
  * Downloads via HuggingFace datasets
  * Pre-tokenises fully and saves to disk
  * Chunks into fixed-length sequences

Phase 2 – Chat SFT
  Supported: tatsu-lab/alpaca, HuggingFaceH4/ultrachat_200k
  * Formats as "User: ...\nAssistant: ..."
  * Concatenates into a single token stream
  * Chunked into fixed-length sequences
  * Uses next-token prediction objective
"""

import os
import json
from typing import List, Optional

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_token_stream(
    token_ids: List[int],
    seq_len: int,
) -> List[List[int]]:
    """Split a flat list of token ids into non-overlapping chunks of seq_len."""
    chunks = []
    for start in range(0, len(token_ids) - seq_len, seq_len):
        chunks.append(token_ids[start: start + seq_len + 1])  # +1 for label shift
    return chunks


# ---------------------------------------------------------------------------
# Pre-train dataset
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """
    Foundation pre-training dataset.

    Automatically downloads wikitext-103 or openwebtext,
    pre-tokenises, chunks into seq_len+1 sequences, and caches to disk.
    The +1 accounts for the label shift (predict token t+1 at position t).
    """

    SUPPORTED = {
        "wikitext-103": ("wikitext", "wikitext-103-raw-v1"),
        "openwebtext":  ("openwebtext", None),
    }

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        dataset_name: str = "wikitext-103",
        split: str = "train",
        cache_dir: str = "data_cache",
        max_samples: Optional[int] = None,
    ):
        self.seq_len = seq_len
        cache_file = os.path.join(
            cache_dir,
            f"{dataset_name.replace('/', '_')}_{split}_seq{seq_len}.pt",
        )
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(cache_file):
            print(f"[PretrainDataset] Loading cached chunks from {cache_file}")
            self.chunks = torch.load(cache_file)
        else:
            print(f"[PretrainDataset] Building cache for {dataset_name}/{split} …")
            self.chunks = self._build(tokenizer, dataset_name, split, max_samples)
            torch.save(self.chunks, cache_file)
            print(f"[PretrainDataset] Saved {len(self.chunks)} chunks → {cache_file}")

    # ------------------------------------------------------------------
    def _build(self, tokenizer, dataset_name, split, max_samples) -> torch.Tensor:
        from datasets import load_dataset  # type: ignore

        if dataset_name not in self.SUPPORTED:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Choose from {list(self.SUPPORTED)}"
            )
        hf_name, config_name = self.SUPPORTED[dataset_name]
        if config_name:
            ds = load_dataset(hf_name, config_name, split=split)
        else:
            ds = load_dataset(hf_name, split=split)

        text_key = "text"

        # Flatten into one big token stream
        all_ids: List[int] = []
        for i, example in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            text = example[text_key].strip()
            if not text:
                continue
            all_ids.extend(tokenizer.encode(text, add_eos=True))

        chunks = _chunk_token_stream(all_ids, self.seq_len)
        # Convert to tensor: (n_chunks, seq_len+1)
        return torch.tensor(chunks, dtype=torch.long)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        chunk = self.chunks[idx]          # (seq_len + 1,)
        input_ids = chunk[:-1]            # (seq_len,)
        labels    = chunk[1:]             # (seq_len,)
        return input_ids, labels


# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning (chat) dataset.

    Supported: tatsu-lab/alpaca, HuggingFaceH4/ultrachat_200k.
    Formats each example as:

        User: <instruction>\nAssistant: <response><eos>

    Concatenates the full token stream and chunks it.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        dataset_name: str = "tatsu-lab/alpaca",
        split: str = "train",
        cache_dir: str = "data_cache",
        max_samples: Optional[int] = None,
    ):
        self.seq_len = seq_len
        safe_name = dataset_name.replace("/", "_")
        cache_file = os.path.join(
            cache_dir,
            f"sft_{safe_name}_{split}_seq{seq_len}.pt",
        )
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(cache_file):
            print(f"[SFTDataset] Loading cached chunks from {cache_file}")
            self.chunks = torch.load(cache_file)
        else:
            print(f"[SFTDataset] Building cache for {dataset_name}/{split} …")
            self.chunks = self._build(tokenizer, dataset_name, split, max_samples)
            torch.save(self.chunks, cache_file)
            print(f"[SFTDataset] Saved {len(self.chunks)} chunks → {cache_file}")

    # ------------------------------------------------------------------
    def _build(self, tokenizer, dataset_name, split, max_samples) -> torch.Tensor:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(dataset_name, split=split)

        all_ids: List[int] = []
        for i, example in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            text = _format_sft_example(example, dataset_name)
            if not text:
                continue
            all_ids.extend(tokenizer.encode(text, add_eos=True))

        chunks = _chunk_token_stream(all_ids, self.seq_len)
        return torch.tensor(chunks, dtype=torch.long)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# SFT formatting helpers
# ---------------------------------------------------------------------------

def _format_sft_example(example: dict, dataset_name: str) -> str:
    """Format a dataset example into a User/Assistant dialogue string."""
    if "alpaca" in dataset_name:
        instruction = example.get("instruction", "").strip()
        inp         = example.get("input", "").strip()
        output      = example.get("output", "").strip()
        if inp:
            user_text = f"{instruction}\n{inp}"
        else:
            user_text = instruction
        return f"User: {user_text}\nAssistant: {output}"

    elif "ultrachat" in dataset_name:
        # ultrachat_200k stores messages as a list of {"role", "content"} dicts
        messages = example.get("messages", [])
        parts = []
        for msg in messages:
            role    = msg.get("role", "").strip()
            content = msg.get("content", "").strip()
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n".join(parts)

    else:
        # Generic fallback: look for common field names
        for key in ("text", "content", "prompt"):
            if key in example:
                return str(example[key]).strip()
        return ""
