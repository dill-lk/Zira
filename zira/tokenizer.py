"""
Tokenizer wrapper.
Wraps a SentencePiece or HuggingFace tokenizer into a consistent API
used by the rest of Zira.

Priority:
  1. Load from a local SentencePiece model file (*.model)
  2. Load from a HuggingFace tokenizer directory / identifier
     (requires `transformers` to be installed)

If neither is available, the caller must supply an already-loaded
tokenizer via ZiraTokenizer(tokenizer=...).
"""

import os
from typing import List, Optional, Union


class ZiraTokenizer:
    """
    Thin wrapper providing:
        .encode(text)  → List[int]
        .decode(ids)   → str
        .bos_id        → int
        .eos_id        → int
        .pad_id        → int
        .vocab_size    → int
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hf_name: Optional[str] = None,
        tokenizer=None,
    ):
        self._tok = None

        if tokenizer is not None:
            # Pre-built tokenizer passed in directly
            self._tok = tokenizer
            self._backend = "external"

        elif model_path is not None and os.path.isfile(model_path):
            # SentencePiece
            try:
                import sentencepiece as spm
            except ImportError as e:
                raise ImportError(
                    "sentencepiece is required for .model files. "
                    "Install with: pip install sentencepiece"
                ) from e
            sp = spm.SentencePieceProcessor()
            sp.Load(model_path)
            self._tok = sp
            self._backend = "sentencepiece"

        elif hf_name is not None:
            # HuggingFace tokenizer
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "transformers is required for HuggingFace tokenizers. "
                    "Install with: pip install transformers"
                ) from e
            self._tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
            self._backend = "huggingface"

        else:
            raise ValueError(
                "Provide one of: model_path (SentencePiece *.model), "
                "hf_name (HuggingFace identifier), or tokenizer= (pre-built)."
            )

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        if self._backend == "sentencepiece":
            return self._tok.GetPieceSize()
        return len(self._tok)

    @property
    def bos_id(self) -> int:
        if self._backend == "sentencepiece":
            return self._tok.bos_id()
        return self._tok.bos_token_id or 1

    @property
    def eos_id(self) -> int:
        if self._backend == "sentencepiece":
            return self._tok.eos_id()
        return self._tok.eos_token_id or 2

    @property
    def pad_id(self) -> int:
        if self._backend == "sentencepiece":
            return self._tok.pad_id()
        pid = self._tok.pad_token_id
        return pid if pid is not None else self.eos_id

    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        if self._backend == "sentencepiece":
            ids: List[int] = self._tok.EncodeAsIds(text)
        else:
            ids = self._tok.encode(text, add_special_tokens=False)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        if self._backend == "sentencepiece":
            return self._tok.DecodeIds(ids)
        return self._tok.decode(ids, skip_special_tokens=True)
