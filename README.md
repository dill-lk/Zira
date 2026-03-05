<div align="center">

# 🔥 Zira

**Production-grade decoder-only Transformer — built from scratch for Google Colab Free TPU**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dill-lk/Zira/blob/main/colab_launcher.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

</div>

---

Zira is a clean, modular Transformer training system optimised specifically for **Google Colab Free TPU (v5e-1)** via `torch_xla`. It supports foundation pre-training and supervised fine-tuning (SFT/chat), with a single-click Colab launcher for zero-setup experimentation.

## ✨ Highlights

- **No HuggingFace Trainer / AutoModel** — every component is implemented from scratch
- **TPU-first** — `xm.xla_device()`, `xm.optimizer_step()`, `MpDeviceLoader`, BF16, static shapes
- **GPU / CPU fallback** — runs anywhere without code changes
- **Modular architecture** — swap attention, FFN, or positional encoding independently
- **Two-phase training** — foundation pre-training → chat SFT in one repo
- **Resume support** — checkpoints save to Google Drive across Colab sessions

---

## 🚀 Quick Start

### Option 1 — One-click Colab (recommended)

Click the badge above, or use this direct link:

> **[▶ Open `colab_launcher.ipynb` in Google Colab](https://colab.research.google.com/github/dill-lk/Zira/blob/main/colab_launcher.ipynb)**

The notebook will:
1. Detect TPU / GPU / CPU automatically
2. Install `torch_xla` and all dependencies
3. Clone this repository
4. (Optionally) mount Google Drive for persistent checkpoints
5. Run pre-training and SFT with a single cell each

### Option 2 — Local / script

```bash
# Clone
git clone https://github.com/dill-lk/Zira.git
cd Zira

# Install dependencies
pip install torch datasets transformers sentencepiece

# Pre-train Zira-Micro on wikitext-103
python -m zira.train_pretrain \
    --model micro \
    --dataset wikitext-103 \
    --tokenizer huggyllama/llama-7b \
    --batch_size 8 \
    --steps 50000

# SFT with Alpaca (load pretrained checkpoint)
python -m zira.train_sft \
    --model micro \
    --pretrain_ckpt checkpoints/ckpt_step0050000.pt \
    --dataset tatsu-lab/alpaca

# Generate text
python -m zira.generate \
    --model micro \
    --checkpoint checkpoints/ckpt_step0050000.pt \
    --prompt "Once upon a time" \
    --max_new_tokens 200 \
    --temperature 0.8
```

---

## 🏗️ Architecture

Zira implements a **decoder-only Transformer** with modern improvements:

| Component | Implementation |
|-----------|---------------|
| Positional encoding | Rotary Positional Embeddings (RoPE) |
| Attention | Grouped Query Attention (GQA) |
| Feedforward | SwiGLU (`silu(gate) * up → down`) |
| Normalisation | RMSNorm (pre-norm layout) |
| Residual | `x = x + Attn(Norm(x))`, `x = x + FFN(Norm(x))` |
| Weight tying | `lm_head.weight = embed_tokens.weight` |
| Precision | BF16 on TPU, BF16 autocast on GPU |
| Biases | None (all linear layers are bias-free) |

---

## 📐 Model Families

| Model | Layers | d_model | Heads (KV) | FFN | Seq Len | ~Params | Target Use |
|-------|--------|---------|------------|-----|---------|---------|------------|
| **Zira-Micro** | 12 | 512 | 8 (4) | 2048 | 1024 | ~64M | Fast debug (<2 hrs on TPU) |
| **Zira-Small** | 16 | 640 | 10 (5) | 2560 | 1024 | ~119M | Full SFT in one Colab session |
| **Zira-Compact** | 16 | 768 | 12 (4) | 3072 | 1024 | ~163M | Balanced pretrain + SFT |
| **Zira-Base** | 24 | 1024 | 16 (8) | 4096 | 2048 | ~410M | Foundation pretrain across sessions |
| **Zira-Pro** | 32 | 1280 | 20 (5) | 5120 | 2048 | ~801M | Near full TPU utilisation |

Each config prints its parameter count on initialisation:

```python
from zira.config import ZiraMicro
cfg = ZiraMicro()
# [ZiraConfig] Zira-Micro | layers=12, d_model=512, heads=8(kv=4), ffn=2048, seq=1024 | ~63.6M params
```

---

## 📁 Repository Structure

```
Zira/
│
├── colab_launcher.ipynb      # ← One-click Colab notebook
│
└── zira/
    ├── config.py             # Model family definitions & ZiraConfig dataclass
    ├── model.py              # ZiraModel — full decoder-only Transformer
    ├── rope.py               # Rotary Positional Embeddings
    ├── attention.py          # Grouped Query Attention (GQA) + causal mask
    ├── ffn.py                # SwiGLU feedforward
    ├── transformer_block.py  # Pre-Norm block (RMSNorm + GQA + SwiGLU)
    ├── tokenizer.py          # Tokenizer wrapper (SentencePiece / HuggingFace)
    ├── dataset.py            # Pre-tokenised datasets (pretrain + SFT)
    ├── utils.py              # LR scheduler, checkpointing, throughput tracker
    ├── train_pretrain.py     # Phase 1 — foundation pre-training loop
    ├── train_sft.py          # Phase 2 — supervised fine-tuning loop
    └── generate.py           # Autoregressive text generation
```

---

## 🗄️ Data Pipeline

### Phase 1 — Foundation Pre-training

| Dataset | HuggingFace ID | Notes |
|---------|---------------|-------|
| WikiText-103 | `wikitext / wikitext-103-raw-v1` | Default; fast download |
| OpenWebText | `openwebtext` | Larger; use with `--max_samples` for Colab |

The full corpus is **pre-tokenised once and cached to disk**. Subsequent runs load from cache instantly. No tokenisation happens inside `__getitem__`.

### Phase 2 — Chat SFT

| Dataset | HuggingFace ID | Format |
|---------|---------------|--------|
| Alpaca | `tatsu-lab/alpaca` | Instruction → Response |
| UltraChat 200k | `HuggingFaceH4/ultrachat_200k` | Multi-turn dialogue |

All examples are formatted as:

```
User: <instruction>
Assistant: <response>
```

…concatenated into a single token stream and chunked. Standard next-token prediction loss.

---

## 🔧 Configuration Reference

All hyperparameters live in `ZiraConfig`. Pass overrides to any factory:

```python
from zira.config import ZiraMicro, get_config

# Override specific fields
cfg = ZiraMicro(lr=1e-4, batch_size=16, max_steps=20_000)

# Or use the registry
cfg = get_config("base", lr=2e-4, seq_len=1024)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 32000 | Vocabulary size |
| `layers` | 12 | Number of Transformer blocks |
| `d_model` | 512 | Hidden dimension |
| `heads` | 8 | Query attention heads |
| `kv_heads` | 4 | Key/value heads (GQA; set `= heads` to disable) |
| `head_dim` | 64 | Dimension per head (`d_model = heads × head_dim`) |
| `ffn_hidden` | 2048 | FFN intermediate dimension |
| `seq_len` | 1024 | Static sequence length |
| `lr` | 3e-4 | Peak learning rate |
| `warmup_steps` | 2000 | Linear warmup duration |
| `max_steps` | 100000 | Total training steps |
| `grad_accum_steps` | 1 | Gradient accumulation |
| `grad_clip` | 1.0 | Gradient norm clipping |
| `save_every` | 1000 | Checkpoint frequency (steps) |

---

## ⚡ TPU Details

Zira is designed to be fully **XLA graph-compilation safe**:

- All sequence lengths are **static** (no padding / dynamic shapes)
- The causal mask is **pre-computed** and registered as a buffer
- RoPE cos/sin tables are **pre-built** up to `max_seq_len`
- No Python-side branching on tensor values
- Uses `xm.optimizer_step()` instead of `optimizer.step()` on TPU
- `MpDeviceLoader` wraps the DataLoader for efficient TPU prefetch
- `DistributedSampler` for multi-core runs via `xmp.spawn`

---

## 💾 Checkpointing & Resume

Checkpoints save model weights, optimizer state, and scheduler step:

```python
# Automatically save to Google Drive in Colab
python -m zira.train_pretrain --model micro --use_drive

# Resume from a specific checkpoint
python -m zira.train_pretrain --model micro --resume checkpoints/ckpt_step0010000.pt

# Resume from the latest checkpoint in a directory
python -m zira.train_pretrain --model micro --resume checkpoints/
```

---

## 🧠 Text Generation

```python
from zira.config import get_config
from zira.model import ZiraModel
from zira.tokenizer import ZiraTokenizer
from zira.generate import generate
from zira.utils import load_checkpoint

cfg = get_config("micro")
model = ZiraModel(cfg)
load_checkpoint("checkpoints/ckpt_step0050000.pt", model)
tokenizer = ZiraTokenizer(hf_name="huggyllama/llama-7b")

output = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
print(output)
```

CLI flags for `zira.generate`:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `micro` | Model size |
| `--checkpoint` | — | Path to `.pt` checkpoint (required) |
| `--tokenizer` | `huggyllama/llama-7b` | HuggingFace tokenizer |
| `--prompt` | `"Once upon a time"` | Input prompt |
| `--max_new_tokens` | 200 | Maximum tokens to generate |
| `--temperature` | 0.8 | Sampling temperature |
| `--top_k` | 50 | Top-k filter (0 = disabled) |
| `--top_p` | 0.95 | Nucleus sampling threshold |

---

## 📦 Requirements

```
torch >= 2.0
datasets
transformers
sentencepiece
torch_xla  # TPU only — installed automatically in the Colab notebook
```

Install everything at once:

```bash
pip install torch datasets transformers sentencepiece
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).
