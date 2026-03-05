"""
Zira model configurations.
Defines model families from ~30M to ~450M parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class ZiraConfig:
    # Architecture
    vocab_size: int = 32000
    layers: int = 12
    d_model: int = 512
    heads: int = 8          # query heads
    kv_heads: int = 4       # key/value heads for GQA (set equal to heads to disable GQA)
    head_dim: int = 64
    ffn_hidden: int = 2048
    seq_len: int = 1024

    # Regularisation
    dropout: float = 0.0    # keep 0 for TPU / large-scale runs

    # Numeric
    norm_eps: float = 1e-5

    # Training
    max_batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100_000
    grad_accum_steps: int = 1

    # Checkpoint
    save_every: int = 1000
    checkpoint_dir: str = "checkpoints"

    # Dataset
    dataset_cache_dir: str = "data_cache"

    # Internal (computed on post_init)
    n_params: int = field(default=0, init=False, repr=False)

    # Model family name (informational)
    name: str = "Zira-Micro"

    def __post_init__(self) -> None:
        assert self.d_model == self.heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal heads * head_dim "
            f"({self.heads} * {self.head_dim})"
        )
        assert self.heads % self.kv_heads == 0, (
            f"heads ({self.heads}) must be divisible by kv_heads ({self.kv_heads})"
        )
        self.n_params = self._count_params()
        print(
            f"[ZiraConfig] {self.name} | "
            f"layers={self.layers}, d_model={self.d_model}, "
            f"heads={self.heads}(kv={self.kv_heads}), "
            f"ffn={self.ffn_hidden}, seq={self.seq_len} | "
            f"~{self.n_params / 1e6:.1f}M params"
        )

    # ------------------------------------------------------------------
    # Approximate parameter count (without biases, with weight tying)
    # ------------------------------------------------------------------
    def _count_params(self) -> int:
        d = self.d_model
        v = self.vocab_size
        # Embedding table (shared with lm_head → count once)
        emb = v * d
        # Per-layer: attention projections + ffn + two RMSNorm (no bias)
        # Attention: Q=(d, heads*head_dim), K=(d, kv_heads*head_dim),
        #            V=(d, kv_heads*head_dim), O=(heads*head_dim, d)
        q_proj = d * (self.heads * self.head_dim)
        k_proj = d * (self.kv_heads * self.head_dim)
        v_proj = d * (self.kv_heads * self.head_dim)
        o_proj = (self.heads * self.head_dim) * d
        attn = q_proj + k_proj + v_proj + o_proj
        # FFN (SwiGLU): gate_proj + up_proj + down_proj
        ffn = d * self.ffn_hidden * 2 + self.ffn_hidden * d
        # RMSNorm: 2 per layer (pre-attn, pre-ffn) + 1 final
        norms = self.layers * 2 * d + d
        return emb + self.layers * (attn + ffn) + norms


# ---------------------------------------------------------------------------
# Predefined model families
# ---------------------------------------------------------------------------

def ZiraMicro(**kwargs) -> ZiraConfig:
    """~64M parameter model – fast debug runs."""
    defaults = dict(
        name="Zira-Micro",
        layers=12,
        d_model=512,
        heads=8,
        kv_heads=4,
        head_dim=64,
        ffn_hidden=2048,
        seq_len=1024,
        warmup_steps=1000,
        max_steps=50_000,
    )
    defaults.update(kwargs)
    return ZiraConfig(**defaults)


def ZiraSmall(**kwargs) -> ZiraConfig:
    """~119M parameter model – full SFT in one Colab session."""
    defaults = dict(
        name="Zira-Small",
        layers=16,
        d_model=640,
        heads=10,
        kv_heads=5,
        head_dim=64,
        ffn_hidden=2560,
        seq_len=1024,
        warmup_steps=1500,
        max_steps=80_000,
    )
    defaults.update(kwargs)
    return ZiraConfig(**defaults)


def ZiraCompact(**kwargs) -> ZiraConfig:
    """~163M parameter model."""
    defaults = dict(
        name="Zira-Compact",
        layers=16,
        d_model=768,
        heads=12,
        kv_heads=4,
        head_dim=64,
        ffn_hidden=3072,
        seq_len=1024,
        warmup_steps=2000,
        max_steps=100_000,
    )
    defaults.update(kwargs)
    return ZiraConfig(**defaults)


def ZiraBase(**kwargs) -> ZiraConfig:
    """~410M parameter model – foundation pretrain across multiple sessions."""
    defaults = dict(
        name="Zira-Base",
        layers=24,
        d_model=1024,
        heads=16,
        kv_heads=8,
        head_dim=64,
        ffn_hidden=4096,
        seq_len=2048,
        warmup_steps=2000,
        max_steps=200_000,
        lr=2e-4,
    )
    defaults.update(kwargs)
    return ZiraConfig(**defaults)


def ZiraPro(**kwargs) -> ZiraConfig:
    """~801M parameter model – near full TPU utilisation."""
    defaults = dict(
        name="Zira-Pro",
        layers=32,
        d_model=1280,
        heads=20,
        kv_heads=5,
        head_dim=64,
        ffn_hidden=5120,
        seq_len=2048,
        warmup_steps=3000,
        max_steps=300_000,
        lr=1e-4,
    )
    defaults.update(kwargs)
    return ZiraConfig(**defaults)


# Registry for easy lookup by name
MODEL_REGISTRY = {
    "micro":   ZiraMicro,
    "small":   ZiraSmall,
    "compact": ZiraCompact,
    "base":    ZiraBase,
    "pro":     ZiraPro,
}


def get_config(name: str, **kwargs) -> ZiraConfig:
    """Return a ZiraConfig by short name (case-insensitive)."""
    key = name.lower().replace("zira-", "")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)
