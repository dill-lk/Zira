"""
Transformer Block with Pre-Norm layout.
Applies RMSNorm → Attention → residual, then RMSNorm → FFN → residual.
"""

import torch
import torch.nn as nn

from .config import ZiraConfig
from .attention import GroupedQueryAttention
from .ffn import SwiGLUFFN


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no re-centering)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class TransformerBlock(nn.Module):
    """
    A single decoder-only Transformer block.

    Layout:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(self, config: ZiraConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.attn  = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.ffn   = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
