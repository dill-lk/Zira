"""
Grouped Query Attention (GQA) with RoPE and causal masking.
Supports Multi-Head Attention (kv_heads == heads) and GQA (kv_heads < heads).
No bias terms, XLA-friendly static shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .config import ZiraConfig
from .rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """
    Causal self-attention with Grouped Query Attention and RoPE.

    For GQA: kv_heads < heads, and each KV head is shared across
    (heads // kv_heads) query heads.
    """

    def __init__(self, config: ZiraConfig):
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.groups = config.heads // config.kv_heads   # queries per kv head

        # Projections – no bias
        self.q_proj = nn.Linear(config.d_model, config.heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.heads * config.head_dim, config.d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Rotary embeddings
        self.rope = RotaryEmbedding(config.head_dim, config.seq_len)

        # Static causal mask (upper-triangular, registered as buffer)
        mask = torch.full((config.seq_len, config.seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        B, S, _ = x.shape

        # Project and reshape to (B, heads, S, head_dim)
        q = self.q_proj(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        q, k = self.rope(q, k)

        # Expand KV heads to match query heads for GQA
        # (B, kv_heads, S, head_dim) → (B, heads, S, head_dim)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        # Scaled dot-product attention with static causal mask
        # attn_bias: (1, 1, S, S) slice of precomputed mask
        attn_bias = self.causal_mask[:S, :S].unsqueeze(0).unsqueeze(0)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale + attn_bias
        attn_probs = F.softmax(attn_weights, dim=-1)

        # (B, heads, S, head_dim) → (B, S, heads * head_dim)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.heads * self.head_dim)
        return self.o_proj(out)
