"""
Rotary Positional Embeddings (RoPE).
Implements position-dependent rotation of query/key vectors.
Fully static – no dynamic branching, XLA-friendly.
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """
    Pre-computes cos/sin tables for RoPE up to `max_seq_len`.
    Stored as non-parameter buffers so they are moved with .to(device).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        # Inverse frequencies: shape (head_dim // 2,)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-build tables for the full sequence length
        self._build_cache(max_seq_len)

    # ------------------------------------------------------------------
    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)       # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)      # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    # ------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to queries and keys.

        Args:
            q: (batch, heads, seq_len, head_dim)
            k: (batch, kv_heads, seq_len, head_dim)

        Returns:
            Rotated (q, k) tensors with the same shape.
        """
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :]   # (1, 1, seq_len, head_dim)
        sin = self.sin_cached[:, :, :seq_len, :]

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
