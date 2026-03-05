"""
Zira – Decoder-only Transformer.

Features:
  * Pre-Norm residual (RMSNorm)
  * Rotary Positional Embeddings (RoPE)
  * Grouped Query Attention (GQA)
  * SwiGLU Feed-Forward
  * Weight tying: token embedding ↔ lm_head
  * No bias in any linear layer
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import ZiraConfig
from .transformer_block import TransformerBlock, RMSNorm


class ZiraModel(nn.Module):
    """
    Full decoder-only Transformer.

    Args:
        config: ZiraConfig instance describing the architecture.
    """

    def __init__(self, config: ZiraConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.layers)]
        )

        # Final normalisation
        self.norm = RMSNorm(config.d_model, config.norm_eps)

        # Language model head (no bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: lm_head and embedding share the same weight matrix
        self.lm_head.weight = self.embed_tokens.weight

        # Initialise weights
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: (batch, seq_len) int64 token ids
            labels:    (batch, seq_len) int64 token ids; if provided,
                       computes cross-entropy loss (next-token prediction).

        Returns:
            If labels is None:  logits (batch, seq_len, vocab_size)
            If labels provided: (loss scalar, logits)
        """
        x = self.embed_tokens(input_ids)  # (B, S, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)          # (B, S, vocab_size)

        if labels is None:
            return logits

        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
        )
        return loss, logits

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of (trainable) parameters."""
        return sum(
            p.numel()
            for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )
