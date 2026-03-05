"""
SwiGLU Feed-Forward Network.
FFN(x) = (SiLU(gate_proj(x)) * up_proj(x)) @ down_proj
No bias terms. XLA-friendly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ZiraConfig


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feedforward block.

    Uses two separate up-projections (gate and up) followed by
    element-wise gating and a single down-projection.
    """

    def __init__(self, config: ZiraConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) * up, then project down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
