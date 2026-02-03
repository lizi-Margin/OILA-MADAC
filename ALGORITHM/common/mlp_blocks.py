import torch
import torch.nn as nn
from typing import Optional

class ResidualMLPBlock(nn.Module):
    """Lightweight residual MLP block with LayerNorm and activation clamp."""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0, clamp_value: float = 10.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        self.clamp_value = clamp_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.lin1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.norm2(out)
        out = out + residual
        return out.clamp_(-self.clamp_value, self.clamp_value)

