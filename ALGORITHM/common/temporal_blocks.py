import torch
import torch.nn as nn
from typing import Optional, Tuple

from ALGORITHM.common.attention import SimpleAttention


class SimpleMultiHeadTemporalAttention(nn.Module):
    """Multi-head self-attention over temporal tokens built from SimpleAttention heads."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert dim % num_heads == 0, (
            f"Temporal attention hidden size {dim} must be divisible by num_heads {num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.heads = nn.ModuleList([SimpleAttention(self.head_dim) for _ in range(num_heads)])
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len) boolean mask where True indicates padded positions.
        """
        bsz, seq_len, _ = x.shape
        x_heads = x.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        outputs = []
        attn_mask = mask.bool() if mask is not None else None
        for head_idx, attn in enumerate(self.heads):
            head_in = x_heads[:, head_idx, :, :]  # (batch, seq_len, head_dim)
            head_out = attn(k=head_in, q=head_in, v=head_in, mask=attn_mask)
            outputs.append(head_out)
        concat = torch.cat(outputs, dim=-1)
        return self.dropout(self.out_proj(concat))


class TemporalAttentionBlock(nn.Module):
    """LayerNorm -> Multi-head temporal attention -> residual -> FFN -> residual."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = SimpleMultiHeadTemporalAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_in = self.attn_norm(x)
        attn_out = self.attn(attn_in, mask=mask)
        x = x + self.dropout(attn_out)
        ffn_in = self.ffn_norm(x)
        ffn_out = self.ffn(ffn_in)
        return x + self.dropout(ffn_out)


class TemporalSequenceEncoder(nn.Module):
    """Stack of TemporalAttentionBlocks with optional sliding-window state management."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        window: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            TemporalAttentionBlock(dim=dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        )
        self.window = window

    def forward_sequence(
        self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            sequence: (batch, n_agent, seq_len, dim)
            mask: (batch, n_agent, seq_len) boolean mask for padded timesteps.
        """
        bsz, n_agent, seq_len, dim = sequence.shape
        x = sequence.view(bsz * n_agent, seq_len, dim)
        mask_flat = None
        if mask is not None:
            mask_flat = mask.view(bsz * n_agent, seq_len)
        for layer in self.layers:
            x = layer(x, mask=mask_flat)
        return x.view(bsz, n_agent, seq_len, dim)

    def forward_step(
        self,
        token: torch.Tensor,
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            token: (batch, n_agent, dim) current timestep embeddings.
            prev_state: tuple(sequence, mask) with shapes
                sequence -> (batch, n_agent, <=window, dim)
                mask -> (batch, n_agent, <=window) boolean
            token_mask: (batch, n_agent) boolean mask marking invalid tokens.
        Returns:
            output: (batch, n_agent, dim) encoded representation of the latest token.
            new_state: updated tuple(sequence, mask) with window trimming applied.
        """
        seq_prev, mask_prev = (None, None) if prev_state is None else prev_state
        token = token.unsqueeze(2)  # (batch, n_agent, 1, dim)
        if seq_prev is None:
            sequence = token
        else:
            sequence = torch.cat([seq_prev, token], dim=2)
        if mask_prev is None:
            mask = token_mask.unsqueeze(2) if token_mask is not None else None
        else:
            if token_mask is not None:
                token_mask_exp = token_mask.unsqueeze(2)
            else:
                token_mask_exp = mask_prev.new_zeros(token.shape[:3])
            mask = torch.cat([mask_prev, token_mask_exp], dim=2)
        if self.window > 0 and sequence.shape[2] > self.window:
            sequence = sequence[:, :, -self.window :, :]
            if mask is not None:
                mask = mask[:, :, -self.window :]
        encoded = self.forward_sequence(sequence, mask)
        latest = encoded[:, :, -1, :]
        return latest, (sequence, mask)
