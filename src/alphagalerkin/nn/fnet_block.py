"""FNet block: FFT-based token mixing for fast MCTS evaluation.

Replaces self-attention with FFT for O(N log N) global mixing.
No learnable parameters in the spectral domain -- mixing happens
purely through the Fourier transform.

Reference: Lee-Thorp et al., "FNet: Mixing Tokens with Fourier
Transforms" (2021).
"""
from __future__ import annotations

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger("nn.fnet_block")


class FNetMixing(nn.Module):
    """FFT-based token mixing layer.

    Applies real-valued FFT along the sequence dimension,
    takes the real part, providing global token interaction
    without learnable attention weights.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix tokens via FFT.

        Args:
            x: shape (batch, seq_len, hidden_dim) or
                (seq_len, hidden_dim).

        Returns:
            Mixed tensor, same shape as input.

        """
        # Use rfft for efficiency (real input), then take real part
        # FFT along sequence dimension (dim=-2)
        return torch.fft.fft(x, dim=-2).real


class FNetBlock(nn.Module):
    """FNet block: FFT mixing + feedforward.

    Architecture:
        x -> LayerNorm -> FFT mixing -> residual
          -> LayerNorm -> FFN -> residual

    This provides O(N log N) global mixing without attention
    weights, enabling fast batch MCTS leaf evaluation.

    Parameters
    ----------
    hidden_dim:
        Model hidden dimension.
    expansion_factor:
        FFN intermediate dimension = hidden_dim * expansion_factor.
    dropout:
        Dropout probability.

    """

    def __init__(
        self,
        hidden_dim: int = 128,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mixing = FNetMixing()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FFT mixing and FFN.

        Args:
            x: shape (batch, seq_len, hidden_dim) or
                (seq_len, hidden_dim).

        Returns:
            Processed tensor, same shape.

        """
        needs_batch = x.dim() == 2
        if needs_batch:
            x = x.unsqueeze(0)

        # FFT mixing with residual
        normed = self.norm1(x)
        x = x + self.mixing(normed)

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        if needs_batch:
            x = x.squeeze(0)

        return x
