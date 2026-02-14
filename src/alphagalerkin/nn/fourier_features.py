"""Fourier features for resolution-independent positional encoding.

Encodes continuous coordinates using random Fourier features,
enabling the network to handle arbitrary mesh resolutions
without retraining.
"""
from __future__ import annotations

import math

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger("nn.fourier_features")


class FourierPositionalEncoding(nn.Module):
    """Random Fourier features for continuous positional encoding.

    Maps continuous coordinates to a high-dimensional feature space
    using sinusoidal projections with learnable or fixed frequencies.

    Parameters
    ----------
    coord_dim:
        Input coordinate dimension (e.g. 2 for 2D).
    num_frequencies:
        Number of frequency components per coordinate.
    learnable:
        If True, frequency matrix B is learnable.
        If False, B is sampled from N(0, sigma^2) and frozen.
    sigma:
        Standard deviation for frequency initialization.

    Output dimension: coord_dim + 2 * num_frequencies * coord_dim
    (original coords + sin + cos for each frequency).

    """

    def __init__(
        self,
        coord_dim: int = 2,
        num_frequencies: int = 32,
        learnable: bool = False,
        sigma: float = 10.0,
    ) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.num_frequencies = num_frequencies
        self.output_dim = coord_dim + 2 * num_frequencies * coord_dim

        # Frequency matrix B: (num_frequencies * coord_dim, coord_dim)
        B = torch.randn(num_frequencies * coord_dim, coord_dim) * sigma
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        logger.debug(
            "fourier_features.init",
            coord_dim=coord_dim,
            num_frequencies=num_frequencies,
            output_dim=self.output_dim,
            learnable=learnable,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates with Fourier features.

        Args:
            coords: Coordinate tensor, shape (..., coord_dim).

        Returns:
            Encoded features, shape (..., output_dim).

        """
        # Project: coords @ B^T -> (..., num_frequencies * coord_dim)
        projected = torch.matmul(coords, self.B.T)
        projected = 2.0 * math.pi * projected

        # Concatenate: [original_coords, sin(projected), cos(projected)]
        return torch.cat([
            coords,
            torch.sin(projected),
            torch.cos(projected),
        ], dim=-1)
