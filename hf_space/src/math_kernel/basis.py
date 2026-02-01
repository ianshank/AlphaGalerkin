"""Basis functions for Galerkin projection on the continuous domain.

The board is treated as a domain Omega = [0,1]^2. We project functions
onto orthogonal basis functions to enable resolution-independent learning.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class BasisFunction(Protocol):
    """Protocol for basis function implementations."""

    def evaluate(
        self,
        coords: Float[Tensor, "batch n 2"],
        frequencies: int,
    ) -> Float[Tensor, "batch n features"]:
        """Evaluate basis functions at given coordinates.

        Args:
            coords: Normalized coordinates in [0, 1]^2.
            frequencies: Number of frequency components.

        Returns:
            Basis function evaluations at each coordinate.

        """
        ...


class FourierBasis(nn.Module):
    """Fourier basis functions for spectral representation.

    Implements random Fourier features following:
    "Fourier Features Let Networks Learn High Frequency Functions in Low
    Dimensional Domains" (Tancik et al., 2020)

    The basis functions are:
        phi_k(x) = [cos(2*pi*B*x), sin(2*pi*B*x)]
    where B is a random frequency matrix.
    """

    def __init__(
        self,
        n_features: int,
        scale: float = 1.0,
        learnable: bool = False,
    ) -> None:
        """Initialize Fourier basis.

        Args:
            n_features: Number of Fourier features (output dimension // 2).
            scale: Standard deviation for frequency sampling.
            learnable: Whether to make frequencies learnable.

        """
        super().__init__()
        self.n_features = n_features
        self.scale = scale

        # Random frequency matrix B ~ N(0, scale^2)
        # Shape: (2, n_features) for 2D input coordinates
        b_matrix = torch.randn(2, n_features) * scale
        if learnable:
            self.b_matrix = nn.Parameter(b_matrix)
        else:
            self.register_buffer("b_matrix", b_matrix)

    def evaluate(
        self,
        coords: Float[Tensor, "batch n 2"],
    ) -> Float[Tensor, "batch n features"]:
        """Evaluate Fourier features at given coordinates.

        Args:
            coords: Normalized coordinates in [0, 1]^2.

        Returns:
            Fourier features of shape (batch, n, 2*n_features).

        """
        # Project coordinates onto frequency basis
        # coords: (batch, n, 2), b_matrix: (2, n_features)
        projection = torch.einsum("bnd,df->bnf", coords, self.b_matrix)

        # Apply 2*pi scaling
        projection = 2 * math.pi * projection

        # Concatenate sin and cos features
        features = torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)

        return features

    def forward(
        self,
        coords: Float[Tensor, "batch n 2"],
    ) -> Float[Tensor, "batch n features"]:
        """Forward pass (alias for evaluate)."""
        return self.evaluate(coords)


class ChebyshevBasis(nn.Module):
    """Chebyshev polynomial basis functions.

    Chebyshev polynomials of the first kind T_n(x) are useful for
    spectral methods due to their optimal approximation properties.

    T_0(x) = 1
    T_1(x) = x
    T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    """

    def __init__(
        self,
        max_degree: int,
    ) -> None:
        """Initialize Chebyshev basis.

        Args:
            max_degree: Maximum polynomial degree.

        """
        super().__init__()
        self.max_degree = max_degree

    def _chebyshev_1d(
        self,
        x: Float[Tensor, ...],
        max_degree: int,
    ) -> Float[Tensor, "... degree"]:
        """Evaluate 1D Chebyshev polynomials.

        Args:
            x: Input values in [-1, 1].
            max_degree: Maximum degree (exclusive).

        Returns:
            Chebyshev polynomial values for degrees 0 to max_degree-1.

        """
        # Recurrence relation
        polynomials = [torch.ones_like(x)]  # T_0 = 1

        if max_degree > 1:
            polynomials.append(x)  # T_1 = x

        for _ in range(2, max_degree):
            # T_n = 2*x*T_{n-1} - T_{n-2}
            t_n = 2 * x * polynomials[-1] - polynomials[-2]
            polynomials.append(t_n)

        return torch.stack(polynomials, dim=-1)

    def evaluate(
        self,
        coords: Float[Tensor, "batch n 2"],
    ) -> Float[Tensor, "batch n features"]:
        """Evaluate 2D Chebyshev features at given coordinates.

        Args:
            coords: Normalized coordinates in [0, 1]^2.

        Returns:
            Tensor product of Chebyshev polynomials.

        """
        # Map [0, 1] to [-1, 1] for Chebyshev domain
        x = 2 * coords[..., 0] - 1
        y = 2 * coords[..., 1] - 1

        # Evaluate 1D polynomials
        t_x = self._chebyshev_1d(x, self.max_degree)  # (batch, n, degree)
        t_y = self._chebyshev_1d(y, self.max_degree)  # (batch, n, degree)

        # Tensor product basis: T_i(x) * T_j(y)
        # Result shape: (batch, n, degree * degree)
        features = torch.einsum("bni,bnj->bnij", t_x, t_y)
        features = rearrange(features, "b n i j -> b n (i j)")

        return features

    def forward(
        self,
        coords: Float[Tensor, "batch n 2"],
    ) -> Float[Tensor, "batch n features"]:
        """Forward pass (alias for evaluate)."""
        return self.evaluate(coords)


def create_grid_coordinates(
    board_size: int,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> Float[Tensor, "batch n 2"]:
    """Create normalized grid coordinates for a square board.

    Maps discrete board positions to continuous coordinates in [0, 1]^2.
    Uses cell-centered coordinates: position (i, j) maps to
    ((i + 0.5) / board_size, (j + 0.5) / board_size).

    Args:
        board_size: Size of the board (e.g., 9, 13, 19).
        batch_size: Batch size.
        device: Target device.

    Returns:
        Normalized coordinates of shape (batch, board_size^2, 2).

    """
    # Create grid indices
    indices = torch.arange(board_size, device=device, dtype=torch.float32)

    # Create meshgrid (cell-centered)
    y_coords, x_coords = torch.meshgrid(indices, indices, indexing="ij")

    # Normalize to [0, 1] (cell-centered)
    x_coords = (x_coords + 0.5) / board_size
    y_coords = (y_coords + 0.5) / board_size

    # Flatten and stack
    coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=-1)

    # Expand for batch dimension
    coords = coords.unsqueeze(0).expand(batch_size, -1, -1)

    return coords
