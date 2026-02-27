"""Basis functions for Galerkin projection on the continuous domain.

The board is treated as a domain Omega = [0,1]^2. We project functions
onto orthogonal basis functions to enable resolution-independent learning.

Supports both PyTorch and JAX backends. The original PyTorch classes
(FourierBasis, ChebyshevBasis) are kept unchanged for backward compatibility.
JAX-backed equivalents (JaxFourierBasis, JaxChebyshevBasis) are provided
when JAX/Flax are installed.  Factory functions select the right class
based on a ``backend`` parameter.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import structlog
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional JAX / Flax imports
# ---------------------------------------------------------------------------
try:
    import flax.linen as fnn
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


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


# ===================================================================
# PyTorch implementations (unchanged for backward compatibility)
# ===================================================================


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


# ===================================================================
# JAX / Flax implementations (guarded by HAS_JAX)
# ===================================================================

if HAS_JAX:

    class JaxFourierBasis(fnn.Module):
        """JAX/Flax Fourier basis functions for spectral representation.

        Implements random Fourier features following:
        "Fourier Features Let Networks Learn High Frequency Functions in Low
        Dimensional Domains" (Tancik et al., 2020)

        The basis functions are:
            phi_k(x) = [cos(2*pi*B*x), sin(2*pi*B*x)]
        where B is a random frequency matrix.

        This is the Flax equivalent of :class:`FourierBasis`.  When
        ``learnable=True`` the frequency matrix is stored as a trainable
        ``param``; otherwise it is stored as a non-trainable ``variable``
        in the ``'constants'`` collection.

        Attributes:
            n_features: Number of Fourier features (output dimension // 2).
            scale: Standard deviation for frequency sampling.
            learnable: Whether to make frequencies learnable.

        """

        n_features: int
        scale: float = 1.0
        learnable: bool = False

        @fnn.compact
        def __call__(self, coords: Any) -> Any:
            """Evaluate Fourier features at given coordinates.

            Args:
                coords: JAX array of shape (batch, n, 2) with normalized
                    coordinates in [0, 1]^2.

            Returns:
                Fourier features of shape (batch, n, 2 * n_features).

            """
            if self.learnable:
                b_matrix = self.param(
                    "b_matrix",
                    fnn.initializers.normal(self.scale),
                    (2, self.n_features),
                )
            else:
                b_matrix = self.variable(
                    "constants",
                    "b_matrix",
                    lambda: (
                        jax.random.normal(self.make_rng("params"), (2, self.n_features))
                        * self.scale
                    ),
                ).value

            # Project coordinates onto frequency basis
            # coords: (batch, n, 2), b_matrix: (2, n_features)
            projection = jnp.einsum("bnd,df->bnf", coords, b_matrix)

            # Apply 2*pi scaling
            projection = 2 * jnp.pi * projection

            # Concatenate cos and sin features
            features = jnp.concatenate([jnp.cos(projection), jnp.sin(projection)], axis=-1)

            return features

    class JaxChebyshevBasis(fnn.Module):
        """JAX/Flax Chebyshev polynomial basis functions.

        Chebyshev polynomials of the first kind T_n(x) are useful for
        spectral methods due to their optimal approximation properties.

        T_0(x) = 1
        T_1(x) = x
        T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)

        This is the Flax equivalent of :class:`ChebyshevBasis`.
        It is a pure-function module with no trainable parameters.

        Attributes:
            max_degree: Maximum polynomial degree.

        """

        max_degree: int

        @staticmethod
        def _chebyshev_1d(x: Any, max_degree: int) -> Any:
            """Evaluate 1D Chebyshev polynomials.

            Args:
                x: JAX array of input values in [-1, 1].
                max_degree: Maximum degree (exclusive).

            Returns:
                Chebyshev polynomial values stacked along the last axis,
                shape ``(..., max_degree)``.

            """
            polynomials = [jnp.ones_like(x)]  # T_0 = 1

            if max_degree > 1:
                polynomials.append(x)  # T_1 = x

            for _ in range(2, max_degree):
                # T_n = 2*x*T_{n-1} - T_{n-2}
                t_n = 2 * x * polynomials[-1] - polynomials[-2]
                polynomials.append(t_n)

            return jnp.stack(polynomials, axis=-1)

        @fnn.compact
        def __call__(self, coords: Any) -> Any:
            """Evaluate 2D Chebyshev features at given coordinates.

            Args:
                coords: JAX array of shape (batch, n, 2) with normalized
                    coordinates in [0, 1]^2.

            Returns:
                Tensor product of Chebyshev polynomials, shape
                ``(batch, n, max_degree * max_degree)``.

            """
            # Map [0, 1] to [-1, 1] for Chebyshev domain
            x = 2 * coords[..., 0] - 1
            y = 2 * coords[..., 1] - 1

            # Evaluate 1D polynomials
            t_x = self._chebyshev_1d(x, self.max_degree)  # (batch, n, degree)
            t_y = self._chebyshev_1d(y, self.max_degree)  # (batch, n, degree)

            # Tensor product basis: T_i(x) * T_j(y)
            # Result shape: (batch, n, degree * degree)
            features = jnp.einsum("bni,bnj->bnij", t_x, t_y)
            batch, n, i, j = features.shape
            features = features.reshape(batch, n, i * j)

            return features

    def create_grid_coordinates_jax(
        board_size: int,
        batch_size: int = 1,
    ) -> Any:
        """Create normalized grid coordinates for a square board using JAX.

        Maps discrete board positions to continuous coordinates in [0, 1]^2.
        Uses cell-centered coordinates: position (i, j) maps to
        ((i + 0.5) / board_size, (j + 0.5) / board_size).

        Args:
            board_size: Size of the board (e.g., 9, 13, 19).
            batch_size: Batch size.

        Returns:
            JAX array of normalized coordinates with shape
            (batch, board_size^2, 2).

        """
        # Create grid indices
        indices = jnp.arange(board_size, dtype=jnp.float32)

        # Create meshgrid (cell-centered)
        y_coords, x_coords = jnp.meshgrid(indices, indices, indexing="ij")

        # Normalize to [0, 1] (cell-centered)
        x_coords = (x_coords + 0.5) / board_size
        y_coords = (y_coords + 0.5) / board_size

        # Flatten and stack
        coords = jnp.stack([x_coords.flatten(), y_coords.flatten()], axis=-1)

        # Expand for batch dimension using broadcast
        coords = jnp.broadcast_to(
            coords[jnp.newaxis, ...], (batch_size, board_size * board_size, 2)
        )

        return coords


# ===================================================================
# Factory functions
# ===================================================================


def create_fourier_basis(
    n_features: int,
    scale: float = 1.0,
    learnable: bool = False,
    backend: str = "torch",
) -> Any:
    """Create a Fourier basis instance for the specified backend.

    Args:
        n_features: Number of Fourier features (output dimension // 2).
        scale: Standard deviation for frequency sampling.
        learnable: Whether to make frequencies learnable.
        backend: Backend framework to use (``"torch"`` or ``"jax"``).

    Returns:
        A ``FourierBasis`` (PyTorch ``nn.Module``) when ``backend="torch"``,
        or a ``JaxFourierBasis`` (Flax ``nn.Module``) when ``backend="jax"``.

    Raises:
        ImportError: If ``backend="jax"`` but JAX/Flax are not installed.
        ValueError: If an unknown backend name is provided.

    """
    log = logger.bind(
        factory="create_fourier_basis",
        backend=backend,
        n_features=n_features,
        scale=scale,
        learnable=learnable,
    )

    if backend == "jax":
        if not HAS_JAX:
            raise ImportError(
                "JAX and Flax are required for the 'jax' backend. "
                "Install with: pip install 'alphagalerkin[jax]'"
            )
        log.info("factory.created", cls="JaxFourierBasis")
        return JaxFourierBasis(
            n_features=n_features,
            scale=scale,
            learnable=learnable,
        )

    if backend == "torch":
        log.info("factory.created", cls="FourierBasis")
        return FourierBasis(
            n_features=n_features,
            scale=scale,
            learnable=learnable,
        )

    msg = f"Unknown backend: {backend!r}. Supported: 'torch', 'jax'"
    raise ValueError(msg)


def create_chebyshev_basis(
    max_degree: int,
    backend: str = "torch",
) -> Any:
    """Create a Chebyshev basis instance for the specified backend.

    Args:
        max_degree: Maximum polynomial degree.
        backend: Backend framework to use (``"torch"`` or ``"jax"``).

    Returns:
        A ``ChebyshevBasis`` (PyTorch ``nn.Module``) when ``backend="torch"``,
        or a ``JaxChebyshevBasis`` (Flax ``nn.Module``) when ``backend="jax"``.

    Raises:
        ImportError: If ``backend="jax"`` but JAX/Flax are not installed.
        ValueError: If an unknown backend name is provided.

    """
    log = logger.bind(
        factory="create_chebyshev_basis",
        backend=backend,
        max_degree=max_degree,
    )

    if backend == "jax":
        if not HAS_JAX:
            raise ImportError(
                "JAX and Flax are required for the 'jax' backend. "
                "Install with: pip install 'alphagalerkin[jax]'"
            )
        log.info("factory.created", cls="JaxChebyshevBasis")
        return JaxChebyshevBasis(max_degree=max_degree)

    if backend == "torch":
        log.info("factory.created", cls="ChebyshevBasis")
        return ChebyshevBasis(max_degree=max_degree)

    msg = f"Unknown backend: {backend!r}. Supported: 'torch', 'jax'"
    raise ValueError(msg)
