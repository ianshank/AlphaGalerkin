"""Discrete CNN baseline for the honest zero-shot-transfer benchmark.

This module provides the *discrete* foil against which the resolution-independent
:class:`~src.experiments.physics_model.PhysicsOperator` is measured. A plain
convolutional network learns a **fixed-pixel-radius** stencil. Because the Poisson
discretisation uses grid spacing ``h = 1 / (n + 1)`` (see
:meth:`src.physics.poisson.PoissonSolver._solve_spectral`), the discrete Green's
function is length-scale dependent: a stencil learned at ``9x9`` sits at the wrong
physical scale on a ``19x19`` grid. The CNN therefore **cannot** transfer zero-shot
and must be *retrained at the target resolution* — that mandatory retraining is
exactly the limitation the benchmark quantifies.

The network is fully convolutional (no ``Linear`` layer tied to the grid size), so it
is architecturally runnable at any resolution; the benchmark nonetheless retrains it
per resolution on purpose. It consumes the same flattened ``PoissonSample`` charge
field the operator sees (``(B, N)`` with ``N = grid_size ** 2``), reshapes to a
``(B, 1, grid_size, grid_size)`` image, and predicts the flattened potential ``(B, N)``
so both arms score MSE against the identical ``PoissonSample.potential`` targets.

No hardcoded architecture values: every knob is a constructor argument surfaced from
:class:`~src.poc.scenarios.transfer_baseline_compare_config.TransferBaselineCompareConfig`.
"""

from __future__ import annotations

import math

import structlog
from torch import Tensor, nn

logger = structlog.get_logger(__name__)

# Search bound for match_cnn_channels: how far above the analytic channel estimate we
# are willing to look before giving up and returning the closest candidate found.
DEFAULT_CHANNEL_SEARCH_SPAN: int = 64


def _infer_grid_size(n_points: int) -> int:
    """Recover the square grid side length from a flattened point count.

    Args:
        n_points: Number of flattened grid points (``grid_size ** 2``).

    Returns:
        The grid side length ``grid_size``.

    Raises:
        ValueError: If ``n_points`` is not a positive perfect square.

    """
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")
    side = int(round(math.sqrt(n_points)))
    if side * side != n_points:
        raise ValueError(
            f"Charge vector length {n_points} is not a perfect square; "
            "DiscreteCNNBaseline requires a square grid."
        )
    return side


class _ResidualConvBlock(nn.Module):
    """A residual convolution block: Conv -> (BatchNorm) -> GELU -> (Dropout) + skip.

    The convolution preserves spatial extent (``padding = kernel_size // 2`` for the
    odd kernels this benchmark uses), so the block is resolution-agnostic.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        use_batchnorm: bool,
        dropout: float,
    ) -> None:
        """Initialise the block.

        Args:
            channels: Number of feature channels (constant across the block).
            kernel_size: Odd convolution kernel size.
            use_batchnorm: Whether to apply ``BatchNorm2d`` after the convolution.
            dropout: Dropout probability in ``[0, 1)``; ``0.0`` disables dropout.

        """
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.norm: nn.Module = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.activation = nn.GELU()
        self.dropout: nn.Module = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block.

        Args:
            x: Input feature map ``(B, C, H, W)``.

        Returns:
            Output feature map ``(B, C, H, W)`` (same shape as ``x``).

        """
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return x + out


class DiscreteCNNBaseline(nn.Module):
    """Fully-convolutional discrete baseline that must be retrained per resolution.

    Maps a flattened charge field to a flattened potential field through a stack of
    same-resolution convolutions. It is the honest "you would otherwise have to
    retrain" foil for the resolution-independent operator.
    """

    def __init__(
        self,
        n_layers: int = 6,
        channels: int = 32,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the CNN baseline.

        Args:
            n_layers: Number of residual convolution blocks (``>= 0``).
            channels: Feature channels used throughout the trunk (``>= 1``).
            kernel_size: Odd convolution kernel size (``>= 1``).
            use_batchnorm: Whether residual blocks use ``BatchNorm2d``.
            dropout: Dropout probability in ``[0, 1)``.

        Raises:
            ValueError: If ``kernel_size`` is even or any bound is violated.

        """
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd int, got {kernel_size}")
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if n_layers < 0:
            raise ValueError(f"n_layers must be >= 0, got {n_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.n_layers = n_layers
        self.channels = channels
        self.kernel_size = kernel_size

        padding = kernel_size // 2
        self.input_conv = nn.Conv2d(1, channels, kernel_size, padding=padding)
        self.blocks = nn.ModuleList(
            _ResidualConvBlock(channels, kernel_size, use_batchnorm, dropout)
            for _ in range(n_layers)
        )
        self.output_conv = nn.Conv2d(channels, 1, kernel_size, padding=padding)

    def forward(self, charges: Tensor) -> Tensor:
        """Predict the flattened potential from a flattened charge field.

        Args:
            charges: Charge field, either flattened ``(B, N)`` (``N = grid_size ** 2``)
                or already gridded ``(B, 1, H, W)`` / ``(B, H, W)``.

        Returns:
            Predicted potential flattened to ``(B, N)`` to match
            ``PoissonSample.potential``.

        """
        if charges.ndim == 2:
            batch, n_points = charges.shape
            side = _infer_grid_size(n_points)
            grid = charges.view(batch, 1, side, side)
        elif charges.ndim == 3:
            batch, side, width = charges.shape
            if side != width:
                raise ValueError(f"Expected a square grid, got ({side}, {width})")
            grid = charges.view(batch, 1, side, side)
        elif charges.ndim == 4:
            grid = charges
            batch, _, side, width = charges.shape
            if side != width:
                raise ValueError(f"Expected a square grid, got ({side}, {width})")
        else:
            raise ValueError(f"Unsupported charges rank {charges.ndim}; expected 2, 3 or 4")

        h = self.input_conv(grid)
        for block in self.blocks:
            h = block(h)
        out = self.output_conv(h)
        return out.view(batch, side * side)


def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters of a module.

    Args:
        module: Any ``nn.Module``.

    Returns:
        Total number of elements across all parameters.

    """
    return sum(p.numel() for p in module.parameters())


def match_cnn_channels(
    target_n_params: int,
    n_layers: int,
    kernel_size: int,
    *,
    use_batchnorm: bool = True,
    tolerance: float = 0.15,
    search_span: int = DEFAULT_CHANNEL_SEARCH_SPAN,
) -> int:
    """Find the CNN channel width whose parameter count is closest to a target.

    Used to size the CNN baseline near the operator's parameter count so the
    *secondary* matched-parameter sanity metric is auditable. Parameter-count parity
    is deliberately **not** the benchmark's gated axis (raw parity between a CNN and a
    Fourier-Galerkin transformer is cosmetic); this helper simply makes the number
    honest and reproducible.

    Starts from the analytic estimate ``C ~= sqrt(target / (n_layers * k^2))`` (the
    trunk dominates the parameter count) and searches integer widths around it,
    returning the width minimising ``|n_params - target|``.

    Args:
        target_n_params: Parameter count to match (typically the operator's).
        n_layers: CNN depth to size for.
        kernel_size: CNN kernel size to size for.
        use_batchnorm: Whether the sized CNN uses batch norm (affects the count).
        tolerance: Relative band; a warning is logged if the best width misses it.
        search_span: How many channels above the analytic seed to scan.

    Returns:
        The channel width whose built ``DiscreteCNNBaseline`` is closest to target.

    Raises:
        ValueError: If ``target_n_params`` is not positive.

    """
    if target_n_params <= 0:
        raise ValueError(f"target_n_params must be positive, got {target_n_params}")

    denom = max(n_layers, 1) * kernel_size * kernel_size
    seed = int(round(math.sqrt(target_n_params / denom)))
    low = max(1, seed - search_span)
    high = seed + search_span

    best_channels = low
    best_gap = math.inf
    for channels in range(low, high + 1):
        candidate = DiscreteCNNBaseline(
            n_layers=n_layers,
            channels=channels,
            kernel_size=kernel_size,
            use_batchnorm=use_batchnorm,
        )
        gap = abs(count_parameters(candidate) - target_n_params)
        if gap < best_gap:
            best_gap = gap
            best_channels = channels

    relative_gap = best_gap / target_n_params
    if relative_gap > tolerance:
        logger.warning(
            "cnn_channel_match_outside_tolerance",
            target_n_params=target_n_params,
            best_channels=best_channels,
            relative_gap=relative_gap,
            tolerance=tolerance,
        )
    return best_channels
