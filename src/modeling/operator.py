"""Neural Operator model for physics simulation.

Provides a unified interface for operator learning using either
FNO or Galerkin-based architectures.
"""

from __future__ import annotations

from typing import Literal, cast

import structlog
from jaxtyping import Float
from torch import Tensor, nn

from src.modeling.fno_layer import FNO2d
from src.modeling.galerkin_operator import Galerkin2d

logger = structlog.get_logger(__name__)


class NeuralOperator(nn.Module):
    """Resolution-independent neural operator for PDE solving.
    
    Supports multiple backends:
    - 'fno': Fourier Neural Operator
    - 'galerkin': Galerkin Attention based (using existing AlphaGalerkin blocks)
    
    Example:
        >>> model = NeuralOperator(in_channels=1, out_channels=1, width=64)
        >>> x = torch.randn(4, 1, 16, 16)  # Train resolution
        >>> y = model(x)
        >>> # Inference at higher resolution
        >>> x_hi = torch.randn(4, 1, 64, 64)
        >>> y_hi = model(x_hi)  # Works without retraining!

    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        n_layers: int = 4,
        modes: int = 12,
        n_heads: int | None = None,
        backend: Literal["fno", "galerkin"] = "fno",
    ) -> None:
        """Initialize neural operator.

        Args:
            in_channels: Input field channels.
            out_channels: Output field channels.
            width: Hidden dimension.
            n_layers: Number of operator layers.
            modes: Fourier modes (for FNO backend).
            n_heads: Attention heads (for Galerkin backend). Default: width // 16.
            backend: Architecture backend.

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backend = backend

        self.model: FNO2d | Galerkin2d
        if backend == "fno":
            self.model = FNO2d(
                in_channels=in_channels,
                out_channels=out_channels,
                width=width,
                modes1=modes,
                modes2=modes,
                n_layers=n_layers,
            )
        elif backend == "galerkin":
            # Use explicit n_heads or derive from width
            galerkin_heads = n_heads if n_heads is not None else max(1, width // 16)
            self.model = Galerkin2d(
                in_channels=in_channels,
                out_channels=out_channels,
                width=width,
                n_layers=n_layers,
                n_heads=galerkin_heads,
            )
        else:
            raise NotImplementedError(f"Backend '{backend}' not yet implemented")

        logger.info(
            "neural_operator_initialized",
            backend=backend,
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            n_layers=n_layers,
        )

    def forward(
        self,
        x: Float[Tensor, "batch c h w"],
        coords: Float[Tensor, "batch h w 2"] | None = None,
    ) -> Float[Tensor, "batch c h w"]:
        """Forward pass.

        Args:
            x: Input field.
            coords: Optional coordinate grid.

        Returns:
            Predicted output field.

        """
        return cast(Tensor, self.model(x, coords))

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
