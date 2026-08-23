"""Fourier Neural Operator layers.

Implements spectral convolution in Fourier space for resolution-independent
operator learning.
"""

from __future__ import annotations

from typing import cast

import structlog
import torch
from jaxtyping import Float
from torch import Tensor, nn

logger = structlog.get_logger(__name__)


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution via FFT.

    Performs convolution in Fourier space by learning complex weights
    for the low-frequency modes. This enables resolution independence.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 12,
        modes2: int = 12,
    ) -> None:
        """Initialize spectral convolution.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            modes1: Fourier modes in first dimension.
            modes2: Fourier modes in second dimension.

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Scale for initialization
        scale = 1 / (in_channels * out_channels)

        # Complex weights for Fourier modes
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

        logger.debug(
            "spectral_conv_initialized",
            in_channels=in_channels,
            out_channels=out_channels,
            modes=(modes1, modes2),
        )

    def compl_mul2d(
        self,
        input: Float[Tensor, "batch in h w"],
        weights: Float[Tensor, "in out m1 m2"],
    ) -> Float[Tensor, "batch out h w"]:
        """Complex multiplication in Fourier space."""
        # (batch, in, x, y), (in, out, x, y) -> (batch, out, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(
        self,
        x: Float[Tensor, "batch c h w"],
    ) -> Float[Tensor, "batch c h w"]:
        """Forward pass through spectral convolution.

        Args:
            x: Input tensor (batch, channels, height, width).

        Returns:
            Output tensor (batch, channels, height, width).

        """
        batchsize = x.shape[0]

        # Transform to Fourier domain
        x_ft = torch.fft.rfft2(x)

        # Compute convolution for low-frequency modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Dynamic modes (clamp to input resolution)
        modes1 = min(x_ft.size(-2), self.modes1)
        modes2 = min(x_ft.size(-1), self.modes2)

        # Apply weights to low-frequency modes
        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2]
        )
        out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(
            x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, :modes1, :modes2]
        )

        # Transform back to spatial domain
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x


class FNOBlock(nn.Module):
    """Fourier Neural Operator block.

    Combines spectral convolution with a local linear transform
    and non-linearity.
    """

    def __init__(
        self,
        width: int,
        modes1: int = 12,
        modes2: int = 12,
        activation: str = "gelu",
    ) -> None:
        """Initialize FNO block.

        Args:
            width: Channel width.
            modes1: Fourier modes dim 1.
            modes2: Fourier modes dim 2.
            activation: Activation function name.

        """
        super().__init__()

        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.local_conv = nn.Conv2d(width, width, kernel_size=1)

        self.activation: nn.Module
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

    def forward(
        self,
        x: Float[Tensor, "batch c h w"],
    ) -> Float[Tensor, "batch c h w"]:
        """Forward pass with residual connection."""
        # Parallel spectral and local paths
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)

        # Combine and activate
        return cast(Tensor, self.activation(x1 + x2))


class FNO2d(nn.Module):
    """Full Fourier Neural Operator for 2D problems.

    Maps input field a(x) -> output field u(x).
    Resolution-independent by design.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
        activation: str = "gelu",
    ) -> None:
        """Initialize FNO.

        Args:
            in_channels: Input field channels.
            out_channels: Output field channels.
            width: Hidden dimension width.
            modes1: Fourier modes dim 1.
            modes2: Fourier modes dim 2.
            n_layers: Number of FNO blocks.
            activation: Activation function.

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_layers = n_layers

        # Lift input to hidden dimension
        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for coordinates

        # FNO layers
        self.layers = nn.ModuleList(
            [FNOBlock(width, modes1, modes2, activation) for _ in range(n_layers)]
        )

        # Project back to output channels
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        self.activation: nn.Module
        if activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        logger.debug(
            "fno2d_initialized",
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            n_layers=n_layers,
            modes=(modes1, modes2),
            activation=activation,
        )

    def forward(
        self,
        x: Float[Tensor, "batch c h w"],
        coords: Float[Tensor, "batch h w 2"] | None = None,
    ) -> Float[Tensor, "batch c h w"]:
        """Forward pass through FNO.

        Args:
            x: Input field (batch, in_channels, h, w).
            coords: Optional grid coordinates (batch, h, w, 2).

        Returns:
            Output field (batch, out_channels, h, w).

        """
        batch_size, _, h, w = x.shape

        logger.debug(
            "fno2d_forward_start",
            batch_size=batch_size,
            input_resolution=(h, w),
            coords_provided=coords is not None,
        )

        # Generate coordinates if not provided
        if coords is None:
            grid_x = torch.linspace(0, 1, h, device=x.device)
            grid_y = torch.linspace(0, 1, w, device=x.device)
            grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
            coords = torch.stack([grid_x, grid_y], dim=-1)
            coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Reshape to (batch, h, w, c)
        x = x.permute(0, 2, 3, 1)

        # Concatenate with coordinates
        x = torch.cat([x, coords], dim=-1)

        # Lift to hidden dimension: (batch, h, w, width)
        x = self.fc0(x)

        # Reshape for convolutions: (batch, width, h, w)
        x = x.permute(0, 3, 1, 2)

        # FNO layers
        for layer in self.layers:
            x = layer(x)

        # Reshape back: (batch, h, w, width)
        x = x.permute(0, 2, 3, 1)

        # Project to output: (batch, h, w, out_channels)
        x = self.fc1(x)
        x = cast(Tensor, self.activation(x))
        x = self.fc2(x)

        # Final shape: (batch, out_channels, h, w)
        x = x.permute(0, 3, 1, 2)

        return x
