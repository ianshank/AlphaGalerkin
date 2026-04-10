"""Flow field encoder for compressible flow MCTS neural evaluator.

Encodes the 2D flow field (density, velocity, pressure, temperature,
species) into a fixed-size latent representation suitable for the
AlphaGalerkin neural architecture.

The encoder is resolution-independent: it maps from arbitrary mesh
sizes to a fixed grid via interpolation, then applies convolutional
feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


@dataclass
class FlowFieldEncoding:
    """Encoded flow field for MCTS evaluation.

    Attributes:
        tensor: Multi-channel tensor (batch, channels, H, W).
        channels: Channel names for interpretability.

    """

    tensor: torch.Tensor
    channels: list[str]


class FlowFieldEncoder(nn.Module):
    """Resolution-independent flow field encoder.

    Maps variable-size 2D flow fields to a fixed (C, H_enc, W_enc)
    tensor via adaptive pooling, then applies a small CNN for
    feature extraction.

    Input channels:
    - density (normalized by freestream)
    - x-velocity (normalized by freestream)
    - y-velocity (normalized by freestream)
    - pressure (normalized by freestream)
    - Mach number
    - shock indicator (from ShockDetector)
    - mesh density (DOF per cell)
    """

    N_INPUT_CHANNELS = 7
    CHANNEL_NAMES = [
        "density",
        "velocity_x",
        "velocity_y",
        "pressure",
        "mach",
        "shock_indicator",
        "mesh_density",
    ]

    def __init__(
        self,
        encode_size: int = 32,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encode_size = encode_size
        self.latent_dim = latent_dim

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((encode_size, encode_size))

        # Feature extraction CNN
        self.features = nn.Sequential(
            nn.Conv2d(self.N_INPUT_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode flow field to latent vector.

        Args:
            x: Input tensor (batch, N_INPUT_CHANNELS, H, W).

        Returns:
            Latent representation (batch, latent_dim).

        """
        x = self.pool(x)
        return self.features(x)

    @staticmethod
    def encode_flow_field(
        density: NDArray[np.float64],
        velocity_x: NDArray[np.float64],
        velocity_y: NDArray[np.float64],
        pressure: NDArray[np.float64],
        mach: NDArray[np.float64],
        shock_indicator: NDArray[np.float64] | None = None,
        freestream_rho: float = 1.0,
        freestream_u: float = 1.0,
        freestream_p: float = 1.0,
    ) -> FlowFieldEncoding:
        """Convert numpy flow fields to normalized torch tensor.

        All quantities are normalized by freestream values to produce
        O(1) inputs for the neural network.
        """
        if shock_indicator is None:
            shock_indicator = np.zeros_like(density)

        mesh_density = np.ones_like(density)

        channels = np.stack(
            [
                density / max(freestream_rho, 1e-30),
                velocity_x / max(freestream_u, 1e-30),
                velocity_y / max(freestream_u, 1e-30),
                pressure / max(freestream_p, 1e-30),
                mach,
                shock_indicator,
                mesh_density,
            ],
            axis=0,
        )  # (7, H, W)

        tensor = torch.from_numpy(channels.astype(np.float32)).unsqueeze(0)

        return FlowFieldEncoding(
            tensor=tensor,
            channels=FlowFieldEncoder.CHANNEL_NAMES,
        )
