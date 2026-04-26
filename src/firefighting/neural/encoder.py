"""Fire state encoder for MCTS neural evaluator.

Encodes the fire simulation state (temperature, fuel, wind,
fire perimeter) into a fixed-size latent representation for
the AlphaGalerkin neural architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


@dataclass
class FireStateEncoding:
    """Encoded fire state for MCTS evaluation.

    Attributes:
        tensor: Multi-channel tensor (batch, channels, H, W).
        channels: Channel names.

    """

    tensor: torch.Tensor
    channels: list[str]


class FireStateEncoder(nn.Module):
    """Resolution-independent fire state encoder.

    Maps variable-size 2D fire simulation grids to a fixed
    (C, H_enc, W_enc) tensor, then extracts features.

    Input channels:
    - temperature (normalized)
    - fuel remaining fraction
    - wind speed
    - wind direction (encoded as sin/cos)
    - fire front indicator (from level set)
    - terrain slope
    """

    N_INPUT_CHANNELS = 7
    CHANNEL_NAMES = [
        "temperature",
        "fuel_remaining",
        "wind_speed",
        "wind_dir_sin",
        "wind_dir_cos",
        "fire_front",
        "terrain_slope",
    ]

    def __init__(
        self,
        encode_size: int = 32,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encode_size = encode_size
        self.latent_dim = latent_dim

        self.pool = nn.AdaptiveAvgPool2d((encode_size, encode_size))

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
        """Encode fire state to latent vector.

        Args:
            x: (batch, N_INPUT_CHANNELS, H, W).

        Returns:
            Latent representation (batch, latent_dim).

        """
        x = self.pool(x)
        return self.features(x)

    @staticmethod
    def encode_fire_state(
        temperature: NDArray[np.float64],
        fuel_remaining: NDArray[np.float64],
        wind_u: NDArray[np.float64],
        wind_v: NDArray[np.float64],
        fire_front: NDArray[np.float64] | None = None,
        terrain_slope: NDArray[np.float64] | None = None,
        max_temperature: float = 2000.0,
    ) -> FireStateEncoding:
        """Convert numpy fire state to normalized torch tensor."""
        ny, nx = temperature.shape
        if fire_front is None:
            fire_front = np.zeros((ny, nx))
        if terrain_slope is None:
            terrain_slope = np.zeros((ny, nx))

        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        wind_dir = np.arctan2(wind_v, wind_u + 1e-10)

        channels = np.stack(
            [
                temperature / max_temperature,
                fuel_remaining,
                wind_speed / max(wind_speed.max(), 1e-10),
                np.sin(wind_dir),
                np.cos(wind_dir),
                fire_front,
                terrain_slope / max(np.abs(terrain_slope).max(), 1e-10)
                if terrain_slope.max() > 0
                else terrain_slope,
            ],
            axis=0,
        )

        tensor = torch.from_numpy(channels.astype(np.float32)).unsqueeze(0)
        return FireStateEncoding(
            tensor=tensor,
            channels=FireStateEncoder.CHANNEL_NAMES,
        )
