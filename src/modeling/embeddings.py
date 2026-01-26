"""Continuous embeddings for resolution-independent Go representation.

The board is treated as a continuous domain Omega = [0,1]^2. Stone positions
are mapped to continuous coordinates and projected onto a Fourier basis.
This enables zero-shot transfer between different board sizes.
"""

from __future__ import annotations

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from src.math_kernel.basis import FourierBasis, create_grid_coordinates


class FourierFeatures(nn.Module):
    """Fourier feature encoding for positional information.

    Maps 2D coordinates to high-dimensional feature space using
    random Fourier features. This is essential for the neural operator
    to learn high-frequency functions.

    The encoding is:
        gamma(x) = [cos(2*pi*B*x), sin(2*pi*B*x)]

    where B is a learnable or fixed frequency matrix.
    """

    def __init__(
        self,
        n_features: int,
        scale: float = 1.0,
        learnable: bool = True,
        include_coordinates: bool = True,
    ) -> None:
        """Initialize Fourier features.

        Args:
            n_features: Number of Fourier frequencies.
            scale: Standard deviation for frequency initialization.
            learnable: Whether frequencies are learnable.
            include_coordinates: Whether to concatenate raw coordinates.

        """
        super().__init__()
        self.n_features = n_features
        self.include_coordinates = include_coordinates

        self.fourier_basis = FourierBasis(
            n_features=n_features,
            scale=scale,
            learnable=learnable,
        )

    @property
    def output_dim(self) -> int:
        """Output dimension of Fourier features."""
        dim = 2 * self.n_features  # cos + sin
        if self.include_coordinates:
            dim += 2  # raw x, y
        return dim

    def forward(
        self,
        coords: Float[Tensor, "batch n 2"],
    ) -> Float[Tensor, "batch n features"]:
        """Encode coordinates with Fourier features.

        Args:
            coords: Normalized coordinates in [0, 1]^2.

        Returns:
            Fourier feature embeddings.

        """
        # Compute Fourier features
        features = self.fourier_basis(coords)

        # Optionally include raw coordinates
        if self.include_coordinates:
            features = torch.cat([coords, features], dim=-1)

        return features


class ContinuousEmbedding(nn.Module):
    """Continuous embedding layer for Go board representation.

    Combines:
    1. Input feature projection (stone colors, move history, etc.)
    2. Fourier positional encoding (resolution-independent)
    3. Optional learnable per-position embedding

    This design ensures no hard-coded board sizes in the model.
    """

    def __init__(
        self,
        input_channels: int,
        d_model: int,
        n_fourier_features: int = 128,
        fourier_scale: float = 1.0,
        use_learnable_positions: bool = False,
    ) -> None:
        """Initialize continuous embedding.

        Args:
            input_channels: Number of input feature channels.
            d_model: Output embedding dimension.
            n_fourier_features: Number of Fourier feature frequencies.
            fourier_scale: Scale for Fourier features.
            use_learnable_positions: Whether to add learnable position embedding
                (NOTE: This breaks resolution independence! Use only for comparison.)

        """
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model
        self.use_learnable_positions = use_learnable_positions

        # Fourier positional encoding
        self.fourier_features = FourierFeatures(
            n_features=n_fourier_features,
            scale=fourier_scale,
            learnable=True,
        )

        # Input projection
        # Combines input channels with Fourier features
        fourier_dim = self.fourier_features.output_dim
        self.input_projection = nn.Linear(
            input_channels + fourier_dim, d_model
        )

        # Optional learnable position embedding (for baseline comparison)
        if use_learnable_positions:
            # This will be dynamically created based on board size
            self._position_embedding: nn.Parameter | None = None
            self._position_embedding_size: int = 0

    def _get_position_embedding(
        self,
        n_positions: int,
        device: torch.device,
    ) -> Float[Tensor, "1 n d"]:
        """Get or create learnable position embedding.

        Args:
            n_positions: Number of positions.
            device: Target device.

        Returns:
            Position embedding tensor.

        """
        if not self.use_learnable_positions:
            return torch.zeros(1, n_positions, self.d_model, device=device)

        # Create or resize position embedding
        if (
            self._position_embedding is None
            or self._position_embedding_size != n_positions
        ):
            # Create new embedding
            self._position_embedding = nn.Parameter(
                torch.randn(1, n_positions, self.d_model) * 0.02
            ).to(device)
            self._position_embedding_size = n_positions

        return self._position_embedding

    def forward(
        self,
        x: Float[Tensor, "batch channels height width"],
        coords: Float[Tensor, "batch n 2"] | None = None,
    ) -> Float[Tensor, "batch n d_model"]:
        """Embed board state with continuous positional encoding.

        Args:
            x: Input board state (batch, channels, height, width).
            coords: Optional pre-computed coordinates. If None, creates
                   uniform grid coordinates.

        Returns:
            Embedded sequence (batch, n, d_model) where n = height * width.

        """
        batch_size, channels, height, width = x.shape
        n_positions = height * width

        # Create coordinates if not provided
        if coords is None:
            # Assume square board
            assert height == width, "Non-square boards require explicit coordinates"
            coords = create_grid_coordinates(
                board_size=height,
                batch_size=batch_size,
                device=x.device,
            )

        # Flatten spatial dimensions to sequence
        x_flat = rearrange(x, "b c h w -> b (h w) c")

        # Compute Fourier features for positions
        pos_features = self.fourier_features(coords)

        # Concatenate input features with positional encoding
        combined = torch.cat([x_flat, pos_features], dim=-1)

        # Project to model dimension
        embeddings = self.input_projection(combined)

        # Add learnable position embedding if enabled
        if self.use_learnable_positions:
            pos_emb = self._get_position_embedding(n_positions, x.device)
            embeddings = embeddings + pos_emb

        return embeddings


class StoneEmbedding(nn.Module):
    """Embedding layer specifically for Go stone representation.

    Provides semantic embeddings for:
    - Empty intersections
    - Black stones
    - White stones
    - Special markers (ko, liberty counts, etc.)
    """

    def __init__(
        self,
        d_model: int,
        n_stone_types: int = 3,  # empty, black, white
        n_special_features: int = 14,  # liberties, history, etc.
    ) -> None:
        """Initialize stone embedding.

        Args:
            d_model: Embedding dimension.
            n_stone_types: Number of stone types (typically 3).
            n_special_features: Number of additional feature planes.

        """
        super().__init__()
        self.d_model = d_model

        # Stone type embedding
        self.stone_embedding = nn.Embedding(n_stone_types, d_model)

        # Special feature projection
        self.feature_projection = nn.Linear(n_special_features, d_model)

        # Combine embeddings
        self.combiner = nn.Linear(2 * d_model, d_model)

    def forward(
        self,
        stone_types: Float[Tensor, "batch height width"],
        features: Float[Tensor, "batch features height width"],
    ) -> Float[Tensor, "batch height*width d_model"]:
        """Embed stone configuration.

        Args:
            stone_types: Stone type indices (0=empty, 1=black, 2=white).
            features: Additional feature planes.

        Returns:
            Stone embeddings.

        """
        batch, height, width = stone_types.shape

        # Flatten spatial dimensions
        stone_flat = rearrange(stone_types.long(), "b h w -> b (h w)")
        features_flat = rearrange(features, "b f h w -> b (h w) f")

        # Embed stone types
        stone_emb = self.stone_embedding(stone_flat)

        # Project special features
        feature_emb = self.feature_projection(features_flat)

        # Combine
        combined = torch.cat([stone_emb, feature_emb], dim=-1)
        output = self.combiner(combined)

        return output
