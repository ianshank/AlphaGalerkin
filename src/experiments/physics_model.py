"""Neural Operator for Physics Field Prediction.

This model learns to predict the potential field φ from coordinates and charges,
approximating the Green's function of the Laplacian operator.

Architecture:
    Input: (x, y) coordinates + charge density ρ(x,y)
    → Fourier Features (high-frequency encoding)
    → Galerkin Linear Attention layers
    → FNet mixing (optional)
    → Scalar field output φ(x,y)

Key constraints (from template):
    - NO hardcoded grid sizes
    - Input treated as bag of points (B, N, D)
    - 1/N normalization (Monte Carlo integration)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.modeling.attention import GalerkinAttention
from src.modeling.embeddings import FourierFeatures
from src.modeling.fnet import FNetBlock


class PhysicsOperator(nn.Module):
    """Neural operator for learning the Green's function.

    Maps (coordinates, charges) → potential field.
    Resolution-independent: trained on 9x9, runs on any N x N.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_fourier_features: int = 64,
        fourier_scale: float = 10.0,
        use_fnet: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """Initialize physics operator.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of Galerkin layers.
            n_fourier_features: Number of Fourier feature frequencies.
            fourier_scale: Scale for Fourier features (controls frequency).
            use_fnet: Whether to use FNet mixing layers.
            dropout: Dropout rate.

        """
        super().__init__()

        self.d_model = d_model
        self.use_fnet = use_fnet

        # Fourier feature encoding for coordinates
        # Maps (x, y) → [sin(ωx), cos(ωx), sin(ωy), cos(ωy), ...]
        self.coord_encoder = FourierFeatures(
            input_dim=2,  # (x, y) coordinates
            n_features=n_fourier_features,
            scale=fourier_scale,
        )

        # Input projection: Fourier features + charge → d_model
        fourier_output_dim = 2 * n_fourier_features  # sin + cos
        self.input_proj = nn.Linear(fourier_output_dim + 1, d_model)  # +1 for charge

        # Galerkin attention layers (O(N) complexity)
        self.galerkin_layers = nn.ModuleList([
            GalerkinBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Optional FNet layers for mixing
        if use_fnet:
            self.fnet_layers = nn.ModuleList([
                FNetBlock(d_model, d_ffn=4 * d_model, dropout=dropout)
                for _ in range(n_layers // 2)
            ])
        else:
            self.fnet_layers = None

        # Output head: project to scalar potential
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        coords: Tensor,  # Float[batch, n_points, 2]
        charges: Tensor,  # Float[batch, n_points]
    ) -> Tensor:  # Float[batch, n_points]
        """Forward pass: predict potential from coords and charges.

        Args:
            coords: Point coordinates (batch, n_points, 2), normalized to [0, 1].
            charges: Charge density (batch, n_points) at each point.

        Returns:
            Predicted potential (batch, n_points) at each point.

        """
        batch_size, n_points, _ = coords.shape

        # Encode coordinates with Fourier features
        coord_features = self.coord_encoder(coords)  # (B, N, 2*F)

        # Concatenate with charges
        charges_expanded = charges.unsqueeze(-1)  # (B, N, 1)
        features = torch.cat([coord_features, charges_expanded], dim=-1)

        # Project to model dimension
        x = self.input_proj(features)  # (B, N, D)

        # Apply Galerkin layers
        for i, galerkin in enumerate(self.galerkin_layers):
            x = galerkin(x)

            # Interleave with FNet if enabled
            if self.fnet_layers is not None and i < len(self.fnet_layers):
                # FNet needs to know the spatial structure
                # For bag-of-points, we treat it as 1D sequence
                x = self.fnet_layers[i](x, grid_size=1)  # 1D mode

        # Output projection
        potential = self.output_head(x).squeeze(-1)  # (B, N)

        return potential


class GalerkinBlock(nn.Module):
    """Galerkin attention block with LayerNorm and FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize Galerkin block."""
        super().__init__()

        d_ffn = d_ffn or 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GalerkinAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,  # Float[batch, n, d]
    ) -> Tensor:  # Float[batch, n, d]
        """Forward with pre-norm and residual connections."""
        # Attention with residual
        x = x + self.attn(self.norm1(x))

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class PhysicsLoss(nn.Module):
    """Loss function for physics field prediction.

    Combines MSE loss with optional physics-informed regularization.
    """

    def __init__(
        self,
        physics_weight: float = 0.0,
    ) -> None:
        """Initialize loss.

        Args:
            physics_weight: Weight for physics regularization (Laplacian constraint).

        """
        super().__init__()
        self.physics_weight = physics_weight

    def forward(
        self,
        pred: Tensor,  # Float[batch, n]
        target: Tensor,  # Float[batch, n]
        charges: Tensor | None = None,  # Float[batch, n]
        coords: Tensor | None = None,  # Float[batch, n, 2]
    ) -> Tensor:  # Float[]
        """Compute loss.

        Args:
            pred: Predicted potential (batch, n).
            target: Ground truth potential (batch, n).
            charges: Optional charges (batch, n) for physics regularization.
            coords: Optional coordinates (batch, n, 2) for physics regularization.

        Returns:
            Scalar loss value.

        """
        # MSE loss
        mse = torch.nn.functional.mse_loss(pred, target)

        # Optional physics regularization (not used in basic PoC)
        if self.physics_weight > 0 and charges is not None:
            # Could add Laplacian constraint here
            pass

        return mse
