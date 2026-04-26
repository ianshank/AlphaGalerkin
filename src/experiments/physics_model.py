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

import structlog
import torch
from torch import Tensor, nn

from src.modeling.attention import GalerkinAttention
from src.modeling.embeddings import FourierFeatures
from src.modeling.fnet import FNetBlock

logger = structlog.get_logger(__name__)


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
        input_dim: int = 2,
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
            input_dim: Spatial dimension of input coordinates (2 for planar,
                3 for volumetric domains such as the helical heat exchanger).

        """
        super().__init__()

        self.d_model = d_model
        self.use_fnet = use_fnet
        self.input_dim = input_dim

        logger.debug(
            "physics_operator_init",
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_fourier_features=n_fourier_features,
            fourier_scale=fourier_scale,
            use_fnet=use_fnet,
            dropout=dropout,
            input_dim=input_dim,
        )

        # Fourier feature encoding for coordinates. Maps an arbitrary-dim
        # input into [coords, sin(B*coords), cos(B*coords)] features, so
        # the same backbone serves both the 2D Poisson and 3D Noyron HX
        # use cases.
        self.coord_encoder = FourierFeatures(
            n_features=n_fourier_features,
            scale=fourier_scale,
            learnable=True,
            include_coordinates=True,
            input_dim=input_dim,
        )

        # Input projection: Fourier features + charge → d_model
        # output_dim = 2 * n_fourier_features (sin + cos) + 2 (raw coords)
        fourier_output_dim = self.coord_encoder.output_dim
        self.input_proj = nn.Linear(fourier_output_dim + 1, d_model)  # +1 for charge

        # Galerkin attention layers (O(N) complexity)
        self.galerkin_layers = nn.ModuleList(
            [GalerkinBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )

        # Optional FNet layers for mixing
        if use_fnet:
            self.fnet_layers = nn.ModuleList(
                [
                    FNetBlock(d_model, d_ffn=4 * d_model, dropout=dropout)
                    for _ in range(n_layers // 2)
                ]
            )
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
        coords: Tensor,  # Float[batch, n_points, input_dim]
        charges: Tensor,  # Float[batch, n_points]
    ) -> Tensor:  # Float[batch, n_points]
        """Forward pass: predict potential from coords and charges.

        Args:
            coords: Point coordinates (batch, n_points, input_dim), normalized
                to [0, 1]. input_dim is 2 for planar problems, 3 for
                volumetric (e.g. helical heat exchanger) problems.
            charges: Source-density field (batch, n_points) at each point.

        Returns:
            Predicted potential (batch, n_points) at each point.

        """
        batch_size, n_points, coord_dim = coords.shape
        if coord_dim != self.input_dim:
            raise ValueError(
                f"PhysicsOperator was built for input_dim={self.input_dim}, "
                f"but got coords with last dim {coord_dim}"
            )

        logger.debug(
            "physics_operator_forward_start",
            batch_size=batch_size,
            n_points=n_points,
            coords_shape=list(coords.shape),
            charges_shape=list(charges.shape),
        )

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
                # For bag-of-points, we treat it as 1D sequence (board_size=None)
                x = self.fnet_layers[i](x, board_size=None)  # 1D mode

        # Output projection
        potential = self.output_head(x).squeeze(-1)  # (B, N)

        logger.debug(
            "physics_operator_forward_complete",
            output_shape=list(potential.shape),
            output_finite=bool(potential.isfinite().all()),
        )

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

    Combines MSE loss with optional physics-informed regularization
    via a discrete Laplacian constraint.  When ``physics_weight > 0``
    and coordinates/charges are supplied, the loss penalises
    deviations from the Poisson equation ``-\u0394\u03c6 = \u03c1``.

    The Laplacian is approximated using finite differences on the
    nearest neighbours in the coordinate cloud, making this
    resolution-independent (works on any N-point cloud).
    """

    def __init__(
        self,
        physics_weight: float = 0.0,
        laplacian_eps: float = 1e-6,
    ) -> None:
        """Initialize loss.

        Args:
            physics_weight: Weight for physics regularization (Laplacian constraint).
            laplacian_eps: Small constant added to distance denominators for
                numerical stability.

        """
        super().__init__()
        self.physics_weight = physics_weight
        self.laplacian_eps = laplacian_eps

    @staticmethod
    def _compute_laplacian(
        pred: Tensor,
        coords: Tensor,
        eps: float = 1e-6,
    ) -> Tensor:
        """Approximate the Laplacian via autodiff.

        Uses ``torch.autograd.grad`` to compute second derivatives of
        the predicted field with respect to coordinates.

        Args:
            pred: Predicted potential (batch, n).
            coords: Coordinates (batch, n, 2) **with requires_grad**.
            eps: Stability epsilon (unused here, kept for API symmetry).

        Returns:
            Laplacian estimate (batch, n).

        """
        # pred depends on coords through the model; compute grad
        grad_outputs = torch.ones_like(pred)
        (grad_phi,) = torch.autograd.grad(
            outputs=pred,
            inputs=coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )
        # grad_phi shape: (batch, n, 2) -> [dphi/dx, dphi/dy]
        laplacian = torch.zeros_like(pred)
        for dim_idx in range(coords.shape[-1]):
            grad_dim = grad_phi[..., dim_idx]
            # For linear functions the first derivative is constant (no grad_fn).
            # In that case the second derivative is zero — nothing to add.
            if grad_dim.grad_fn is None and not grad_dim.requires_grad:
                continue
            grad_dim_outputs = torch.ones_like(grad_dim)
            try:
                grad2_result = torch.autograd.grad(
                    outputs=grad_dim,
                    inputs=coords,
                    grad_outputs=grad_dim_outputs,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            except RuntimeError:
                # Second derivative is zero (e.g., linear function in this dim)
                grad2_result = None
            if grad2_result is not None:
                laplacian = laplacian + grad2_result[..., dim_idx]
        return laplacian

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
                Must have ``requires_grad=True`` when ``physics_weight > 0``.

        Returns:
            Scalar loss value (MSE + weighted Laplacian residual).

        """
        # MSE loss
        mse = torch.nn.functional.mse_loss(pred, target)

        # Physics-informed Laplacian regularization: -Lap(phi) = rho
        if self.physics_weight > 0 and charges is not None and coords is not None:
            try:
                laplacian = self._compute_laplacian(pred, coords, eps=self.laplacian_eps)
                # Poisson residual: -Lap(phi) - rho should be zero
                residual = (-laplacian) - charges
                physics_loss = torch.mean(residual**2)
                mse = mse + self.physics_weight * physics_loss

                logger.debug(
                    "physics_loss_computed",
                    mse=mse.item(),
                    physics_loss=physics_loss.item(),
                    weight=self.physics_weight,
                )
            except RuntimeError as e:
                # Autograd may fail if coords don't require grad
                logger.debug(
                    "physics_loss_skipped",
                    reason=str(e),
                )

        return mse
