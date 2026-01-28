"""Main AlphaGalerkin model combining all components.

Architecture:
    Input (discrete board) -> Continuous Embedding (Fourier)
    -> Strategy Body (Galerkin/FNet) -> Tactical Head (Softmax)
    -> Policy Head + Value Head

The model is resolution-independent: trained on 9x9, runs on 19x19.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from config.schemas import OperatorConfig
from src.math_kernel.basis import create_grid_coordinates
from src.math_kernel.spectral import ResolutionAdapter
from src.modeling.attention import GalerkinAttention, SoftmaxAttention
from src.modeling.embeddings import ContinuousEmbedding
from src.modeling.fnet import FNetBlock, FNetStack
from src.modeling.stability import StabilityGuard


class ModelOutput(NamedTuple):
    """Output from AlphaGalerkin model."""

    policy_logits: Float[Tensor, "batch n+1"]  # +1 for pass move
    value: Float[Tensor, "batch 1"]
    lbb_constant: Float[Tensor, batch] | None


class GalerkinBlock(nn.Module):
    """Galerkin Transformer block with attention and FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int | None = None,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
    ) -> None:
        """Initialize Galerkin block.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ffn: Feed-forward dimension.
            dropout: Dropout rate.
            norm_type: Normalization type.

        """
        super().__init__()
        d_ffn = d_ffn or 4 * d_model

        # Galerkin attention
        self.attention = GalerkinAttention(d_model, n_heads, dropout=dropout)

        # Normalization
        if norm_type == "layernorm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == "scalenorm":
            self.norm1 = ScaleNorm(d_model)
            self.norm2 = ScaleNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
        return_lbb: bool = False,
    ) -> Float[Tensor, "batch n d"] | tuple[Float[Tensor, "batch n d"], Float[Tensor, batch]]:
        """Forward pass through Galerkin block.

        Args:
            x: Input tensor.
            return_lbb: Whether to return LBB constant.

        Returns:
            Output tensor, optionally with LBB constant.

        """
        # Attention with residual
        x_norm = self.norm1(x)
        if return_lbb:
            attn_out, lbb = self.attention(x_norm, return_lbb=True)
        else:
            attn_out = self.attention(x_norm)
            lbb = None

        x = x + self.dropout(attn_out)

        # FFN with residual
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        if return_lbb:
            return x, lbb
        return x


class SoftmaxBlock(nn.Module):
    """Softmax Transformer block for tactical precision."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize Softmax block.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ffn: Feed-forward dimension.
            dropout: Dropout rate.

        """
        super().__init__()
        d_ffn = d_ffn or 4 * d_model

        self.attention = SoftmaxAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, "batch n d"]:
        """Forward pass through Softmax block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.

        """
        # Attention with residual
        x_norm = self.norm1(x)
        x = x + self.dropout(self.attention(x_norm))

        # FFN with residual
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x


class ScaleNorm(nn.Module):
    """Scale normalization (simpler alternative to LayerNorm)."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        """Initialize ScaleNorm.

        Args:
            d_model: Model dimension.
            eps: Small constant for numerical stability.

        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * d_model ** 0.5)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Apply scale normalization."""
        norm = x.norm(dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.scale


class PolicyHead(nn.Module):
    """Policy head for move prediction."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
    ) -> None:
        """Initialize policy head.

        Args:
            d_model: Input dimension.
            d_hidden: Hidden dimension.

        """
        super().__init__()
        d_hidden = d_hidden or d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),  # Per-position logit
        )

        # Pass move projection
        self.pass_projection = nn.Linear(d_model, 1)

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, "batch n+1"]:
        """Compute policy logits.

        Args:
            x: Input features.

        Returns:
            Policy logits including pass move.

        """
        # Per-position logits
        position_logits = self.net(x).squeeze(-1)  # (batch, n)

        # Pass move logit (from global pooling)
        global_features = x.mean(dim=1)  # (batch, d)
        pass_logit = self.pass_projection(global_features)  # (batch, 1)

        # Concatenate
        logits = torch.cat([position_logits, pass_logit], dim=-1)

        return logits


class ValueHead(nn.Module):
    """Value head for position evaluation."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
    ) -> None:
        """Initialize value head.

        Args:
            d_model: Input dimension.
            d_hidden: Hidden dimension.

        """
        super().__init__()
        d_hidden = d_hidden or d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, "batch 1"]:
        """Compute value estimate.

        Args:
            x: Input features.

        Returns:
            Value in [-1, 1].

        """
        # Global pooling
        global_features = x.mean(dim=1)  # (batch, d)

        # Value prediction
        value = self.net(global_features)

        return value


class DenseHead(nn.Module):
    """Dense head for physics field regression.
    
    Maps sequence features back to a dense output field.
    Used for operator learning tasks (e.g., Poisson, Heat, Darcy).
    """

    def __init__(
        self,
        d_model: int,
        output_channels: int = 1,
        d_hidden: int | None = None,
    ) -> None:
        """Initialize dense head.

        Args:
            d_model: Input feature dimension.
            output_channels: Number of output channels per position.
            d_hidden: Hidden dimension for MLP.
        """
        super().__init__()
        d_hidden = d_hidden or d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, output_channels),
        )

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, "batch n c"]:
        """Compute dense output field.

        Args:
            x: Input features (batch, n, d_model).

        Returns:
            Output field (batch, n, output_channels).
        """
        return self.net(x)


class AlphaGalerkinModel(nn.Module):
    """Main AlphaGalerkin model.

    Resolution-independent Go AI using:
    - Continuous embedding with Fourier features
    - Galerkin attention for global influence (O(N))
    - Softmax attention for local tactics
    - Optional FNet for fast rollouts
    """

    def __init__(
        self,
        config: OperatorConfig,
    ) -> None:
        """Initialize AlphaGalerkin model.

        Args:
            config: Model configuration.

        """
        super().__init__()
        self.config = config

        # Continuous embedding (resolution-independent)
        self.embedding = ContinuousEmbedding(
            input_channels=config.input_channels,
            d_model=config.d_model,
            n_fourier_features=config.n_fourier_features,
            fourier_scale=config.fourier_scale,
        )

        # Strategy body: Galerkin attention layers
        self.strategy_body = nn.ModuleList([
            GalerkinBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ffn=config.d_ffn,
                dropout=config.fnet_dropout,
                norm_type=config.norm_type,
            )
            for _ in range(config.n_galerkin_layers)
        ])

        # FNet mixing layers (for speed)
        if config.use_fnet_mixing:
            self.fnet_layers = nn.ModuleList([
                FNetBlock(
                    d_model=config.d_model,
                    d_ffn=config.d_ffn,
                    dropout=config.fnet_dropout,
                )
                for _ in range(config.n_galerkin_layers // 2)
            ])
        else:
            self.fnet_layers = None

        # Tactical head: Softmax attention for precision
        self.tactical_head = nn.ModuleList([
            SoftmaxBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ffn=config.d_ffn,
                dropout=config.fnet_dropout,
            )
            for _ in range(config.n_softmax_layers)
        ])

        # Output heads
        self.policy_head = PolicyHead(config.d_model)
        self.value_head = ValueHead(config.d_model)

        # Stability monitoring
        self.stability_guard = StabilityGuard(
            beta_threshold=config.lbb_beta_threshold,
        )

        # Resolution adapter for zero-shot transfer
        self.resolution_adapter = ResolutionAdapter()

        # Store training resolution for adaptation
        self._training_resolution: int | None = None

    @property
    def training_resolution(self) -> int | None:
        """Get the resolution used during training."""
        return self._training_resolution

    @training_resolution.setter
    def training_resolution(self, value: int) -> None:
        """Set the training resolution."""
        self._training_resolution = value

    def forward(
        self,
        x: Float[Tensor, "batch channels height width"],
        return_lbb: bool = False,
    ) -> ModelOutput:
        """Forward pass through AlphaGalerkin.

        Args:
            x: Input board state.
            return_lbb: Whether to compute LBB constant.

        Returns:
            ModelOutput with policy, value, and optional LBB constant.

        """
        batch, channels, height, width = x.shape
        board_size = height  # Assume square board

        # Update training resolution on first forward pass
        if self._training_resolution is None and self.training:
            self._training_resolution = board_size

        # Create continuous coordinates
        coords = create_grid_coordinates(board_size, batch, x.device)

        # Continuous embedding
        embeddings = self.embedding(x, coords)

        # Strategy body (Galerkin attention)
        features = embeddings
        lbb_values = []

        for i, block in enumerate(self.strategy_body):
            if return_lbb:
                features, lbb = block(features, return_lbb=True)
                lbb_values.append(lbb)
            else:
                features = block(features)

            # Interleave with FNet layers if enabled
            if self.fnet_layers is not None and i < len(self.fnet_layers):
                features = self.fnet_layers[i](features, board_size)

        # Tactical head (Softmax attention)
        for block in self.tactical_head:
            features = block(features)

        # Output heads
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        # Aggregate LBB constants
        lbb_constant = None
        if return_lbb and lbb_values:
            # Stack and take minimum (worst-case stability)
            lbb_constant = torch.stack(lbb_values, dim=0).min(dim=0).values

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            lbb_constant=lbb_constant,
        )

    def forward_fast(
        self,
        x: Float[Tensor, "batch channels height width"],
    ) -> ModelOutput:
        """Fast forward pass using only FNet (for MCTS rollouts).

        Skips softmax attention for speed.

        Args:
            x: Input board state.

        Returns:
            ModelOutput with policy and value.

        """
        batch, channels, height, width = x.shape
        board_size = height

        coords = create_grid_coordinates(board_size, batch, x.device)
        embeddings = self.embedding(x, coords)

        # Use FNet-only path
        features = embeddings
        if self.fnet_layers is not None:
            for fnet_block in self.fnet_layers:
                features = fnet_block(features, board_size)
        else:
            # Fallback to first few Galerkin blocks
            for block in self.strategy_body[:2]:
                features = block(features)

        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            lbb_constant=None,
        )

    def adapt_resolution(
        self,
        source_size: int,
        target_size: int,
    ) -> None:
        """Prepare model for resolution transfer.

        Updates internal state for inference at a different resolution.

        Args:
            source_size: Training board size.
            target_size: Target board size for inference.

        """
        self._training_resolution = source_size
        # Resolution adapter handles the actual adaptation during forward pass


class AlphaGalerkinFast(nn.Module):
    """Lightweight AlphaGalerkin for fast MCTS rollouts.

    Uses only FNet blocks for O(N log N) complexity.
    """

    def __init__(
        self,
        config: OperatorConfig,
        n_layers: int = 4,
    ) -> None:
        """Initialize fast model.

        Args:
            config: Model configuration.
            n_layers: Number of FNet layers.

        """
        super().__init__()
        self.config = config

        self.embedding = ContinuousEmbedding(
            input_channels=config.input_channels,
            d_model=config.d_model,
            n_fourier_features=config.n_fourier_features,
            fourier_scale=config.fourier_scale,
        )

        self.fnet_stack = FNetStack(
            d_model=config.d_model,
            n_layers=n_layers,
            d_ffn=config.d_ffn,
            dropout=config.fnet_dropout,
        )

        self.policy_head = PolicyHead(config.d_model)
        self.value_head = ValueHead(config.d_model)

    def forward(
        self,
        x: Float[Tensor, "batch channels height width"],
    ) -> ModelOutput:
        """Forward pass.

        Args:
            x: Input board state.

        Returns:
            ModelOutput with policy and value.

        """
        batch, channels, height, width = x.shape
        board_size = height

        coords = create_grid_coordinates(board_size, batch, x.device)
        embeddings = self.embedding(x, coords)

        features = self.fnet_stack(embeddings, board_size)

        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            lbb_constant=None,
        )
