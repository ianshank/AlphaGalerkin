"""Attention mechanisms for AlphaGalerkin.

Implements two types of attention:
1. GalerkinAttention: O(N) linear attention for global influence modeling
2. SoftmaxAttention: Standard attention for local tactical precision

The Galerkin attention approximates the integral operator (Green's function)
while softmax attention preserves injectivity for precise local reading.
"""

from __future__ import annotations

import math

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn


class GalerkinAttention(nn.Module):
    """Galerkin Linear Attention for O(N) global influence modeling.

    Implements the Petrov-Galerkin projection:
        Output = Q * (K^T V / n)

    where the 1/n normalization is the Monte Carlo integral approximation.

    This is mathematically equivalent to solving:
        (Kf)(x) = integral_Omega K(x, xi) f(xi) d(xi)

    Key properties:
    - O(N) complexity instead of O(N^2)
    - Resolution independent (no hard-coded sequence length)
    - Approximates integral operators (Green's function)

    LBB Stability:
    The inf-sup condition requires dim(Key) >= dim(Query).
    We monitor the minimum singular value of K^T K / n.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_key: int | None = None,
        d_value: int | None = None,
        dropout: float = 0.0,
        normalize_features: bool = True,
    ) -> None:
        """Initialize Galerkin attention.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_key: Key/Query dimension per head (default: d_model // n_heads).
            d_value: Value dimension per head (default: d_model // n_heads).
            dropout: Dropout rate.
            normalize_features: Whether to normalize Q and K before attention.

        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_key or d_model // n_heads
        self.d_value = d_value or d_model // n_heads
        self.normalize_features = normalize_features

        # Projections
        self.to_q = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        self.to_k = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        self.to_v = nn.Linear(d_model, n_heads * self.d_value, bias=False)
        self.to_out = nn.Linear(n_heads * self.d_value, d_model)

        self.dropout = nn.Dropout(dropout)

        # For LBB monitoring
        self._last_lbb_constant: Float[Tensor, batch] | None = None

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
        return_lbb: bool = False,
    ) -> Float[Tensor, "batch n d"] | tuple[Float[Tensor, "batch n d"], Float[Tensor, batch]]:
        """Apply Galerkin attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            return_lbb: Whether to return LBB stability constant.

        Returns:
            Output tensor, optionally with LBB constant.

        """
        batch, n, _ = x.shape

        # Project to Q, K, V
        q = self.to_q(x)  # (batch, n, heads * d_key)
        k = self.to_k(x)  # (batch, n, heads * d_key)
        v = self.to_v(x)  # (batch, n, heads * d_value)

        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_heads)

        # Optional feature normalization (helps with stability)
        if self.normalize_features:
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)

        # Galerkin projection: Q * (K^T V / n)
        # Step 1: K^T V (Monte Carlo integral)
        # Shape: (batch, heads, d_key, d_value)
        context = einsum(k, v, "b h n k, b h n v -> b h k v") / n

        # Step 2: Q * Context
        # Shape: (batch, heads, n, d_value)
        output = einsum(q, context, "b h n q, b h q v -> b h n v")

        # Reshape back
        output = rearrange(output, "b h n d -> b n (h d)")

        # Output projection
        output = self.to_out(output)
        output = self.dropout(output)

        if return_lbb:
            # Compute LBB constant
            lbb = self._compute_lbb_constant(k)
            self._last_lbb_constant = lbb
            return output, lbb

        return output

    def _compute_lbb_constant(
        self,
        k: Float[Tensor, "batch heads n d_key"],
    ) -> Float[Tensor, batch]:
        """Compute LBB stability constant (minimum singular value).

        Args:
            k: Key tensor.

        Returns:
            Minimum singular value across heads for each batch.

        """
        batch, heads, n, d_key = k.shape

        # Compute Gram matrix K^T K / n for each head
        gram = einsum(k, k, "b h n i, b h n j -> b h i j") / n

        # Average over heads for overall stability
        gram_avg = gram.mean(dim=1)

        # Compute singular values
        singular_values = torch.linalg.svdvals(gram_avg)

        # Return minimum singular value
        return singular_values.min(dim=-1).values


class SoftmaxAttention(nn.Module):
    """Standard scaled dot-product attention for tactical precision.

    While Galerkin attention is efficient for global influence,
    softmax attention preserves injectivity - critical for precise
    local reading in life & death situations.

    Uses the standard formulation:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_key: int | None = None,
        d_value: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize softmax attention.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_key: Key/Query dimension per head.
            d_value: Value dimension per head.
            dropout: Dropout rate.

        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_key or d_model // n_heads
        self.d_value = d_value or d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_key)

        # Projections
        self.to_q = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        self.to_k = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        self.to_v = nn.Linear(d_model, n_heads * self.d_value, bias=False)
        self.to_out = nn.Linear(n_heads * self.d_value, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
        mask: Float[Tensor, "batch n n"] | None = None,
    ) -> Float[Tensor, "batch n d"]:
        """Apply softmax attention.

        Args:
            x: Input tensor.
            mask: Optional attention mask.

        Returns:
            Output tensor.

        """
        batch, n, _ = x.shape

        # Project to Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_heads)

        # Scaled dot-product attention
        attn_scores = einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax normalization
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = einsum(attn_weights, v, "b h i j, b h j d -> b h i d")

        # Reshape and project output
        output = rearrange(output, "b h n d -> b n (h d)")
        output = self.to_out(output)

        return output


class HybridAttention(nn.Module):
    """Hybrid attention combining Galerkin (global) and Softmax (local).

    For Go, we need both:
    - Global influence patterns (territory, thickness) -> Galerkin
    - Local tactical precision (life & death) -> Softmax

    This layer applies both and combines them with a learnable gate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        galerkin_ratio: float = 0.7,
        learnable_gate: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initialize hybrid attention.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            galerkin_ratio: Initial ratio for Galerkin vs Softmax.
            learnable_gate: Whether the gate is learnable.
            dropout: Dropout rate.

        """
        super().__init__()
        self.galerkin = GalerkinAttention(d_model, n_heads, dropout=dropout)
        self.softmax = SoftmaxAttention(d_model, n_heads, dropout=dropout)

        if learnable_gate:
            self.gate = nn.Parameter(torch.tensor(galerkin_ratio))
        else:
            self.register_buffer("gate", torch.tensor(galerkin_ratio))

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, "batch n d"]:
        """Apply hybrid attention.

        Args:
            x: Input tensor.

        Returns:
            Weighted combination of Galerkin and Softmax attention.

        """
        # Apply both attention types
        galerkin_out = self.galerkin(x)
        softmax_out = self.softmax(x)

        # Combine with gate (sigmoid for [0, 1] range)
        gate = torch.sigmoid(self.gate)
        output = gate * galerkin_out + (1 - gate) * softmax_out

        return output
