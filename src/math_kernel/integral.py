"""Integral approximation methods for Galerkin projection.

The Galerkin method solves integral equations by projecting onto a finite
basis. For the continuous operator formulation, we approximate integrals
using Monte Carlo methods, which naturally adapt to any resolution.
"""

from __future__ import annotations

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn


class MonteCarloIntegral(nn.Module):
    """Monte Carlo integration for Galerkin projection.

    Approximates the integral:
        I[f] = integral_Omega f(x) dx

    Using Monte Carlo sampling:
        I[f] approx (|Omega| / N) * sum_{i=1}^N f(x_i)

    For uniform grids on [0,1]^2, this simplifies to 1/N normalization.
    """

    def __init__(self) -> None:
        """Initialize Monte Carlo integrator."""
        super().__init__()

    def integrate(
        self,
        values: Float[Tensor, "batch n ..."],
        weights: Float[Tensor, "batch n"] | None = None,
    ) -> Float[Tensor, "batch ..."]:
        """Compute Monte Carlo integral over spatial dimension.

        Args:
            values: Function values at sample points.
            weights: Optional quadrature weights (default: uniform 1/n).

        Returns:
            Integrated values.

        """
        n = values.shape[1]

        if weights is None:
            # Uniform weights for Monte Carlo
            return values.mean(dim=1)
        else:
            # Weighted quadrature
            # Normalize weights to sum to 1
            weights = weights / weights.sum(dim=1, keepdim=True)
            return einsum(values, weights, "b n ..., b n -> b ...")

    def forward(
        self,
        values: Float[Tensor, "batch n ..."],
        weights: Float[Tensor, "batch n"] | None = None,
    ) -> Float[Tensor, "batch ..."]:
        """Forward pass (alias for integrate)."""
        return self.integrate(values, weights)


class GalerkinProjection(nn.Module):
    """Galerkin projection for approximating integral operators.

    Given a kernel K(x, xi), the integral operator is:
        (Kf)(x) = integral_Omega K(x, xi) f(xi) d(xi)

    In the Galerkin method, we approximate this by:
    1. Projecting f onto a test basis: <phi_j, f> = integral phi_j(x) f(x) dx
    2. Representing K in the trial/test basis: K_ij = integral integral phi_i K phi_j
    3. Solving the projected system

    For neural operator learning, this becomes the attention mechanism:
        Output = Q * (K^T V / n)

    where the 1/n normalization is the Monte Carlo integral approximation.

    LBB Stability Condition:
    For the projection to be stable, we need:
        inf_{v in V} sup_{u in U} b(u,v) / (||u|| ||v||) >= beta > 0
    This translates to: sigma_min(K) >= beta
    """

    def __init__(
        self,
        d_model: int,
        d_key: int,
        d_value: int,
    ) -> None:
        """Initialize Galerkin projection.

        Args:
            d_model: Input/output dimension.
            d_key: Key/Query dimension (must satisfy LBB: d_key >= d_query).
            d_value: Value dimension.

        """
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value

        # Projections for Q, K, V
        self.to_query = nn.Linear(d_model, d_key)
        self.to_key = nn.Linear(d_model, d_key)
        self.to_value = nn.Linear(d_model, d_value)
        self.to_output = nn.Linear(d_value, d_model)

        # Monte Carlo integrator
        self.integrator = MonteCarloIntegral()

    def project(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, "batch n d"]:
        """Apply Galerkin projection (linear attention).

        Args:
            x: Input tensor.

        Returns:
            Projected output.

        """
        q = self.to_query(x)  # (batch, n, d_key)
        k = self.to_key(x)  # (batch, n, d_key)
        v = self.to_value(x)  # (batch, n, d_value)

        n = x.shape[1]

        # Galerkin projection: Q * (K^T V / n)
        # Step 1: K^T V - project values onto key basis (Monte Carlo integral)
        # This computes: integral K(x, xi) V(xi) d(xi)
        context = einsum(k, v, "b n k, b n v -> b k v") / n

        # Step 2: Q * Context - reconstruct in query basis
        output = einsum(q, context, "b n q, b q v -> b n v")

        # Project back to model dimension
        output = self.to_output(output)

        return output

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, "batch n d"]:
        """Forward pass (alias for project)."""
        return self.project(x)

    def compute_lbb_constant(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, batch]:
        """Compute the LBB stability constant (minimum singular value).

        The LBB condition requires:
            beta = sigma_min(K^T K / n) > 0

        A larger beta indicates better numerical stability.

        Args:
            x: Input tensor for computing key matrix.

        Returns:
            Minimum singular value for each batch element.

        """
        k = self.to_key(x)  # (batch, n, d_key)
        n = x.shape[1]

        # Compute K^T K / n (Gram matrix)
        gram = einsum(k, k, "b n k, b n l -> b k l") / n

        # Compute singular values
        singular_values = torch.linalg.svdvals(gram)

        # Return minimum singular value (LBB constant)
        return singular_values.min(dim=-1).values


class PetrovGalerkinProjection(nn.Module):
    """Petrov-Galerkin projection with different trial and test spaces.

    In Petrov-Galerkin methods, we use different basis functions for
    the trial space (approximation) and test space (residual orthogonality).

    This allows for greater flexibility in satisfying the LBB condition
    by choosing appropriate pairings of trial and test functions.
    """

    def __init__(
        self,
        d_model: int,
        d_trial: int,
        d_test: int,
        d_value: int,
    ) -> None:
        """Initialize Petrov-Galerkin projection.

        Args:
            d_model: Input/output dimension.
            d_trial: Trial space dimension (for Keys).
            d_test: Test space dimension (for Queries).
            d_value: Value dimension.

        """
        super().__init__()
        self.d_model = d_model
        self.d_trial = d_trial
        self.d_test = d_test
        self.d_value = d_value

        # LBB stability requires dim(trial) >= dim(test)
        if d_trial < d_test:
            raise ValueError(
                f"LBB violation: d_trial ({d_trial}) must be >= d_test ({d_test})"
            )

        # Projections
        self.to_query = nn.Linear(d_model, d_test)
        self.to_key = nn.Linear(d_model, d_trial)
        self.to_value = nn.Linear(d_model, d_value)
        self.to_output = nn.Linear(d_value, d_model)

    def project(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, "batch n d"]:
        """Apply Petrov-Galerkin projection.

        Args:
            x: Input tensor.

        Returns:
            Projected output.

        """
        q = self.to_query(x)  # (batch, n, d_test)
        k = self.to_key(x)  # (batch, n, d_trial)
        v = self.to_value(x)  # (batch, n, d_value)

        n = x.shape[1]

        # Petrov-Galerkin projection
        # Context: K^T V / n (project onto trial basis)
        context = einsum(k, v, "b n k, b n v -> b k v") / n

        # To map from trial (d_trial) to test (d_test) space,
        # we need to handle dimension mismatch
        # For now, we truncate/pad the context to match query dimension
        if self.d_trial > self.d_test:
            # Truncate: use first d_test dimensions
            context = context[:, : self.d_test, :]
        elif self.d_trial < self.d_test:
            # Pad: should not happen due to LBB check in __init__
            raise RuntimeError("LBB violation detected at runtime")

        # Reconstruct: Q * Context
        output = einsum(q, context, "b n q, b q v -> b n v")

        return self.to_output(output)

    def forward(
        self,
        x: Float[Tensor, "batch n d"],
    ) -> Float[Tensor, "batch n d"]:
        """Forward pass (alias for project)."""
        return self.project(x)
