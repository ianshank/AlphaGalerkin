"""Self-tests for the closed-form OU / jump-OU references.

The analytic module is the independent oracle every AC compares against, so it
gets its own verification: hand-computed 1D closed forms, an in-test RK4
integration of the Lyapunov ODE (independent of the package's integrators),
λ=0 reduction, and density normalization.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pde.stochastic.analytic import (
    gaussian_density_on_grid,
    jump_ou_covariance,
    jump_ou_mean,
    ou_covariance,
    ou_mean,
)

# Pinned 2D OU problem from the spec.
A_2D = torch.tensor([[-1.0, 0.3], [0.0, -0.8]], dtype=torch.float64)
B_2D = torch.tensor([0.1, -0.2], dtype=torch.float64)
G_2D = torch.diag(torch.tensor([0.4, 0.3], dtype=torch.float64))
M0_2D = torch.tensor([1.0, -0.5], dtype=torch.float64)
P0_2D = torch.diag(torch.tensor([0.3, 0.2], dtype=torch.float64))


class TestOuMean1D:
    def test_matches_hand_closed_form(self):
        theta, m0, t = 1.0, 1.0, 0.7
        result = ou_mean(torch.tensor([[-theta]]), torch.tensor([0.0]), torch.tensor([m0]), t)
        assert abs(float(result[0]) - m0 * math.exp(-theta * t)) < 1e-12

    def test_with_bias_reaches_stationary_mean(self):
        # dm/dt = -θm + b → m(∞) = b/θ
        theta, b = 2.0, 1.0
        result = ou_mean(torch.tensor([[-theta]]), torch.tensor([b]), torch.tensor([0.0]), 20.0)
        assert abs(float(result[0]) - b / theta) < 1e-10

    def test_t_zero_is_identity(self):
        result = ou_mean(torch.tensor([[-1.0]]), torch.tensor([0.5]), torch.tensor([1.5]), 0.0)
        assert abs(float(result[0]) - 1.5) < 1e-14


class TestOuCovariance1D:
    def test_matches_hand_closed_form(self):
        # P(t) = q/(2θ) + (P0 − q/(2θ)) e^{−2θt} with q = g².
        theta, g, p0, t = 1.0, 0.5, 0.5, 0.9
        q = g * g
        expected = q / (2 * theta) + (p0 - q / (2 * theta)) * math.exp(-2 * theta * t)
        result = ou_covariance(
            torch.tensor([[-theta]]), torch.tensor([[q]]), torch.tensor([[p0]]), t
        )
        assert abs(float(result[0, 0]) - expected) < 1e-12

    def test_stationary_limit(self):
        theta, q = 1.5, 0.3
        result = ou_covariance(
            torch.tensor([[-theta]], dtype=torch.float64),
            torch.tensor([[q]], dtype=torch.float64),
            torch.tensor([[1.0]], dtype=torch.float64),
            30.0,
        )
        assert abs(float(result[0, 0]) - q / (2 * theta)) < 1e-10


class TestOuCovariance2D:
    def _rk4_lyapunov(self, a, q, p0, t, n_steps=4000):
        """In-test RK4 for dP/dt = AP + PAᵀ + Q — independent verification."""

        def rhs(p):
            return a @ p + p @ a.T + q

        p = p0.clone()
        h = t / n_steps
        for _ in range(n_steps):
            k1 = rhs(p)
            k2 = rhs(p + 0.5 * h * k1)
            k3 = rhs(p + 0.5 * h * k2)
            k4 = rhs(p + h * k3)
            p = p + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return p

    def test_matches_independent_rk4(self):
        q = G_2D @ G_2D.T
        expected = self._rk4_lyapunov(A_2D, q, P0_2D, 1.0)
        result = ou_covariance(A_2D, q, P0_2D, 1.0)
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-10)

    def test_symmetric_and_psd(self):
        q = G_2D @ G_2D.T
        result = ou_covariance(A_2D, q, P0_2D, 0.5)
        torch.testing.assert_close(result, result.T)
        eigvals = torch.linalg.eigvalsh(result)
        assert bool((eigvals > 0).all())

    def test_mean_2d_matches_independent_rk4(self):
        def rhs(m):
            return A_2D @ m + B_2D

        m = M0_2D.clone()
        n_steps, t = 4000, 1.0
        h = t / n_steps
        for _ in range(n_steps):
            k1 = rhs(m)
            k2 = rhs(m + 0.5 * h * k1)
            k3 = rhs(m + 0.5 * h * k2)
            k4 = rhs(m + h * k3)
            m = m + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        result = ou_mean(A_2D, B_2D, M0_2D, t)
        torch.testing.assert_close(result, m, rtol=1e-9, atol=1e-11)


class TestJumpOu:
    def test_lambda_zero_reduces_to_ou(self):
        a = torch.tensor([[-1.0]])
        q = torch.tensor([[0.09]])
        p0 = torch.tensor([[0.1]])
        mu = torch.tensor([0.5])
        sigma = torch.tensor([[0.04]])
        m0 = torch.tensor([0.0])
        t = 1.0
        torch.testing.assert_close(
            jump_ou_mean(a, torch.tensor([0.0]), 0.0, mu, m0, t),
            ou_mean(a, torch.tensor([0.0]), m0, t),
        )
        torch.testing.assert_close(
            jump_ou_covariance(a, q, 0.0, mu, sigma, p0, t),
            ou_covariance(a, q, p0, t),
        )

    def test_jump_shifts_match_effective_coefficients_1d(self):
        # Pinned jump-OU problem: A=-1, b=0, g=0.3, λ=2, μ=0.5, Σ=0.04.
        theta, g, lam, mu, sigma2 = 1.0, 0.3, 2.0, 0.5, 0.04
        p0, m0, t = 0.1, 0.0, 1.0
        b_eff = lam * mu
        q_eff = g * g + lam * (sigma2 + mu * mu)
        expected_mean = (b_eff / theta) * (1 - math.exp(-theta * t)) + m0 * math.exp(-theta * t)
        expected_cov = q_eff / (2 * theta) + (p0 - q_eff / (2 * theta)) * math.exp(-2 * theta * t)
        f64 = torch.float64
        got_mean = jump_ou_mean(
            torch.tensor([[-theta]], dtype=f64),
            torch.tensor([0.0], dtype=f64),
            lam,
            torch.tensor([mu], dtype=f64),
            torch.tensor([m0], dtype=f64),
            t,
        )
        got_cov = jump_ou_covariance(
            torch.tensor([[-theta]], dtype=f64),
            torch.tensor([[g * g]], dtype=f64),
            lam,
            torch.tensor([mu], dtype=f64),
            torch.tensor([[sigma2]], dtype=f64),
            torch.tensor([[p0]], dtype=f64),
            t,
        )
        assert abs(float(got_mean[0]) - expected_mean) < 1e-12
        assert abs(float(got_cov[0, 0]) - expected_cov) < 1e-12

    @settings(max_examples=20, deadline=None)
    @given(lam=st.floats(min_value=0.0, max_value=5.0))
    def test_covariance_monotone_in_rate(self, lam):
        a = torch.tensor([[-1.0]])
        q = torch.tensor([[0.09]])
        p0 = torch.tensor([[0.1]])
        mu = torch.tensor([0.5])
        sigma = torch.tensor([[0.04]])
        base = jump_ou_covariance(a, q, 0.0, mu, sigma, p0, 1.0)
        bumped = jump_ou_covariance(a, q, lam, mu, sigma, p0, 1.0)
        assert float(bumped[0, 0]) >= float(base[0, 0]) - 1e-12


class TestGaussianDensity:
    def test_matches_manual_formula_1d(self):
        mean = torch.tensor([0.3], dtype=torch.float64)
        cov = torch.tensor([[0.25]], dtype=torch.float64)
        x = torch.tensor([[0.0], [0.3], [1.0]], dtype=torch.float64)
        got = gaussian_density_on_grid(mean, cov, x)
        var = 0.25
        for i, xi in enumerate([0.0, 0.3, 1.0]):
            expected = math.exp(-0.5 * (xi - 0.3) ** 2 / var) / math.sqrt(2 * math.pi * var)
            assert abs(float(got[i]) - expected) < 1e-12

    def test_integrates_to_one_2d(self):
        mean = torch.tensor([0.0, 0.0])
        cov = torch.tensor([[0.3, 0.1], [0.1, 0.2]])
        n = 201
        axis = torch.linspace(-5.0, 5.0, n, dtype=torch.float64)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        density = gaussian_density_on_grid(mean, cov, coords).reshape(n, n)
        dx = float(axis[1] - axis[0])
        integral = float(density.sum()) * dx * dx
        assert abs(integral - 1.0) < 1e-3

    def test_bad_coords_shape_rejected(self):
        with pytest.raises(ValueError, match="coords must have shape"):
            gaussian_density_on_grid(torch.tensor([0.0]), torch.tensor([[1.0]]), torch.zeros(3, 2))

    def test_nonsquare_matrix_rejected(self):
        with pytest.raises(ValueError, match="square"):
            ou_covariance(torch.zeros(2, 3), torch.eye(2), torch.eye(2), 1.0)
