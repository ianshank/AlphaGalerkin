"""Theoretical-rate validation for the convergence-rate utilities.

Closes Section 4.3 of docs/PLAN_2026-04-27.md.

The existing ``tests/research/test_pde_benchmarks.py::TestAttachConvergenceRates``
covers structural cases (single-result, zero-error, NaN, different
methods). What was missing — and what this module adds — is validation
that the log-log slope produces the correct *theoretical* rate when
fed synthetic data crafted to match known analytic orders (the
spectral / FDM / FEM theoretical rates that SBIR proposals cite).

Two utilities under test:

1. ``PDEBenchmarkRunner._attach_convergence_rates`` — pairwise log-log
   slope between successive refinement levels (uses ``log(N_i+1/N_i)``
   as the DOF ratio, so the rate is "error reduction per unit DOF
   doubling on a log scale").

2. ``BurgersOperator.convergence_rate`` — full polyfit of
   ``log(error) = p * log(h) + C`` over a refinement study (different
   convention: takes ``h`` not ``N_dof``, slope is the order of the
   error in ``h``).

These two compute slightly different things — one is per-DOF, the
other is per-h — so the tests assert the right rate against the right
contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import BurgersOperator
from src.research.pde_benchmarks import PDEBenchmarkResult, PDEBenchmarkRunner

# ---------------------------------------------------------------------------
# PDEBenchmarkRunner._attach_convergence_rates: theoretical-rate validation
# ---------------------------------------------------------------------------


def _result(n_dof: int, l2_error: float, method: str = "fdm") -> PDEBenchmarkResult:
    return PDEBenchmarkResult(
        benchmark_name="theoretical",
        method_name=method,
        n_dof=n_dof,
        l2_error=l2_error,
        wall_time_seconds=0.0,
    )


class TestAttachConvergenceRatesTheoretical:
    """Feed synthetic results crafted to match a known analytical rate.

    The DOF-ratio convention used by ``_attach_convergence_rates`` is
    ``rate = log(e_prev / e_cur) / log(n_cur / n_prev)``. So if the
    error scales like ``e ~ C * (1/n)^p`` (DOF-based), the computed
    rate should equal ``p``.
    """

    def test_first_order_rate(self) -> None:
        # e ~ 1/n  =>  rate = 1.0
        results = [_result(n, 1.0 / n) for n in (16, 64, 256, 1024)]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        rates = [r.convergence_rate for r in updated if r.convergence_rate is not None]
        assert len(rates) == 3  # one rate per pairwise step
        for r in rates:
            assert r == pytest.approx(1.0, rel=1e-9)

    def test_second_order_rate(self) -> None:
        # e ~ 1/n^2  =>  rate = 2.0 (second-order finite difference)
        results = [_result(n, 1.0 / n**2) for n in (16, 64, 256, 1024)]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        rates = [r.convergence_rate for r in updated if r.convergence_rate is not None]
        assert len(rates) == 3
        for r in rates:
            assert r == pytest.approx(2.0, rel=1e-9)

    def test_fourth_order_rate(self) -> None:
        # e ~ 1/n^4  =>  rate = 4.0 (RK4-like spatial discretization)
        results = [_result(n, 1.0 / n**4) for n in (16, 64, 256)]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        rates = [r.convergence_rate for r in updated if r.convergence_rate is not None]
        assert len(rates) == 2
        for r in rates:
            assert r == pytest.approx(4.0, rel=1e-9)

    def test_spectral_rate_is_super_algebraic(self) -> None:
        """Spectral methods produce a super-algebraic rate.

        ``error ~ exp(-c*n)`` — the fitted log-log rate is large and
        increases with n (no fixed asymptotic order).
        """
        results = [_result(n, math.exp(-0.5 * n)) for n in (8, 16, 32)]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        rates = [r.convergence_rate for r in updated if r.convergence_rate is not None]
        # Spectral fits as a "super-algebraic" rate: larger n -> larger fitted rate.
        assert len(rates) == 2
        assert rates[1] > rates[0]
        # First-step rate already > 4 (super-algebraic threshold).
        assert rates[0] > 4.0

    def test_pre_asymptotic_regime_rate_drift(self) -> None:
        """Pre-asymptotic errors produce drifting rates.

        Errors that don't yet follow a clean power law (e.g., constant
        offset + power-law term) yield successive rates that drift
        rather than stay constant — regression guard against hidden
        averaging in the per-pair computation.
        """
        # Mix of "constant offset + 1/n^2" — pre-asymptotic.
        results = [_result(n, 0.01 + 1.0 / n**2) for n in (16, 64, 256, 1024)]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        rates = [r.convergence_rate for r in updated if r.convergence_rate is not None]
        # Successive rates should NOT all be equal (drift toward 0 as 1/n^2 -> 0).
        assert len(rates) == 3
        assert not all(r == pytest.approx(rates[0], rel=1e-2) for r in rates[1:])
        # In the pre-asymptotic regime the constant offset dominates,
        # so the apparent rate decreases toward 0.
        assert rates[-1] < rates[0]

    def test_methods_are_independent(self) -> None:
        """Mixing methods must not cross-contaminate the rate computation."""
        results = [
            _result(16, 1.0 / 16, method="fdm"),
            _result(64, 1.0 / 64, method="fdm"),
            _result(16, 1.0 / 16**2, method="amr"),
            _result(64, 1.0 / 64**2, method="amr"),
        ]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        by_method: dict[str, list[float]] = {}
        for r in updated:
            if r.convergence_rate is not None:
                by_method.setdefault(r.method_name, []).append(r.convergence_rate)
        assert by_method["fdm"][0] == pytest.approx(1.0, rel=1e-9)
        assert by_method["amr"][0] == pytest.approx(2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# BurgersOperator.convergence_rate: polyfit log-log slope on (h, error)
# ---------------------------------------------------------------------------


@pytest.fixture
def burgers_operator() -> BurgersOperator:
    config = PDEConfig(
        name="burgers_conv",
        pde_type=PDEType.BURGERS,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        diffusion_coeff=0.01,
        is_time_dependent=True,
    )
    return BurgersOperator(config)


class TestBurgersConvergenceRate:
    """Direct tests of ``BurgersOperator.convergence_rate``.

    Different convention from ``_attach_convergence_rates``: here we
    pass ``h`` (mesh size, decreasing) and errors, and the method fits
    ``log(error) = p * log(h) + C``. So if ``e ~ h^2`` the fitted rate
    is ``p = 2.0``.
    """

    def test_first_order_h_rate(self, burgers_operator: BurgersOperator) -> None:
        # e = C * h^1
        h_values = [0.1, 0.05, 0.025, 0.0125]
        errors = [0.5 * h for h in h_values]
        rate = burgers_operator.convergence_rate(h_values, errors)
        assert rate == pytest.approx(1.0, rel=1e-9)

    def test_second_order_h_rate(self, burgers_operator: BurgersOperator) -> None:
        # e = C * h^2 (FDM second-order)
        h_values = [0.1, 0.05, 0.025, 0.0125]
        errors = [0.5 * h**2 for h in h_values]
        rate = burgers_operator.convergence_rate(h_values, errors)
        assert rate == pytest.approx(2.0, rel=1e-9)

    def test_fourth_order_h_rate(self, burgers_operator: BurgersOperator) -> None:
        # e = C * h^4 (RK4-class spatial)
        h_values = [0.1, 0.05, 0.025]
        errors = [0.5 * h**4 for h in h_values]
        rate = burgers_operator.convergence_rate(h_values, errors)
        assert rate == pytest.approx(4.0, rel=1e-9)

    def test_polyfit_robust_to_constant_offset_in_log_space(
        self, burgers_operator: BurgersOperator
    ) -> None:
        """Verifies the fit handles the unknown ``C`` correctly.

        ``log(C * h^p) = log C + p * log h`` — the slope only depends
        on ``p``, not the constant. Different ``C`` values must yield
        the same fitted rate.
        """
        h_values = [0.1, 0.05, 0.025, 0.0125]
        errors_a = [1e-2 * h**2 for h in h_values]
        errors_b = [1e-6 * h**2 for h in h_values]
        rate_a = burgers_operator.convergence_rate(h_values, errors_a)
        rate_b = burgers_operator.convergence_rate(h_values, errors_b)
        assert rate_a == pytest.approx(rate_b, rel=1e-9)
        assert rate_a == pytest.approx(2.0, rel=1e-9)

    def test_recovers_rate_from_noisy_synthetic_data(
        self, burgers_operator: BurgersOperator
    ) -> None:
        """Polyfit recovers the rate from noisy synthetic data.

        Adds small (5%) multiplicative noise to a clean power-law
        signal; over 8 refinement levels the recovered rate should
        match the true rate to within ~5%.
        """
        rng = np.random.default_rng(seed=42)
        h_values = [2 ** (-k) for k in range(2, 10)]  # 8 levels, h=0.25 .. ~2e-3
        # 5% multiplicative noise on top of e = h^2.
        errors = [(1.0 + 0.05 * rng.standard_normal()) * h**2 for h in h_values]
        rate = burgers_operator.convergence_rate(h_values, errors)
        # 8 levels with 5% noise should recover rate within ~5% of true.
        assert rate == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Cross-utility consistency: per-DOF and per-h conventions agree on
# the rate for a uniform 2D grid where N_dof = (1/h)^2.
# ---------------------------------------------------------------------------


class TestConvergenceConventionConsistency:
    """The per-DOF and per-h conventions agree under explicit conversion.

    For a uniform 2D grid, ``N_dof = (1/h)^2``, so a per-h rate of
    ``p`` corresponds to a per-DOF rate of ``p/2``. This class
    verifies both utilities agree under that conversion.
    """

    def test_2d_grid_per_h_p_equals_per_dof_p_over_2(
        self, burgers_operator: BurgersOperator
    ) -> None:
        # Build a 4-level refinement: h_k = 0.1 * 2^-k, N_k = (1/h_k)^2.
        h_values = [0.1 * (2 ** (-k)) for k in range(4)]
        # e = C * h^2 (second-order FDM in 2D).
        errors = [0.5 * h**2 for h in h_values]
        n_dof = [int(round(1.0 / h)) ** 2 for h in h_values]

        # Per-h slope = 2.0 (Burgers operator convention).
        rate_per_h = burgers_operator.convergence_rate(h_values, errors)
        assert rate_per_h == pytest.approx(2.0, rel=1e-9)

        # Per-DOF slope using the runner's convention: error vs DOF.
        # With N = (1/h)^2 and e = h^2, e = (1/N) so per-DOF rate = 1.0.
        results = [_result(n, e, method="fdm") for n, e in zip(n_dof, errors, strict=True)]
        updated = PDEBenchmarkRunner._attach_convergence_rates(results)
        rates_per_dof = [r.convergence_rate for r in updated if r.convergence_rate is not None]
        for r in rates_per_dof:
            assert r == pytest.approx(1.0, rel=1e-9)

        # Conversion: rate_per_h / dim = rate_per_dof for uniform grids.
        # 2D: rate_per_dof = rate_per_h / 2.
        assert rate_per_h / 2.0 == pytest.approx(rates_per_dof[0], rel=1e-9)
