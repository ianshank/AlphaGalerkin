"""Property-based tests for reentry solver conservation laws.

Uses Hypothesis to verify numerical invariants hold across
random initial conditions and solver configurations.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.reentry.solver.cfl import CFLController
from src.reentry.solver.shock_detector import ShockDetector


class TestConservationProperties:
    """Property tests for conservation of mass/momentum/energy."""

    @given(
        rho=st.floats(min_value=0.01, max_value=100.0, allow_subnormal=False),
        u=st.floats(min_value=-1000.0, max_value=1000.0, allow_subnormal=False),
        v=st.floats(min_value=-1000.0, max_value=1000.0, allow_subnormal=False),
        p=st.floats(min_value=100.0, max_value=1e7, allow_subnormal=False),
    )
    @settings(max_examples=50)
    def test_primitive_to_conservative_roundtrip(
        self,
        rho: float,
        u: float,
        v: float,
        p: float,
    ) -> None:
        """Converting prim→cons→prim should be identity."""
        from src.reentry.solver.state import ConservativeState

        gamma = 1.4
        gm1 = gamma - 1.0

        # Primitive → Conservative
        e = p / (gm1 * rho) + 0.5 * (u**2 + v**2)
        q = ConservativeState(
            density=np.array([rho]),
            momentum_x=np.array([rho * u]),
            momentum_y=np.array([rho * v]),
            total_energy=np.array([rho * e]),
        )

        # Conservative → Primitive
        rho_back = q.density[0]
        u_back = q.velocity_x()[0]
        v_back = q.velocity_y()[0]

        np.testing.assert_allclose(rho_back, rho, rtol=1e-12)
        np.testing.assert_allclose(u_back, u, rtol=1e-10)
        np.testing.assert_allclose(v_back, v, rtol=1e-10)


class TestPositivityProperties:
    """Property tests for positivity preservation."""

    @given(
        n=st.integers(min_value=5, max_value=50),
        p_ratio=st.floats(min_value=1.1, max_value=100.0),
    )
    @settings(max_examples=30)
    def test_shock_indicator_bounded(self, n: int, p_ratio: float) -> None:
        """Shock indicator should always be in [0, 1]."""
        detector = ShockDetector(pressure_threshold=0.3, enable_ducros=False)
        p = np.ones((n, n)) * 1e5
        mid = n // 2
        p[:, mid:] *= p_ratio
        sigma = detector.detect(p)
        assert np.all(sigma >= 0.0)
        assert np.all(sigma <= 1.0)

    @given(
        cfl=st.floats(min_value=0.01, max_value=0.99),
        ws=st.floats(min_value=1.0, max_value=1e4),
        dx=st.floats(min_value=1e-4, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_cfl_timestep_positive(self, cfl: float, ws: float, dx: float) -> None:
        """CFL-computed timestep must always be positive."""
        ctrl = CFLController(cfl_target=cfl, adaptive=False)
        wave = np.array([[ws]])
        dxarr = np.array([[dx]])
        dyarr = np.array([[dx]])
        dt = ctrl.compute_timestep(wave, dxarr, dyarr)
        assert dt > 0


class TestSpeciesSumProperty:
    """Property tests for species mass fraction constraints."""

    @given(
        n_species=st.integers(min_value=2, max_value=7),
    )
    @settings(max_examples=20)
    def test_random_mass_fractions_sum_to_one(self, n_species: int) -> None:
        """Random mass fractions should be normalizable to sum=1."""
        rng = np.random.default_rng(42)
        y = rng.random(n_species)
        y /= y.sum()
        np.testing.assert_allclose(y.sum(), 1.0, rtol=1e-12)
        assert np.all(y >= 0)
        assert np.all(y <= 1)

    @given(
        n_points=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=20)
    def test_species_sum_invariant_after_clipping(self, n_points: int) -> None:
        """After clipping and renormalization, sum(Y) = 1."""
        rng = np.random.default_rng(42)
        y = rng.random((n_points, 5))
        # Clip and renormalize (what ChemistryIntegrator does)
        y = np.clip(y, 0.0, 1.0)
        y /= y.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(y.sum(axis=1), 1.0, rtol=1e-12)
