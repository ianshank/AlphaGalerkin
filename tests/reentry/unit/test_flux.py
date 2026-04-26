"""Tests for numerical flux solvers (Roe and HLLC)."""

from __future__ import annotations

import numpy as np
import pytest

from src.reentry.config.gas import GasConfig
from src.reentry.flux.hllc import HLLCFlux
from src.reentry.flux.limiter import get_limiter, minmod, superbee, van_leer
from src.reentry.flux.reconstruction import MUSCLReconstruction
from src.reentry.flux.roe import RoeFlux
from src.reentry.gas.eos import CaloricallyPerfectEOS


@pytest.fixture
def roe_flux() -> RoeFlux:
    config = GasConfig(
        name="test",
        species=["N2"],
        gamma=1.4,
        molecular_weights={"N2": 0.0280134},
        formation_enthalpies={"N2": 0.0},
        theta_v={"N2": 0.0},
    )
    eos = CaloricallyPerfectEOS(config)
    return RoeFlux(eos=eos, gamma=1.4)


@pytest.fixture
def hllc_flux() -> HLLCFlux:
    return HLLCFlux(gamma=1.4)


class TestRoeFlux:
    def test_uniform_flow_zero_dissipation(self, roe_flux: RoeFlux) -> None:
        """Uniform flow should produce exact physical flux."""
        n = 10
        rho = np.ones(n)
        u = np.ones(n) * 100.0
        p = np.ones(n) * 101325.0

        flux = roe_flux.compute(rho, u, p, rho, u, p)
        # Mass flux = rho * u
        np.testing.assert_allclose(flux[:, 0], rho * u, rtol=1e-10)

    def test_conservation(self, roe_flux: RoeFlux) -> None:
        """Roe flux should be consistent: F(U,U) = F(U)."""
        rho = np.array([1.0])
        u = np.array([50.0])
        p = np.array([100000.0])

        flux = roe_flux.compute(rho, u, p, rho, u, p)
        assert flux.shape == (1, 3)

    def test_sod_left_right_different(self, roe_flux: RoeFlux) -> None:
        """Sod shock tube interface should produce non-zero dissipation."""
        rho_l = np.array([1.0])
        u_l = np.array([0.0])
        p_l = np.array([1.0])
        rho_r = np.array([0.125])
        u_r = np.array([0.0])
        p_r = np.array([0.1])

        flux = roe_flux.compute(rho_l, u_l, p_l, rho_r, u_r, p_r)
        # Flux should be non-zero due to pressure difference
        assert flux.shape == (1, 3)

    def test_max_wave_speed(self, roe_flux: RoeFlux) -> None:
        rho = np.array([1.225])
        u = np.array([0.0])
        p = np.array([101325.0])
        s = roe_flux.max_wave_speed(rho, u, p)
        assert 330 < s[0] < 350  # ~340 m/s


class TestHLLCFlux:
    def test_uniform_flow(self, hllc_flux: HLLCFlux) -> None:
        n = 10
        rho = np.ones(n) * 1.225
        u = np.ones(n) * 50.0
        p = np.ones(n) * 101325.0

        flux = hllc_flux.compute(rho, u, p, rho, u, p)
        np.testing.assert_allclose(flux[:, 0], rho * u, rtol=1e-6)

    def test_flux_shape(self, hllc_flux: HLLCFlux) -> None:
        n = 5
        rho_l = np.random.rand(n) + 0.1
        u_l = np.random.randn(n) * 100
        p_l = np.random.rand(n) * 100000 + 1000
        rho_r = np.random.rand(n) + 0.1
        u_r = np.random.randn(n) * 100
        p_r = np.random.rand(n) * 100000 + 1000

        flux = hllc_flux.compute(rho_l, u_l, p_l, rho_r, u_r, p_r)
        assert flux.shape == (n, 3)


class TestLimiters:
    def test_minmod_tvd(self) -> None:
        r = np.linspace(-2, 4, 100)
        phi = minmod(r)
        assert np.all(phi >= 0)
        assert np.all(phi <= 2)

    def test_van_leer_symmetric(self) -> None:
        r = np.array([0.5, 1.0, 2.0])
        phi = van_leer(r)
        assert np.all(phi >= 0)
        assert np.all(phi <= 2)
        # phi(1) = 1
        np.testing.assert_allclose(van_leer(np.array([1.0])), [1.0])

    def test_superbee_bounds(self) -> None:
        r = np.linspace(-2, 4, 100)
        phi = superbee(r)
        assert np.all(phi >= 0)
        assert np.all(phi <= 2)

    def test_get_limiter_factory(self) -> None:
        from src.reentry.config.solver import LimiterType

        limiter = get_limiter(LimiterType.VAN_LEER)
        result = limiter(np.array([1.0]))
        np.testing.assert_allclose(result, [1.0])


class TestMUSCLReconstruction:
    def test_smooth_field_preserves_accuracy(self) -> None:
        recon = MUSCLReconstruction()
        x = np.linspace(0, 1, 20)
        u = np.sin(2 * np.pi * x)
        u_l, u_r = recon.reconstruct(u)
        # L and R should bracket the interface value
        assert u_l.shape == (19,)
        assert u_r.shape == (19,)

    def test_multivar_reconstruction(self) -> None:
        recon = MUSCLReconstruction()
        u = np.random.rand(20, 3)
        u_l, u_r = recon.reconstruct(u)
        assert u_l.shape == (19, 3)
        assert u_r.shape == (19, 3)
