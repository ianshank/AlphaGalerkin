"""Tests for gas mixture and transport to boost coverage."""

from __future__ import annotations

import numpy as np

from src.reentry.config.gas import GasConfig
from src.reentry.gas.mixture import GasMixture
from src.reentry.gas.transport import BlottnerTransport


def _gas_config() -> GasConfig:
    return GasConfig(
        name="test",
        species=["N2", "O2"],
        molecular_weights={"N2": 0.0280134, "O2": 0.0319988},
        formation_enthalpies={"N2": 0.0, "O2": 0.0},
        theta_v={"N2": 3395.0, "O2": 2239.0},
    )


class TestGasMixture:
    def test_create(self) -> None:
        config = _gas_config()
        mix = GasMixture(config)
        assert mix.n_species == 2
        assert "N2" in mix.species_names

    def test_mixture_gas_constant(self) -> None:
        config = _gas_config()
        mix = GasMixture(config)
        y = np.array([[0.76, 0.24]])
        r_mix = mix.mixture_gas_constant(y)
        assert 280 < r_mix < 300  # Between N2 and O2 specific R

    def test_mixture_cv_cp(self) -> None:
        config = _gas_config()
        mix = GasMixture(config)
        t = np.array([300.0])
        y = np.array([[0.76, 0.24]])
        cv = mix.mixture_cv(t, y)
        cp = mix.mixture_cp(t, y)
        assert np.all(cp > cv)
        assert np.all(cv > 0)

    def test_mixture_gamma(self) -> None:
        config = _gas_config()
        mix = GasMixture(config)
        t = np.array([300.0])
        y = np.array([[0.76, 0.24]])
        gamma = mix.mixture_gamma(t, y)
        assert np.all(gamma > 1.3)
        assert np.all(gamma < 1.5)

    def test_mass_to_mole(self) -> None:
        config = _gas_config()
        mix = GasMixture(config)
        y = np.array([[0.76, 0.24]])
        x = mix.mass_to_mole_fractions(y)
        np.testing.assert_allclose(x.sum(), 1.0, rtol=1e-10)

    def test_validate_mass_fractions(self) -> None:
        config = _gas_config()
        mix = GasMixture(config)
        y = np.array([[0.76, 0.24]])
        assert mix.validate_mass_fractions(y)

    def test_enthalpy(self) -> None:
        config = _gas_config()
        mix = GasMixture(config)
        t = np.array([1000.0])
        y = np.array([[0.76, 0.24]])
        h = mix.mixture_enthalpy(t, y)
        assert np.all(h > 0)


class TestBlottnerTransport:
    def test_create(self) -> None:
        config = _gas_config()
        transport = BlottnerTransport(config)
        assert transport is not None

    def test_species_viscosity(self) -> None:
        config = _gas_config()
        transport = BlottnerTransport(config)
        mu = transport.species_viscosity("N2", 1000.0)
        assert mu > 0

    def test_mixture_viscosity(self) -> None:
        config = _gas_config()
        transport = BlottnerTransport(config)
        t = np.array([1000.0])
        y = np.array([[0.76, 0.24]])
        mu = transport.mixture_viscosity(t, y)
        assert np.all(mu > 0)

    def test_species_conductivity(self) -> None:
        config = _gas_config()
        transport = BlottnerTransport(config)
        k = transport.species_conductivity("N2", 1000.0)
        assert k > 0

    def test_mixture_conductivity(self) -> None:
        config = _gas_config()
        transport = BlottnerTransport(config)
        t = np.array([1000.0])
        y = np.array([[0.76, 0.24]])
        k = transport.mixture_conductivity(t, y)
        assert np.all(k > 0)
