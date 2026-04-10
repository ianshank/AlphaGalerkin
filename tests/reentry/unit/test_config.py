"""Tests for reentry configuration schemas."""

from __future__ import annotations

import pytest

from src.reentry.config.chemistry import ChemistryConfig, ChemistryMechanism
from src.reentry.config.freestream import FreestreamConfig
from src.reentry.config.gas import GasConfig
from src.reentry.config.mesh import ReentryMeshConfig
from src.reentry.config.solver import FluxScheme, LimiterType, ReentrySolverConfig
from src.reentry.config.trajectory import TrajectoryPoint
from src.reentry.config.wall import CatalyticModel, WallConfig


class TestGasConfig:
    def test_default_5_species(self) -> None:
        config = GasConfig(name="test")
        assert config.n_species == 5
        assert "N2" in config.species
        assert "O" in config.species

    def test_gas_constant(self) -> None:
        config = GasConfig(name="test")
        r_n2 = config.gas_constant("N2")
        assert 290 < r_n2 < 300  # R_u / M_N2 ≈ 296.8

    def test_missing_species_data_raises(self) -> None:
        with pytest.raises(ValueError, match="missing from molecular_weights"):
            GasConfig(name="bad", species=["N2", "Xe"])

    def test_mixture_molecular_weight(self) -> None:
        config = GasConfig(name="test")
        mw = config.mixture_molecular_weight({"N2": 0.767, "O2": 0.233})
        assert 0.028 < mw < 0.030  # Air ~0.029


class TestFreestreamConfig:
    def test_valid_freestream(self) -> None:
        config = FreestreamConfig(
            name="fire2_1636",
            mach=35.7,
            velocity_m_s=11360.0,
            density_kg_m3=4.855e-4,
            temperature_K=210.0,
        )
        assert config.mach == 35.7

    def test_mass_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            FreestreamConfig(
                name="bad",
                mach=10,
                velocity_m_s=3000,
                density_kg_m3=0.01,
                temperature_K=300,
                mass_fractions={"N2": 0.5, "O2": 0.3},
            )


class TestSolverConfig:
    def test_default_solver(self) -> None:
        config = ReentrySolverConfig(name="test")
        assert config.flux_scheme == FluxScheme.ROE
        assert config.limiter == LimiterType.VAN_LEER
        assert config.cfl == 0.5

    def test_cfl_ramp_validation(self) -> None:
        with pytest.raises(ValueError, match="cfl_ramp_start must be less"):
            ReentrySolverConfig(
                name="bad",
                adaptive_cfl=True,
                cfl=0.5,
                cfl_ramp_start=0.8,
            )


class TestMeshConfig:
    def test_default_mesh(self) -> None:
        config = ReentryMeshConfig(name="test")
        assert config.nx == 100
        assert config.ny == 50

    def test_invalid_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="x_max must be greater"):
            ReentryMeshConfig(name="bad", x_min=1.0, x_max=0.0)

    def test_amr_threshold_validation(self) -> None:
        with pytest.raises(ValueError, match="amr_coarsen_threshold must be less"):
            ReentryMeshConfig(
                name="bad",
                amr_error_threshold=0.05,
                amr_coarsen_threshold=0.1,
            )


class TestChemistryConfig:
    def test_default_park(self) -> None:
        config = ChemistryConfig(name="test")
        assert config.mechanism == ChemistryMechanism.PARK_1993
        assert config.enable_two_temperature is True


class TestWallConfig:
    def test_default_wall(self) -> None:
        config = WallConfig(name="test")
        assert config.catalytic_model == CatalyticModel.NON_CATALYTIC


class TestTrajectoryConfig:
    def test_fire2_trajectory_point(self) -> None:
        point = TrajectoryPoint(
            name="t1636",
            time_s=1636.0,
            altitude_km=53.04,
            velocity_m_s=11360.0,
            density_kg_m3=4.855e-4,
            temperature_K=210.0,
            mach=35.7,
        )
        assert point.mach == 35.7
