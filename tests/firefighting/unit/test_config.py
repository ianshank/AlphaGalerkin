"""Tests for firefighting configuration schemas."""

from __future__ import annotations

import pytest

from src.firefighting.config.edge import EdgeConfig, EdgeDevice
from src.firefighting.config.fire import FireConfig, FuelCategory
from src.firefighting.config.sensor import SensorConfig
from src.firefighting.config.solver import FireSolverConfig
from src.firefighting.config.terrain import TerrainConfig
from src.firefighting.config.wind import WindConfig


class TestFireConfig:
    def test_default_grass_fire(self) -> None:
        config = FireConfig(name="test")
        assert config.fuel_category == FuelCategory.SHORT_GRASS
        assert config.ignition_temperature_K > config.ambient_temperature_K

    def test_invalid_temperatures_raises(self) -> None:
        with pytest.raises(ValueError, match="ignition_temperature must be greater"):
            FireConfig(
                name="bad",
                ignition_temperature_K=200.0,
                ambient_temperature_K=300.0,
            )


class TestFireSolverConfig:
    def test_default_solver(self) -> None:
        config = FireSolverConfig(name="test")
        assert config.nx == 100
        assert config.ny == 100

    def test_explicit_stability_check(self) -> None:
        """Forward Euler with too-large timestep should raise."""
        # dx = 1m, alpha = 1.0 => max_dt = 0.25 * 1^2 / 1.0 = 0.25s
        with pytest.raises(ValueError, match="exceeds diffusion stability"):
            FireSolverConfig(
                name="bad",
                nx=100,
                ny=100,
                domain_size_x_m=100.0,
                domain_size_y_m=100.0,
                dt_s=1.0,  # > 0.25s stability limit
                thermal_diffusivity_m2_s=1.0,
                time_integration="forward_euler",
            )


class TestEdgeConfig:
    def test_jetson_defaults(self) -> None:
        config = EdgeConfig(name="test")
        assert config.device == EdgeDevice.JETSON_ORIN_NANO
        assert config.max_memory_mb == 4096
        assert config.max_latency_ms == 500.0

    def test_latency_budgets_sum(self) -> None:
        config = EdgeConfig(name="test")
        total = (
            config.sensor_ingest_budget_ms
            + config.mcts_budget_ms
            + config.pde_solve_budget_ms
            + config.output_budget_ms
        )
        assert total == config.max_latency_ms


class TestSensorConfig:
    def test_defaults(self) -> None:
        config = SensorConfig(name="test")
        assert config.stale_threshold_s == 5.0
        assert config.min_confidence > 0


class TestTerrainConfig:
    def test_flat_terrain(self) -> None:
        config = TerrainConfig(name="test", flat_terrain=True)
        assert config.flat_terrain is True


class TestWindConfig:
    def test_default_wind(self) -> None:
        config = WindConfig(name="test")
        assert config.default_speed_m_s == 5.0
