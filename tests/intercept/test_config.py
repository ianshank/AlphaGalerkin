"""Tests for intercept module Pydantic configuration schemas."""

from __future__ import annotations

import pytest

from src.intercept.config import (
    AssignmentConfig,
    AtmosphereConfig,
    DynamicsConfig,
    EdgeConfig,
    EngagementConfig,
    GuidanceConfig,
    GuidanceLawType,
    InterceptorConfig,
    InterceptorType,
    MCTSInterceptConfig,
    SensorConfig,
    ThreatConfig,
    ThreatType,
)


class TestThreatConfig:
    def test_defaults(self) -> None:
        config = ThreatConfig(name="test")
        assert config.threat_type == ThreatType.BALLISTIC
        assert config.mass_kg == 200.0
        assert config.max_g == 5.0

    def test_custom_values(self) -> None:
        config = ThreatConfig(
            name="cruise",
            threat_type=ThreatType.CRUISE,
            mass_kg=500.0,
            cd_0=0.05,
            max_g=9.0,
        )
        assert config.threat_type == ThreatType.CRUISE
        assert config.mass_kg == 500.0
        assert config.cd_0 == 0.05

    def test_negative_mass_rejected(self) -> None:
        with pytest.raises(Exception):
            ThreatConfig(name="bad", mass_kg=-1.0)

    def test_zero_area_rejected(self) -> None:
        with pytest.raises(Exception):
            ThreatConfig(name="bad", reference_area_m2=0.0)


class TestInterceptorConfig:
    def test_defaults(self) -> None:
        config = InterceptorConfig(name="test")
        assert config.interceptor_type == InterceptorType.MISSILE
        assert config.max_g == 30.0
        assert config.kill_radius_m == 2.0

    def test_drone_type(self) -> None:
        config = InterceptorConfig(
            name="drone",
            interceptor_type=InterceptorType.ROTOR_DRONE,
            max_speed_ms=100.0,
        )
        assert config.interceptor_type == InterceptorType.ROTOR_DRONE


class TestAtmosphereConfig:
    def test_defaults(self) -> None:
        config = AtmosphereConfig(name="test")
        assert config.wind_speed_ms == 0.0

    def test_negative_wind_rejected(self) -> None:
        with pytest.raises(Exception):
            AtmosphereConfig(name="bad", wind_speed_ms=-5.0)


class TestGuidanceConfig:
    def test_defaults(self) -> None:
        config = GuidanceConfig(name="test")
        assert config.law_type == GuidanceLawType.PN
        assert config.navigation_constant == 3.0

    def test_navigation_constant_bounds(self) -> None:
        with pytest.raises(Exception):
            GuidanceConfig(name="bad", navigation_constant=0.0)
        with pytest.raises(Exception):
            GuidanceConfig(name="bad", navigation_constant=11.0)


class TestMCTSInterceptConfig:
    def test_defaults(self) -> None:
        config = MCTSInterceptConfig(name="test")
        assert config.action_grid_size == 7
        assert config.n_simulations == 100

    def test_action_space_size(self) -> None:
        config = MCTSInterceptConfig(name="test", action_grid_size=3)
        # 3^3 + 1 = 28 actions
        assert config.action_grid_size == 3


class TestEngagementConfig:
    def test_defaults(self) -> None:
        config = EngagementConfig(name="test")
        assert config.max_time_s == 60.0
        assert config.dt_s == 0.02

    def test_dt_greater_than_max_time_rejected(self) -> None:
        with pytest.raises(Exception):
            EngagementConfig(name="bad", dt_s=100.0, max_time_s=10.0)

    def test_nested_configs(self) -> None:
        config = EngagementConfig(
            name="test",
            threat=ThreatConfig(name="t", threat_type=ThreatType.DRONE),
            guidance=GuidanceConfig(name="g", law_type=GuidanceLawType.APN),
        )
        assert config.threat.threat_type == ThreatType.DRONE
        assert config.guidance.law_type == GuidanceLawType.APN

    def test_hash_determinism(self) -> None:
        c1 = EngagementConfig(name="test")
        # Same object should produce same hash
        assert c1.compute_hash() == c1.compute_hash()

    def test_hash_changes_with_values(self) -> None:
        c1 = EngagementConfig(name="test", max_time_s=60.0)
        c2 = EngagementConfig(name="test", max_time_s=120.0)
        assert c1.compute_hash() != c2.compute_hash()


class TestOtherConfigs:
    def test_sensor_config(self) -> None:
        config = SensorConfig(name="test")
        assert config.update_rate_hz == 10.0

    def test_assignment_config(self) -> None:
        config = AssignmentConfig(name="test")
        assert config.max_threats == 50

    def test_edge_config(self) -> None:
        config = EdgeConfig(name="test")
        assert config.max_guidance_latency_ms == 20.0

    def test_dynamics_config(self) -> None:
        config = DynamicsConfig(name="test")
        assert config.dt == 0.01
