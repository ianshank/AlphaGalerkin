"""Adversarial engagement scenarios.

Tests worst-case conditions to verify system robustness:
- High closure speeds
- Energy exhaustion / breakoff
- Sensor dropout
- Swarm saturation
- Crossing geometry
"""

from __future__ import annotations

import torch

from src.intercept.assignment import HungarianAssigner, TriageLogic, build_cost_matrix
from src.intercept.config import AssignmentConfig, DynamicsConfig, GuidanceConfig
from src.intercept.dynamics import RigidBody6DOF, create_initial_state
from src.intercept.guidance import (
    EnergyTracker,
    ProportionalNavigation,
    ZeroEffortMissGuidance,
)
from src.intercept.intercept_game import run_engagement
from src.intercept.sensors import StalenessTracker
from src.intercept.tracking import create_initial_track


class TestHighClosureSpeed:
    def test_500ms_closure(self) -> None:
        """Engagement with > 500 m/s combined closure should complete."""
        pn = ProportionalNavigation()
        config = GuidanceConfig(name="test", navigation_constant=4.0, max_acceleration_g=40.0)
        dyn = RigidBody6DOF(DynamicsConfig(name="test", dt=0.01, g0=0.0))

        interceptor = create_initial_state(position=[0.0, 0.0, -5000.0], velocity=[300.0, 0.0, 0.0])
        threat = create_initial_state(position=[3000.0, 30.0, -5000.0], velocity=[-250.0, 0.0, 0.0])

        history = run_engagement(
            interceptor_state=interceptor,
            threat_state=threat,
            guidance_law=pn,
            guidance_config=config,
            dynamics=dyn,
            max_time=10.0,
            dt=0.005,
        )
        # Should complete without crash
        assert len(history) > 5
        min_range = min(s.range_m for s in history)
        assert min_range < 20.0  # should get reasonably close at high speed


class TestEnergyExhaustion:
    def test_breakoff_on_fuel_depletion(self) -> None:
        """Energy tracker should signal exhaustion correctly."""
        tracker = EnergyTracker(initial_fuel_mass_kg=0.5, specific_impulse_s=200.0)

        # Burn fuel rapidly
        for _ in range(100):
            tracker.update(accel_magnitude=300.0, dt=0.01, mass=50.0)

        assert tracker.is_exhausted()
        assert tracker.fuel_fraction < 0.01


class TestCrossingGeometry:
    def test_90_degree_crossing(self) -> None:
        """ZEM guidance should handle 90-degree crossing engagement."""
        zem = ZeroEffortMissGuidance()
        config = GuidanceConfig(name="test", navigation_constant=4.0, max_acceleration_g=30.0)
        dyn = RigidBody6DOF(DynamicsConfig(name="test", dt=0.01, g0=0.0))

        interceptor = create_initial_state(position=[0.0, 0.0, -5000.0], velocity=[300.0, 0.0, 0.0])
        # Threat moving perpendicular (East)
        threat = create_initial_state(
            position=[5000.0, -2000.0, -5000.0], velocity=[0.0, 300.0, 0.0]
        )

        history = run_engagement(
            interceptor_state=interceptor,
            threat_state=threat,
            guidance_law=zem,
            guidance_config=config,
            dynamics=dyn,
            max_time=20.0,
            dt=0.005,
        )
        assert len(history) > 5
        min_range = min(s.range_m for s in history)
        # Crossing engagement is hard; verify engagement completes and converges
        # ZEM guidance should at least reduce range significantly from initial
        initial_range = history[0].range_m
        assert min_range < initial_range * 0.8


class TestSensorDropout:
    def test_confidence_decay_during_dropout(self) -> None:
        """Confidence should decay exponentially during sensor dropout."""
        tracker = StalenessTracker(half_life_s=3.0)
        tracker.update("t1", 0.0)

        # Simulate 10s dropout
        conf_0 = tracker.confidence("t1", 0.0)
        conf_3 = tracker.confidence("t1", 3.0)
        conf_6 = tracker.confidence("t1", 6.0)
        conf_10 = tracker.confidence("t1", 10.0)

        assert conf_0 > 0.99
        assert 0.45 < conf_3 < 0.55  # ~0.5 at half-life
        assert conf_6 < conf_3
        assert conf_10 < 0.15


class TestSwarmSaturation:
    def test_50_threats_10_interceptors_triage(self) -> None:
        """50 threats vs 10 interceptors: triage should prioritize correctly."""
        threats = [
            create_initial_track(
                position=[5000.0 + i * 200, float(i * 50), -3000.0],
                velocity=[-200.0, 0.0, 0.0],
                track_id=f"t{i}",
            )
            for i in range(50)
        ]

        engaged, dropped = TriageLogic.triage(threats, n_interceptors=10)
        assert len(engaged) == 10
        assert len(dropped) == 40

        # Engaged should be the closest/fastest threats
        engaged_ranges = [torch.norm(threats[i].position).item() for i in engaged]
        dropped_ranges = [torch.norm(threats[i].position).item() for i in dropped]
        # On average, engaged threats should be closer
        assert sum(engaged_ranges) / len(engaged_ranges) < sum(dropped_ranges) / len(dropped_ranges)

    def test_50x10_assignment_completes(self) -> None:
        """50x10 Hungarian assignment should complete in reasonable time."""
        threats = [
            create_initial_track(
                position=[5000.0 + i * 200, float(i * 50), -3000.0],
                velocity=[-200.0, 0.0, 0.0],
                track_id=f"t{i}",
            )
            for i in range(50)
        ]
        interceptors = [
            create_initial_state(
                position=[0.0, float(i * 100), -3000.0],
                velocity=[300.0, 0.0, 0.0],
            )
            for i in range(10)
        ]

        # Only assign to top 10 threats after triage
        engaged, _ = TriageLogic.triage(threats, n_interceptors=10)
        engaged_threats = [threats[i] for i in engaged]
        engaged_ids = [f"t{i}" for i in engaged]
        int_ids = [f"i{i}" for i in range(10)]

        cost = build_cost_matrix(engaged_threats, interceptors)
        solver = HungarianAssigner()
        config = AssignmentConfig(name="test")
        result = solver.solve(cost, engaged_ids, int_ids, config)

        assert len(result.assignments) == 10
        assert result.computation_time_ms < 200.0


class TestTailChase:
    def test_slower_interceptor_diverges(self) -> None:
        """Interceptor slower than threat should detect divergence."""
        pn = ProportionalNavigation()
        config = GuidanceConfig(
            name="test",
            navigation_constant=3.0,
            breakoff_miss_m=100.0,
            breakoff_tgo_s=5.0,
        )
        dyn = RigidBody6DOF(DynamicsConfig(name="test", dt=0.01, g0=0.0))

        # Interceptor slower than threat in tail chase
        interceptor = create_initial_state(position=[0.0, 0.0, -5000.0], velocity=[100.0, 0.0, 0.0])
        threat = create_initial_state(position=[2000.0, 50.0, -5000.0], velocity=[200.0, 0.0, 0.0])

        history = run_engagement(
            interceptor_state=interceptor,
            threat_state=threat,
            guidance_law=pn,
            guidance_config=config,
            dynamics=dyn,
            max_time=30.0,
            dt=0.01,
        )

        # Interceptor can't catch threat - engagement should terminate
        # (either via breakoff or divergence detection or max time)
        assert len(history) > 5
        # The min range should never reach kill radius (2m)
        min_range = min(s.range_m for s in history)
        assert min_range > 50.0
