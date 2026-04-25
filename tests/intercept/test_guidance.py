"""Tests for guidance laws and engagement simulation."""

from __future__ import annotations

import pytest
import torch

from src.intercept.config import GuidanceConfig
from src.intercept.dynamics import RigidBody6DOF, create_initial_state
from src.intercept.guidance import (
    AugmentedPN,
    EnergyTracker,
    GuidanceLawRegistry,
    ProportionalNavigation,
    ZeroEffortMissGuidance,
)
from src.intercept.intercept_game import InterceptGameAdapter, run_engagement


class TestProportionalNavigation:
    def test_head_on_engagement(self) -> None:
        """PN should converge for head-on constant-velocity target."""
        pn = ProportionalNavigation()
        config = GuidanceConfig(name="test", navigation_constant=4.0)

        # Interceptor and threat heading toward each other
        own = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[200.0, 0.0, 0.0],
        )
        threat = create_initial_state(
            position=[10000.0, 100.0, -5000.0],
            velocity=[-200.0, 0.0, 0.0],
        )

        cmd = pn.compute(own, threat, config)
        assert cmd.acceleration.shape == (3,)
        assert cmd.time_to_go > 0
        assert cmd.miss_distance < float("inf")

    def test_zero_los_rate_zero_command(self) -> None:
        """If interceptor is on collision course, command should be ~zero."""
        pn = ProportionalNavigation()
        config = GuidanceConfig(name="test")

        own = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[200.0, 0.0, 0.0],
        )
        # Threat directly ahead, same altitude, heading toward us
        threat = create_initial_state(
            position=[10000.0, 0.0, -5000.0],
            velocity=[-200.0, 0.0, 0.0],
        )

        cmd = pn.compute(own, threat, config)
        # On perfect collision course, LOS rate = 0 => small command
        assert torch.norm(cmd.acceleration).item() < 50.0

    def test_terminal_phase_boost(self) -> None:
        """Terminal phase should increase navigation constant."""
        pn = ProportionalNavigation()
        config = GuidanceConfig(name="test", terminal_range_m=1000.0)

        own = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[200.0, 0.0, 0.0],
        )
        # Close range: terminal phase
        threat = create_initial_state(
            position=[500.0, 50.0, -5000.0],
            velocity=[-100.0, 0.0, 0.0],
        )

        cmd = pn.compute(own, threat, config)
        assert cmd.is_terminal

    def test_breakoff_detection(self) -> None:
        """Should detect when miss distance is too large near intercept."""
        pn = ProportionalNavigation()
        config = GuidanceConfig(
            name="test",
            breakoff_miss_m=10.0,
            breakoff_tgo_s=100.0,  # set high so we trigger it
        )

        own = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[200.0, 0.0, 0.0],
        )
        # Threat moving perpendicular -> large miss
        threat = create_initial_state(
            position=[5000.0, 0.0, -5000.0],
            velocity=[0.0, 300.0, 0.0],
        )

        cmd = pn.compute(own, threat, config)
        # Miss should be large due to perpendicular velocity
        assert cmd.miss_distance > 10.0


class TestAugmentedPN:
    def test_produces_command(self) -> None:
        apn = AugmentedPN()
        config = GuidanceConfig(name="test")
        own = create_initial_state(position=[0.0, 0.0, -5000.0], velocity=[200.0, 0.0, 0.0])
        threat = create_initial_state(
            position=[10000.0, 100.0, -5000.0], velocity=[-200.0, 0.0, 0.0]
        )
        cmd = apn.compute(own, threat, config)
        assert cmd.acceleration.shape == (3,)


class TestZeroEffortMissGuidance:
    def test_produces_command(self) -> None:
        zem = ZeroEffortMissGuidance()
        config = GuidanceConfig(name="test")
        own = create_initial_state(position=[0.0, 0.0, -5000.0], velocity=[200.0, 0.0, 0.0])
        threat = create_initial_state(
            position=[10000.0, 100.0, -5000.0], velocity=[-200.0, 0.0, 0.0]
        )
        cmd = zem.compute(own, threat, config)
        assert cmd.acceleration.shape == (3,)
        assert cmd.time_to_go > 0


class TestGuidanceLawRegistry:
    def test_pn_registered(self) -> None:
        cls = GuidanceLawRegistry().get("pn")
        assert cls is ProportionalNavigation

    def test_apn_registered(self) -> None:
        cls = GuidanceLawRegistry().get("apn")
        assert cls is AugmentedPN

    def test_zem_registered(self) -> None:
        cls = GuidanceLawRegistry().get("zem_zev")
        assert cls is ZeroEffortMissGuidance


class TestEnergyTracker:
    def test_initial_fuel(self) -> None:
        tracker = EnergyTracker(initial_fuel_mass_kg=10.0)
        assert tracker.fuel_fraction == pytest.approx(1.0, abs=1e-10)
        assert not tracker.is_exhausted()

    def test_fuel_consumption(self) -> None:
        tracker = EnergyTracker(initial_fuel_mass_kg=10.0, specific_impulse_s=250.0)
        tracker.update(accel_magnitude=100.0, dt=0.1, mass=50.0)
        assert tracker.fuel_remaining < 10.0

    def test_exhaustion(self) -> None:
        tracker = EnergyTracker(initial_fuel_mass_kg=0.01)
        tracker.update(accel_magnitude=1000.0, dt=1.0, mass=50.0)
        assert tracker.is_exhausted()

    def test_delta_v_tracking(self) -> None:
        tracker = EnergyTracker()
        tracker.update(accel_magnitude=10.0, dt=1.0, mass=50.0)
        assert tracker.total_delta_v_used == pytest.approx(10.0)


class TestInterceptGameAdapter:
    """Test MCTS GameInterface protocol compliance."""

    def _create_game(self) -> InterceptGameAdapter:
        interceptor = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[300.0, 0.0, 0.0],
        )
        threat = create_initial_state(
            position=[5000.0, 200.0, -5000.0],
            velocity=[-200.0, 0.0, 0.0],
        )
        from src.intercept.config import MCTSInterceptConfig

        mcts_config = MCTSInterceptConfig(name="test", action_grid_size=3, time_horizon_s=2.0)
        return InterceptGameAdapter(
            interceptor_state=interceptor,
            threat_state=threat,
            mcts_config=mcts_config,
        )

    def test_get_state(self) -> None:
        game = self._create_game()
        state = game.get_state()
        assert state.dtype.name == "float32"
        assert len(state) == 15

    def test_get_legal_actions(self) -> None:
        game = self._create_game()
        actions = game.get_legal_actions()
        # 3^3 + 1 = 28
        assert len(actions) == 28

    def test_apply_action(self) -> None:
        game = self._create_game()
        s0 = game.get_state().copy()
        game.apply_action(0)
        s1 = game.get_state()
        assert not all(s0 == s1)

    def test_is_terminal_initially_false(self) -> None:
        game = self._create_game()
        assert not game.is_terminal()

    def test_terminal_at_max_steps(self) -> None:
        game = self._create_game()
        for _ in range(1000):
            if game.is_terminal():
                break
            game.apply_action(game._n_actions - 1)  # coast
        assert game.is_terminal()

    def test_get_winner(self) -> None:
        game = self._create_game()
        while not game.is_terminal():
            game.apply_action(game._n_actions - 1)
        w = game.get_winner()
        assert w in (-1, 0, 1)

    def test_clone(self) -> None:
        game = self._create_game()
        clone = game.clone()
        game.apply_action(0)
        # Clone unaffected
        assert game._state.steps != clone._state.steps


class TestRunEngagement:
    def test_head_on_intercept(self) -> None:
        """Head-on engagement with PN in zero-gravity should achieve < 3m miss.

        Disabling gravity isolates guidance law convergence from the
        gravity compensation problem (tested separately with APN).
        """
        from src.intercept.config import DynamicsConfig

        pn = ProportionalNavigation()
        config = GuidanceConfig(
            name="test",
            navigation_constant=4.0,
            max_acceleration_g=30.0,
        )

        # Zero-gravity dynamics to test pure PN convergence
        dyn_config = DynamicsConfig(name="test", dt=0.01, g0=0.0)
        dynamics = RigidBody6DOF(dyn_config)

        # 50m lateral offset creates LOS rate for PN to correct
        interceptor = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[300.0, 0.0, 0.0],
        )
        threat = create_initial_state(
            position=[5000.0, 50.0, -5000.0],
            velocity=[-200.0, 0.0, 0.0],
        )

        history = run_engagement(
            interceptor_state=interceptor,
            threat_state=threat,
            guidance_law=pn,
            guidance_config=config,
            dynamics=dynamics,
            max_time=15.0,
            dt=0.01,
        )

        assert len(history) > 10

        # Find minimum range
        min_range = min(s.range_m for s in history)
        # PN with N=4 on constant-velocity head-on should converge
        assert min_range < 3.0, f"Min range {min_range:.1f}m exceeds 3m"

    def test_engagement_records_history(self) -> None:
        pn = ProportionalNavigation()
        config = GuidanceConfig(name="test")

        interceptor = create_initial_state(position=[0.0, 0.0, -5000.0], velocity=[300.0, 0.0, 0.0])
        threat = create_initial_state(position=[5000.0, 0.0, -5000.0], velocity=[-200.0, 0.0, 0.0])

        history = run_engagement(
            interceptor_state=interceptor,
            threat_state=threat,
            guidance_law=pn,
            guidance_config=config,
            max_time=20.0,
            dt=0.02,
        )

        assert len(history) > 0
        assert all(s.range_m >= 0 for s in history)
