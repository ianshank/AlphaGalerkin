"""Tests for threat trajectory prediction and MCTS game."""

from __future__ import annotations

import numpy as np
import pytest

from src.intercept.aero import SimpleAeroModel
from src.intercept.atmosphere import ISAAtmosphere
from src.intercept.dynamics import RigidBody6DOF, create_initial_state
from src.intercept.threat_model import (
    ThreatMCTSGame,
    ThreatPredictor,
)


class TestThreatPredictor:
    def test_ballistic_prediction_length(self) -> None:
        predictor = ThreatPredictor()
        state = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[200.0, 0.0, 10.0],
        )
        traj = predictor.predict_ballistic(state, horizon_s=10.0, dt=0.1)
        assert len(traj.positions) == 101  # 100 steps + initial
        assert len(traj.times) == 101

    def test_ballistic_trajectory_forward(self) -> None:
        predictor = ThreatPredictor()
        state = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[200.0, 0.0, 0.0],
        )
        traj = predictor.predict_ballistic(state, horizon_s=5.0, dt=0.1)
        # Should move North
        assert traj.final_position[0] > 500.0

    def test_ballistic_confidence(self) -> None:
        predictor = ThreatPredictor()
        state = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[200.0, 0.0, 0.0],
        )
        traj = predictor.predict_ballistic(state, horizon_s=5.0)
        assert traj.confidence == 1.0

    def test_predicted_trajectory_duration(self) -> None:
        predictor = ThreatPredictor()
        state = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[100.0, 0.0, 0.0],
        )
        traj = predictor.predict_ballistic(state, horizon_s=10.0, dt=0.5)
        assert traj.duration == pytest.approx(10.0, abs=0.6)


class TestThreatMCTSGame:
    """Test MCTS GameInterface protocol compliance."""

    def _create_game(self) -> ThreatMCTSGame:
        state = create_initial_state(
            position=[0.0, 0.0, -5000.0],
            velocity=[200.0, 0.0, 0.0],
        )
        return ThreatMCTSGame(
            initial_state=state,
            dynamics=RigidBody6DOF(),
            aero_model=SimpleAeroModel(),
            atmosphere=ISAAtmosphere(),
            max_g=5.0,
            horizon_s=2.0,
            dt=0.1,
        )

    def test_get_state_returns_array(self) -> None:
        game = self._create_game()
        state = game.get_state()
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32

    def test_get_legal_actions(self) -> None:
        game = self._create_game()
        actions = game.get_legal_actions()
        assert len(actions) == game.N_ACTIONS
        assert actions == list(range(game.N_ACTIONS))

    def test_apply_action_advances_state(self) -> None:
        game = self._create_game()
        s0 = game.get_state().copy()
        game.apply_action(0)  # first action
        s1 = game.get_state()
        # State should change
        assert not np.allclose(s0, s1)

    def test_coast_action(self) -> None:
        game = self._create_game()
        coast_action = game.N_ACTIONS - 1
        game.apply_action(coast_action)
        assert not game.is_terminal()

    def test_is_terminal_at_horizon(self) -> None:
        game = self._create_game()
        # Run until terminal
        for _ in range(100):
            if game.is_terminal():
                break
            game.apply_action(game.N_ACTIONS - 1)  # coast
        assert game.is_terminal()

    def test_get_winner(self) -> None:
        game = self._create_game()
        # Coast to end
        while not game.is_terminal():
            game.apply_action(game.N_ACTIONS - 1)
        winner = game.get_winner()
        assert winner in (-1, 0, 1)

    def test_clone_independence(self) -> None:
        game = self._create_game()
        clone = game.clone()
        game.apply_action(0)
        # Clone should not be affected
        s_game = game.get_state()
        s_clone = clone.get_state()
        assert not np.allclose(s_game, s_clone)

    def test_decode_coast(self) -> None:
        game = self._create_game()
        accel = game._decode_action(game.N_ACTIONS - 1)
        assert np.allclose(accel, np.zeros(3))

    def test_decode_non_coast(self) -> None:
        game = self._create_game()
        accel = game._decode_action(0)
        # Should be non-zero
        assert np.linalg.norm(accel) > 0

    def test_n_actions_consistent(self) -> None:
        game = self._create_game()
        assert game.N_ACTIONS == 26 * 3 + 1  # 26 directions * 3 magnitudes + coast
