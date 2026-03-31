"""Tests for PettingZoo adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.games.pettingzoo_adapter import HAS_PETTINGZOO, PettingZooAdapter


@pytest.fixture
def mock_game() -> MagicMock:
    """Create a mock GameInterface."""
    game = MagicMock()
    game.name = "mock_game"
    game.default_board_size = 9
    game.action_space_size = 82  # 9*9 + 1 (pass)
    game.state_channels = 17
    game.get_observation_shape.return_value = (17, 9, 9)

    # initial_state returns a mock GameState
    state = MagicMock()
    state.board_size = 9
    game.initial_state.return_value = state

    # to_tensor returns a suitable tensor
    import torch

    game.to_tensor.return_value = torch.zeros(17, 9, 9)

    # validate_action returns True by default
    game.validate_action.return_value = True

    # is_terminal
    game.is_terminal.return_value = False

    return game


@pytest.mark.skipif(not HAS_PETTINGZOO, reason="pettingzoo not installed")
class TestPettingZooAdapter:
    """Tests for PettingZooAdapter (requires pettingzoo)."""

    def test_init(self, mock_game: MagicMock) -> None:
        adapter = PettingZooAdapter(mock_game, n_agents=2)
        assert adapter.n_agents == 2
        assert len(adapter.possible_agents) == 2
        assert adapter.possible_agents == ["agent_0", "agent_1"]

    def test_reset(self, mock_game: MagicMock) -> None:
        adapter = PettingZooAdapter(mock_game, n_agents=2)
        obs, infos = adapter.reset()

        assert len(obs) == 2
        assert "agent_0" in obs
        assert "agent_1" in obs
        assert obs["agent_0"].shape == (17, 9, 9)
        mock_game.initial_state.assert_called_once()

    def test_step(self, mock_game: MagicMock) -> None:
        adapter = PettingZooAdapter(mock_game, n_agents=2)
        adapter.reset()

        # Mock apply_action to return a new state
        new_state = MagicMock()
        mock_game.apply_action.return_value = new_state

        actions = {"agent_0": 0, "agent_1": 1}
        obs, rewards, terminations, truncations, infos = adapter.step(actions)

        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)

    def test_step_terminal(self, mock_game: MagicMock) -> None:
        adapter = PettingZooAdapter(mock_game, n_agents=2)
        adapter.reset()

        new_state = MagicMock()
        mock_game.apply_action.return_value = new_state
        mock_game.is_terminal.return_value = True
        result = MagicMock()
        result.winner = 1
        mock_game.get_result.return_value = result

        actions = {"agent_0": 0, "agent_1": 1}
        obs, rewards, terminations, truncations, infos = adapter.step(actions)

        assert all(terminations.values())

    def test_observation_space(self, mock_game: MagicMock) -> None:
        adapter = PettingZooAdapter(mock_game, n_agents=2)
        space = adapter.observation_space("agent_0")
        assert space.shape == (17, 9, 9)

    def test_action_space(self, mock_game: MagicMock) -> None:
        adapter = PettingZooAdapter(mock_game, n_agents=2)
        space = adapter.action_space("agent_0")
        assert space.n == 82

    def test_repr(self, mock_game: MagicMock) -> None:
        adapter = PettingZooAdapter(mock_game, n_agents=3)
        r = repr(adapter)
        assert "PettingZooAdapter" in r
        assert "mock_game" in r

    def test_step_without_reset_raises(self, mock_game: MagicMock) -> None:
        adapter = PettingZooAdapter(mock_game, n_agents=2)
        with pytest.raises(RuntimeError, match="reset"):
            adapter.step({"agent_0": 0})


class TestPettingZooAdapterNoDeps:
    """Tests that work regardless of pettingzoo availability."""

    def test_has_pettingzoo_flag(self) -> None:
        assert isinstance(HAS_PETTINGZOO, bool)

    def test_import_without_pettingzoo(self) -> None:
        """Verify module can be imported even without pettingzoo."""
        # This test passing means the import succeeded
        from src.games.pettingzoo_adapter import PettingZooAdapter  # noqa: F811

        assert PettingZooAdapter is not None

    @pytest.mark.skipif(HAS_PETTINGZOO, reason="Test for missing deps")
    def test_raises_without_pettingzoo(self, mock_game: MagicMock) -> None:
        """When pettingzoo is not installed, init should raise."""
        with pytest.raises(RuntimeError, match="pettingzoo"):
            PettingZooAdapter(mock_game)
