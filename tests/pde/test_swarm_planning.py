"""Tests for swarm planning game."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from src.pde.games.swarm_planning import (
    ACTION_NAMES,
    N_ACTIONS_PER_AGENT,
    SwarmPlanningConfig,
    SwarmPlanningGame,
    SwarmState,
)

# --- Fixtures ---


@pytest.fixture
def default_config() -> SwarmPlanningConfig:
    """Create a default SwarmPlanningConfig."""
    return SwarmPlanningConfig(name="test_swarm")


@pytest.fixture
def small_config() -> SwarmPlanningConfig:
    """Create a small config for fast testing."""
    return SwarmPlanningConfig(
        name="small_swarm",
        n_agents=3,
        n_obstacles=2,
        max_steps=5,
        domain_size=(50.0, 50.0, 25.0),
        communication_range=100.0,
        collision_radius=1.0,
        coverage_grid_res=10,
        seed=42,
    )


@pytest.fixture
def game(small_config: SwarmPlanningConfig) -> SwarmPlanningGame:
    """Create a SwarmPlanningGame from small config."""
    return SwarmPlanningGame(small_config)


@pytest.fixture
def initial_state(game: SwarmPlanningGame) -> SwarmState:
    """Create an initial SwarmState."""
    return game.get_initial_state(seed=42)


# --- Config tests ---


class TestSwarmPlanningConfig:
    """Tests for SwarmPlanningConfig."""

    def test_default_config(self) -> None:
        config = SwarmPlanningConfig(name="test")
        assert config.n_agents == 10
        assert config.max_steps == 100
        assert config.collision_radius == 2.0
        assert config.communication_range == 50.0

    def test_custom_config(self) -> None:
        config = SwarmPlanningConfig(
            name="custom",
            n_agents=20,
            domain_size=(200.0, 200.0, 100.0),
            max_velocity=15.0,
        )
        assert config.n_agents == 20
        assert config.domain_size == (200.0, 200.0, 100.0)
        assert config.max_velocity == 15.0

    def test_n_agents_min(self) -> None:
        with pytest.raises(ValidationError):
            SwarmPlanningConfig(name="bad", n_agents=1)

    def test_n_agents_max(self) -> None:
        with pytest.raises(ValidationError):
            SwarmPlanningConfig(name="bad", n_agents=1001)

    def test_collision_less_than_communication(self) -> None:
        with pytest.raises(ValidationError, match="collision_radius"):
            SwarmPlanningConfig(
                name="bad",
                collision_radius=50.0,
                communication_range=10.0,
            )

    def test_zero_obstacles_allowed(self) -> None:
        config = SwarmPlanningConfig(name="no_obs", n_obstacles=0)
        assert config.n_obstacles == 0

    @pytest.mark.parametrize(
        "field,value",
        [
            ("max_velocity", -1.0),
            ("collision_radius", 0.0),
            ("communication_range", -5.0),
            ("max_steps", 0),
        ],
    )
    def test_invalid_field_values(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError):
            SwarmPlanningConfig(name="bad", **{field: value})


# --- Game initialization tests ---


class TestSwarmPlanningGameInit:
    """Tests for game initialization."""

    def test_action_space_size(self, game: SwarmPlanningGame) -> None:
        assert game.action_space_size == 7
        assert game.action_space_size == N_ACTIONS_PER_AGENT

    def test_initial_state_shape(self, game: SwarmPlanningGame, initial_state: SwarmState) -> None:
        assert initial_state.positions.shape == (3, 3)  # n_agents=3, 3D
        assert initial_state.velocities.shape == (3, 3)

    def test_initial_velocities_zero(self, initial_state: SwarmState) -> None:
        np.testing.assert_array_equal(initial_state.velocities, np.zeros((3, 3)))

    def test_initial_step_zero(self, initial_state: SwarmState) -> None:
        assert initial_state.step == 0
        assert initial_state.current_agent == 0
        assert not initial_state.is_terminal

    def test_initial_obstacles(self, game: SwarmPlanningGame, initial_state: SwarmState) -> None:
        assert initial_state.obstacles.shape == (2, 4)  # n_obstacles=2

    def test_initial_coverage_map(self, initial_state: SwarmState) -> None:
        assert initial_state.coverage_map is not None
        assert initial_state.coverage_map.shape == (10, 10)

    def test_positions_within_domain(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        domain = np.array(game.config.domain_size)
        assert np.all(initial_state.positions >= 0.0)
        assert np.all(initial_state.positions <= domain)

    def test_seed_reproducibility(self, small_config: SwarmPlanningConfig) -> None:
        g1 = SwarmPlanningGame(small_config)
        g2 = SwarmPlanningGame(small_config)
        s1 = g1.get_initial_state(seed=123)
        s2 = g2.get_initial_state(seed=123)
        np.testing.assert_array_equal(s1.positions, s2.positions)
        np.testing.assert_array_equal(s1.obstacles, s2.obstacles)

    def test_no_obstacles_config(self) -> None:
        config = SwarmPlanningConfig(
            name="no_obs", n_agents=2, n_obstacles=0, communication_range=50.0
        )
        game = SwarmPlanningGame(config)
        state = game.get_initial_state()
        assert state.obstacles.shape == (0, 4)


# --- Action tests ---


class TestSwarmPlanningActions:
    """Tests for action application."""

    def test_legal_actions(self, game: SwarmPlanningGame, initial_state: SwarmState) -> None:
        actions = game.get_legal_actions(initial_state)
        assert actions == list(range(7))

    def test_no_actions_when_terminal(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        terminal = initial_state.clone()
        terminal.is_terminal = True
        assert game.get_legal_actions(terminal) == []

    def test_apply_forward_action(self, game: SwarmPlanningGame, initial_state: SwarmState) -> None:
        # Action 4 = forward (+x)
        new_state = game.apply_action(initial_state, 4)
        # Agent 0 should have moved in +x direction
        assert new_state.positions[0][0] >= initial_state.positions[0][0]
        # Velocity should be set
        assert new_state.velocities[0][0] == pytest.approx(game.config.max_velocity)

    def test_apply_hover_action(self, game: SwarmPlanningGame, initial_state: SwarmState) -> None:
        # Action 6 = hover
        new_state = game.apply_action(initial_state, 6)
        np.testing.assert_array_almost_equal(new_state.positions[0], initial_state.positions[0])

    def test_round_robin_agent_control(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        state = initial_state
        assert state.current_agent == 0

        state = game.apply_action(state, 0)
        assert state.current_agent == 1

        state = game.apply_action(state, 0)
        assert state.current_agent == 2

        state = game.apply_action(state, 0)
        assert state.current_agent == 0  # Wrapped around

    def test_step_increments_on_full_round(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        state = initial_state
        assert state.step == 0

        # 3 agents => 3 actions = 1 full round
        for _ in range(3):
            state = game.apply_action(state, 6)
        assert state.step == 1

    def test_invalid_action_raises(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        with pytest.raises(ValueError, match="Invalid action"):
            game.apply_action(initial_state, 99)
        with pytest.raises(ValueError, match="Invalid action"):
            game.apply_action(initial_state, -1)

    def test_action_on_terminal_raises(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        terminal = initial_state.clone()
        terminal.is_terminal = True
        with pytest.raises(ValueError, match="terminal"):
            game.apply_action(terminal, 0)

    def test_positions_clamped_to_domain(self) -> None:
        config = SwarmPlanningConfig(
            name="clamp_test",
            n_agents=2,
            n_obstacles=0,
            domain_size=(10.0, 10.0, 10.0),
            max_velocity=100.0,
            communication_range=50.0,
            dt=1.0,
        )
        game = SwarmPlanningGame(config)
        state = game.get_initial_state(seed=0)
        # Apply forward many times to push out of bounds
        for _ in range(10):
            state = game.apply_action(state, 4)  # forward +x
            state = game.apply_action(state, 4)  # agent 1 too
        domain = np.array(config.domain_size)
        assert np.all(state.positions >= 0.0)
        assert np.all(state.positions <= domain)


# --- Terminal and reward tests ---


class TestSwarmPlanningTerminalAndReward:
    """Tests for terminal conditions and reward computation."""

    def test_terminal_at_max_steps(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        state = initial_state
        # 5 max_steps, 3 agents => 15 actions to reach step 5
        for _ in range(15):
            state = game.apply_action(state, 6)
        assert game.is_terminal(state)
        assert state.is_terminal

    def test_reward_is_float(self, game: SwarmPlanningGame, initial_state: SwarmState) -> None:
        reward = game.compute_reward(initial_state)
        assert isinstance(reward, float)

    def test_collision_penalty_increases_with_close_agents(self) -> None:
        config = SwarmPlanningConfig(
            name="collision_test",
            n_agents=2,
            n_obstacles=0,
            collision_radius=5.0,
            communication_range=50.0,
            collision_penalty=100.0,
        )
        game = SwarmPlanningGame(config)

        # State with agents very close together
        state = game.get_initial_state(seed=0)
        state.positions[0] = [10.0, 10.0, 10.0]
        state.positions[1] = [10.1, 10.0, 10.0]  # Within collision radius

        reward_close = game.compute_reward(state)

        # State with agents far apart
        state2 = state.clone()
        state2.positions[1] = [90.0, 90.0, 40.0]

        reward_far = game.compute_reward(state2)

        # Close agents should have worse reward (collision penalty)
        assert reward_close < reward_far


# --- PDE-connected method tests ---


class TestSwarmPlanningPDEMethods:
    """Tests for PDE-connected methods."""

    def test_potential_field_shape(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        potentials = game.compute_potential_field(initial_state)
        assert potentials.shape == (game.config.n_agents,)

    def test_potential_field_positive(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        potentials = game.compute_potential_field(initial_state)
        assert np.all(potentials >= 0.0)

    def test_potential_field_no_obstacles(self) -> None:
        config = SwarmPlanningConfig(
            name="no_obs", n_agents=3, n_obstacles=0, communication_range=50.0
        )
        game = SwarmPlanningGame(config)
        state = game.get_initial_state()
        potentials = game.compute_potential_field(state)
        np.testing.assert_array_equal(potentials, np.zeros(3))

    def test_communication_graph_shape(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        graph = game.compute_communication_graph(initial_state)
        n = game.config.n_agents
        assert graph.shape == (n, n)
        assert graph.dtype == np.bool_

    def test_communication_graph_symmetric(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        graph = game.compute_communication_graph(initial_state)
        np.testing.assert_array_equal(graph, graph.T)

    def test_communication_graph_no_self_loops(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        graph = game.compute_communication_graph(initial_state)
        assert not np.any(np.diag(graph))

    def test_coverage_in_range(self, game: SwarmPlanningGame, initial_state: SwarmState) -> None:
        coverage = game.compute_coverage(initial_state)
        assert 0.0 <= coverage <= 1.0

    def test_coverage_increases_over_time(
        self, game: SwarmPlanningGame, initial_state: SwarmState
    ) -> None:
        cov_initial = game.compute_coverage(initial_state)

        # Move agents around to cover more area
        state = initial_state
        for _ in range(6):  # 2 full rounds
            state = game.apply_action(state, 4)  # forward
        cov_after = game.compute_coverage(state)

        assert cov_after >= cov_initial


# --- Utility tests ---


class TestSwarmPlanningUtilities:
    """Tests for utility methods."""

    def test_action_to_string(self, game: SwarmPlanningGame) -> None:
        assert game.action_to_string(0) == "up"
        assert game.action_to_string(6) == "hover"
        assert "invalid" in game.action_to_string(99)

    @pytest.mark.parametrize("action_idx", range(N_ACTIONS_PER_AGENT))
    def test_all_action_names_valid(self, game: SwarmPlanningGame, action_idx: int) -> None:
        name = game.action_to_string(action_idx)
        assert name in ACTION_NAMES

    def test_clone_game(self, game: SwarmPlanningGame) -> None:
        cloned = game.clone()
        assert cloned.config.n_agents == game.config.n_agents
        assert cloned is not game

    def test_state_clone(self, initial_state: SwarmState) -> None:
        cloned = initial_state.clone()
        assert cloned is not initial_state
        np.testing.assert_array_equal(cloned.positions, initial_state.positions)
        # Mutate clone, original unchanged
        cloned.positions[0] = [999.0, 999.0, 999.0]
        assert initial_state.positions[0][0] != 999.0

    def test_repr(self, game: SwarmPlanningGame) -> None:
        r = repr(game)
        assert "SwarmPlanningGame" in r
        assert "n_agents=3" in r
