"""Shared fixtures for agent-physics integration tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agents.base import AgentState, BaseAgent
from src.agents.config import (
    AgentConfig,
    AgentType,
    CollocationConfig,
    CollocationStrategy,
    CouplingConfig,
    CouplingPairConfig,
    CouplingType,
    DecompositionConfig,
    DecompositionStrategy,
    MessageBusConfig,
    MultiPhysicsConfig,
    OrchestratorConfig,
    SolverAgentConfig,
)
from src.agents.message import MessageBus
from src.agents.registry import AgentRegistry
from src.pde.config import BoundaryCondition, PDEConfig, PDEType

# ------------------------------------------------------------------ #
# Registry cleanup                                                     #
# ------------------------------------------------------------------ #


@pytest.fixture(autouse=True)
def _reset_registries() -> None:  # noqa: PT004
    """Reset singleton registries after each test for isolation."""
    yield  # type: ignore[misc]
    AgentRegistry().clear()
    # Don't clear CollocationRegistry — allocators are registered at import
    # time via decorators and clearing causes factory tests to fail.


# ------------------------------------------------------------------ #
# PDE configs                                                          #
# ------------------------------------------------------------------ #


@pytest.fixture
def poisson_pde_config() -> PDEConfig:
    """Create a Poisson PDE config for testing."""
    return PDEConfig(
        name="poisson_test",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        boundary_condition=BoundaryCondition.DIRICHLET,
        boundary_value=0.0,
    )


@pytest.fixture
def heat_pde_config() -> PDEConfig:
    """Create a Heat PDE config for testing."""
    return PDEConfig(
        name="heat_test",
        pde_type=PDEType.HEAT,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        boundary_condition=BoundaryCondition.DIRICHLET,
        boundary_value=0.0,
        is_time_dependent=True,
        time_start=0.0,
        time_end=1.0,
    )


# ------------------------------------------------------------------ #
# Agent configs                                                        #
# ------------------------------------------------------------------ #


@pytest.fixture
def sample_agent_config() -> AgentConfig:
    """Create a basic agent config."""
    return AgentConfig(
        name="test_agent",
        agent_type=AgentType.SOLVER,
        max_steps=100,
        error_tolerance=0.01,
        computational_budget=1.0,
    )


@pytest.fixture
def sample_solver_config() -> SolverAgentConfig:
    """Create a solver agent config."""
    return SolverAgentConfig(
        name="test_solver",
        game_mode="basis_selection",
        n_simulations=10,
        temperature_start=1.0,
        temperature_end=0.1,
        temperature_decay_steps=50,
        max_steps=100,
        computational_budget=1.0,
    )


@pytest.fixture
def sample_decomposition_config() -> DecompositionConfig:
    """Create a decomposition config."""
    return DecompositionConfig(
        name="test_decomp",
        strategy=DecompositionStrategy.OPERATOR_SPLITTING,
        max_subproblems=4,
        overlap_fraction=0.1,
    )


@pytest.fixture
def sample_coupling_config() -> CouplingConfig:
    """Create a coupling config."""
    return CouplingConfig(
        name="test_coupling",
        coupling_type=CouplingType.DIRICHLET_NEUMANN,
        tolerance=1e-4,
        max_iterations=50,
        relaxation_factor=0.5,
        convergence_window=3,
    )


@pytest.fixture
def sample_collocation_config() -> CollocationConfig:
    """Create a collocation config."""
    return CollocationConfig(
        name="test_collocation",
        strategy=CollocationStrategy.UNIFORM,
        n_points=100,
        adaptation_rate=0.5,
        min_points=10,
        max_points=10000,
    )


@pytest.fixture
def sample_message_bus_config() -> MessageBusConfig:
    """Create a message bus config."""
    return MessageBusConfig(
        name="test_bus",
        buffer_size=100,
        enable_logging=False,
    )


@pytest.fixture
def sample_multi_physics_config(
    poisson_pde_config: PDEConfig,
    heat_pde_config: PDEConfig,
) -> MultiPhysicsConfig:
    """Create a multi-physics config with two coupled physics."""
    return MultiPhysicsConfig(
        name="test_multi_physics",
        physics=[poisson_pde_config, heat_pde_config],
        coupling_pairs=[
            CouplingPairConfig(
                name="poisson_heat_coupling",
                physics_a="poisson_test",
                physics_b="heat_test",
                interface_type=CouplingType.DIRICHLET_NEUMANN,
            ),
        ],
        global_tolerance=0.01,
        max_schwarz_iterations=20,
        budget_allocation={
            "poisson_test": 0.5,
            "heat_test": 0.5,
        },
    )


@pytest.fixture
def sample_orchestrator_config(
    sample_multi_physics_config: MultiPhysicsConfig,
    sample_decomposition_config: DecompositionConfig,
    sample_solver_config: SolverAgentConfig,
    sample_coupling_config: CouplingConfig,
    sample_collocation_config: CollocationConfig,
    sample_message_bus_config: MessageBusConfig,
) -> OrchestratorConfig:
    """Create a full orchestrator config."""
    return OrchestratorConfig(
        name="test_orchestrator",
        multi_physics=sample_multi_physics_config,
        decomposition=sample_decomposition_config,
        solver_defaults=sample_solver_config,
        coupling=sample_coupling_config,
        collocation=sample_collocation_config,
        message_bus=sample_message_bus_config,
        parallel_solvers=False,
    )


# ------------------------------------------------------------------ #
# Message bus                                                          #
# ------------------------------------------------------------------ #


@pytest.fixture
def message_bus(sample_message_bus_config: MessageBusConfig) -> MessageBus:
    """Create a message bus for testing."""
    return MessageBus(sample_message_bus_config)


# ------------------------------------------------------------------ #
# Mock PDE game                                                        #
# ------------------------------------------------------------------ #


@pytest.fixture
def mock_pde_state() -> MagicMock:
    """Create a mock PDEState."""
    state = MagicMock()
    state.coords = np.random.default_rng(42).random((25, 2)).astype(np.float32)
    state.solution = np.zeros(25, dtype=np.float32)
    state.residuals = np.random.default_rng(42).random(25).astype(np.float32)
    state.error_estimate = 1.0
    state.dof = 25
    state.step = 0
    state.budget_remaining = 1.0
    return state


@pytest.fixture
def mock_pde_game(mock_pde_state: MagicMock) -> MagicMock:
    """Create a mock PDEGame."""
    game = MagicMock()
    game.action_space_size = 32
    game.state_channels = 4

    current_state = MagicMock()
    current_state.error_estimate = 1.0
    current_state.dof = 25
    current_state.step = 0

    game.get_initial_state.return_value = mock_pde_state
    game.get_valid_actions.return_value = list(range(10))
    game.is_terminal.return_value = False
    game.to_tensor.return_value = MagicMock(
        detach=MagicMock(
            return_value=MagicMock(
                cpu=MagicMock(
                    return_value=MagicMock(
                        numpy=MagicMock(
                            return_value=MagicMock(
                                astype=MagicMock(
                                    return_value=np.random.default_rng(42)
                                    .random((4, 5, 5))
                                    .astype(np.float32)
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    def apply_action_side_effect(state: object, action: int) -> MagicMock:
        new_state = MagicMock()
        new_state.error_estimate = mock_pde_state.error_estimate * 0.9
        mock_pde_state.error_estimate = new_state.error_estimate
        new_state.dof = 26
        new_state.step = mock_pde_state.step + 1
        mock_pde_state.step = new_state.step
        return new_state

    game.apply_action.side_effect = apply_action_side_effect
    return game


@pytest.fixture
def mock_evaluator() -> MagicMock:
    """Create a mock MCTS evaluator."""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = MagicMock(
        policy=np.ones(32, dtype=np.float32) / 32,
        value=0.0,
    )
    return evaluator


# ------------------------------------------------------------------ #
# Concrete test agent                                                  #
# ------------------------------------------------------------------ #


class ConcreteTestAgent(BaseAgent):
    """Minimal concrete agent for testing BaseAgent."""

    def __init__(
        self,
        config: AgentConfig,
        message_bus: MessageBus | None = None,
        agent_id: str | None = None,
        steps_to_terminal: int = 5,
        error_per_step: float = 0.2,
    ) -> None:
        super().__init__(config, message_bus, agent_id)
        self._steps_to_terminal = steps_to_terminal
        self._error_per_step = error_per_step
        self._initial_error = 1.0

    def setup(self) -> None:
        self._state.error_history.append(self._initial_error)

    def step(self) -> AgentState:
        self._state.step += 1
        current_error = self._state.error_history[-1] * (1.0 - self._error_per_step)
        self._state.error_history.append(current_error)
        self.update_budget(1.0 / self._steps_to_terminal)
        return self._state

    def reset(self) -> None:
        self._state = self._create_initial_state()

    def get_metrics(self) -> dict[str, float]:
        return {
            "step": float(self._state.step),
            "error": self._state.error_history[-1] if self._state.error_history else 1.0,
        }


@pytest.fixture
def concrete_agent(
    sample_agent_config: AgentConfig,
    message_bus: MessageBus,
) -> ConcreteTestAgent:
    """Create a concrete test agent with message bus."""
    return ConcreteTestAgent(
        config=sample_agent_config,
        message_bus=message_bus,
        agent_id="test_concrete",
    )
