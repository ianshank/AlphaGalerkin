"""SolverAgent wrapping PDEGame + MCTS for single-PDE solving.

Each SolverAgent manages one ``PDEGameAdapter`` and ``MCTS`` instance,
running a search loop with a configurable temperature schedule.

Example:
    from src.agents.solver import SolverAgent
    from src.agents.config import SolverAgentConfig
    from src.pde.games.basis_selection import BasisSelectionGame
    from src.mcts.evaluator import RandomEvaluator

    config = SolverAgentConfig(name="solver", n_simulations=100)
    solver = SolverAgent(config, pde_game=game, evaluator=evaluator)
    result = solver.run()

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.base import AgentState, BaseAgent
from src.agents.config import MessageType
from src.templates.logging import create_logger_class

if TYPE_CHECKING:
    from src.agents.config import SolverAgentConfig
    from src.agents.message import MessageBus
    from src.mcts.evaluator import Evaluator
    from src.mcts.search import MCTS
    from src.pde.game import PDEGame
    from src.pde.mcts_adapter import PDEGameAdapter
    from src.templates.base import ExecutionStatus

SolverLogger = create_logger_class("SolverAgent")


class SolverAgent(BaseAgent):
    """Agent that solves a single PDE via MCTS-guided Galerkin approximation.

    Wraps an existing ``PDEGame`` and ``Evaluator`` without modifying
    their interfaces. The MCTS search budget and temperature schedule
    are controlled by ``SolverAgentConfig``.

    Args:
        config: Solver agent configuration.
        pde_game: The PDE game instance to solve.
        evaluator: Neural network or random evaluator for MCTS leaf nodes.
        message_bus: Optional bus for inter-agent communication.
        agent_id: Optional explicit ID.

    """

    def __init__(
        self,
        config: SolverAgentConfig,
        pde_game: PDEGame,
        evaluator: Evaluator,
        message_bus: MessageBus | None = None,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(config, message_bus, agent_id)
        self._pde_game = pde_game
        self._evaluator = evaluator
        self._adapter: PDEGameAdapter | None = None
        self._mcts: MCTS | None = None
        self._solver_logger = SolverLogger(
            component="solver",
            run_id=self._agent_id,
        )

    @property
    def solver_config(self) -> SolverAgentConfig:
        """Typed access to solver-specific config."""
        from src.agents.config import SolverAgentConfig

        assert isinstance(self.config, SolverAgentConfig)
        return self.config

    @property
    def adapter(self) -> PDEGameAdapter | None:
        """Access to the underlying PDEGameAdapter."""
        return self._adapter

    @property
    def current_error(self) -> float:
        """Current error estimate from the PDE game."""
        if self._adapter is not None:
            return self._adapter.current_error
        return float("inf")

    @property
    def error_reduction(self) -> float:
        """Total error reduction from initial state."""
        if self._adapter is not None:
            return self._adapter.error_reduction
        return 0.0

    def setup(self) -> None:
        """Create PDEGameAdapter and MCTS instance."""
        from src.mcts.search import MCTS
        from src.pde.mcts_adapter import PDEGameAdapter

        self._adapter = PDEGameAdapter(self._pde_game)
        self._mcts = MCTS(
            evaluator=self._evaluator,
            c_puct=self.solver_config.c_puct,
            n_simulations=self.solver_config.n_simulations,
            dirichlet_alpha=self.solver_config.dirichlet_alpha,
            dirichlet_epsilon=self.solver_config.dirichlet_epsilon,
        )

        initial_error = self._adapter.current_error
        self._state.error_history.append(initial_error)

        self._solver_logger.info(
            "solver_setup_complete",
            game_mode=self.solver_config.game_mode,
            n_simulations=self.solver_config.n_simulations,
            initial_error=initial_error,
        )

    def _compute_temperature(self) -> float:
        """Compute current temperature from linear decay schedule."""
        cfg = self.solver_config
        if cfg.temperature_decay_steps <= 0:
            return cfg.temperature_end
        progress = min(1.0, self._state.step / cfg.temperature_decay_steps)
        return cfg.temperature_start + progress * (cfg.temperature_end - cfg.temperature_start)

    def step(self) -> AgentState:
        """Execute one MCTS search + action step.

        Returns:
            Updated agent state.

        """
        assert self._adapter is not None, "Must call setup() before step()"
        assert self._mcts is not None, "Must call setup() before step()"

        if self._adapter.is_terminal():
            self._state.status = self._terminal_status()
            return self._state

        temperature = self._compute_temperature()

        self._mcts.search(self._adapter)
        action = self._mcts.get_action(self._adapter, temperature=temperature)
        self._adapter.apply_action(action)
        self._mcts.advance(action)

        self._state.step += 1
        self._state.error_history.append(self._adapter.current_error)
        self.update_budget(self.solver_config.budget_per_step)
        self._state.metrics = self.get_metrics()

        if self._message_bus is not None:
            self.send_message(
                receiver="*",
                msg_type=MessageType.STATE_UPDATE,
                payload={
                    "agent_id": self._agent_id,
                    "step": self._state.step,
                    "error": self._adapter.current_error,
                    "action": action,
                },
            )

        self._solver_logger.debug(
            "solver_step",
            step=self._state.step,
            action=action,
            error=self._adapter.current_error,
            temperature=temperature,
        )

        return self._state

    def _terminal_status(self) -> ExecutionStatus:
        """Determine terminal status."""
        from src.templates.base import ExecutionStatus

        return ExecutionStatus.COMPLETED

    def reset(self) -> None:
        """Reset adapter, MCTS, and state to initial conditions."""
        if self._adapter is not None:
            self._adapter.reset()
        if self._mcts is not None and hasattr(self._mcts, "reset"):
            self._mcts.reset()
        self._state = self._create_initial_state()
        if self._adapter is not None:
            self._state.error_history.append(self._adapter.current_error)

    def get_metrics(self) -> dict[str, float]:
        """Return current solver metrics."""
        metrics: dict[str, float] = {
            "step": float(self._state.step),
            "budget_used": self._state.budget_used,
            "budget_remaining": self._state.budget_remaining,
        }
        if self._adapter is not None:
            metrics["error"] = self._adapter.current_error
            metrics["error_reduction"] = self._adapter.error_reduction
            metrics["dof"] = float(self._adapter.state.dof)
        return metrics
