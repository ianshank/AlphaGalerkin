"""MetaAgent orchestrating decomposition, solving, and coupling.

The MetaAgent coordinates the full multi-physics pipeline:
1. Decompose the problem into subproblems
2. Create and run SolverAgents for each subproblem
3. Use a CouplingAgent to enforce interface conditions
4. Iterate until global convergence

Example:
    from src.agents.meta import MetaAgent
    from src.agents.config import OrchestratorConfig

    meta = MetaAgent(config=orch_config, message_bus=bus)
    meta.setup()
    result = meta.run()

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from src.agents.base import AgentState, BaseAgent
from src.agents.config import MessageType
from src.agents.coupling import CouplingAgent
from src.agents.decomposition import DecompositionAgent, SubproblemSpec
from src.agents.solver import SolverAgent
from src.templates.base import ExecutionStatus
from src.templates.logging import create_logger_class

if TYPE_CHECKING:
    from src.agents.config import OrchestratorConfig
    from src.agents.message import MessageBus
    from src.mcts.evaluator import Evaluator
    from src.pde.game import PDEGame

MetaLogger = create_logger_class("MetaAgent")


class MetaAgent(BaseAgent):
    """Top-level agent orchestrating multi-physics problem solving.

    Coordinates decomposition, solving, and coupling agents. Supports
    adaptive strategy switching when a solver's error stalls.

    Args:
        config: Orchestrator configuration composing all sub-configs.
        message_bus: Message bus for inter-agent communication.
        agent_id: Optional explicit ID.
        game_factory: Optional factory to create PDEGame from config.
            Signature: ``(SubproblemSpec) -> PDEGame``.
        evaluator_factory: Optional factory to create Evaluator for a game.
            Signature: ``(PDEGame) -> Evaluator``.

    """

    def __init__(
        self,
        config: OrchestratorConfig,
        message_bus: MessageBus | None = None,
        agent_id: str | None = None,
        game_factory: Callable[[SubproblemSpec], PDEGame] | None = None,
        evaluator_factory: Callable[[PDEGame], Evaluator] | None = None,
    ) -> None:
        super().__init__(config.decomposition, message_bus, agent_id)
        self._orch_config = config
        self._game_factory = game_factory
        self._evaluator_factory = evaluator_factory
        self._decomp_agent: DecompositionAgent | None = None
        self._solver_agents: dict[str, SolverAgent] = {}
        self._coupling_agent: CouplingAgent | None = None
        self._subproblems: list[SubproblemSpec] = []
        self._meta_logger = MetaLogger(
            component="meta",
            run_id=self._agent_id,
        )
        self._stall_counters: dict[str, int] = {}
        self._stall_threshold: int = config.decomposition.stall_threshold
        self._stall_tolerance: float = config.decomposition.stall_tolerance

    @property
    def solver_agents(self) -> dict[str, SolverAgent]:
        """Access to individual solver agents."""
        return self._solver_agents

    @property
    def coupling_agent(self) -> CouplingAgent | None:
        """Access to the coupling agent."""
        return self._coupling_agent

    @property
    def subproblems(self) -> list[SubproblemSpec]:
        """Decomposed subproblem specifications."""
        return self._subproblems

    def setup(self) -> None:
        """Initialize all sub-agents.

        1. Create and run DecompositionAgent
        2. Create SolverAgent per subproblem
        3. Create CouplingAgent for interface management
        """
        self._decomp_agent = DecompositionAgent(
            config=self._orch_config.decomposition,
            multi_physics=self._orch_config.multi_physics,
            message_bus=self._message_bus,
        )
        self._decomp_agent.setup()
        self._subproblems = self._decomp_agent.decompose()

        for spec in self._subproblems:
            self._create_solver_for_subproblem(spec)

        if len(self._subproblems) > 1:
            self._coupling_agent = CouplingAgent(
                config=self._orch_config.coupling,
                message_bus=self._message_bus,
            )
            self._coupling_agent.setup()

        self._meta_logger.info(
            "meta_setup_complete",
            n_subproblems=len(self._subproblems),
            n_solvers=len(self._solver_agents),
            has_coupling=self._coupling_agent is not None,
        )

    def _create_solver_for_subproblem(self, spec: SubproblemSpec) -> None:
        """Create a SolverAgent for a subproblem specification.

        Uses the game_factory and evaluator_factory if provided,
        otherwise creates a minimal setup for testing.
        """
        solver_config = spec.solver_config or self._orch_config.solver_defaults
        solver_config = solver_config.with_overrides(
            name=f"solver_{spec.name}",
            computational_budget=self._orch_config.solver_defaults.computational_budget
            * spec.budget_fraction,
        )

        pde_game: PDEGame | None = None
        evaluator: Evaluator | None = None

        if self._game_factory is not None:
            pde_game = self._game_factory(spec)
        if self._evaluator_factory is not None and pde_game is not None:
            evaluator = self._evaluator_factory(pde_game)

        if pde_game is not None and evaluator is not None:
            solver = SolverAgent(
                config=solver_config,
                pde_game=pde_game,
                evaluator=evaluator,
                message_bus=self._message_bus,
                agent_id=f"solver_{spec.name}",
            )
            solver.setup()
            self._solver_agents[spec.name] = solver
            self._stall_counters[spec.name] = 0
        else:
            self._meta_logger.warning(
                "solver_creation_skipped",
                subproblem=spec.name,
                has_game_factory=self._game_factory is not None,
                has_evaluator_factory=self._evaluator_factory is not None,
            )

    def step(self) -> AgentState:
        """Execute one orchestration step.

        1. Each solver runs one step
        2. Coupling agent exchanges boundary data (if multi-physics)
        3. Check global convergence
        4. Detect and handle stalled solvers

        Returns:
            Updated agent state.

        """
        for name, solver in self._solver_agents.items():
            if not solver.is_terminal:
                prev_error = solver.current_error
                solver.step()
                new_error = solver.current_error

                if (
                    prev_error > 0
                    and abs(prev_error - new_error) / prev_error < self._stall_tolerance
                ):
                    self._stall_counters[name] = self._stall_counters.get(name, 0) + 1
                else:
                    self._stall_counters[name] = 0
                    self._meta_logger.debug(
                        "stall_counter_reset",
                        solver=name,
                        error=new_error,
                    )

                if self._stall_counters.get(name, 0) >= self._stall_threshold:
                    self._handle_stalled_solver(name)

        if self._coupling_agent is not None and not self._coupling_agent.is_terminal:
            self._coupling_agent.step()

        self._state.step += 1
        self._update_global_state()
        self._state.metrics = self.get_metrics()

        if self._check_global_convergence():
            self._state.status = ExecutionStatus.COMPLETED

        return self._state

    def _handle_stalled_solver(self, solver_name: str) -> None:
        """Handle a solver whose error has stalled.

        Publishes a STRATEGY_CHANGE message for external consumers.
        """
        self._stall_counters[solver_name] = 0

        if self._message_bus is not None:
            self.send_message(
                receiver="*",
                msg_type=MessageType.STRATEGY_CHANGE,
                payload={
                    "solver": solver_name,
                    "reason": "error_stall",
                },
            )

        self._meta_logger.warning(
            "solver_stalled",
            solver=solver_name,
            threshold=self._stall_threshold,
        )

    def _update_global_state(self) -> None:
        """Update global state from sub-agent states."""
        errors = []
        total_budget_used = 0.0
        for solver in self._solver_agents.values():
            if solver.state.error_history:
                errors.append(solver.state.error_history[-1])
            total_budget_used += solver.state.budget_used

        if errors:
            global_error = max(errors)
            self._state.error_history.append(global_error)

        self._state.budget_used = total_budget_used
        self._state.budget_remaining = max(
            0.0, self.config.computational_budget - total_budget_used
        )

    def _check_global_convergence(self) -> bool:
        """Check if all solvers have converged or terminated.

        Respects ``MultiPhysicsConfig.global_tolerance`` and
        ``max_schwarz_iterations`` in addition to per-solver
        terminal status and coupling convergence.
        """
        if not self._solver_agents:
            return True

        mp_config = self._orch_config.multi_physics

        # Enforce outer-iteration cap
        if self._state.step >= mp_config.max_schwarz_iterations:
            self._meta_logger.info(
                "max_schwarz_iterations_reached",
                step=self._state.step,
                limit=mp_config.max_schwarz_iterations,
            )
            return True

        all_solvers_done = all(solver.is_terminal for solver in self._solver_agents.values())

        coupling_done = True
        if self._coupling_agent is not None:
            coupling_done = self._coupling_agent.is_converged()

        # Check global error against tolerance
        if self._state.error_history:
            global_error = self._state.error_history[-1]
            if global_error < mp_config.global_tolerance:
                return True

        return all_solvers_done and coupling_done

    def reset(self) -> None:
        """Reset all sub-agents."""
        for solver in self._solver_agents.values():
            solver.reset()
        if self._coupling_agent is not None:
            self._coupling_agent.reset()
        self._stall_counters = dict.fromkeys(self._solver_agents, 0)
        self._state = self._create_initial_state()

    def get_metrics(self) -> dict[str, float]:
        """Aggregate metrics from all sub-agents."""
        metrics: dict[str, float] = {
            "global_step": float(self._state.step),
            "n_active_solvers": sum(1.0 for s in self._solver_agents.values() if not s.is_terminal),
            "n_total_solvers": float(len(self._solver_agents)),
            "total_budget_used": self._state.budget_used,
        }

        if self._state.error_history:
            metrics["global_error"] = self._state.error_history[-1]

        for name, solver in self._solver_agents.items():
            solver_metrics = solver.get_metrics()
            for key, value in solver_metrics.items():
                metrics[f"solver_{name}_{key}"] = value

        if self._coupling_agent is not None:
            coupling_metrics = self._coupling_agent.get_metrics()
            for key, value in coupling_metrics.items():
                metrics[f"coupling_{key}"] = value

        return metrics
