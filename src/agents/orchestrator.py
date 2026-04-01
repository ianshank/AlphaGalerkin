"""AgentOrchestrator — BaseExecutable entry point for agent-physics integration.

Wraps the MetaAgent in the standard ``BaseExecutable`` pattern,
producing an ``ExecutionResult`` with timing, metrics, and error handling.

Example:
    from src.agents.orchestrator import AgentOrchestrator
    from src.agents.config import OrchestratorConfig

    config = OrchestratorConfig(name="run", multi_physics=mp_config)
    orchestrator = AgentOrchestrator(config)
    result = orchestrator.run()
    print(result.metrics)

"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.agents.message import MessageBus
from src.agents.meta import MetaAgent
from src.templates.base import BaseExecutable, ExecutionResult, ExecutionStatus
from src.templates.logging import create_logger_class

if TYPE_CHECKING:
    from src.agents.config import OrchestratorConfig
    from src.agents.decomposition import SubproblemSpec
    from src.mcts.evaluator import Evaluator
    from src.pde.game import PDEGame

OrchestratorLogger = create_logger_class("AgentOrchestrator")


class AgentOrchestrator(BaseExecutable["OrchestratorConfig"]):
    """Main entry point for running the agent-physics pipeline.

    Creates a ``MessageBus``, ``MetaAgent``, and runs the full
    decomposition → solving → coupling loop. Returns results in
    the standard ``ExecutionResult`` format.

    Args:
        config: Orchestrator configuration.
        run_id: Optional run identifier.
        game_factory: Optional factory ``(SubproblemSpec) -> PDEGame``.
        evaluator_factory: Optional factory ``(PDEGame) -> Evaluator``.

    """

    _executable_name = "agent_orchestrator"
    _logger_class = OrchestratorLogger

    def __init__(
        self,
        config: OrchestratorConfig,
        run_id: str | None = None,
        game_factory: Callable[[SubproblemSpec], PDEGame] | None = None,
        evaluator_factory: Callable[[PDEGame], Evaluator] | None = None,
    ) -> None:
        super().__init__(config, run_id)
        self._game_factory = game_factory
        self._evaluator_factory = evaluator_factory

    def execute(self) -> ExecutionResult:
        """Run the full agent-physics orchestration pipeline.

        Returns:
            ExecutionResult with status, metrics, and artifacts.

        """
        try:
            bus = MessageBus(self.config.message_bus)

            meta = MetaAgent(
                config=self.config,
                message_bus=bus,
                game_factory=self._game_factory,
                evaluator_factory=self._evaluator_factory,
            )

            final_state = meta.run(
                max_steps=self.config.decomposition.max_steps,
            )

            metrics = final_state.metrics.copy()
            metrics["total_steps"] = float(final_state.step)
            metrics["budget_used"] = final_state.budget_used

            artifacts: dict[str, Any] = {
                "error_history": list(final_state.error_history),
                "n_subproblems": len(meta.subproblems),
                "subproblem_names": [s.name for s in meta.subproblems],
            }

            for name, solver in meta.solver_agents.items():
                artifacts[f"solver_{name}_error_history"] = list(solver.state.error_history)

            return self._create_result(
                status=final_state.status,
                metrics=metrics,
                artifacts=artifacts,
            )

        except Exception as e:
            return self._create_result(
                status=ExecutionStatus.FAILED,
                error=str(e),
            )
