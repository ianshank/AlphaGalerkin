"""CouplingAgent for managing interface conditions between subdomains.

Handles boundary data exchange and convergence checking in
Schwarz-type domain decomposition iterations.

Example:
    from src.agents.coupling import CouplingAgent
    from src.agents.config import CouplingConfig, CouplingType

    config = CouplingConfig(
        name="coupling",
        coupling_type=CouplingType.ROBIN_ROBIN,
        relaxation_factor=0.5,
    )
    agent = CouplingAgent(config)
    agent.setup()
    updated_bcs = agent.exchange_boundary_data(solver_data)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.agents.base import AgentState, BaseAgent
from src.agents.config import CouplingType, MessageType
from src.templates.base import ExecutionStatus
from src.templates.logging import create_logger_class

if TYPE_CHECKING:
    from src.agents.config import CouplingConfig
    from src.agents.message import MessageBus

CouplingLogger = create_logger_class("CouplingAgent")


class CouplingAgent(BaseAgent):
    """Agent that enforces coupling conditions between solver subdomains.

    Implements Dirichlet-Neumann, Robin-Robin, and Mortar coupling
    with configurable relaxation and convergence detection.

    Args:
        config: Coupling agent configuration.
        message_bus: Optional bus for inter-agent communication.
        agent_id: Optional explicit ID.

    """

    def __init__(
        self,
        config: CouplingConfig,
        message_bus: MessageBus | None = None,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(config, message_bus, agent_id)
        self._previous_bcs: dict[str, NDArray[np.float32]] = {}
        self._residual_history: list[float] = []
        self._coupling_logger = CouplingLogger(
            component="coupling",
            run_id=self._agent_id,
        )

    @property
    def coupling_config(self) -> CouplingConfig:
        """Typed access to coupling-specific config."""
        from src.agents.config import CouplingConfig

        assert isinstance(self.config, CouplingConfig)
        return self.config

    def setup(self) -> None:
        """Initialize coupling state."""
        self._previous_bcs = {}
        self._residual_history = []
        self._coupling_logger.info(
            "coupling_setup",
            coupling_type=self.coupling_config.coupling_type.value,
            relaxation=self.coupling_config.relaxation_factor,
            tolerance=self.coupling_config.tolerance,
        )

    def step(self) -> AgentState:
        """Process one coupling iteration.

        Receives boundary data from solver agents via the message bus,
        applies coupling conditions, and checks convergence.

        Returns:
            Updated agent state.

        """
        boundary_messages = self.receive_messages(MessageType.BOUNDARY_DATA)
        solver_data: dict[str, NDArray[np.float32]] = {}
        for msg in boundary_messages:
            if "agent_id" in msg.payload and "boundary_values" in msg.payload:
                solver_data[msg.payload["agent_id"]] = np.array(
                    msg.payload["boundary_values"], dtype=np.float32
                )

        if solver_data:
            updated_bcs = self.exchange_boundary_data(solver_data)
            interface_residual = self._compute_interface_residual(solver_data, updated_bcs)
            self._residual_history.append(interface_residual)
            self._state.error_history.append(interface_residual)
        else:
            self._residual_history.append(float("inf"))

        self._state.step += 1
        self.update_budget(self.coupling_config.budget_per_step)

        is_converged = self.is_converged()
        if is_converged or self._state.step >= self.coupling_config.max_iterations:
            self._state.status = ExecutionStatus.COMPLETED

        if self._message_bus is not None:
            self.send_message(
                receiver="*",
                msg_type=MessageType.CONVERGENCE_CHECK,
                payload={
                    "converged": is_converged,
                    "iteration": self._state.step,
                    "residual": self._residual_history[-1]
                    if self._residual_history
                    else float("inf"),
                },
            )

        self._state.metrics = self.get_metrics()
        return self._state

    def exchange_boundary_data(
        self,
        solver_data: dict[str, NDArray[np.float32]],
    ) -> dict[str, NDArray[np.float32]]:
        """Apply coupling conditions and return updated boundary data.

        Uses relaxation to blend new boundary data with previous values:
        ``bc_new = (1 - omega) * bc_old + omega * bc_received``

        Args:
            solver_data: Boundary values from each solver, keyed by agent ID.

        Returns:
            Updated boundary conditions for each solver.

        """
        omega = self.coupling_config.relaxation_factor
        coupling_type = self.coupling_config.coupling_type
        updated: dict[str, NDArray[np.float32]] = {}

        solver_ids = list(solver_data.keys())

        for solver_id in solver_ids:
            received = solver_data[solver_id]

            if solver_id in self._previous_bcs:
                old = self._previous_bcs[solver_id]
                if old.shape == received.shape:
                    if coupling_type in (
                        CouplingType.ROBIN_ROBIN,
                        CouplingType.DIRICHLET_NEUMANN,
                        CouplingType.MORTAR,
                    ):
                        blended = (1.0 - omega) * old + omega * received
                    else:
                        blended = received
                    updated[solver_id] = blended.astype(np.float32)
                else:
                    updated[solver_id] = received
            else:
                updated[solver_id] = received

            self._previous_bcs[solver_id] = updated[solver_id].copy()

        return updated

    def _compute_interface_residual(
        self,
        solver_data: dict[str, NDArray[np.float32]],
        updated_bcs: dict[str, NDArray[np.float32]],
    ) -> float:
        """Compute the interface residual (mismatch between solvers).

        Args:
            solver_data: Original boundary data from solvers.
            updated_bcs: Updated boundary conditions after coupling.

        Returns:
            L2 norm of the interface mismatch.

        """
        total_residual = 0.0
        n_pairs = 0
        for solver_id in solver_data:
            if solver_id in updated_bcs:
                diff = solver_data[solver_id] - updated_bcs[solver_id]
                total_residual += float(np.sqrt(np.mean(diff**2)))
                n_pairs += 1
        return total_residual / max(n_pairs, 1)

    def is_converged(self) -> bool:
        """Check if the coupling has converged.

        Convergence requires the interface residual to be below tolerance
        for the last ``convergence_window`` consecutive steps.

        Returns:
            True if converged.

        """
        window = self.coupling_config.convergence_window
        tol = self.coupling_config.tolerance

        if len(self._residual_history) < window:
            return False

        recent = self._residual_history[-window:]
        return all(r < tol for r in recent)

    def reset(self) -> None:
        """Reset coupling state."""
        self._previous_bcs = {}
        self._residual_history = []
        self._state = self._create_initial_state()

    def get_metrics(self) -> dict[str, float]:
        """Return coupling metrics."""
        metrics: dict[str, float] = {
            "iteration": float(self._state.step),
            "converged": 1.0 if self.is_converged() else 0.0,
        }
        if self._residual_history:
            metrics["interface_residual"] = self._residual_history[-1]
            metrics["min_residual"] = min(self._residual_history)
        return metrics
