"""Domain-free state for sequential refinement games.

``RefinementState`` is the neutral counterpart to ``src.pde.game.PDEState``: it
carries only what a refinement *engine* needs — a field of ``values``, per-unit
``indicators``, and the four scalars MCTS reads (``error_estimate``, ``dof``,
``budget_remaining``, ``step``). It deliberately does **not** know about PDE
concepts (coords, basis coefficients, mesh levels); those live on the PDE
domain's own state. ``PDEState`` is unchanged and gains
``to_refinement()`` / ``from_refinement()`` converters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class RefinementLike(Protocol):
    """The four scalar fields the MCTS adapter reads off any refinement state.

    Both ``RefinementState`` and ``src.pde.game.PDEState`` satisfy this, so the
    ``RefinementGameAdapter`` can operate on either without conversion.
    """

    error_estimate: float
    dof: int
    budget_remaining: float
    step: int


@dataclass
class RefinementState:
    """State of a sequential-refinement episode.

    Attributes:
        values: Domain field values (e.g. a PDE solution). Shape is domain-defined.
        indicators: Per-unit refinement indicators (e.g. per-element error).
            Drives which unit to refine next.
        error_estimate: Current global error/objective estimate MCTS optimises.
        dof: Active degrees of freedom (cost proxy).
        step: Number of actions applied so far.
        budget_remaining: Remaining computational budget.
        history: Actions applied so far, in order.

    """

    values: NDArray[np.float32]
    indicators: NDArray[np.float32]
    error_estimate: float = 1.0
    dof: int = 0
    step: int = 0
    budget_remaining: float = 1e6
    history: list[int] = field(default_factory=list)

    def clone(self) -> RefinementState:
        """Return a deep copy safe to mutate in a sibling MCTS branch."""
        return RefinementState(
            values=self.values.copy(),
            indicators=self.indicators.copy(),
            error_estimate=self.error_estimate,
            dof=self.dof,
            step=self.step,
            budget_remaining=self.budget_remaining,
            history=list(self.history),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain Python types (arrays become lists)."""
        return {
            "values": self.values.tolist(),
            "indicators": self.indicators.tolist(),
            "error_estimate": self.error_estimate,
            "dof": self.dof,
            "step": self.step,
            "budget_remaining": self.budget_remaining,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RefinementState:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            values=np.asarray(data["values"], dtype=np.float32),
            indicators=np.asarray(data["indicators"], dtype=np.float32),
            error_estimate=float(data["error_estimate"]),
            dof=int(data["dof"]),
            step=int(data["step"]),
            budget_remaining=float(data["budget_remaining"]),
            history=list(data.get("history", [])),
        )
