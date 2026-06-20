"""Pydantic configuration for the eval-harness adapter (no hardcoded values).

Two models, both ``extra="forbid"`` so typos fail loud:

- :class:`BasisCellParams` validates the ``inputs`` dict the harness ``callable``
  target hands to :func:`~src.integrations.eval_harness.target.run_basis_cell`
  (one MCTS basis-selection cell).
- :class:`OracleDatasetParams` validates the ``params`` of the ``basis_oracle``
  dataset and is the spec consumed by
  :func:`~src.integrations.eval_harness.dataset.build_dataset_jsonl`. It sweeps
  ``pde_families × seeds`` and emits one :class:`BasisCellParams` per cell via
  :meth:`OracleDatasetParams.iter_cells`.

The two share the per-cell knobs through :class:`_BasisKnobs` so a new knob is
declared exactly once.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ArmName = Literal["random", "trained", "llm"]
"""The three MCTS evaluator arms, mirroring ``_centaur_common.build_arm_evaluator``."""


class _BasisKnobs(BaseModel):
    """Per-cell MCTS basis-selection knobs shared by cell + dataset configs."""

    model_config = ConfigDict(extra="forbid")

    arm: ArmName = Field(
        default="random",
        description="MCTS evaluator arm: 'random', 'trained' (checkpoint), or 'llm' (LM Studio).",
    )
    max_basis_functions: int = Field(
        default=8,
        ge=1,
        description="Maximum bases the game may add before terminating.",
    )
    n_candidate_bases: int = Field(
        default=16,
        ge=1,
        description="Size of the candidate basis library (== MCTS action space).",
    )
    target_residual: float = Field(
        default=1e-2,
        gt=0.0,
        description="Error tolerance that terminates the cell (also the residual gate).",
    )
    max_rollouts: int = Field(
        default=256,
        ge=1,
        description="Hard cap on accumulated MCTS simulations per cell.",
    )
    n_simulations: int = Field(
        default=16,
        ge=1,
        description="MCTS simulations per macro-step (action selection).",
    )
    topk: int = Field(
        default=3,
        ge=1,
        description="Top-k cutoff for the policy-accuracy scorer at the root state.",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description="Trained-model checkpoint path (required only when arm='trained').",
    )
    lm_studio: dict[str, Any] | None = Field(
        default=None,
        description="LMStudioConfig kwargs (used only when arm='llm'); None uses defaults.",
    )

    @model_validator(mode="after")
    def _check_topk_within_action_space(self) -> _BasisKnobs:
        """``topk`` cannot exceed the candidate library size."""
        if self.topk > self.n_candidate_bases:
            raise ValueError(
                f"topk ({self.topk}) cannot exceed n_candidate_bases ({self.n_candidate_bases})"
            )
        return self


class BasisCellParams(_BasisKnobs):
    """Inputs for one MCTS basis-selection cell (one EvalItem ``inputs`` dict)."""

    pde_family: str = Field(
        ...,
        min_length=1,
        description="PDE registry name (a key of _centaur_common.PDE_TYPE_MAP).",
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Per-cell RNG seed (numpy + torch are seeded before the solve).",
    )


class OracleDatasetParams(_BasisKnobs):
    """Spec for the ``basis_oracle`` dataset: a ``pde_families × seeds`` sweep."""

    pde_families: list[str] = Field(
        ...,
        min_length=1,
        description="PDE registry names to sweep (each a key of PDE_TYPE_MAP).",
    )
    seeds: list[int] = Field(
        default_factory=lambda: [0],
        min_length=1,
        description="Per-PDE seeds to sweep; the cartesian product defines the items.",
    )

    def iter_cells(self) -> Iterator[BasisCellParams]:
        """Yield one :class:`BasisCellParams` per ``(pde_family, seed)`` cell."""
        knobs = {name: getattr(self, name) for name in _BasisKnobs.model_fields}
        for pde_family in self.pde_families:
            for seed in self.seeds:
                yield BasisCellParams(pde_family=pde_family, seed=seed, **knobs)


__all__ = ["ArmName", "BasisCellParams", "OracleDatasetParams"]
