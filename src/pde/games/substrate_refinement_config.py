"""Pydantic config for the substrate-backed ``RefinementGame`` (Slice E)."""

from __future__ import annotations

from pydantic import Field, field_validator

from src.research.substrates.config import SubstrateConfig
from src.research.substrates.factory import OperatorName
from src.templates.config import BaseModuleConfig


class SubstrateRefinementConfig(BaseModuleConfig):
    """Engine + substrate knobs for ``SubstrateRefinementGame``.

    All tunables are typed fields — no hardcoded magic at call sites.
    """

    substrate: SubstrateConfig = Field(
        default_factory=lambda: SubstrateConfig(
            name="substrate_refinement_default",
            kind="tensor_grid",
            initial_side=4,
        ),
        description="Substrate identity and solve/mark knobs (looked up by kind).",
    )
    operator_name: OperatorName = Field(
        default="poisson",
        description="Which exact-solution Poisson operator to pair with the substrate.",
    )
    lshape_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Domain scale for operator_name='lshape_poisson'.",
    )
    max_steps: int = Field(
        default=8,
        ge=1,
        le=10_000,
        description="Maximum single-element refine actions before terminal.",
    )
    error_tolerance: float = Field(
        default=1e-4,
        gt=0.0,
        description="Terminate when error_estimate drops to this tolerance.",
    )
    computational_budget: float = Field(
        default=1e6,
        gt=0.0,
        description="Initial budget_remaining; each refine subtracts refine_cost.",
    )
    refine_cost: float = Field(
        default=1.0,
        gt=0.0,
        description="Budget consumed per successful refine action.",
    )
    max_action_space: int = Field(
        default=4096,
        ge=1,
        le=1_000_000,
        description=(
            "Fixed action-index range for MCTS/evaluators. Valid actions are the "
            "refinable unit indices in [0, n_units) capped by this bound."
        ),
    )
    reward_cost_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Subtracted from error reduction in get_reward (0 = error-only).",
    )
    winner_error_threshold: float = Field(
        default=0.1,
        gt=0.0,
        description="Terminal winner=1 iff error_estimate is strictly below this.",
    )

    @field_validator("substrate")
    @classmethod
    def _substrate_kind_supported(cls, value: SubstrateConfig) -> SubstrateConfig:
        """Reject kinds the factory does not know how to build."""
        allowed: tuple[str, ...] = ("tensor_grid", "skfem_tri")
        if value.kind not in allowed:
            raise ValueError(f"substrate.kind must be one of {allowed}, got {value.kind!r}")
        return value


__all__ = ["SubstrateRefinementConfig"]
