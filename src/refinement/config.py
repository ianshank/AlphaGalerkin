"""Configuration for domain-free refinement games.

``RefinementGameConfig`` carries the engine-level knobs shared by every
refinement domain and a **generic** ``domain_config`` payload typed to the
concrete domain's config. Making the config generic (rather than typing the
payload as the ``BaseModuleConfig`` base) is what keeps ``mypy --strict`` precise
*and* prevents Pydantic v2 from coercing the payload to the base and dropping the
subclass's fields — the field-loss bug class the round-trip test guards against.

There is deliberately **no** ``PDEType`` / ``pde_config`` here: the engine is
domain-agnostic, and the PDE-shaped config (``src.pde.config.PDEGameConfig``)
stays exactly as it is.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import Field

from src.templates.config import BaseModuleConfig

TDomain = TypeVar("TDomain", bound=BaseModuleConfig)


class RefinementGameConfig(BaseModuleConfig, Generic[TDomain]):
    """Engine-level config for a sequential refinement game.

    Parametrise with the concrete domain config type to keep its fields typed:
    ``RefinementGameConfig[LambdaSchedulingConfig](domain_config=...)``.
    """

    max_steps: int = Field(
        default=30,
        ge=1,
        le=10_000,
        description="Maximum refinement actions before the episode terminates.",
    )
    error_tolerance: float = Field(
        default=1e-4,
        gt=0.0,
        description="Convergence tolerance on the error/objective estimate.",
    )
    computational_budget: float = Field(
        default=1e6,
        gt=0.0,
        description="Total budget the episode may spend (cost units).",
    )
    use_intermediate_rewards: bool = Field(
        default=False,
        description=(
            "Opt in to per-edge reward shaping in MCTS (get_reward along the "
            "selection path). Default False reproduces terminal-only backup."
        ),
    )
    reward_discount: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Discount gamma applied to intermediate rewards.",
    )
    domain_config: TDomain = Field(
        ...,
        description="Concrete domain payload (a BaseModuleConfig subclass).",
    )
