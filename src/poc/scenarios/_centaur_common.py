"""Shared MCTS basis-selection primitives for "centaur" PoC scenarios.

Brown's centaur loop — search (MCTS) + a learned/LLM policy prior — is used
by three surfaces in this codebase:

    * :mod:`src.poc.scenarios.llm_prior_ablation` (random / trained / LLM arms)
    * :mod:`src.poc.scenarios.scaling_law` (residual vs MCTS-simulation budget)
    * :mod:`src.agents.research_loop` (sweep across a manifest of problems)

To keep a single source of truth for the operator/game construction and the
inner MCTS rollout loop, those primitives live here as free functions. Every
caller injects its own configuration values, so there are no hardcoded
budgets, tolerances, or library sizes in this module.

The functions are intentionally side-effect free with respect to global RNG:
seeding is the caller's responsibility (so each scenario controls its own
reproducibility contract).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

from src.mcts.evaluator import RandomEvaluator
from src.mcts.search import MCTS
from src.pde.config import (
    BasisSelectionConfig,
    PDEConfig,
    PDEGameConfig,
    PDEType,
)
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.mcts_adapter import PDEGameAdapter
from src.pde.registry import get_pde_operator

if TYPE_CHECKING:
    import torch

    from src.integrations.lm_studio.client import LMStudioClient
    from src.mcts.evaluator import Evaluator
    from src.modeling.model import AlphaGalerkinModel
    from src.pde.operators import PDEOperator
    from src.poc.logging import ScenarioLogger


# Canonical mapping from PDE registry name to PDEType enum. Shared by every
# centaur scenario so a new operator is wired in exactly once. ``poisson_lshaped``
# reuses the POISSON enum because it is a domain variant, not a distinct type.
PDE_TYPE_MAP: dict[str, PDEType] = {
    "poisson": PDEType.POISSON,
    "burgers": PDEType.BURGERS,
    "heat": PDEType.HEAT,
    "advection_diffusion": PDEType.ADVECTION_DIFFUSION,
    "navier_stokes": PDEType.NAVIER_STOKES,
    "poisson_lshaped": PDEType.POISSON,
    "helmholtz": PDEType.HELMHOLTZ,
    "biharmonic": PDEType.BIHARMONIC,
}


@runtime_checkable
class SupportsWarning(Protocol):
    """Structural type for the structlog-style loggers used by callers."""

    def warning(self, event: str, **kwargs: object) -> object:
        """Emit a warning-level structured event."""
        ...


class CellOutcome(NamedTuple):
    """Result of a single MCTS basis-selection cell.

    Attributes:
        rollouts_used: Total MCTS simulations accumulated to reach the stop
            condition (target residual, terminal state, or budget cap).
        final_residual: ``adapter.current_error`` at the stop point.

    """

    rollouts_used: int
    final_residual: float


def build_pde_operator(pde_name: str) -> PDEOperator:
    """Instantiate a registered PDE operator by name.

    Args:
        pde_name: Registry name (must be a key of :data:`PDE_TYPE_MAP`).

    Returns:
        A constructed :class:`~src.pde.operators.PDEOperator`.

    Raises:
        ValueError: If ``pde_name`` has no :data:`PDE_TYPE_MAP` entry.

    """
    if pde_name not in PDE_TYPE_MAP:
        raise ValueError(f"PDE {pde_name!r} has no PDEType mapping; known: {sorted(PDE_TYPE_MAP)}")
    pde_config = PDEConfig(name=pde_name, pde_type=PDE_TYPE_MAP[pde_name])
    operator_cls = get_pde_operator(pde_name)
    return operator_cls(pde_config)


def build_basis_game(
    pde_name: str,
    operator: PDEOperator,
    *,
    max_basis_functions: int,
    n_candidate_bases: int,
    target_residual: float,
) -> BasisSelectionGame:
    """Build a :class:`BasisSelectionGame` around an operator.

    Args:
        pde_name: Used only to name the nested configs.
        operator: The PDE operator the game approximates.
        max_basis_functions: Maximum bases the game may add before terminating.
        n_candidate_bases: Size of the candidate library (== action space).
        target_residual: Error tolerance that terminates the game.

    Returns:
        A configured :class:`BasisSelectionGame`.

    """
    basis_config = BasisSelectionConfig(
        name=f"{pde_name}_basis",
        max_basis_functions=max_basis_functions,
        n_candidate_bases=n_candidate_bases,
    )
    game_config = PDEGameConfig(
        name=f"{pde_name}_game",
        pde_config=operator.config,
        game_mode="basis_selection",
        basis_config=basis_config,
        error_tolerance=target_residual,
    )
    return BasisSelectionGame(operator, game_config)


def enumerate_basis_descriptions(game: BasisSelectionGame) -> list[str]:
    """Return human-readable descriptions of every candidate basis action."""
    return [game.action_to_string(i) for i in range(game.action_space_size)]


def build_arm_evaluator(
    arm: str,
    *,
    game: BasisSelectionGame,
    pde_name: str,
    basis_descriptions: list[str],
    seed: int,
    lm_client: LMStudioClient | None = None,
    trained_model: AlphaGalerkinModel | None = None,
    device: torch.device | str | None = None,
    scenario_logger: ScenarioLogger | None = None,
) -> Evaluator:
    """Construct the MCTS evaluator for an arm.

    Centralises the random / trained / LLM switch so the three centaur
    scenarios share one definition. Heavy evaluators are imported lazily so
    the cold path stays light when an arm is unused.

    Args:
        arm: One of ``"random"``, ``"trained"``, ``"llm"``.
        game: The basis-selection game (provides the action-space size).
        pde_name: PDE family label forwarded to the LLM prompt.
        basis_descriptions: Candidate-basis descriptions for the LLM prompt.
        seed: Per-cell seed forwarded to the LLM client for determinism.
        lm_client: A constructed ``LMStudioClient`` (required for ``"llm"``).
        trained_model: A loaded model (required for ``"trained"``).
        device: Torch device for the trained evaluator.
        scenario_logger: Optional structured logger forwarded to the LLM evaluator.

    Returns:
        An object satisfying the MCTS :class:`Evaluator` protocol.

    Raises:
        ValueError: For an unknown ``arm``.
        RuntimeError: If a required resource for the arm is missing.

    """
    if arm == "random":
        return RandomEvaluator(n_actions=game.action_space_size)
    if arm == "trained":
        if trained_model is None:
            raise RuntimeError("trained arm requested but trained_model is None")
        if device is None:
            raise RuntimeError("trained arm requested but device is None")
        from src.mcts.evaluator import FNetEvaluator

        return FNetEvaluator(model=trained_model, device=device)
    if arm == "llm":
        if lm_client is None:
            raise RuntimeError("LLM arm requested but lm_client is None")
        from src.integrations.lm_studio.evaluator import LMStudioEvaluator

        return LMStudioEvaluator(
            lm_client,
            action_space_size=game.action_space_size,
            pde_family=pde_name,
            basis_descriptions=basis_descriptions,
            seed=seed,
            scenario_logger=scenario_logger,
        )
    raise ValueError(f"unknown arm {arm!r}")


def run_basis_selection_cell(
    *,
    game: BasisSelectionGame,
    evaluator: Evaluator,
    target_residual: float,
    max_rollouts: int,
    n_simulations: int,
    scenario_logger: SupportsWarning | None = None,
) -> CellOutcome:
    """Run one MCTS basis-selection cell to its stop condition.

    The caller is responsible for seeding global RNG *before* calling this
    function (``MCTS.__init__`` has no seed kwarg). The loop reuses the
    subtree rooted at each chosen action via :meth:`MCTS.advance` so search
    work is not discarded between macro-steps.

    Args:
        game: A fresh basis-selection game.
        evaluator: The arm's MCTS evaluator.
        target_residual: Stop once ``adapter.current_error`` drops to/below this.
        max_rollouts: Hard cap on accumulated simulations.
        n_simulations: Simulations per macro-step (action selection).
        scenario_logger: Optional logger for the early-exit warning.

    Returns:
        A :class:`CellOutcome` with rollouts used and final residual.

    """
    adapter = PDEGameAdapter(game)
    rollouts_used = 0

    if adapter.current_error <= target_residual:
        return CellOutcome(rollouts_used, float(adapter.current_error))

    mcts = MCTS(evaluator=evaluator, n_simulations=n_simulations)

    while (
        not adapter.is_terminal()
        and adapter.current_error > target_residual
        and rollouts_used + n_simulations <= max_rollouts
    ):
        action = mcts.get_action(adapter, temperature=0.0, add_noise=False)
        if action < 0:
            if scenario_logger is not None:
                scenario_logger.warning(
                    "cell_loop_early_exit",
                    reason="evaluator_returned_invalid_action",
                    action=action,
                    rollouts_used=rollouts_used,
                    current_error=float(adapter.current_error),
                )
            break
        adapter.apply_action(action)
        rollouts_used += n_simulations
        mcts.advance(action)

    return CellOutcome(rollouts_used, float(adapter.current_error))


__all__ = [
    "PDE_TYPE_MAP",
    "CellOutcome",
    "SupportsWarning",
    "build_arm_evaluator",
    "build_basis_game",
    "build_pde_operator",
    "enumerate_basis_descriptions",
    "run_basis_selection_cell",
]
