"""The harness ``callable`` target: run one MCTS basis-selection cell.

Wired into a harness ``EvalConfig`` as
``{type: callable, params: {function: "src.integrations.eval_harness.target:run_basis_cell"}}``.
The built-in ``CallableTarget`` calls ``run_basis_cell(item.inputs)`` (a dict) and
wraps the returned dict in a ``TargetOutput`` (capturing latency/errors), so this
function takes a dict and returns a dict.

For each cell it (1) queries the arm's policy at the root state to record the
``chosen_action`` / ``topk_actions`` the :class:`PolicyTopKScorer` compares against
the oracle label, and (2) runs the full MCTS rollout to record ``final_residual`` /
``rollouts_used`` for the :class:`FinalResidualScorer`. Both reuse the shared
``_centaur_common`` primitives verbatim — the cell loop is not re-implemented here.

All heavy imports (torch, numpy, the PDE game) are local so importing this module
costs nothing on a base install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from src.integrations.eval_harness.config import BasisCellParams
    from src.integrations.lm_studio.client import LMStudioClient
    from src.mcts.evaluator import Evaluator
    from src.modeling.model import AlphaGalerkinModel


def run_basis_cell(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run one MCTS basis-selection cell and return its serialisable result.

    Args:
        inputs: A :class:`BasisCellParams`-shaped dict (validated here).

    Returns:
        A dict with ``final_residual``, ``rollouts_used``, ``chosen_action``,
        ``topk_actions``, ``value``, and the echoed ``pde_family`` / ``arm`` /
        ``seed`` — the fields the scorers consume.

    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from src.integrations.eval_harness.config import BasisCellParams  # noqa: PLC0415
    from src.pde.mcts_adapter import PDEGameAdapter  # noqa: PLC0415
    from src.poc.scenarios._centaur_common import (  # noqa: PLC0415
        build_basis_game,
        build_pde_operator,
        enumerate_basis_descriptions,
        run_basis_selection_cell,
    )

    params = BasisCellParams.model_validate(inputs)
    operator = build_pde_operator(params.pde_family)
    game = build_basis_game(
        params.pde_family,
        operator,
        max_basis_functions=params.max_basis_functions,
        n_candidate_bases=params.n_candidate_bases,
        target_residual=params.target_residual,
    )
    descriptions = enumerate_basis_descriptions(game)

    lm_client = _build_lm_client(params)
    trained_model, device = _load_trained_model(params)
    try:
        # (1) Root-state policy query -> chosen / top-k actions for PolicyTopKScorer.
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        root_evaluator = _build_evaluator(
            params, game, descriptions, lm_client, trained_model, device
        )
        adapter = PDEGameAdapter(game)
        chosen_action, topk_actions, value = _query_root_policy(
            root_evaluator, adapter.get_state(), adapter.get_legal_actions(), params.topk
        )

        # (2) Full MCTS rollout -> final residual for FinalResidualScorer. Re-seed
        # so the rollout is reproducible regardless of the root query's RNG draws.
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        cell_evaluator = _build_evaluator(
            params, game, descriptions, lm_client, trained_model, device
        )
        outcome = run_basis_selection_cell(
            game=game,
            evaluator=cell_evaluator,
            target_residual=params.target_residual,
            max_rollouts=params.max_rollouts,
            n_simulations=params.n_simulations,
        )
    finally:
        if lm_client is not None:
            lm_client.close()

    return {
        "pde_family": params.pde_family,
        "arm": params.arm,
        "seed": params.seed,
        "final_residual": float(outcome.final_residual),
        "rollouts_used": int(outcome.rollouts_used),
        "chosen_action": chosen_action,
        "topk_actions": topk_actions,
        "value": value,
    }


def _build_evaluator(
    params: BasisCellParams,
    game: Any,
    descriptions: list[str],
    lm_client: LMStudioClient | None,
    trained_model: AlphaGalerkinModel | None,
    device: torch.device | None,
) -> Evaluator:
    """Construct the arm evaluator via the shared centaur factory."""
    from src.poc.scenarios._centaur_common import build_arm_evaluator  # noqa: PLC0415

    return build_arm_evaluator(
        params.arm,
        game=game,
        pde_name=params.pde_family,
        basis_descriptions=descriptions,
        seed=params.seed,
        lm_client=lm_client,
        trained_model=trained_model,
        device=device,
    )


def _query_root_policy(
    evaluator: Evaluator,
    state: Any,
    legal_actions: list[int],
    topk: int,
) -> tuple[int | None, list[int], float]:
    """Return ``(chosen_action, topk_actions, value)`` from one policy evaluation."""
    import numpy as np  # noqa: PLC0415

    if not legal_actions:
        return None, [], 0.0
    result = evaluator.evaluate(state, legal_actions)
    policy = np.asarray(result.policy, dtype=np.float64)
    ranked = sorted(legal_actions, key=lambda action: float(policy[action]), reverse=True)
    return int(ranked[0]), [int(action) for action in ranked[:topk]], float(result.value)


def _build_lm_client(params: BasisCellParams) -> LMStudioClient | None:  # pragma: no cover
    """Build an ``LMStudioClient`` for the LLM arm, else ``None`` (live server only)."""
    if params.arm != "llm":
        return None
    from src.integrations.lm_studio.client import LMStudioClient  # noqa: PLC0415
    from src.integrations.lm_studio.config import LMStudioConfig  # noqa: PLC0415

    return LMStudioClient(LMStudioConfig(**(params.lm_studio or {})))


def _load_trained_model(  # pragma: no cover - trained arm needs a checkpoint (integration only)
    params: BasisCellParams,
) -> tuple[AlphaGalerkinModel | None, torch.device | None]:
    """Load the trained model + device for the trained arm, else ``(None, None)``."""
    if params.arm != "trained":
        return None, None
    if not params.checkpoint_path:
        raise ValueError("arm='trained' requires checkpoint_path")
    from src.poc.device import resolve_device  # noqa: PLC0415
    from src.training.checkpoint import create_model_from_checkpoint  # noqa: PLC0415

    device = resolve_device("auto", context="eval_harness_target")
    model, _config = create_model_from_checkpoint(
        params.checkpoint_path, device=str(device), strict=False
    )
    return model, device


__all__ = ["run_basis_cell"]
