"""Deterministic prompt builder for the LM Studio policy-prior evaluator.

The prompt encodes the MCTS basis-selection state (residual statistics,
current basis indices, legal actions, basis library descriptions) as a
compact JSON-of-state plus a short instruction. Determinism is critical so
that ``prompt_hash`` is stable across runs with the same state — that hash
is the cache-key candidate and the log correlation field.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

PROMPT_HASH_LENGTH = 16
"""Number of hex characters retained from sha256 for the prompt hash."""

RESIDUAL_PREVIEW_LENGTH = 8
"""How many residual-channel samples to include verbatim in the prompt."""


def _summarise_residual_channel(state: NDArray[np.float32]) -> dict[str, float]:
    """Extract compact residual statistics from a state tensor.

    Args:
        state: ``(channels, H, W)`` state tensor. Channel 1 is the residual
            per ``BasisSelectionGame.to_tensor``.

    Returns:
        Dict of named scalars suitable for JSON serialisation.

    """
    if state.ndim < 3 or state.shape[0] < 3:
        return {"mean": 0.0, "abs_mean": 0.0, "abs_max": 0.0, "l2": 0.0}
    residual = state[1].astype(np.float64, copy=False)
    abs_residual = np.abs(residual)
    return {
        "mean": float(residual.mean()),
        "abs_mean": float(abs_residual.mean()),
        "abs_max": float(abs_residual.max()),
        "l2": float(np.sqrt(np.mean(residual**2))),
    }


def _selected_basis_indices(state: NDArray[np.float32]) -> list[int]:
    """Extract indices of already-selected bases from the state tensor.

    ``BasisSelectionGame.to_tensor`` packs binary indicators in channels
    ``[3, 3 + max_basis_functions)``. A channel of all-ones indicates the
    basis at that offset has been selected.
    """
    if state.ndim < 3 or state.shape[0] <= 3:
        return []
    indicators = state[3:]
    selected: list[int] = []
    for i, channel in enumerate(indicators):
        if float(channel.mean()) > 0.5:
            selected.append(i)
    return selected


def build_policy_prompt(
    state: NDArray[np.float32],
    *,
    legal_actions: list[int],
    action_space_size: int,
    pde_family: str,
    basis_descriptions: list[str],
) -> str:
    """Render a deterministic prompt for the LLM policy prior.

    Args:
        state: ``(channels, H, W)`` state tensor.
        legal_actions: Indices the LLM is allowed to recommend.
        action_space_size: Total action-space size (length of ``logits``).
        pde_family: Human-readable PDE name (e.g. ``"poisson"``).
        basis_descriptions: ``len == action_space_size``; the LLM gets a
            short description of each candidate basis.

    Returns:
        The rendered prompt string. Identical inputs always produce
        identical output (no time, no random salt).

    """
    if len(basis_descriptions) != action_space_size:
        raise ValueError(
            "basis_descriptions length "
            f"({len(basis_descriptions)}) must equal action_space_size "
            f"({action_space_size})"
        )
    residual_stats = _summarise_residual_channel(state)
    already_selected = _selected_basis_indices(state)
    legal_actions_sorted = sorted(set(legal_actions))

    payload: dict[str, Any] = {
        "task": "policy_prior_for_galerkin_basis_selection",
        "pde_family": pde_family,
        "action_space_size": action_space_size,
        "legal_actions": legal_actions_sorted,
        "already_selected": already_selected,
        "residual_stats": residual_stats,
        "basis_library": [
            {"index": i, "description": basis_descriptions[i]} for i in range(action_space_size)
        ],
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    instructions = (
        "You are a numerical-analysis assistant guiding an MCTS search "
        "over Galerkin basis functions. Given the residual statistics and "
        "the library of candidate bases, return a JSON object with three "
        "keys exactly:\n"
        '  "logits": a list of length action_space_size with one real-valued '
        "score per basis (any real number; the consumer applies softmax with "
        "illegal-action masking),\n"
        '  "value": a single float in [-1, 1] estimating the position '
        "quality after one step of basis addition,\n"
        '  "reasoning": one short sentence describing your top choice.\n'
        "Output JSON only — no markdown fences, no commentary."
    )
    return f"{instructions}\n\nSTATE:\n{payload_json}\n"


def prompt_hash(prompt: str) -> str:
    """Stable short hash of a prompt for log correlation."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:PROMPT_HASH_LENGTH]
