"""Prompt-builder and hash-stability tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.integrations.lm_studio.prompt import (
    PROMPT_HASH_LENGTH,
    build_policy_prompt,
    prompt_hash,
)


def _state(channels: int = 5, side: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((channels, side, side)).astype(np.float32)


def _basis_descriptions(n: int) -> list[str]:
    return [f"basis_{i}" for i in range(n)]


def test_prompt_deterministic_for_same_inputs() -> None:
    state = _state()
    a = build_policy_prompt(
        state,
        legal_actions=[0, 2, 3],
        action_space_size=4,
        pde_family="poisson",
        basis_descriptions=_basis_descriptions(4),
    )
    b = build_policy_prompt(
        state,
        legal_actions=[0, 2, 3],
        action_space_size=4,
        pde_family="poisson",
        basis_descriptions=_basis_descriptions(4),
    )
    assert a == b
    assert prompt_hash(a) == prompt_hash(b)
    assert len(prompt_hash(a)) == PROMPT_HASH_LENGTH


def test_hash_changes_when_state_changes() -> None:
    state_a = _state()
    state_b = state_a.copy()
    state_b[1, 0, 0] += 1.0  # perturb residual channel
    args = {
        "legal_actions": [0, 1, 2, 3],
        "action_space_size": 4,
        "pde_family": "poisson",
        "basis_descriptions": _basis_descriptions(4),
    }
    prompt_a = build_policy_prompt(state_a, **args)
    prompt_b = build_policy_prompt(state_b, **args)
    assert prompt_a != prompt_b
    assert prompt_hash(prompt_a) != prompt_hash(prompt_b)


def test_legal_actions_appear_in_prompt() -> None:
    state = _state()
    prompt = build_policy_prompt(
        state,
        legal_actions=[3, 1],
        action_space_size=4,
        pde_family="poisson",
        basis_descriptions=_basis_descriptions(4),
    )
    assert '"legal_actions":[1,3]' in prompt  # sorted + deduped


def test_basis_descriptions_appear_in_prompt() -> None:
    state = _state()
    prompt = build_policy_prompt(
        state,
        legal_actions=[0, 1],
        action_space_size=2,
        pde_family="poisson",
        basis_descriptions=["fourier(k=1,1)", "rbf(c=0.50,0.50)"],
    )
    assert "fourier(k=1,1)" in prompt
    assert "rbf(c=0.50,0.50)" in prompt


@pytest.mark.parametrize(
    "shape",
    [(2, 4, 4), (4, 4), (1,)],
)
def test_summarise_residual_handles_low_dim_state(shape: tuple[int, ...]) -> None:
    """Low-channel / low-rank state must not crash the prompt builder.

    Covers the early-return guard inside ``_summarise_residual_channel``.
    """
    state = np.zeros(shape, dtype=np.float32)
    prompt = build_policy_prompt(
        state,
        legal_actions=[0],
        action_space_size=1,
        pde_family="poisson",
        basis_descriptions=["b_0"],
    )
    # Residual stats are always emitted; the early-return path supplies
    # zeros rather than crashing on a malformed state.
    assert '"residual_stats"' in prompt


def test_prompt_omits_already_selected_field() -> None:
    """`already_selected` was removed: see prompt.py docstring for rationale.

    ``BasisSelectionGame.to_tensor`` packs channels 3+ as selection-order
    slots, not one-hot action indicators, so reading them produced
    misleading "selected basis indices". The prompt now relies solely on
    ``legal_actions`` (which already excludes selected bases) to communicate
    the constraint to the LLM.
    """
    state = np.zeros((5, 4, 4), dtype=np.float32)
    state[3].fill(1.0)  # the 1st selection-slot is "occupied" in the state
    prompt = build_policy_prompt(
        state,
        legal_actions=[1],
        action_space_size=2,
        pde_family="poisson",
        basis_descriptions=["b_0", "b_1"],
    )
    assert "already_selected" not in prompt
    # Sanity: the legal_actions constraint is still present.
    assert '"legal_actions":[1]' in prompt


def test_mismatched_basis_descriptions_raises() -> None:
    state = _state()
    with pytest.raises(ValueError):
        build_policy_prompt(
            state,
            legal_actions=[0],
            action_space_size=4,
            pde_family="poisson",
            basis_descriptions=["only_one"],
        )
