"""Greedy oracle equals brute-force argmin (requires the full torch stack)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from src.integrations.eval_harness.oracle import greedy_basis_oracle  # noqa: E402
from src.poc.scenarios._centaur_common import (  # noqa: E402
    build_basis_game,
    build_pde_operator,
)


def test_greedy_oracle_matches_bruteforce() -> None:
    operator = build_pde_operator("poisson")
    game = build_basis_game(
        "poisson",
        operator,
        max_basis_functions=4,
        n_candidate_bases=8,
        target_residual=1e-3,
    )
    state = game.get_initial_state()
    legal = game.get_valid_actions(state)
    assert legal, "expected legal actions at the initial state"

    brute = {a: float(game.apply_action(state, a).error_estimate) for a in legal}
    expected_ranked = sorted(legal, key=lambda a: brute[a])

    label = greedy_basis_oracle(game, state)
    assert label["ranked_actions"] == expected_ranked
    assert label["greedy_action"] == expected_ranked[0]
    assert label["greedy_residual"] == pytest.approx(brute[expected_ranked[0]])
    assert len(label["ranked_actions"]) == len(legal)
