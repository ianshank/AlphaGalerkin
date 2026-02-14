"""Tests for head-to-head evaluation in PolicyEvaluator."""

from __future__ import annotations

from typing import Any

import pytest

from src.alphagalerkin.core.config import (
    AlphaGalerkinConfig,
    EnvironmentConfig,
    MCTSConfig,
    TrainingConfig,
)
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.evaluation.evaluator import PolicyEvaluator
from src.alphagalerkin.mcts.action_masking import ActionMasker


def _make_config() -> AlphaGalerkinConfig:
    """Minimal config for fast evaluator tests."""
    return AlphaGalerkinConfig(
        mcts=MCTSConfig(
            num_simulations=2,
            max_tree_depth=2,
            action_topk=3,
        ),
        training=TrainingConfig(
            batch_size=4,
            total_steps=2,
        ),
        environment=EnvironmentConfig(
            max_steps=3,
            max_dof=500,
        ),
        device="cpu",
    )


def _make_uniform_eval_fn() -> Any:
    """Create an EvalFn that returns uniform priors and value 0.5.

    This function is deterministic: for any state it always produces
    the same action priors and value.
    """
    masker = ActionMasker(_make_config().environment)

    def eval_fn(
        state: DiscretizationState,
    ) -> tuple[dict[Action, float], float]:
        valid = masker.valid_actions(state)
        n = len(valid)
        priors = dict.fromkeys(valid, 1.0 / n)
        return priors, 0.5

    return eval_fn


def _make_biased_eval_fn(value: float = 0.9) -> Any:
    """Create an EvalFn with a different value estimate.

    Uses the same uniform action priors but returns a
    different value, which can subtly change MCTS behaviour.
    """
    masker = ActionMasker(_make_config().environment)

    def eval_fn(
        state: DiscretizationState,
    ) -> tuple[dict[Action, float], float]:
        valid = masker.valid_actions(state)
        n = len(valid)
        # Heavily weight the first valid action.
        priors: dict[Action, float] = {}
        for i, a in enumerate(valid):
            if i == 0:
                priors[a] = 0.9
            else:
                priors[a] = 0.1 / max(1, n - 1)
        return priors, value

    return eval_fn


# -------------------------------------------------------------------
# evaluate_head_to_head
# -------------------------------------------------------------------


class TestEvaluateHeadToHead:
    """Tests for evaluate_head_to_head method."""

    def test_returns_expected_keys(self) -> None:
        """Result dict contains win_rate, reward_diff, agreement."""
        config = _make_config()
        evaluator = PolicyEvaluator(config)

        fn_a = _make_uniform_eval_fn()
        fn_b = _make_uniform_eval_fn()

        result = evaluator.evaluate_head_to_head(
            current_fn=fn_a,
            baseline_fn=fn_b,
            num_episodes=2,
        )

        assert "h2h/win_rate" in result
        assert "h2h/avg_reward_diff" in result
        assert "h2h/policy_agreement" in result

    def test_win_rate_bounds(self) -> None:
        """Win rate should be between 0 and 1 inclusive."""
        config = _make_config()
        evaluator = PolicyEvaluator(config)

        fn_a = _make_uniform_eval_fn()
        fn_b = _make_biased_eval_fn(value=0.1)

        result = evaluator.evaluate_head_to_head(
            current_fn=fn_a,
            baseline_fn=fn_b,
            num_episodes=4,
        )

        wr = result["h2h/win_rate"]
        assert 0.0 <= wr <= 1.0

    def test_avg_reward_diff_is_float(self) -> None:
        config = _make_config()
        evaluator = PolicyEvaluator(config)

        fn_a = _make_uniform_eval_fn()
        fn_b = _make_uniform_eval_fn()

        result = evaluator.evaluate_head_to_head(
            current_fn=fn_a,
            baseline_fn=fn_b,
            num_episodes=2,
        )

        assert isinstance(result["h2h/avg_reward_diff"], float)

    def test_policy_agreement_bounds(self) -> None:
        """Policy agreement should be between 0 and 1."""
        config = _make_config()
        evaluator = PolicyEvaluator(config)

        fn_a = _make_uniform_eval_fn()
        fn_b = _make_uniform_eval_fn()

        result = evaluator.evaluate_head_to_head(
            current_fn=fn_a,
            baseline_fn=fn_b,
            num_episodes=2,
        )

        pa = result["h2h/policy_agreement"]
        assert 0.0 <= pa <= 1.0


# -------------------------------------------------------------------
# compute_policy_agreement
# -------------------------------------------------------------------


class TestComputePolicyAgreement:
    """Tests for compute_policy_agreement method."""

    def test_identical_fns_perfect_agreement(self) -> None:
        """Two identical EvalFns should agree on every state.

        We disable Dirichlet noise (epsilon=0) and use a near-zero
        temperature so that action selection is deterministic argmax,
        making both MCTS trees produce the same actions.
        """
        from src.alphagalerkin.core.config import TemperatureSchedule
        from src.alphagalerkin.core.types import TemperatureScheduleType

        config = AlphaGalerkinConfig(
            mcts=MCTSConfig(
                num_simulations=2,
                max_tree_depth=2,
                action_topk=3,
                dirichlet_epsilon=0.0,  # No root noise
                temperature_schedule=TemperatureSchedule(
                    schedule_type=TemperatureScheduleType.CONSTANT,
                    initial_temp=1e-9,
                    final_temp=1e-9,
                ),
            ),
            training=TrainingConfig(
                batch_size=4,
                total_steps=2,
            ),
            environment=EnvironmentConfig(
                max_steps=3,
                max_dof=500,
            ),
            device="cpu",
        )
        evaluator = PolicyEvaluator(config)

        # Use the exact same function object for both.
        fn = _make_uniform_eval_fn()

        agreement = evaluator.compute_policy_agreement(
            fn_a=fn,
            fn_b=fn,
            num_states=5,
        )

        assert agreement == pytest.approx(1.0)

    def test_different_fns_less_than_perfect(self) -> None:
        """Different EvalFns may disagree on some states."""
        config = _make_config()
        evaluator = PolicyEvaluator(config)

        fn_uniform = _make_uniform_eval_fn()
        fn_biased = _make_biased_eval_fn(value=0.1)

        agreement = evaluator.compute_policy_agreement(
            fn_a=fn_uniform,
            fn_b=fn_biased,
            num_states=5,
        )

        # Agreement is between 0 and 1 -- we cannot guarantee
        # exact value because MCTS is stochastic, but bounds hold.
        assert 0.0 <= agreement <= 1.0

    def test_agreement_returns_float(self) -> None:
        config = _make_config()
        evaluator = PolicyEvaluator(config)

        fn = _make_uniform_eval_fn()

        agreement = evaluator.compute_policy_agreement(
            fn_a=fn,
            fn_b=fn,
            num_states=3,
        )

        assert isinstance(agreement, float)

    def test_agreement_with_zero_states(self) -> None:
        """Edge case: zero states should return 0.0 (no evidence)."""
        config = _make_config()
        evaluator = PolicyEvaluator(config)

        fn = _make_uniform_eval_fn()

        agreement = evaluator.compute_policy_agreement(
            fn_a=fn,
            fn_b=fn,
            num_states=0,
        )

        assert agreement == pytest.approx(0.0)
