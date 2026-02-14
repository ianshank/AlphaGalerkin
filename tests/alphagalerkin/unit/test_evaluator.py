"""Tests for PolicyEvaluator (src/alphagalerkin/evaluation/evaluator.py).

Focuses on evaluate_from_checkpoint and edge cases not covered by
test_head_to_head.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_config(**kwargs: Any) -> AlphaGalerkinConfig:
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
            max_steps=kwargs.get("max_steps", 3),
            max_dof=500,
        ),
        device="cpu",
        **{k: v for k, v in kwargs.items() if k != "max_steps"},
    )


def _make_eval_fn(config: AlphaGalerkinConfig | None = None) -> Any:
    """Create an EvalFn that returns uniform priors and value 0.5."""
    if config is None:
        config = _make_config()
    masker = ActionMasker(config.environment)

    def eval_fn(
        state: DiscretizationState,
    ) -> tuple[dict[Action, float], float]:
        valid = masker.valid_actions(state)
        n = len(valid)
        priors = dict.fromkeys(valid, 1.0 / n)
        return priors, 0.5

    return eval_fn


# -------------------------------------------------------------------
# evaluate (basic / edge cases)
# -------------------------------------------------------------------


class TestEvaluate:
    """Tests for the evaluate method."""

    def test_evaluate_returns_expected_keys(self) -> None:
        config = _make_config()
        evaluator = PolicyEvaluator(config)
        fn = _make_eval_fn(config)

        result = evaluator.evaluate(eval_fn=fn, num_episodes=2)

        expected_keys = {
            "eval/avg_reward",
            "eval/avg_length",
            "eval/avg_final_dof",
            "eval/std_reward",
            "eval/min_reward",
            "eval/max_reward",
        }
        assert set(result.keys()) == expected_keys

    def test_evaluate_single_episode(self) -> None:
        """Single episode should work without error."""
        config = _make_config()
        evaluator = PolicyEvaluator(config)
        fn = _make_eval_fn(config)

        result = evaluator.evaluate(eval_fn=fn, num_episodes=1)

        assert isinstance(result["eval/avg_reward"], float)
        # With 1 episode, std should be 0
        assert result["eval/std_reward"] == pytest.approx(0.0)

    def test_evaluate_values_are_float(self) -> None:
        config = _make_config()
        evaluator = PolicyEvaluator(config)
        fn = _make_eval_fn(config)

        result = evaluator.evaluate(eval_fn=fn, num_episodes=2)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be float but is {type(value)}"

    def test_evaluate_length_within_max_steps(self) -> None:
        max_steps = 5
        config = _make_config(max_steps=max_steps)
        evaluator = PolicyEvaluator(config)
        fn = _make_eval_fn(config)

        result = evaluator.evaluate(eval_fn=fn, num_episodes=3)

        assert result["eval/avg_length"] <= max_steps


# -------------------------------------------------------------------
# evaluate_head_to_head (supplementary edge cases)
# -------------------------------------------------------------------


class TestEvaluateHeadToHeadEdgeCases:
    """Supplementary edge cases for evaluate_head_to_head."""

    def test_single_episode(self) -> None:
        """Single episode should produce valid metrics."""
        config = _make_config()
        evaluator = PolicyEvaluator(config)
        fn = _make_eval_fn(config)

        result = evaluator.evaluate_head_to_head(
            current_fn=fn,
            baseline_fn=fn,
            num_episodes=1,
        )

        assert 0.0 <= result["h2h/win_rate"] <= 1.0
        assert isinstance(result["h2h/avg_reward_diff"], float)
        assert 0.0 <= result["h2h/policy_agreement"] <= 1.0


# -------------------------------------------------------------------
# compute_policy_agreement (supplementary)
# -------------------------------------------------------------------


class TestComputePolicyAgreementEdgeCases:
    """Supplementary edge cases for compute_policy_agreement."""

    def test_single_state(self) -> None:
        """Agreement with one state is either 0 or 1."""
        config = _make_config()
        evaluator = PolicyEvaluator(config)
        fn = _make_eval_fn(config)

        agreement = evaluator.compute_policy_agreement(
            fn_a=fn,
            fn_b=fn,
            num_states=1,
        )

        assert agreement in (0.0, 1.0)


# -------------------------------------------------------------------
# evaluate_from_checkpoint
# -------------------------------------------------------------------


class TestEvaluateFromCheckpoint:
    """Tests for evaluate_from_checkpoint with mocked dependencies."""

    @patch("src.alphagalerkin.evaluation.evaluator.PolicyEvaluator.evaluate")
    @patch("src.alphagalerkin.nn.model.AlphaGalerkinNetwork")
    @patch("torch.load")
    def test_loads_checkpoint_and_evaluates(
        self,
        mock_torch_load: MagicMock,
        mock_network_cls: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """Should load checkpoint, build network, and call evaluate."""
        # Set up mock checkpoint
        mock_torch_load.return_value = {
            "model_state_dict": {"dummy": "weights"},
            "step": 1000,
        }

        # Set up mock network
        mock_network = MagicMock()
        mock_network.predict.return_value = ({}, 0.5)
        mock_network_cls.return_value = mock_network

        # Set up mock evaluate return
        mock_evaluate.return_value = {
            "eval/avg_reward": 1.0,
            "eval/avg_length": 5.0,
            "eval/avg_final_dof": 100.0,
            "eval/std_reward": 0.1,
            "eval/min_reward": 0.5,
            "eval/max_reward": 1.5,
        }

        config = _make_config()
        evaluator = PolicyEvaluator(config)

        result = evaluator.evaluate_from_checkpoint(
            checkpoint_path=Path("/fake/checkpoint.pt"),
            num_episodes=5,
        )

        # torch.load should have been called
        mock_torch_load.assert_called_once()
        # evaluate should have been called with num_episodes=5
        mock_evaluate.assert_called_once()
        call_kwargs = mock_evaluate.call_args
        assert call_kwargs[1]["num_episodes"] == 5
        # Result should match mock return
        assert result["eval/avg_reward"] == 1.0

    @patch("src.alphagalerkin.evaluation.evaluator.PolicyEvaluator.evaluate")
    @patch("src.alphagalerkin.nn.model.AlphaGalerkinNetwork")
    @patch("torch.load")
    def test_network_set_to_eval_mode(
        self,
        mock_torch_load: MagicMock,
        mock_network_cls: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """Network should be put in eval mode before inference."""
        mock_torch_load.return_value = {
            "model_state_dict": {"dummy": "weights"},
            "step": 500,
        }

        mock_network = MagicMock()
        mock_network_cls.return_value = mock_network

        mock_evaluate.return_value = {
            "eval/avg_reward": 0.0,
            "eval/avg_length": 1.0,
            "eval/avg_final_dof": 10.0,
            "eval/std_reward": 0.0,
            "eval/min_reward": 0.0,
            "eval/max_reward": 0.0,
        }

        config = _make_config()
        evaluator = PolicyEvaluator(config)

        evaluator.evaluate_from_checkpoint(
            checkpoint_path=Path("/fake/checkpoint.pt"),
            num_episodes=1,
        )

        mock_network.eval.assert_called_once()
        mock_network.load_state_dict.assert_called_once_with({"dummy": "weights"})

    @patch("src.alphagalerkin.evaluation.evaluator.PolicyEvaluator.evaluate")
    @patch("src.alphagalerkin.nn.model.AlphaGalerkinNetwork")
    @patch("torch.load")
    def test_default_num_episodes(
        self,
        mock_torch_load: MagicMock,
        mock_network_cls: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """Default num_episodes should be 10."""
        mock_torch_load.return_value = {
            "model_state_dict": {},
            "step": 0,
        }
        mock_network = MagicMock()
        mock_network_cls.return_value = mock_network
        mock_evaluate.return_value = {
            "eval/avg_reward": 0.0,
            "eval/avg_length": 1.0,
            "eval/avg_final_dof": 10.0,
            "eval/std_reward": 0.0,
            "eval/min_reward": 0.0,
            "eval/max_reward": 0.0,
        }

        config = _make_config()
        evaluator = PolicyEvaluator(config)

        evaluator.evaluate_from_checkpoint(
            checkpoint_path=Path("/fake/checkpoint.pt"),
        )

        call_kwargs = mock_evaluate.call_args
        assert call_kwargs[1]["num_episodes"] == 10

    @patch("src.alphagalerkin.evaluation.evaluator.PolicyEvaluator.evaluate")
    @patch("src.alphagalerkin.nn.model.AlphaGalerkinNetwork")
    @patch("torch.load")
    def test_checkpoint_without_step_key(
        self,
        mock_torch_load: MagicMock,
        mock_network_cls: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """Checkpoint without 'step' key should still work."""
        mock_torch_load.return_value = {
            "model_state_dict": {},
        }
        mock_network = MagicMock()
        mock_network_cls.return_value = mock_network
        mock_evaluate.return_value = {
            "eval/avg_reward": 0.0,
            "eval/avg_length": 1.0,
            "eval/avg_final_dof": 10.0,
            "eval/std_reward": 0.0,
            "eval/min_reward": 0.0,
            "eval/max_reward": 0.0,
        }

        config = _make_config()
        evaluator = PolicyEvaluator(config)

        # Should not raise
        result = evaluator.evaluate_from_checkpoint(
            checkpoint_path=Path("/fake/checkpoint.pt"),
            num_episodes=2,
        )
        assert isinstance(result, dict)
