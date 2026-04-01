"""Tests for resolution invariance verification tool.

Tests cover the core verification functions using mocked models,
avoiding the need for actual trained model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from src.tools.verify_invariance import (
    create_synthetic_input,
    verify_forward_pass,
    verify_lbb_stability,
    verify_semantic_validity,
)

# --- Helper: Mock model output ---


@dataclass
class MockModelOutput:
    """Simulates AlphaGalerkinModel output."""

    policy_logits: torch.Tensor
    value: torch.Tensor
    lbb_constant: torch.Tensor | None = None


def make_mock_model(
    board_size: int = 9,
    batch_size: int = 4,
    deterministic: bool = True,
    nan_policy: bool = False,
    nan_value: bool = False,
    inf_policy: bool = False,
    inf_value: bool = False,
    bad_value_range: bool = False,
    bad_policy_shape: bool = False,
    bad_value_shape: bool = False,
    uniform_policy: bool = False,
    constant_value: bool = False,
    lbb_constant: float | None = None,
    raise_exception: bool = False,
) -> MagicMock:
    """Create a mock model that returns controlled outputs."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)

    # Create a parameter to satisfy next(model.parameters()).device
    param = torch.zeros(1)
    model.parameters = MagicMock(return_value=iter([param]))

    n_actions = board_size**2 + 1

    # Cache for deterministic outputs keyed by input shape
    _output_cache: dict[tuple, MockModelOutput] = {}

    def forward(x, return_lbb=False):
        if raise_exception:
            raise RuntimeError("Model forward failed")

        bs = x.shape[0]
        n_act = x.shape[2] ** 2 + 1

        # For deterministic behavior, use cached output for same shape
        cache_key = (bs, n_act, x.shape[2])
        if deterministic and cache_key in _output_cache:
            cached = _output_cache[cache_key]
            output = MockModelOutput(
                policy_logits=cached.policy_logits.clone(),
                value=cached.value.clone(),
                lbb_constant=cached.lbb_constant,
            )
            if return_lbb and lbb_constant is not None:
                output.lbb_constant = torch.tensor([lbb_constant] * bs)
            return output

        if bad_policy_shape:
            policy = torch.randn(bs, 10)  # Wrong shape
        elif nan_policy:
            policy = torch.full((bs, n_act), float("nan"))
        elif inf_policy:
            policy = torch.full((bs, n_act), float("inf"))
        elif uniform_policy:
            policy = torch.zeros(bs, n_act)  # All equal -> uniform
        else:
            # Use a seeded generator for reproducibility
            g = torch.Generator()
            g.manual_seed(42 + bs + n_act)
            policy = torch.randn(bs, n_act, generator=g)

        if bad_value_shape:
            value = torch.zeros(bs, 2)  # Wrong shape
        elif nan_value:
            value = torch.full((bs, 1), float("nan"))
        elif inf_value:
            value = torch.full((bs, 1), float("inf"))
        elif bad_value_range:
            value = torch.full((bs, 1), 5.0)  # Out of [-1, 1]
        elif constant_value:
            value = torch.full((bs, 1), 0.5)
        else:
            g = torch.Generator()
            g.manual_seed(123 + bs)
            value = torch.tanh(torch.randn(bs, 1, generator=g))

        output = MockModelOutput(policy_logits=policy, value=value)

        if return_lbb and lbb_constant is not None:
            output.lbb_constant = torch.tensor([lbb_constant] * bs)

        # Cache for deterministic mode
        if deterministic:
            _output_cache[cache_key] = output

        return output

    model.side_effect = forward
    model.__call__ = forward
    return model


# --- create_synthetic_input Tests ---


class TestCreateSyntheticInput:
    """Tests for create_synthetic_input."""

    def test_default_shape(self) -> None:
        """Returns tensor with correct default shape."""
        x = create_synthetic_input(batch_size=2, board_size=9)
        assert x.shape == (2, 17, 9, 9)

    def test_custom_channels(self) -> None:
        """Supports custom input channels."""
        x = create_synthetic_input(batch_size=1, board_size=5, input_channels=3)
        assert x.shape == (1, 3, 5, 5)

    def test_device_cpu(self) -> None:
        """Creates tensor on CPU."""
        x = create_synthetic_input(batch_size=1, board_size=9, device=torch.device("cpu"))
        assert x.device.type == "cpu"

    @pytest.mark.parametrize("board_size", [5, 9, 13, 19, 25])
    def test_various_board_sizes(self, board_size: int) -> None:
        """Works for various board sizes."""
        x = create_synthetic_input(batch_size=1, board_size=board_size)
        assert x.shape == (1, 17, board_size, board_size)


# --- verify_forward_pass Tests ---


class TestVerifyForwardPass:
    """Tests for verify_forward_pass."""

    def test_pass_on_valid_output(self) -> None:
        """Returns passed=True for valid model output."""
        model = make_mock_model(board_size=9)
        results = verify_forward_pass(model, board_size=9, batch_size=4)
        assert results["passed"] is True
        assert results["board_size"] == 9
        assert results["batch_size"] == 4
        assert len(results["errors"]) == 0

    def test_records_inference_time(self) -> None:
        """Records inference time in results."""
        model = make_mock_model(board_size=9)
        results = verify_forward_pass(model, board_size=9)
        assert "inference_time_ms" in results
        assert results["inference_time_ms"] >= 0

    def test_records_output_statistics(self) -> None:
        """Records policy entropy and value statistics."""
        model = make_mock_model(board_size=9)
        results = verify_forward_pass(model, board_size=9)
        assert "policy_entropy" in results
        assert "value_mean" in results
        assert "value_std" in results

    def test_fail_on_nan_policy(self) -> None:
        """Fails when policy contains NaN."""
        model = make_mock_model(board_size=9, nan_policy=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False
        assert any("NaN" in e for e in results["errors"])

    def test_fail_on_nan_value(self) -> None:
        """Fails when value contains NaN."""
        model = make_mock_model(board_size=9, nan_value=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False
        assert any("NaN" in e for e in results["errors"])

    def test_fail_on_inf_policy(self) -> None:
        """Fails when policy contains Inf."""
        model = make_mock_model(board_size=9, inf_policy=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False
        assert any("Inf" in e for e in results["errors"])

    def test_fail_on_inf_value(self) -> None:
        """Fails when value contains Inf."""
        model = make_mock_model(board_size=9, inf_value=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False

    def test_fail_on_value_out_of_range(self) -> None:
        """Fails when value is outside [-1, 1]."""
        model = make_mock_model(board_size=9, bad_value_range=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False
        assert any("out of range" in e for e in results["errors"])

    def test_exception_handling(self) -> None:
        """Handles model exceptions gracefully."""
        model = make_mock_model(board_size=9, raise_exception=True)
        results = verify_forward_pass(model, board_size=9)
        assert results["passed"] is False
        assert any("Exception" in e for e in results["errors"])


# --- verify_lbb_stability Tests ---


class TestVerifyLBBStability:
    """Tests for verify_lbb_stability."""

    def test_pass_on_positive_lbb(self) -> None:
        """Passes when LBB constant is above threshold."""
        model = make_mock_model(board_size=9, lbb_constant=0.5)
        model.config = MagicMock()
        model.config.lbb_beta_threshold = 0.1
        results = verify_lbb_stability(model, board_size=9)
        assert results["passed"] is True
        assert results["lbb_min"] == 0.5

    def test_fail_below_threshold(self) -> None:
        """Fails when LBB constant is below threshold."""
        model = make_mock_model(board_size=9, lbb_constant=0.01)
        model.config = MagicMock()
        model.config.lbb_beta_threshold = 0.1
        results = verify_lbb_stability(model, board_size=9)
        assert results["passed"] is False
        assert any("below threshold" in e for e in results["errors"])

    def test_fail_on_negative_lbb(self) -> None:
        """Fails when LBB constant is non-positive."""
        model = make_mock_model(board_size=9, lbb_constant=-0.1)
        model.config = MagicMock()
        model.config.lbb_beta_threshold = 0.1
        results = verify_lbb_stability(model, board_size=9)
        assert results["passed"] is False
        assert any("non-positive" in e for e in results["errors"])

    def test_fail_when_lbb_not_returned(self) -> None:
        """Fails when model doesn't return LBB constant."""
        model = make_mock_model(board_size=9, lbb_constant=None)
        model.config = MagicMock()
        model.config.lbb_beta_threshold = 0.1
        results = verify_lbb_stability(model, board_size=9)
        assert results["passed"] is False
        assert any("not returned" in e for e in results["errors"])

    def test_exception_handling(self) -> None:
        """Handles model exceptions gracefully."""
        model = make_mock_model(board_size=9, raise_exception=True)
        model.config = MagicMock()
        model.config.lbb_beta_threshold = 0.1
        results = verify_lbb_stability(model, board_size=9)
        assert results["passed"] is False
        assert any("Exception" in e for e in results["errors"])


# --- verify_semantic_validity Tests ---


class TestVerifySemanticValidity:
    """Tests for verify_semantic_validity."""

    def test_pass_on_valid_model(self) -> None:
        """Passes for model with deterministic varied output."""
        model = make_mock_model(board_size=9)
        results = verify_semantic_validity(model, board_size=9)
        assert results["passed"] is True
        assert "max_policy_prob" in results
        assert "uniform_prob" in results

    def test_exception_handling(self) -> None:
        """Handles model exceptions gracefully."""
        model = make_mock_model(board_size=9, raise_exception=True)
        results = verify_semantic_validity(model, board_size=9)
        assert results["passed"] is False
        assert any("Exception" in e for e in results["errors"])
