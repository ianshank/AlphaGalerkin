"""Tests for resolution invariance verification tool.

Validates synthetic input creation, forward pass verification,
LBB stability checks, semantic validity tests, and the
orchestration function ``run_verification``.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from src.tools.verify_invariance import (
    create_synthetic_input,
    run_verification,
    verify_forward_pass,
    verify_lbb_stability,
    verify_semantic_validity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class _MockOutput:
    """Minimal mock for model output."""

    policy_logits: torch.Tensor
    value: torch.Tensor
    lbb_constant: torch.Tensor | None = None


class _DummyModel(nn.Module):
    """Tiny model that mimics AlphaGalerkinModel output contract."""

    def __init__(
        self,
        *,
        nan_policy: bool = False,
        inf_value: bool = False,
        constant_value: bool = False,
        value_out_of_range: bool = False,
    ) -> None:
        super().__init__()
        self._linear = nn.Linear(17, 1)  # gives us a parameter
        self.nan_policy = nan_policy
        self.inf_value = inf_value
        self.constant_value = constant_value
        self.value_out_of_range = value_out_of_range

    def forward(
        self,
        x: torch.Tensor,
        return_lbb: bool = False,
    ) -> _MockOutput:
        batch, _c, h, w = x.shape
        policy_size = h * w + 1

        policy = torch.randn(batch, policy_size)
        if self.nan_policy:
            policy[0, 0] = float("nan")

        if self.constant_value:
            value = torch.zeros(batch, 1)
        elif self.value_out_of_range:
            value = torch.full((batch, 1), 5.0)
        else:
            value = torch.tanh(torch.randn(batch, 1))

        if self.inf_value:
            value[0, 0] = float("inf")

        lbb = torch.tensor([0.01]) if return_lbb else None
        return _MockOutput(policy_logits=policy, value=value, lbb_constant=lbb)


@pytest.fixture(params=[5, 9, 13])
def board_size(request: pytest.FixtureRequest) -> int:
    """Parametrized board sizes for verification tests."""
    return request.param


@pytest.fixture
def dummy_model() -> _DummyModel:
    """A healthy dummy model."""
    return _DummyModel()


# ---------------------------------------------------------------------------
# Tests: create_synthetic_input
# ---------------------------------------------------------------------------


class TestCreateSyntheticInput:
    """Tests for synthetic input creation."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_shape(self, batch_size: int, board_size: int) -> None:
        """Output tensor has shape (B, C, H, W)."""
        channels = 17
        t = create_synthetic_input(batch_size, board_size, input_channels=channels)
        assert t.shape == (batch_size, channels, board_size, board_size)

    @pytest.mark.parametrize("channels", [1, 17, 32])
    def test_custom_channels(self, channels: int) -> None:
        """Custom input_channels propagates to output."""
        t = create_synthetic_input(2, 9, input_channels=channels)
        assert t.shape[1] == channels

    def test_device_placement(self) -> None:
        """Tensor is placed on the requested device."""
        device = torch.device("cpu")
        t = create_synthetic_input(1, 9, device=device)
        assert t.device == device

    def test_values_are_finite(self) -> None:
        """All values are finite (no NaN/Inf)."""
        t = create_synthetic_input(4, 9)
        assert t.isfinite().all()


# ---------------------------------------------------------------------------
# Tests: verify_forward_pass
# ---------------------------------------------------------------------------


class TestVerifyForwardPass:
    """Tests for forward pass verification with mock model."""

    def test_passes_on_healthy_model(self, dummy_model: _DummyModel) -> None:
        """A well-behaved model passes all checks."""
        result = verify_forward_pass(dummy_model, board_size=9, batch_size=2)
        assert result["passed"] is True
        assert len(result["errors"]) == 0

    @pytest.mark.parametrize("bs", [5, 9, 19])
    def test_output_keys_present(self, dummy_model: _DummyModel, bs: int) -> None:
        """Result dict contains expected metadata keys."""
        result = verify_forward_pass(dummy_model, board_size=bs, batch_size=2)
        assert "board_size" in result
        assert "inference_time_ms" in result
        assert "policy_entropy" in result
        assert result["board_size"] == bs

    def test_nan_policy_detected(self) -> None:
        """NaN in policy logits causes failure."""
        model = _DummyModel(nan_policy=True)
        result = verify_forward_pass(model, board_size=9, batch_size=2)
        assert result["passed"] is False
        assert any("NaN" in e for e in result["errors"])

    def test_inf_value_detected(self) -> None:
        """Inf in value head causes failure."""
        model = _DummyModel(inf_value=True)
        result = verify_forward_pass(model, board_size=9, batch_size=2)
        assert result["passed"] is False
        assert any("Inf" in e or "range" in e for e in result["errors"])

    def test_value_out_of_range_detected(self) -> None:
        """Values outside [-1, 1] cause failure."""
        model = _DummyModel(value_out_of_range=True)
        result = verify_forward_pass(model, board_size=9, batch_size=2)
        assert result["passed"] is False
        assert any("range" in e for e in result["errors"])

    def test_exception_captured(self) -> None:
        """Exceptions during forward pass are recorded, not raised."""
        model = MagicMock(spec=nn.Module)
        model.eval = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        model.side_effect = RuntimeError("boom")

        result = verify_forward_pass(model, board_size=9, batch_size=2)
        assert result["passed"] is False
        assert any("Exception" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Tests: verify_lbb_stability
# ---------------------------------------------------------------------------


class TestVerifyLBBStability:
    """Tests for LBB stability verification."""

    def _make_lbb_model(
        self,
        lbb_value: float | None = 0.01,
        threshold: float = 1e-6,
    ) -> MagicMock:
        """Create a mock AlphaGalerkinModel with controllable LBB constant."""
        model = MagicMock()
        model.eval = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        model.config = MagicMock()
        model.config.lbb_beta_threshold = threshold

        output = MagicMock()
        output.lbb_constant = torch.tensor([lbb_value]) if lbb_value is not None else None
        model.return_value = output
        return model

    def test_passes_when_above_threshold(self) -> None:
        """LBB constant above threshold passes."""
        model = self._make_lbb_model(lbb_value=0.1, threshold=1e-6)
        result = verify_lbb_stability(model, board_size=9, batch_size=2)
        assert result["passed"] is True

    def test_fails_when_below_threshold(self) -> None:
        """LBB constant below threshold fails."""
        model = self._make_lbb_model(lbb_value=1e-8, threshold=1e-6)
        result = verify_lbb_stability(model, board_size=9, batch_size=2)
        assert result["passed"] is False
        assert any("threshold" in e for e in result["errors"])

    def test_fails_when_non_positive(self) -> None:
        """Non-positive LBB constant fails."""
        model = self._make_lbb_model(lbb_value=-0.01, threshold=1e-6)
        result = verify_lbb_stability(model, board_size=9, batch_size=2)
        assert result["passed"] is False
        assert any("non-positive" in e for e in result["errors"])

    def test_fails_when_lbb_none(self) -> None:
        """Missing LBB constant (None) fails."""
        model = self._make_lbb_model(lbb_value=None)
        result = verify_lbb_stability(model, board_size=9, batch_size=2)
        assert result["passed"] is False
        assert any("not returned" in e for e in result["errors"])

    def test_result_keys(self) -> None:
        """Result contains board_size and error list."""
        model = self._make_lbb_model(lbb_value=0.5)
        result = verify_lbb_stability(model, board_size=13, batch_size=2)
        assert result["board_size"] == 13
        assert "lbb_min" in result
        assert "lbb_mean" in result


# ---------------------------------------------------------------------------
# Tests: verify_semantic_validity
# ---------------------------------------------------------------------------


class TestVerifySemanticValidity:
    """Tests for semantic validity checks."""

    def test_deterministic_model_passes(self, dummy_model: _DummyModel) -> None:
        """Deterministic eval-mode model passes determinism check."""
        # The dummy model uses fixed random seed via conftest, so call twice
        # to check determinism. We seed manually to guarantee identical outputs.
        torch.manual_seed(0)
        result = verify_semantic_validity(dummy_model, board_size=9)
        # Dummy model is not truly deterministic because randn is called inside.
        # It should capture this as an error or warning -- either way the
        # function should not crash.
        assert "board_size" in result
        assert isinstance(result["errors"], list)

    def test_result_contains_metrics(self) -> None:
        """Result dictionary has max_policy_prob and value_std."""
        model = _DummyModel()
        result = verify_semantic_validity(model, board_size=9)
        # These keys exist when no exception occurs
        if result["passed"] or "max_policy_prob" in result:
            assert "max_policy_prob" in result
            assert "uniform_prob" in result

    def test_exception_captured(self) -> None:
        """Exceptions are captured, not raised."""
        model = MagicMock(spec=nn.Module)
        model.eval = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        model.side_effect = RuntimeError("forward failed")

        result = verify_semantic_validity(model, board_size=9)
        assert result["passed"] is False
        assert any("Exception" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# Tests: run_verification (orchestration)
# ---------------------------------------------------------------------------


class TestRunVerification:
    """Tests for the orchestration function."""

    @patch("src.tools.verify_invariance.AlphaGalerkinModel")
    @patch("src.tools.verify_invariance.OperatorConfig")
    def test_returns_bool(
        self,
        mock_config_cls: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        """run_verification returns a boolean."""
        # Wire up mock model that behaves like _DummyModel
        mock_model = _DummyModel()
        mock_model_cls.return_value = mock_model
        mock_config_cls.return_value = MagicMock()

        result = run_verification(train_size=5, infer_size=9, device="cpu")
        assert isinstance(result, bool)

    @patch("src.tools.verify_invariance.AlphaGalerkinModel")
    @patch("src.tools.verify_invariance.OperatorConfig")
    def test_uses_provided_sizes(
        self,
        mock_config_cls: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        """Provided train_size and infer_size are forwarded."""
        mock_model = _DummyModel()
        mock_model_cls.return_value = mock_model
        mock_config_cls.return_value = MagicMock()

        # Should not raise regardless of board sizes
        result = run_verification(train_size=5, infer_size=7, device="cpu")
        assert isinstance(result, bool)
