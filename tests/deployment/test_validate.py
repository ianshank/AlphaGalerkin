"""Tests for model validation utilities.

This module tests the ModelValidator class, ValidationResult dataclass,
and related factory functions, mocking ONNX Runtime sessions and onnx
package operations to ensure deterministic, fast tests.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import structlog
import torch
from torch import nn

from src.deployment.validate import (
    ModelValidator,
    ValidationResult,
    create_validator,
    validate_export,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
DEFAULT_BOARD_SIZE = 9
DEFAULT_CHANNELS = 17
DEFAULT_TOLERANCE = 1e-5
DEFAULT_RELATIVE_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyModel(nn.Module):
    """Minimal model for validation tests producing (policy, value) tuple."""

    def __init__(self, in_channels: int = DEFAULT_CHANNELS) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.conv(x)
        policy = out[:, 0].flatten(1)
        value = out[:, 1].mean(dim=(1, 2), keepdim=False)
        return policy, value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def validator() -> ModelValidator:
    """Provide a default ModelValidator."""
    return ModelValidator(
        tolerance=DEFAULT_TOLERANCE,
        relative_tolerance=DEFAULT_RELATIVE_TOLERANCE,
    )


@pytest.fixture()
def dummy_model() -> nn.Module:
    """Provide a lightweight dummy PyTorch model."""
    torch.manual_seed(SEED)
    return _DummyModel(in_channels=DEFAULT_CHANNELS)


@pytest.fixture()
def dummy_onnx_path(tmp_path: Path) -> Path:
    """Create a dummy ONNX model file path."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"\x00" * 1024)
    return model_path


@pytest.fixture()
def sample_test_inputs() -> list[torch.Tensor]:
    """Provide deterministic test input tensors."""
    torch.manual_seed(SEED)
    return [
        torch.randn(1, DEFAULT_CHANNELS, DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE)
        for _ in range(3)
    ]


# ---------------------------------------------------------------------------
# Tests for create_validator factory
# ---------------------------------------------------------------------------


class TestCreateValidator:
    """Tests for the create_validator factory function."""

    def test_creates_validator_with_defaults(self) -> None:
        """Factory returns ModelValidator with default tolerance."""
        v = create_validator()
        assert isinstance(v, ModelValidator)
        assert v.tolerance == DEFAULT_TOLERANCE

    @pytest.mark.parametrize("tolerance", [1e-3, 1e-5, 1e-7])
    def test_creates_validator_with_tolerance(self, tolerance: float) -> None:
        """Factory correctly sets absolute tolerance."""
        v = create_validator(tolerance=tolerance)
        assert v.tolerance == tolerance

    def test_creates_validator_with_relative_tolerance(self) -> None:
        """Factory forwards relative_tolerance kwarg."""
        v = create_validator(tolerance=1e-4, relative_tolerance=1e-3)
        assert v.relative_tolerance == 1e-3

    def test_default_relative_tolerance(self) -> None:
        """Validator uses 1e-4 relative tolerance by default."""
        v = ModelValidator()
        assert v.relative_tolerance == DEFAULT_RELATIVE_TOLERANCE


# ---------------------------------------------------------------------------
# Tests for generate_test_inputs
# ---------------------------------------------------------------------------


class TestGenerateTestInputs:
    """Tests for ModelValidator.generate_test_inputs."""

    @pytest.mark.parametrize("n_samples", [1, 5, 10])
    def test_produces_correct_number_of_inputs(
        self, validator: ModelValidator, n_samples: int
    ) -> None:
        """generate_test_inputs returns the requested number of tensors."""
        import random

        random.seed(SEED)
        torch.manual_seed(SEED)

        inputs = validator.generate_test_inputs(
            n_samples=n_samples,
            board_sizes=[DEFAULT_BOARD_SIZE],
            input_channels=DEFAULT_CHANNELS,
        )
        assert len(inputs) == n_samples

    def test_inputs_are_tensors_with_correct_ndim(
        self, validator: ModelValidator
    ) -> None:
        """Each generated input is a 4D tensor (B, C, H, W)."""
        import random

        random.seed(SEED)
        torch.manual_seed(SEED)

        inputs = validator.generate_test_inputs(
            n_samples=3,
            board_sizes=[DEFAULT_BOARD_SIZE],
            input_channels=DEFAULT_CHANNELS,
        )
        for inp in inputs:
            assert isinstance(inp, torch.Tensor)
            assert inp.ndim == 4
            assert inp.shape[0] == 1
            assert inp.shape[1] == DEFAULT_CHANNELS

    def test_produces_varying_board_sizes(self, validator: ModelValidator) -> None:
        """When multiple board sizes given, inputs can have different spatial dims."""
        import random

        random.seed(SEED)
        torch.manual_seed(SEED)

        board_sizes = [9, 13, 19]
        inputs = validator.generate_test_inputs(
            n_samples=30,
            board_sizes=board_sizes,
            input_channels=DEFAULT_CHANNELS,
        )
        observed_sizes = {inp.shape[2] for inp in inputs}
        # With 30 samples and seed=42, expect at least 2 different sizes
        assert len(observed_sizes) >= 2
        assert observed_sizes.issubset(set(board_sizes))

    @pytest.mark.parametrize("board_size", [9, 13, 19])
    def test_spatial_dims_match_board_size(
        self, validator: ModelValidator, board_size: int
    ) -> None:
        """Spatial dimensions (H, W) equal the chosen board size."""
        import random

        random.seed(SEED)
        torch.manual_seed(SEED)

        inputs = validator.generate_test_inputs(
            n_samples=3,
            board_sizes=[board_size],
            input_channels=DEFAULT_CHANNELS,
        )
        for inp in inputs:
            assert inp.shape[2] == board_size
            assert inp.shape[3] == board_size


# ---------------------------------------------------------------------------
# Tests for ValidationResult dataclass
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """ValidationResult can be created with all required fields."""
        result = ValidationResult(
            passed=True,
            max_policy_diff=1e-6,
            max_value_diff=1e-7,
            mean_policy_diff=5e-7,
            mean_value_diff=3e-7,
            pytorch_time_ms=10.0,
            onnx_time_ms=5.0,
            speedup_ratio=2.0,
            n_samples_tested=10,
            failed_samples=0,
        )
        assert result.passed is True
        assert result.failed_samples == 0
        assert result.error_message is None

    def test_creation_with_error_message(self) -> None:
        """ValidationResult supports an optional error_message."""
        result = ValidationResult(
            passed=False,
            max_policy_diff=float("inf"),
            max_value_diff=float("inf"),
            mean_policy_diff=float("inf"),
            mean_value_diff=float("inf"),
            pytorch_time_ms=0.0,
            onnx_time_ms=0.0,
            speedup_ratio=0.0,
            n_samples_tested=1,
            failed_samples=1,
            error_message="Runtime error during inference",
        )
        assert result.passed is False
        assert result.error_message is not None

    def test_is_dataclass_and_serializable(self) -> None:
        """ValidationResult is a proper dataclass with asdict support."""
        result = ValidationResult(
            passed=True,
            max_policy_diff=0.0,
            max_value_diff=0.0,
            mean_policy_diff=0.0,
            mean_value_diff=0.0,
            pytorch_time_ms=1.0,
            onnx_time_ms=0.5,
            speedup_ratio=2.0,
            n_samples_tested=5,
            failed_samples=0,
        )
        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        expected_keys = {
            "passed",
            "max_policy_diff",
            "max_value_diff",
            "mean_policy_diff",
            "mean_value_diff",
            "pytorch_time_ms",
            "onnx_time_ms",
            "speedup_ratio",
            "n_samples_tested",
            "failed_samples",
            "error_message",
        }
        assert set(result_dict.keys()) == expected_keys

    @pytest.mark.parametrize(
        "passed,failed",
        [(True, 0), (False, 1), (False, 5)],
    )
    def test_passed_and_failed_consistency(self, passed: bool, failed: int) -> None:
        """Passed flag and failed_samples should be semantically consistent."""
        result = ValidationResult(
            passed=passed,
            max_policy_diff=0.0 if passed else 1.0,
            max_value_diff=0.0 if passed else 1.0,
            mean_policy_diff=0.0,
            mean_value_diff=0.0,
            pytorch_time_ms=1.0,
            onnx_time_ms=1.0,
            speedup_ratio=1.0,
            n_samples_tested=10,
            failed_samples=failed,
        )
        if passed:
            assert result.failed_samples == 0
        else:
            assert result.failed_samples > 0


# ---------------------------------------------------------------------------
# Tests for validate with mocked ONNX Runtime session
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for ModelValidator.validate with mocked ONNX Runtime."""

    def _make_mock_runtime(
        self,
        policy_output: np.ndarray,
        value_output: np.ndarray,
        inference_time_ms: float = 1.0,
    ) -> MagicMock:
        """Create a mock ONNXRuntime that returns deterministic results."""
        mock_runtime = MagicMock()
        mock_result = MagicMock()
        mock_result.policy = policy_output
        mock_result.value = value_output
        mock_result.inference_time_ms = inference_time_ms
        mock_runtime.run.return_value = mock_result
        return mock_runtime

    def test_validate_passes_when_outputs_match(
        self,
        validator: ModelValidator,
        dummy_model: nn.Module,
        dummy_onnx_path: Path,
        sample_test_inputs: list[torch.Tensor],
    ) -> None:
        """validate() returns passed=True when ONNX output matches PyTorch."""
        dummy_model.eval()

        # Get actual PyTorch outputs to feed into mock runtime
        with torch.no_grad():
            pt_policy, pt_value = dummy_model(sample_test_inputs[0])

        mock_runtime = self._make_mock_runtime(
            policy_output=pt_policy.numpy(),
            value_output=pt_value.numpy(),
        )

        with patch("src.deployment.runtime.ONNXRuntime", return_value=mock_runtime):
            result = validator.validate(
                dummy_model,
                dummy_onnx_path,
                sample_test_inputs[:1],
            )

        assert result.passed is True
        assert result.failed_samples == 0
        assert result.n_samples_tested == 1

    def test_validate_fails_when_outputs_differ(
        self,
        validator: ModelValidator,
        dummy_model: nn.Module,
        dummy_onnx_path: Path,
        sample_test_inputs: list[torch.Tensor],
    ) -> None:
        """validate() returns passed=False when outputs exceed tolerance."""
        dummy_model.eval()

        # Return intentionally wrong outputs
        wrong_policy = np.ones((1, DEFAULT_BOARD_SIZE * DEFAULT_BOARD_SIZE), dtype=np.float32)
        wrong_value = np.array([999.0], dtype=np.float32)

        mock_runtime = self._make_mock_runtime(
            policy_output=wrong_policy,
            value_output=wrong_value,
        )

        with patch("src.deployment.runtime.ONNXRuntime", return_value=mock_runtime):
            result = validator.validate(
                dummy_model,
                dummy_onnx_path,
                sample_test_inputs[:1],
            )

        assert result.passed is False
        assert result.failed_samples > 0

    def test_validate_multiple_samples(
        self,
        validator: ModelValidator,
        dummy_model: nn.Module,
        dummy_onnx_path: Path,
        sample_test_inputs: list[torch.Tensor],
    ) -> None:
        """validate() tests all provided input samples."""
        dummy_model.eval()

        # Return matching outputs for each sample
        with torch.no_grad():
            pt_policy, pt_value = dummy_model(sample_test_inputs[0])

        mock_runtime = self._make_mock_runtime(
            policy_output=pt_policy.numpy(),
            value_output=pt_value.numpy(),
        )

        with patch("src.deployment.runtime.ONNXRuntime", return_value=mock_runtime):
            result = validator.validate(
                dummy_model,
                dummy_onnx_path,
                sample_test_inputs,
            )

        assert result.n_samples_tested == len(sample_test_inputs)

    def test_validate_reports_speedup_ratio(
        self,
        validator: ModelValidator,
        dummy_model: nn.Module,
        dummy_onnx_path: Path,
        sample_test_inputs: list[torch.Tensor],
    ) -> None:
        """validate() computes speedup ratio between PyTorch and ONNX times."""
        dummy_model.eval()

        with torch.no_grad():
            pt_policy, pt_value = dummy_model(sample_test_inputs[0])

        mock_runtime = self._make_mock_runtime(
            policy_output=pt_policy.numpy(),
            value_output=pt_value.numpy(),
            inference_time_ms=2.0,
        )

        with patch("src.deployment.runtime.ONNXRuntime", return_value=mock_runtime):
            result = validator.validate(
                dummy_model,
                dummy_onnx_path,
                sample_test_inputs[:1],
            )

        assert result.onnx_time_ms > 0
        assert result.pytorch_time_ms >= 0
        # speedup_ratio = pytorch_time / onnx_time
        assert result.speedup_ratio >= 0

    def test_validate_handles_single_tensor_input(
        self,
        validator: ModelValidator,
        dummy_model: nn.Module,
        dummy_onnx_path: Path,
    ) -> None:
        """validate() accepts a single Tensor (not just list)."""
        dummy_model.eval()
        torch.manual_seed(SEED)
        single_input = torch.randn(1, DEFAULT_CHANNELS, DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE)

        with torch.no_grad():
            pt_policy, pt_value = dummy_model(single_input)

        mock_runtime = self._make_mock_runtime(
            policy_output=pt_policy.numpy(),
            value_output=pt_value.numpy(),
        )

        with patch("src.deployment.runtime.ONNXRuntime", return_value=mock_runtime):
            result = validator.validate(
                dummy_model,
                dummy_onnx_path,
                single_input,
            )

        assert result.n_samples_tested == 1

    def test_validate_handles_runtime_error_gracefully(
        self,
        validator: ModelValidator,
        dummy_model: nn.Module,
        dummy_onnx_path: Path,
        sample_test_inputs: list[torch.Tensor],
    ) -> None:
        """validate() catches exceptions during inference and marks sample as failed."""
        dummy_model.eval()

        mock_runtime = MagicMock()
        mock_runtime.run.side_effect = RuntimeError("ONNX inference failed")

        with patch("src.deployment.runtime.ONNXRuntime", return_value=mock_runtime):
            result = validator.validate(
                dummy_model,
                dummy_onnx_path,
                sample_test_inputs[:1],
            )

        assert result.passed is False
        assert result.failed_samples == 1


# ---------------------------------------------------------------------------
# Tests for validate_shapes
# ---------------------------------------------------------------------------


class TestValidateShapes:
    """Tests for ModelValidator.validate_shapes."""

    def _make_mock_onnx_model(
        self,
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
    ) -> MagicMock:
        """Create a mock onnx model with specified input/output shapes.

        Args:
            inputs: List of dicts with 'name' and 'dims' keys.
                Each dim is either an int (fixed) or a str (dynamic).
            outputs: Same format as inputs.

        """
        mock_model = MagicMock()
        mock_graph = MagicMock()

        mock_inputs = []
        for inp_spec in inputs:
            mock_inp = MagicMock()
            mock_inp.name = inp_spec["name"]
            dims = []
            for d in inp_spec["dims"]:
                mock_dim = MagicMock()
                if isinstance(d, str):
                    mock_dim.dim_param = d
                    mock_dim.dim_value = 0
                else:
                    mock_dim.dim_param = ""
                    mock_dim.dim_value = d
                dims.append(mock_dim)
            mock_inp.type.tensor_type.shape.dim = dims
            mock_inputs.append(mock_inp)

        mock_outputs = []
        for out_spec in outputs:
            mock_out = MagicMock()
            mock_out.name = out_spec["name"]
            dims = []
            for d in out_spec["dims"]:
                mock_dim = MagicMock()
                if isinstance(d, str):
                    mock_dim.dim_param = d
                    mock_dim.dim_value = 0
                else:
                    mock_dim.dim_param = ""
                    mock_dim.dim_value = d
                dims.append(mock_dim)
            mock_out.type.tensor_type.shape.dim = dims
            mock_outputs.append(mock_out)

        mock_graph.input = mock_inputs
        mock_graph.output = mock_outputs
        mock_model.graph = mock_graph
        return mock_model

    def test_validate_shapes_all_match(
        self, validator: ModelValidator, dummy_onnx_path: Path
    ) -> None:
        """validate_shapes returns True for all shapes when they match."""
        mock_model = self._make_mock_onnx_model(
            inputs=[{"name": "board_state", "dims": ["batch", 17, "height", "width"]}],
            outputs=[
                {"name": "policy", "dims": ["batch", 361]},
                {"name": "value", "dims": ["batch", 1]},
            ],
        )

        mock_onnx = MagicMock()
        mock_onnx.load.return_value = mock_model

        with patch.dict("sys.modules", {"onnx": mock_onnx}):
            results = validator.validate_shapes(
                dummy_onnx_path,
                expected_inputs={"board_state": (-1, 17, -1, -1)},
                expected_outputs={
                    "policy": (-1, 361),
                    "value": (-1, 1),
                },
            )

        assert results["input_board_state"] is True
        assert results["output_policy"] is True
        assert results["output_value"] is True

    def test_validate_shapes_mismatch_detected(
        self, validator: ModelValidator, dummy_onnx_path: Path
    ) -> None:
        """validate_shapes returns False when fixed dims don't match."""
        mock_model = self._make_mock_onnx_model(
            inputs=[{"name": "board_state", "dims": ["batch", 17, 9, 9]}],
            outputs=[{"name": "policy", "dims": ["batch", 81]}],
        )

        mock_onnx = MagicMock()
        mock_onnx.load.return_value = mock_model

        with patch.dict("sys.modules", {"onnx": mock_onnx}):
            results = validator.validate_shapes(
                dummy_onnx_path,
                expected_inputs={"board_state": (-1, 17, 19, 19)},
                expected_outputs={"policy": (-1, 361)},
            )

        # Input spatial dims 9 != 19
        assert results["input_board_state"] is False
        # Output policy dim 81 != 361
        assert results["output_policy"] is False

    def test_validate_shapes_dynamic_dims_accepted(
        self, validator: ModelValidator, dummy_onnx_path: Path
    ) -> None:
        """Dynamic dims (-1) match any expected value."""
        mock_model = self._make_mock_onnx_model(
            inputs=[{"name": "board_state", "dims": ["batch", 17, "height", "width"]}],
            outputs=[],
        )

        mock_onnx = MagicMock()
        mock_onnx.load.return_value = mock_model

        with patch.dict("sys.modules", {"onnx": mock_onnx}):
            results = validator.validate_shapes(
                dummy_onnx_path,
                expected_inputs={"board_state": (-1, 17, 19, 19)},
                expected_outputs={},
            )

        # Dynamic dims should match anything
        assert results["input_board_state"] is True

    def test_validate_shapes_onnx_unavailable(
        self, validator: ModelValidator, dummy_onnx_path: Path
    ) -> None:
        """validate_shapes returns error dict when onnx is not installed."""
        with patch.dict("sys.modules", {"onnx": None}):
            results = validator.validate_shapes(
                dummy_onnx_path,
                expected_inputs={},
                expected_outputs={},
            )

        assert "error" in results

    def test_validate_shapes_unknown_inputs_pass(
        self, validator: ModelValidator, dummy_onnx_path: Path
    ) -> None:
        """Inputs not listed in expected_inputs default to True."""
        mock_model = self._make_mock_onnx_model(
            inputs=[{"name": "extra_input", "dims": [1, 3, 9, 9]}],
            outputs=[],
        )

        mock_onnx = MagicMock()
        mock_onnx.load.return_value = mock_model

        with patch.dict("sys.modules", {"onnx": mock_onnx}):
            results = validator.validate_shapes(
                dummy_onnx_path,
                expected_inputs={},
                expected_outputs={},
            )

        assert results["input_extra_input"] is True


# ---------------------------------------------------------------------------
# Tests for compare_quantized
# ---------------------------------------------------------------------------


class TestCompareQuantized:
    """Tests for ModelValidator.compare_quantized with mocked runtimes."""

    def test_compare_returns_expected_keys(
        self,
        validator: ModelValidator,
        tmp_path: Path,
    ) -> None:
        """compare_quantized returns dict with all expected metric keys."""
        original_path = tmp_path / "original.onnx"
        quantized_path = tmp_path / "quantized.onnx"
        original_path.write_bytes(b"\x00" * 1024)
        quantized_path.write_bytes(b"\x00" * 512)

        rng = np.random.default_rng(SEED)
        shape = (1, DEFAULT_CHANNELS, DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE)
        test_inputs = [
            rng.standard_normal(shape).astype(np.float32)
            for _ in range(3)
        ]

        # Mock both runtimes to return similar outputs
        board_sq = DEFAULT_BOARD_SIZE * DEFAULT_BOARD_SIZE
        policy = rng.standard_normal((1, board_sq)).astype(np.float32)
        value = np.array([0.5], dtype=np.float32)

        def make_mock_runtime() -> MagicMock:
            mock_rt = MagicMock()
            mock_result = MagicMock()
            mock_result.policy = policy
            mock_result.value = value
            mock_result.inference_time_ms = 1.0
            mock_rt.run.return_value = mock_result
            return mock_rt

        mock_original = make_mock_runtime()
        mock_quantized = make_mock_runtime()

        with patch(
            "src.deployment.runtime.ONNXRuntime",
            side_effect=[mock_original, mock_quantized],
        ):
            results = validator.compare_quantized(
                original_path,
                quantized_path,
                test_inputs,
            )

        expected_keys = {
            "max_policy_diff",
            "mean_policy_diff",
            "max_value_diff",
            "mean_value_diff",
            "original_time_ms",
            "quantized_time_ms",
            "speedup_ratio",
            "n_samples",
        }
        assert set(results.keys()) == expected_keys
        assert results["n_samples"] == 3

    def test_compare_detects_quantization_drift(
        self,
        validator: ModelValidator,
        tmp_path: Path,
    ) -> None:
        """compare_quantized reports nonzero diff when outputs differ."""
        original_path = tmp_path / "original.onnx"
        quantized_path = tmp_path / "quantized.onnx"
        original_path.write_bytes(b"\x00" * 1024)
        quantized_path.write_bytes(b"\x00" * 512)

        rng = np.random.default_rng(SEED)
        shape = (1, DEFAULT_CHANNELS, DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE)
        test_inputs = [
            rng.standard_normal(shape).astype(np.float32)
        ]

        original_policy = np.zeros((1, DEFAULT_BOARD_SIZE * DEFAULT_BOARD_SIZE), dtype=np.float32)
        quantized_policy = np.ones((1, DEFAULT_BOARD_SIZE * DEFAULT_BOARD_SIZE), dtype=np.float32)

        mock_original = MagicMock()
        mock_original_result = MagicMock()
        mock_original_result.policy = original_policy
        mock_original_result.value = np.array([0.0], dtype=np.float32)
        mock_original_result.inference_time_ms = 2.0
        mock_original.run.return_value = mock_original_result

        mock_quantized = MagicMock()
        mock_quantized_result = MagicMock()
        mock_quantized_result.policy = quantized_policy
        mock_quantized_result.value = np.array([0.0], dtype=np.float32)
        mock_quantized_result.inference_time_ms = 1.0
        mock_quantized.run.return_value = mock_quantized_result

        with patch(
            "src.deployment.runtime.ONNXRuntime",
            side_effect=[mock_original, mock_quantized],
        ):
            results = validator.compare_quantized(
                original_path,
                quantized_path,
                test_inputs,
            )

        assert results["max_policy_diff"] == pytest.approx(1.0)
        assert results["speedup_ratio"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Tests for validate_export convenience function
# ---------------------------------------------------------------------------


class TestValidateExportConvenience:
    """Tests for the validate_export convenience function."""

    def test_validate_export_uses_default_board_sizes(
        self,
        dummy_model: nn.Module,
        dummy_onnx_path: Path,
    ) -> None:
        """validate_export uses [9, 13, 19] as default board sizes."""
        dummy_model.eval()

        with torch.no_grad():
            pt_policy, pt_value = dummy_model(
                torch.randn(1, DEFAULT_CHANNELS, 9, 9)
            )

        mock_runtime = MagicMock()
        mock_result = MagicMock()
        mock_result.policy = pt_policy.numpy()
        mock_result.value = pt_value.numpy()
        mock_result.inference_time_ms = 1.0
        mock_runtime.run.return_value = mock_result

        with patch("src.deployment.runtime.ONNXRuntime", return_value=mock_runtime):
            result = validate_export(
                dummy_model,
                dummy_onnx_path,
                n_samples=3,
            )

        assert isinstance(result, ValidationResult)
        assert result.n_samples_tested == 3

    def test_validate_export_custom_tolerance(
        self,
        dummy_model: nn.Module,
        dummy_onnx_path: Path,
    ) -> None:
        """validate_export respects custom tolerance parameter."""
        dummy_model.eval()

        # Return slightly different outputs
        mock_runtime = MagicMock()
        mock_result = MagicMock()
        mock_result.policy = np.zeros((1, 81), dtype=np.float32)
        mock_result.value = np.array([0.0], dtype=np.float32)
        mock_result.inference_time_ms = 1.0
        mock_runtime.run.return_value = mock_result

        # With a very large tolerance, differences should be acceptable
        with patch("src.deployment.runtime.ONNXRuntime", return_value=mock_runtime):
            result = validate_export(
                dummy_model,
                dummy_onnx_path,
                n_samples=1,
                board_sizes=[9],
                tolerance=1e6,  # Very large tolerance
            )

        assert result.passed is True
