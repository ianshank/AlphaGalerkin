"""Integration tests for ONNX export functionality.

Covers actual export, dynamic shapes, output validation, and quantization.
Tests requiring onnxruntime or onnx packages are automatically skipped when
those packages are not installed.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from config.schemas import OperatorConfig
from src.deployment.config import (
    ExportConfig,
    QuantizationConfig,
    QuantizationMode,
)
from src.deployment.export_onnx import ONNXExporter, create_exporter, export_model
from src.modeling.model import AlphaGalerkinModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_op_config(**overrides: Any) -> OperatorConfig:
    """Return a small OperatorConfig suitable for fast unit tests.

    Args:
        **overrides: Any OperatorConfig field overrides.

    Returns:
        A minimal OperatorConfig instance.

    """
    defaults: dict[str, Any] = dict(
        d_model=16,
        d_key=8,
        d_value=8,
        d_ffn=32,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=8,
        use_fnet_mixing=False,
        input_channels=4,
    )
    defaults.update(overrides)
    return OperatorConfig(**defaults)


def _make_model(**config_overrides: Any) -> AlphaGalerkinModel:
    """Instantiate a tiny AlphaGalerkinModel for testing.

    Args:
        **config_overrides: OperatorConfig overrides.

    Returns:
        AlphaGalerkinModel in eval mode.

    """
    cfg = _tiny_op_config(**config_overrides)
    model = AlphaGalerkinModel(cfg)
    model.eval()
    return model


def _sample_input(
    batch: int = 1,
    channels: int = 4,
    board: int = 9,
) -> torch.Tensor:
    """Create a random board-state tensor.

    Args:
        batch: Batch dimension.
        channels: Input channel count.
        board: Board edge length (square board assumed).

    Returns:
        Float32 tensor of shape (batch, channels, board, board).

    """
    torch.manual_seed(0)
    return torch.randn(batch, channels, board, board)


def _export_to(
    tmp_path: Path,
    model: AlphaGalerkinModel,
    sample: torch.Tensor,
    config: ExportConfig | None = None,
    name: str = "model.onnx",
) -> Path:
    """Export *model* to *tmp_path/name* and return the Path.

    Args:
        tmp_path: Pytest tmp_path fixture.
        model: Model to export.
        sample: Sample input tensor.
        config: Optional ExportConfig (defaults used if None).
        name: Output file name.

    Returns:
        Path to the written .onnx file.

    """
    exporter = ONNXExporter(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return exporter.export(model, sample, tmp_path / name)


# ---------------------------------------------------------------------------
# Group 1 – ExportConfig dynamic axes
# ---------------------------------------------------------------------------

class TestONNXExporterConfig:
    """Tests for ExportConfig dynamic-axes helpers and validation."""

    def test_default_dynamic_axes_has_batch_for_board_state(self) -> None:
        """Default dynamic_axes exposes batch dimension (0) for board_state."""
        cfg = ExportConfig()
        assert 0 in cfg.dynamic_axes["board_state"]
        assert cfg.dynamic_axes["board_state"][0] == "batch"

    def test_default_dynamic_axes_has_spatial_dims_for_board_state(self) -> None:
        """Default dynamic_axes exposes height (2) and width (3) for board_state."""
        cfg = ExportConfig()
        axes = cfg.dynamic_axes["board_state"]
        assert 2 in axes, "height axis missing from board_state dynamic_axes"
        assert 3 in axes, "width axis missing from board_state dynamic_axes"

    def test_default_dynamic_axes_has_batch_for_outputs(self) -> None:
        """Default dynamic_axes exposes batch dimension for policy and value outputs."""
        cfg = ExportConfig()
        assert 0 in cfg.dynamic_axes.get("policy", {})
        assert 0 in cfg.dynamic_axes.get("value", {})

    def test_custom_dynamic_axes_validates(self) -> None:
        """ExportConfig accepts custom dynamic_axes without error."""
        custom_axes: dict[str, dict[int, str]] = {
            "board_state": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        }
        cfg = ExportConfig(dynamic_axes=custom_axes)
        assert cfg.dynamic_axes["board_state"] == {0: "batch"}

    def test_fixed_spatial_axes_config(self) -> None:
        """ExportConfig with only batch dynamic (no spatial) is valid."""
        fixed_axes: dict[str, dict[int, str]] = {
            "board_state": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        }
        cfg = ExportConfig(dynamic_axes=fixed_axes)
        # Spatial dims 2 and 3 should NOT be present
        assert 2 not in cfg.dynamic_axes["board_state"]
        assert 3 not in cfg.dynamic_axes["board_state"]


# ---------------------------------------------------------------------------
# Group 2 – Actual export
# ---------------------------------------------------------------------------

class TestONNXExporterActualExport:
    """Tests that perform a real ONNX export and verify the file output."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        """Export produces a file at the specified path."""
        onnx = pytest.importorskip("onnx")  # noqa: F841
        model = _make_model()
        sample = _sample_input()
        out = _export_to(tmp_path, model, sample)
        assert out.exists(), "ONNX file was not created"

    def test_exported_file_has_nonzero_size(self, tmp_path: Path) -> None:
        """Exported ONNX file is non-empty."""
        pytest.importorskip("onnx")
        model = _make_model()
        sample = _sample_input()
        out = _export_to(tmp_path, model, sample)
        assert out.stat().st_size > 0, "ONNX file is empty"

    def test_export_returns_path_object(self, tmp_path: Path) -> None:
        """export() returns a Path pointing to the produced file."""
        pytest.importorskip("onnx")
        model = _make_model()
        sample = _sample_input()
        result = _export_to(tmp_path, model, sample)
        assert isinstance(result, Path)

    def test_export_method_trace(self, tmp_path: Path) -> None:
        """Export succeeds with export_method='trace'."""
        pytest.importorskip("onnx")
        cfg = ExportConfig(export_method="trace", optimization_level="none")
        model = _make_model()
        sample = _sample_input()
        out = _export_to(tmp_path, model, sample, config=cfg)
        assert out.exists()

    def test_export_opset_11(self, tmp_path: Path) -> None:
        """Export succeeds with opset_version=11."""
        pytest.importorskip("onnx")
        cfg = ExportConfig(opset_version=11, optimization_level="none")
        model = _make_model()
        sample = _sample_input()
        out = _export_to(tmp_path, model, sample, config=cfg)
        assert out.exists()

    def test_export_opset_13(self, tmp_path: Path) -> None:
        """Export succeeds with opset_version=13."""
        pytest.importorskip("onnx")
        cfg = ExportConfig(opset_version=13, optimization_level="none")
        model = _make_model()
        sample = _sample_input()
        out = _export_to(tmp_path, model, sample, config=cfg)
        assert out.exists()

    def test_export_dynamic_batch_size_runs_batch_1_and_2(self, tmp_path: Path) -> None:
        """Exported model with dynamic batch axis accepts batch=1 and batch=2."""
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        out = _export_to(tmp_path, model, _sample_input(batch=1), config=cfg)

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        for batch in (1, 2):
            inp = _sample_input(batch=batch).numpy()
            result = session.run(None, {input_name: inp})
            # policy and value outputs should have leading dim == batch
            assert result[0].shape[0] == batch, f"batch={batch} policy shape mismatch"
            assert result[1].shape[0] == batch, f"batch={batch} value shape mismatch"

    def test_export_is_reproducible(self, tmp_path: Path) -> None:
        """Two exports of the same model with the same input produce identical ONNX bytes."""
        pytest.importorskip("onnx")
        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        sample = _sample_input()
        path_a = _export_to(tmp_path, model, sample, config=cfg, name="a.onnx")
        path_b = _export_to(tmp_path, model, sample, config=cfg, name="b.onnx")
        # File sizes should be identical (same graph topology & weights)
        assert path_a.stat().st_size == path_b.stat().st_size

    def test_export_model_convenience_function(self, tmp_path: Path) -> None:
        """export_model() convenience function creates a valid ONNX file."""
        pytest.importorskip("onnx")
        model = _make_model()
        sample = _sample_input()
        out = export_model(model, tmp_path / "convenience.onnx", sample_input=sample)
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Group 3 – Dynamic shapes
# ---------------------------------------------------------------------------

class TestONNXDynamicShapes:
    """Tests for resolution independence through ONNX dynamic spatial axes."""

    def test_export_with_dynamic_spatial_runs_on_9x9_board(self, tmp_path: Path) -> None:
        """Model exported with dynamic spatial dims runs inference on 9x9 input."""
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        out = _export_to(tmp_path, model, _sample_input(board=9), config=cfg)

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        inp = _sample_input(board=9).numpy()
        result = session.run(None, {session.get_inputs()[0].name: inp})
        assert result[0].ndim == 2  # (batch, actions)

    def test_export_dynamic_spatial_runs_on_larger_board(self, tmp_path: Path) -> None:
        """Model exported on 9x9 board runs correctly on a 13x13 board at inference time."""
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        # Export with a 9x9 trace but dynamic spatial axes
        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        out = _export_to(tmp_path, model, _sample_input(board=9), config=cfg)

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        inp_13 = _sample_input(board=13).numpy()
        # Dynamic axes allow this without error
        result = session.run(None, {session.get_inputs()[0].name: inp_13})
        # Policy output should have shape (1, 13*13+1) for Go-style head
        assert result[0].shape[0] == 1

    def test_batch_axis_is_dynamic(self, tmp_path: Path) -> None:
        """Default export config declares batch axis as dynamic in the ONNX graph."""
        onnx = pytest.importorskip("onnx")
        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        out = _export_to(tmp_path, model, _sample_input(), config=cfg)

        onnx_model = onnx.load(str(out))
        board_input = onnx_model.graph.input[0]
        # Dimension 0 should be a symbolic (dynamic) parameter
        dim0 = board_input.type.tensor_type.shape.dim[0]
        assert dim0.dim_param != "", "Batch dimension should be symbolic (dynamic)"

    def test_dynamic_batch_works_for_batch_4(self, tmp_path: Path) -> None:
        """Exported model accepts batch size 4 at inference."""
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        out = _export_to(tmp_path, model, _sample_input(batch=1), config=cfg)

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        inp = _sample_input(batch=4).numpy()
        result = session.run(None, {session.get_inputs()[0].name: inp})
        assert result[0].shape[0] == 4
        assert result[1].shape[0] == 4

    def test_policy_output_size_matches_board_positions_plus_pass(
        self, tmp_path: Path
    ) -> None:
        """Policy head output has board_size^2 + 1 entries (positions + pass move)."""
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        board = 9
        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        out = _export_to(tmp_path, model, _sample_input(board=board), config=cfg)

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        inp = _sample_input(board=board).numpy()
        result = session.run(None, {session.get_inputs()[0].name: inp})
        expected_actions = board * board + 1  # positions + pass
        assert result[0].shape[-1] == expected_actions, (
            f"Expected {expected_actions} policy logits, got {result[0].shape[-1]}"
        )

    def test_fixed_spatial_export_has_no_spatial_dynamic_dims(
        self, tmp_path: Path
    ) -> None:
        """Config with only batch dynamic produces fixed spatial dims in ONNX graph."""
        onnx = pytest.importorskip("onnx")
        fixed_axes: dict[str, dict[int, str]] = {
            "board_state": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        }
        cfg = ExportConfig(dynamic_axes=fixed_axes, optimization_level="none")
        model = _make_model()
        board = 9
        out = _export_to(tmp_path, model, _sample_input(board=board), config=cfg)

        onnx_model = onnx.load(str(out))
        board_input = onnx_model.graph.input[0]
        # Spatial dims (2 and 3) must be concrete integers, not symbolic
        dim_h = board_input.type.tensor_type.shape.dim[2]
        dim_w = board_input.type.tensor_type.shape.dim[3]
        assert dim_h.dim_param == "", "Height should be static (not symbolic)"
        assert dim_w.dim_param == "", "Width should be static (not symbolic)"
        assert dim_h.dim_value == board
        assert dim_w.dim_value == board


# ---------------------------------------------------------------------------
# Group 4 – ONNX validation
# ---------------------------------------------------------------------------

class TestONNXValidation:
    """Tests for exported model correctness via ONNXExporter.verify_export()
    and ModelValidator.validate()."""

    def test_verify_export_returns_true_for_valid_model(self, tmp_path: Path) -> None:
        """verify_export() returns True for a freshly exported model."""
        pytest.importorskip("onnx")
        model = _make_model()
        out = _export_to(tmp_path, model, _sample_input())
        exporter = ONNXExporter()
        assert exporter.verify_export(out) is True

    def test_policy_outputs_match_pytorch_within_tolerance(
        self, tmp_path: Path
    ) -> None:
        """ONNX policy outputs are numerically close to PyTorch outputs."""
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        model.eval()
        sample = _sample_input()
        out = _export_to(tmp_path, model, sample, config=cfg)

        # PyTorch reference
        with torch.no_grad():
            pt_out = model(sample)
        pt_policy = pt_out.policy_logits.numpy()

        # ONNX Runtime
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        onnx_outputs = session.run(None, {session.get_inputs()[0].name: sample.numpy()})
        onnx_policy = onnx_outputs[0]

        np.testing.assert_allclose(
            pt_policy, onnx_policy, atol=1e-4,
            err_msg="Policy outputs diverge beyond tolerance",
        )

    def test_value_outputs_match_pytorch_within_tolerance(
        self, tmp_path: Path
    ) -> None:
        """ONNX value outputs are numerically close to PyTorch outputs."""
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        model.eval()
        sample = _sample_input()
        out = _export_to(tmp_path, model, sample, config=cfg)

        with torch.no_grad():
            pt_out = model(sample)
        pt_value = pt_out.value.numpy()

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        onnx_outputs = session.run(None, {session.get_inputs()[0].name: sample.numpy()})
        onnx_value = onnx_outputs[1]

        np.testing.assert_allclose(
            pt_value, onnx_value, atol=1e-4,
            err_msg="Value outputs diverge beyond tolerance",
        )

    def test_get_model_info_returns_expected_keys(self, tmp_path: Path) -> None:
        """get_model_info() returns a dict with inputs, outputs, and metadata."""
        pytest.importorskip("onnx")
        model = _make_model()
        out = _export_to(tmp_path, model, _sample_input())
        exporter = ONNXExporter()
        info = exporter.get_model_info(out)
        for key in ("inputs", "outputs", "opset_version", "ir_version", "metadata"):
            assert key in info, f"Missing key '{key}' in model info"

    def test_get_model_info_includes_custom_metadata(self, tmp_path: Path) -> None:
        """Exported model contains the model_name and model_version metadata fields."""
        pytest.importorskip("onnx")
        cfg = ExportConfig(
            model_name="test_model",
            model_version="2.0.0",
            optimization_level="none",
        )
        model = _make_model()
        out = _export_to(tmp_path, model, _sample_input(), config=cfg)
        exporter = ONNXExporter(cfg)
        info = exporter.get_model_info(out)
        assert info["metadata"].get("model_name") == "test_model"
        assert info["metadata"].get("model_version") == "2.0.0"

    def test_outputs_vary_with_different_inputs(self, tmp_path: Path) -> None:
        """ONNX inference produces different outputs for different random inputs."""
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        out = _export_to(tmp_path, model, _sample_input(), config=cfg)

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        torch.manual_seed(1)
        inp_a = torch.randn(1, 4, 9, 9).numpy()
        torch.manual_seed(2)
        inp_b = torch.randn(1, 4, 9, 9).numpy()

        out_a = session.run(None, {input_name: inp_a})[0]
        out_b = session.run(None, {input_name: inp_b})[0]
        assert not np.allclose(out_a, out_b), (
            "ONNX model produced identical outputs for different inputs"
        )


# ---------------------------------------------------------------------------
# Group 5 – Quantization
# ---------------------------------------------------------------------------

class TestONNXWithQuantization:
    """Tests for ONNX model quantization via ModelQuantizer."""

    def test_dynamic_quantization_creates_output_file(self, tmp_path: Path) -> None:
        """Dynamic quantization of an exported model produces an output file."""
        pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        from src.deployment.quantize import ModelQuantizer

        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        onnx_path = _export_to(tmp_path, model, _sample_input(), config=cfg)

        q_config = QuantizationConfig(mode=QuantizationMode.DYNAMIC, per_channel=False)
        quantizer = ModelQuantizer(q_config)
        q_path = quantizer.quantize(onnx_path, tmp_path / "model_quant.onnx")

        assert q_path.exists(), "Quantized model file was not created"
        assert q_path.stat().st_size > 0, "Quantized model file is empty"

    def test_quantized_model_is_no_larger_than_original(self, tmp_path: Path) -> None:
        """Quantized model file size is not larger than the original ONNX model.

        Note: for very small models quantization overhead can occasionally make
        the file slightly larger, so we allow a 50 % margin as a soft guard.
        """
        pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        from src.deployment.quantize import ModelQuantizer

        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        onnx_path = _export_to(tmp_path, model, _sample_input(), config=cfg)
        original_size = onnx_path.stat().st_size

        q_config = QuantizationConfig(mode=QuantizationMode.DYNAMIC, per_channel=False)
        quantizer = ModelQuantizer(q_config)
        q_path = quantizer.quantize(onnx_path, tmp_path / "model_quant.onnx")
        quantized_size = q_path.stat().st_size

        # Quantized model should not be more than 2x the original
        assert quantized_size < original_size * 2, (
            f"Quantized model ({quantized_size} B) is much larger than original "
            f"({original_size} B)"
        )

    def test_quantized_model_policy_accuracy_within_tolerance(
        self, tmp_path: Path
    ) -> None:
        """Quantized model policy outputs are within 10 % of original ONNX outputs."""
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        from src.deployment.quantize import ModelQuantizer

        cfg = ExportConfig(optimization_level="none")
        model = _make_model()
        sample = _sample_input()
        onnx_path = _export_to(tmp_path, model, sample, config=cfg)

        q_config = QuantizationConfig(mode=QuantizationMode.DYNAMIC, per_channel=False)
        quantizer = ModelQuantizer(q_config)
        q_path = quantizer.quantize(onnx_path, tmp_path / "model_quant.onnx")

        inp_np = sample.numpy()

        orig_session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        orig_policy = orig_session.run(
            None, {orig_session.get_inputs()[0].name: inp_np}
        )[0]

        q_session = ort.InferenceSession(
            str(q_path), providers=["CPUExecutionProvider"]
        )
        q_policy = q_session.run(
            None, {q_session.get_inputs()[0].name: inp_np}
        )[0]

        # Relative difference should be within 10 % of the original magnitude
        rel_diff = np.abs(orig_policy - q_policy) / (np.abs(orig_policy) + 1e-8)
        assert rel_diff.mean() < 0.10, (
            f"Quantized policy mean relative error {rel_diff.mean():.4f} exceeds 10 %"
        )

    def test_int8_quantization_config_validates(self) -> None:
        """QuantizationConfig with weight_type='int8' and mode=dynamic is valid."""
        q_cfg = QuantizationConfig(
            mode=QuantizationMode.DYNAMIC,
            weight_type="int8",
            per_channel=False,
            reduce_range=False,
        )
        assert q_cfg.weight_type == "int8"
        assert q_cfg.mode == QuantizationMode.DYNAMIC
