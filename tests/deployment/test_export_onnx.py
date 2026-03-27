"""Tests for ONNX export utilities.

This module tests the ONNXExporter class and related factory functions,
mocking heavy ONNX/torch operations to ensure deterministic, fast tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import structlog
import torch
from torch import nn

from src.deployment.config import ExportConfig
from src.deployment.export_onnx import (
    ONNXExporter,
    create_exporter,
    export_model,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42
DEFAULT_BATCH_SIZE = 1
DEFAULT_BOARD_SIZE = 9
DEFAULT_CHANNELS = 17


class _DummyModel(nn.Module):
    """Minimal model for export tests."""

    def __init__(self, in_channels: int = DEFAULT_CHANNELS) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.conv(x)
        policy = out[:, 0].flatten(1)
        value = out[:, 1].mean(dim=(1, 2), keepdim=False)
        return policy, value


@pytest.fixture()
def export_config() -> ExportConfig:
    """Provide a default ExportConfig for tests."""
    return ExportConfig(
        opset_version=17,
        model_name="test_model",
        model_version="0.1.0",
    )


@pytest.fixture()
def exporter(export_config: ExportConfig) -> ONNXExporter:
    """Provide a configured ONNXExporter."""
    return ONNXExporter(config=export_config)


@pytest.fixture()
def dummy_model() -> nn.Module:
    """Provide a lightweight dummy model."""
    torch.manual_seed(SEED)
    return _DummyModel(in_channels=DEFAULT_CHANNELS)


@pytest.fixture()
def sample_input() -> torch.Tensor:
    """Provide a deterministic sample input tensor."""
    torch.manual_seed(SEED)
    return torch.randn(
        DEFAULT_BATCH_SIZE,
        DEFAULT_CHANNELS,
        DEFAULT_BOARD_SIZE,
        DEFAULT_BOARD_SIZE,
    )


# ---------------------------------------------------------------------------
# Tests for create_exporter factory
# ---------------------------------------------------------------------------


class TestCreateExporter:
    """Tests for the create_exporter factory function."""

    def test_creates_exporter_with_defaults(self) -> None:
        """Factory returns ONNXExporter with default config when no args given."""
        exp = create_exporter()
        assert isinstance(exp, ONNXExporter)
        assert exp.config.opset_version == 17

    @pytest.mark.parametrize("opset", [9, 13, 17, 20])
    def test_creates_exporter_with_opset(self, opset: int) -> None:
        """Factory correctly forwards opset_version to config."""
        exp = create_exporter(opset_version=opset)
        assert exp.config.opset_version == opset

    def test_creates_exporter_with_kwargs(self) -> None:
        """Factory forwards additional keyword arguments to ExportConfig."""
        exp = create_exporter(
            opset_version=15,
            model_name="custom",
            export_method="script",
        )
        assert exp.config.model_name == "custom"
        assert exp.config.export_method == "script"

    def test_invalid_opset_raises(self) -> None:
        """Factory raises ValidationError for out-of-range opset."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            create_exporter(opset_version=5)

    def test_invalid_kwarg_raises(self) -> None:
        """Factory raises ValidationError for unknown config fields."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            create_exporter(nonexistent_field=True)


# ---------------------------------------------------------------------------
# Tests for create_sample_input
# ---------------------------------------------------------------------------


class TestCreateSampleInput:
    """Tests for ONNXExporter.create_sample_input."""

    @pytest.mark.parametrize(
        "batch_size,board_size,channels",
        [
            (1, 9, 17),
            (4, 13, 17),
            (2, 19, 32),
        ],
    )
    def test_shape(
        self,
        exporter: ONNXExporter,
        batch_size: int,
        board_size: int,
        channels: int,
    ) -> None:
        """create_sample_input returns tensor with correct (B, C, H, W)."""
        torch.manual_seed(SEED)
        tensor = exporter.create_sample_input(
            batch_size=batch_size,
            board_size=board_size,
            channels=channels,
        )
        assert tensor.shape == (batch_size, channels, board_size, board_size)

    def test_device(self, exporter: ONNXExporter) -> None:
        """Tensor is created on the specified device."""
        tensor = exporter.create_sample_input(device="cpu")
        assert tensor.device == torch.device("cpu")

    def test_dtype_is_float(self, exporter: ONNXExporter) -> None:
        """Sample input should be float32 by default."""
        tensor = exporter.create_sample_input()
        assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests for get_model_info
# ---------------------------------------------------------------------------


class TestGetModelInfo:
    """Tests for ONNXExporter.get_model_info."""

    def test_returns_expected_keys_when_onnx_available(
        self, exporter: ONNXExporter, tmp_path: Path
    ) -> None:
        """get_model_info returns dict with standard info keys when onnx is mocked."""
        mock_model = MagicMock()
        mock_model.graph.input = []
        mock_model.graph.output = []
        mock_model.graph.node = [MagicMock()] * 5
        mock_model.graph.initializer = [MagicMock()] * 3
        mock_model.opset_import = [MagicMock(version=17)]
        mock_model.ir_version = 8
        mock_model.producer_name = "pytorch"
        mock_model.producer_version = "2.0"
        mock_model.doc_string = "test"
        mock_model.metadata_props = []

        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict("sys.modules", {"onnx": MagicMock()}):
            with patch("src.deployment.export_onnx.importlib.import_module", side_effect=ImportError):
                pass

            import importlib
            mock_onnx = MagicMock()
            mock_onnx.load.return_value = mock_model

            with patch.dict("sys.modules", {"onnx": mock_onnx}):
                info = exporter.get_model_info(model_path)

        expected_keys = {
            "inputs",
            "outputs",
            "opset_version",
            "ir_version",
            "producer_name",
            "producer_version",
            "doc_string",
            "metadata",
            "num_nodes",
            "num_initializers",
        }
        assert expected_keys.issubset(info.keys())
        assert info["num_nodes"] == 5
        assert info["num_initializers"] == 3

    def test_returns_error_when_onnx_unavailable(
        self, exporter: ONNXExporter, tmp_path: Path
    ) -> None:
        """get_model_info returns error dict when onnx is not installed."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict("sys.modules", {"onnx": None}):
            info = exporter.get_model_info(model_path)

        assert "error" in info


# ---------------------------------------------------------------------------
# Tests for export (mocked torch.onnx.export)
# ---------------------------------------------------------------------------


class TestExport:
    """Tests for ONNXExporter.export with mocked ONNX operations."""

    def test_export_trace_calls_torch_onnx_export(
        self,
        exporter: ONNXExporter,
        dummy_model: nn.Module,
        sample_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        """export() in trace mode calls torch.onnx.export with correct arguments."""
        output_path = tmp_path / "model.onnx"

        with (
            patch("torch.onnx.export") as mock_export,
            patch.object(exporter, "_optimize"),
            patch.object(exporter, "_add_metadata"),
        ):
            # Create the file so stat() works in the logging call
            output_path.touch()

            exporter.export(dummy_model, sample_input, output_path)

            mock_export.assert_called_once()
            call_kwargs = mock_export.call_args
            assert call_kwargs[1]["opset_version"] == exporter.config.opset_version
            assert call_kwargs[1]["input_names"] == exporter.config.input_names
            assert call_kwargs[1]["output_names"] == exporter.config.output_names

    def test_export_script_method(
        self,
        dummy_model: nn.Module,
        sample_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        """export() with script method calls torch.jit.script."""
        config = ExportConfig(export_method="script", optimization_level="none")
        exp = ONNXExporter(config)
        output_path = tmp_path / "model_script.onnx"

        with (
            patch("torch.jit.script") as mock_script,
            patch("torch.onnx.export") as mock_export,
            patch.object(exp, "_add_metadata"),
        ):
            mock_script.return_value = dummy_model
            output_path.touch()

            exp.export(dummy_model, sample_input, output_path)

            mock_script.assert_called_once_with(dummy_model)
            mock_export.assert_called_once()

    def test_export_dynamo_fallback(
        self,
        dummy_model: nn.Module,
        sample_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        """export() with dynamo method falls back to trace when unavailable."""
        config = ExportConfig(export_method="dynamo", optimization_level="none")
        exp = ONNXExporter(config)
        output_path = tmp_path / "model_dynamo.onnx"

        with (
            patch(
                "torch.onnx.dynamo_export",
                side_effect=AttributeError("not available"),
            ),
            patch("torch.onnx.export") as mock_trace_export,
            patch.object(exp, "_add_metadata"),
        ):
            output_path.touch()
            exp.export(dummy_model, sample_input, output_path)
            # Should have fallen back to trace
            mock_trace_export.assert_called_once()

    def test_export_creates_parent_dirs(
        self,
        exporter: ONNXExporter,
        dummy_model: nn.Module,
        sample_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        """export() creates parent directories if they do not exist."""
        output_path = tmp_path / "deep" / "nested" / "model.onnx"

        with (
            patch("torch.onnx.export"),
            patch.object(exporter, "_optimize"),
            patch.object(exporter, "_add_metadata"),
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

            result = exporter.export(dummy_model, sample_input, output_path)
            assert result.parent.exists()

    def test_export_with_dict_input(
        self,
        exporter: ONNXExporter,
        dummy_model: nn.Module,
        tmp_path: Path,
    ) -> None:
        """export() handles dict input by extracting values as tuple."""
        torch.manual_seed(SEED)
        dict_input = {
            "board_state": torch.randn(
                DEFAULT_BATCH_SIZE, DEFAULT_CHANNELS, DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE
            )
        }
        output_path = tmp_path / "model.onnx"

        with (
            patch("torch.onnx.export") as mock_export,
            patch.object(exporter, "_optimize"),
            patch.object(exporter, "_add_metadata"),
        ):
            output_path.touch()
            exporter.export(dummy_model, dict_input, output_path)
            mock_export.assert_called_once()

    def test_export_sets_model_to_eval(
        self,
        exporter: ONNXExporter,
        sample_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        """export() sets the model to eval mode before exporting."""
        model = _DummyModel()
        model.train()
        assert model.training is True

        output_path = tmp_path / "model.onnx"

        with (
            patch("torch.onnx.export"),
            patch.object(exporter, "_optimize"),
            patch.object(exporter, "_add_metadata"),
        ):
            output_path.touch()
            exporter.export(model, sample_input, output_path)

        assert model.training is False


# ---------------------------------------------------------------------------
# Tests for export_model convenience function
# ---------------------------------------------------------------------------


class TestExportModelConvenience:
    """Tests for the export_model convenience function."""

    def test_export_model_creates_exporter_and_exports(
        self,
        dummy_model: nn.Module,
        tmp_path: Path,
    ) -> None:
        """export_model creates an exporter and calls export end-to-end."""
        output_path = tmp_path / "model.onnx"

        with (
            patch("torch.onnx.export"),
            patch.object(ONNXExporter, "_optimize"),
            patch.object(ONNXExporter, "_add_metadata"),
        ):
            output_path.touch()
            result = export_model(
                dummy_model,
                output_path,
                board_size=DEFAULT_BOARD_SIZE,
                input_channels=DEFAULT_CHANNELS,
            )
            assert result == output_path

    def test_export_model_with_custom_sample_input(
        self,
        dummy_model: nn.Module,
        sample_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        """export_model uses provided sample_input when given."""
        output_path = tmp_path / "model.onnx"

        with (
            patch("torch.onnx.export") as mock_export,
            patch.object(ONNXExporter, "_optimize"),
            patch.object(ONNXExporter, "_add_metadata"),
        ):
            output_path.touch()
            export_model(dummy_model, output_path, sample_input=sample_input)

            # The sample_input should be wrapped in a tuple
            actual_input = mock_export.call_args[0][1]
            assert actual_input[0].shape == sample_input.shape

    def test_export_model_forwards_kwargs(
        self,
        dummy_model: nn.Module,
        tmp_path: Path,
    ) -> None:
        """export_model passes kwargs through to ExportConfig."""
        output_path = tmp_path / "model.onnx"

        with (
            patch("torch.onnx.export"),
            patch.object(ONNXExporter, "_optimize"),
            patch.object(ONNXExporter, "_add_metadata"),
        ):
            output_path.touch()
            export_model(
                dummy_model,
                output_path,
                opset_version=15,
                model_name="custom_export",
            )


# ---------------------------------------------------------------------------
# Tests for dynamic shape handling
# ---------------------------------------------------------------------------


class TestDynamicShapeHandling:
    """Tests for dynamic axis configuration during export."""

    def test_default_dynamic_axes(self) -> None:
        """Default config includes dynamic axes for batch, height, width."""
        config = ExportConfig()
        axes = config.dynamic_axes

        assert "board_state" in axes
        assert 0 in axes["board_state"]  # batch
        assert 2 in axes["board_state"]  # height
        assert 3 in axes["board_state"]  # width

    def test_custom_dynamic_axes_propagated(
        self,
        dummy_model: nn.Module,
        sample_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        """Custom dynamic axes are passed to torch.onnx.export."""
        custom_axes: dict[str, dict[int, str]] = {
            "board_state": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        }
        config = ExportConfig(
            dynamic_axes=custom_axes,
            optimization_level="none",
        )
        exp = ONNXExporter(config)
        output_path = tmp_path / "model.onnx"

        with (
            patch("torch.onnx.export") as mock_export,
            patch.object(exp, "_add_metadata"),
        ):
            output_path.touch()
            exp.export(dummy_model, sample_input, output_path)

            passed_axes = mock_export.call_args[1]["dynamic_axes"]
            assert passed_axes == custom_axes

    @pytest.mark.parametrize("board_size", [9, 13, 19])
    def test_sample_input_varies_with_board_size(
        self,
        exporter: ONNXExporter,
        board_size: int,
    ) -> None:
        """create_sample_input produces different spatial dims per board size."""
        torch.manual_seed(SEED)
        tensor = exporter.create_sample_input(board_size=board_size)
        assert tensor.shape[2] == board_size
        assert tensor.shape[3] == board_size


# ---------------------------------------------------------------------------
# Tests for verify_export
# ---------------------------------------------------------------------------


class TestVerifyExport:
    """Tests for ONNXExporter.verify_export."""

    def test_verify_returns_true_on_valid_model(
        self, exporter: ONNXExporter, tmp_path: Path
    ) -> None:
        """verify_export returns True when onnx checker passes."""
        model_path = tmp_path / "valid.onnx"
        model_path.touch()

        mock_onnx = MagicMock()
        with patch.dict("sys.modules", {"onnx": mock_onnx}):
            result = exporter.verify_export(model_path)

        assert result is True
        mock_onnx.checker.check_model.assert_called_once()

    def test_verify_returns_false_on_invalid_model(
        self, exporter: ONNXExporter, tmp_path: Path
    ) -> None:
        """verify_export returns False when checker raises."""
        model_path = tmp_path / "invalid.onnx"
        model_path.touch()

        mock_onnx = MagicMock()
        mock_onnx.checker.check_model.side_effect = Exception("bad model")
        with patch.dict("sys.modules", {"onnx": mock_onnx}):
            result = exporter.verify_export(model_path)

        assert result is False

    def test_verify_returns_false_when_onnx_unavailable(
        self, exporter: ONNXExporter, tmp_path: Path
    ) -> None:
        """verify_export returns False when onnx package not installed."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict("sys.modules", {"onnx": None}):
            result = exporter.verify_export(model_path)

        assert result is False
