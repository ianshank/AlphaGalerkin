"""Tests for model quantization utilities.

This module tests the ModelQuantizer class, CalibrationDataReader,
and related factory functions, mocking heavy ONNX Runtime quantization
operations to ensure deterministic, fast tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import structlog

from src.deployment.config import QuantizationConfig, QuantizationMode
from src.deployment.quantize import (
    CalibrationDataReader,
    ModelQuantizer,
    create_quantizer,
    quantize_model,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
DEFAULT_BOARD_SIZE = 9
DEFAULT_CHANNELS = 17
DEFAULT_CALIBRATION_SAMPLES = 20
DEFAULT_INPUT_NAME = "board_state"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def quantization_config() -> QuantizationConfig:
    """Provide a default QuantizationConfig for tests."""
    return QuantizationConfig(
        mode=QuantizationMode.DYNAMIC,
        weight_type="int8",
        per_channel=True,
        calibration_samples=DEFAULT_CALIBRATION_SAMPLES,
        input_name=DEFAULT_INPUT_NAME,
    )


@pytest.fixture()
def static_quantization_config() -> QuantizationConfig:
    """Provide a QuantizationConfig for static quantization tests."""
    return QuantizationConfig(
        mode=QuantizationMode.STATIC,
        weight_type="int8",
        activation_type="int8",
        calibration_method="entropy",
        calibration_samples=DEFAULT_CALIBRATION_SAMPLES,
        input_name=DEFAULT_INPUT_NAME,
    )


@pytest.fixture()
def quantizer(quantization_config: QuantizationConfig) -> ModelQuantizer:
    """Provide a configured ModelQuantizer with dynamic mode."""
    return ModelQuantizer(config=quantization_config)


@pytest.fixture()
def static_quantizer(static_quantization_config: QuantizationConfig) -> ModelQuantizer:
    """Provide a configured ModelQuantizer with static mode."""
    return ModelQuantizer(config=static_quantization_config)


@pytest.fixture()
def dummy_onnx_model(tmp_path: Path) -> Path:
    """Create a dummy ONNX model file for testing."""
    model_path = tmp_path / "model.onnx"
    # Write some bytes so stat().st_size is nonzero
    model_path.write_bytes(b"\x00" * 1024)
    return model_path


@pytest.fixture()
def calibration_data_iterator() -> list[np.ndarray]:
    """Provide deterministic calibration data samples."""
    rng = np.random.default_rng(SEED)
    return [
        rng.standard_normal((1, DEFAULT_CHANNELS, DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE)).astype(
            np.float32
        )
        for _ in range(DEFAULT_CALIBRATION_SAMPLES)
    ]


# ---------------------------------------------------------------------------
# Tests for create_quantizer factory
# ---------------------------------------------------------------------------


class TestCreateQuantizer:
    """Tests for the create_quantizer factory function."""

    def test_creates_quantizer_with_defaults(self) -> None:
        """Factory returns ModelQuantizer with default dynamic config."""
        q = create_quantizer()
        assert isinstance(q, ModelQuantizer)
        assert q.config.mode == QuantizationMode.DYNAMIC

    @pytest.mark.parametrize("mode", ["dynamic", "static"])
    def test_creates_quantizer_with_mode(self, mode: str) -> None:
        """Factory correctly sets quantization mode."""
        q = create_quantizer(mode=mode)
        assert q.config.mode == QuantizationMode(mode)

    def test_creates_quantizer_with_kwargs(self) -> None:
        """Factory forwards additional keyword arguments to QuantizationConfig."""
        q = create_quantizer(
            mode="dynamic",
            weight_type="uint8",
            per_channel=False,
        )
        assert q.config.weight_type == "uint8"
        assert q.config.per_channel is False

    def test_invalid_mode_raises(self) -> None:
        """Factory raises ValueError for unknown quantization mode."""
        with pytest.raises(ValueError):
            create_quantizer(mode="nonexistent_mode")

    def test_invalid_kwarg_raises(self) -> None:
        """Factory raises ValidationError for unknown config fields."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            create_quantizer(mode="dynamic", nonexistent_field=True)

    def test_none_config_uses_defaults(self) -> None:
        """ModelQuantizer with None config falls back to defaults."""
        q = ModelQuantizer(config=None)
        assert q.config.mode == QuantizationMode.DYNAMIC
        assert q.config.weight_type == "int8"


# ---------------------------------------------------------------------------
# Tests for generate_calibration_data
# ---------------------------------------------------------------------------


class TestGenerateCalibrationData:
    """Tests for ModelQuantizer.generate_calibration_data."""

    @pytest.mark.parametrize("n_samples", [1, 5, 10])
    def test_produces_correct_number_of_samples(
        self,
        quantizer: ModelQuantizer,
        n_samples: int,
    ) -> None:
        """generate_calibration_data yields the requested number of samples."""
        import random

        random.seed(SEED)
        np.random.seed(SEED)

        data = list(
            quantizer.generate_calibration_data(
                n_samples=n_samples,
                board_sizes=[DEFAULT_BOARD_SIZE],
                input_channels=DEFAULT_CHANNELS,
            )
        )
        assert len(data) == n_samples

    @pytest.mark.parametrize("board_size", [9, 13, 19])
    def test_produces_correct_shapes(
        self,
        quantizer: ModelQuantizer,
        board_size: int,
    ) -> None:
        """Each calibration sample has shape (1, C, H, W)."""
        import random

        random.seed(SEED)
        np.random.seed(SEED)

        data = list(
            quantizer.generate_calibration_data(
                n_samples=3,
                board_sizes=[board_size],
                input_channels=DEFAULT_CHANNELS,
            )
        )
        for sample in data:
            assert sample.shape == (1, DEFAULT_CHANNELS, board_size, board_size)
            assert sample.dtype == np.float32

    def test_varies_board_sizes(self, quantizer: ModelQuantizer) -> None:
        """When multiple board sizes given, samples can have different spatial dims."""
        import random

        random.seed(SEED)
        np.random.seed(SEED)

        board_sizes = [9, 13, 19]
        data = list(
            quantizer.generate_calibration_data(
                n_samples=30,
                board_sizes=board_sizes,
                input_channels=DEFAULT_CHANNELS,
            )
        )
        observed_sizes = {sample.shape[2] for sample in data}
        # With 30 samples and seed=42, we expect to see at least 2 different sizes
        assert len(observed_sizes) >= 2
        assert observed_sizes.issubset(set(board_sizes))


# ---------------------------------------------------------------------------
# Tests for CalibrationDataReader interface
# ---------------------------------------------------------------------------


class TestCalibrationDataReader:
    """Tests for the CalibrationDataReader that feeds static quantization."""

    def test_get_next_returns_samples_in_order(
        self, calibration_data_iterator: list[np.ndarray]
    ) -> None:
        """get_next returns samples sequentially and None when exhausted."""
        reader = CalibrationDataReader(
            data_generator=iter(calibration_data_iterator),
            input_name=DEFAULT_INPUT_NAME,
        )

        for i in range(DEFAULT_CALIBRATION_SAMPLES):
            result = reader.get_next()
            assert result is not None
            assert DEFAULT_INPUT_NAME in result
            np.testing.assert_array_equal(
                result[DEFAULT_INPUT_NAME], calibration_data_iterator[i]
            )

        # Exhausted
        assert reader.get_next() is None

    def test_rewind_resets_to_beginning(
        self, calibration_data_iterator: list[np.ndarray]
    ) -> None:
        """rewind() allows re-reading samples from the start."""
        reader = CalibrationDataReader(
            data_generator=iter(calibration_data_iterator),
            input_name=DEFAULT_INPUT_NAME,
        )

        # Consume all
        for _ in range(DEFAULT_CALIBRATION_SAMPLES):
            reader.get_next()
        assert reader.get_next() is None

        # Rewind
        reader.rewind()
        first = reader.get_next()
        assert first is not None
        np.testing.assert_array_equal(
            first[DEFAULT_INPUT_NAME], calibration_data_iterator[0]
        )

    def test_set_range_moves_index(
        self, calibration_data_iterator: list[np.ndarray]
    ) -> None:
        """set_range(start, end) sets the current read index."""
        reader = CalibrationDataReader(
            data_generator=iter(calibration_data_iterator),
            input_name=DEFAULT_INPUT_NAME,
        )

        start_idx = 5
        reader.set_range(start_idx, DEFAULT_CALIBRATION_SAMPLES)
        result = reader.get_next()
        assert result is not None
        np.testing.assert_array_equal(
            result[DEFAULT_INPUT_NAME], calibration_data_iterator[start_idx]
        )

    def test_custom_input_name(self) -> None:
        """CalibrationDataReader uses the specified input_name as dict key."""
        rng = np.random.default_rng(SEED)
        data = [rng.standard_normal((1, 3, 9, 9)).astype(np.float32)]
        custom_name = "custom_input"

        reader = CalibrationDataReader(
            data_generator=iter(data),
            input_name=custom_name,
        )
        result = reader.get_next()
        assert result is not None
        assert custom_name in result


# ---------------------------------------------------------------------------
# Tests for quantize with dynamic mode (mocked onnxruntime.quantization)
# ---------------------------------------------------------------------------


class TestQuantizeDynamic:
    """Tests for ModelQuantizer.quantize in dynamic mode."""

    def test_quantize_dynamic_calls_quantize_dynamic(
        self,
        quantizer: ModelQuantizer,
        dummy_onnx_model: Path,
        tmp_path: Path,
    ) -> None:
        """quantize() in dynamic mode calls onnxruntime.quantization.quantize_dynamic."""
        output_path = tmp_path / "model_quant.onnx"
        output_path.write_bytes(b"\x00" * 512)  # Smaller "quantized" file

        mock_quant_type = MagicMock()
        mock_quant_type.QInt8 = "QInt8"
        mock_quant_type.QUInt8 = "QUInt8"
        mock_quant_type.QInt4 = "QInt4"

        mock_quantize_dynamic = MagicMock()

        mock_module = MagicMock()
        mock_module.quantize_dynamic = mock_quantize_dynamic
        mock_module.QuantType = mock_quant_type

        with patch.dict(
            "sys.modules",
            {
                "onnxruntime": MagicMock(),
                "onnxruntime.quantization": mock_module,
            },
        ):
            quantizer.quantize(dummy_onnx_model, output_path)

        mock_quantize_dynamic.assert_called_once()
        call_kwargs = mock_quantize_dynamic.call_args[1]
        assert call_kwargs["model_input"] == str(dummy_onnx_model)
        assert call_kwargs["model_output"] == str(output_path)

    def test_quantize_dynamic_auto_output_path(
        self,
        quantizer: ModelQuantizer,
        dummy_onnx_model: Path,
    ) -> None:
        """quantize() generates output path when not provided."""
        expected_stem = dummy_onnx_model.stem + f"_quant_{quantizer.config.weight_type}"
        expected_output = dummy_onnx_model.with_stem(expected_stem)
        # Pre-create the expected output file
        expected_output.write_bytes(b"\x00" * 512)

        mock_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "onnxruntime": MagicMock(),
                "onnxruntime.quantization": mock_module,
            },
        ):
            result = quantizer.quantize(dummy_onnx_model)

        assert result == expected_output


# ---------------------------------------------------------------------------
# Tests for quantize with static mode
# ---------------------------------------------------------------------------


class TestQuantizeStatic:
    """Tests for ModelQuantizer.quantize in static mode."""

    def test_quantize_static_calls_quantize_static(
        self,
        static_quantizer: ModelQuantizer,
        dummy_onnx_model: Path,
        tmp_path: Path,
        calibration_data_iterator: list[np.ndarray],
    ) -> None:
        """quantize() in static mode calls onnxruntime.quantization.quantize_static."""
        output_path = tmp_path / "model_static_quant.onnx"
        output_path.write_bytes(b"\x00" * 512)

        mock_quantize_static = MagicMock()
        mock_quant_type = MagicMock()
        mock_quant_type.QInt8 = "QInt8"
        mock_quant_type.QUInt8 = "QUInt8"
        mock_quant_type.QInt4 = "QInt4"
        mock_quant_format = MagicMock()
        mock_calibration_method = MagicMock()
        mock_calibration_method.MinMax = "MinMax"
        mock_calibration_method.Entropy = "Entropy"
        mock_calibration_method.Percentile = "Percentile"

        mock_module = MagicMock()
        mock_module.quantize_static = mock_quantize_static
        mock_module.QuantType = mock_quant_type
        mock_module.QuantFormat = mock_quant_format
        mock_module.CalibrationMethod = mock_calibration_method

        with patch.dict(
            "sys.modules",
            {
                "onnxruntime": MagicMock(),
                "onnxruntime.quantization": mock_module,
            },
        ):
            static_quantizer.quantize(
                dummy_onnx_model,
                output_path,
                calibration_data=iter(calibration_data_iterator),
            )

        mock_quantize_static.assert_called_once()
        call_kwargs = mock_quantize_static.call_args[1]
        assert call_kwargs["model_input"] == str(dummy_onnx_model)
        assert call_kwargs["model_output"] == str(output_path)

    def test_static_quantize_requires_calibration_data(
        self,
        static_quantizer: ModelQuantizer,
        dummy_onnx_model: Path,
        tmp_path: Path,
    ) -> None:
        """quantize() in static mode raises ValueError without calibration data."""
        output_path = tmp_path / "model_static_quant.onnx"

        with pytest.raises(
            ValueError, match="Calibration data required for static quantization"
        ):
            static_quantizer.quantize(dummy_onnx_model, output_path)

    @pytest.mark.parametrize(
        "calibration_method", ["minmax", "entropy", "percentile"]
    )
    def test_static_quantize_calibration_methods(
        self,
        calibration_method: str,
        dummy_onnx_model: Path,
        tmp_path: Path,
        calibration_data_iterator: list[np.ndarray],
    ) -> None:
        """Static quantization supports all calibration methods."""
        config = QuantizationConfig(
            mode=QuantizationMode.STATIC,
            calibration_method=calibration_method,
        )
        q = ModelQuantizer(config=config)
        output_path = tmp_path / "model_quant.onnx"
        output_path.write_bytes(b"\x00" * 512)

        mock_quantize_static = MagicMock()
        mock_calibration_method = MagicMock()
        mock_calibration_method.MinMax = "MinMax"
        mock_calibration_method.Entropy = "Entropy"
        mock_calibration_method.Percentile = "Percentile"

        mock_module = MagicMock()
        mock_module.quantize_static = mock_quantize_static
        mock_module.QuantType = MagicMock()
        mock_module.QuantFormat = MagicMock()
        mock_module.CalibrationMethod = mock_calibration_method

        with patch.dict(
            "sys.modules",
            {
                "onnxruntime": MagicMock(),
                "onnxruntime.quantization": mock_module,
            },
        ):
            q.quantize(
                dummy_onnx_model,
                output_path,
                calibration_data=iter(calibration_data_iterator),
            )

        mock_quantize_static.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for unsupported quantization modes
# ---------------------------------------------------------------------------


class TestQuantizeUnsupportedMode:
    """Tests for ValueError on unsupported quantization modes."""

    def test_qat_mode_raises_value_error(
        self,
        dummy_onnx_model: Path,
        tmp_path: Path,
    ) -> None:
        """QAT mode is not implemented and raises ValueError."""
        config = QuantizationConfig(mode=QuantizationMode.QAT)
        q = ModelQuantizer(config=config)
        output_path = tmp_path / "model_qat.onnx"

        with pytest.raises(ValueError, match="Unsupported quantization mode"):
            q.quantize(dummy_onnx_model, output_path)


# ---------------------------------------------------------------------------
# Tests for quantize_model convenience function
# ---------------------------------------------------------------------------


class TestQuantizeModelConvenience:
    """Tests for the quantize_model convenience function."""

    def test_quantize_model_creates_quantizer_and_quantizes(
        self,
        dummy_onnx_model: Path,
        tmp_path: Path,
    ) -> None:
        """quantize_model creates a quantizer and calls quantize end-to-end."""
        output_path = tmp_path / "model_quant.onnx"
        output_path.write_bytes(b"\x00" * 512)

        mock_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "onnxruntime": MagicMock(),
                "onnxruntime.quantization": mock_module,
            },
        ):
            result = quantize_model(
                dummy_onnx_model,
                output_path,
                mode="dynamic",
            )

        assert result == output_path

    def test_quantize_model_forwards_kwargs(
        self,
        dummy_onnx_model: Path,
        tmp_path: Path,
    ) -> None:
        """quantize_model passes kwargs through to QuantizationConfig."""
        output_path = tmp_path / "model_quant.onnx"
        output_path.write_bytes(b"\x00" * 512)

        mock_module = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "onnxruntime": MagicMock(),
                "onnxruntime.quantization": mock_module,
            },
        ):
            quantize_model(
                dummy_onnx_model,
                output_path,
                mode="dynamic",
                weight_type="uint8",
                per_channel=False,
            )


# ---------------------------------------------------------------------------
# Tests for generate_calibration_from_dataset
# ---------------------------------------------------------------------------


class TestGenerateCalibrationFromDataset:
    """Tests for ModelQuantizer.generate_calibration_from_dataset."""

    def test_from_dict_dataset(self, quantizer: ModelQuantizer) -> None:
        """Generates calibration data from a dataset returning dicts."""
        import torch

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 5

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                torch.manual_seed(SEED + idx)
                return {
                    "board_state": torch.randn(
                        DEFAULT_CHANNELS, DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE
                    )
                }

        dataset = DictDataset()
        data = list(quantizer.generate_calibration_from_dataset(dataset, n_samples=3))
        assert len(data) == 3
        for sample in data:
            assert sample.shape == (1, DEFAULT_CHANNELS, DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE)
            assert sample.dtype == np.float32

    def test_from_tuple_dataset(self, quantizer: ModelQuantizer) -> None:
        """Generates calibration data from a dataset returning tuples."""
        import torch

        class TupleDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 5

            def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
                torch.manual_seed(SEED + idx)
                return (
                    torch.randn(DEFAULT_CHANNELS, DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE),
                    idx,
                )

        dataset = TupleDataset()
        data = list(quantizer.generate_calibration_from_dataset(dataset, n_samples=2))
        assert len(data) == 2
        for sample in data:
            # Should have batch dim added
            assert sample.ndim == 4
            assert sample.dtype == np.float32

    def test_n_samples_none_uses_all(self, quantizer: ModelQuantizer) -> None:
        """When n_samples is None, uses entire dataset length."""
        import torch

        class SmallDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 3

            def __getitem__(self, idx: int) -> torch.Tensor:
                torch.manual_seed(SEED + idx)
                return torch.randn(
                    DEFAULT_CHANNELS, DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE
                )

        dataset = SmallDataset()
        data = list(quantizer.generate_calibration_from_dataset(dataset))
        assert len(data) == 3
