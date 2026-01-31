"""Model quantization utilities for efficient deployment.

This module provides utilities for quantizing ONNX models to reduce
model size and improve inference speed on edge devices.

Features:
    - Dynamic quantization (weights only)
    - Static quantization (weights + activations)
    - Calibration data generation
    - Quantization validation
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch

from src.deployment.config import QuantizationConfig, QuantizationMode

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = structlog.get_logger(__name__)


class CalibrationDataReader:
    """Provides calibration data for static quantization.

    Implements the interface expected by ONNX Runtime quantization tools.

    Attributes:
        data_generator: Generator that yields calibration samples.
        input_name: Name of the model input.

    """

    def __init__(
        self,
        data_generator: Iterator[np.ndarray],
        input_name: str = "board_state",
    ) -> None:
        """Initialize calibration data reader.

        Args:
            data_generator: Generator yielding numpy arrays.
            input_name: Name of the model input.

        """
        self.input_name = input_name
        self._enum_data: list[dict[str, np.ndarray]] = [
            {input_name: data} for data in data_generator
        ]
        self._current_index: int = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Get next calibration sample.

        Returns:
            Dictionary mapping input name to numpy array, or None if exhausted.

        """
        if self._current_index >= len(self._enum_data):
            return None
        result = self._enum_data[self._current_index]
        self._current_index += 1
        return result

    def set_range(self, start_index: int, end_index: int) -> None:
        """Set range for calibration (required interface method).

        Args:
            start_index: Index to start reading from.
            end_index: End index (unused, kept for interface compatibility).

        """
        self._current_index = start_index

    def rewind(self) -> None:
        """Rewind the data reader to the beginning."""
        self._current_index = 0


class ModelQuantizer:
    """Quantizes ONNX models for efficient deployment.

    Supports multiple quantization modes including dynamic and static
    quantization with various calibration methods.

    Attributes:
        config: Quantization configuration.

    """

    def __init__(self, config: QuantizationConfig | None = None) -> None:
        """Initialize quantizer.

        Args:
            config: Quantization configuration. Uses defaults if None.

        """
        self.config = config or QuantizationConfig()
        self._logger = structlog.get_logger(__name__).bind(
            mode=self.config.mode.value,
            weight_type=self.config.weight_type,
        )

    def quantize(
        self,
        model_path: str | Path,
        output_path: str | Path | None = None,
        calibration_data: Iterator[np.ndarray] | None = None,
    ) -> Path:
        """Quantize an ONNX model.

        Args:
            model_path: Path to input ONNX model.
            output_path: Path for output quantized model.
            calibration_data: Calibration data for static quantization.

        Returns:
            Path to quantized model.

        """
        model_path = Path(model_path)

        if output_path is None:
            suffix = f"_quant_{self.config.weight_type}"
            output_path = model_path.with_stem(model_path.stem + suffix)
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._logger.info(
            "starting_quantization",
            input_path=str(model_path),
            output_path=str(output_path),
        )

        if self.config.mode == QuantizationMode.DYNAMIC:
            self._quantize_dynamic(model_path, output_path)
        elif self.config.mode == QuantizationMode.STATIC:
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            self._quantize_static(model_path, output_path, calibration_data)
        else:
            raise ValueError(f"Unsupported quantization mode: {self.config.mode}")

        # Log results
        original_size = model_path.stat().st_size / (1024 * 1024)
        quantized_size = output_path.stat().st_size / (1024 * 1024)

        self._logger.info(
            "quantization_completed",
            original_size_mb=f"{original_size:.2f}",
            quantized_size_mb=f"{quantized_size:.2f}",
            compression_ratio=f"{original_size / quantized_size:.2f}x",
        )

        return output_path

    def _quantize_dynamic(
        self,
        model_path: Path,
        output_path: Path,
    ) -> None:
        """Apply dynamic quantization.

        Args:
            model_path: Input model path.
            output_path: Output model path.

        """
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            weight_type = self._get_quant_type(self.config.weight_type)

            quantize_dynamic(
                model_input=str(model_path),
                model_output=str(output_path),
                weight_type=weight_type,
                per_channel=self.config.per_channel,
                reduce_range=self.config.reduce_range,
                nodes_to_exclude=self.config.nodes_to_exclude or [],
            )

        except ImportError as e:
            self._logger.error(
                "quantization_import_error",
                error=str(e),
                message="Install onnxruntime-tools for quantization",
            )
            raise

    def _quantize_static(
        self,
        model_path: Path,
        output_path: Path,
        calibration_data: Iterator[np.ndarray],
    ) -> None:
        """Apply static quantization with calibration.

        Args:
            model_path: Input model path.
            output_path: Output model path.
            calibration_data: Calibration data generator.

        """
        try:
            from onnxruntime.quantization import (
                CalibrationMethod,
                QuantFormat,
                QuantType,
                quantize_static,
            )

            # Create calibration data reader
            # NOTE: input_name must match the ONNX model's input tensor name, not operator names
            calibration_reader = CalibrationDataReader(
                data_generator=calibration_data,
                input_name=self.config.input_name,
            )

            weight_type = self._get_quant_type(self.config.weight_type)
            activation_type = self._get_quant_type(self.config.activation_type)

            # Map calibration method
            calib_method_map = {
                "minmax": CalibrationMethod.MinMax,
                "entropy": CalibrationMethod.Entropy,
                "percentile": CalibrationMethod.Percentile,
            }
            calibration_method = calib_method_map.get(
                self.config.calibration_method,
                CalibrationMethod.Entropy,
            )

            quantize_static(
                model_input=str(model_path),
                model_output=str(output_path),
                calibration_data_reader=calibration_reader,
                weight_type=weight_type,
                activation_type=activation_type,
                quant_format=QuantFormat.QDQ,
                calibrate_method=calibration_method,
                per_channel=self.config.per_channel,
                reduce_range=self.config.reduce_range,
                nodes_to_exclude=self.config.nodes_to_exclude or [],
            )

        except ImportError as e:
            self._logger.error(
                "quantization_import_error",
                error=str(e),
                message="Install onnxruntime-tools for static quantization",
            )
            raise

    def _get_quant_type(self, type_str: str) -> Any:
        """Convert type string to ONNX Runtime QuantType.

        Args:
            type_str: Type string ("int8", "uint8", "int4").

        Returns:
            QuantType enum value.

        """
        try:
            from onnxruntime.quantization import QuantType

            type_map = {
                "int8": QuantType.QInt8,
                "uint8": QuantType.QUInt8,
                "int4": QuantType.QInt4,
            }
            return type_map.get(type_str, QuantType.QInt8)
        except ImportError:
            return None

    def generate_calibration_data(
        self,
        n_samples: int,
        board_sizes: list[int],
        input_channels: int = 17,
    ) -> Iterator[np.ndarray]:
        """Generate random calibration data.

        Args:
            n_samples: Number of calibration samples.
            board_sizes: Available board sizes to sample from.
            input_channels: Number of input channels.

        Yields:
            Calibration data samples as numpy arrays.

        """
        import random

        for _ in range(n_samples):
            board_size = random.choice(board_sizes)
            data = np.random.randn(1, input_channels, board_size, board_size).astype(np.float32)
            yield data

    def generate_calibration_from_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        n_samples: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Generate calibration data from a PyTorch dataset.

        Args:
            dataset: PyTorch dataset.
            n_samples: Number of samples (None = all).

        Yields:
            Calibration data samples as numpy arrays.

        """
        n_samples = n_samples or len(dataset)
        n_samples = min(n_samples, len(dataset))

        for i in range(n_samples):
            sample = dataset[i]

            # Handle different dataset formats
            if isinstance(sample, dict):
                data = sample.get("board_state", sample.get("input"))
            elif isinstance(sample, (tuple, list)):
                data = sample[0]
            else:
                data = sample

            if isinstance(data, torch.Tensor):
                data = data.numpy()

            # Add batch dimension if needed
            if data.ndim == 3:
                data = np.expand_dims(data, 0)

            yield data.astype(np.float32)


def create_quantizer(
    mode: str = "dynamic",
    **kwargs: Any,
) -> ModelQuantizer:
    """Factory function to create model quantizer.

    Args:
        mode: Quantization mode.
        **kwargs: Additional configuration options.

    Returns:
        Configured ModelQuantizer instance.

    """
    config = QuantizationConfig(mode=QuantizationMode(mode), **kwargs)
    return ModelQuantizer(config)


def quantize_model(
    model_path: str | Path,
    output_path: str | Path | None = None,
    mode: str = "dynamic",
    **kwargs: Any,
) -> Path:
    """Convenience function to quantize a model.

    Args:
        model_path: Path to input ONNX model.
        output_path: Path for output quantized model.
        mode: Quantization mode.
        **kwargs: Additional quantization options.

    Returns:
        Path to quantized model.

    """
    quantizer = create_quantizer(mode=mode, **kwargs)
    return quantizer.quantize(model_path, output_path)
