"""Deployment utilities for AlphaGalerkin.

This module provides utilities for exporting models to ONNX format
and optimizing them for deployment on various platforms.

Key Components:
    - ExportConfig: Configuration for ONNX export
    - QuantizationConfig: Configuration for model quantization
    - ONNXExporter: PyTorch to ONNX conversion
    - ONNXRuntime: ONNX Runtime inference wrapper
    - ModelValidator: Validation of exported models

Usage:
    from src.deployment import ExportConfig, ONNXExporter

    config = ExportConfig(opset_version=17)
    exporter = ONNXExporter(config)
    onnx_path = exporter.export(model, sample_input)

    # Quantize for edge deployment
    quantized_path = exporter.quantize(onnx_path)
"""

from src.deployment.config import (
    ExportConfig,
    QuantizationConfig,
    RuntimeConfig,
)
from src.deployment.export_onnx import ONNXExporter
from src.deployment.quantize import ModelQuantizer
from src.deployment.runtime import ONNXRuntime
from src.deployment.validate import ModelValidator

__all__ = [
    "ExportConfig",
    "QuantizationConfig",
    "RuntimeConfig",
    "ONNXExporter",
    "ModelQuantizer",
    "ONNXRuntime",
    "ModelValidator",
]
