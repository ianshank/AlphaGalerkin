"""Tests for ONNX export utilities.

Covers ONNXExporter initialization and configuration.
Actual export tests are skipped when ONNX is not available.
"""

from __future__ import annotations

from src.deployment.config import ExportConfig
from src.deployment.export_onnx import ONNXExporter


class TestONNXExporter:
    """Tests for ONNXExporter class."""

    def test_init_default_config(self) -> None:
        """Exporter initializes with default config when None provided."""
        exporter = ONNXExporter()
        assert isinstance(exporter.config, ExportConfig)
        assert exporter.config.opset_version == 17

    def test_init_custom_config(self) -> None:
        """Exporter uses provided config."""
        config = ExportConfig(opset_version=13)
        exporter = ONNXExporter(config=config)
        assert exporter.config.opset_version == 13

    def test_config_stored(self) -> None:
        """Config is accessible after init."""
        config = ExportConfig(
            opset_version=15,
            input_names=["input"],
            output_names=["output"],
        )
        exporter = ONNXExporter(config=config)
        assert exporter.config is config
        assert exporter.config.input_names == ["input"]
        assert exporter.config.output_names == ["output"]

    def test_multiple_exporters(self) -> None:
        """Multiple exporters can coexist with different configs."""
        e1 = ONNXExporter(ExportConfig(opset_version=13))
        e2 = ONNXExporter(ExportConfig(opset_version=17))
        assert e1.config.opset_version == 13
        assert e2.config.opset_version == 17

    def test_default_export_method(self) -> None:
        """Default export method is 'trace'."""
        exporter = ONNXExporter()
        assert exporter.config.export_method == "trace"

    def test_custom_dynamic_axes(self) -> None:
        """Exporter accepts config with dynamic axes."""
        config = ExportConfig(
            dynamic_axes={
                "board_state": {0: "batch", 2: "height", 3: "width"},
            }
        )
        exporter = ONNXExporter(config=config)
        assert "board_state" in exporter.config.dynamic_axes
