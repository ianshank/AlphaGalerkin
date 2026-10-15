"""``ModelQuantizer`` drops ``graph.value_info`` before dynamic quantization (hygiene B45).

Defect class: torch.onnx's dynamo exporter, the default since torch 2.9,
records ``graph.value_info`` for weights too, and onnxruntime's dynamic
quantizer rewrites each ``Gemm`` with ``transB=1`` into a ``MatMul``,
transposing the weight in place while keeping that record. Its next shape
inference rejects the stale shape ("Inferred shape and existing shape
differ"), so every exported ``Linear`` on a 2-D input failed to quantize.
Static quantization does no such rewrite and passed, so it is unchanged.

The end-to-end reproduction runs real onnxruntime in
``tests/deployment/test_onnx_runtime_regressions.py``. These tests pin the
contract with ``onnx`` and ``onnxruntime`` replaced by stubs in
``sys.modules``, so they run without either installed, as in the fast lane
and the ``coverage-gates`` job.

Planted defects (harden-a-guard), 4/4 killed, each by the named test(s):

1. The file path handed to onnxruntime again (the pre-fix state):
   ``TestDynamicQuantization::test_onnxruntime_receives_the_model_without
   _value_info``.
2. The model loaded but ``value_info`` kept:
   ``TestDynamicQuantization::test_onnxruntime_receives_the_model_without
   _value_info``.
3. The ``value_info_stripped`` event logged with a wrong count:
   ``TestDynamicQuantization::test_the_strip_is_logged_with_its_count``.
4. Static quantization stripped too:
   ``TestStaticQuantization::test_onnxruntime_still_receives_the_path``.

All killers run under ``-m "not gpu_required and not fem_required"``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.deployment.config import QuantizationConfig, QuantizationMode
from src.deployment.quantize import ModelQuantizer

VALUE_INFO_ENTRIES = 3
CALIBRATION_SHAPE = (1, 4)
CALIBRATION_SAMPLES = 2
DYNAMIC = QuantizationConfig(mode=QuantizationMode.DYNAMIC, per_channel=False)
STATIC = QuantizationConfig(mode=QuantizationMode.STATIC, per_channel=False)


class _RecordingLogger:
    """Stands in for the quantizer's bound structlog logger."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def _record(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    info = debug = warning = error = _record


class _OnnxStack:
    """Stub ``onnx`` and ``onnxruntime.quantization``, recording what onnxruntime receives."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.loaded = SimpleNamespace(
            graph=SimpleNamespace(value_info=[f"shape_{i}" for i in range(VALUE_INFO_ENTRIES)])
        )
        self.load_paths: list[str] = []
        self.calls: dict[str, tuple[Any, str]] = {}
        stack = self

        def load(path: str) -> SimpleNamespace:
            stack.load_paths.append(path)
            return stack.loaded

        def quantizer(kind: str) -> Any:
            def quantize(model_input: Any, model_output: str, **_: Any) -> None:
                stack.calls[kind] = (model_input, model_output)
                Path(model_output).write_bytes(b"quantized")

            return quantize

        onnx = ModuleType("onnx")
        onnx.load = load  # type: ignore[attr-defined]
        quantization = ModuleType("onnxruntime.quantization")
        quantization.quantize_dynamic = quantizer("dynamic")  # type: ignore[attr-defined]
        quantization.quantize_static = quantizer("static")  # type: ignore[attr-defined]
        quantization.QuantType = SimpleNamespace(QInt8="QInt8", QUInt8="QUInt8", QInt4="QInt4")  # type: ignore[attr-defined]
        quantization.QuantFormat = SimpleNamespace(QDQ="QDQ")  # type: ignore[attr-defined]
        quantization.CalibrationMethod = SimpleNamespace(  # type: ignore[attr-defined]
            MinMax="MinMax", Entropy="Entropy", Percentile="Percentile"
        )
        onnxruntime = ModuleType("onnxruntime")
        onnxruntime.quantization = quantization  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "onnx", onnx)
        monkeypatch.setitem(sys.modules, "onnxruntime", onnxruntime)
        monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quantization)


@pytest.fixture()
def onnx_stack(monkeypatch: pytest.MonkeyPatch) -> _OnnxStack:
    return _OnnxStack(monkeypatch)


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"exported model")
    return path


class TestDynamicQuantization:
    def test_onnxruntime_receives_the_model_without_value_info(
        self, onnx_stack: _OnnxStack, tmp_path: Path
    ) -> None:
        model_path = _model_file(tmp_path)
        ModelQuantizer(DYNAMIC).quantize(model_path, tmp_path / "quantized.onnx")

        model_input, model_output = onnx_stack.calls["dynamic"]
        assert model_input is onnx_stack.loaded
        assert list(model_input.graph.value_info) == []
        assert onnx_stack.load_paths == [str(model_path)]
        assert model_output == str(tmp_path / "quantized.onnx")

    def test_the_strip_is_logged_with_its_count(
        self, onnx_stack: _OnnxStack, tmp_path: Path
    ) -> None:
        model_path = _model_file(tmp_path)
        quantizer = ModelQuantizer(DYNAMIC)
        logger = _RecordingLogger()
        quantizer._logger = logger  # type: ignore[assignment]
        quantizer.quantize(model_path, tmp_path / "quantized.onnx")

        events = [fields for event, fields in logger.events if event == "value_info_stripped"]
        assert events == [{"input_path": str(model_path), "entries_removed": VALUE_INFO_ENTRIES}]


class TestStaticQuantization:
    def test_onnxruntime_still_receives_the_path(
        self, onnx_stack: _OnnxStack, tmp_path: Path
    ) -> None:
        """Static quantization rewrites no Gemm, passed before the fix, and is untouched."""
        model_path = _model_file(tmp_path)
        calibration = (
            np.zeros(CALIBRATION_SHAPE, dtype=np.float32) for _ in range(CALIBRATION_SAMPLES)
        )
        ModelQuantizer(STATIC).quantize(model_path, tmp_path / "quantized.onnx", calibration)

        model_input, _ = onnx_stack.calls["static"]
        assert model_input == str(model_path)
        assert onnx_stack.load_paths == []
