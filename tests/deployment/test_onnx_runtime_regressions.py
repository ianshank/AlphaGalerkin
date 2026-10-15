"""ONNX Runtime regressions behind hygiene B45's last five failures.

Once the ``ModelOutput`` dataclass was flattened (``export_adapters``), two
more defects still failed the real ``AlphaGalerkinModel`` export suites.
Neither depends on that model, so each is reproduced here on a minimal
tensor-returning one:

1. Dynamic quantization rejected every dynamo export with a Linear layer.
   ``ModelQuantizer`` handed the file to onnxruntime, whose Gemm->MatMul
   rewrite transposes ``transB`` weights in place but keeps the exporter's
   recorded ``graph.value_info``; the shape inference that follows fails with
   "Inferred shape and existing shape differ".
2. A dynamic axis exported from a size-1 sample was silently fixed at 1.
   torch.export 0/1-specializes it and torch.onnx's fallback bakes that 1
   into the graph, so a model that uses its batch size in a reshape fails in
   ONNX Runtime at batch 2 although the graph declares the axis dynamic.

These tests need onnx, onnxruntime and onnxscript (``[test-extras]``) and
skip per test without them. CI runs this file in ``test-extras``' blocking
"Run ONNX export/validate suites" step.

Planted defects (harden-a-guard), 4/4 killed, each by the named test(s):

1. ``ModelQuantizer`` hands onnxruntime the file path again (the pre-fix
   state): ``TestDynamicQuantizationOfADynamoExport::test_tensor_mlp
   _quantizes_and_runs``.
2. The model loaded but ``graph.value_info`` kept:
   ``TestDynamicQuantizationOfADynamoExport::test_tensor_mlp_quantizes_and
   _runs``.
3. The stripped model written back over the caller's file:
   ``TestDynamicQuantizationOfADynamoExport::test_quantization_leaves_the
   _input_file_untouched``.
4. The size-1 enlargement removed from the export path (the pre-fix state):
   ``TestSizeOneDynamicAxis::test_batch_dependent_model_runs_at_every_batch
   [2]`` and ``[4]``.

These killers skip without the ONNX stack; CI runs them where it is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from src.deployment.config import ExportConfig, QuantizationConfig, QuantizationMode
from src.deployment.export_onnx import ONNXExporter
from src.deployment.quantize import ModelQuantizer

pytestmark = pytest.mark.onnx_required

FEATURES = 8
ROWS = 3
HIDDEN = 16
COORDINATE_DIMS = 2
FREQUENCIES = 4
# The example size torch.export 0/1-specializes on a dynamic axis.
SIZE_ONE_SAMPLE_BATCH = 1
RUNTIME_BATCHES = (1, 2, 4)
QUANTIZATION_BATCH = 8
FP32_ABS_TOL = 1e-5
# Same INT8 bound as tests/integration/test_onnx_e2e.py: dynamic quantization
# of the linear layers costs precision, not correctness.
INT8_MAE_THRESHOLD = 0.5
INPUT_NAME = "input"
OUTPUT_NAME = "output"


class _MLP(nn.Module):
    """Linear layers on a 2-D input: the dynamo exporter emits ``Gemm`` with ``transB=1``."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, FEATURES)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _ExpandedGridModel(nn.Module):
    """``AlphaGalerkinModel``'s batch-pinning pattern, at toy size.

    ``create_grid_coordinates`` expands one coordinate grid to the batch and
    ``FourierBasis`` contracts it with ``einsum``. That einsum makes
    torch.export add ``Eq(batch, 1)`` as a guard and a runtime assert, which
    pins a size-1 sample's batch even under the ``backed_size_oblivious``
    mode torch.onnx enables. The ``Linear`` on the 3-D result then lowers to
    a reshape with the pinned 1 baked in, so ONNX Runtime fails at batch 2.
    A batch-dependent reshape alone does not reproduce this on torch 2.14,
    and neither does this model with ``@`` in place of the einsum: the
    oblivious mode keeps both batches dynamic.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("grid", torch.rand(1, ROWS, COORDINATE_DIMS))
        self.register_buffer("frequencies", torch.randn(COORDINATE_DIMS, FREQUENCIES))
        self.linear = nn.Linear(FEATURES + FREQUENCIES, FEATURES)

    def forward(self, x: Tensor) -> Tensor:
        grid = self.grid.expand(x.shape[0], -1, -1)
        positions = torch.einsum("bnd,df->bnf", grid, self.frequencies)
        return self.linear(torch.cat([x, positions], dim=-1))


def _onnxruntime() -> Any:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    return pytest.importorskip("onnxruntime")


def _export(model: nn.Module, sample: Tensor, path: Path, dynamic_batch: bool) -> Path:
    axes: dict[str, dict[int, str]] = (
        {INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}} if dynamic_batch else {}
    )
    config = ExportConfig(
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=axes,
        optimization_level="none",
    )
    return ONNXExporter(config).export(model.eval(), sample, path)


def _run(ort: Any, path: Path, x: Tensor) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run([OUTPUT_NAME], {INPUT_NAME: x.numpy()})[0]


class TestDynamicQuantizationOfADynamoExport:
    def test_tensor_mlp_quantizes_and_runs(self, tmp_path: Path) -> None:
        ort = _onnxruntime()
        torch.manual_seed(0)
        model = _MLP().eval()
        sample = torch.randn(QUANTIZATION_BATCH, FEATURES)
        fp32 = _export(model, sample, tmp_path / "mlp.onnx", dynamic_batch=False)

        config = QuantizationConfig(mode=QuantizationMode.DYNAMIC, per_channel=False)
        int8 = ModelQuantizer(config).quantize(fp32, tmp_path / "mlp_int8.onnx")

        with torch.no_grad():
            expected = model(sample).numpy()
        mae = float(np.mean(np.abs(_run(ort, int8, sample) - expected)))
        assert mae < INT8_MAE_THRESHOLD, f"INT8 MAE {mae:.3f} exceeds {INT8_MAE_THRESHOLD}"

    def test_quantization_leaves_the_input_file_untouched(self, tmp_path: Path) -> None:
        _onnxruntime()
        torch.manual_seed(0)
        fp32 = _export(
            _MLP(), torch.randn(QUANTIZATION_BATCH, FEATURES), tmp_path / "mlp.onnx", False
        )
        before = fp32.read_bytes()

        config = QuantizationConfig(mode=QuantizationMode.DYNAMIC, per_channel=False)
        ModelQuantizer(config).quantize(fp32, tmp_path / "mlp_int8.onnx")

        assert fp32.read_bytes() == before


@pytest.fixture(scope="module")
def batch_dependent_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[nn.Module, Path]:
    """``_ExpandedGridModel`` exported once, from a batch-1 sample, batch declared dynamic."""
    _onnxruntime()
    torch.manual_seed(0)
    model = _ExpandedGridModel().eval()
    sample = torch.randn(SIZE_ONE_SAMPLE_BATCH, ROWS, FEATURES)
    path = tmp_path_factory.mktemp("batch_dependent") / "model.onnx"
    return model, _export(model, sample, path, dynamic_batch=True)


class TestSizeOneDynamicAxis:
    @pytest.mark.parametrize("batch", RUNTIME_BATCHES)
    def test_batch_dependent_model_runs_at_every_batch(
        self, batch: int, batch_dependent_export: tuple[nn.Module, Path]
    ) -> None:
        ort = _onnxruntime()
        model, path = batch_dependent_export
        torch.manual_seed(batch)
        x = torch.randn(batch, ROWS, FEATURES)

        with torch.no_grad():
            expected = model(x).numpy()
        np.testing.assert_allclose(_run(ort, path, x), expected, atol=FP32_ABS_TOL)
