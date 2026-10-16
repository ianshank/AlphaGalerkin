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

A third finding, B50, is not a failure yet but would become one on a torch
upgrade: every export handed ``torch.onnx.export`` its ``dynamic_axes``,
which the dynamo path converts through a function deprecated for removal,
warning each time. ``TestNoDeprecatedDynamicAxesRoute`` fails if either
warning (``tests/support/onnx_export.py``) is emitted, on a tensor model and
on ``_GridHeads``, which returns a ``ModelOutput`` and so exports through
``TupleOutputWrapper`` like ``AlphaGalerkinModel``. Its premise test fails
if torch stops emitting the warnings when given ``dynamic_axes``, which
would otherwise leave those tests passing on nothing.

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
   [2]`` and ``[4]``; since B50 also ``TestNoDeprecatedDynamicAxesRoute::
   test_dataclass_model_matches_pytorch_at_every_batch[2]`` and ``[4]``.

B50, 3 of its 9 planted defects killed here (the rest in
``test_export_adapters.py``'s docstring, which numbers them 13-20):

5. ``dynamic_axes`` passed again instead of ``dynamic_shapes`` (the pre-fix
   call): ``TestNoDeprecatedDynamicAxesRoute::test_tensor_export_takes_no
   _deprecated_route`` and ``...::test_dataclass_export_takes_no_deprecated
   _route``.
6. Every input passed as static: ``TestNoDeprecatedDynamicAxesRoute::test
   _dataclass_model_matches_pytorch_at_every_batch[1]`` and ``[4]``.
7. A warning marker reworded, as a torch release might: ``TestNoDeprecated
   DynamicAxesRoute::test_premise_torch_flags_dynamic_axes_with_both
   _messages``.

These killers skip without the ONNX stack; CI runs them where it is installed.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from src.deployment.config import ExportConfig, QuantizationConfig, QuantizationMode
from src.deployment.export_onnx import ONNXExporter
from src.deployment.quantize import ModelQuantizer
from src.modeling.model import ModelOutput
from tests.support.onnx_export import (
    DEPRECATED_DYNAMIC_AXES_MESSAGES,
    deprecated_dynamic_axes_warnings,
)

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
# ModelOutput's two tensor fields, in the order the exporter flattens them.
POLICY_NAME = "policy"
VALUE_NAME = "value"
# A dynamic batch the exporter traces as it is: neither 1 nor the enlarged 2.
UNENLARGED_SAMPLE_BATCH = 3


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


class _GridHeads(_ExpandedGridModel):
    """``_ExpandedGridModel`` returning a ``ModelOutput``: exported as ``AlphaGalerkinModel`` is.

    The dataclass output sends it through ``TupleOutputWrapper``, whose
    ``forward(x, *args)`` signature is what ``dynamic_shapes`` must bind to; a
    batch-1 sample is enlarged to 2; and ``dynamic_axes`` names the input and
    both outputs, like the default ``ExportConfig``.
    """

    def forward(self, x: Tensor) -> ModelOutput:  # type: ignore[override]
        features = super().forward(x)
        return ModelOutput(
            policy_logits=features.flatten(1),
            value=torch.tanh(features.mean(dim=(1, 2))).unsqueeze(-1),
        )


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


def _export_heads(model: nn.Module, sample: Tensor, path: Path) -> Path:
    """Export a ``ModelOutput`` model the way the default ``ExportConfig`` exports one."""
    config = ExportConfig(
        input_names=[INPUT_NAME],
        output_names=[POLICY_NAME, VALUE_NAME],
        dynamic_axes={name: {0: "batch"} for name in (INPUT_NAME, POLICY_NAME, VALUE_NAME)},
        optimization_level="none",
    )
    return ONNXExporter(config).export(model.eval(), sample, path)


@pytest.fixture(scope="module")
def heads_export(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[nn.Module, Path, list[str]]:
    """``_GridHeads`` exported once from a batch-1 sample, and its deprecated-route warnings."""
    _onnxruntime()
    torch.manual_seed(0)
    model = _GridHeads().eval()
    sample = torch.randn(SIZE_ONE_SAMPLE_BATCH, ROWS, FEATURES)
    path = tmp_path_factory.mktemp("heads") / "model.onnx"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _export_heads(model, sample, path)
    return model, path, deprecated_dynamic_axes_warnings(caught)


class TestNoDeprecatedDynamicAxesRoute:
    """The exporter passes ``dynamic_shapes``, never torch's deprecated ``dynamic_axes`` route."""

    def test_premise_torch_flags_dynamic_axes_with_both_messages(self, tmp_path: Path) -> None:
        """Without this, a reworded torch warning would leave the tests below passing vacuously."""
        _onnxruntime()
        torch.manual_seed(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.onnx.export(
                _MLP().eval(),
                (torch.randn(UNENLARGED_SAMPLE_BATCH, FEATURES),),
                str(tmp_path / "direct.onnx"),
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                dynamic_axes={INPUT_NAME: {0: "batch"}},
                dynamo=True,
            )
        flagged = deprecated_dynamic_axes_warnings(caught)
        assert len(flagged) == len(DEPRECATED_DYNAMIC_AXES_MESSAGES), (
            "torch no longer emits one warning per DEPRECATED_DYNAMIC_AXES_MESSAGES entry "
            f"when given dynamic_axes (got {flagged}). If it reworded them, update the "
            "markers; if it removed the conversion, this premise and its guards can go."
        )

    def test_tensor_export_takes_no_deprecated_route(self, tmp_path: Path) -> None:
        _onnxruntime()
        torch.manual_seed(0)
        sample = torch.randn(UNENLARGED_SAMPLE_BATCH, FEATURES)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _export(_MLP(), sample, tmp_path / "mlp.onnx", dynamic_batch=True)
        assert deprecated_dynamic_axes_warnings(caught) == []

    def test_dataclass_export_takes_no_deprecated_route(
        self, heads_export: tuple[nn.Module, Path, list[str]]
    ) -> None:
        _, _, flagged = heads_export
        assert flagged == []

    @pytest.mark.parametrize("batch", RUNTIME_BATCHES)
    def test_dataclass_model_matches_pytorch_at_every_batch(
        self, batch: int, heads_export: tuple[nn.Module, Path, list[str]]
    ) -> None:
        ort = _onnxruntime()
        model, path, _ = heads_export
        torch.manual_seed(batch)
        x = torch.randn(batch, ROWS, FEATURES)

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        policy, value = session.run([POLICY_NAME, VALUE_NAME], {INPUT_NAME: x.numpy()})
        with torch.no_grad():
            expected = model(x)
        np.testing.assert_allclose(policy, expected.policy_logits.numpy(), atol=FP32_ABS_TOL)
        np.testing.assert_allclose(value, expected.value.numpy(), atol=FP32_ABS_TOL)
