"""End-to-end ONNX export + runtime integration test.

Closes the B2 deliverable from the implementation plan: a single
test that exercises the full PyTorch → ONNX → ONNX Runtime pipeline,
including dynamic batch shapes and a quantization branch.

The test is gated by the existing ``onnx_required`` pytest marker (so
machines without ``onnx``/``onnxruntime`` installed skip cleanly via
``pytest.importorskip``).  All thresholds (accuracy delta, timing
budgets) are defined as module-level constants so they can be tuned
without spelunking the test body — never inline magic numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

# Skip the entire module when onnx infrastructure is unavailable.
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from src.deployment.config import ExportConfig  # noqa: E402  — after importorskip
from src.deployment.export_onnx import ONNXExporter  # noqa: E402

pytestmark = pytest.mark.onnx_required


# ---------------------------------------------------------------------------
# Tunables — surfaced as constants so reviewers see thresholds in one place
# ---------------------------------------------------------------------------


# Maximum allowed absolute difference between PyTorch and ONNX outputs
# in FP32 mode.  ONNX exporters frequently round-trip with sub-float
# differences caused by graph reordering; we accept up to 1e-4.
FP32_ABS_TOL = 1e-4

# Maximum acceptable mean-absolute error for INT8-quantised models.
# A relaxed threshold reflects the inherent precision loss of dynamic
# quantization on linear layers.
INT8_MAE_THRESHOLD = 0.5

# Static shapes used for the smoke test
INPUT_DIM = 32
OUTPUT_DIM = 8
BATCH_SIZES_TO_TEST = (1, 4, 32)


# ---------------------------------------------------------------------------
# Tiny model used by all tests in this module
# ---------------------------------------------------------------------------


class _SimpleMLP(nn.Module):
    """Two-layer MLP with no batch-dependent ops — ONNX-friendly."""

    def __init__(self, in_dim: int = INPUT_DIM, out_dim: int = OUTPUT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.fixture()
def model() -> _SimpleMLP:
    torch.manual_seed(42)
    m = _SimpleMLP()
    m.eval()
    return m


@pytest.fixture()
def sample_input() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(1, INPUT_DIM)


# ---------------------------------------------------------------------------
# Round-trip: FP32 export and FP32 inference
# ---------------------------------------------------------------------------


def _static_export_config() -> ExportConfig:
    """Config with custom IO names and an empty ``dynamic_axes`` map.

    The default :class:`ExportConfig` ships AlphaGalerkin-specific
    axis names (``board_state``/``policy``/``value``).  Tests using a
    plain MLP must override both the IO names and the dynamic-axes
    map so the new dynamo exporter does not try to apply mappings
    against non-existent inputs.
    """
    return ExportConfig(
        opset_version=17,
        export_method="trace",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={},
    )


def _dynamic_export_config() -> ExportConfig:
    return ExportConfig(
        opset_version=17,
        export_method="trace",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
    )


class TestONNXFP32Roundtrip:
    """PyTorch → ONNX FP32 → ONNX Runtime FP32."""

    def test_export_creates_valid_onnx(
        self,
        model: _SimpleMLP,
        sample_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        exporter = ONNXExporter(_static_export_config())
        out = tmp_path / "model.onnx"
        exporter.export(model=model, sample_input=sample_input, output_path=out)
        assert out.exists()
        # ONNX integrity check
        onnx_model = onnx.load(str(out))
        onnx.checker.check_model(onnx_model)

    def test_runtime_matches_pytorch(
        self,
        model: _SimpleMLP,
        sample_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "model.onnx"
        ONNXExporter(_static_export_config()).export(model, sample_input, out)

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            torch_out = model(sample_input).cpu().numpy()
        ort_out = session.run(["output"], {"input": sample_input.cpu().numpy()})[0]

        assert torch_out.shape == ort_out.shape
        np.testing.assert_allclose(torch_out, ort_out, atol=FP32_ABS_TOL)

    @pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
    def test_dynamic_batch_dim(
        self,
        model: _SimpleMLP,
        tmp_path: Path,
        batch_size: int,
    ) -> None:
        """Verify dynamic batch axis works for batch=1, 4, 32."""
        sample = torch.randn(1, INPUT_DIM)
        out = tmp_path / "model_dyn.onnx"
        ONNXExporter(_dynamic_export_config()).export(model, sample, out)

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = np.random.randn(batch_size, INPUT_DIM).astype(np.float32)
        y = session.run(["output"], {"input": x})[0]
        assert y.shape == (batch_size, OUTPUT_DIM)
        # Compare against the eager PyTorch model — accuracy must hold for any batch
        with torch.no_grad():
            torch_out = model(torch.from_numpy(x)).cpu().numpy()
        np.testing.assert_allclose(y, torch_out, atol=FP32_ABS_TOL)


# ---------------------------------------------------------------------------
# Quantized round-trip
# ---------------------------------------------------------------------------


class TestONNXQuantization:
    """Dynamic INT8 quantization via onnxruntime.quantization."""

    def test_quantized_runs_and_close_enough(
        self,
        model: _SimpleMLP,
        sample_input: torch.Tensor,
        tmp_path: Path,
    ) -> None:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            from onnxruntime.quantization.shape_inference import quant_pre_process
        except ImportError:
            pytest.skip("onnxruntime.quantization not available in this environment")

        fp32_path = tmp_path / "model_fp32.onnx"
        prepared_path = tmp_path / "model_prepared.onnx"
        int8_path = tmp_path / "model_int8.onnx"
        # Trace at the same batch size we'll evaluate at — keeps
        # quantization shape-inference happy regardless of dynamic-axes
        # metadata, which is brittle on the dynamo exporter.
        eval_batch = 8
        sample_for_quant = torch.randn(eval_batch, INPUT_DIM)
        ONNXExporter(_static_export_config()).export(
            model, sample_for_quant, fp32_path
        )
        quant_pre_process(str(fp32_path), str(prepared_path))

        quantize_dynamic(
            str(prepared_path),
            str(int8_path),
            weight_type=QuantType.QInt8,
        )
        assert int8_path.exists()

        # Compare quantized ONNX vs full-precision PyTorch
        x = np.random.randn(eval_batch, INPUT_DIM).astype(np.float32)
        with torch.no_grad():
            torch_out = model(torch.from_numpy(x)).cpu().numpy()
        session = ort.InferenceSession(
            str(int8_path), providers=["CPUExecutionProvider"]
        )
        ort_out = session.run(["output"], {"input": x})[0]

        # INT8 has noticeable but bounded MAE
        mae = float(np.mean(np.abs(torch_out - ort_out)))
        assert mae < INT8_MAE_THRESHOLD, f"MAE {mae:.3f} exceeds {INT8_MAE_THRESHOLD}"
