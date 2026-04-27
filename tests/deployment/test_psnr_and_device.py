"""Track A — PSNR gate and GPU-primary device routing.

Tests for the PSNR validation gate and device-routing additions to
:mod:`src.deployment`.

Covers:

* New ``ExportConfig.export_device`` and ``DeploymentConfig.validation_device``
  Pydantic fields with ``Literal["cuda","cpu","auto"]`` validation.
* New ``DeploymentConfig.accuracy_threshold_psnr_db`` field with bounded
  validator.
* ``ModelValidator`` accepts ``accuracy_threshold_psnr_db`` and surfaces
  PSNR + ``psnr_passed`` on :class:`ValidationResult`.
* The ``compute_psnr_db`` helper handles edge cases (empty input,
  zero MSE, constant reference).

These tests do not require CUDA and run cleanly on CPU CI runners.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from src.deployment.config import (
    DeploymentConfig,
    ExportConfig,
)
from src.deployment.validate import (
    ModelValidator,
    ValidationResult,
    compute_psnr_db,
)

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestExportDeviceField:
    def test_default_is_cuda_for_gpu_primary_rig(self) -> None:
        cfg = ExportConfig()
        assert cfg.export_device == "cuda"

    @pytest.mark.parametrize("device", ["cuda", "cpu", "auto"])
    def test_accepts_literal_values(self, device: str) -> None:
        cfg = ExportConfig(export_device=device)  # type: ignore[arg-type]
        assert cfg.export_device == device

    def test_rejects_unknown_device(self) -> None:
        with pytest.raises(ValidationError):
            ExportConfig(export_device="tpu")  # type: ignore[arg-type]


class TestValidationDeviceField:
    def test_default_is_auto_for_ci_compat(self) -> None:
        cfg = DeploymentConfig()
        assert cfg.validation_device == "auto"

    @pytest.mark.parametrize("device", ["cuda", "cpu", "auto"])
    def test_accepts_literal_values(self, device: str) -> None:
        cfg = DeploymentConfig(validation_device=device)  # type: ignore[arg-type]
        assert cfg.validation_device == device

    def test_rejects_unknown_device(self) -> None:
        with pytest.raises(ValidationError):
            DeploymentConfig(validation_device="rocm")  # type: ignore[arg-type]


class TestAccuracyThresholdPsnrField:
    def test_default_is_35_db(self) -> None:
        cfg = DeploymentConfig()
        assert cfg.accuracy_threshold_psnr_db == 35.0

    def test_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            DeploymentConfig(accuracy_threshold_psnr_db=-1.0)

    def test_zero_is_accepted_no_threshold(self) -> None:
        cfg = DeploymentConfig(accuracy_threshold_psnr_db=0.0)
        assert cfg.accuracy_threshold_psnr_db == 0.0


# ---------------------------------------------------------------------------
# compute_psnr_db helper
# ---------------------------------------------------------------------------


class TestComputePsnrDb:
    def test_empty_input_returns_none(self) -> None:
        assert compute_psnr_db([]) is None

    def test_perfect_parity_returns_none(self) -> None:
        ref = np.array([1.0, 2.0, 3.0])
        # MSE is 0; helper signals "undefined" so the strict tolerance
        # gate carries the assertion.
        assert compute_psnr_db([(ref, ref.copy())]) is None

    def test_known_psnr_round_trip(self) -> None:
        # Reference range [0, 10], MSE = 1.0
        # PSNR = 10 * log10(100 / 1) = 20 dB
        ref = np.array([0.0, 5.0, 10.0])
        candidate = ref + 1.0
        psnr = compute_psnr_db([(ref, candidate)])
        assert psnr is not None
        assert psnr == pytest.approx(20.0, abs=1e-9)

    def test_constant_reference_uses_floor_peak(self) -> None:
        # peak == 0 would divide by zero; the helper clamps to 1e-12.
        ref = np.full(5, 7.0)
        candidate = ref + 0.5
        psnr = compute_psnr_db([(ref, candidate)])
        # We don't assert a specific value — just that it's finite and
        # the helper didn't raise.
        assert psnr is not None and np.isfinite(psnr)

    def test_aggregation_across_pairs(self) -> None:
        # Two identical pairs => same PSNR as a single pair.
        ref = np.array([0.0, 10.0])
        candidate = np.array([1.0, 9.0])
        single = compute_psnr_db([(ref, candidate)])
        double = compute_psnr_db([(ref, candidate), (ref, candidate)])
        assert single is not None and double is not None
        assert single == pytest.approx(double, abs=1e-9)


# ---------------------------------------------------------------------------
# ModelValidator surface (no real ONNX runtime required)
# ---------------------------------------------------------------------------


class TestModelValidatorPsnrSurface:
    def test_constructor_accepts_psnr_threshold(self) -> None:
        v = ModelValidator(accuracy_threshold_psnr_db=42.0)
        assert v.accuracy_threshold_psnr_db == 42.0

    def test_constructor_default_psnr_threshold_is_none(self) -> None:
        v = ModelValidator()
        assert v.accuracy_threshold_psnr_db is None

    def test_validation_result_has_psnr_fields(self) -> None:
        # Smoke: dataclass surface is forward-compatible.
        result = ValidationResult(
            passed=True,
            max_policy_diff=0.0,
            max_value_diff=0.0,
            mean_policy_diff=0.0,
            mean_value_diff=0.0,
            pytorch_time_ms=0.0,
            onnx_time_ms=0.0,
            speedup_ratio=1.0,
            n_samples_tested=0,
            failed_samples=0,
        )
        # Defaults preserve backwards compatibility.
        assert result.policy_psnr_db is None
        assert result.value_psnr_db is None
        assert result.psnr_threshold_db is None
        assert result.psnr_passed is None


# ---------------------------------------------------------------------------
# create_sample_input device routing
# ---------------------------------------------------------------------------


class TestSampleInputDeviceRouting:
    def test_cpu_export_device_routes_to_cpu(self) -> None:
        from src.deployment.export_onnx import ONNXExporter

        cfg = ExportConfig(export_device="cpu")
        exporter = ONNXExporter(cfg)
        sample = exporter.create_sample_input(batch_size=1, board_size=9, channels=4)
        assert sample.device.type == "cpu"

    def test_explicit_device_override_wins(self) -> None:
        # Even with a CUDA-default config, an explicit device kwarg must
        # be honored — required for unit tests on CPU CI runners.
        from src.deployment.export_onnx import ONNXExporter

        cfg = ExportConfig(export_device="cuda")  # default
        exporter = ONNXExporter(cfg)
        sample = exporter.create_sample_input(
            batch_size=1, board_size=9, channels=4, device=torch.device("cpu")
        )
        assert sample.device.type == "cpu"

    def test_cuda_export_device_fails_loud_when_cuda_missing(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA is available; cannot test fail-loud path")
        from src.deployment.export_onnx import ONNXExporter

        cfg = ExportConfig(export_device="cuda")
        exporter = ONNXExporter(cfg)
        with pytest.raises(RuntimeError, match="CUDA is not"):
            exporter.create_sample_input(batch_size=1, board_size=9, channels=4)
