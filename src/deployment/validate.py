"""Model validation utilities for deployment.

This module provides utilities for validating exported ONNX models
against the original PyTorch models to ensure correctness.

Features:
    - Numerical accuracy comparison
    - Shape validation
    - Performance comparison
    - Batch validation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from torch import Tensor, nn

from src.constants import DEFAULT_BOARD_SIZES

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


def _compute_psnr_db(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float | None:
    """Aggregate PSNR (dB) across a list of (reference, candidate) pairs.

    PSNR = 10 * log10(peak^2 / mse) where ``peak`` is the dynamic range
    of the reference (max - min, clamped to a small positive floor to
    avoid division by zero on constant references) and ``mse`` is the
    mean squared error.  Returns ``None`` when no pairs are supplied or
    when the aggregate MSE is exactly zero (perfect parity), in which
    case the strict tolerance gate carries the assertion.
    """
    if not pairs:
        return None
    sq_err: list[float] = []
    peak_max = -np.inf
    peak_min = np.inf
    for reference, candidate in pairs:
        sq_err.append(float(np.mean((reference - candidate) ** 2)))
        peak_max = max(peak_max, float(np.max(reference)))
        peak_min = min(peak_min, float(np.min(reference)))
    mse = float(np.mean(sq_err))
    if mse <= 0.0:
        return None
    peak = max(peak_max - peak_min, 1e-12)
    return 10.0 * float(np.log10((peak**2) / mse))


@dataclass
class ValidationResult:
    """Result from model validation.

    The PSNR fields are populated only when ``policy_psnr_db`` /
    ``value_psnr_db`` can be computed (i.e. the reference output is not
    identically zero).  ``policy_psnr_db is None`` means the numeric
    floor was hit and the strict tolerance gate carries the assertion.
    Backwards compatible: legacy callers that only inspect
    ``passed`` / ``*_diff`` fields are unaffected.
    """

    passed: bool
    max_policy_diff: float
    max_value_diff: float
    mean_policy_diff: float
    mean_value_diff: float
    pytorch_time_ms: float
    onnx_time_ms: float
    speedup_ratio: float
    n_samples_tested: int
    failed_samples: int
    error_message: str | None = None
    policy_psnr_db: float | None = None
    value_psnr_db: float | None = None
    psnr_threshold_db: float | None = None
    psnr_passed: bool | None = None


class ModelValidator:
    """Validates ONNX models against PyTorch models.

    Compares outputs between PyTorch and ONNX Runtime to ensure
    the exported model produces correct results.

    Attributes:
        tolerance: Maximum allowed difference between outputs.
        relative_tolerance: Relative tolerance for comparison.

    """

    def __init__(
        self,
        tolerance: float = 1e-5,
        relative_tolerance: float = 1e-4,
        *,
        accuracy_threshold_psnr_db: float | None = None,
    ) -> None:
        """Initialize validator.

        Args:
            tolerance: Absolute tolerance for comparison.
            relative_tolerance: Relative tolerance for comparison.
            accuracy_threshold_psnr_db: Optional minimum PSNR (dB) for
                the ONNX outputs vs. the PyTorch reference.  When
                provided, :meth:`validate` populates the PSNR fields on
                :class:`ValidationResult` and gates ``psnr_passed``
                accordingly.  ``None`` (the default) preserves the
                pre-PR behaviour: PSNR is not measured.

        """
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.accuracy_threshold_psnr_db = accuracy_threshold_psnr_db

        self._logger = structlog.get_logger(__name__).bind(
            tolerance=tolerance,
            relative_tolerance=relative_tolerance,
        )

    def validate(
        self,
        pytorch_model: nn.Module,
        onnx_path: str | Path,
        test_inputs: list[Tensor] | Tensor,
        device: torch.device | str = "auto",
    ) -> ValidationResult:
        """Validate ONNX model against PyTorch model.

        Args:
            pytorch_model: Original PyTorch model.
            onnx_path: Path to ONNX model.
            test_inputs: Test input tensors.
            device: Device for PyTorch reference inference.  Strings
                ``"cuda"``, ``"cpu"``, ``"auto"`` are routed through
                :func:`src.poc.device.resolve_device` (``"cuda"`` fails
                loud if unavailable; ``"auto"`` falls back silently).
                The legacy default ``"cpu"`` would silently mask GPU
                regressions on the user's training rig, so the new
                default is ``"auto"``.  Explicit ``torch.device``
                objects are respected as-is.

        Returns:
            ValidationResult with comparison metrics.

        """
        from src.deployment.runtime import ONNXRuntime

        onnx_path = Path(onnx_path)

        if isinstance(device, str):
            from src.poc.device import resolve_device

            device = resolve_device(device, context="ModelValidator.validate")

        # Prepare test inputs
        if isinstance(test_inputs, Tensor):
            test_inputs = [test_inputs]

        # Initialize ONNX Runtime
        runtime = ONNXRuntime(onnx_path)

        # Move PyTorch model to device and eval mode
        pytorch_model = pytorch_model.to(device)
        pytorch_model.eval()

        # Comparison metrics
        policy_diffs: list[float] = []
        value_diffs: list[float] = []
        pytorch_times: list[float] = []
        onnx_times: list[float] = []
        failed_samples = 0
        # Per-sample (reference, candidate) pairs for PSNR aggregation.
        # Populated only when self.accuracy_threshold_psnr_db is not None
        # to keep the legacy path allocation-free.
        psnr_pairs_policy: list[tuple[np.ndarray, np.ndarray]] = []
        psnr_pairs_value: list[tuple[np.ndarray, np.ndarray]] = []

        self._logger.info(
            "starting_validation",
            n_samples=len(test_inputs),
            onnx_path=str(onnx_path),
        )

        for i, test_input in enumerate(test_inputs):
            try:
                # Ensure input is on correct device
                input_tensor = test_input.to(device)

                # PyTorch inference
                import time

                start = time.perf_counter()
                with torch.no_grad():
                    pytorch_output = pytorch_model(input_tensor)
                pytorch_time = (time.perf_counter() - start) * 1000
                pytorch_times.append(pytorch_time)

                # Convert PyTorch output to numpy
                if hasattr(pytorch_output, "policy_logits"):
                    pytorch_policy = pytorch_output.policy_logits.cpu().numpy()
                    pytorch_value = pytorch_output.value.cpu().numpy()
                elif isinstance(pytorch_output, tuple):
                    pytorch_policy = pytorch_output[0].cpu().numpy()
                    pytorch_value = (
                        pytorch_output[1].cpu().numpy()
                        if len(pytorch_output) > 1
                        else np.array([0.0])
                    )
                else:
                    pytorch_policy = pytorch_output.cpu().numpy()
                    pytorch_value = np.array([0.0])

                # ONNX Runtime inference
                input_numpy = input_tensor.cpu().numpy()
                onnx_result = runtime.run(input_numpy)
                onnx_times.append(onnx_result.inference_time_ms)

                # Compare outputs
                policy_diff = np.abs(pytorch_policy - onnx_result.policy).max()
                policy_diffs.append(policy_diff)

                if len(pytorch_value.shape) > 0 and len(onnx_result.value.shape) > 0:
                    value_diff = np.abs(pytorch_value - onnx_result.value).max()
                else:
                    value_diff = 0.0
                value_diffs.append(value_diff)

                # Check if sample passed
                if policy_diff > self.tolerance or value_diff > self.tolerance:
                    failed_samples += 1
                    self._logger.warning(
                        "sample_exceeded_tolerance",
                        sample=i,
                        policy_diff=policy_diff,
                        value_diff=value_diff,
                    )

                # PSNR aggregation (only when threshold gate is active).
                if self.accuracy_threshold_psnr_db is not None:
                    psnr_pairs_policy.append((pytorch_policy, onnx_result.policy))
                    if len(pytorch_value.shape) > 0 and len(onnx_result.value.shape) > 0:
                        psnr_pairs_value.append((pytorch_value, onnx_result.value))

            except Exception as e:
                self._logger.error(
                    "validation_error",
                    sample=i,
                    error=str(e),
                )
                failed_samples += 1
                policy_diffs.append(float("inf"))
                value_diffs.append(float("inf"))

        # Close runtime
        runtime.close()

        # Compute final metrics
        passed = failed_samples == 0
        max_policy_diff = max(policy_diffs) if policy_diffs else float("inf")
        max_value_diff = max(value_diffs) if value_diffs else float("inf")
        mean_policy_diff = np.mean([d for d in policy_diffs if d != float("inf")])
        mean_value_diff = np.mean([d for d in value_diffs if d != float("inf")])
        avg_pytorch_time = np.mean(pytorch_times) if pytorch_times else 0
        avg_onnx_time = np.mean(onnx_times) if onnx_times else 0
        speedup = avg_pytorch_time / avg_onnx_time if avg_onnx_time > 0 else 0

        # Aggregate PSNR across collected (reference, candidate) pairs.
        policy_psnr_db: float | None = None
        value_psnr_db: float | None = None
        psnr_passed: bool | None = None
        if self.accuracy_threshold_psnr_db is not None:
            policy_psnr_db = _compute_psnr_db(psnr_pairs_policy)
            value_psnr_db = _compute_psnr_db(psnr_pairs_value)
            # An output stream "passes" PSNR if either (a) it is undefined
            # because the reference was zero / no pairs were collected
            # (in which case the strict tolerance gate already covers
            # correctness) or (b) it meets the threshold.
            policy_ok = policy_psnr_db is None or policy_psnr_db >= self.accuracy_threshold_psnr_db
            value_ok = value_psnr_db is None or value_psnr_db >= self.accuracy_threshold_psnr_db
            psnr_passed = policy_ok and value_ok
            passed = passed and psnr_passed

        result = ValidationResult(
            passed=passed,
            max_policy_diff=max_policy_diff,
            max_value_diff=max_value_diff,
            mean_policy_diff=mean_policy_diff,
            mean_value_diff=mean_value_diff,
            pytorch_time_ms=avg_pytorch_time,
            onnx_time_ms=avg_onnx_time,
            speedup_ratio=speedup,
            n_samples_tested=len(test_inputs),
            failed_samples=failed_samples,
            policy_psnr_db=policy_psnr_db,
            value_psnr_db=value_psnr_db,
            psnr_threshold_db=self.accuracy_threshold_psnr_db,
            psnr_passed=psnr_passed,
        )

        self._logger.info(
            "validation_completed",
            passed=passed,
            max_policy_diff=max_policy_diff,
            max_value_diff=max_value_diff,
            speedup=f"{speedup:.2f}x",
            policy_psnr_db=policy_psnr_db,
            value_psnr_db=value_psnr_db,
        )

        return result

    def validate_shapes(
        self,
        onnx_path: str | Path,
        expected_inputs: dict[str, tuple[int, ...]],
        expected_outputs: dict[str, tuple[int, ...]],
    ) -> dict[str, bool]:
        """Validate input/output shapes of ONNX model.

        Args:
            onnx_path: Path to ONNX model.
            expected_inputs: Expected input shapes.
            expected_outputs: Expected output shapes.

        Returns:
            Dictionary with shape validation results.

        """
        try:
            import onnx

            model = onnx.load(str(onnx_path))
            graph = model.graph

            results = {}

            # Check inputs
            for inp in graph.input:
                shape = []
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(-1)  # Dynamic
                    else:
                        shape.append(dim.dim_value)

                expected = expected_inputs.get(inp.name)
                if expected:
                    # Compare, allowing -1 for dynamic dims
                    match = True
                    for s, e in zip(shape, expected, strict=False):
                        if s != -1 and e != -1 and s != e:
                            match = False
                    results[f"input_{inp.name}"] = match
                else:
                    results[f"input_{inp.name}"] = True

            # Check outputs
            for out in graph.output:
                shape = []
                for dim in out.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(-1)
                    else:
                        shape.append(dim.dim_value)

                expected = expected_outputs.get(out.name)
                if expected:
                    match = True
                    for s, e in zip(shape, expected, strict=False):
                        if s != -1 and e != -1 and s != e:
                            match = False
                    results[f"output_{out.name}"] = match
                else:
                    results[f"output_{out.name}"] = True

            return results

        except ImportError:
            return {"error": "onnx package not available"}
        except Exception as e:
            return {"error": str(e)}

    def compare_quantized(
        self,
        original_path: str | Path,
        quantized_path: str | Path,
        test_inputs: list[np.ndarray],
    ) -> dict[str, Any]:
        """Compare original and quantized model outputs.

        Args:
            original_path: Path to original ONNX model.
            quantized_path: Path to quantized ONNX model.
            test_inputs: Test input arrays.

        Returns:
            Dictionary with comparison results.

        """
        from src.deployment.runtime import ONNXRuntime

        # Initialize runtimes
        original_runtime = ONNXRuntime(original_path)
        quantized_runtime = ONNXRuntime(quantized_path)

        policy_diffs = []
        value_diffs = []
        original_times = []
        quantized_times = []

        for test_input in test_inputs:
            # Run on original
            original_result = original_runtime.run(test_input)
            original_times.append(original_result.inference_time_ms)

            # Run on quantized
            quantized_result = quantized_runtime.run(test_input)
            quantized_times.append(quantized_result.inference_time_ms)

            # Compare
            policy_diff = np.abs(original_result.policy - quantized_result.policy).max()
            policy_diffs.append(policy_diff)

            if len(original_result.value.shape) > 0 and len(quantized_result.value.shape) > 0:
                value_diff = np.abs(original_result.value - quantized_result.value).max()
                value_diffs.append(value_diff)

        # Cleanup
        original_runtime.close()
        quantized_runtime.close()

        # Calculate metrics
        avg_original_time = np.mean(original_times)
        avg_quantized_time = np.mean(quantized_times)

        return {
            "max_policy_diff": max(policy_diffs),
            "mean_policy_diff": np.mean(policy_diffs),
            "max_value_diff": max(value_diffs) if value_diffs else 0.0,
            "mean_value_diff": np.mean(value_diffs) if value_diffs else 0.0,
            "original_time_ms": avg_original_time,
            "quantized_time_ms": avg_quantized_time,
            "speedup_ratio": avg_original_time / avg_quantized_time
            if avg_quantized_time > 0
            else 0,
            "n_samples": len(test_inputs),
        }

    def generate_test_inputs(
        self,
        n_samples: int,
        board_sizes: list[int],
        input_channels: int = 17,
    ) -> list[Tensor]:
        """Generate random test inputs.

        Args:
            n_samples: Number of test samples.
            board_sizes: Available board sizes.
            input_channels: Number of input channels.

        Returns:
            List of test input tensors.

        """
        import random

        inputs = []
        for _ in range(n_samples):
            board_size = random.choice(board_sizes)
            inp = torch.randn(1, input_channels, board_size, board_size)
            inputs.append(inp)

        return inputs


def create_validator(
    tolerance: float = 1e-5,
    **kwargs: Any,
) -> ModelValidator:
    """Factory function to create model validator.

    Args:
        tolerance: Comparison tolerance.
        **kwargs: Additional options.

    Returns:
        Configured ModelValidator instance.

    """
    return ModelValidator(tolerance=tolerance, **kwargs)


def validate_export(
    pytorch_model: nn.Module,
    onnx_path: str | Path,
    n_samples: int = 10,
    board_sizes: list[int] | None = None,
    tolerance: float = 1e-5,
) -> ValidationResult:
    """Convenience function to validate an exported model.

    Args:
        pytorch_model: Original PyTorch model.
        onnx_path: Path to exported ONNX model.
        n_samples: Number of test samples.
        board_sizes: Board sizes for test inputs.
        tolerance: Comparison tolerance.

    Returns:
        ValidationResult with comparison metrics.

    """
    board_sizes = board_sizes or list(DEFAULT_BOARD_SIZES)
    validator = create_validator(tolerance=tolerance)
    test_inputs = validator.generate_test_inputs(n_samples, board_sizes)
    return validator.validate(pytorch_model, onnx_path, test_inputs)
