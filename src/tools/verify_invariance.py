"""Resolution invariance verification tool.

Verifies that a model trained on one resolution can run inference
on a different resolution without errors and produces valid outputs.

Usage:
    python -m src.tools.verify_invariance --train-size 9 --infer-size 19
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import structlog
import torch
from torch import nn

from config.schemas import OperatorConfig
from src.modeling.model import AlphaGalerkinModel

logger = structlog.get_logger(__name__)


def create_synthetic_input(
    batch_size: int,
    board_size: int,
    input_channels: int = 17,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create synthetic input tensor for testing.

    Args:
        batch_size: Batch size.
        board_size: Board size.
        input_channels: Number of input channels.
        device: Target device.

    Returns:
        Random input tensor.

    """
    return torch.randn(batch_size, input_channels, board_size, board_size, device=device)


def verify_forward_pass(
    model: nn.Module,
    board_size: int,
    batch_size: int = 4,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Verify forward pass works for given board size.

    Args:
        model: Model to test.
        board_size: Board size to test.
        batch_size: Batch size.
        device: Device for computation.

    Returns:
        Dictionary with verification results.

    """
    model.eval()
    device = device or next(model.parameters()).device

    results: dict[str, Any] = {
        "board_size": board_size,
        "batch_size": batch_size,
        "passed": False,
        "errors": [],
    }

    try:
        # Create input
        x = create_synthetic_input(batch_size, board_size, device=device)

        # Forward pass
        with torch.no_grad():
            start_time = time.perf_counter()
            output = model(x)
            inference_time = time.perf_counter() - start_time

        results["inference_time_ms"] = inference_time * 1000

        # Verify output shapes
        expected_policy_size = board_size**2 + 1
        if output.policy_logits.shape != (batch_size, expected_policy_size):
            results["errors"].append(
                f"Policy shape mismatch: expected ({batch_size}, {expected_policy_size}), "
                f"got {output.policy_logits.shape}"
            )

        if output.value.shape != (batch_size, 1):
            results["errors"].append(
                f"Value shape mismatch: expected ({batch_size}, 1), got {output.value.shape}"
            )

        # Verify value range
        if not ((output.value >= -1).all() and (output.value <= 1).all()):
            results["errors"].append(
                f"Value out of range [-1, 1]: min={output.value.min()}, max={output.value.max()}"
            )

        # Verify no NaN/Inf
        if torch.isnan(output.policy_logits).any():
            results["errors"].append("NaN in policy logits")

        if torch.isnan(output.value).any():
            results["errors"].append("NaN in value")

        if torch.isinf(output.policy_logits).any():
            results["errors"].append("Inf in policy logits")

        if torch.isinf(output.value).any():
            results["errors"].append("Inf in value")

        # Verify policy can be softmaxed
        policy_probs = torch.softmax(output.policy_logits, dim=-1)
        if not torch.allclose(
            policy_probs.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-4
        ):
            results["errors"].append("Policy probabilities don't sum to 1")

        results["passed"] = len(results["errors"]) == 0

        # Store some output statistics
        results["policy_entropy"] = (
            -(policy_probs * torch.log(policy_probs + 1e-10)).sum(dim=-1).mean().item()
        )
        results["value_mean"] = output.value.mean().item()
        results["value_std"] = output.value.std().item()

    except Exception as e:
        results["errors"].append(f"Exception: {e!s}")

    return results


def verify_lbb_stability(
    model: AlphaGalerkinModel,
    board_size: int,
    batch_size: int = 4,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Verify LBB stability condition.

    Args:
        model: Model to test.
        board_size: Board size.
        batch_size: Batch size.
        device: Device for computation.

    Returns:
        Dictionary with LBB verification results.

    """
    model.eval()
    device = device or next(model.parameters()).device

    results: dict[str, Any] = {
        "board_size": board_size,
        "passed": False,
        "errors": [],
    }

    try:
        x = create_synthetic_input(batch_size, board_size, device=device)

        with torch.no_grad():
            output = model(x, return_lbb=True)

        if output.lbb_constant is None:
            results["errors"].append("LBB constant not returned")
        else:
            lbb_min = output.lbb_constant.min().item()
            lbb_mean = output.lbb_constant.mean().item()
            lbb_max = output.lbb_constant.max().item()

            results["lbb_min"] = lbb_min
            results["lbb_mean"] = lbb_mean
            results["lbb_max"] = lbb_max

            # Check stability threshold
            threshold = model.config.lbb_beta_threshold
            if lbb_min < threshold:
                results["errors"].append(f"LBB constant below threshold: {lbb_min} < {threshold}")

            if lbb_min <= 0:
                results["errors"].append(f"LBB constant non-positive: {lbb_min}")

        results["passed"] = len(results["errors"]) == 0

    except Exception as e:
        results["errors"].append(f"Exception: {e!s}")

    return results


def verify_semantic_validity(
    model: nn.Module,
    board_size: int,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Verify that outputs are semantically valid (influence maps).

    Tests:
    - Policy concentrates probability on some moves (not uniform)
    - Value varies with input (not constant)
    - Output is deterministic in eval mode

    Args:
        model: Model to test.
        board_size: Board size.
        device: Device for computation.

    Returns:
        Verification results.

    """
    model.eval()
    device = device or next(model.parameters()).device

    results: dict[str, Any] = {
        "board_size": board_size,
        "passed": False,
        "errors": [],
    }

    try:
        # Test determinism
        x = torch.randn(1, 17, board_size, board_size, device=device)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        if not torch.allclose(out1.policy_logits, out2.policy_logits, atol=1e-6):
            results["errors"].append("Model not deterministic in eval mode")

        if not torch.allclose(out1.value, out2.value, atol=1e-6):
            results["errors"].append("Value not deterministic in eval mode")

        # Test that policy is not uniform
        policy_probs = torch.softmax(out1.policy_logits, dim=-1)
        max_prob = policy_probs.max().item()
        uniform_prob = 1.0 / policy_probs.shape[-1]

        # Max probability should be noticeably higher than uniform
        if max_prob < uniform_prob * 1.5:
            results["warnings"] = results.get("warnings", [])
            results["warnings"].append(
                f"Policy may be too uniform: max_prob={max_prob:.4f}, uniform={uniform_prob:.4f}"
            )

        results["max_policy_prob"] = max_prob
        results["uniform_prob"] = uniform_prob

        # Test that value varies with input
        x_different = torch.randn(4, 17, board_size, board_size, device=device)

        with torch.no_grad():
            out_different = model(x_different)

        value_std = out_different.value.std().item()
        if value_std < 0.01:
            results["warnings"] = results.get("warnings", [])
            results["warnings"].append(f"Value may be too constant: std={value_std:.4f}")

        results["value_std"] = value_std

        results["passed"] = len(results["errors"]) == 0

    except Exception as e:
        results["errors"].append(f"Exception: {e!s}")

    return results


def run_verification(
    train_size: int = 9,
    infer_size: int = 19,
    device: str = "cpu",
    verbose: bool = True,
) -> bool:
    """Run full verification suite.

    Args:
        train_size: Size used for "training" (model initialization).
        infer_size: Size to test inference on.
        device: Device for computation.
        verbose: Whether to print detailed output.

    Returns:
        True if all verifications pass, False otherwise.

    """
    logger.info(
        "starting_verification",
        train_size=train_size,
        infer_size=infer_size,
        device=device,
    )

    # Create model
    torch.manual_seed(42)
    config = OperatorConfig(
        d_model=128,
        n_heads=4,
        n_galerkin_layers=4,
        n_softmax_layers=2,
        n_fourier_features=64,
        input_channels=17,
    )
    model = AlphaGalerkinModel(config)
    model.to(device)
    model.eval()

    # Set training resolution
    model.training_resolution = train_size

    all_passed = True
    results_summary = {}

    # Test 1: Forward pass on training size
    logger.info("testing_train_size", board_size=train_size)
    results = verify_forward_pass(model, train_size, device=torch.device(device))
    results_summary[f"forward_train_{train_size}"] = results
    if not results["passed"]:
        all_passed = False
        logger.error("forward_pass_failed", **results)
    else:
        logger.info("forward_pass_passed", **results)

    # Test 2: Forward pass on inference size (CRITICAL - zero-shot transfer)
    logger.info("testing_infer_size", board_size=infer_size)
    results = verify_forward_pass(model, infer_size, device=torch.device(device))
    results_summary[f"forward_infer_{infer_size}"] = results
    if not results["passed"]:
        all_passed = False
        logger.error("forward_pass_failed", **results)
    else:
        logger.info("forward_pass_passed", **results)

    # Test 3: LBB stability
    logger.info("testing_lbb_stability")
    for size in [train_size, infer_size]:
        results = verify_lbb_stability(model, size, device=torch.device(device))
        results_summary[f"lbb_{size}"] = results
        if not results["passed"]:
            all_passed = False
            logger.error("lbb_stability_failed", **results)
        else:
            logger.info("lbb_stability_passed", **results)

    # Test 4: Semantic validity
    logger.info("testing_semantic_validity")
    for size in [train_size, infer_size]:
        results = verify_semantic_validity(model, size, device=torch.device(device))
        results_summary[f"semantic_{size}"] = results
        if not results["passed"]:
            all_passed = False
            logger.error("semantic_validity_failed", **results)
        else:
            logger.info("semantic_validity_passed", **results)

    # Test 5: Multiple resolutions
    logger.info("testing_multiple_resolutions")
    for size in [5, 9, 13, 19, 25]:
        results = verify_forward_pass(model, size, batch_size=2, device=torch.device(device))
        results_summary[f"forward_{size}"] = results
        if not results["passed"]:
            all_passed = False
            logger.error("multi_resolution_failed", board_size=size, **results)

    # Final summary
    if all_passed:
        logger.info(
            "verification_passed",
            message="All resolution invariance tests passed!",
            train_size=train_size,
            infer_size=infer_size,
        )
    else:
        logger.error(
            "verification_failed",
            message="Some tests failed. See above for details.",
        )

    return all_passed


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify resolution invariance of AlphaGalerkin model"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=9,
        help="Board size used for training (default: 9)",
    )
    parser.add_argument(
        "--infer-size",
        type=int,
        default=19,
        help="Board size for inference (default: 19)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation (default: cpu)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

    # Run verification
    passed = run_verification(
        train_size=args.train_size,
        infer_size=args.infer_size,
        device=args.device,
        verbose=args.verbose,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
