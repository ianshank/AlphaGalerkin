"""Benchmarking utilities for AlphaGalerkin demo notebooks.

Provides reusable, configurable benchmarking functions with proper
warmup, timing, and result formatting.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    # Fallback to standard logging if structlog not available
    logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


@dataclass
class BenchmarkResult:
    """Result of a single benchmark measurement."""

    name: str
    seq_length: int
    time_ms: float
    batch_size: int
    n_runs: int

    @property
    def throughput(self) -> float:
        """Compute throughput in items per second."""
        if self.time_ms <= 0:
            return 0.0
        return (self.batch_size * 1000) / self.time_ms


def benchmark_module(
    module: nn.Module,
    input_tensor: Tensor,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> float:
    """Benchmark a module's forward pass.

    Args:
        module: PyTorch module to benchmark.
        input_tensor: Input tensor for the forward pass.
        n_warmup: Number of warmup runs (not timed).
        n_runs: Number of timed runs for averaging.

    Returns:
        Average time per forward pass in milliseconds.

    """
    module.eval()

    # Warmup runs
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = module(input_tensor)

    # Synchronize if CUDA
    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = module(input_tensor)

    # Synchronize again for accurate timing
    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / n_runs) * 1000

    logger.debug(
        "benchmark_complete",
        module=module.__class__.__name__,
        avg_ms=avg_ms,
        n_runs=n_runs,
    )

    return avg_ms


def benchmark_attention(
    galerkin_attn: nn.Module,
    softmax_attn: nn.Module,
    seq_lengths: Sequence[int],
    d_model: int,
    batch_size: int = 4,
    n_warmup: int = 5,
    n_runs: int = 50,
    device: torch.device | str = "cpu",
) -> tuple[list[BenchmarkResult], list[BenchmarkResult]]:
    """Benchmark Galerkin vs Softmax attention across sequence lengths.

    Args:
        galerkin_attn: Galerkin attention module.
        softmax_attn: Softmax attention module.
        seq_lengths: Sequence lengths to benchmark (e.g., [25, 81, 169, 361]).
        d_model: Model dimension.
        batch_size: Batch size for benchmarking.
        n_warmup: Warmup iterations.
        n_runs: Timed iterations.
        device: Device to run on.

    Returns:
        Tuple of (galerkin_results, softmax_results).

    Raises:
        ValueError: If seq_lengths is empty.
        RuntimeError: If module execution fails.

    """
    # Validate inputs
    if not seq_lengths:
        raise ValueError("seq_lengths cannot be empty")

    galerkin_results = []
    softmax_results = []

    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        # Ensure modules are on the correct device
        if hasattr(galerkin_attn, "to"):
            galerkin_attn.to(device)
        if hasattr(softmax_attn, "to"):
            softmax_attn.to(device)

        # Benchmark Galerkin with error handling
        try:
            galerkin_ms = benchmark_module(galerkin_attn, x, n_warmup, n_runs)
        except Exception as e:
            logger.error("galerkin_benchmark_failed", seq_len=seq_len, error=str(e))
            raise RuntimeError(f"Galerkin attention failed at seq_len={seq_len}: {e}") from e

        galerkin_results.append(
            BenchmarkResult(
                name="Galerkin",
                seq_length=seq_len,
                time_ms=galerkin_ms,
                batch_size=batch_size,
                n_runs=n_runs,
            )
        )

        # Benchmark Softmax with error handling
        try:
            softmax_ms = benchmark_module(softmax_attn, x, n_warmup, n_runs)
        except Exception as e:
            logger.error("softmax_benchmark_failed", seq_len=seq_len, error=str(e))
            raise RuntimeError(f"Softmax attention failed at seq_len={seq_len}: {e}") from e

        softmax_results.append(
            BenchmarkResult(
                name="Softmax",
                seq_length=seq_len,
                time_ms=softmax_ms,
                batch_size=batch_size,
                n_runs=n_runs,
            )
        )

        logger.debug(
            "attention_benchmark",
            seq_len=seq_len,
            galerkin_ms=galerkin_ms,
            softmax_ms=softmax_ms,
            speedup=softmax_ms / galerkin_ms if galerkin_ms > 0 else 0,
        )

    return galerkin_results, softmax_results


def benchmark_model_throughput(
    model: nn.Module,
    board_size: int,
    input_channels: int = 17,
    batch_size: int = 32,
    n_evals: int = 100,
    device: torch.device | str = "cpu",
) -> float:
    """Benchmark model throughput for MCTS rollouts.

    Args:
        model: Model to benchmark.
        board_size: Board size (height/width).
        input_channels: Number of input channels.
        batch_size: Batch size.
        n_evals: Number of evaluation batches.
        device: Device to run on.

    Returns:
        Throughput in positions evaluated per second.

    Raises:
        ValueError: If board_size or batch_size is non-positive.
        RuntimeError: If model execution fails.

    """
    # Validate inputs
    if board_size <= 0:
        raise ValueError(f"board_size must be positive, got {board_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_evals <= 0:
        raise ValueError(f"n_evals must be positive, got {n_evals}")

    # Ensure model is on the correct device
    if hasattr(model, "to"):
        model.to(device)

    model.eval()
    x = torch.randn(batch_size, input_channels, board_size, board_size, device=device)

    # Warmup with error handling
    try:
        with torch.no_grad():
            _ = model(x)
    except Exception as e:
        logger.error(
            "model_warmup_failed",
            model=model.__class__.__name__,
            board_size=board_size,
            error=str(e),
        )
        raise RuntimeError(f"Model warmup failed: {e}") from e

    if x.is_cuda:
        torch.cuda.synchronize()

    # Benchmark with error handling
    try:
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_evals):
                _ = model(x)

        if x.is_cuda:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
    except Exception as e:
        logger.error(
            "model_benchmark_failed",
            model=model.__class__.__name__,
            board_size=board_size,
            error=str(e),
        )
        raise RuntimeError(f"Model benchmark failed: {e}") from e

    evals_per_sec = (batch_size * n_evals) / elapsed

    logger.debug(
        "throughput_benchmark",
        model=model.__class__.__name__,
        board_size=board_size,
        evals_per_sec=evals_per_sec,
    )

    return evals_per_sec


def format_benchmark_table(
    galerkin_results: list[BenchmarkResult],
    softmax_results: list[BenchmarkResult],
    board_labels: Sequence[str] | None = None,
) -> str:
    """Format benchmark results as a printable table.

    Args:
        galerkin_results: Galerkin benchmark results.
        softmax_results: Softmax benchmark results.
        board_labels: Optional labels for each row.

    Returns:
        Formatted table string.

    Raises:
        ValueError: If result lists have different lengths.

    """
    # Handle empty results
    if not galerkin_results and not softmax_results:
        return "No benchmark results to display"

    # Validate matching lengths
    if len(galerkin_results) != len(softmax_results):
        raise ValueError(
            f"Result length mismatch: galerkin={len(galerkin_results)}, "
            f"softmax={len(softmax_results)}"
        )

    # Validate labels if provided
    if board_labels is not None and len(board_labels) != len(galerkin_results):
        raise ValueError(
            f"Label count mismatch: labels={len(board_labels)}, results={len(galerkin_results)}"
        )

    lines = []
    header = f"{'Positions':^10} | {'Galerkin (ms)':^14} | {'Softmax (ms)':^14} | {'Speedup':^10}"
    lines.append(header)
    lines.append("-" * 55)

    for i, (g, s) in enumerate(zip(galerkin_results, softmax_results, strict=True)):
        speedup = s.time_ms / g.time_ms if g.time_ms > 0 else 0
        label = board_labels[i] if board_labels else str(g.seq_length)
        lines.append(f"{label:^10} | {g.time_ms:^14.3f} | {s.time_ms:^14.3f} | {speedup:^10.2f}x")

    return "\n".join(lines)


def format_throughput_table(
    results: list[tuple[str, int, float, float]],
) -> str:
    """Format throughput comparison results.

    Args:
        results: List of (board_label, size, full_throughput, fast_throughput).

    Returns:
        Formatted table string.

    """
    lines = []
    header = f"{'Board':^10} | {'Full Model':^15} | {'Fast Model':^15} | {'Speedup':^10}"
    lines.append(header)
    lines.append("-" * 55)

    for label, _size, full_tp, fast_tp in results:
        speedup = fast_tp / full_tp if full_tp > 0 else 0
        lines.append(f"{label:^10} | {full_tp:^15,.0f} | {fast_tp:^15,.0f} | {speedup:^10.2f}x")

    return "\n".join(lines)
