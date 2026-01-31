#!/usr/bin/env python3
"""FNet Speed Benchmark.

Benchmarks FNet O(N log N) vs Softmax Attention O(N²) scaling.

This validates the theoretical speedup of FFT-based mixing for
high-speed MCTS rollouts.

Usage:
    python -m src.experiments.benchmark_fnet
    python -m src.experiments.benchmark_fnet --sizes 81,169,361,625,900
    python -m src.experiments.benchmark_fnet --n-warmup 10 --n-iterations 100
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger(__name__)

# Module-level constants with documented rationale
# These define thresholds for scaling analysis

# Scaling exponent threshold for sub-quadratic vs quadratic classification
# FNet should have exponent ~1.0-1.1 (N log N ≈ N^1 for log-log fitting)
# Softmax should have exponent ~2.0 (N²)
# 1.5 is the midpoint that cleanly separates these two regimes
# - exponent < 1.5: Sub-quadratic (good for FNet)
# - exponent > 1.5: Quadratic or worse (expected for Softmax)
SUBQUADRATIC_EXPONENT_THRESHOLD: float = 1.5


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    n_tokens: int
    fnet_time_ms: float
    softmax_time_ms: float
    speedup: float
    fnet_memory_mb: float
    softmax_memory_mb: float


class FNetMixingLayer(nn.Module):
    """FNet-style FFT mixing layer.

    Uses 2D FFT for O(N log N) token mixing on grid-structured inputs.
    """

    def __init__(self, d_model: int) -> None:
        """Initialize FNet mixing layer."""
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: int,
    ) -> torch.Tensor:
        """Apply FFT-based mixing.

        Args:
            x: Input tensor (batch, n_tokens, d_model).
            grid_size: Grid dimension (sqrt of n_tokens).

        Returns:
            Mixed tensor (batch, n_tokens, d_model).

        """
        batch, n_tokens, d = x.shape

        # Reshape to grid
        x_grid = x.view(batch, grid_size, grid_size, d)

        # 2D FFT mixing (real-valued for efficiency)
        x_freq = torch.fft.rfft2(x_grid, dim=(1, 2), norm="ortho")
        x_mixed = torch.fft.irfft2(x_freq, s=(grid_size, grid_size), dim=(1, 2), norm="ortho")

        # Reshape back
        x_mixed = x_mixed.view(batch, n_tokens, d)

        return self.norm(x + x_mixed)


class SoftmaxAttentionLayer(nn.Module):
    """Standard softmax attention for comparison.

    O(N²) complexity in number of tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """Initialize softmax attention layer."""
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply softmax attention.

        Args:
            x: Input tensor (batch, n_tokens, d_model).

        Returns:
            Output tensor (batch, n_tokens, d_model).

        """
        batch, n_tokens, _ = x.shape

        qkv = self.qkv(x).reshape(batch, n_tokens, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, tokens, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores: O(N²)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, n_tokens, self.d_model)
        out = self.proj(out)

        return self.norm(x + out)


def benchmark_layer(
    layer: nn.Module,
    x: torch.Tensor,
    n_warmup: int,
    n_iterations: int,
    grid_size: int | None = None,
) -> tuple[float, float]:
    """Benchmark a layer's forward pass.

    Args:
        layer: Layer to benchmark.
        x: Input tensor.
        n_warmup: Number of warmup iterations.
        n_iterations: Number of timed iterations.
        grid_size: Grid size for FNet layer.

    Returns:
        Tuple of (mean time ms, peak memory MB).

    """
    device = x.device

    # Warmup
    for _ in range(n_warmup):
        if grid_size is not None:  # noqa: SIM108
            layer(x, grid_size)
        else:
            layer(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    times = []
    for _ in range(n_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        if grid_size is not None:  # noqa: SIM108
            layer(x, grid_size)
        else:
            layer(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append((time.perf_counter() - start) * 1000)  # ms

    mean_time = np.mean(times)

    # Memory usage
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        peak_memory = 0.0  # Can't easily measure CPU memory

    return mean_time, peak_memory


def run_benchmark(
    sizes: list[int],
    d_model: int = 128,
    batch_size: int = 32,
    n_warmup: int = 10,
    n_iterations: int = 100,
    device: str = "auto",
) -> list[BenchmarkResult]:
    """Run FNet vs Softmax benchmark across sizes.

    Args:
        sizes: List of n_tokens (must be perfect squares).
        d_model: Model dimension.
        batch_size: Batch size.
        n_warmup: Warmup iterations.
        n_iterations: Timed iterations.
        device: Device to use.

    Returns:
        List of benchmark results.

    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_device = torch.device(device)
    logger.info("benchmark_starting", device=device, sizes=sizes)

    # Create layers
    fnet = FNetMixingLayer(d_model).to(torch_device)
    softmax = SoftmaxAttentionLayer(d_model).to(torch_device)

    fnet.eval()
    softmax.eval()

    results = []

    for n_tokens in sizes:
        grid_size = int(np.sqrt(n_tokens))
        assert grid_size * grid_size == n_tokens, "n_tokens must be a perfect square"

        logger.info("benchmarking_size", n_tokens=n_tokens, grid_size=grid_size)

        # Create input
        x = torch.randn(batch_size, n_tokens, d_model, device=torch_device)

        with torch.no_grad():
            # Benchmark FNet
            fnet_time, fnet_mem = benchmark_layer(
                fnet, x, n_warmup, n_iterations, grid_size
            )

            # Benchmark Softmax
            softmax_time, softmax_mem = benchmark_layer(
                softmax, x, n_warmup, n_iterations
            )

        speedup = softmax_time / fnet_time if fnet_time > 0 else float("inf")

        result = BenchmarkResult(
            n_tokens=n_tokens,
            fnet_time_ms=fnet_time,
            softmax_time_ms=softmax_time,
            speedup=speedup,
            fnet_memory_mb=fnet_mem,
            softmax_memory_mb=softmax_mem,
        )
        results.append(result)

        logger.info(
            "benchmark_result",
            n_tokens=n_tokens,
            fnet_ms=f"{fnet_time:.3f}",
            softmax_ms=f"{softmax_time:.3f}",
            speedup=f"{speedup:.2f}x",
        )

    return results


def analyze_scaling(results: list[BenchmarkResult]) -> dict[str, float]:
    """Analyze scaling behavior from benchmark results.

    FNet should scale as O(N log N), Softmax as O(N²).

    Args:
        results: List of benchmark results.

    Returns:
        Dictionary with scaling analysis.

    """
    n_tokens = np.array([r.n_tokens for r in results])
    fnet_times = np.array([r.fnet_time_ms for r in results])
    softmax_times = np.array([r.softmax_time_ms for r in results])

    # Fit log-log regression to estimate scaling exponent
    # t = c * N^α → log(t) = log(c) + α * log(N)

    log_n = np.log(n_tokens)

    # FNet scaling (should be ~1.0-1.1 for N log N)
    log_fnet = np.log(fnet_times)
    fnet_coeffs = np.polyfit(log_n, log_fnet, 1)
    fnet_exponent = fnet_coeffs[0]

    # Softmax scaling (should be ~2.0 for N²)
    log_softmax = np.log(softmax_times)
    softmax_coeffs = np.polyfit(log_n, log_softmax, 1)
    softmax_exponent = softmax_coeffs[0]

    # Theoretical ratio: N² / (N log N) = N / log N
    # At large N, speedup should grow with N

    speedups = np.array([r.speedup for r in results])

    return {
        "fnet_scaling_exponent": float(fnet_exponent),
        "softmax_scaling_exponent": float(softmax_exponent),
        "fnet_expected_exponent": 1.0,  # N log N ≈ N for analysis
        "softmax_expected_exponent": 2.0,  # N²
        "mean_speedup": float(np.mean(speedups)),
        "max_speedup": float(np.max(speedups)),
        "speedup_at_largest": float(speedups[-1]),
        "fnet_scales_subquadratic": bool(fnet_exponent < SUBQUADRATIC_EXPONENT_THRESHOLD),
        "softmax_scales_quadratic": bool(softmax_exponent > SUBQUADRATIC_EXPONENT_THRESHOLD),
    }


def main() -> None:
    """Run the FNet benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark FNet vs Softmax attention")
    parser.add_argument(
        "--sizes",
        type=str,
        default="81,169,361,625,900",
        help="Comma-separated list of n_tokens (must be perfect squares)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model dimension",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=100,
        help="Number of timed iterations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmarks",
        help="Directory to save results",
    )

    args = parser.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]

    results = run_benchmark(
        sizes=sizes,
        d_model=args.d_model,
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        n_iterations=args.n_iterations,
        device=args.device,
    )

    scaling = analyze_scaling(results)

    # Print results
    print("\n" + "=" * 70)
    print("FNET VS SOFTMAX ATTENTION BENCHMARK")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print()

    print(f"{'N Tokens':<12} {'Grid':<8} {'FNet (ms)':<12} {'Softmax (ms)':<14} {'Speedup':<10}")
    print("-" * 56)

    for r in results:
        grid = int(np.sqrt(r.n_tokens))
        print(
            f"{r.n_tokens:<12} {grid}x{grid:<4} {r.fnet_time_ms:<12.3f} "
            f"{r.softmax_time_ms:<14.3f} {r.speedup:.2f}x"
        )

    print()
    print("Scaling Analysis:")
    print(f"  FNet exponent: {scaling['fnet_scaling_exponent']:.3f} (expected ~1.0 for N log N)")
    print(f"  Softmax exponent: {scaling['softmax_scaling_exponent']:.3f} (expected ~2.0 for N^2)")
    print(f"  Mean speedup: {scaling['mean_speedup']:.2f}x")
    print(f"  Max speedup: {scaling['max_speedup']:.2f}x")
    print()

    if scaling["fnet_scales_subquadratic"]:
        print("[PASS] FNet scales sub-quadratically (O(N log N) verified)")
    else:
        print("[FAIL] FNet scaling unexpected")

    if scaling["softmax_scales_quadratic"]:
        print("[PASS] Softmax scales quadratically (O(N^2) verified)")
    else:
        print("[FAIL] Softmax scaling unexpected")

    print("=" * 70)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": {
            "sizes": sizes,
            "d_model": args.d_model,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "results": [
            {
                "n_tokens": r.n_tokens,
                "fnet_time_ms": r.fnet_time_ms,
                "softmax_time_ms": r.softmax_time_ms,
                "speedup": r.speedup,
            }
            for r in results
        ],
        "scaling": scaling,
    }

    with open(output_dir / "fnet_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir / 'fnet_benchmark.json'}")


if __name__ == "__main__":
    main()
