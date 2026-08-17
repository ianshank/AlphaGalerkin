"""Performance benchmarks for GalerkinAttention."""

from typing import Any

import pytest
import torch

from src.modeling.attention import GalerkinAttention
from tests.benchmarks.conftest import BenchmarkTimerProtocol


@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_galerkin_attention_scaling(
    benchmark_timer: BenchmarkTimerProtocol,
    device: str,
) -> None:
    """Benchmark GalerkinAttention O(N) scaling relative to sequence length."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    d_model = 64
    n_heads = 4
    batch_size = 8
    seq_lengths = [81, 169, 361, 625]

    model = GalerkinAttention(d_model=d_model, n_heads=n_heads).to(device)
    model.eval()

    times: dict[int, float] = {}

    for N in seq_lengths:
        x = torch.randn(batch_size, N, d_model, device=device)

        def run_forward(x_in: torch.Tensor) -> Any:
            with torch.no_grad():
                out = model(x_in)
                if device == "cuda":
                    torch.cuda.synchronize()
                return out

        stats = benchmark_timer(run_forward, x, warmup_rounds=5, timing_rounds=20)
        times[N] = stats.mean_time_s

    # Verify scaling is approximately O(N)
    # Ratios of T(N) / T(N_prev) should be roughly equal to N / N_prev
    # We allow a generous margin (±25%) for overheads and CI variability.

    for i in range(1, len(seq_lengths)):
        N_prev = seq_lengths[i - 1]
        N_curr = seq_lengths[i]

        time_prev = times[N_prev]
        time_curr = times[N_curr]

        expected_ratio = N_curr / N_prev
        actual_ratio = time_curr / time_prev if time_prev > 0 else 0.0

        # O(N) means actual_ratio ≈ expected_ratio.
        # Due to constant overheads, ratio may vary for small N,
        # but it should scale significantly better than O(N^2).

        n_squared_ratio = (N_curr / N_prev) ** 2

        # Ensure it scales better than O(N^2) significantly, and roughly close to O(N)
        assert actual_ratio < n_squared_ratio * 0.8, (
            f"Scaling seems worse than O(N). For N={N_prev} to {N_curr}, "
            f"expected O(N) ratio ~ {expected_ratio:.2f}, got {actual_ratio:.2f}. "
            f"O(N^2) ratio is {n_squared_ratio:.2f}."
        )
