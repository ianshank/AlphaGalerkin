"""Performance benchmarks for FNetMixing compared to standard Attention."""

from typing import Any

import pytest
import torch
from torch import nn

from src.modeling.fnet import FNetMixing
from tests.benchmarks.conftest import BenchmarkTimerProtocol


@pytest.mark.benchmark
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fnet_vs_attention(
    benchmark_timer: BenchmarkTimerProtocol,
    device: str,
) -> None:
    """Benchmark FNet vs MultiheadAttention speedup."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    d_model = 64
    n_heads = 4
    batch_size = 16
    seq_lengths = [128, 256, 512]

    fnet = FNetMixing(use_2d=False).to(device)
    mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True).to(device)

    fnet.eval()
    mha.eval()

    fnet_times: dict[int, float] = {}
    mha_times: dict[int, float] = {}

    for N in seq_lengths:
        x = torch.randn(batch_size, N, d_model, device=device)

        def run_fnet(x_in: torch.Tensor) -> Any:
            with torch.no_grad():
                out = fnet(x_in)
                if device == "cuda":
                    torch.cuda.synchronize()
                return out

        def run_mha(x_in: torch.Tensor) -> Any:
            with torch.no_grad():
                out, _ = mha(x_in, x_in, x_in, need_weights=False)
                if device == "cuda":
                    torch.cuda.synchronize()
                return out

        # Warmup and test
        out_fnet = run_fnet(x)

        # Verify finite non-NaN outputs
        assert torch.isfinite(out_fnet).all(), "FNet output contains NaN or Inf"

        stats_fnet = benchmark_timer(run_fnet, x, warmup_rounds=10, timing_rounds=30)
        fnet_times[N] = stats_fnet.mean_time_s

        stats_mha = benchmark_timer(run_mha, x, warmup_rounds=10, timing_rounds=30)
        mha_times[N] = stats_mha.mean_time_s

    # Verify that FNet achieves substantial speedup (>= 1.5x) for sequence lengths >= 256
    for N in seq_lengths:
        if N >= 256:
            time_mha = mha_times[N]
            time_fnet = fnet_times[N]

            # Substantial speedup calculation
            if time_fnet > 0:
                speedup = time_mha / time_fnet
                assert speedup >= 1.5, (
                    f"FNet did not achieve >= 1.5x speedup at N={N}. "
                    f"MHA time: {time_mha:.5f}, FNet time: {time_fnet:.5f}, Speedup: {speedup:.2f}x"
                )
