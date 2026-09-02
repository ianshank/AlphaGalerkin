"""Performance benchmarks for FNetMixing compared to standard Attention."""

from typing import Any

import pytest
import torch
from torch import nn

from src.modeling.fnet import FNetMixing
from tests.benchmarks.conftest import BenchmarkTimerProtocol

#: Minimum FNet-over-attention speedup demanded at the larger sequence lengths.
#: Unchanged from the bare ``1.5`` this file used to assert; named so the number
#: has one home and a rationale. Measured on an idle box at N=512 the real
#: figure is ~41x by median, so this is a *floor* with two orders of magnitude of
#: headroom, not a tuned threshold -- which is why the historical failures were
#: outlier artefacts rather than genuine regressions.
MIN_FNET_SPEEDUP = 1.5

#: Sequence lengths at or above which the O(N log N) claim should dominate the
#: constant overheads. Below this the two are legitimately comparable.
SPEEDUP_ASSERTED_ABOVE_N = 256

#: Timing budget. Ten warmups to settle allocator/threadpool state, thirty timed
#: rounds so the median has something to be a median *of* -- with five rounds a
#: single stall moves it.
WARMUP_ROUNDS = 10
TIMING_ROUNDS = 30


def fnet_speedup(mha_time_s: float, fnet_time_s: float) -> float:
    """Speedup, floored so a zero-time sample cannot divide by zero.

    Extracted as a named function purely so ``TestSpeedupPredicate`` can drive it
    on synthetic inputs. A predicate that only ever runs behind a real
    measurement is a predicate whose thresholds nothing checks -- the trap
    recorded in ``tests/research/test_amr_arena_interpretability.py``, where
    widening a band left every solve-driven test green.
    """
    if fnet_time_s <= 0.0:
        return float("inf")
    return mha_time_s / fnet_time_s


class TestSpeedupPredicate:
    """Unit-tests the predicate itself, with no tensors and no timing."""

    def test_a_real_speedup_passes(self) -> None:
        assert fnet_speedup(10.0, 1.0) >= MIN_FNET_SPEEDUP

    def test_a_slowdown_fails(self) -> None:
        """The case a too-loose threshold would wave through."""
        assert fnet_speedup(1.0, 2.0) < MIN_FNET_SPEEDUP

    def test_parity_fails(self) -> None:
        assert fnet_speedup(1.0, 1.0) < MIN_FNET_SPEEDUP

    def test_zero_time_does_not_divide_by_zero(self) -> None:
        assert fnet_speedup(1.0, 0.0) == float("inf")


class TestBenchmarkTimerEdges:
    """The shared fixture's ``if not times:`` early return had no caller.

    Every site passes ``timing_rounds`` >= 5, so ``times`` was never empty and
    the all-zeros ``BenchmarkStats`` branch was unmeasured. Zero rounds is a
    legitimate way to ask "warm up only"; it must not divide by zero.
    """

    def test_zero_timing_rounds_returns_all_zero_stats(
        self, benchmark_timer: BenchmarkTimerProtocol
    ) -> None:
        calls = 0

        def fn() -> None:
            nonlocal calls
            calls += 1

        stats = benchmark_timer(fn, warmup_rounds=2, timing_rounds=0)
        assert calls == 2, "warmups still run"
        assert stats.mean_time_s == stats.median_time_s == stats.throughput_per_s == 0.0
        assert stats.robust_time_s() == 0.0


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

        stats_fnet = benchmark_timer(
            run_fnet, x, warmup_rounds=WARMUP_ROUNDS, timing_rounds=TIMING_ROUNDS
        )
        fnet_times[N] = stats_fnet.robust_time_s()

        stats_mha = benchmark_timer(
            run_mha, x, warmup_rounds=WARMUP_ROUNDS, timing_rounds=TIMING_ROUNDS
        )
        mha_times[N] = stats_mha.robust_time_s()

    # Compared on the MEDIAN, not the mean (see conftest.ROBUST_STATISTIC): a
    # single stalled round out of thirty more than doubles the mean even on an
    # idle box, and this test runs in CI's blocking lane.
    for N in seq_lengths:
        if N >= SPEEDUP_ASSERTED_ABOVE_N:
            time_mha = mha_times[N]
            time_fnet = fnet_times[N]
            speedup = fnet_speedup(time_mha, time_fnet)
            assert speedup >= MIN_FNET_SPEEDUP, (
                f"FNet did not achieve >= {MIN_FNET_SPEEDUP}x speedup at N={N}. "
                f"MHA time: {time_mha:.5f}, FNet time: {time_fnet:.5f}, Speedup: {speedup:.2f}x"
            )
