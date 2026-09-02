import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol

import pytest

#: Statistic the ratio assertions in this directory compare on.
#:
#: **Median, not mean, and that is the whole point.** These tests run in CI's
#: blocking lane (``benchmark`` is registered in pyproject.toml and applied
#: here, but appears in no ``-m`` filter in ci.yml or the Makefile), so a
#: contended runner reds the build. Measured on an *idle* box (loadavg 0.06),
#: 30 rounds, N=512::
#:
#:     fnet  mean=0.953ms  median=0.438ms  min=0.291ms  max=13.228ms
#:     speedup  by-mean=19.424  by-median=41.577  by-min=59.542
#:
#: One outlier at 30x the median more than doubles the mean. This is **not**
#: threshold-widening -- the assertions need 1.5x and the code delivers 19-42x
#: whichever statistic is used; they fail on outliers, not on margin. ``min``
#: would be the most permissive and is deliberately *not* chosen: the median
#: still degrades if the code genuinely gets slower, which is the signal these
#: tests exist for.
ROBUST_STATISTIC: Final[str] = "median_time_s"


@dataclass
class BenchmarkStats:
    mean_time_s: float
    median_time_s: float
    std_time_s: float
    min_time_s: float
    max_time_s: float
    throughput_per_s: float

    def robust_time_s(self) -> float:
        """The statistic ratio assertions compare on. See ``ROBUST_STATISTIC``."""
        return float(getattr(self, ROBUST_STATISTIC))


class BenchmarkTimerProtocol(Protocol):
    def __call__(
        self,
        func: Callable[..., Any],
        *args: Any,
        warmup_rounds: int = 2,
        timing_rounds: int = 5,
        **kwargs: Any,
    ) -> BenchmarkStats: ...


@pytest.fixture
def benchmark_timer() -> BenchmarkTimerProtocol:
    """Execute a callable with warmup rounds and timing rounds.

    Previously opened with a ``request.getfixturevalue("benchmark")`` branch
    that delegated to ``pytest-benchmark`` when present. That branch was
    **dead**: ``pytest-benchmark`` is installed nowhere and declared in no
    dependency file (not in ``pyproject.toml``'s ``dev``/``test-extras``
    extras, not in any requirements file), and no fixture named ``benchmark``
    exists anywhere under ``tests/`` -- so ``getfixturevalue`` always raised and
    the delegate was always ``None``. Sixteen unreachable lines in a shared
    fixture that would have silently changed the timing semantics of every
    assertion in this directory the day someone ran ``pip install
    pytest-benchmark``, with no test covering the switch. Deleted rather than
    left as a trapdoor; wiring pytest-benchmark deliberately, with a declared
    dependency, is the way to bring it back.
    """

    def timer(
        func: Callable[..., Any],
        *args: Any,
        warmup_rounds: int = 2,
        timing_rounds: int = 5,
        **kwargs: Any,
    ) -> BenchmarkStats:
        # Warmup
        for _ in range(warmup_rounds):
            func(*args, **kwargs)

        times = []
        for _ in range(timing_rounds):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

        if not times:
            return BenchmarkStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        mean_time = sum(times) / len(times)
        median_time = statistics.median(times)
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        std_time = math.sqrt(variance)
        min_time = min(times)
        max_time = max(times)
        throughput = 1.0 / mean_time if mean_time > 0 else 0.0

        return BenchmarkStats(
            mean_time_s=mean_time,
            median_time_s=median_time,
            std_time_s=std_time,
            min_time_s=min_time,
            max_time_s=max_time,
            throughput_per_s=throughput,
        )

    return timer


# NOTE: no ``pytest_configure`` marker registration here. ``benchmark`` is
# registered in pyproject.toml's ``[tool.pytest.ini_options] markers`` with the
# description "performance and scaling benchmark tests". This file used to
# re-register it with a *different* wording, giving one marker two sources of
# truth -- and under ``--strict-markers`` the pyproject entry is the one that
# matters, so the local copy was inert as well as divergent.
