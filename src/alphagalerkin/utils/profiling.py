"""Performance profiling utilities for AlphaGalerkin.

Provides:
- PerformanceTracker: Accumulates timing samples via ``track()``
  and produces a summary ``report()``.
- profile_function: Decorator that records every call into a
  module-level ``PerformanceTracker``.
- cprofile_section: Context manager that runs ``cProfile`` over a
  block and writes the stats to a file.
"""

from __future__ import annotations

import cProfile
import functools
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from typing import ParamSpec, TypeVar

    P = ParamSpec("P")
    R = TypeVar("R")

logger = structlog.get_logger("utils.profiling")


# ------------------------------------------------------------------ #
#  PerformanceTracker                                                 #
# ------------------------------------------------------------------ #


@dataclass
class _SectionStats:
    """Accumulated statistics for a single tracked section."""

    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0


class PerformanceTracker:
    """Collect wall-clock timings for named code sections.

    Thread-safety is *not* guaranteed; use one tracker per thread or
    protect externally if concurrent access is needed.

    Example::

        tracker = PerformanceTracker()

        with tracker.track("forward"):
            out = model(x)

        with tracker.track("backward"):
            loss.backward()

        print(tracker.report())

    """

    def __init__(self) -> None:
        self._sections: dict[str, _SectionStats] = {}

    # -- public API ------------------------------------------------ #

    @contextmanager
    def track(
        self,
        name: str,
    ) -> Generator[None, None, None]:
        """Time a code block and accumulate the measurement.

        Args:
            name: Logical section name (e.g. ``"forward"``).

        Yields:
            Nothing; timing is recorded automatically on exit.

        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            stats = self._sections.setdefault(name, _SectionStats())
            stats.total_time += elapsed
            stats.call_count += 1
            if elapsed < stats.min_time:
                stats.min_time = elapsed
            if elapsed > stats.max_time:
                stats.max_time = elapsed

    def report(self) -> dict[str, dict[str, float]]:
        """Return a summary of all tracked sections.

        Returns:
            Mapping from section name to a dict with keys
            ``total_time``, ``call_count``, ``mean_time``,
            ``min_time``, and ``max_time``.

        """
        result: dict[str, dict[str, float]] = {}
        for name, stats in self._sections.items():
            mean = stats.total_time / stats.call_count if stats.call_count > 0 else 0.0
            result[name] = {
                "total_time": round(stats.total_time, 6),
                "call_count": float(stats.call_count),
                "mean_time": round(mean, 6),
                "min_time": round(stats.min_time, 6),
                "max_time": round(stats.max_time, 6),
            }
        logger.debug("profiling.report", sections=list(result.keys()))
        return result

    def reset(self) -> None:
        """Clear all accumulated measurements."""
        self._sections.clear()


# ------------------------------------------------------------------ #
#  Module-level tracker used by the decorator                         #
# ------------------------------------------------------------------ #

_global_tracker = PerformanceTracker()


def get_global_tracker() -> PerformanceTracker:
    """Return the module-level ``PerformanceTracker`` instance."""
    return _global_tracker


# ------------------------------------------------------------------ #
#  profile_function decorator                                         #
# ------------------------------------------------------------------ #


def profile_function(
    func: Callable[..., Any] | None = None,
    *,
    tracker: PerformanceTracker | None = None,
    name: str | None = None,
) -> Any:
    """Decorator that records each call in a ``PerformanceTracker``.

    Can be used with or without arguments::

        @profile_function
        def my_fn(): ...

        @profile_function(name="custom_name")
        def my_fn(): ...

    Args:
        func: The function to wrap (set automatically when used
            without parentheses).
        tracker: Tracker instance; defaults to the module-level
            global tracker.
        name: Override section name; defaults to
            ``"<module>.<qualname>"``.

    Returns:
        Wrapped function (or decorator if called with keyword args).

    """
    effective_tracker = tracker or _global_tracker

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        section_name = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            with effective_tracker.track(section_name):
                return fn(*args, **kwargs)

        return _wrapper

    # Support bare ``@profile_function`` (no parentheses).
    if func is not None:
        return _decorator(func)
    return _decorator


# ------------------------------------------------------------------ #
#  cprofile_section context manager                                   #
# ------------------------------------------------------------------ #


@contextmanager
def cprofile_section(
    output_path: Path | str | None = None,
    sort_by: str = "cumulative",
    top_n: int = 30,
) -> Generator[cProfile.Profile, None, None]:
    """Run ``cProfile`` over a block and optionally dump stats.

    Args:
        output_path: If given, write binary pstats to this path.
            Parent directories are created automatically.
        sort_by: Sort key for the logged summary (see
            ``pstats.SortKey``).
        top_n: Number of top entries to include in the log summary.

    Yields:
        The live ``cProfile.Profile`` object (can be inspected after
        the block).

    Example::

        with cprofile_section("profile.prof", top_n=20) as prof:
            train_one_epoch()
        # profile.prof is written; summary logged at DEBUG level.

    """
    prof = cProfile.Profile()
    prof.enable()
    try:
        yield prof
    finally:
        prof.disable()

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            prof.dump_stats(str(out))
            logger.debug(
                "profiling.cprofile_saved",
                path=str(out),
            )

        # Build a short text summary for the log.
        stats = pstats.Stats(prof)
        stats.sort_stats(sort_by)
        # pstats prints to stdout by default; capture via stream.
        import io

        stream = io.StringIO()
        stats.stream = stream  # type: ignore[attr-defined]
        stats.print_stats(top_n)
        logger.debug(
            "profiling.cprofile_summary",
            sort_by=sort_by,
            top_n=top_n,
            summary=stream.getvalue(),
        )
