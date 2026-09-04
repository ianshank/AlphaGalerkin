"""Shared ``--record-baseline`` / ``--baseline`` CLI wiring for harness scripts.

Three harness entry points expose the same regression-gating contract:

    ``--record-baseline <path>``  write this run's stable metrics, exit 0
    ``--baseline <path>``         diff against a committed baseline, exit 1 on regression
    ``--tolerance-pct <float>``   per-metric tolerance when recording

``scripts/run_transfer_baseline_compare.py`` and
``scripts/run_stochastic_galerkin_compare.py`` each grew their own copy of the
argument definitions and the ~20-line dispatch block. Adding a third copy to
``scripts/run_lshape_amr.py`` -- which had no baseline flags at all, so the
L-shape headline could not be regression-gated from its own CLI -- would have
made three. This module is that shared implementation.

What is deliberately *not* shared: which metrics are stable, and which are
higher-better. Those are per-harness judgements (a wall-clock ratio is unstable
on the L-shape but there is no wall-clock metric in the stochastic run), so each
caller passes its own filter and its own higher-better set. Sharing the
*mechanism* while keeping the *policy* local is the split that stops one
harness's tolerance decision silently becoming another's.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from typing import Final

from src.poc.baselines.registry import ObservedMetrics, ScenarioBaselineRegistry

#: Default per-metric regression tolerance, in percent, when recording a
#: baseline. Matches the value the two pre-existing scripts already used, so
#: adopting this helper changes no recorded document.
DEFAULT_CLI_TOLERANCE_PCT: Final[float] = 15.0

#: Exit code for "no regression" (also used for a successful record).
EXIT_OK: Final[int] = 0

#: Exit code for "regression detected". Note this collides with the exit code a
#: harness returns for a failed acceptance threshold -- by design, since both
#: mean "this run is not acceptable" -- so a caller distinguishing the two must
#: read the printed report, not the code alone.
EXIT_REGRESSION: Final[int] = 1

#: Filter applied to a run's metrics before recording. Returns the subset that
#: is stable enough to gate on.
StableMetricFilter = Callable[[Mapping[str, float]], dict[str, float]]


def add_baseline_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_tolerance_pct: float = DEFAULT_CLI_TOLERANCE_PCT,
) -> argparse.ArgumentParser:
    """Add the three baseline flags plus ``--git-sha`` to *parser*.

    Args:
        parser: The harness's argument parser.
        default_tolerance_pct: Tolerance recorded when ``--tolerance-pct`` is
            not given. Surfaced as a parameter rather than fixed so a harness
            with genuinely noisier metrics can widen its own default without
            editing this module.

    Returns:
        The same parser, for chaining.

    """
    parser.add_argument(
        "--record-baseline",
        dest="record_baseline",
        default=None,
        help="Record this run's stable metrics as a baseline JSON and exit 0.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Diff this run against a committed baseline JSON; exit 1 on regression.",
    )
    parser.add_argument(
        "--tolerance-pct",
        dest="tolerance_pct",
        type=float,
        default=default_tolerance_pct,
        help=(
            "Per-metric regression tolerance when recording a baseline "
            f"(default {default_tolerance_pct:g})."
        ),
    )
    parser.add_argument("--git-sha", dest="git_sha", default="", help="Provenance: commit SHA.")
    return parser


def handle_baseline_flags(
    args: argparse.Namespace,
    *,
    observed: ObservedMetrics,
    scenario_name: str,
    stable_filter: StableMetricFilter,
    higher_better_metrics: Iterable[str] = (),
    description: str = "",
) -> int | None:
    """Act on ``--record-baseline`` / ``--baseline``; return an exit code or None.

    Args:
        args: Parsed arguments carrying the flags from
            :func:`add_baseline_arguments`.
        observed: This run's metrics, in ``{scenario: {metric: value}}`` form.
        scenario_name: Key into *observed* naming this harness's scenario.
        stable_filter: Narrows the recorded metrics to the gate-worthy subset.
            Applied on record only -- a diff compares whatever the baseline
            declares, so an unstable metric that was never recorded cannot
            later fail the gate.
        higher_better_metrics: Metric names where a larger value is an
            improvement. Everything else is treated as lower-better.
        description: Free-text label stored in the recorded document.

    Returns:
        ``EXIT_OK`` after recording, ``EXIT_REGRESSION`` if recording selected no
        metrics, ``EXIT_OK``/``EXIT_REGRESSION`` after a diff, or ``None`` when
        neither flag was given -- in which case the caller falls through to its
        own acceptance-threshold verdict. Returning ``None`` rather than
        ``EXIT_OK`` is what keeps that verdict reachable: collapsing the two
        would make every run exit 0.

    """
    if args.record_baseline:
        selected = stable_filter(observed[scenario_name])
        if not selected:
            # An empty baseline is worse than no baseline: `compare` has nothing
            # to check, so every later `--baseline` run exits 0 and the gate
            # reports green while measuring nothing. That happens silently the
            # moment a metric is renamed and `stable_filter` stops matching --
            # the same vacuity class as a coverage gate whose target is swallowed
            # by `omit`. `poc.cli record-baseline` already refuses this case; the
            # shared helper must not be the way around it.
            print(
                f"\nNo stable metrics selected for {scenario_name!r} "
                f"(of {len(observed[scenario_name])} recorded); refusing to write an "
                f"empty baseline to {args.record_baseline}."
            )
            return EXIT_REGRESSION
        stable = {scenario_name: selected}
        registry = ScenarioBaselineRegistry.from_observed(
            stable,
            higher_better_metrics=tuple(higher_better_metrics),
            tolerance_pct=args.tolerance_pct,
            description=description,
            git_sha=args.git_sha,
        )
        registry.save(args.record_baseline)
        print(f"\nBaseline recorded -> {args.record_baseline}")
        return EXIT_OK

    if args.baseline:
        registry = ScenarioBaselineRegistry.load(args.baseline)
        report = registry.compare(observed, baseline_path=args.baseline)
        print("\nRegression diff vs baseline:")
        print(report.summary() if hasattr(report, "summary") else report)
        return EXIT_REGRESSION if report.has_regressions else EXIT_OK

    return None
