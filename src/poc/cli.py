#!/usr/bin/env python3
"""CLI entry point for PoC scenario framework.

Usage:
    # Run all scenarios
    python -m src.poc.cli run

    # Run specific scenario
    python -m src.poc.cli run --scenario transfer

    # Run from config file
    python -m src.poc.cli run --config config/scenarios/poc_full.yaml

    # List available scenarios
    python -m src.poc.cli list

    # Show scenario details
    python -m src.poc.cli info transfer

    # Compare two runs
    python -m src.poc.cli compare run_a run_b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import structlog

from src.poc.logging import configure_logging
from src.poc.registry import ScenarioRegistry
from src.poc.results import ResultCollector
from src.poc.runner import ScenarioRunner

logger = structlog.get_logger(__name__)

# Metric-name suffixes whose larger value is *better*. Used to record metric
# direction in a baseline without hardcoding a per-metric table; extend at the
# CLI with ``--higher-better``. Names are matched by suffix because centaur
# metrics are arm-prefixed (e.g. ``random_residual_fit_r2``).
DEFAULT_HIGHER_BETTER_SUFFIXES: tuple[str, ...] = (
    "_fit_r2",
    "_r2",
    "solved_fraction",
    "_reduction_pct",
    "accept_rate",
)


def _load_run_result_dicts(output_dir: str, run_id: str) -> list[dict[str, Any]]:
    """Read every result JSON written under a run id, across both layouts.

    Supports the two on-disk layouts the harness writes:

    - PoC scenarios (``src/poc/results.py``):
      ``{output_dir}/results/{run_id}/*.json``
    - Research loop (``src/agents/cli.py``):
      ``{output_dir}/{run_id}/result.json``

    The PoC layout is preferred when present; otherwise the research layout is
    used (so ``--output-dir outputs/agents/research --run-id <id>`` records a
    research-loop run). Raises ``FileNotFoundError`` when neither layout has a
    run dir, and ``ValueError`` for a corrupt result file, so the CLI fails
    loud rather than recording an empty or partial baseline.
    """
    base = Path(output_dir)
    poc_dir = base / "results" / run_id
    research_dir = base / run_id
    if poc_dir.is_dir():
        run_dir = poc_dir
    elif research_dir.is_dir():
        run_dir = research_dir
    else:
        raise FileNotFoundError(
            f"no results found for run_id {run_id!r} under {poc_dir} or {research_dir}"
        )
    dicts: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"result file {path} is not valid JSON: {exc}") from exc
        if isinstance(raw, dict):
            dicts.append(raw)
    return dicts


def _resolve_higher_better(
    observed: dict[str, dict[str, float]],
    *,
    extra_names: list[str],
    extra_suffixes: list[str],
) -> set[str]:
    """Pick the exact metric names recorded as higher-better.

    A metric is higher-better if its name ends with one of the default
    suffixes (or an extra suffix), or is named explicitly via ``extra_names``.
    """
    suffixes = (*DEFAULT_HIGHER_BETTER_SUFFIXES, *extra_suffixes)
    explicit = set(extra_names)
    higher: set[str] = set()
    for scenario_metrics in observed.values():
        for metric in scenario_metrics:
            if metric in explicit or any(metric.endswith(s) for s in suffixes):
                higher.add(metric)
    return higher


def register_builtin_scenarios() -> None:
    """Register all built-in scenarios."""
    # Import to trigger @scenario decorator registration
    from src.poc.scenarios import (  # noqa: F401
        ComplexityScenario,
        StabilityScenario,
        TransferScenario,
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run scenarios."""
    register_builtin_scenarios()

    runner = ScenarioRunner(
        output_dir=args.output_dir,
        max_workers=args.parallel,
        fail_fast=args.fail_fast,
    )

    if args.config:
        # Run from config file
        results = runner.run_from_config(args.config)
    elif args.scenario:
        # Run specific scenario
        result = runner.run(
            args.scenario,
            name=args.scenario,
            description=f"CLI run of {args.scenario}",
        )
        results = [result]
    else:
        # Run all scenarios
        results = runner.run_all(filter_tier=args.tier)

    # Return exit code based on results
    if all(r.passed for r in results):
        return 0
    return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List available scenarios."""
    register_builtin_scenarios()

    registry = ScenarioRegistry()
    scenarios = registry.get_all()

    if not scenarios:
        print("No scenarios registered.")
        return 0

    print("\nAvailable Scenarios:")
    print("=" * 60)

    for name, scenario_cls in sorted(scenarios.items()):
        # Get default config for description
        try:
            scenario = scenario_cls(name=name, description="temp")
            desc = scenario.config.description
            tier = scenario.config.tier.value
        except Exception:
            desc = "(no description)"
            tier = "unknown"

        print(f"\n  {name}")
        print(f"    Tier: {tier}")
        print(f"    {desc}")

    print("\n" + "=" * 60)
    print(f"Total: {len(scenarios)} scenarios")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed scenario information."""
    register_builtin_scenarios()

    registry = ScenarioRegistry()
    scenario_cls = registry.get(args.scenario)

    if not scenario_cls:
        print(f"Scenario '{args.scenario}' not found.")
        print(f"Available: {registry.list_scenarios()}")
        return 1

    # Create instance with defaults
    try:
        scenario = scenario_cls(name=args.scenario, description="info query")
        config = scenario.config
    except Exception as e:
        print(f"Error creating scenario: {e}")
        return 1

    print(f"\nScenario: {args.scenario}")
    print("=" * 60)
    print(f"Description: {config.description}")
    print(f"Tier: {config.tier.value}")
    print(f"Config class: {scenario_cls.config_class.__name__}")
    print()

    # Print config fields
    print("Configuration Fields:")
    print("-" * 40)

    for field_name, field_info in config.model_fields.items():
        default = getattr(config, field_name)
        field_type = field_info.annotation
        print(f"  {field_name}: {field_type}")
        print(f"    Default: {default}")
        if field_info.description:
            print(f"    Description: {field_info.description}")

    print()
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two runs."""
    collector = ResultCollector(output_dir=args.output_dir)

    comparison = collector.compare_runs(args.run_a, args.run_b)

    print(f"\nComparing: {args.run_a} vs {args.run_b}")
    print("=" * 60)

    for comp in comparison["comparisons"]:
        scenario = comp["scenario"]
        print(f"\n{scenario}:")

        if "only_in" in comp:
            print(f"  Only in run {comp['only_in']}")
            continue

        if comp.get("status_changed"):
            print(f"  Status: {comp['status_a']} -> {comp['status_b']}")
        else:
            print(f"  Status: {comp.get('status_a', 'unknown')} (unchanged)")

        if "metric_changes" in comp:
            for metric, changes in comp["metric_changes"].items():
                delta = changes["delta"]
                pct = changes["pct_change"]
                direction = "better" if delta < 0 else "worse" if delta > 0 else "same"
                print(
                    f"  {metric}: {changes['a']:.4f} -> {changes['b']:.4f} "
                    f"({delta:+.4f}, {pct:+.1f}%, {direction})"
                )

    print("\n" + "=" * 60)
    return 0


def cmd_record_baseline(args: argparse.Namespace) -> int:
    """Record a headline baseline document from a completed run's metrics."""
    from src.poc.baselines import ScenarioBaselineRegistry, observed_from_result_dicts

    result_dicts = _load_run_result_dicts(args.output_dir, args.run_id)
    observed = observed_from_result_dicts(result_dicts)
    if not any(observed.values()):
        print(f"No numeric metrics found for run {args.run_id!r}; nothing to record.")
        return 1
    higher_better = _resolve_higher_better(
        observed,
        extra_names=_split_csv(args.higher_better),
        extra_suffixes=_split_csv(args.higher_better_suffix),
    )
    registry = ScenarioBaselineRegistry.from_observed(
        observed,
        higher_better_metrics=higher_better,
        tolerance_pct=args.tolerance_pct,
        description=args.description,
        hardware_tag=args.hardware_tag,
        git_sha=args.git_sha,
        llm_backend=args.llm_backend,
    )
    registry.save(args.out)
    n_entries = len(registry.document.entries)
    print(f"Recorded {n_entries} metric entries to {args.out} (run {args.run_id}).")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Diff a completed run's metrics against a recorded baseline.

    Exit code is non-zero when any metric regressed beyond tolerance, so this
    is usable as a CI gate.
    """
    from src.poc.baselines import ScenarioBaselineRegistry, observed_from_result_dicts

    registry = ScenarioBaselineRegistry.load(args.baseline)
    result_dicts = _load_run_result_dicts(args.output_dir, args.run_id)
    observed = observed_from_result_dicts(result_dicts)
    report = registry.compare(observed, baseline_path=str(args.baseline))

    print(f"\nBaseline diff: {args.baseline} vs run {args.run_id}")
    print("=" * 60)
    for diff in report.diffs:
        print(
            f"  [{diff.status:>9}] {diff.key}: "
            f"{diff.baseline_value:.6g} -> {diff.observed_value:.6g} "
            f"({diff.delta_pct:+.1f}% vs ±{diff.tolerance_pct:.1f}%)"
        )
    if report.missing_in_observed:
        print(f"\n  Missing in run: {', '.join(report.missing_in_observed)}")
    print("\n" + "=" * 60)
    print(f"{len(report.regressions)} regression(s), {len(report.improvements)} improvement(s).")
    return 1 if report.has_regressions else 0


def _split_csv(value: str | None) -> list[str]:
    """Split a comma-separated CLI value into a trimmed, non-empty list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AlphaGalerkin PoC Scenario Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run scenarios")
    run_parser.add_argument(
        "--scenario",
        type=str,
        help="Specific scenario to run",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    run_parser.add_argument(
        "--tier",
        choices=["unit", "functional", "integration"],
        help="Filter scenarios by tier",
    )
    run_parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    run_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/poc",
        help="Output directory for results",
    )

    # List command
    subparsers.add_parser("list", help="List available scenarios")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show scenario details")
    info_parser.add_argument("scenario", type=str, help="Scenario name")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two runs")
    compare_parser.add_argument("run_a", type=str, help="First run ID")
    compare_parser.add_argument("run_b", type=str, help="Second run ID")
    compare_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/poc",
        help="Output directory for results",
    )

    # Record-baseline command
    record_parser = subparsers.add_parser(
        "record-baseline", help="Record a headline baseline from a run's metrics"
    )
    record_parser.add_argument("--run-id", type=str, required=True, help="Run id to record from")
    record_parser.add_argument("--out", type=str, required=True, help="Baseline JSON output path")
    record_parser.add_argument(
        "--output-dir", type=str, default="outputs/poc", help="Results directory"
    )
    record_parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=10.0,
        help="Per-metric regression tolerance to record (percent).",
    )
    record_parser.add_argument(
        "--higher-better",
        type=str,
        default="",
        help="Comma-separated exact metric names whose larger value is better.",
    )
    record_parser.add_argument(
        "--higher-better-suffix",
        type=str,
        default="",
        help="Comma-separated extra metric-name suffixes treated as higher-better.",
    )
    record_parser.add_argument("--description", type=str, default="", help="Baseline note.")
    record_parser.add_argument("--hardware-tag", type=str, default="", help="Hardware identifier.")
    record_parser.add_argument("--git-sha", type=str, default="", help="Commit sha provenance.")
    record_parser.add_argument("--llm-backend", type=str, default="", help="LLM backend used.")

    # Diff command
    diff_parser = subparsers.add_parser(
        "diff", help="Diff a run's metrics against a recorded baseline (CI gate)"
    )
    diff_parser.add_argument("--baseline", type=str, required=True, help="Baseline JSON path")
    diff_parser.add_argument("--run-id", type=str, required=True, help="Run id to compare")
    diff_parser.add_argument(
        "--output-dir", type=str, default="outputs/poc", help="Results directory"
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(level=args.log_level)

    # Dispatch to command handler
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "record-baseline":
        return cmd_record_baseline(args)
    elif args.command == "diff":
        return cmd_diff(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
