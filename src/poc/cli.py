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
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.poc.logging import configure_logging
from src.poc.registry import ScenarioRegistry
from src.poc.results import ResultCollector
from src.poc.runner import ScenarioRunner

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/poc",
        help="Output directory for results",
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

    # List command
    subparsers.add_parser("list", help="List available scenarios")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show scenario details")
    info_parser.add_argument("scenario", type=str, help="Scenario name")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two runs")
    compare_parser.add_argument("run_a", type=str, help="First run ID")
    compare_parser.add_argument("run_b", type=str, help="Second run ID")

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
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
