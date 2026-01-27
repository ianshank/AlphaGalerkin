"""Command-line interface for validation framework.

Provides a CLI for running validation scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import structlog

from src.validation.config import (
    GPUTrainingConfig,
    MergeReadinessConfig,
    ToleranceConfig,
    TransferValidationConfig,
    ValidationConfig,
)
from src.validation.runner import ValidationRunner

logger = structlog.get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="AlphaGalerkin Validation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all validations
  python -m src.validation.cli run

  # Run specific validation
  python -m src.validation.cli run --only gpu_training

  # Run with custom config
  python -m src.validation.cli run --config config/validation.yaml

  # Check merge readiness
  python -m src.validation.cli merge-check --pr 7

  # Analyze tolerance issues
  python -m src.validation.cli tolerance-check

  # Show configuration
  python -m src.validation.cli config --show
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run validations")
    run_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    run_parser.add_argument(
        "--only",
        type=str,
        choices=["tolerance", "gpu_training", "transfer", "merge_readiness"],
        help="Run only specified validation",
    )
    run_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run validations in parallel",
    )
    run_parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop on first failure",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validation",
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Merge check command
    merge_parser = subparsers.add_parser("merge-check", help="Check PR merge readiness")
    merge_parser.add_argument(
        "--pr",
        type=int,
        required=True,
        help="PR number to check (required)",
    )
    merge_parser.add_argument(
        "--allow-failures",
        type=int,
        default=0,
        help="Maximum allowed test failures",
    )

    # Tolerance check command
    tolerance_parser = subparsers.add_parser(
        "tolerance-check",
        help="Analyze tolerance issues in tests",
    )
    tolerance_parser.add_argument(
        "--test-dir",
        type=str,
        default="tests/",
        help="Directory to scan for tests",
    )
    tolerance_parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply suggested fixes",
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration",
    )
    config_parser.add_argument(
        "--generate",
        type=str,
        metavar="FILE",
        help="Generate default configuration file",
    )

    return parser


def run_validations(args: argparse.Namespace) -> int:
    """Run validations based on arguments.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code (0 for success).
    """
    # Build config
    config_kwargs: dict[str, Any] = {
        "parallel": args.parallel,
        "stop_on_failure": args.stop_on_failure,
        "output_dir": args.output_dir,
        "verbose": args.verbose,
    }

    # Handle --only flag
    if args.only:
        config_kwargs["run_tolerance_fix"] = args.only == "tolerance"
        config_kwargs["run_gpu_training"] = args.only == "gpu_training"
        config_kwargs["run_transfer_validation"] = args.only == "transfer"
        config_kwargs["run_merge_readiness"] = args.only == "merge_readiness"

    # Load from file if provided
    if args.config:
        import yaml

        from src.validation.utils import deep_merge

        with open(args.config) as f:
            file_config = yaml.safe_load(f)
        # Use deep_merge to properly merge nested Pydantic model configs
        config_kwargs = deep_merge(file_config, config_kwargs)

    config = ValidationConfig(**config_kwargs)
    runner = ValidationRunner(config=config)

    results = runner.run_all()

    # Print summary
    print("\n" + runner.get_summary())

    # Return exit code
    failed = sum(1 for r in results.values() if not r.passed)
    return 1 if failed > 0 else 0


def run_merge_check(args: argparse.Namespace) -> int:
    """Run merge readiness check.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from src.validation.scenarios.merge_readiness import MergeReadinessChecker

    config = MergeReadinessConfig(
        pr_number=args.pr,
        max_allowed_failures=args.allow_failures,
    )

    checker = MergeReadinessChecker(config=config)
    print(checker.get_summary())

    result = checker.run()
    return 0 if result.passed else 1


def run_tolerance_check(args: argparse.Namespace) -> int:
    """Run tolerance analysis.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    from src.validation.scenarios.tolerance_fixer import ToleranceTestFixer

    config = ToleranceConfig()
    fixer = ToleranceTestFixer(config=config, test_dirs=[args.test_dir])

    result = fixer.run()

    # Print issues
    if result.details.get("issues"):
        print("\nTolerance Issues Found:")
        print("=" * 60)

        for issue in result.details["issues"]:
            print(f"\nFile: {issue['file_path']}:{issue['line_number']}")
            print(f"Test: {issue['test_name']}")
            print(f"Reason: {issue['reason']}")
            print(f"Current: rtol={issue['current_rtol']}, atol={issue['current_atol']}")
            print(f"Suggested: rtol={issue['suggested_rtol']}, atol={issue['suggested_atol']}")

        print("\n" + "=" * 60)
        print(f"Total issues: {len(result.details['issues'])}")
    else:
        print("No tolerance issues found.")

    return 0


def manage_config(args: argparse.Namespace) -> int:
    """Manage configuration.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    if args.show:
        config = ValidationConfig()
        print(json.dumps(config.model_dump(), indent=2, default=str))
        return 0

    if args.generate:
        import yaml

        config = ValidationConfig()
        with open(args.generate, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False)
        print(f"Configuration written to {args.generate}")
        return 0

    print("Use --show or --generate")
    return 1


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return run_validations(args)
    elif args.command == "merge-check":
        return run_merge_check(args)
    elif args.command == "tolerance-check":
        return run_tolerance_check(args)
    elif args.command == "config":
        return manage_config(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
