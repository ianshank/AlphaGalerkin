"""CLI to run the L-shaped Poisson MCTS-vs-Dörfler AMR comparison.

Loads a scenario YAML, runs the ``lshape_amr_compare`` scenario, and writes the
committed ``results/lshape_mcts_vs_dorfler.{csv,png}`` artifacts. Config-driven
with per-field overrides; no hardcoded budgets or paths.

Usage:
    python -m scripts.run_lshape_amr --config config/scenarios/lshape_amr_compare_demo.yaml
    python -m scripts.run_lshape_amr --config config/scenarios/lshape_amr_compare_cpu.yaml \
        --output-dir results --seed 7

Exit code is 0 iff the primary acceptance threshold
(``l2_error_ratio_at_matched_dof < 1.0``) passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from src.poc.config import load_config_from_dict
from src.poc.logging import configure_logging
from src.poc.scenarios.lshape_amr_compare import LShapeAMRCompareScenario
from src.poc.scenarios.lshape_amr_compare_config import (
    SCENARIO_NAME,
    LShapeAMRCompareConfig,
)


def load_scenario_dict(config_path: str | Path) -> dict[str, Any]:
    """Load the single ``lshape_amr_compare`` scenario dict from a YAML file.

    Supports both a top-level ``scenarios:`` list and a bare single-scenario
    mapping (mirroring ``ScenarioRunner.load_config``).

    Args:
        config_path: Path to the scenario YAML.

    Returns:
        The scenario mapping (with ``name == lshape_amr_compare``).

    Raises:
        ValueError: If no matching scenario is found in the file.

    """
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config {config_path} did not parse to a mapping")

    scenarios = raw.get("scenarios")
    if isinstance(scenarios, list):
        for entry in scenarios:
            if isinstance(entry, dict) and entry.get("name") == SCENARIO_NAME:
                return entry
        raise ValueError(f"No {SCENARIO_NAME!r} scenario found in {config_path}")
    if raw.get("name") == SCENARIO_NAME:
        return raw
    raise ValueError(f"No {SCENARIO_NAME!r} scenario found in {config_path}")


def apply_overrides(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides onto a scenario dict (only non-None values)."""
    overrides = {
        "seed": args.seed,
        "max_dof": args.max_dof,
        "n_simulations": args.n_simulations,
        "output_dir": args.output_dir,
        "device": args.device,
    }
    merged = dict(data)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def build_config(config_path: str | Path, args: argparse.Namespace) -> LShapeAMRCompareConfig:
    """Load, override, and validate the scenario config."""
    data = apply_overrides(load_scenario_dict(config_path), args)
    config = load_config_from_dict(data, scenario_type=SCENARIO_NAME)
    # ``load_config_from_dict`` dispatches by scenario name; verify by class
    # *name* rather than ``isinstance``. Under some pytest import modes the
    # config module can be imported under two keys, so identity-based checks are
    # fragile while the dispatch itself is correct.
    if type(config).__name__ != LShapeAMRCompareConfig.__name__:
        raise TypeError(f"expected {LShapeAMRCompareConfig.__name__}, got {type(config).__name__}")
    return cast("LShapeAMRCompareConfig", config)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/scenarios/lshape_amr_compare_demo.yaml",
        help="Path to the scenario YAML.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the RNG seed.")
    parser.add_argument(
        "--max-dof", dest="max_dof", type=int, default=None, help="Override max_dof."
    )
    parser.add_argument(
        "--n-simulations",
        dest="n_simulations",
        type=int,
        default=None,
        help="Override MCTS n_simulations.",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None, help="Override the artifact output dir."
    )
    parser.add_argument("--device", default=None, help="Override the device (cpu/cuda).")
    parser.add_argument("--log-level", default="INFO", help="structlog level (default INFO).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparison scenario and return an exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level)

    config = build_config(args.config, args)
    scenario = LShapeAMRCompareScenario(config)
    result = scenario.run()

    print(result.summary())
    print("\nMetrics:")
    for name, value in sorted(result.metrics.items()):
        print(f"  {name}: {value:.6g}")
    print("\nArtifacts:")
    for name, path in result.artifacts.items():
        print(f"  {name}: {path}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
