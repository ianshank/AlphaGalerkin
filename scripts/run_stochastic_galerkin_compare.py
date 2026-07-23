"""CLI for the stochastic-Galerkin (NKE) comparison scenario.

Loads a scenario YAML, runs ``stochastic_galerkin_compare``, and writes the
``results/stochastic_galerkin_compare.{csv,png}`` artifacts. Config-driven
with per-field overrides; no hardcoded budgets or paths.

Two regression-harness modes reuse ``src.poc.baselines.ScenarioBaselineRegistry``:

* ``--record-baseline PATH`` records the run's stable metrics as a baseline JSON.
* ``--baseline PATH`` diffs the run against a committed baseline and exits **1 on
  regression** (direction-aware).

Without either flag the exit code is the scenario's own acceptance threshold
(``stochastic_density_mse < stochastic_mse_gate``).

Usage:
    python -m scripts.run_stochastic_galerkin_compare \
        --config config/scenarios/stochastic_galerkin_compare_ci.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from src.poc.baselines import ScenarioBaselineRegistry
from src.poc.config import load_config_from_dict
from src.poc.logging import configure_logging
from src.poc.scenarios.stochastic_galerkin_compare import StochasticGalerkinCompareScenario
from src.poc.scenarios.stochastic_galerkin_compare_config import (
    SCENARIO_NAME,
    StochasticGalerkinCompareConfig,
)

# Only the stable arm MSEs are recorded into a regression baseline: wall-clocks
# and the derived ratio drift with hardware/BLAS; n_params is redundant with the
# config hash.
STABLE_BASELINE_METRICS: tuple[str, ...] = (
    "stochastic_density_mse",
    "deterministic_density_mse",
    "deterministic_density_mse_median",
)


def _stable_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Filter to the stable headline metrics recorded into a regression baseline."""
    return {k: v for k, v in metrics.items() if k in STABLE_BASELINE_METRICS}


def load_scenario_dict(config_path: str | Path) -> dict[str, Any]:
    """Load the ``stochastic_galerkin_compare`` scenario dict from a YAML file.

    Supports both a top-level ``scenarios:`` list and a bare single-scenario
    mapping (mirroring ``ScenarioRunner.load_config``).
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
        "n_epochs": args.n_epochs,
        "n_seeds": args.n_seeds,
        "grid_n": args.grid_n,
        "output_dir": args.output_dir,
        "device": args.device,
    }
    merged = dict(data)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def build_config(
    config_path: str | Path, args: argparse.Namespace
) -> StochasticGalerkinCompareConfig:
    """Load, override, and validate the scenario config."""
    data = apply_overrides(load_scenario_dict(config_path), args)
    config = load_config_from_dict(data, scenario_type=SCENARIO_NAME)
    # Verify by class *name* rather than isinstance (robust to dual-import under
    # some pytest modes); the dispatch itself is correct.
    if type(config).__name__ != StochasticGalerkinCompareConfig.__name__:
        raise TypeError(
            f"expected {StochasticGalerkinCompareConfig.__name__}, got {type(config).__name__}"
        )
    return cast("StochasticGalerkinCompareConfig", config)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/scenarios/stochastic_galerkin_compare_ci.yaml",
        help="Path to the scenario YAML.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the RNG seed.")
    parser.add_argument(
        "--n-epochs", dest="n_epochs", type=int, default=None, help="Override training epochs."
    )
    parser.add_argument(
        "--n-seeds", dest="n_seeds", type=int, default=None, help="Override the seed sweep size."
    )
    parser.add_argument(
        "--grid-n", dest="grid_n", type=int, default=None, help="Override the eval grid size."
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None, help="Override the artifact output dir."
    )
    parser.add_argument("--device", default=None, help="Override the device (cpu/cuda).")
    parser.add_argument("--log-level", default="INFO", help="structlog level (default INFO).")
    parser.add_argument(
        "--record-baseline",
        dest="record_baseline",
        default=None,
        help="Record this run's metrics as a baseline JSON and exit 0.",
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
        default=25.0,
        help="Per-metric regression tolerance when recording a baseline (default 25).",
    )
    parser.add_argument("--git-sha", dest="git_sha", default="", help="Provenance: commit SHA.")
    return parser


def _observed(result: Any) -> dict[str, dict[str, float]]:
    """Adapt a ScenarioResult's metrics to the ``scenario -> metric -> value`` shape."""
    return {SCENARIO_NAME: {k: float(v) for k, v in result.metrics.items()}}


def main(argv: list[str] | None = None) -> int:
    """Run the comparison scenario and return an exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level)

    config = build_config(args.config, args)
    scenario = StochasticGalerkinCompareScenario(config)
    result = scenario.run()

    print(result.summary())
    print("\nMetrics:")
    for name, value in sorted(result.metrics.items()):
        print(f"  {name}: {value:.6g}")
    print("\nArtifacts:")
    for name, path in result.artifacts.items():
        print(f"  {name}: {path}")

    observed = _observed(result)

    if args.record_baseline:
        stable = {SCENARIO_NAME: _stable_metrics(observed[SCENARIO_NAME])}
        # All recorded stable metrics are lower-better (MSEs).
        registry = ScenarioBaselineRegistry.from_observed(
            stable,
            higher_better_metrics=(),
            tolerance_pct=args.tolerance_pct,
            description="stochastic_galerkin_compare headline (stable metrics only)",
            git_sha=args.git_sha,
        )
        registry.save(args.record_baseline)
        print(f"\nBaseline recorded -> {args.record_baseline}")
        return 0

    if args.baseline:
        registry = ScenarioBaselineRegistry.load(args.baseline)
        report = registry.compare(observed, baseline_path=args.baseline)
        print("\nRegression diff vs baseline:")
        print(report.summary() if hasattr(report, "summary") else report)
        return 1 if report.has_regressions else 0

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
