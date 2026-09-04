"""CLI to run the honest zero-shot-transfer comparison (operator vs retrained CNN).

Loads a scenario YAML, runs the ``transfer_baseline_compare`` scenario, and writes the
committed ``results/transfer_baseline_compare.{csv,png}`` artifacts. Config-driven with
per-field overrides; no hardcoded budgets or paths.

Two regression-harness modes reuse ``src.poc.baselines.ScenarioBaselineRegistry``:

* ``--record-baseline PATH`` records the run's metrics as a baseline JSON and exits 0.
* ``--baseline PATH`` diffs the run against a committed baseline and exits **1 on
  regression** (direction-aware). This is the CI gate — it fails on a *regression from
  the recorded number*, NOT on the operator losing to the CNN (an honest loss is still
  a green CI run, it just shows in the recorded number).

Without either flag the exit code is the scenario's own acceptance threshold
(``transfer_mse_ratio_<t>x<t> < transfer_ratio_pass_threshold``).

Usage:
    python -m scripts.run_transfer_baseline_compare \
        --config config/scenarios/transfer_baseline_compare_ci.yaml
    python -m scripts.run_transfer_baseline_compare \
        --config config/scenarios/transfer_baseline_compare_ci.yaml \
        --baseline config/baselines/transfer_ci.json          # CI regression gate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from src.poc.baselines import add_baseline_arguments, handle_baseline_flags
from src.poc.config import load_config_from_dict
from src.poc.logging import configure_logging
from src.poc.scenarios.transfer_baseline_compare import TransferBaselineCompareScenario
from src.poc.scenarios.transfer_baseline_compare_config import (
    SCENARIO_NAME,
    TransferBaselineCompareConfig,
)

# Only these stable, meaningful metrics are recorded into a regression baseline. The
# volatile spread metrics (win_fraction, seed_std/min/max) and provenance counts are
# excluded so the regression gate flags gross breakage — not cross-environment BLAS
# drift or a binary win-fraction flip on a 2-seed tripwire. Substrings are matched so
# the resolution-suffixed names (e.g. transfer_mse_ratio_19x19) are covered.
STABLE_BASELINE_SUBSTRINGS: tuple[str, ...] = (
    "transfer_mse_ratio_",
    "mse_alphagalerkin_zeroshot_",
    "mse_cnn_retrained_",
    "mse_cnn_zeroshot_",
)

#: Free-text label stored in a recorded baseline document. Named rather than
#: inlined because it is persisted: the committed
#: ``config/baselines/transfer_ci.json`` carries this exact string, so a silent
#: edit would make a re-recorded document differ from the one in git for a
#: reason unrelated to any measurement.
BASELINE_DESCRIPTION: str = "transfer_baseline_compare headline (stable metrics only)"


def _stable_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Filter to the stable headline metrics recorded into a regression baseline."""
    return {
        k: v
        for k, v in metrics.items()
        if any(sub in k for sub in STABLE_BASELINE_SUBSTRINGS)
        and not k.endswith("_matched_compute")
    }


def load_scenario_dict(config_path: str | Path) -> dict[str, Any]:
    """Load the single ``transfer_baseline_compare`` scenario dict from a YAML file.

    Supports both a top-level ``scenarios:`` list and a bare single-scenario mapping
    (mirroring ``ScenarioRunner.load_config``).
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
        "n_train_samples": args.n_train_samples,
        "target_resolution": args.target_resolution,
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
) -> TransferBaselineCompareConfig:
    """Load, override, and validate the scenario config."""
    data = apply_overrides(load_scenario_dict(config_path), args)
    config = load_config_from_dict(data, scenario_type=SCENARIO_NAME)
    # Verify by class *name* rather than isinstance (robust to dual-import under
    # some pytest modes); the dispatch itself is correct.
    if type(config).__name__ != TransferBaselineCompareConfig.__name__:
        raise TypeError(
            f"expected {TransferBaselineCompareConfig.__name__}, got {type(config).__name__}"
        )
    return cast("TransferBaselineCompareConfig", config)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/scenarios/transfer_baseline_compare_ci.yaml",
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
        "--n-train-samples",
        dest="n_train_samples",
        type=int,
        default=None,
        help="Override the per-arm training sample count.",
    )
    parser.add_argument(
        "--target-resolution",
        dest="target_resolution",
        type=int,
        default=None,
        help="Override the zero-shot target resolution.",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None, help="Override the artifact output dir."
    )
    parser.add_argument("--device", default=None, help="Override the device (cpu/cuda).")
    parser.add_argument("--log-level", default="INFO", help="structlog level (default INFO).")
    # The four baseline flags come from the shared helper rather than being
    # redeclared here, so a fix to the gate's semantics (e.g. failing on a
    # metric the run did not produce) reaches every harness at once. The
    # recorded tolerance is unchanged: 15.0 is this helper's default.
    add_baseline_arguments(parser)
    return parser


def _observed(result: Any) -> dict[str, dict[str, float]]:
    """Adapt a ScenarioResult's metrics to the ``scenario -> metric -> value`` shape."""
    return {SCENARIO_NAME: {k: float(v) for k, v in result.metrics.items()}}


def main(argv: list[str] | None = None) -> int:
    """Run the comparison scenario and return an exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level)

    config = build_config(args.config, args)
    scenario = TransferBaselineCompareScenario(config)
    result = scenario.run()

    print(result.summary())
    print("\nMetrics:")
    for name, value in sorted(result.metrics.items()):
        print(f"  {name}: {value:.6g}")
    print("\nArtifacts:")
    for name, path in result.artifacts.items():
        print(f"  {name}: {path}")

    observed = _observed(result)

    # Every recorded stable metric is lower-better: the win_fraction / seed-spread
    # metrics (the only higher-better ones) are excluded by _stable_metrics above, so
    # no higher-better override is needed here.
    baseline_exit = handle_baseline_flags(
        args,
        observed=observed,
        scenario_name=SCENARIO_NAME,
        stable_filter=_stable_metrics,
        higher_better_metrics=(),
        description=BASELINE_DESCRIPTION,
    )
    if baseline_exit is not None:
        return baseline_exit

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
