r"""CLI for the MCTS-vs-classical AMR arena on ``SkfemTriSubstrate``.

Usage:
    python -m scripts.run_mcts_classical_amr_arena \
        --config config/scenarios/mcts_classical_amr_arena_ci.yaml
    python -m scripts.run_mcts_classical_amr_arena \
        --config config/scenarios/mcts_classical_amr_arena.yaml --proposal-grade

Exit code is 0 iff ``l2_error_ratio_at_matched_dof < 1.0`` (an honest FAIL is
a legitimate research outcome, not a crash). ``--proposal-grade`` additionally
rejects ``dirty: true`` / ``config_hash: unknown`` on the sidecar.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from src.poc.baselines import add_baseline_arguments, handle_baseline_flags
from src.poc.baselines.registry import ObservedMetrics
from src.poc.config import load_config_from_dict
from src.poc.logging import configure_logging
from src.poc.scenarios.mcts_classical_amr_arena import MCTSClassicalAMRArenaScenario
from src.poc.scenarios.mcts_classical_amr_arena_config import (
    SCENARIO_NAME,
    MCTSClassicalAMRArenaConfig,
)
from src.research.run_manifest import (
    ProposalGradeError,
    assert_proposal_grade,
    load_run_manifest,
    manifest_path_for,
)


def load_scenario_dict(config_path: str | Path) -> dict[str, Any]:
    """Load the arena scenario mapping from YAML (list or bare mapping)."""
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
        "n_seeds": args.n_seeds,
        "n_simulations": args.n_simulations,
        "max_dof": args.max_dof,
        "max_steps": args.max_steps,
        "output_dir": args.output_dir,
        "device": args.device,
        "require_adequacy_precondition": args.require_adequacy,
    }
    merged = dict(data)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def build_config(config_path: str | Path, args: argparse.Namespace) -> MCTSClassicalAMRArenaConfig:
    """Load, override, and validate the scenario config."""
    data = apply_overrides(load_scenario_dict(config_path), args)
    config = load_config_from_dict(data, scenario_type=SCENARIO_NAME)
    if type(config).__name__ != MCTSClassicalAMRArenaConfig.__name__:
        raise TypeError(
            f"expected {MCTSClassicalAMRArenaConfig.__name__}, got {type(config).__name__}"
        )
    return cast("MCTSClassicalAMRArenaConfig", config)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the arena CLI."""
    parser = argparse.ArgumentParser(description="MCTS vs classical AMR arena")
    parser.add_argument(
        "--config",
        default="config/scenarios/mcts_classical_amr_arena.yaml",
        help="Scenario YAML path",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-simulations", type=int, default=None)
    parser.add_argument("--max-dof", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--require-adequacy",
        dest="require_adequacy",
        action="store_true",
        default=None,
        help="Force the adequacy abort (overrides YAML false).",
    )
    parser.add_argument(
        "--skip-adequacy",
        dest="require_adequacy",
        action="store_false",
        help="Disable the adequacy abort (CI / tensor_grid hosts).",
    )
    parser.add_argument(
        "--proposal-grade",
        action="store_true",
        help="Fail if the sidecar is dirty or has config_hash unknown.",
    )
    parser.add_argument("--log-level", default="INFO")
    add_baseline_arguments(parser)
    return parser


def stable_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Drop wall-clock-derived keys from regression baselines."""
    blocked = ("wall", "time", "error_per_dof_ratio_mcts_over_dorfler")
    return {
        name: float(value)
        for name, value in metrics.items()
        if not any(token in name for token in blocked)
    }


def main(argv: list[str] | None = None) -> int:
    """Run the arena and return an exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level)
    config = build_config(args.config, args)
    scenario = MCTSClassicalAMRArenaScenario(config)
    result = scenario.run()
    print(result.summary())
    print("\nMetrics:")
    for name, value in sorted(result.metrics.items()):
        print(f"  {name}: {value:.6g}")
    print("\nArtifacts:")
    for name, path in result.artifacts.items():
        print(f"  {name}: {path}")

    if args.proposal_grade:
        csv_path = result.artifacts.get("csv")
        if not csv_path:
            print("proposal-grade: no CSV artifact", file=sys.stderr)
            return 2
        try:
            assert_proposal_grade(load_run_manifest(manifest_path_for(csv_path)))
        except ProposalGradeError as exc:
            print(f"proposal-grade: {exc}", file=sys.stderr)
            return 2

    observed: ObservedMetrics = {
        SCENARIO_NAME: {key: float(value) for key, value in result.metrics.items()}
    }
    baseline_exit = handle_baseline_flags(
        args,
        observed=observed,
        scenario_name=SCENARIO_NAME,
        stable_filter=stable_metrics,
        higher_better_metrics=("mcts_win_fraction",),
        description="mcts_classical_amr_arena (stable metrics)",
    )
    if baseline_exit is not None:
        return baseline_exit
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
