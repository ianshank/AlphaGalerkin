r"""CLI for the Gate 1 look-ahead-vs-greedy go/no-go (pre-registered).

Usage:
    python -m scripts.run_lookahead_vs_greedy \
        --config config/scenarios/lookahead_vs_greedy_lshape.yaml --proposal-grade
    python -m scripts.run_lookahead_vs_greedy \
        --config config/scenarios/lookahead_vs_greedy_zshape.yaml --proposal-grade

The pre-registration is ``specs/lookahead_vs_greedy.spec.md``; the YAMLs restate
its values. The exit code reports the *process*, never the verdict -- a NO-GO is
a research outcome, not a failure:

    0  completed (GO or NO-GO; the verdict is printed and recorded in the sidecar)
    1  error (invalid config, invalid run, unexpected exception)
    2  not proposal-grade (``--proposal-grade`` on a tree that is not provably clean)
    3  the testbed aborted on its adequacy precondition (no comparison ran)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Final, cast

import yaml

from src.poc.config import ScenarioResult, ScenarioStatus, load_config_from_dict
from src.poc.logging import configure_logging
from src.poc.scenarios.lookahead_vs_greedy import LookaheadVsGreedyScenario
from src.poc.scenarios.lookahead_vs_greedy_config import SCENARIO_NAME, LookaheadVsGreedyConfig
from src.research.lookahead_vs_greedy import ABORTED, ADEQUACY_ABORTED_METRIC
from src.research.run_manifest import (
    ProposalGradeError,
    RunRecorder,
    assert_proposal_grade,
    load_run_manifest,
    manifest_path_for,
)

EXIT_COMPLETED: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_NOT_PROPOSAL_GRADE: Final[int] = 2
EXIT_ADEQUACY_ABORT: Final[int] = 3
DEFAULT_CONFIG: Final[str] = "config/scenarios/lookahead_vs_greedy_lshape.yaml"


def load_scenario_dict(config_path: str | Path) -> dict[str, Any]:
    """The ``lookahead_vs_greedy`` mapping from a YAML (``scenarios:`` list or bare mapping)."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config {config_path} did not parse to a mapping")
    scenarios = raw.get("scenarios")
    candidates = scenarios if isinstance(scenarios, list) else [raw]
    for entry in candidates:
        if isinstance(entry, dict) and entry.get("name") == SCENARIO_NAME:
            return dict(entry)
    raise ValueError(f"no {SCENARIO_NAME!r} scenario found in {config_path}")


def apply_overrides(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Overlay the CLI's non-``None`` overrides on the scenario mapping."""
    overrides = {
        "seed": args.seed,
        "robust_n_seeds": args.n_seeds,
        "output_dir": args.output_dir,
    }
    merged = dict(data)
    merged.update({key: value for key, value in overrides.items() if value is not None})
    return merged


def build_config(config_path: str | Path, args: argparse.Namespace) -> LookaheadVsGreedyConfig:
    """Load, override and validate the testbed config."""
    data = apply_overrides(load_scenario_dict(config_path), args)
    config = load_config_from_dict(data, scenario_type=SCENARIO_NAME)
    # By name, not isinstance: the scenario package can be re-imported under test.
    if type(config).__name__ != LookaheadVsGreedyConfig.__name__:
        raise TypeError(f"expected {LookaheadVsGreedyConfig.__name__}, got {type(config).__name__}")
    return cast("LookaheadVsGreedyConfig", config)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the Gate 1 CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Gate 1 go/no-go: MCTS look-ahead vs the best classical marking policy "
            "(greedy, Dörfler, uniform) on one pre-registered testbed. Exit code: 0 "
            "completed (GO or NO-GO), 1 error, 2 not proposal-grade, 3 adequacy abort."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Testbed YAML path.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir.")
    parser.add_argument("--seed", type=int, default=None, help="Override the base seed.")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Override robust_n_seeds (C3 needs min_robust_wins <= this).",
    )
    parser.add_argument(
        "--proposal-grade",
        action="store_true",
        help="Refuse to start on a tree that is not provably clean; re-verify the sidecar.",
    )
    parser.add_argument("--log-level", default="INFO", help="structlog level.")
    return parser


def _print_result(result: ScenarioResult) -> None:
    """Human-readable summary: status, gated metrics, artifacts."""
    print(result.summary())
    print("\nMetrics:")
    for name, value in sorted(result.metrics.items()):
        print(f"  {name}: {value:.6g}")
    print("\nArtifacts:")
    for name, path in result.artifacts.items():
        print(f"  {name}: {path}")


def exit_code_for(result: ScenarioResult, *, proposal_grade: bool) -> int:
    """Map a finished scenario to the CLI's exit code (never to the verdict)."""
    if result.metrics.get(ADEQUACY_ABORTED_METRIC) == ABORTED:
        print("adequacy precondition failed: testbed aborted, no comparison ran", file=sys.stderr)
        return EXIT_ADEQUACY_ABORT
    if result.status is ScenarioStatus.ERROR:
        print(f"error: {result.error_message}", file=sys.stderr)
        return EXIT_ERROR
    if proposal_grade:
        csv_path = result.artifacts.get("csv")
        if not csv_path:
            print("proposal-grade: no CSV artifact", file=sys.stderr)
            return EXIT_NOT_PROPOSAL_GRADE
        try:
            assert_proposal_grade(load_run_manifest(manifest_path_for(csv_path)))
        except ProposalGradeError as exc:
            print(f"proposal-grade: {exc}", file=sys.stderr)
            return EXIT_NOT_PROPOSAL_GRADE
    return EXIT_COMPLETED


def main(argv: list[str] | None = None) -> int:
    """Run one testbed; the exit code reports the process, not the verdict."""
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level)
    try:
        config = build_config(args.config, args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.proposal_grade:
        # Pre-flight only: the scenario takes its own snapshot before computing.
        try:
            RunRecorder.start(config_hash=config.compute_hash(), proposal_grade=True)
        except ProposalGradeError as exc:
            print(f"proposal-grade: {exc}", file=sys.stderr)
            return EXIT_NOT_PROPOSAL_GRADE
    result = LookaheadVsGreedyScenario(config, proposal_grade=args.proposal_grade).run()
    _print_result(result)
    return exit_code_for(result, proposal_grade=args.proposal_grade)


if __name__ == "__main__":
    sys.exit(main())
