#!/usr/bin/env python
"""CLI for the AlphaGalerkin codec performance benchmark.

Usage:
    # Run a smoke benchmark (CI default, ~10 seconds on CPU)
    python -m scripts.benchmark_codec run --config config/perf/smoke.yaml

    # Run with a JSON report and a regression gate vs a recorded baseline
    python -m scripts.benchmark_codec run \
        --config config/perf/default.yaml \
        --output reports/perf.json \
        --baseline docs/perf/baseline_v1.json

    # Record a fresh baseline from a benchmark run
    python -m scripts.benchmark_codec record-baseline \
        --config config/perf/default.yaml \
        --output docs/perf/baseline_v1.json \
        --hardware-tag rtx-3060-12g

The CLI is a thin shell over ``src.video_compression.perf.PerfBenchmark``;
all measurement logic lives there. This file is responsible for argument
parsing, config loading (YAML/JSON), logging configuration, and exit
codes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from src.templates.logging import configure_module_logging, create_logger_class
from src.video_compression.config import CodecConfig
from src.video_compression.perf import (
    BaselineRegistry,
    PerfBenchmark,
    PerfBenchmarkConfig,
    baseline_from_report,
    report_from_result,
)

_Logger = create_logger_class("benchmark_codec")


# ----------------------------------------------------------------- helpers


def _load_dict(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON file into a dict.

    Dispatches on suffix; we don't sniff content because misnamed files
    are a more common source of bugs than ambiguous content.
    """
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(
            f"unsupported config suffix {suffix!r}; expected .yaml/.yml/.json",
        )
    if data is None:
        raise ValueError(f"config file is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping; got {type(data).__name__}")
    return data


def _load_perf_config(path: Path) -> tuple[PerfBenchmarkConfig, CodecConfig | None]:
    """Load a benchmark config plus an optional embedded codec config.

    Convention: if the YAML has a top-level ``codec:`` key, that subtree is
    parsed as a ``CodecConfig`` and the rest is parsed as a
    ``PerfBenchmarkConfig``. Otherwise a default codec config is used.
    """
    raw = _load_dict(path)
    codec_raw = raw.pop("codec", None)
    perf_config = PerfBenchmarkConfig.model_validate(raw)
    codec_config: CodecConfig | None = None
    if codec_raw is not None:
        codec_config = CodecConfig.model_validate(codec_raw)
    return perf_config, codec_config


def _git_sha_or_empty() -> str:
    """Best-effort git SHA tag; never raises on non-git dirs."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""
    return out.stdout.strip()


# ------------------------------------------------------------- subcommands


def _cmd_run(args: argparse.Namespace) -> int:
    logger = _Logger("cli", subcommand="run")

    perf_config, codec_config = _load_perf_config(args.config)
    if args.output:
        perf_config = perf_config.with_overrides(output_path=str(args.output))
    if args.baseline:
        perf_config = perf_config.with_overrides(baseline_path=str(args.baseline))
    if args.tolerance is not None:
        perf_config = perf_config.with_overrides(
            regression_tolerance_pct=args.tolerance,
        )
    if args.device is not None:
        perf_config = perf_config.with_overrides(device_preference=args.device)

    logger.info(
        "cli.run.starting",
        config_path=str(args.config),
        output=str(args.output) if args.output else None,
        baseline=str(args.baseline) if args.baseline else None,
    )

    benchmark = PerfBenchmark(config=perf_config, codec_config=codec_config)
    result = benchmark.run()

    logger.info(
        "cli.run.completed",
        status=result.status.value,
        duration_s=result.duration_seconds,
        n_cells_total=result.metrics.get("n_cells_total"),
        n_cells_failed=result.metrics.get("n_cells_failed"),
        avg_throughput_fps=result.metrics.get("avg_throughput_fps"),
    )

    # Exit non-zero on benchmark failure (e.g. regression detected).
    return 0 if result.is_success() else 1


def _cmd_record_baseline(args: argparse.Namespace) -> int:
    logger = _Logger("cli", subcommand="record-baseline")

    perf_config, codec_config = _load_perf_config(args.config)
    # Recording always wants a clean run with no prior baseline gate.
    perf_config = perf_config.with_overrides(
        baseline_path=None,
        output_path=None,
    )

    logger.info(
        "cli.record.starting",
        config_path=str(args.config),
        output=str(args.output),
    )

    benchmark = PerfBenchmark(config=perf_config, codec_config=codec_config)
    result = benchmark.run()
    if not result.is_success():
        logger.error(
            "cli.record.benchmark_failed",
            error=result.error,
        )
        return 1

    report = report_from_result(result)
    if any(c.failed for c in report.cells):
        logger.error(
            "cli.record.has_failed_cells",
            n_failed=sum(1 for c in report.cells if c.failed),
            hint="refusing to record a baseline that contains failed cells",
        )
        return 1

    document = baseline_from_report(
        report,
        description=args.description,
        hardware_tag=args.hardware_tag,
        git_sha=_git_sha_or_empty(),
    )
    BaselineRegistry(document).save(args.output)
    logger.info(
        "cli.record.completed",
        path=str(args.output),
        n_entries=len(document.entries),
    )
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    """Diff a saved report JSON against a baseline.

    Used by CI when the benchmark and the gate run in different jobs.
    """
    logger = _Logger("cli", subcommand="diff")

    report_raw = json.loads(Path(args.report).read_text())
    # Reuse the report rehydrator from the perf package.
    from src.video_compression.perf.benchmark import _report_from_dict

    report = _report_from_dict(report_raw)

    registry = BaselineRegistry.load(args.baseline)
    diff = registry.compare_report(report, tolerance_pct=args.tolerance)
    diff_dict = diff.to_dict()
    diff_dict["baseline_path"] = str(args.baseline)

    if args.output:
        Path(args.output).write_text(json.dumps(diff_dict, indent=2, sort_keys=True))

    logger.info(
        "cli.diff.completed",
        n_diffs=diff_dict["n_diffs"],
        n_regressions=diff_dict["n_regressions"],
        n_improvements=diff_dict["n_improvements"],
    )
    return 0 if diff_dict["n_regressions"] == 0 else 1


# --------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_codec",
        description="AlphaGalerkin video codec performance benchmark",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger level (default: INFO).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Shorthand for --log-level DEBUG.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a benchmark sweep.")
    run_p.add_argument("--config", type=Path, required=True)
    run_p.add_argument("--output", type=Path, default=None)
    run_p.add_argument("--baseline", type=Path, default=None)
    run_p.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Override regression_tolerance_pct from config.",
    )
    run_p.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default=None,
        help="Override device_preference from config.",
    )
    run_p.set_defaults(func=_cmd_run)

    record_p = sub.add_parser(
        "record-baseline",
        help="Run a benchmark and persist its report as a baseline.",
    )
    record_p.add_argument("--config", type=Path, required=True)
    record_p.add_argument("--output", type=Path, required=True)
    record_p.add_argument("--description", default="")
    record_p.add_argument(
        "--hardware-tag",
        default="unknown",
        help="Free-form hardware identifier, e.g. 'rtx-3060-12g'.",
    )
    record_p.set_defaults(func=_cmd_record_baseline)

    diff_p = sub.add_parser(
        "diff",
        help="Compare a saved report JSON against a baseline.",
    )
    diff_p.add_argument("--report", type=Path, required=True)
    diff_p.add_argument("--baseline", type=Path, required=True)
    diff_p.add_argument("--tolerance", type=float, default=5.0)
    diff_p.add_argument("--output", type=Path, default=None)
    diff_p.set_defaults(func=_cmd_diff)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = "DEBUG" if args.debug else args.log_level
    configure_module_logging(level=level)

    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - exercised via tests via main()
    sys.exit(main())
