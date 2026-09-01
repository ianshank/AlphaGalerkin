"""Tests for ``src.poc.cli``'s ``eval-harness`` subcommand (``cmd_eval_harness``).

``run_eval`` is imported *lazily* inside ``cmd_eval_harness`` from
``src.integrations.eval_harness.runner`` -- the module itself is always
importable (``src.integrations.eval_harness.runner`` does not import the
optional ``eval_harness`` package at module scope), but *calling* the real
``run_eval`` needs the optional ``[eval-harness]`` git dependency, which is
not part of the ``[dev]`` extra CI installs.

These tests monkeypatch ``run_eval`` on its *source* module
(``src.integrations.eval_harness.runner``), which is exactly what the local
``from ... import run_eval`` inside ``cmd_eval_harness`` re-resolves on every
call -- so this exercises the full CLI command body hermetically, without the
optional dependency and without a Langfuse server.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pytest

import src.integrations.eval_harness.runner as eval_harness_runner
from src.poc.baselines import ScenarioBaselineRegistry
from src.poc.cli import cmd_eval_harness


@pytest.fixture(autouse=True)
def _clean_eval_output_dir_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cmd_eval_harness`` calls ``os.environ.setdefault(...)`` directly.

    That bypasses ``monkeypatch``'s own env tracking/restore, so clean up
    explicitly on both sides to keep these tests hermetic and order-independent.
    """
    monkeypatch.delenv("EVAL_OUTPUT_DIR", raising=False)
    yield
    os.environ.pop("EVAL_OUTPUT_DIR", None)


class _FakeAggregate:
    """Duck-types the fields ``cmd_eval_harness`` reads off an aggregate score."""

    def __init__(self, mean: float, pass_rate: float | None, count: int) -> None:
        self.mean = mean
        self.pass_rate = pass_rate
        self.count = count


class _FakeRunResult:
    """Duck-types ``eval_harness.core.types.RunResult`` for the fields used here."""

    def __init__(self, run_id: str, config_name: str, aggregate: dict[str, _FakeAggregate]) -> None:
        self.run_id = run_id
        self.config_name = config_name
        self.aggregate = aggregate


def test_cmd_eval_harness_prints_aggregates_without_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No ``--baseline``: prints the run id, config name, and per-score aggregates."""
    fake_result = _FakeRunResult(
        run_id="eh1",
        config_name="demo_eval",
        aggregate={
            "accuracy": _FakeAggregate(mean=0.87, pass_rate=0.9, count=10),
            "latency_ms": _FakeAggregate(mean=120.5, pass_rate=None, count=10),
        },
    )
    monkeypatch.setattr(eval_harness_runner, "run_eval", lambda config_path, offline: fake_result)
    args = argparse.Namespace(
        config="unused.yaml", online=False, baseline="", output_dir=str(tmp_path)
    )

    rc = cmd_eval_harness(args)

    assert rc == 0
    captured = capsys.readouterr()
    assert "eh1" in captured.out
    assert "demo_eval" in captured.out
    assert "accuracy" in captured.out
    assert "latency_ms" in captured.out
    assert "n/a" in captured.out  # pass_rate=None formats as "n/a"


def test_cmd_eval_harness_online_flag_maps_to_offline_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--online`` inverts to ``offline=False`` on the ``run_eval`` call."""
    recorded: dict[str, Any] = {}

    def _fake_run_eval(config_path: str, offline: bool) -> _FakeRunResult:
        recorded["config_path"] = config_path
        recorded["offline"] = offline
        return _FakeRunResult(run_id="eh2", config_name="demo", aggregate={})

    monkeypatch.setattr(eval_harness_runner, "run_eval", _fake_run_eval)
    args = argparse.Namespace(
        config="my_config.yaml", online=True, baseline="", output_dir=str(tmp_path)
    )

    rc = cmd_eval_harness(args)

    assert rc == 0
    assert recorded == {"config_path": "my_config.yaml", "offline": False}


def test_cmd_eval_harness_default_offline_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default (no ``--online``) calls ``run_eval`` with ``offline=True``."""
    recorded: dict[str, Any] = {}

    def _fake_run_eval(config_path: str, offline: bool) -> _FakeRunResult:
        recorded["offline"] = offline
        return _FakeRunResult(run_id="eh2b", config_name="demo", aggregate={})

    monkeypatch.setattr(eval_harness_runner, "run_eval", _fake_run_eval)
    args = argparse.Namespace(
        config="my_config.yaml", online=False, baseline="", output_dir=str(tmp_path)
    )

    cmd_eval_harness(args)

    assert recorded == {"offline": True}


def test_cmd_eval_harness_with_baseline_reports_regression(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--baseline`` diffs the run's persisted metrics; a regression exits 1."""
    run_id = "eh3"
    run_dir = tmp_path / "results" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "eval.json").write_text(
        json.dumps({"scenario_name": "eval_demo", "metrics": {"accuracy": 0.5}})
    )

    baseline_path = tmp_path / "baseline.json"
    registry = ScenarioBaselineRegistry.from_observed(
        {"eval_demo": {"accuracy": 0.9}},
        higher_better_metrics={"accuracy"},
        tolerance_pct=5.0,
    )
    registry.save(str(baseline_path))

    fake_result = _FakeRunResult(
        run_id=run_id,
        config_name="demo_eval",
        aggregate={"accuracy": _FakeAggregate(mean=0.5, pass_rate=0.5, count=4)},
    )
    monkeypatch.setattr(eval_harness_runner, "run_eval", lambda config_path, offline: fake_result)
    args = argparse.Namespace(
        config="unused.yaml", online=False, baseline=str(baseline_path), output_dir=str(tmp_path)
    )

    rc = cmd_eval_harness(args)

    assert rc == 1
    captured = capsys.readouterr()
    assert "regression" in captured.out


def test_cmd_eval_harness_with_baseline_clean_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run matching its baseline within tolerance exits 0."""
    run_id = "eh4"
    run_dir = tmp_path / "results" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "eval.json").write_text(
        json.dumps({"scenario_name": "eval_demo", "metrics": {"accuracy": 0.9}})
    )

    baseline_path = tmp_path / "baseline.json"
    registry = ScenarioBaselineRegistry.from_observed(
        {"eval_demo": {"accuracy": 0.9}},
        higher_better_metrics={"accuracy"},
        tolerance_pct=5.0,
    )
    registry.save(str(baseline_path))

    fake_result = _FakeRunResult(
        run_id=run_id,
        config_name="demo_eval",
        aggregate={"accuracy": _FakeAggregate(mean=0.9, pass_rate=1.0, count=4)},
    )
    monkeypatch.setattr(eval_harness_runner, "run_eval", lambda config_path, offline: fake_result)
    args = argparse.Namespace(
        config="unused.yaml", online=False, baseline=str(baseline_path), output_dir=str(tmp_path)
    )

    rc = cmd_eval_harness(args)

    assert rc == 0
