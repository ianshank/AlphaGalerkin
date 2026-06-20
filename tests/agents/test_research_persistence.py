"""Tests for research-loop result persistence (WS2 Part B)."""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.cli import _persist_research_result
from src.poc.baselines import observed_from_result_dicts
from src.templates.base import ExecutionResult, ExecutionStatus


def _result() -> ExecutionResult:
    return ExecutionResult(
        run_id="abc123",
        name="research_loop",
        status=ExecutionStatus.COMPLETED,
        metrics={"solved_fraction": 1.0, "n_problems": 3.0},
    )


def test_persist_writes_run_scoped_json(tmp_path: Path) -> None:
    path = _persist_research_result(_result(), tmp_path)
    assert path == tmp_path / "abc123" / "result.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["name"] == "research_loop"
    assert data["metrics"]["solved_fraction"] == 1.0


def test_persisted_result_is_baseline_diffable(tmp_path: Path) -> None:
    path = _persist_research_result(_result(), tmp_path)
    observed = observed_from_result_dicts([json.loads(path.read_text())])
    # ExecutionResult uses ``name`` (not ``scenario_name``) as the run label.
    assert observed["research_loop"]["solved_fraction"] == 1.0


def test_persist_round_trips_through_from_dict(tmp_path: Path) -> None:
    path = _persist_research_result(_result(), tmp_path)
    restored = ExecutionResult.from_dict(json.loads(path.read_text()))
    assert restored.run_id == "abc123"
    assert restored.metrics["n_problems"] == 3.0
