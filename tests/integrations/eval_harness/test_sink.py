"""Load-bearing bridge test: ScenarioResultSink -> the existing PoC baseline gate.

Proves a harness ``RunResult`` lands in the PoC results layout, that
``observed_from_result_dicts`` parses it, and that the existing
``ScenarioBaselineRegistry`` flags a seeded regression — i.e. the harness reuses
AlphaGalerkin's authoritative gate with no new gating logic. CPU; no torch.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eval_harness.core.types import RunResult, ScoreAggregate

from src.integrations.eval_harness.sink import ScenarioResultSink
from src.poc.baselines.registry import (
    ScenarioBaselineRegistry,
    observed_from_result_dicts,
)


def _run_result(run_id: str, *, residual_mean: float, pass_rate: float) -> RunResult:
    now = datetime.now(timezone.utc)
    return RunResult(
        run_id=run_id,
        config_name="basis_eval",
        items=[],
        aggregate={
            "final_residual": ScoreAggregate(count=2, mean=residual_mean, pass_rate=pass_rate),
            "policy_topk": ScoreAggregate(count=2, mean=pass_rate, pass_rate=pass_rate),
        },
        started_at=now,
        finished_at=now,
    )


def _read_results(output_dir: Path, run_id: str) -> list[dict[str, object]]:
    run_dir = output_dir / "results" / run_id
    return [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))]


def test_sink_writes_scenario_result_shape(tmp_path: Path) -> None:
    sink = ScenarioResultSink(output_dir=str(tmp_path))
    sink.emit(_run_result("run-a", residual_mean=0.02, pass_rate=0.5))

    dicts = _read_results(tmp_path, "run-a")
    assert len(dicts) == 1
    doc = dicts[0]
    assert doc["scenario_name"] == "basis_eval"
    assert doc["metrics"]["final_residual_mean"] == 0.02
    assert doc["metrics"]["final_residual_pass_rate"] == 0.5
    assert doc["metrics"]["policy_topk_pass_rate"] == 0.5


def test_sink_omits_none_pass_rate(tmp_path: Path) -> None:
    sink = ScenarioResultSink(output_dir=str(tmp_path))
    now = datetime.now(timezone.utc)
    run = RunResult(
        run_id="r-none",
        config_name="x",
        items=[],
        aggregate={"m": ScoreAggregate(count=1, mean=0.1, pass_rate=None)},
        started_at=now,
        finished_at=now,
    )
    sink.emit(run)
    doc = _read_results(tmp_path, "r-none")[0]
    assert doc["metrics"]["m_mean"] == 0.1
    assert "m_pass_rate" not in doc["metrics"]


def test_sink_output_feeds_baseline_gate_and_flags_regression(tmp_path: Path) -> None:
    sink = ScenarioResultSink(output_dir=str(tmp_path))

    # Baseline run: good residual + high pass-rate.
    sink.emit(_run_result("base", residual_mean=0.01, pass_rate=0.9))
    baseline_observed = observed_from_result_dicts(_read_results(tmp_path, "base"))
    registry = ScenarioBaselineRegistry.from_observed(
        baseline_observed,
        # *_pass_rate is higher-better; residual_mean is lower-better (default).
        higher_better_metrics={"final_residual_pass_rate", "policy_topk_pass_rate"},
        tolerance_pct=10.0,
    )

    # Self-diff is clean.
    assert not registry.compare(baseline_observed).has_regressions

    # Worse run: residual up 5x, pass-rate down — both must register as regressions.
    sink.emit(_run_result("worse", residual_mean=0.05, pass_rate=0.2))
    worse_observed = observed_from_result_dicts(_read_results(tmp_path, "worse"))
    report = registry.compare(worse_observed)
    assert report.has_regressions
    regressed = {d.metric_name for d in report.regressions}
    assert "final_residual_mean" in regressed
    assert "final_residual_pass_rate" in regressed
