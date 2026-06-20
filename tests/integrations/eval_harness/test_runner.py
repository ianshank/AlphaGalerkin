"""Offline end-to-end run through the harness engine (requires the torch stack).

Drives the full chain — register_all -> basis_oracle dataset -> callable target
(run_basis_cell, random arm) -> scorers -> ScenarioResultSink -> RunResult — with
the dependency-free NullLangfuseClient, exactly as CPU-with-torch CI does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.integrations.eval_harness.runner import run_eval  # noqa: E402


def _config(results_dir: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "run": {"name": "basis_eval_offline", "seed": 0},
        "dataset": {
            "type": "basis_oracle",
            "params": {
                "pde_families": ["poisson"],
                "seeds": [0],
                "arm": "random",
                "max_basis_functions": 3,
                "n_candidate_bases": 8,
                "target_residual": 1e-2,
                "max_rollouts": 32,
                "n_simulations": 8,
                "topk": 3,
            },
        },
        "target": {
            "type": "callable",
            "params": {"function": "src.integrations.eval_harness.target:run_basis_cell"},
        },
        "scorers": [
            {"type": "final_residual", "params": {"target_residual": 1e-2}},
            {"type": "policy_topk", "params": {"k": 3}},
        ],
        "sinks": [{"type": "scenario_result", "params": {"output_dir": str(results_dir)}}],
        "gate": {"rules": [{"score": "final_residual", "metric": "mean", "max": 1.0}]},
    }


def test_run_eval_offline_end_to_end(tmp_path: Path) -> None:
    results_dir = tmp_path / "poc"
    config_path = tmp_path / "cfg.json"  # yaml.safe_load parses JSON
    config_path.write_text(json.dumps(_config(results_dir)))

    result = run_eval(str(config_path), offline=True)

    assert "final_residual" in result.aggregate
    assert "policy_topk" in result.aggregate
    assert result.aggregate["final_residual"].count == 1

    emitted = results_dir / "results" / result.run_id / "eval_harness.json"
    assert emitted.is_file()
    doc = json.loads(emitted.read_text())
    assert doc["scenario_name"] == "basis_eval_offline"
    assert "final_residual_mean" in doc["metrics"]
