"""Smoke tests for the λ-scheduling artifact CLI (negative-result harness)."""

from __future__ import annotations

from pathlib import Path

from scripts.run_lambda_scheduling import main, run
from src.thermo.config import LambdaSchedulingConfig


def test_run_writes_artifacts_and_reports_negative(tmp_path: Path) -> None:
    config = LambdaSchedulingConfig(
        name="thermo_test",
        n_initial_windows=4,
        max_windows=8,
        batch_samples=100,
        sample_budget=1000,
        n_seeds=2,
        n_simulations=4,
        error_tolerance=0.05,
        surrogate_bias_sweep=[0.0, 0.25],
        primary_bias=0.25,
    )
    headline = run(config, tmp_path)
    assert (tmp_path / "lambda_scheduling.png").exists()
    assert (tmp_path / "lambda_scheduling.csv").exists()
    # Both bias levels reported; the negative result means MCTS >= greedy (ratio ~>=1).
    assert set(headline) == {
        "final_ratio_mcts_over_greedy_bias_0",
        "final_ratio_mcts_over_greedy_bias_0.25",
    }
    assert all(v > 0.0 for v in headline.values())


def test_main_cli(tmp_path: Path) -> None:
    rc = main(
        [
            "--output-dir",
            str(tmp_path),
            "--n-seeds",
            "2",
            "--n-simulations",
            "4",
            "--sample-budget",
            "1000",
        ]
    )
    assert rc == 0
    assert (tmp_path / "lambda_scheduling.csv").exists()
