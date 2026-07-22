"""Smoke tests for the TransferBaselineCompareScenario (mechanism only).

These run in ``test-fast`` (no ``slow`` marker). They assert the MECHANISM — the
scenario runs, records a finite ratio, evaluates the threshold, writes the CSV — but
NOT the direction of the result, so the CI fast tier is green whichever arm wins.
"""

from __future__ import annotations

import logging
import math

import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

from src.poc.config import ScenarioStatus  # noqa: E402
from src.poc.scenarios.transfer_baseline_compare import (  # noqa: E402
    TransferBaselineCompareScenario,
)
from src.poc.scenarios.transfer_baseline_compare_config import (  # noqa: E402
    TransferBaselineCompareConfig,
)


def _tiny_config(tmp_path, **overrides):  # type: ignore[no-untyped-def]
    params = {
        "target_resolution": 13,
        "train_resolution": 9,
        "secondary_resolutions": [9],
        "n_train_samples": 64,
        "n_eval_samples": 16,
        "batch_size": 16,
        "n_epochs": 1,
        "n_seeds": 2,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "n_fourier_features": 4,
        "use_fnet": False,
        "cnn_n_layers": 1,
        "cnn_channels": 4,
        "output_dir": str(tmp_path),
    }
    params.update(overrides)
    return TransferBaselineCompareConfig(**params)


class TestScenarioSmoke:
    def test_runs_and_records_ratio(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        scenario = TransferBaselineCompareScenario(_tiny_config(tmp_path))
        result = scenario.run()

        # Mechanism: completed (not ERROR), regardless of who wins.
        assert result.status in (ScenarioStatus.PASSED, ScenarioStatus.FAILED)

        ratio = result.metrics.get("transfer_mse_ratio_13x13")
        assert ratio is not None and math.isfinite(ratio) and ratio > 0

        # Both the gated ratio and the matched-compute variant present.
        assert "transfer_mse_ratio_13x13_matched_compute" in result.metrics
        assert "mse_alphagalerkin_zeroshot_13x13" in result.metrics
        assert "mse_cnn_retrained_13x13" in result.metrics
        assert "param_count_ratio" in result.metrics

        # Threshold was evaluated for the gated metric.
        assert "transfer_mse_ratio_13x13" in result.threshold_results

        # CSV artifact written.
        csv_path = tmp_path / "transfer_baseline_compare.csv"
        assert csv_path.exists()
        assert "csv" in result.artifacts

    def test_passed_matches_threshold_direction(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        # A deliberately lenient gate makes any finite ratio pass — proves the gate wiring.
        scenario = TransferBaselineCompareScenario(
            _tiny_config(tmp_path, transfer_ratio_pass_threshold=100.0)
        )
        result = scenario.run()
        assert result.passed is True
        assert result.threshold_results["transfer_mse_ratio_13x13"] is True

    def test_failed_when_gate_impossible(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        # A gate of ~0 can never be met by a positive ratio → honest FAILED, not ERROR.
        scenario = TransferBaselineCompareScenario(
            _tiny_config(tmp_path, transfer_ratio_pass_threshold=1e-9)
        )
        result = scenario.run()
        assert result.status == ScenarioStatus.FAILED
        assert result.passed is False
