"""Tests for the operator-vs-CNN transfer comparison harness.

Uses tiny configs so the real training loops run in well under a second per model.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import structlog
import torch

# Keep the per-forward DEBUG logging out of the test output (and marginally faster).
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

from src.research.transfer_baseline_compare import (  # noqa: E402
    SEED_PRIME_STRIDE,
    MultiSeedTransferComparison,
    TransferComparisonParams,
    build_cnn,
    build_operator,
    evaluate_on_grid,
    export_csv,
    export_plot,
    resolve_cnn_channels,
    resolved_seeds,
    run_multiseed_transfer_comparison,
    run_transfer_comparison,
)


def _tiny(**overrides: object) -> TransferComparisonParams:
    base: dict[str, object] = {
        "seed": 0,
        "device": "cpu",
        "train_resolution": 9,
        "target_resolution": 13,
        "secondary_resolutions": (9,),
        "n_train_samples": 32,
        "n_eval_samples": 16,
        "n_charges": 5,
        "batch_size": 16,
        "n_epochs": 1,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "n_fourier_features": 4,
        "use_fnet": False,
        "cnn_n_layers": 1,
        "cnn_kernel_size": 3,
        "cnn_channels": 4,
        "matched_budget_mode": "grad_steps",
        "n_seeds": 2,
    }
    base.update(overrides)
    return TransferComparisonParams(**base)  # type: ignore[arg-type]


class TestParamsValidation:
    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("train_resolution", 2, "train_resolution"),
            ("target_resolution", 4, "target_resolution"),
            ("n_seeds", 0, "n_seeds"),
            ("batch_size", 0, "batch_size"),
            ("n_epochs", 0, "n_epochs"),
            ("matched_budget_mode", "bogus", "matched_budget_mode"),
        ],
    )
    def test_rejects(self, field: str, value: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _tiny(**{field: value})

    def test_all_resolutions_dedups_and_sorts(self) -> None:
        p = _tiny(target_resolution=19, secondary_resolutions=(13, 9, 19))
        assert p.all_resolutions == [9, 13, 19]


class TestSeedDerivation:
    def test_resolved_seeds(self) -> None:
        assert resolved_seeds(42, 3) == [42, 42 + SEED_PRIME_STRIDE, 42 + 2 * SEED_PRIME_STRIDE]

    def test_resolved_seeds_single(self) -> None:
        assert resolved_seeds(5, 1) == [5]


class TestBuildersAndEval:
    def test_resolve_cnn_channels_explicit(self) -> None:
        assert resolve_cnn_channels(_tiny(cnn_channels=17), 999_999) == 17

    def test_resolve_cnn_channels_auto_matches(self) -> None:
        p = _tiny(cnn_channels=None)
        op = build_operator(p, torch.device("cpu"))
        n = sum(x.numel() for x in op.parameters())
        channels = resolve_cnn_channels(p, n)
        assert channels >= 1

    def test_build_cnn_and_operator_types(self) -> None:
        p = _tiny()
        assert build_operator(p, torch.device("cpu")).__class__.__name__ == "PhysicsOperator"
        assert build_cnn(p, 4, torch.device("cpu")).__class__.__name__ == "DiscreteCNNBaseline"

    def test_shared_eval_set_is_seed_independent(self) -> None:
        """AC2: eval seed = eval_seed_base + resolution, independent of the training seed."""
        p = _tiny()
        cnn = build_cnn(p, 4, torch.device("cpu"))
        mse_a = evaluate_on_grid(cnn, 13, _tiny(seed=1), torch.device("cpu"), forward="cnn")
        mse_b = evaluate_on_grid(cnn, 13, _tiny(seed=2), torch.device("cpu"), forward="cnn")
        assert mse_a == pytest.approx(mse_b)


class TestSingleSeed:
    def test_run_produces_all_arms_and_ratio(self) -> None:
        result = run_transfer_comparison(_tiny())
        assert result.operator.arm == "operator"
        assert result.cnn_retrained.arm == "cnn_retrained"
        assert result.cnn_zeroshot.arm == "cnn_zeroshot"
        assert np.isfinite(result.transfer_mse_ratio)
        assert result.transfer_mse_ratio > 0
        # Operator curve covers all evaluated resolutions.
        assert set(result.operator.mse_by_resolution) == {9, 13}

    def test_grad_steps_matched_compute_aliases_retrained(self) -> None:
        result = run_transfer_comparison(_tiny(matched_budget_mode="grad_steps"))
        assert result.cnn_matched_compute.mse_at_target == result.cnn_retrained.mse_at_target
        assert result.transfer_mse_ratio == pytest.approx(result.transfer_mse_ratio_matched_compute)

    def test_wall_clock_matched_compute_runs(self) -> None:
        result = run_transfer_comparison(_tiny(matched_budget_mode="wall_clock"))
        assert np.isfinite(result.transfer_mse_ratio_matched_compute)
        assert result.cnn_matched_compute.arm == "cnn_matched_compute"

    def test_metrics_keys_are_resolution_suffixed(self) -> None:
        m = run_transfer_comparison(_tiny()).metrics()
        for key in (
            "transfer_mse_ratio_13x13",
            "transfer_mse_ratio_13x13_matched_compute",
            "mse_alphagalerkin_zeroshot_13x13",
            "mse_cnn_retrained_13x13",
            "mse_cnn_zeroshot_13x13",
            "param_count_ratio",
            "mse_9x9",
            "mse_13x13",
        ):
            assert key in m, key

    def test_deterministic(self) -> None:
        r1 = run_transfer_comparison(_tiny(seed=3))
        r2 = run_transfer_comparison(_tiny(seed=3))
        assert r1.transfer_mse_ratio == pytest.approx(r2.transfer_mse_ratio)


class TestMultiSeed:
    def test_median_and_spread(self) -> None:
        comp = run_multiseed_transfer_comparison(_tiny(n_seeds=3))
        assert len(comp.per_seed) == 3
        assert comp.seeds == resolved_seeds(0, 3)
        m = comp.metrics()
        assert m["n_seeds"] == 3.0
        assert m["transfer_mse_ratio_13x13"] == pytest.approx(float(np.median(comp.ratios)))
        assert 0.0 <= m["alphagalerkin_win_fraction"] <= 1.0
        assert m["transfer_ratio_seed_min"] <= m["transfer_ratio_seed_max"]

    def test_representative_is_median_seed(self) -> None:
        comp = run_multiseed_transfer_comparison(_tiny(n_seeds=3))
        rep = comp.representative
        assert rep in comp.per_seed


class TestArtifacts:
    def test_export_csv_rows(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        comp = run_multiseed_transfer_comparison(_tiny(n_seeds=2))
        path = export_csv(comp, tmp_path / "t.csv")
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        # header + per-seed × arm × resolution rows (operator: 2 res, others: 1 each).
        assert lines[0].startswith("problem,seed,arm,resolution")
        assert len(lines) > 1
        assert any(",operator," in line for line in lines)
        assert any(",cnn_retrained," in line for line in lines)

    def test_export_plot(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        comp = run_multiseed_transfer_comparison(_tiny(n_seeds=2))
        path = export_plot(comp, tmp_path / "t.png")
        # matplotlib is a test dependency; the PNG should be written.
        assert path is not None and path.exists()

    def test_multiseed_dataclass_shapes(self) -> None:
        comp = run_multiseed_transfer_comparison(_tiny(n_seeds=2))
        assert isinstance(comp, MultiSeedTransferComparison)
        assert len(comp.ratios) == 2
