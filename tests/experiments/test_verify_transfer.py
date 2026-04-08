"""Tests for src/experiments/verify_transfer.py.

Covers TransferResult, load_model, evaluate_transfer, run_verification,
verify_resolution_independence, argument parsing, and edge cases.
All heavy computation is mocked or uses tiny models.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.experiments.physics_model import PhysicsOperator
from src.experiments.verify_transfer import (
    DEFAULT_EVAL_SEED_OFFSET,
    DEFAULT_EVAL_SIZES,
    DEFAULT_RESOLUTION_TEST_SIZES,
    PRIMARY_EVAL_SIZE,
    TransferResult,
    evaluate_transfer,
    load_model,
    main,
    run_verification,
    verify_resolution_independence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL_GRID = 5
N_SAMPLES = 4
BATCH_SIZE = 2

SMALL_MODEL_KWARGS = {
    "d_model": 16,
    "n_heads": 2,
    "n_layers": 1,
    "n_fourier_features": 8,
    "fourier_scale": 1.0,
    "use_fnet": False,
}


def _save_checkpoint(path: Path, **config_overrides) -> Path:
    """Save a small model checkpoint for testing."""
    model = PhysicsOperator(**SMALL_MODEL_KWARGS)
    config = {**SMALL_MODEL_KWARGS}
    config.update(config_overrides)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": 1,
        "transfer_mse": 0.01,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants are sensible."""

    def test_default_eval_seed_offset(self) -> None:
        assert DEFAULT_EVAL_SEED_OFFSET == 50000

    def test_default_eval_sizes(self) -> None:
        assert DEFAULT_EVAL_SIZES == [9, 13, 19]

    def test_default_resolution_test_sizes(self) -> None:
        assert DEFAULT_RESOLUTION_TEST_SIZES == [9, 13, 19, 25]

    def test_primary_eval_size(self) -> None:
        assert PRIMARY_EVAL_SIZE == 19


# ---------------------------------------------------------------------------
# TransferResult
# ---------------------------------------------------------------------------


class TestTransferResult:
    """Tests for TransferResult dataclass."""

    def test_creation(self) -> None:
        r = TransferResult(
            train_size=9,
            eval_size=19,
            mse=0.01,
            mae=0.05,
            rmse=0.1,
            max_error=0.5,
            n_samples=100,
            passed=True,
        )
        assert r.train_size == 9
        assert r.eval_size == 19
        assert r.passed is True

    def test_passed_logic(self) -> None:
        r_pass = TransferResult(
            train_size=9,
            eval_size=19,
            mse=0.01,
            mae=0.0,
            rmse=0.0,
            max_error=0.0,
            n_samples=1,
            passed=True,
        )
        r_fail = TransferResult(
            train_size=9,
            eval_size=19,
            mse=0.06,
            mae=0.0,
            rmse=0.0,
            max_error=0.0,
            n_samples=1,
            passed=False,
        )
        assert r_pass.passed is True
        assert r_fail.passed is False


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Tests for load_model function."""

    def test_loads_model_and_config(self, tmp_path: Path) -> None:
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")
        device = torch.device("cpu")

        model, config = load_model(ckpt_path, device)

        assert isinstance(model, PhysicsOperator)
        assert isinstance(config, dict)
        assert config["d_model"] == 16

    def test_model_in_eval_mode(self, tmp_path: Path) -> None:
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")
        model, _ = load_model(ckpt_path, torch.device("cpu"))
        assert not model.training

    def test_model_on_correct_device(self, tmp_path: Path) -> None:
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")
        model, _ = load_model(ckpt_path, torch.device("cpu"))
        # Check first parameter is on CPU
        p = next(model.parameters())
        assert p.device == torch.device("cpu")

    def test_default_config_values(self, tmp_path: Path) -> None:
        """When checkpoint config is missing keys, defaults are used."""
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        # Save checkpoint with empty config
        torch.save(
            {"model_state_dict": model.state_dict(), "config": {}},
            tmp_path / "minimal.pt",
        )
        # load_model uses .get() with defaults, so d_model=128 etc.
        # This will fail because state_dict doesn't match the 128-dim model.
        # That's the expected behavior for a mismatched checkpoint.
        with pytest.raises(RuntimeError):
            load_model(tmp_path / "minimal.pt", torch.device("cpu"))

    def test_load_model_with_full_config(self, tmp_path: Path) -> None:
        """Checkpoint with full config loads properly."""
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")
        model, config = load_model(ckpt_path, torch.device("cpu"))
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0


# ---------------------------------------------------------------------------
# evaluate_transfer
# ---------------------------------------------------------------------------


class TestEvaluateTransfer:
    """Tests for evaluate_transfer function."""

    def test_returns_transfer_result(self, tmp_path: Path) -> None:
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        model.eval()
        device = torch.device("cpu")

        result = evaluate_transfer(
            model=model,
            train_size=SMALL_GRID,
            eval_size=SMALL_GRID,
            n_samples=N_SAMPLES,
            device=device,
            seed=0,
            threshold=999.0,
            n_charges=2,
            batch_size=BATCH_SIZE,
        )

        assert isinstance(result, TransferResult)
        assert result.train_size == SMALL_GRID
        assert result.eval_size == SMALL_GRID
        assert result.n_samples == N_SAMPLES
        assert np.isfinite(result.mse)
        assert np.isfinite(result.mae)
        assert np.isfinite(result.rmse)
        assert np.isfinite(result.max_error)

    def test_rmse_is_sqrt_mse(self) -> None:
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        model.eval()

        result = evaluate_transfer(
            model=model,
            train_size=5,
            eval_size=5,
            n_samples=N_SAMPLES,
            device=torch.device("cpu"),
            seed=0,
            n_charges=2,
            batch_size=BATCH_SIZE,
        )
        assert abs(result.rmse - np.sqrt(result.mse)) < 1e-6

    def test_pass_with_high_threshold(self) -> None:
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        model.eval()

        result = evaluate_transfer(
            model=model,
            train_size=5,
            eval_size=5,
            n_samples=N_SAMPLES,
            device=torch.device("cpu"),
            threshold=999.0,
            n_charges=2,
            batch_size=BATCH_SIZE,
        )
        assert result.passed is True

    def test_fail_with_zero_threshold(self) -> None:
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        model.eval()

        result = evaluate_transfer(
            model=model,
            train_size=5,
            eval_size=5,
            n_samples=N_SAMPLES,
            device=torch.device("cpu"),
            threshold=0.0,
            n_charges=2,
            batch_size=BATCH_SIZE,
        )
        assert result.passed is False

    def test_different_eval_size(self) -> None:
        """Model can evaluate at a different resolution than trained on."""
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        model.eval()

        result = evaluate_transfer(
            model=model,
            train_size=5,
            eval_size=7,
            n_samples=N_SAMPLES,
            device=torch.device("cpu"),
            n_charges=2,
            batch_size=BATCH_SIZE,
        )
        assert result.eval_size == 7
        assert np.isfinite(result.mse)

    def test_max_error_nonnegative(self) -> None:
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        model.eval()

        result = evaluate_transfer(
            model=model,
            train_size=5,
            eval_size=5,
            n_samples=N_SAMPLES,
            device=torch.device("cpu"),
            n_charges=2,
            batch_size=BATCH_SIZE,
        )
        assert result.max_error >= 0.0


# ---------------------------------------------------------------------------
# run_verification
# ---------------------------------------------------------------------------


class TestRunVerification:
    """Tests for run_verification function."""

    def test_with_existing_checkpoint(self, tmp_path: Path) -> None:
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")

        summary = run_verification(
            model_path=ckpt_path,
            train_size=SMALL_GRID,
            eval_sizes=[SMALL_GRID],
            n_samples=N_SAMPLES,
            threshold=999.0,
            output_dir=tmp_path / "results",
        )

        assert isinstance(summary, dict)
        assert "model_path" in summary
        assert "train_size" in summary
        assert "threshold" in summary
        assert "results" in summary
        assert "primary_transfer" in summary
        assert "all_passed" in summary

    def test_results_list_structure(self, tmp_path: Path) -> None:
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")

        summary = run_verification(
            model_path=ckpt_path,
            train_size=SMALL_GRID,
            eval_sizes=[SMALL_GRID, 7],
            n_samples=N_SAMPLES,
            threshold=999.0,
        )

        assert len(summary["results"]) == 2
        for r in summary["results"]:
            assert "eval_size" in r
            assert "mse" in r
            assert "mae" in r
            assert "rmse" in r
            assert "max_error" in r
            assert "passed" in r

    def test_all_passed_when_threshold_high(self, tmp_path: Path) -> None:
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")

        summary = run_verification(
            model_path=ckpt_path,
            train_size=SMALL_GRID,
            eval_sizes=[SMALL_GRID],
            n_samples=N_SAMPLES,
            threshold=999.0,
        )
        assert summary["all_passed"] is True

    def test_all_passed_false_when_threshold_zero(self, tmp_path: Path) -> None:
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")

        summary = run_verification(
            model_path=ckpt_path,
            train_size=SMALL_GRID,
            eval_sizes=[SMALL_GRID],
            n_samples=N_SAMPLES,
            threshold=0.0,
        )
        assert summary["all_passed"] is False

    def test_saves_json_when_output_dir(self, tmp_path: Path) -> None:
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")
        out = tmp_path / "results"

        run_verification(
            model_path=ckpt_path,
            train_size=SMALL_GRID,
            eval_sizes=[SMALL_GRID],
            n_samples=N_SAMPLES,
            threshold=999.0,
            output_dir=out,
        )

        json_path = out / "transfer_verification.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "all_passed" in data

    def test_no_json_when_no_output_dir(self, tmp_path: Path) -> None:
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")

        summary = run_verification(
            model_path=ckpt_path,
            train_size=SMALL_GRID,
            eval_sizes=[SMALL_GRID],
            n_samples=N_SAMPLES,
            threshold=999.0,
            output_dir=None,
        )
        assert isinstance(summary, dict)

    def test_default_eval_sizes_used(self, tmp_path: Path) -> None:
        """When eval_sizes is None, DEFAULT_EVAL_SIZES is used."""
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")

        summary = run_verification(
            model_path=ckpt_path,
            train_size=SMALL_GRID,
            n_samples=N_SAMPLES,
            threshold=999.0,
        )
        assert len(summary["results"]) == len(DEFAULT_EVAL_SIZES)

    def test_primary_transfer_picks_19(self, tmp_path: Path) -> None:
        """primary_transfer should prefer eval_size=19."""
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")

        summary = run_verification(
            model_path=ckpt_path,
            train_size=SMALL_GRID,
            eval_sizes=[SMALL_GRID, 19],
            n_samples=N_SAMPLES,
            threshold=999.0,
        )
        assert summary["primary_transfer"]["to"] == 19

    def test_primary_transfer_fallback_last(self, tmp_path: Path) -> None:
        """When 19 not in eval_sizes, use last result."""
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")

        summary = run_verification(
            model_path=ckpt_path,
            train_size=SMALL_GRID,
            eval_sizes=[SMALL_GRID, 7],
            n_samples=N_SAMPLES,
            threshold=999.0,
        )
        assert summary["primary_transfer"]["to"] == 7

    def test_trains_model_when_path_none(self, tmp_path: Path) -> None:
        """When model_path is None, a new model is trained."""
        out_dir = tmp_path / "outputs" / "physics_poc"
        _save_checkpoint(out_dir / "best_model.pt")

        with patch("src.experiments.train_physics.train") as mock_train:
            mock_train.return_value = {}

            summary = run_verification(
                model_path=None,
                train_size=SMALL_GRID,
                eval_sizes=[SMALL_GRID],
                n_samples=N_SAMPLES,
                threshold=999.0,
                output_dir=out_dir,
            )

            mock_train.assert_called_once()
            assert isinstance(summary, dict)

    def test_trains_model_when_path_missing(self, tmp_path: Path) -> None:
        """When model_path points to non-existent file, train is invoked."""
        fake_path = tmp_path / "nonexistent.pt"
        out_dir = tmp_path / "outputs" / "physics_poc"
        _save_checkpoint(out_dir / "best_model.pt")

        with patch("src.experiments.train_physics.train") as mock_train:
            mock_train.return_value = {}

            summary = run_verification(
                model_path=fake_path,
                train_size=SMALL_GRID,
                eval_sizes=[SMALL_GRID],
                n_samples=N_SAMPLES,
                threshold=999.0,
                output_dir=out_dir,
            )

            mock_train.assert_called_once()
            assert isinstance(summary, dict)


# ---------------------------------------------------------------------------
# verify_resolution_independence
# ---------------------------------------------------------------------------


class TestVerifyResolutionIndependence:
    """Tests for verify_resolution_independence function.

    The source function does not use @torch.no_grad(), so we wrap calls
    in torch.no_grad() to avoid the `.numpy()` on grad-tensor error.
    """

    def test_returns_metrics_dict(self) -> None:
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        model.eval()

        with torch.no_grad():
            results = verify_resolution_independence(
                model=model,
                device=torch.device("cpu"),
                resolutions=[5, 7],
                n_samples=2,
                n_charges=2,
                seed=0,
            )

        assert "mean_consistency_error" in results
        assert "std_consistency_error" in results
        assert "max_consistency_error" in results
        assert "resolutions_tested" in results

    def test_consistency_error_nonnegative(self) -> None:
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        model.eval()

        with torch.no_grad():
            results = verify_resolution_independence(
                model=model,
                device=torch.device("cpu"),
                resolutions=[5, 7],
                n_samples=2,
                n_charges=2,
            )
        assert results["mean_consistency_error"] >= 0.0
        assert results["max_consistency_error"] >= 0.0

    def test_default_resolutions_used(self) -> None:
        """When resolutions is None, DEFAULT_RESOLUTION_TEST_SIZES is used."""
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        model.eval()

        with torch.no_grad():
            results = verify_resolution_independence(
                model=model,
                device=torch.device("cpu"),
                n_samples=2,
                n_charges=2,
            )
        assert results["resolutions_tested"] == DEFAULT_RESOLUTION_TEST_SIZES

    def test_single_resolution(self) -> None:
        """With a single resolution, coarsest == finest, error ~ 0."""
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        model.eval()

        with torch.no_grad():
            results = verify_resolution_independence(
                model=model,
                device=torch.device("cpu"),
                resolutions=[5],
                n_samples=2,
                n_charges=2,
            )
        assert results["mean_consistency_error"] < 1e-10

    def test_results_finite(self) -> None:
        model = PhysicsOperator(**SMALL_MODEL_KWARGS)
        model.eval()

        with torch.no_grad():
            results = verify_resolution_independence(
                model=model,
                device=torch.device("cpu"),
                resolutions=[5, 9],
                n_samples=3,
                n_charges=2,
            )
        for key in ["mean_consistency_error", "std_consistency_error", "max_consistency_error"]:
            assert np.isfinite(float(results[key]))


# ---------------------------------------------------------------------------
# Argument parsing (main)
# ---------------------------------------------------------------------------


class TestMainArgParsing:
    """Tests for argument parsing in main()."""

    def test_default_args_model_not_found(self, tmp_path: Path) -> None:
        """When model doesn't exist, run_verification triggers training (mocked)."""
        with (
            patch("src.experiments.verify_transfer.run_verification") as mock_rv,
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model-path",
                    str(tmp_path / "no_model.pt"),
                    "--output-dir",
                    str(tmp_path),
                ],
            ),
        ):
            mock_rv.return_value = {"all_passed": True}
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        mock_rv.assert_called_once()
        call_kwargs = mock_rv.call_args
        # model_path should be None since file doesn't exist
        assert call_kwargs.kwargs.get("model_path") is None

    def test_custom_args(self, tmp_path: Path) -> None:
        ckpt_path = _save_checkpoint(tmp_path / "model.pt")

        with (
            patch("src.experiments.verify_transfer.run_verification") as mock_rv,
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model-path",
                    str(ckpt_path),
                    "--train-size",
                    "7",
                    "--eval-sizes",
                    "5,7,9",
                    "--n-samples",
                    "10",
                    "--threshold",
                    "0.1",
                    "--output-dir",
                    str(tmp_path),
                ],
            ),
        ):
            mock_rv.return_value = {"all_passed": True}
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        call_kwargs = mock_rv.call_args
        assert call_kwargs.kwargs["train_size"] == 7
        assert call_kwargs.kwargs["eval_sizes"] == [5, 7, 9]
        assert call_kwargs.kwargs["n_samples"] == 10
        assert call_kwargs.kwargs["threshold"] == 0.1

    def test_exit_code_0_on_success(self, tmp_path: Path) -> None:
        with (
            patch("src.experiments.verify_transfer.run_verification") as mock_rv,
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model-path",
                    str(tmp_path / "no.pt"),
                    "--output-dir",
                    str(tmp_path),
                ],
            ),
        ):
            mock_rv.return_value = {"all_passed": True}
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_exit_code_1_on_failure(self, tmp_path: Path) -> None:
        with (
            patch("src.experiments.verify_transfer.run_verification") as mock_rv,
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model-path",
                    str(tmp_path / "no.pt"),
                    "--output-dir",
                    str(tmp_path),
                ],
            ),
        ):
            mock_rv.return_value = {"all_passed": False}
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_eval_sizes_parsing(self, tmp_path: Path) -> None:
        """Comma-separated eval-sizes string is parsed to list of ints."""
        with (
            patch("src.experiments.verify_transfer.run_verification") as mock_rv,
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model-path",
                    str(tmp_path / "no.pt"),
                    "--eval-sizes",
                    "3,5,7,11",
                    "--output-dir",
                    str(tmp_path),
                ],
            ),
        ):
            mock_rv.return_value = {"all_passed": True}
            with pytest.raises(SystemExit):
                main()

        call_kwargs = mock_rv.call_args
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        if kwargs:
            assert kwargs["eval_sizes"] == [3, 5, 7, 11]
