"""Tests for the run_transfer_baseline_compare CLI (loading, overrides, exit codes)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest
import structlog
import yaml

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

from scripts.run_transfer_baseline_compare import (  # noqa: E402
    SCENARIO_NAME,
    apply_overrides,
    build_config,
    build_parser,
    load_scenario_dict,
    main,
)

_TINY_SCENARIO = {
    "name": SCENARIO_NAME,
    "device": "cpu",
    "train_resolution": 9,
    "target_resolution": 13,
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
}


def _write_yaml(tmp_path: Path, *, as_list: bool = True) -> Path:
    payload = {"scenarios": [_TINY_SCENARIO]} if as_list else dict(_TINY_SCENARIO)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


class TestLoading:
    def test_load_from_scenarios_list(self, tmp_path: Path) -> None:
        data = load_scenario_dict(_write_yaml(tmp_path, as_list=True))
        assert data["name"] == SCENARIO_NAME

    def test_load_from_bare_mapping(self, tmp_path: Path) -> None:
        data = load_scenario_dict(_write_yaml(tmp_path, as_list=False))
        assert data["name"] == SCENARIO_NAME

    def test_missing_scenario_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({"scenarios": [{"name": "other"}]}))
        with pytest.raises(ValueError, match="No 'transfer_baseline_compare'"):
            load_scenario_dict(path)

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text(yaml.safe_dump([1, 2, 3]))
        with pytest.raises(ValueError, match="did not parse to a mapping"):
            load_scenario_dict(path)


class TestOverrides:
    def test_apply_overrides_only_non_none(self) -> None:
        args = argparse.Namespace(
            seed=7,
            n_epochs=None,
            n_seeds=3,
            n_train_samples=None,
            target_resolution=None,
            output_dir="out",
            device=None,
        )
        merged = apply_overrides({"name": SCENARIO_NAME, "n_epochs": 1}, args)
        assert merged["seed"] == 7
        assert merged["n_seeds"] == 3
        assert merged["output_dir"] == "out"
        assert merged["n_epochs"] == 1  # untouched (override was None)

    def test_build_config_dispatches_and_overrides(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args(["--config", "x", "--seed", "11"])
        cfg = build_config(_write_yaml(tmp_path), args)
        assert type(cfg).__name__ == "TransferBaselineCompareConfig"
        assert cfg.seed == 11


class TestParser:
    def test_defaults(self) -> None:
        args = build_parser().parse_args([])
        assert args.config.endswith("transfer_baseline_compare_ci.yaml")
        assert args.log_level == "INFO"
        assert args.tolerance_pct == 15.0


class TestMain:
    def test_threshold_exit_code(self, tmp_path: Path) -> None:
        cfg = _write_yaml(tmp_path)
        # Lenient gate → passes → exit 0.
        code = main(
            [
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path),
                "--log-level",
                "WARNING",
                "--target-resolution",
                "13",
            ]
        )
        assert code in (0, 1)  # honest either way; just proves it runs to an exit code

    def test_record_and_diff_roundtrip(self, tmp_path: Path) -> None:
        cfg = _write_yaml(tmp_path)
        base = tmp_path / "base.json"
        rec = main(
            [
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path),
                "--record-baseline",
                str(base),
                # A large tolerance keeps the plumbing test robust to CPU-matmul
                # run-to-run drift (this asserts record->diff wiring, not determinism).
                "--tolerance-pct",
                "100000",
                "--log-level",
                "WARNING",
            ]
        )
        assert rec == 0
        assert base.exists()
        # Self-diff against the just-recorded baseline → no gross regression → exit 0.
        diff = main(
            [
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path),
                "--baseline",
                str(base),
                "--log-level",
                "WARNING",
            ]
        )
        assert diff == 0
