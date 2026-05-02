"""Tests for the ``report`` subcommand of ``scripts/train_compression_zoo.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from src.video_compression.config import CodecConfig
from src.video_compression.zoo.config import (
    DeviceAssignmentStrategy,
    ModelZooEntryConfig,
)
from src.video_compression.zoo.h265_baseline import (
    H265BaselineDocument,
    H265BaselineEntry,
    H265BaselineRegistry,
)
from src.video_compression.zoo.storage import VideoCodecZoo

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_compression_zoo.py"


@pytest.fixture(scope="module")
def cli_module():
    spec = importlib.util.spec_from_file_location(
        "train_compression_zoo_report_script",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.fail(f"unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_compression_zoo_report_script"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _write_codec_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(CodecConfig(name="codec_test").to_yaml_dict()),
        encoding="utf-8",
    )


def _write_manifest(
    path: Path,
    codec_path: Path,
    storage_root: Path,
    *,
    entry_specs: list[tuple[str, float, float, float]],
) -> None:
    """Write a manifest with the given entries.

    Each entry_spec is (entry_id, lambda_rd, target_bpp, target_psnr_db).
    """
    entries = [
        {
            "entry_id": eid,
            "lambda_rd": lam,
            "target_bpp": bpp,
            "target_psnr_db": psnr,
            "train_steps": 1000,
            "batch_size": 2,
            "scheduler": {"name": "scheduler", "warmup_steps": 1},
        }
        for eid, lam, bpp, psnr in entry_specs
    ]
    path.write_text(
        yaml.safe_dump(
            {
                "name": "report_smoke",
                "storage_root": str(storage_root),
                "default_codec_config_ref": str(codec_path),
                "device_preference": "cpu",
                "device_assignment_strategy": (
                    DeviceAssignmentStrategy.SINGLE_DEVICE.value
                ),
                "entries": entries,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _save_metrics_for_entries(
    storage_root: Path,
    entry_specs: list[tuple[str, float, float, float]],
    *,
    realized_offset_psnr: float = 2.0,
) -> None:
    """Save trained-entry artifacts so VideoCodecZoo can read them back.

    Realized PSNR is offset above target so the AlphaGalerkin curve sits
    above the baseline (test better than reference) by default.
    """
    zoo = VideoCodecZoo(storage_root)
    for eid, lam, bpp, psnr in entry_specs:
        e = ModelZooEntryConfig(
            entry_id=eid,
            lambda_rd=lam,
            target_bpp=bpp,
            target_psnr_db=psnr,
            train_steps=1000,
        )
        zoo.save_entry(
            e,
            {"model": {"w": torch.tensor([0.0])}},
            {
                "bpp": bpp,
                "psnr_db": psnr + realized_offset_psnr,
                "ms_ssim": 0.95,
            },
        )


def _write_baseline(
    path: Path,
    *,
    sequence_id: str = "akiyo",
    rates_psnrs: list[tuple[float, float]] | None = None,
) -> None:
    pairs = rates_psnrs or [
        (0.1, 28.0),
        (0.2, 30.0),
        (0.4, 33.0),
        (0.8, 36.0),
    ]
    entries = [
        H265BaselineEntry(
            name=f"{sequence_id}|cif|30|libx265|p{i}",
            cell_key=f"{sequence_id}|cif|30|libx265|p{i}",
            sequence_id=sequence_id,
            codec="libx265",
            crf=22 + 2 * i,
            width=352,
            height=288,
            fps=30.0,
            bpp=bpp,
            psnr_db=psnr,
        )
        for i, (bpp, psnr) in enumerate(pairs)
    ]
    doc = H265BaselineDocument(name="ref", entries=entries)
    H265BaselineRegistry(doc).save(path)


def _entry_specs_above_baseline() -> list[tuple[str, float, float, float]]:
    """Build entry specs whose realized curve sits ~2 dB above the baseline."""
    return [
        ("lambda_high_rate", 0.0016, 0.8, 36.0),
        ("lambda_mid", 0.015, 0.4, 33.0),
        ("lambda_low_rate", 0.18, 0.1, 28.0),
    ]


# --------------------------------------------------------------------------
# Argparse
# --------------------------------------------------------------------------


class TestReportArgparse:
    def test_subcommand_registered(self, cli_module) -> None:
        parser = cli_module.build_parser()
        args = parser.parse_args([
            "report",
            "--manifest", "m.yaml",
            "--baseline", "b.json",
            "--baseline-sequence-id", "akiyo",
        ])
        assert args.cmd == "report"
        assert args.baseline_sequence_id == "akiyo"
        assert args.baseline_codec == "libx265"
        assert args.primary_lambda_rd is None
        assert args.gate_pct is None
        assert args.allow_non_monotone is False

    def test_subcommand_missing_required_flags(self, cli_module) -> None:
        parser = cli_module.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["report", "--manifest", "m.yaml"])

    def test_subcommand_overrides_pass_through(self, cli_module) -> None:
        parser = cli_module.build_parser()
        args = parser.parse_args([
            "report",
            "--manifest", "m.yaml",
            "--baseline", "b.json",
            "--baseline-sequence-id", "akiyo",
            "--baseline-codec", "libaom-av1",
            "--primary-lambda-rd", "0.03",
            "--gate-pct", "-10.0",
            "--allow-non-monotone",
            "--output", "o.json",
        ])
        assert args.baseline_codec == "libaom-av1"
        assert args.primary_lambda_rd == pytest.approx(0.03)
        assert args.gate_pct == pytest.approx(-10.0)
        assert args.allow_non_monotone is True
        assert str(args.output) == "o.json"


# --------------------------------------------------------------------------
# End-to-end report command
# --------------------------------------------------------------------------


class TestReportCommand:
    def test_report_writes_json_when_test_better(
        self,
        cli_module,
        tmp_path: Path,
    ) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        baseline_path = tmp_path / "h265_baseline.json"
        storage_root = tmp_path / "zoo_store"
        output_path = tmp_path / "bd_rate_report.json"

        _write_codec_config(codec_path)
        specs = _entry_specs_above_baseline()
        _write_manifest(manifest_path, codec_path, storage_root, entry_specs=specs)
        _save_metrics_for_entries(storage_root, specs, realized_offset_psnr=4.0)
        _write_baseline(baseline_path)

        rc = cli_module.main([
            "report",
            "--manifest", str(manifest_path),
            "--baseline", str(baseline_path),
            "--baseline-sequence-id", "akiyo",
            "--primary-lambda-rd", "0.015",
            "--gate-pct", "-15.0",
            "--output", str(output_path),
        ])

        # +4 dB shift across an overlapping range should clear the -15 % gate.
        assert rc == 0
        assert output_path.exists()
        payload = json.loads(output_path.read_text())
        assert payload["gate_status"] == "passed"
        assert payload["bd_rate_pct"] < 0.0

    def test_report_returns_nonzero_when_gate_fails(
        self,
        cli_module,
        tmp_path: Path,
    ) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        baseline_path = tmp_path / "h265_baseline.json"
        storage_root = tmp_path / "zoo_store"
        output_path = tmp_path / "bd_rate_report.json"

        _write_codec_config(codec_path)
        specs = _entry_specs_above_baseline()
        _write_manifest(manifest_path, codec_path, storage_root, entry_specs=specs)
        # Realized PSNR equal to baseline → BD-rate should be ~0, gate fails.
        _save_metrics_for_entries(storage_root, specs, realized_offset_psnr=0.0)
        # Match the baseline shape one-for-one so the curves overlap fully.
        _write_baseline(
            baseline_path,
            rates_psnrs=[(0.1, 28.0), (0.4, 33.0), (0.8, 36.0)],
        )

        rc = cli_module.main([
            "report",
            "--manifest", str(manifest_path),
            "--baseline", str(baseline_path),
            "--baseline-sequence-id", "akiyo",
            "--primary-lambda-rd", "0.015",
            "--gate-pct", "-15.0",
            "--output", str(output_path),
        ])
        assert rc == 1
        payload = json.loads(output_path.read_text())
        assert payload["gate_status"] in {"failed", "skipped"}

    def test_default_output_path_under_storage_root(
        self,
        cli_module,
        tmp_path: Path,
    ) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        baseline_path = tmp_path / "h265_baseline.json"
        storage_root = tmp_path / "zoo_store"

        _write_codec_config(codec_path)
        specs = _entry_specs_above_baseline()
        _write_manifest(manifest_path, codec_path, storage_root, entry_specs=specs)
        _save_metrics_for_entries(storage_root, specs, realized_offset_psnr=4.0)
        _write_baseline(baseline_path)

        rc = cli_module.main([
            "report",
            "--manifest", str(manifest_path),
            "--baseline", str(baseline_path),
            "--baseline-sequence-id", "akiyo",
        ])
        # Default output: <storage_root>/bd_rate_report.json
        assert (storage_root / "bd_rate_report.json").exists()
        # rc could be 0 or 1 depending on default gate; we only assert
        # the default output path resolution worked.
        assert rc in (0, 1)

    def test_allow_non_monotone_flag_propagates(
        self,
        cli_module,
        tmp_path: Path,
    ) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        baseline_path = tmp_path / "h265_baseline.json"
        storage_root = tmp_path / "zoo_store"
        output_path = tmp_path / "bd_rate_report.json"

        _write_codec_config(codec_path)
        # Build a non-monotone realized curve: middle entry has higher PSNR
        # than the highest-rate entry.
        specs = [
            ("lo", 0.0016, 0.8, 36.0),
            ("mid", 0.015, 0.4, 38.0),  # spike: psnr higher than 'lo'
            ("hi", 0.18, 0.1, 28.0),
        ]
        _write_manifest(manifest_path, codec_path, storage_root, entry_specs=specs)
        _save_metrics_for_entries(storage_root, specs, realized_offset_psnr=0.0)
        _write_baseline(baseline_path)

        # Without --allow-non-monotone the report fails.
        with pytest.raises(Exception):
            cli_module.main([
                "report",
                "--manifest", str(manifest_path),
                "--baseline", str(baseline_path),
                "--baseline-sequence-id", "akiyo",
                "--output", str(output_path),
            ])

        # With the flag it succeeds (gate verdict may be anything).
        rc = cli_module.main([
            "report",
            "--manifest", str(manifest_path),
            "--baseline", str(baseline_path),
            "--baseline-sequence-id", "akiyo",
            "--allow-non-monotone",
            "--output", str(output_path),
        ])
        assert rc in (0, 1)
        assert output_path.exists()
