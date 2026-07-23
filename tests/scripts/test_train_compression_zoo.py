"""Tests for ``scripts/train_compression_zoo.py`` (multi-entry CLI)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.video_compression.config import CodecConfig
from src.video_compression.training.zoo_trainer import ZooTrainingReport
from src.video_compression.zoo.config import (
    DeviceAssignmentStrategy,
    ModelZooEntryConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_compression_zoo.py"


@pytest.fixture(scope="module")
def cli_module():
    spec = importlib.util.spec_from_file_location(
        "train_compression_zoo_script",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.fail(f"unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_compression_zoo_script"] = module
    spec.loader.exec_module(module)
    return module


def _write_codec_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(CodecConfig(name="codec_test").to_yaml_dict()),
        encoding="utf-8",
    )


def _write_manifest(
    path: Path,
    codec_path: Path,
    *,
    entry_ids: tuple[str, ...] = ("lambda_a", "lambda_b"),
) -> None:
    entries = [
        {
            "entry_id": entry_id,
            "lambda_rd": 0.0016 * (idx + 1),
            "target_bpp": 0.25,
            "target_psnr_db": 35.0,
            "train_steps": 1000,
            "batch_size": 2,
            "scheduler": {"name": "scheduler", "warmup_steps": 1},
        }
        for idx, entry_id in enumerate(entry_ids)
    ]
    path.write_text(
        yaml.safe_dump(
            {
                "name": "zoo_multi_smoke",
                "storage_root": str(path.parent / "zoo_store"),
                "default_codec_config_ref": str(codec_path),
                "device_preference": "cpu",
                "device_assignment_strategy": (DeviceAssignmentStrategy.SINGLE_DEVICE.value),
                "entries": entries,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _make_report(entry: ModelZooEntryConfig, *, device: str = "cpu") -> ZooTrainingReport:
    return ZooTrainingReport(
        entry_id=entry.entry_id,
        lambda_rd=entry.lambda_rd,
        target_bpp=entry.target_bpp,
        target_psnr_db=entry.target_psnr_db,
        realized_bpp=entry.target_bpp,
        realized_psnr_db=entry.target_psnr_db,
        realized_ms_ssim=0.97,
        final_loss=0.1,
        step_count=entry.train_steps,
        device=device,
        checkpoint_path=Path("/dev/null/ckpt.pt"),
        tolerance_passed=True,
        bpp_relative_error=0.0,
        psnr_absolute_error_db=0.0,
        train_wallclock_s=0.0,
        eval_wallclock_s=0.0,
    )


# ---------------------------------------------------------------------------
# Argparse surface
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_dry_run_subcommand_parses(self, cli_module) -> None:
        parser = cli_module.build_parser()
        args = parser.parse_args(["dry-run", "--manifest", "m.yaml"])
        assert args.cmd == "dry-run"
        assert str(args.manifest) == "m.yaml"
        assert args.only_entry_id is None

    def test_train_subcommand_parses(self, cli_module) -> None:
        parser = cli_module.build_parser()
        args = parser.parse_args(
            [
                "train",
                "--manifest",
                "m.yaml",
                "--only-entry-id",
                "a",
                "--only-entry-id",
                "b",
            ],
        )
        assert args.cmd == "train"
        assert args.only_entry_id == ["a", "b"]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_completes(self, cli_module, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec_config(codec_path)
        _write_manifest(manifest_path, codec_path)

        rc = cli_module.main(
            [
                "dry-run",
                "--manifest",
                str(manifest_path),
                "--output-root",
                str(tmp_path / "out"),
            ],
        )

        assert rc == 0

    def test_dry_run_with_only_entry_id(self, cli_module, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec_config(codec_path)
        _write_manifest(manifest_path, codec_path)

        rc = cli_module.main(
            [
                "dry-run",
                "--manifest",
                str(manifest_path),
                "--only-entry-id",
                "lambda_b",
                "--output-root",
                str(tmp_path / "out"),
            ],
        )

        assert rc == 0


class TestTrain:
    def test_train_drives_every_entry(
        self,
        cli_module,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec_config(codec_path)
        _write_manifest(manifest_path, codec_path)

        observed: list[str] = []

        def _fake_runner(
            entry: ModelZooEntryConfig,
            device: str,
            zoo: Any,
            codec_config: CodecConfig,
            output_root: Path,
        ) -> ZooTrainingReport:
            observed.append(entry.entry_id)
            return _make_report(entry, device=device)

        # Patch the default runner used inside ZooSweep.
        monkeypatch.setattr(
            "src.video_compression.zoo.sweep.default_entry_runner",
            _fake_runner,
        )

        rc = cli_module.main(
            [
                "train",
                "--manifest",
                str(manifest_path),
                "--output-root",
                str(tmp_path / "out"),
            ],
        )

        assert rc == 0
        assert observed == ["lambda_a", "lambda_b"]

    def test_train_only_entry_id_filters(
        self,
        cli_module,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec_config(codec_path)
        _write_manifest(manifest_path, codec_path)

        observed: list[str] = []

        def _fake_runner(
            entry: ModelZooEntryConfig,
            device: str,
            zoo: Any,
            codec_config: CodecConfig,
            output_root: Path,
        ) -> ZooTrainingReport:
            observed.append(entry.entry_id)
            return _make_report(entry, device=device)

        monkeypatch.setattr(
            "src.video_compression.zoo.sweep.default_entry_runner",
            _fake_runner,
        )

        rc = cli_module.main(
            [
                "train",
                "--manifest",
                str(manifest_path),
                "--only-entry-id",
                "lambda_b",
                "--output-root",
                str(tmp_path / "out"),
            ],
        )

        assert rc == 0
        assert observed == ["lambda_b"]

    def test_train_reports_failure_via_exit_code(
        self,
        cli_module,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec_config(codec_path)
        _write_manifest(manifest_path, codec_path)

        def _boom(*_: Any, **__: Any) -> ZooTrainingReport:
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(
            "src.video_compression.zoo.sweep.default_entry_runner",
            _boom,
        )

        # ZooSweep re-raises on runner failure today, so the CLI also
        # raises — that's the documented contract for this slice.
        with pytest.raises(RuntimeError, match="synthetic failure"):
            cli_module.main(
                [
                    "train",
                    "--manifest",
                    str(manifest_path),
                    "--output-root",
                    str(tmp_path / "out"),
                ],
            )


# ---------------------------------------------------------------------------
# Helper coverage
# ---------------------------------------------------------------------------


class TestSelectedEntries:
    def test_returns_all_when_no_filter(self, cli_module, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec_config(codec_path)
        _write_manifest(manifest_path, codec_path)
        from src.video_compression.zoo import load_manifest as _load

        manifest = _load(manifest_path)
        assert [e.entry_id for e in cli_module._selected_entries(manifest, None)] == [
            "lambda_a",
            "lambda_b",
        ]

    def test_filters_by_id(self, cli_module, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec_config(codec_path)
        _write_manifest(manifest_path, codec_path)
        from src.video_compression.zoo import load_manifest as _load

        manifest = _load(manifest_path)
        selected = cli_module._selected_entries(manifest, ["lambda_b"])
        assert [e.entry_id for e in selected] == ["lambda_b"]
