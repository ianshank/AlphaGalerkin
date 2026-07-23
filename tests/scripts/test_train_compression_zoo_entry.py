"""Tests for ``scripts/train_compression_zoo_entry.py``."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from src.video_compression.config import CodecConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_compression_zoo_entry.py"


@pytest.fixture(scope="module")
def cli_module():
    spec = importlib.util.spec_from_file_location(
        "train_compression_zoo_entry_script",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - sanity
        pytest.fail(f"unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_compression_zoo_entry_script"] = module
    spec.loader.exec_module(module)
    return module


def _write_codec_config(path: Path) -> None:
    path.write_text(yaml.safe_dump(CodecConfig(name="codec_test").to_yaml_dict()), encoding="utf-8")


def _write_manifest(path: Path, codec_path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "name": "zoo_cli_smoke",
                "storage_root": str(path.parent / "zoo_store"),
                "default_codec_config_ref": str(codec_path),
                "device_preference": "cpu",
                "entries": [
                    {
                        "entry_id": "lambda_0016",
                        "lambda_rd": 0.0016,
                        "target_bpp": 0.25,
                        "target_psnr_db": 35.0,
                        "train_steps": 8,
                        "batch_size": 2,
                        "scheduler": {"name": "scheduler", "warmup_steps": 1},
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


class TestArgparse:
    def test_dry_run_subcommand_parses(self, cli_module) -> None:
        parser = cli_module.build_parser()
        args = parser.parse_args(["dry-run", "--manifest", "m.yaml", "--entry-id", "e1"])
        assert args.cmd == "dry-run"
        assert str(args.manifest) == "m.yaml"
        assert args.entry_id == "e1"

    def test_train_subcommand_parses(self, cli_module) -> None:
        parser = cli_module.build_parser()
        args = parser.parse_args(["train", "--manifest", "m.yaml", "--entry-id", "e1"])
        assert args.cmd == "train"
        assert args.entry_id == "e1"


class TestCommands:
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
                "--entry-id",
                "lambda_0016",
                "--device",
                "cpu",
                "--max-steps",
                "4",
            ]
        )

        assert rc == 0

    def test_train_invokes_zoo_trainer(self, cli_module, monkeypatch, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec_config(codec_path)
        _write_manifest(manifest_path, codec_path)
        observed: dict[str, object] = {}

        @dataclass(frozen=True)
        class _FakeReport:
            entry_id: str
            checkpoint_path: Path
            tolerance_passed: bool
            realized_bpp: float
            realized_psnr_db: float

        class _FakeZooTrainer:
            def __init__(self, entry, zoo, *, codec_config, device, output_root) -> None:  # noqa: ANN001
                observed["entry_id"] = entry.entry_id
                observed["train_steps"] = entry.train_steps
                observed["device"] = device
                observed["codec_name"] = codec_config.name
                observed["output_root"] = output_root
                observed["zoo_root"] = getattr(zoo, "root", None)

            def run(self) -> _FakeReport:
                return _FakeReport(
                    entry_id="lambda_0016",
                    checkpoint_path=tmp_path / "checkpoint.pt",
                    tolerance_passed=True,
                    realized_bpp=0.24,
                    realized_psnr_db=35.2,
                )

        monkeypatch.setattr(cli_module, "ZooTrainer", _FakeZooTrainer)

        rc = cli_module.main(
            [
                "train",
                "--manifest",
                str(manifest_path),
                "--entry-id",
                "lambda_0016",
                "--device",
                "cpu",
                "--max-steps",
                "4",
            ]
        )

        assert rc == 0
        assert observed["entry_id"] == "lambda_0016"
        assert observed["train_steps"] == 4
        assert observed["device"] == "cpu"
        assert observed["codec_name"] == "codec_test"
