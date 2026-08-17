"""Slice B tests: parallel sweep + subprocess-per-device entry runner."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.video_compression.config import CodecConfig
from src.video_compression.training.zoo_trainer import ZooTrainingReport
from src.video_compression.zoo import VideoCodecZoo, load_manifest
from src.video_compression.zoo.config import (
    DeviceAssignmentStrategy,
    ModelZooEntryConfig,
    ModelZooManifestConfig,
)
from src.video_compression.zoo.device_planner import DeviceCapability
from src.video_compression.zoo.storage import (
    CHECKPOINT_FILENAME,
    METRICS_FILENAME,
)
from src.video_compression.zoo.sweep import (
    ZooSweep,
    _device_index,
    make_subprocess_entry_runner,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_codec(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(CodecConfig(name="codec_test").to_yaml_dict()),
        encoding="utf-8",
    )


def _write_manifest(
    path: Path,
    codec_path: Path,
    *,
    storage_root: Path,
    entry_ids: tuple[str, ...] = ("entry_a", "entry_b"),
    devices_per_entry: tuple[str, ...] | None = None,
) -> None:
    entries: list[dict[str, Any]] = []
    for idx, entry_id in enumerate(entry_ids):
        entry_dict: dict[str, Any] = {
            "entry_id": entry_id,
            "lambda_rd": 0.0016 * (idx + 1),
            "target_bpp": 0.25,
            "target_psnr_db": 35.0,
            "train_steps": 1000,
            "batch_size": 2,
            "scheduler": {"name": "scheduler", "warmup_steps": 1},
        }
        if devices_per_entry is not None:
            entry_dict["device"] = devices_per_entry[idx]
        entries.append(entry_dict)
    strategy = (
        DeviceAssignmentStrategy.MANUAL.value
        if devices_per_entry is not None
        else DeviceAssignmentStrategy.SINGLE_DEVICE.value
    )
    path.write_text(
        yaml.safe_dump(
            {
                "name": "slice_b_smoke",
                "storage_root": str(storage_root),
                "default_codec_config_ref": str(codec_path),
                "device_preference": "cpu",
                "device_assignment_strategy": strategy,
                "entries": entries,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _make_report(entry: ModelZooEntryConfig, device: str) -> ZooTrainingReport:
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


def _build_sweep(
    manifest: ModelZooManifestConfig,
    zoo: VideoCodecZoo,
    *,
    devices: list[DeviceCapability],
    entry_runner,
) -> ZooSweep:
    return ZooSweep(
        manifest,
        zoo,
        codec_config_for=lambda _e: CodecConfig(name="codec_test"),
        output_root=Path("/tmp/sweep"),
        devices=devices,
        entry_runner=entry_runner,
    )


# ---------------------------------------------------------------------------
# _device_index helper
# ---------------------------------------------------------------------------


class TestDeviceIndex:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("cuda:0", 0),
            ("cuda:1", 1),
            ("cuda:7", 7),
            ("cpu", None),
            ("cuda", None),
            ("cuda:gpu0", None),
        ],
    )
    def test_parses(self, label: str, expected: int | None) -> None:
        assert _device_index(label) == expected


# ---------------------------------------------------------------------------
# ZooSweep.run_parallel
# ---------------------------------------------------------------------------


class TestRunParallel:
    def test_run_parallel_processes_every_entry(self, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec(codec_path)
        _write_manifest(
            manifest_path,
            codec_path,
            storage_root=tmp_path / "store",
            entry_ids=("a", "b"),
            devices_per_entry=("cuda:0", "cuda:1"),
        )
        manifest = load_manifest(manifest_path)
        zoo = VideoCodecZoo(tmp_path / "zoo")

        observed: list[str] = []
        lock = threading.Lock()

        def _runner(
            entry: ModelZooEntryConfig,
            device: str,
            zoo_arg: Any,
            codec_config: CodecConfig,
            output_root: Path,
        ) -> ZooTrainingReport:
            with lock:
                observed.append(entry.entry_id)
            return _make_report(entry, device)

        devices = [
            DeviceCapability(label="cuda:0", name="gpu0", total_vram_mib=16384, is_cuda=True),
            DeviceCapability(label="cuda:1", name="gpu1", total_vram_mib=8192, is_cuda=True),
        ]
        sweep = _build_sweep(manifest, zoo, devices=devices, entry_runner=_runner)

        report = sweep.run_parallel()

        assert report.total == 2
        assert report.trained == 2
        assert report.skipped == 0
        # Reports are returned in manifest order regardless of completion order.
        assert [s.entry_id for s in report.statuses] == ["a", "b"]
        assert sorted(observed) == ["a", "b"]

    def test_run_parallel_actually_overlaps_devices(self, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec(codec_path)
        _write_manifest(
            manifest_path,
            codec_path,
            storage_root=tmp_path / "store",
            entry_ids=("a", "b"),
            devices_per_entry=("cuda:0", "cuda:1"),
        )
        manifest = load_manifest(manifest_path)
        zoo = VideoCodecZoo(tmp_path / "zoo")

        barrier = threading.Barrier(2, timeout=5.0)

        def _runner(
            entry: ModelZooEntryConfig,
            device: str,
            zoo_arg: Any,
            codec_config: CodecConfig,
            output_root: Path,
        ) -> ZooTrainingReport:
            # Both threads must reach the barrier simultaneously; if the
            # dispatcher is sequential this will TimeoutError.
            barrier.wait()
            return _make_report(entry, device)

        devices = [
            DeviceCapability(label="cuda:0", name="gpu0", total_vram_mib=16384, is_cuda=True),
            DeviceCapability(label="cuda:1", name="gpu1", total_vram_mib=8192, is_cuda=True),
        ]
        sweep = _build_sweep(manifest, zoo, devices=devices, entry_runner=_runner)

        report = sweep.run_parallel()
        assert report.trained == 2

    def test_run_parallel_respects_only_entry_ids(self, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec(codec_path)
        _write_manifest(
            manifest_path,
            codec_path,
            storage_root=tmp_path / "store",
            entry_ids=("a", "b"),
            devices_per_entry=("cuda:0", "cuda:1"),
        )
        manifest = load_manifest(manifest_path)
        zoo = VideoCodecZoo(tmp_path / "zoo")

        observed: list[str] = []

        def _runner(
            entry: ModelZooEntryConfig,
            device: str,
            zoo_arg: Any,
            codec_config: CodecConfig,
            output_root: Path,
        ) -> ZooTrainingReport:
            observed.append(entry.entry_id)
            return _make_report(entry, device)

        devices = [
            DeviceCapability(label="cuda:0", name="gpu0", total_vram_mib=16384, is_cuda=True),
            DeviceCapability(label="cuda:1", name="gpu1", total_vram_mib=8192, is_cuda=True),
        ]
        sweep = ZooSweep(
            manifest,
            zoo,
            codec_config_for=lambda _e: CodecConfig(name="codec_test"),
            output_root=tmp_path / "out",
            devices=devices,
            entry_runner=_runner,
            only_entry_ids=["b"],
        )
        report = sweep.run_parallel()
        assert observed == ["b"]
        assert report.total == 1

    def test_run_parallel_no_selected_entries(self, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec(codec_path)
        _write_manifest(
            manifest_path,
            codec_path,
            storage_root=tmp_path / "store",
            entry_ids=("a",),
        )
        manifest = load_manifest(manifest_path)
        zoo = VideoCodecZoo(tmp_path / "zoo")

        sweep = ZooSweep(
            manifest,
            zoo,
            codec_config_for=lambda _e: CodecConfig(name="codec_test"),
            output_root=tmp_path / "out",
            entry_runner=lambda *a, **kw: pytest.fail("should not be called"),
            only_entry_ids=["nonexistent"],
        )
        report = sweep.run_parallel()
        assert report.total == 0
        assert report.trained == 0


# ---------------------------------------------------------------------------
# Subprocess entry runner
# ---------------------------------------------------------------------------


class TestSubprocessEntryRunner:
    def _persist_artifacts(
        self,
        zoo: VideoCodecZoo,
        entry: ModelZooEntryConfig,
    ) -> None:
        """Write the artifacts the parent expects to read back."""
        entry_dir = zoo.entry_dir(entry.entry_id)
        entry_dir.mkdir(parents=True, exist_ok=True)
        (entry_dir / CHECKPOINT_FILENAME).write_bytes(b"\x00")
        import json

        metrics = {
            "loss": 0.1,
            "rate_bpp": 0.24,
            "distortion": 0.01,
            "psnr_db": 34.5,
            "bpp_relative_error": 0.04,
            "psnr_absolute_error_db": 0.5,
            "tolerance_passed": 1.0,
            "step_count": float(entry.train_steps),
            "lambda_rd": float(entry.lambda_rd),
            "ms_ssim": 0.97,
            "train_wallclock_s": 1.2,
            "eval_wallclock_s": 0.3,
        }
        (entry_dir / METRICS_FILENAME).write_text(
            json.dumps({"metrics": metrics, "saved_at": "2026-05-01T00:00:00Z"}),
            encoding="utf-8",
        )

    def test_pins_cuda_visible_devices(self, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec(codec_path)
        _write_manifest(
            manifest_path,
            codec_path,
            storage_root=tmp_path / "store",
            entry_ids=("a",),
        )
        manifest = load_manifest(manifest_path)
        zoo = VideoCodecZoo(tmp_path / "zoo")
        entry = manifest.entries[0]
        self._persist_artifacts(zoo, entry)

        captured: dict[str, Any] = {}

        def _fake_run(argv: list[str], env: dict[str, str]):
            captured["argv"] = argv
            captured["env"] = env
            return subprocess.CompletedProcess(argv, 0, b"", b"")

        runner = make_subprocess_entry_runner(
            manifest_path=manifest_path,
            output_root=tmp_path / "out",
            subprocess_runner=_fake_run,
        )

        report = runner(
            entry,
            "cuda:1",
            zoo,
            CodecConfig(name="codec_test"),
            tmp_path / "out",
        )

        # CUDA_VISIBLE_DEVICES pinned to "1"; child sees cuda:0.
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
        assert "--device" in captured["argv"]
        assert captured["argv"][captured["argv"].index("--device") + 1] == "cuda:0"
        # Parent's report records the *parent-visible* device label.
        assert report.device == "cuda:1"
        assert report.realized_bpp == pytest.approx(0.24)
        assert report.train_wallclock_s == pytest.approx(1.2)

    def test_passes_cpu_device_through(self, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec(codec_path)
        _write_manifest(
            manifest_path,
            codec_path,
            storage_root=tmp_path / "store",
            entry_ids=("a",),
        )
        manifest = load_manifest(manifest_path)
        zoo = VideoCodecZoo(tmp_path / "zoo")
        entry = manifest.entries[0]
        self._persist_artifacts(zoo, entry)

        captured: dict[str, Any] = {}

        def _fake_run(argv: list[str], env: dict[str, str]):
            captured["argv"] = argv
            captured["env"] = env
            return subprocess.CompletedProcess(argv, 0, b"", b"")

        runner = make_subprocess_entry_runner(
            manifest_path=manifest_path,
            output_root=tmp_path / "out",
            subprocess_runner=_fake_run,
        )
        runner(entry, "cpu", zoo, CodecConfig(name="codec_test"), tmp_path / "out")
        # No env pinning for CPU.
        assert "CUDA_VISIBLE_DEVICES" not in captured["env"] or captured["env"].get(
            "CUDA_VISIBLE_DEVICES"
        ) == os.environ.get("CUDA_VISIBLE_DEVICES", "__missing__")
        assert captured["argv"][captured["argv"].index("--device") + 1] == "cpu"

    def test_propagates_nonzero_exit(self, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec(codec_path)
        _write_manifest(
            manifest_path,
            codec_path,
            storage_root=tmp_path / "store",
            entry_ids=("a",),
        )
        manifest = load_manifest(manifest_path)
        zoo = VideoCodecZoo(tmp_path / "zoo")
        entry = manifest.entries[0]

        def _fake_run(argv: list[str], env: dict[str, str]):
            return subprocess.CompletedProcess(argv, 7, b"", b"boom")

        runner = make_subprocess_entry_runner(
            manifest_path=manifest_path,
            output_root=tmp_path / "out",
            subprocess_runner=_fake_run,
        )
        with pytest.raises(RuntimeError, match="exit code 7"):
            runner(
                entry,
                "cuda:0",
                zoo,
                CodecConfig(name="codec_test"),
                tmp_path / "out",
            )

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec(codec_path)
        _write_manifest(
            manifest_path,
            codec_path,
            storage_root=tmp_path / "store",
            entry_ids=("a",),
        )
        manifest = load_manifest(manifest_path)
        zoo = VideoCodecZoo(tmp_path / "zoo")
        entry = manifest.entries[0]

        # Persist metrics.json but NOT the checkpoint.
        entry_dir = zoo.entry_dir(entry.entry_id)
        entry_dir.mkdir(parents=True, exist_ok=True)
        import json

        (entry_dir / METRICS_FILENAME).write_text(
            json.dumps(
                {
                    "metrics": {
                        "loss": 0.1,
                        "rate_bpp": 0.2,
                        "distortion": 0.0,
                        "psnr_db": 30.0,
                        "bpp_relative_error": 0.0,
                        "psnr_absolute_error_db": 0.0,
                        "tolerance_passed": 1.0,
                        "step_count": 1.0,
                        "lambda_rd": 0.001,
                    },
                    "saved_at": "now",
                },
            ),
            encoding="utf-8",
        )

        def _fake_run(argv: list[str], env: dict[str, str]):
            return subprocess.CompletedProcess(argv, 0, b"", b"")

        runner = make_subprocess_entry_runner(
            manifest_path=manifest_path,
            output_root=tmp_path / "out",
            subprocess_runner=_fake_run,
        )
        with pytest.raises(FileNotFoundError, match="expected checkpoint"):
            runner(
                entry,
                "cpu",
                zoo,
                CodecConfig(name="codec_test"),
                tmp_path / "out",
            )

    def test_cuda_pinning_none_passes_label_through(self, tmp_path: Path) -> None:
        codec_path = tmp_path / "codec.yaml"
        manifest_path = tmp_path / "manifest.yaml"
        _write_codec(codec_path)
        _write_manifest(
            manifest_path,
            codec_path,
            storage_root=tmp_path / "store",
            entry_ids=("a",),
        )
        manifest = load_manifest(manifest_path)
        zoo = VideoCodecZoo(tmp_path / "zoo")
        entry = manifest.entries[0]
        self._persist_artifacts(zoo, entry)

        captured: dict[str, Any] = {}

        def _fake_run(argv: list[str], env: dict[str, str]):
            captured["argv"] = argv
            captured["env"] = env
            return subprocess.CompletedProcess(argv, 0, b"", b"")

        runner = make_subprocess_entry_runner(
            manifest_path=manifest_path,
            output_root=tmp_path / "out",
            subprocess_runner=_fake_run,
            cuda_pinning="none",
        )
        runner(
            entry,
            "cuda:1",
            zoo,
            CodecConfig(name="codec_test"),
            tmp_path / "out",
        )
        # No env override; child gets the parent's label verbatim.
        assert captured["env"].get("CUDA_VISIBLE_DEVICES") == os.environ.get(
            "CUDA_VISIBLE_DEVICES",
        )
        assert captured["argv"][captured["argv"].index("--device") + 1] == "cuda:1"
