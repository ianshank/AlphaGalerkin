"""Tests for the Phase 2-D ZooSweep orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.video_compression.config import CodecConfig
from src.video_compression.training.zoo_trainer import ZooTrainingReport
from src.video_compression.zoo.config import (
    DeviceAssignmentStrategy,
    ModelZooEntryConfig,
    ModelZooManifestConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from src.video_compression.zoo.device_planner import DeviceCapability
from src.video_compression.zoo.storage import (
    CHECKPOINT_FILENAME,
    ENTRY_FILENAME,
    VideoCodecZoo,
)
from src.video_compression.zoo.sweep import (
    EntryStatus,
    SweepReport,
    ZooSweep,
    default_entry_runner,
    should_skip,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entry(entry_id: str = "lambda_a", **overrides: Any) -> ModelZooEntryConfig:
    base: dict[str, Any] = {
        "entry_id": entry_id,
        "lambda_rd": 0.01,
        "target_bpp": 0.5,
        "target_psnr_db": 33.0,
        "train_steps": 1000,
        "optimizer": OptimizerConfig(name="opt"),
        "scheduler": SchedulerConfig(name="sched", warmup_steps=10, min_lr_ratio=0.1),
    }
    base.update(overrides)
    return ModelZooEntryConfig(**base)


def _manifest(tmp_path: Path, *entries: ModelZooEntryConfig) -> ModelZooManifestConfig:
    return ModelZooManifestConfig(
        name="m",
        storage_root=str(tmp_path / "zoo"),
        entries=list(entries) if entries else [_entry()],
        device_preference="cpu",
        device_assignment_strategy=DeviceAssignmentStrategy.SINGLE_DEVICE,
    )


def _cpu_only() -> list[DeviceCapability]:
    return [DeviceCapability(label="cpu", name="cpu", total_vram_mib=0.0, is_cuda=False)]


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
# should_skip
# ---------------------------------------------------------------------------


class TestShouldSkip:
    def test_no_checkpoint_means_no_skip(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        skip, reason = should_skip(zoo, _entry())
        assert skip is False
        assert reason is None

    def test_matching_hash_skips(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        entry = _entry()
        entry_dir = zoo.entry_dir(entry.entry_id)
        entry_dir.mkdir(parents=True, exist_ok=True)
        (entry_dir / CHECKPOINT_FILENAME).write_bytes(b"x")
        with (entry_dir / ENTRY_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(entry.to_yaml_dict(), fh, default=str)

        skip, reason = should_skip(zoo, entry)

        assert skip is True
        assert reason is not None
        assert "hash match" in reason

    def test_drift_does_not_skip(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        original = _entry()
        entry_dir = zoo.entry_dir(original.entry_id)
        entry_dir.mkdir(parents=True, exist_ok=True)
        (entry_dir / CHECKPOINT_FILENAME).write_bytes(b"x")
        with (entry_dir / ENTRY_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(original.to_yaml_dict(), fh, default=str)

        # Live entry has a different lambda_rd, which changes the hash.
        live = _entry(lambda_rd=0.02)
        skip, reason = should_skip(zoo, live)

        assert skip is False
        assert reason is not None
        assert "hash drift" in reason

    def test_corrupt_entry_json_does_not_skip(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        entry = _entry()
        entry_dir = zoo.entry_dir(entry.entry_id)
        entry_dir.mkdir(parents=True, exist_ok=True)
        (entry_dir / CHECKPOINT_FILENAME).write_bytes(b"x")
        (entry_dir / ENTRY_FILENAME).write_text("{ not json")

        skip, reason = should_skip(zoo, entry)

        assert skip is False
        assert reason is not None


# ---------------------------------------------------------------------------
# ZooSweep.run
# ---------------------------------------------------------------------------


class TestZooSweep:
    def _codec_config(self, _entry: ModelZooEntryConfig) -> CodecConfig:
        return CodecConfig(name="codec")

    def test_runs_every_entry(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, _entry("a"), _entry("b", lambda_rd=0.02))
        zoo = VideoCodecZoo(tmp_path / "zoo")
        runner = MagicMock(side_effect=lambda e, d, z, c, o: _make_report(e, device=d))

        sweep = ZooSweep(
            manifest,
            zoo,
            codec_config_for=self._codec_config,
            output_root=tmp_path / "outputs",
            devices=_cpu_only(),
            entry_runner=runner,
        )
        report = sweep.run()

        assert isinstance(report, SweepReport)
        assert report.total == 2
        assert report.trained == 2
        assert report.skipped == 0
        assert report.failed == 0
        assert runner.call_count == 2
        assert [s.entry_id for s in report.statuses] == ["a", "b"]
        assert all(s.skipped is False for s in report.statuses)

    def test_skips_entries_with_matching_hash(self, tmp_path: Path) -> None:
        entry_a = _entry("a")
        manifest = _manifest(tmp_path, entry_a, _entry("b", lambda_rd=0.02))
        zoo = VideoCodecZoo(tmp_path / "zoo")
        # Pre-populate entry "a" so it should be skipped.
        a_dir = zoo.entry_dir("a")
        a_dir.mkdir(parents=True, exist_ok=True)
        (a_dir / CHECKPOINT_FILENAME).write_bytes(b"x")
        with (a_dir / ENTRY_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(entry_a.to_yaml_dict(), fh, default=str)

        runner = MagicMock(side_effect=lambda e, d, z, c, o: _make_report(e, device=d))

        sweep = ZooSweep(
            manifest,
            zoo,
            codec_config_for=self._codec_config,
            output_root=tmp_path / "outputs",
            devices=_cpu_only(),
            entry_runner=runner,
        )
        report = sweep.run()

        assert report.skipped == 1
        assert report.trained == 1
        statuses = {s.entry_id: s for s in report.statuses}
        assert statuses["a"].skipped is True
        assert statuses["b"].skipped is False
        # Runner must only be called for the entry that wasn't skipped.
        assert runner.call_count == 1
        called_entry: ModelZooEntryConfig = runner.call_args[0][0]
        assert called_entry.entry_id == "b"

    def test_only_entry_ids_filters(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, _entry("a"), _entry("b", lambda_rd=0.02))
        zoo = VideoCodecZoo(tmp_path / "zoo")
        runner = MagicMock(side_effect=lambda e, d, z, c, o: _make_report(e, device=d))

        sweep = ZooSweep(
            manifest,
            zoo,
            codec_config_for=self._codec_config,
            output_root=tmp_path / "outputs",
            devices=_cpu_only(),
            entry_runner=runner,
            only_entry_ids=["b"],
        )
        report = sweep.run()

        assert report.total == 1
        assert [s.entry_id for s in report.statuses] == ["b"]
        assert runner.call_count == 1

    def test_runner_exception_is_propagated(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, _entry("a"))
        zoo = VideoCodecZoo(tmp_path / "zoo")

        def _boom(*_: Any, **__: Any) -> ZooTrainingReport:
            raise RuntimeError("boom")

        sweep = ZooSweep(
            manifest,
            zoo,
            codec_config_for=self._codec_config,
            output_root=tmp_path / "outputs",
            devices=_cpu_only(),
            entry_runner=_boom,
        )
        with pytest.raises(RuntimeError, match="boom"):
            sweep.run()

    def test_plan_is_exposed(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, _entry("a"))
        zoo = VideoCodecZoo(tmp_path / "zoo")
        sweep = ZooSweep(
            manifest,
            zoo,
            codec_config_for=self._codec_config,
            output_root=tmp_path / "outputs",
            devices=_cpu_only(),
            entry_runner=lambda e, d, z, c, o: _make_report(e, device=d),
        )
        plan = sweep.plan
        assert [a.entry_id for a in plan.assignments] == ["a"]
        # Single CPU rig: every assignment lands on cpu.
        assert all(a.device == "cpu" for a in plan.assignments)


def test_default_entry_runner_signature() -> None:
    # Smoke: confirm the default runner is the one ZooSweep falls back to
    # and matches the EntryRunner protocol.
    assert callable(default_entry_runner)
    # Six positional params: entry, device, zoo, codec_config, output_root.
    # (Python introspection: the function has 5 params plus self = 5.)
    import inspect

    sig = inspect.signature(default_entry_runner)
    assert list(sig.parameters) == [
        "entry",
        "device",
        "zoo",
        "codec_config",
        "output_root",
    ]


def test_entry_status_dataclass_is_frozen() -> None:
    status = EntryStatus(
        entry_id="a",
        device="cpu",
        skipped=True,
        skip_reason="hash match",
        report=None,
    )
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        status.entry_id = "other"  # type: ignore[misc]
