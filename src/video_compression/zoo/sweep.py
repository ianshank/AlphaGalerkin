"""Phase 2-D — multi-entry zoo sweep orchestrator.

This module wires :class:`ZooTrainer` (Phase 2-C) into a manifest-level
sweep driver that:

- Maps every entry to a concrete device via
  :func:`assign_devices` (Phase 2-B device planner).
- Skips entries whose persisted :data:`entry.json` already matches the
  live entry's :func:`compute_hash`. This makes the sweep
  *manifest-hash resumable*: a re-run after a crash or a partial sweep
  picks up exactly where it left off.
- Delegates per-entry training to a configurable ``entry_runner``,
  defaulting to an in-process :class:`ZooTrainer` invocation. Tests
  inject lightweight fakes; future slices will swap in subprocess /
  ``CUDA_VISIBLE_DEVICES``-pinned runners without touching the
  orchestrator surface.

The sweep is intentionally *sequential* in this slice. Subprocess-per-
device parallelism is a separate, additive change in a follow-on
commit.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.templates.logging import create_logger_class
from src.video_compression.config import CodecConfig
from src.video_compression.training.zoo_trainer import (
    ZooTrainer,
    ZooTrainingReport,
)
from src.video_compression.zoo.config import (
    ModelZooEntryConfig,
    ModelZooManifestConfig,
)
from src.video_compression.zoo.device_planner import (
    DeviceCapability,
    DevicePlan,
    assign_devices,
)
from src.video_compression.zoo.storage import (
    ENTRY_FILENAME,
    VideoCodecZoo,
)

_Logger = create_logger_class("ZooSweep")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


EntryRunner = Callable[
    [ModelZooEntryConfig, str, VideoCodecZoo, CodecConfig, Path],
    ZooTrainingReport,
]
"""Strategy for training a single entry.

Args:
    entry: The manifest entry to train.
    device: The resolved device label (``cuda:0``, ``cpu``, ...).
    zoo: Zoo storage handle. Reports + checkpoints land here.
    codec_config: Codec config resolved for this entry.
    output_root: Directory for trainer scratch state.

Returns:
        A :class:`ZooTrainingReport` summarizing the realized metrics.
"""


@dataclass(frozen=True)
class EntryStatus:
    """Per-entry outcome from a sweep run."""

    entry_id: str
    device: str
    skipped: bool
    skip_reason: str | None
    report: ZooTrainingReport | None


@dataclass(frozen=True)
class SweepReport:
    """Aggregate result of running :class:`ZooSweep`."""

    manifest_name: str
    total: int
    trained: int
    skipped: int
    failed: int
    statuses: tuple[EntryStatus, ...]


# ---------------------------------------------------------------------------
# Default in-process entry runner
# ---------------------------------------------------------------------------


def default_entry_runner(
    entry: ModelZooEntryConfig,
    device: str,
    zoo: VideoCodecZoo,
    codec_config: CodecConfig,
    output_root: Path,
) -> ZooTrainingReport:
    """Run one entry in-process via :class:`ZooTrainer`.

    Subprocess / device-pinned variants live in follow-on slices and
    plug into the same :data:`EntryRunner` signature.
    """
    trainer = ZooTrainer(
        entry,
        zoo,
        codec_config=codec_config,
        device=device,
        output_root=output_root,
    )
    return trainer.run()


# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------


def _persisted_entry_hash(zoo: VideoCodecZoo, entry_id: str) -> str | None:
    """Return ``compute_hash()`` of the persisted entry.json, or None."""
    entry_path = zoo.entry_dir(entry_id) / ENTRY_FILENAME
    if not entry_path.exists():
        return None
    try:
        with entry_path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        rehydrated = ModelZooEntryConfig.model_validate(raw)
    except Exception:  # noqa: BLE001 — defensive: stale schema -> retrain
        return None
    return rehydrated.compute_hash()


def should_skip(
    zoo: VideoCodecZoo,
    entry: ModelZooEntryConfig,
) -> tuple[bool, str | None]:
    """Decide whether to skip ``entry`` based on persisted artifacts.

    Returns:
        ``(skip, reason)``. ``skip`` is ``True`` only if every
        invariant is satisfied:

        * The zoo has a checkpoint for this ``entry_id``.
        * The persisted ``entry.json`` rehydrates cleanly.
        * The rehydrated config's :func:`compute_hash` matches the
          live entry's hash.

    """
    if not zoo.has_entry(entry.entry_id):
        return False, None
    persisted_hash = _persisted_entry_hash(zoo, entry.entry_id)
    if persisted_hash is None:
        return False, "persisted entry.json missing or unreadable"
    live_hash = entry.compute_hash()
    if persisted_hash != live_hash:
        return False, f"hash drift {persisted_hash} -> {live_hash}"
    return True, f"hash match {live_hash}"


# ---------------------------------------------------------------------------
# Sweep orchestrator
# ---------------------------------------------------------------------------


class ZooSweep:
    """Run a manifest's entries through a configurable training runner."""

    def __init__(
        self,
        manifest: ModelZooManifestConfig,
        zoo: VideoCodecZoo,
        codec_config_for: Callable[[ModelZooEntryConfig], CodecConfig],
        *,
        output_root: Path,
        devices: Sequence[DeviceCapability] | None = None,
        entry_runner: EntryRunner | None = None,
        only_entry_ids: Iterable[str] | None = None,
    ) -> None:
        """Construct the sweep.

        Args:
            manifest: Validated manifest declaring the entries to run.
            zoo: Zoo storage handle (filesystem-backed in this slice).
            codec_config_for: Callable that resolves the codec config
                for one entry. Path resolution stays with the caller
                (the CLI) so the orchestrator does not couple to
                filesystem layout conventions.
            output_root: Directory under which per-entry trainer
                scratch state lives.
            devices: Pre-scanned device list. ``None`` triggers
                :func:`scan_devices` inside :func:`assign_devices`.
            entry_runner: Strategy for training a single entry. The
                default is :func:`default_entry_runner` (in-process
                ZooTrainer). Tests inject fakes here.
            only_entry_ids: Optional allow-list. When provided, entries
                outside the list are excluded entirely from the sweep
                (they are not even reported as skipped).

        """
        self._manifest = manifest
        self._zoo = zoo
        self._codec_config_for = codec_config_for
        self._output_root = output_root
        self._entry_runner = entry_runner or default_entry_runner
        self._allow: set[str] | None = (
            set(only_entry_ids) if only_entry_ids is not None else None
        )
        self._plan: DevicePlan = assign_devices(
            manifest,
            devices=list(devices) if devices is not None else None,
        )
        self._log = _Logger(
            "ZooSweep",
            manifest=manifest.name,
            strategy=self._plan.strategy.value,
        )

    @property
    def plan(self) -> DevicePlan:
        """Resolved device plan; useful for CLI introspection."""
        return self._plan

    def run(self) -> SweepReport:
        """Drive every entry through the configured runner."""
        statuses: list[EntryStatus] = []
        trained = skipped = failed = 0

        for entry in self._manifest.entries:
            if self._allow is not None and entry.entry_id not in self._allow:
                continue

            device = self._plan.device_for(entry.entry_id)

            skip, reason = should_skip(self._zoo, entry)
            if skip:
                skipped += 1
                self._log.info(
                    "sweep.entry.skipped",
                    entry_id=entry.entry_id,
                    device=device,
                    reason=reason,
                )
                statuses.append(
                    EntryStatus(
                        entry_id=entry.entry_id,
                        device=device,
                        skipped=True,
                        skip_reason=reason,
                        report=None,
                    ),
                )
                continue

            codec_config = self._codec_config_for(entry)
            self._log.info(
                "sweep.entry.start",
                entry_id=entry.entry_id,
                device=device,
                lambda_rd=entry.lambda_rd,
            )
            try:
                report = self._entry_runner(
                    entry,
                    device,
                    self._zoo,
                    codec_config,
                    self._output_root,
                )
            except Exception:
                failed += 1
                self._log.error(
                    "sweep.entry.failed",
                    entry_id=entry.entry_id,
                    device=device,
                )
                raise
            trained += 1
            self._log.info(
                "sweep.entry.completed",
                entry_id=entry.entry_id,
                device=device,
                tolerance_passed=report.tolerance_passed,
                realized_bpp=report.realized_bpp,
                realized_psnr_db=report.realized_psnr_db,
            )
            statuses.append(
                EntryStatus(
                    entry_id=entry.entry_id,
                    device=device,
                    skipped=False,
                    skip_reason=None,
                    report=report,
                ),
            )

        sweep_report = SweepReport(
            manifest_name=self._manifest.name,
            total=len(statuses),
            trained=trained,
            skipped=skipped,
            failed=failed,
            statuses=tuple(statuses),
        )
        self._log.info(
            "sweep.completed",
            total=sweep_report.total,
            trained=sweep_report.trained,
            skipped=sweep_report.skipped,
            failed=sweep_report.failed,
        )
        return sweep_report
