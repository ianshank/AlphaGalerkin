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

import concurrent.futures
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
    CHECKPOINT_FILENAME,
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

    # ------------------------------------------------------------------
    # Per-entry processing (shared by run() and run_parallel())
    # ------------------------------------------------------------------

    def _is_selected(self, entry: ModelZooEntryConfig) -> bool:
        return self._allow is None or entry.entry_id in self._allow

    def _process_entry(self, entry: ModelZooEntryConfig) -> EntryStatus:
        """Train (or skip) one entry; thread-safe for parallel dispatch.

        The orchestrator's instance state (``self._plan``,
        ``self._zoo``, ``self._codec_config_for``, ``self._entry_runner``)
        is read-only after construction, so this method is safe to call
        from multiple worker threads concurrently.
        """
        device = self._plan.device_for(entry.entry_id)
        skip, reason = should_skip(self._zoo, entry)
        if skip:
            self._log.info(
                "sweep.entry.skipped",
                entry_id=entry.entry_id,
                device=device,
                reason=reason,
            )
            return EntryStatus(
                entry_id=entry.entry_id,
                device=device,
                skipped=True,
                skip_reason=reason,
                report=None,
            )

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
            self._log.error(
                "sweep.entry.failed",
                entry_id=entry.entry_id,
                device=device,
            )
            raise
        self._log.info(
            "sweep.entry.completed",
            entry_id=entry.entry_id,
            device=device,
            tolerance_passed=report.tolerance_passed,
            realized_bpp=report.realized_bpp,
            realized_psnr_db=report.realized_psnr_db,
        )
        return EntryStatus(
            entry_id=entry.entry_id,
            device=device,
            skipped=False,
            skip_reason=None,
            report=report,
        )

    def _aggregate(self, statuses: Sequence[EntryStatus]) -> SweepReport:
        trained = sum(
            1 for s in statuses if not s.skipped and s.report is not None
        )
        skipped = sum(1 for s in statuses if s.skipped)
        sweep_report = SweepReport(
            manifest_name=self._manifest.name,
            total=len(statuses),
            trained=trained,
            skipped=skipped,
            failed=0,
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

    def run(self) -> SweepReport:
        """Drive every selected entry sequentially in this process."""
        statuses: list[EntryStatus] = []
        for entry in self._manifest.entries:
            if not self._is_selected(entry):
                continue
            statuses.append(self._process_entry(entry))
        return self._aggregate(statuses)

    def run_parallel(self) -> SweepReport:
        """Drive selected entries with one worker thread per device.

        Entries on the same device are processed sequentially within
        their worker; different devices run concurrently. The default
        in-process ``entry_runner`` is *not* GPU-safe under parallel
        dispatch (two threads sharing the same CUDA context will
        contend); pair this with :func:`make_subprocess_entry_runner`
        when running on real GPUs.
        """
        selected_ids = {
            e.entry_id for e in self._manifest.entries if self._is_selected(e)
        }
        # Preserve manifest order while grouping by device.
        groups: dict[str, list[ModelZooEntryConfig]] = {}
        for entry in self._manifest.entries:
            if entry.entry_id not in selected_ids:
                continue
            groups.setdefault(self._plan.device_for(entry.entry_id), []).append(entry)

        if not groups:
            return self._aggregate([])

        def _worker(entries: list[ModelZooEntryConfig]) -> list[EntryStatus]:
            return [self._process_entry(e) for e in entries]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(groups),
        ) as pool:
            future_to_device = {
                pool.submit(_worker, entries): device
                for device, entries in groups.items()
            }
            results: dict[str, list[EntryStatus]] = {}
            for fut in concurrent.futures.as_completed(future_to_device):
                results[future_to_device[fut]] = fut.result()

        # Re-flatten in the manifest's original entry order so the report is
        # deterministic regardless of which device finished first.
        statuses: list[EntryStatus] = []
        order = {e.entry_id: i for i, e in enumerate(self._manifest.entries)}
        flat = [s for batch in results.values() for s in batch]
        flat.sort(key=lambda s: order[s.entry_id])
        statuses.extend(flat)
        return self._aggregate(statuses)


# ---------------------------------------------------------------------------
# Subprocess-per-device entry runner (Slice B)
# ---------------------------------------------------------------------------


SubprocessRunner = Callable[
    [list[str], dict[str, str]],
    "subprocess.CompletedProcess[Any]",
]
"""Hook for replacing ``subprocess.run`` in tests.

Receives ``(argv, env)`` and must return a completed-process object
whose ``returncode`` is 0 on success.
"""


def _device_index(device: str) -> int | None:
    """Return the GPU index for a ``cuda:N`` label, else ``None``."""
    if not device.startswith("cuda:"):
        return None
    suffix = device.split(":", 1)[1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _read_persisted_report(
    zoo: VideoCodecZoo,
    entry: ModelZooEntryConfig,
    device: str,
) -> ZooTrainingReport:
    """Reconstruct a :class:`ZooTrainingReport` from on-disk artifacts."""
    metrics = zoo.load_metrics(entry.entry_id)
    checkpoint_path = zoo.entry_dir(entry.entry_id) / CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"subprocess runner: expected checkpoint at {checkpoint_path}",
        )
    return ZooTrainingReport(
        entry_id=entry.entry_id,
        lambda_rd=entry.lambda_rd,
        target_bpp=entry.target_bpp,
        target_psnr_db=entry.target_psnr_db,
        realized_bpp=float(metrics["rate_bpp"]),
        realized_psnr_db=float(metrics["psnr_db"]),
        realized_ms_ssim=(
            float(metrics["ms_ssim"]) if "ms_ssim" in metrics else None
        ),
        final_loss=float(metrics["loss"]),
        step_count=int(metrics["step_count"]),
        device=device,
        checkpoint_path=checkpoint_path,
        tolerance_passed=bool(metrics.get("tolerance_passed", 0.0)),
        bpp_relative_error=float(metrics["bpp_relative_error"]),
        psnr_absolute_error_db=float(metrics["psnr_absolute_error_db"]),
        train_wallclock_s=float(metrics.get("train_wallclock_s", 0.0)),
        eval_wallclock_s=float(metrics.get("eval_wallclock_s", 0.0)),
        parent_entry_id=entry.parent_entry_id,
    )


def make_subprocess_entry_runner(
    *,
    manifest_path: Path,
    output_root: Path,
    python_executable: str | None = None,
    module_name: str = "scripts.train_compression_zoo_entry",
    subprocess_runner: SubprocessRunner | None = None,
    cuda_pinning: Literal["env", "none"] = "env",
) -> EntryRunner:
    """Build an :data:`EntryRunner` that delegates to a child process.

    The child re-uses the existing single-entry CLI
    (``train_compression_zoo_entry train``), so trained checkpoints,
    ``entry.json`` and ``metrics.json`` land in the same zoo storage
    that the parent sees. After the child exits 0, the parent reads
    those artifacts back and reconstructs a :class:`ZooTrainingReport`.

    Args:
        manifest_path: Path passed to the child's ``--manifest`` flag.
        output_root: Path passed to the child's ``--output-root`` flag.
        python_executable: Override ``sys.executable`` (e.g. for tests
            running the CLI under a venv). Defaults to ``sys.executable``.
        module_name: Module the child runs via ``-m``. Override only
            when adding a custom CLI entry point.
        subprocess_runner: Hook to replace :func:`subprocess.run`; tests
            inject fakes that record ``(argv, env)`` without forking.
        cuda_pinning: ``"env"`` sets ``CUDA_VISIBLE_DEVICES=<idx>`` for
            ``cuda:N`` devices and translates the child's ``--device``
            flag to ``cuda:0`` (because the child sees only one GPU).
            ``"none"`` disables this and passes the original device
            label through (useful for tests).

    """
    def _default_subprocess_runner(
        argv: list[str],
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[Any]:
        return subprocess.run(argv, env=env, check=False)

    runner: SubprocessRunner = subprocess_runner or _default_subprocess_runner
    interpreter = python_executable or sys.executable

    def _runner(
        entry: ModelZooEntryConfig,
        device: str,
        zoo: VideoCodecZoo,
        codec_config: CodecConfig,  # noqa: ARG001 — child re-resolves from manifest
        output_root_arg: Path,  # noqa: ARG001 — closure value wins for cross-proc consistency
    ) -> ZooTrainingReport:
        env = os.environ.copy()
        child_device = device
        if cuda_pinning == "env":
            idx = _device_index(device)
            if idx is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(idx)
                # Inside the child, the pinned GPU is the only visible
                # one, so it presents as ``cuda:0`` regardless of N.
                child_device = "cuda:0"

        argv = [
            interpreter,
            "-m",
            module_name,
            "train",
            "--manifest",
            str(manifest_path),
            "--entry-id",
            entry.entry_id,
            "--device",
            child_device,
            "--output-root",
            str(output_root),
        ]
        completed = runner(argv, env)
        if completed.returncode != 0:
            raise RuntimeError(
                f"subprocess entry runner failed for entry_id="
                f"{entry.entry_id!r}: exit code {completed.returncode}",
            )
        return _read_persisted_report(zoo, entry, device)

    return _runner
