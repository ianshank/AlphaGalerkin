"""Hardware scan + device assignment for the zoo sweep.

Two-step pipeline:

1. :func:`scan_devices` — enumerates visible CUDA devices and reports
   their total VRAM. Falls back to a single-CPU plan when CUDA is
   unavailable.
2. :func:`assign_devices` — given a manifest and the scanned devices,
   produces an :class:`EntryAssignment` per entry per the manifest's
   :class:`DeviceAssignmentStrategy`.

The reference rig is dual-GPU: RTX 5060 Ti 16 GB at ``cuda:0`` and
RTX 5060 8 GB at ``cuda:1``. The default ``vram_aware`` strategy packs
high-VRAM-need entries onto ``cuda:0`` first, then ``cuda:1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from src.video_compression.zoo.config import (
    DeviceAssignmentStrategy,
    ModelZooEntryConfig,
    ModelZooManifestConfig,
)

logger = structlog.get_logger(__name__)

# CPU is represented as a single virtual device so the planner has a
# uniform device list to reason about.
CPU_DEVICE_LABEL: str = "cpu"


@dataclass(frozen=True)
class DeviceCapability:
    """Static capability summary for one accelerator."""

    label: str  # 'cuda:0' / 'cuda:1' / 'cpu'
    name: str  # 'NVIDIA GeForce RTX 5060 Ti' / 'cpu'
    total_vram_mib: float  # 0.0 for CPU
    is_cuda: bool


@dataclass
class EntryAssignment:
    """One entry pinned to one device."""

    entry_id: str
    device: str  # device label string usable by torch.device()
    reason: str  # why the planner picked this device (for logs)


@dataclass
class DevicePlan:
    """Output of :func:`assign_devices`."""

    strategy: DeviceAssignmentStrategy
    devices: list[DeviceCapability]
    assignments: list[EntryAssignment] = field(default_factory=list)

    def device_for(self, entry_id: str) -> str:
        for a in self.assignments:
            if a.entry_id == entry_id:
                return a.device
        raise KeyError(f"no assignment for entry_id={entry_id!r}")

    def by_device(self) -> dict[str, list[str]]:
        """Group entry_ids by assigned device."""
        out: dict[str, list[str]] = {}
        for a in self.assignments:
            out.setdefault(a.device, []).append(a.entry_id)
        return out


def scan_devices() -> list[DeviceCapability]:
    """Enumerate visible accelerators.

    Returns:
        One :class:`DeviceCapability` per CUDA device, plus a single CPU
        fallback entry. The CPU entry is always last so that any
        ``cuda``-preferred strategy naturally picks GPUs first.

    """
    # Defer the torch import so that test doubles can patch
    # ``src.video_compression.zoo.device_planner.torch`` cleanly.
    import torch

    caps: list[DeviceCapability] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mib = float(props.total_memory) / (1024.0 * 1024.0)
            caps.append(
                DeviceCapability(
                    label=f"cuda:{i}",
                    name=props.name,
                    total_vram_mib=total_mib,
                    is_cuda=True,
                ),
            )
        logger.info(
            "zoo.device_planner.cuda_scan",
            n_devices=len(caps),
            devices=[(c.label, c.name, round(c.total_vram_mib, 1)) for c in caps],
        )
    else:
        logger.info("zoo.device_planner.cuda_unavailable")

    caps.append(
        DeviceCapability(
            label=CPU_DEVICE_LABEL,
            name="cpu",
            total_vram_mib=0.0,
            is_cuda=False,
        ),
    )
    return caps


def _cuda_only(devices: list[DeviceCapability]) -> list[DeviceCapability]:
    return [d for d in devices if d.is_cuda]


def _resolve_explicit_device(
    entry: ModelZooEntryConfig,
    devices: list[DeviceCapability],
) -> str | None:
    """Validate an entry's explicit ``device`` against scanned devices."""
    if entry.device is None:
        return None
    if entry.device == CPU_DEVICE_LABEL:
        return CPU_DEVICE_LABEL
    valid = {d.label for d in devices}
    # Allow bare 'cuda' to map to cuda:0 if any cuda device exists.
    if entry.device == "cuda":
        cuda_devs = _cuda_only(devices)
        if not cuda_devs:
            raise ValueError(
                f"entry {entry.entry_id!r} pinned to 'cuda' but no CUDA "
                f"device is visible",
            )
        return cuda_devs[0].label
    if entry.device not in valid:
        raise ValueError(
            f"entry {entry.entry_id!r} pinned to {entry.device!r}, which is "
            f"not visible. Visible devices: {sorted(valid)}",
        )
    return entry.device


def _assign_round_robin(
    entries: list[ModelZooEntryConfig],
    cuda_devices: list[DeviceCapability],
) -> list[EntryAssignment]:
    if not cuda_devices:
        raise ValueError(
            "round_robin strategy requires at least one CUDA device; "
            "use 'single_device' for CPU sweeps",
        )
    out: list[EntryAssignment] = []
    for i, entry in enumerate(entries):
        dev = cuda_devices[i % len(cuda_devices)]
        out.append(
            EntryAssignment(
                entry_id=entry.entry_id,
                device=dev.label,
                reason=f"round_robin index {i} -> {dev.label}",
            ),
        )
    return out


def _assign_vram_aware(
    entries: list[ModelZooEntryConfig],
    cuda_devices: list[DeviceCapability],
    *,
    initial_remaining: dict[str, float] | None = None,
) -> list[EntryAssignment]:
    """Pack entries onto the GPU with the most remaining headroom.

    Reservation accounting: each device starts with the per-device
    starting headroom from ``initial_remaining`` (which is the device's
    total VRAM minus any pre-debit from explicitly-pinned entries) and
    is debited by ``estimated_vram_mib`` per assignment. Entries are
    processed in descending VRAM order so the largest entry gets the
    largest GPU.

    If no device has enough headroom for an entry, that entry falls back
    to the GPU with the most total VRAM and the planner records the
    over-commit reason — the trainer is responsible for OOM recovery.
    """
    if not cuda_devices:
        raise ValueError(
            "vram_aware strategy requires at least one CUDA device; "
            "use 'single_device' for CPU sweeps",
        )

    if initial_remaining is None:
        # Defensive: simplifies test setups that exercise the helper
        # directly without the public ``assign_devices`` shim.
        initial_remaining = {d.label: d.total_vram_mib for d in cuda_devices}
    remaining = {d.label: initial_remaining.get(d.label, d.total_vram_mib) for d in cuda_devices}
    total: dict[str, float] = {d.label: d.total_vram_mib for d in cuda_devices}

    indexed = sorted(
        enumerate(entries),
        key=lambda pair: (-pair[1].estimated_vram_mib, pair[0]),
    )

    assignments_by_idx: dict[int, EntryAssignment] = {}
    for idx, entry in indexed:
        # Pick the device with the most current headroom.
        best_label = max(remaining, key=lambda lbl: remaining[lbl])
        if remaining[best_label] >= entry.estimated_vram_mib:
            remaining[best_label] -= entry.estimated_vram_mib
            reason = (
                f"vram_aware: packed onto {best_label} "
                f"(headroom {remaining[best_label]:.0f} MiB after "
                f"{entry.estimated_vram_mib:.0f} MiB reservation)"
            )
        else:
            # Over-commit: pick the largest *total* VRAM device and
            # still debit its reservation so subsequent entries see the
            # reduced (possibly negative) headroom and don't get packed
            # onto an already-overloaded GPU.
            best_label = max(total, key=lambda lbl: total[lbl])
            remaining[best_label] -= entry.estimated_vram_mib
            reason = (
                f"vram_aware: over-commit on {best_label} "
                f"(needed {entry.estimated_vram_mib:.0f} MiB, total "
                f"{total[best_label]:.0f} MiB, headroom "
                f"{remaining[best_label]:.0f} MiB after reservation)"
            )
        assignments_by_idx[idx] = EntryAssignment(
            entry_id=entry.entry_id,
            device=best_label,
            reason=reason,
        )

    # Restore manifest order for stability.
    return [assignments_by_idx[i] for i in range(len(entries))]


def _assign_single_device(
    entries: list[ModelZooEntryConfig],
    target: str,
) -> list[EntryAssignment]:
    return [
        EntryAssignment(
            entry_id=entry.entry_id,
            device=target,
            reason=f"single_device -> {target}",
        )
        for entry in entries
    ]


def _assign_manual(
    entries: list[ModelZooEntryConfig],
) -> list[EntryAssignment]:
    """Reject any entry that did not provide an explicit pin.

    By the time this function is called, all entries with a non-``None``
    ``device`` field have already been resolved by
    :func:`_resolve_explicit_device` and excluded from the input list.
    Therefore, in MANUAL mode every remaining entry is, by construction,
    a missing pin and the function raises.
    """
    if not entries:
        return []
    bad = ", ".join(repr(e.entry_id) for e in entries)
    raise ValueError(
        f"manual strategy requires every entry to set 'device'; "
        f"entries with device=None: {bad}",
    )


def _resolve_run_target(preference: str, devices: list[DeviceCapability]) -> str:
    """Resolve ``device_preference`` to a concrete device label."""
    if preference == "auto":
        cuda_devs = _cuda_only(devices)
        return cuda_devs[0].label if cuda_devs else CPU_DEVICE_LABEL
    if preference == "cuda":
        cuda_devs = _cuda_only(devices)
        if not cuda_devs:
            raise ValueError(
                "device_preference='cuda' but no CUDA device is visible; "
                "use 'cpu' or 'auto'",
            )
        return cuda_devs[0].label
    valid = {d.label for d in devices}
    if preference not in valid:
        raise ValueError(
            f"device_preference={preference!r} is not visible. Visible: "
            f"{sorted(valid)}",
        )
    return preference


def assign_devices(
    manifest: ModelZooManifestConfig,
    devices: list[DeviceCapability] | None = None,
) -> DevicePlan:
    """Map every manifest entry to a concrete device.

    Args:
        manifest: Manifest to plan for.
        devices: Optional pre-scanned device list (for tests and reuse
            across multiple manifests). When ``None``, :func:`scan_devices`
            is called.

    Returns:
        A :class:`DevicePlan` whose ``assignments`` list is in manifest
        order. Entries with an explicit ``device`` field always honor it;
        all other entries follow the manifest's strategy.

    """
    devs = devices if devices is not None else scan_devices()
    cuda_devs = _cuda_only(devs)
    strategy = manifest.device_assignment_strategy

    # Pre-resolve any explicit device pinning. These entries are then
    # excluded from the strategy-driven assignment so they don't skew
    # round-robin accounting. For VRAM_AWARE we still debit their
    # ``estimated_vram_mib`` from the target device's headroom so
    # subsequent auto-assignments don't get packed onto an
    # already-occupied GPU.
    explicit: dict[str, EntryAssignment] = {}
    auto_entries: list[ModelZooEntryConfig] = []
    pinned_debit: dict[str, float] = {}
    for entry in manifest.entries:
        target = _resolve_explicit_device(entry, devs)
        if target is not None:
            explicit[entry.entry_id] = EntryAssignment(
                entry_id=entry.entry_id,
                device=target,
                reason=f"explicit device pin -> {target}",
            )
            pinned_debit[target] = pinned_debit.get(target, 0.0) + entry.estimated_vram_mib
        else:
            auto_entries.append(entry)

    if strategy is DeviceAssignmentStrategy.MANUAL:
        # In MANUAL mode every entry must have a pin. Force the check.
        auto_assignments = _assign_manual(auto_entries)
    elif strategy is DeviceAssignmentStrategy.SINGLE_DEVICE:
        target = _resolve_run_target(manifest.device_preference, devs)
        auto_assignments = _assign_single_device(auto_entries, target)
    elif strategy is DeviceAssignmentStrategy.ROUND_ROBIN:
        auto_assignments = _assign_round_robin(auto_entries, cuda_devs)
    elif strategy is DeviceAssignmentStrategy.VRAM_AWARE:
        # Seed remaining-headroom with pinned reservations already debited
        # so a CUDA-pinned entry shrinks that device's budget for the
        # subsequent auto pack.
        seeded_remaining = {
            d.label: d.total_vram_mib - pinned_debit.get(d.label, 0.0)
            for d in cuda_devs
        }
        auto_assignments = _assign_vram_aware(
            auto_entries,
            cuda_devs,
            initial_remaining=seeded_remaining,
        )
    else:  # pragma: no cover - exhaustive enum
        raise ValueError(f"unsupported strategy: {strategy!r}")

    # Reassemble in manifest order.
    auto_by_id = {a.entry_id: a for a in auto_assignments}
    final: list[EntryAssignment] = []
    for entry in manifest.entries:
        if entry.entry_id in explicit:
            final.append(explicit[entry.entry_id])
        else:
            final.append(auto_by_id[entry.entry_id])

    plan = DevicePlan(strategy=strategy, devices=devs, assignments=final)
    logger.info(
        "zoo.device_planner.assigned",
        strategy=strategy.value,
        per_device={d: len(ids) for d, ids in plan.by_device().items()},
    )
    return plan
