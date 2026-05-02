"""Device planner unit tests.

Uses a fake torch CUDA layer so the dual-GPU assignment logic is
exercised without requiring real hardware. The reference rig
(cuda:0 = RTX 5060 Ti 16 GiB, cuda:1 = RTX 5060 8 GiB) is the headline
fixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.video_compression.zoo import device_planner
from src.video_compression.zoo.config import (
    DeviceAssignmentStrategy,
    ModelZooEntryConfig,
    ModelZooManifestConfig,
)
from src.video_compression.zoo.device_planner import (
    CPU_DEVICE_LABEL,
    DeviceCapability,
    assign_devices,
    scan_devices,
)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
@dataclass
class _FakeProps:
    name: str
    total_memory: int  # bytes


class _FakeCuda:
    def __init__(self, devices: list[_FakeProps]) -> None:
        self._devs = devices

    def is_available(self) -> bool:
        return len(self._devs) > 0

    def device_count(self) -> int:
        return len(self._devs)

    def get_device_properties(self, i: int) -> _FakeProps:
        return self._devs[i]


class _FakeTorch:
    def __init__(self, devices: list[_FakeProps]) -> None:
        self.cuda = _FakeCuda(devices)


def _gib_to_bytes(gib: float) -> int:
    return int(gib * 1024 * 1024 * 1024)


@pytest.fixture
def reference_rig(monkeypatch: pytest.MonkeyPatch) -> list[DeviceCapability]:
    """Patch torch with the dual-GPU reference rig and return scanned caps."""
    fake = _FakeTorch(
        [
            _FakeProps(name="NVIDIA GeForce RTX 5060 Ti", total_memory=_gib_to_bytes(15.9)),
            _FakeProps(name="NVIDIA GeForce RTX 5060", total_memory=_gib_to_bytes(8.0)),
        ],
    )
    monkeypatch.setattr(device_planner, "torch", fake, raising=False)
    # Also need to override the local import in scan_devices.
    import sys

    monkeypatch.setitem(sys.modules, "torch", fake)
    return scan_devices()


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> list[DeviceCapability]:
    fake = _FakeTorch([])
    import sys

    monkeypatch.setitem(sys.modules, "torch", fake)
    return scan_devices()


def _entry(
    entry_id: str,
    *,
    vram: float = 4096.0,
    device: str | None = None,
) -> ModelZooEntryConfig:
    return ModelZooEntryConfig(
        entry_id=entry_id,
        lambda_rd=0.01,
        target_bpp=0.5,
        target_psnr_db=33.0,
        train_steps=1000,
        estimated_vram_mib=vram,
        device=device,
    )


# ----------------------------------------------------------------------
# scan_devices
# ----------------------------------------------------------------------
class TestScanDevices:
    def test_dual_gpu_rig(self, reference_rig: list[DeviceCapability]) -> None:
        # Two CUDA + one CPU fallback.
        assert len(reference_rig) == 3
        assert reference_rig[0].label == "cuda:0"
        assert reference_rig[0].is_cuda
        assert reference_rig[0].total_vram_mib > reference_rig[1].total_vram_mib
        assert reference_rig[1].label == "cuda:1"
        assert reference_rig[2].label == CPU_DEVICE_LABEL
        assert not reference_rig[2].is_cuda

    def test_cpu_only_rig(self, cpu_only: list[DeviceCapability]) -> None:
        assert len(cpu_only) == 1
        assert cpu_only[0].label == CPU_DEVICE_LABEL


# ----------------------------------------------------------------------
# assign_devices: VRAM-aware (dual-GPU acceptance test)
# ----------------------------------------------------------------------
class TestVRAMAwareDualGPU:
    def test_large_entry_packs_to_cuda0(self, reference_rig: list[DeviceCapability]) -> None:
        # 12 GiB entry must land on cuda:0 (16 GiB), not cuda:1 (8 GiB).
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry("big", vram=12000.0)],
            device_assignment_strategy=DeviceAssignmentStrategy.VRAM_AWARE,
        )
        plan = assign_devices(manifest, devices=reference_rig)
        assert plan.device_for("big") == "cuda:0"

    def test_small_entry_packs_to_cuda1_after_cuda0_full(
        self, reference_rig: list[DeviceCapability]
    ) -> None:
        # Two 12 GiB entries: first → cuda:0, second won't fit on cuda:0
        # (only 4 GiB headroom left) but does on cuda:1 (8 GiB).
        # ... actually 12 > 8, so this would be over-commit on whichever
        # has more total. Use mixed sizes for realistic packing.
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[
                _entry("big", vram=12000.0),  # → cuda:0
                _entry("small", vram=3000.0),  # → cuda:1 (more headroom)
            ],
            device_assignment_strategy=DeviceAssignmentStrategy.VRAM_AWARE,
        )
        plan = assign_devices(manifest, devices=reference_rig)
        assert plan.device_for("big") == "cuda:0"
        # After cuda:0 has 4 GiB left and cuda:1 has 8 GiB, the small
        # 3 GiB entry should go to cuda:1.
        assert plan.device_for("small") == "cuda:1"

    def test_eight_point_grid_uses_both_cards(self, reference_rig: list[DeviceCapability]) -> None:
        # 8-point grid sized to mirror lambda_grid.yaml: high-VRAM
        # entries should fill cuda:0, lower-VRAM entries land on cuda:1.
        sizes = [12000.0, 11000.0, 9500.0, 8500.0, 7500.0, 6500.0, 5500.0, 4500.0]
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry(f"l{i}", vram=s) for i, s in enumerate(sizes)],
            device_assignment_strategy=DeviceAssignmentStrategy.VRAM_AWARE,
        )
        plan = assign_devices(manifest, devices=reference_rig)
        per_dev = plan.by_device()
        # Both GPUs must be used.
        assert "cuda:0" in per_dev
        assert "cuda:1" in per_dev
        # Largest entry must be on cuda:0.
        assert "l0" in per_dev["cuda:0"]

    def test_over_commit_falls_back_to_largest_total(
        self, reference_rig: list[DeviceCapability]
    ) -> None:
        # 20 GiB entry exceeds both cards; planner picks the largest
        # total VRAM (cuda:0).
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry("huge", vram=20000.0)],
            device_assignment_strategy=DeviceAssignmentStrategy.VRAM_AWARE,
        )
        plan = assign_devices(manifest, devices=reference_rig)
        assert plan.device_for("huge") == "cuda:0"
        assert "over-commit" in plan.assignments[0].reason


# ----------------------------------------------------------------------
# Round-robin
# ----------------------------------------------------------------------
class TestRoundRobin:
    def test_alternates_between_cards(self, reference_rig: list[DeviceCapability]) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry(f"e{i}") for i in range(4)],
            device_assignment_strategy=DeviceAssignmentStrategy.ROUND_ROBIN,
        )
        plan = assign_devices(manifest, devices=reference_rig)
        assert plan.device_for("e0") == "cuda:0"
        assert plan.device_for("e1") == "cuda:1"
        assert plan.device_for("e2") == "cuda:0"
        assert plan.device_for("e3") == "cuda:1"

    def test_round_robin_no_cuda_raises(self, cpu_only: list[DeviceCapability]) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry("e0")],
            device_assignment_strategy=DeviceAssignmentStrategy.ROUND_ROBIN,
        )
        with pytest.raises(ValueError, match="requires at least one CUDA"):
            assign_devices(manifest, devices=cpu_only)


# ----------------------------------------------------------------------
# Single device + manual + explicit pin
# ----------------------------------------------------------------------
class TestSingleDevice:
    def test_cpu_smoke(self, cpu_only: list[DeviceCapability]) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry("a"), _entry("b")],
            device_assignment_strategy=DeviceAssignmentStrategy.SINGLE_DEVICE,
            device_preference="cpu",
        )
        plan = assign_devices(manifest, devices=cpu_only)
        assert plan.device_for("a") == "cpu"
        assert plan.device_for("b") == "cpu"

    def test_auto_picks_first_cuda(self, reference_rig: list[DeviceCapability]) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry("a")],
            device_assignment_strategy=DeviceAssignmentStrategy.SINGLE_DEVICE,
            device_preference="auto",
        )
        plan = assign_devices(manifest, devices=reference_rig)
        assert plan.device_for("a") == "cuda:0"


class TestManualStrategy:
    def test_requires_explicit_pin(self, reference_rig: list[DeviceCapability]) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry("a", device=None)],
            device_assignment_strategy=DeviceAssignmentStrategy.MANUAL,
        )
        with pytest.raises(ValueError, match="manual strategy"):
            assign_devices(manifest, devices=reference_rig)

    def test_honors_pins(self, reference_rig: list[DeviceCapability]) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[
                _entry("a", device="cuda:0"),
                _entry("b", device="cuda:1"),
            ],
            device_assignment_strategy=DeviceAssignmentStrategy.MANUAL,
        )
        plan = assign_devices(manifest, devices=reference_rig)
        assert plan.device_for("a") == "cuda:0"
        assert plan.device_for("b") == "cuda:1"


class TestExplicitPin:
    def test_pin_overrides_strategy(self, reference_rig: list[DeviceCapability]) -> None:
        # Even under VRAM_AWARE, an explicit device pin wins.
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry("forced", vram=14000.0, device="cuda:1")],
            device_assignment_strategy=DeviceAssignmentStrategy.VRAM_AWARE,
        )
        plan = assign_devices(manifest, devices=reference_rig)
        assert plan.device_for("forced") == "cuda:1"
        assert "explicit" in plan.assignments[0].reason

    def test_invalid_pin_rejected(self, reference_rig: list[DeviceCapability]) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry("bad", device="cuda:99")],
        )
        with pytest.raises(ValueError, match="not visible"):
            assign_devices(manifest, devices=reference_rig)
