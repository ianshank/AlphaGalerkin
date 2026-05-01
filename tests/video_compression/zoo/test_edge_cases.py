"""Coverage-closing edge-case tests for the zoo package.

Targets uncovered branches in :mod:`device_planner` and :mod:`storage`:

- ``DevicePlan.device_for`` KeyError on unknown id.
- ``_resolve_explicit_device`` 'cuda' raise when no CUDA visible.
- ``_resolve_explicit_device`` invalid label raise.
- ``_resolve_run_target`` 'cuda' no-CUDA + invalid + cuda:N happy paths.
- ``VideoCodecZoo.list_entries`` returns ``[]`` when root missing.
- ``VideoCodecZoo.load_state_dict`` non-dict bundle TypeError.
- ``VideoCodecZoo.load_metrics`` non-dict payload TypeError.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from src.video_compression.zoo import device_planner
from src.video_compression.zoo.config import (
    DeviceAssignmentStrategy,
    ModelZooEntryConfig,
    ModelZooManifestConfig,
)
from src.video_compression.zoo.device_planner import (
    DeviceCapability,
    DevicePlan,
    EntryAssignment,
    assign_devices,
)
from src.video_compression.zoo.storage import (
    CHECKPOINT_FILENAME,
    METRICS_FILENAME,
    VideoCodecZoo,
)


# ----------------------------------------------------------------------
# Shared fakes (kept local so this file is self-contained)
# ----------------------------------------------------------------------
@dataclass
class _FakeProps:
    name: str
    total_memory: int


class _FakeCuda:
    def __init__(self, devs: list[_FakeProps]) -> None:
        self._devs = devs

    def is_available(self) -> bool:
        return len(self._devs) > 0

    def device_count(self) -> int:
        return len(self._devs)

    def get_device_properties(self, i: int) -> _FakeProps:
        return self._devs[i]


class _FakeTorch:
    def __init__(self, devs: list[_FakeProps]) -> None:
        self.cuda = _FakeCuda(devs)


def _gib(g: float) -> int:
    return int(g * 1024 * 1024 * 1024)


@pytest.fixture
def reference_rig(monkeypatch: pytest.MonkeyPatch) -> list[DeviceCapability]:
    fake = _FakeTorch(
        [
            _FakeProps(name="RTX 5060 Ti", total_memory=_gib(15.9)),
            _FakeProps(name="RTX 5060", total_memory=_gib(8.0)),
        ],
    )
    monkeypatch.setattr(device_planner, "torch", fake, raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return device_planner.scan_devices()


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> list[DeviceCapability]:
    fake = _FakeTorch([])
    monkeypatch.setitem(sys.modules, "torch", fake)
    return device_planner.scan_devices()


def _entry(entry_id: str = "e1", **kw: object) -> ModelZooEntryConfig:
    base: dict[str, object] = {
        "entry_id": entry_id,
        "lambda_rd": 0.01,
        "target_bpp": 0.5,
        "target_psnr_db": 33.0,
        "train_steps": 1000,
    }
    base.update(kw)
    return ModelZooEntryConfig(**base)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# DevicePlan.device_for unknown id
# ----------------------------------------------------------------------
class TestDevicePlanLookup:
    def test_device_for_unknown_raises(self) -> None:
        plan = DevicePlan(
            strategy=DeviceAssignmentStrategy.SINGLE_DEVICE,
            devices=[
                DeviceCapability(
                    label="cpu", name="cpu", total_vram_mib=0.0, is_cuda=False,
                ),
            ],
            assignments=[
                EntryAssignment(entry_id="known", device="cpu", reason="t"),
            ],
        )
        assert plan.device_for("known") == "cpu"
        with pytest.raises(KeyError, match="no assignment"):
            plan.device_for("missing")


# ----------------------------------------------------------------------
# Explicit-device resolution edge cases
# ----------------------------------------------------------------------
class TestExplicitDeviceEdgeCases:
    def test_bare_cuda_resolves_to_cuda0(
        self, reference_rig: list[DeviceCapability]
    ) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry(device="cuda")],
            device_assignment_strategy=DeviceAssignmentStrategy.MANUAL,
        )
        plan = assign_devices(manifest, devices=reference_rig)
        assert plan.device_for("e1") == "cuda:0"

    def test_bare_cuda_no_visible_cuda_raises(
        self, cpu_only: list[DeviceCapability]
    ) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry(device="cuda")],
            device_assignment_strategy=DeviceAssignmentStrategy.MANUAL,
        )
        with pytest.raises(ValueError, match="no CUDA"):
            assign_devices(manifest, devices=cpu_only)

    def test_explicit_cpu_under_vram_aware(
        self, reference_rig: list[DeviceCapability]
    ) -> None:
        # CPU pin must short-circuit the planner without raising.
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[
                _entry(entry_id="auto"),
                _entry(entry_id="forced_cpu", device="cpu"),
            ],
            device_assignment_strategy=DeviceAssignmentStrategy.VRAM_AWARE,
        )
        plan = assign_devices(manifest, devices=reference_rig)
        assert plan.device_for("forced_cpu") == "cpu"
        # auto-assigned entry still goes to a CUDA device.
        assert plan.device_for("auto").startswith("cuda")


# ----------------------------------------------------------------------
# _resolve_run_target paths (covered through SINGLE_DEVICE)
# ----------------------------------------------------------------------
class TestRunTargetResolution:
    def test_cuda_preference_no_cuda_raises(
        self, cpu_only: list[DeviceCapability]
    ) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry()],
            device_assignment_strategy=DeviceAssignmentStrategy.SINGLE_DEVICE,
            device_preference="cuda",
        )
        with pytest.raises(ValueError, match="no CUDA"):
            assign_devices(manifest, devices=cpu_only)

    def test_invalid_preference_raises(
        self, reference_rig: list[DeviceCapability]
    ) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry()],
            device_assignment_strategy=DeviceAssignmentStrategy.SINGLE_DEVICE,
            device_preference="cuda:99",
        )
        with pytest.raises(ValueError, match="not visible"):
            assign_devices(manifest, devices=reference_rig)

    def test_cuda_n_preference_passthrough(
        self, reference_rig: list[DeviceCapability]
    ) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry()],
            device_assignment_strategy=DeviceAssignmentStrategy.SINGLE_DEVICE,
            device_preference="cuda:1",
        )
        plan = assign_devices(manifest, devices=reference_rig)
        assert plan.device_for("e1") == "cuda:1"

    def test_bare_cuda_preference_picks_first_cuda(
        self, reference_rig: list[DeviceCapability]
    ) -> None:
        # SINGLE_DEVICE + 'cuda' must resolve to cuda:0, exercising the
        # ``return cuda_devs[0].label`` branch in _resolve_run_target.
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry()],
            device_assignment_strategy=DeviceAssignmentStrategy.SINGLE_DEVICE,
            device_preference="cuda",
        )
        plan = assign_devices(manifest, devices=reference_rig)
        assert plan.device_for("e1") == "cuda:0"

    def test_vram_aware_no_cuda_raises(
        self, cpu_only: list[DeviceCapability]
    ) -> None:
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry()],
            device_assignment_strategy=DeviceAssignmentStrategy.VRAM_AWARE,
        )
        with pytest.raises(ValueError, match="vram_aware"):
            assign_devices(manifest, devices=cpu_only)

    def test_manual_strategy_unpinned_entry_raises(
        self, reference_rig: list[DeviceCapability]
    ) -> None:
        # Mix of pinned and unpinned entries under MANUAL: the unpinned
        # one must raise the manual-strategy ValueError. This guards the
        # raise branch in ``_assign_manual``.
        manifest = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[
                _entry(entry_id="pinned", device="cuda:0"),
                _entry(entry_id="unpinned"),
            ],
            device_assignment_strategy=DeviceAssignmentStrategy.MANUAL,
        )
        with pytest.raises(ValueError, match="manual strategy requires"):
            assign_devices(manifest, devices=reference_rig)


# ----------------------------------------------------------------------
# VideoCodecZoo edge cases
# ----------------------------------------------------------------------
class TestVideoCodecZooEdges:
    def test_list_entries_empty_when_root_removed(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        # Remove the root after construction to exercise the not-exists
        # branch in list_entries.
        zoo.root.rmdir()
        assert zoo.list_entries() == []

    def test_load_state_dict_non_dict_raises(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        entry_dir = zoo.entry_dir("bad")
        entry_dir.mkdir(parents=True)
        # Write a non-dict checkpoint.
        torch.save([1, 2, 3], entry_dir / CHECKPOINT_FILENAME)
        with pytest.raises(TypeError, match="not a dict"):
            zoo.load_state_dict("bad")

    def test_load_metrics_non_dict_raises(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        entry_dir = zoo.entry_dir("bad")
        entry_dir.mkdir(parents=True)
        # Write a metrics file whose 'metrics' key is not a dict.
        with (entry_dir / METRICS_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump({"metrics": [1, 2, 3], "saved_at": "now"}, fh)
        with pytest.raises(TypeError, match="not a dict"):
            zoo.load_metrics("bad")
