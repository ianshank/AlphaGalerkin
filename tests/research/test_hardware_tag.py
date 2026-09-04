"""Unit tests for ``collect_hardware_tag``.

Written because the function shipped with **no** direct test and its only
assertion lived in an E2E journey that read ``assert tag.strip()`` -- which
holds identically for the ``UNKNOWN`` default the function exists to replace.
Deleting the call site from ``scripts/run_adaptive_vs_uniform.py`` left six
tests passing (verified). A guard that survives the deletion of the code it
guards is not a guard.

These drive the real function with the three probes monkeypatched, so every
branch -- including the two failure paths, which cannot be reached on a healthy
host -- is exercised without a GPU.
"""

from __future__ import annotations

import os
import platform

import pytest

from src.research.run_manifest import (
    CUDA_PROBE_DEVICE_INDEX,
    UNKNOWN,
    UNKNOWN_ARCH,
    collect_hardware_tag,
)


class _FakeCuda:
    """Stand-in for ``torch.cuda`` with a scripted availability and topology."""

    def __init__(self, *, available: bool, count: int = 1, name: str = "FakeGPU") -> None:
        self._available = available
        self._count = count
        self._name = name
        self.requested_index: int | None = None

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int) -> str:
        self.requested_index = index
        return self._name


@pytest.fixture
def _no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the CPU-only path regardless of the host running the tests."""
    import torch

    monkeypatch.setattr(torch, "cuda", _FakeCuda(available=False))


class TestTheTagIsNotTheDefault:
    """The whole point: a real tag must be distinguishable from ``UNKNOWN``."""

    def test_a_healthy_host_produces_something_other_than_unknown(self, _no_cuda: None) -> None:
        """Pin the assertion the E2E journey should have made.

        ``assert tag.strip()`` passes on ``"unknown"``; this does not.
        """
        assert collect_hardware_tag() != UNKNOWN

    def test_the_cpu_tag_names_the_arch_and_core_count(
        self, monkeypatch: pytest.MonkeyPatch, _no_cuda: None
    ) -> None:
        """Both components are present and come from the real probes."""
        monkeypatch.setattr(platform, "machine", lambda: "testarch")
        monkeypatch.setattr(os, "cpu_count", lambda: 7)
        assert collect_hardware_tag() == "testarch-7cpu"


class TestTheCudaBranch:
    """A GPU host must be identifiable from the tag, including its topology."""

    def test_a_cuda_host_records_count_and_device_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dual-card rig must not read as a single-card one."""
        import torch

        fake = _FakeCuda(available=True, count=2, name="RTX 5060 Ti")
        monkeypatch.setattr(torch, "cuda", fake)
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(os, "cpu_count", lambda: 8)

        tag = collect_hardware_tag()

        assert tag == "x86_64-8cpu-2xRTX 5060 Ti"
        assert fake.requested_index == CUDA_PROBE_DEVICE_INDEX

    def test_a_raising_cuda_probe_degrades_to_the_cpu_tag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A driver mismatch must never be why a benchmark run fails."""
        import torch

        class _Exploding:
            def is_available(self) -> bool:
                raise RuntimeError("driver/library version mismatch")

        monkeypatch.setattr(torch, "cuda", _Exploding())
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(os, "cpu_count", lambda: 4)

        assert collect_hardware_tag() == "x86_64-4cpu"


class TestTheDegradedPaths:
    """Both fallbacks are unreachable on a healthy host, so they are driven here."""

    def test_an_empty_machine_string_uses_the_named_arch_fallback(
        self, monkeypatch: pytest.MonkeyPatch, _no_cuda: None
    ) -> None:
        """``UNKNOWN_ARCH``, not ``UNKNOWN``: the CPU count was still collected."""
        monkeypatch.setattr(platform, "machine", lambda: "")
        monkeypatch.setattr(os, "cpu_count", lambda: 2)
        tag = collect_hardware_tag()
        assert tag == f"{UNKNOWN_ARCH}-2cpu"
        assert tag != UNKNOWN, "a partial probe must not look like a total failure"

    def test_an_unknown_core_count_records_zero_rather_than_crashing(
        self, monkeypatch: pytest.MonkeyPatch, _no_cuda: None
    ) -> None:
        """``os.cpu_count()`` returns None on exotic platforms."""
        monkeypatch.setattr(platform, "machine", lambda: "riscv64")
        monkeypatch.setattr(os, "cpu_count", lambda: None)
        assert collect_hardware_tag() == "riscv64-0cpu"

    def test_a_failing_platform_probe_returns_unknown(
        self, monkeypatch: pytest.MonkeyPatch, _no_cuda: None
    ) -> None:
        """Total failure is the one case that legitimately yields ``UNKNOWN``."""

        def _boom() -> str:
            raise OSError("no platform")

        monkeypatch.setattr(platform, "machine", _boom)
        assert collect_hardware_tag() == UNKNOWN


class TestItIsPubliclyExported:
    """The collector is a public provenance API, like its two siblings.

    It was defined *below* ``__all__`` and omitted from it, so
    ``from src.research.run_manifest import *`` did not export it while
    ``collect_git_provenance`` and ``collect_package_versions`` were.
    """

    def test_it_is_in_dunder_all_beside_the_other_collectors(self) -> None:
        from src.research import run_manifest

        assert "collect_hardware_tag" in run_manifest.__all__
        assert "collect_git_provenance" in run_manifest.__all__
