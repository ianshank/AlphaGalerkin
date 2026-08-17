"""Tests for the perf-package device helpers.

Splits into two surfaces:

  * **CPU-safe**: indexed-CUDA parsing, error messaging, fallback to the
    project-wide ``src/poc/device.py`` for bare strings. Always run.
  * **GPU-required**: actual device resolution against real CUDA
    devices. Skipped automatically by the root ``conftest`` when CUDA
    is unavailable (``@pytest.mark.gpu_required``).
"""

from __future__ import annotations

import pytest
import torch

from src.video_compression.perf.device import (
    device_label,
    list_cuda_devices,
    resolve_device,
)

# ---------------------------------------------------------------- CPU-safe


class TestResolveDeviceCPU:
    def test_cpu_returns_cpu(self) -> None:
        device = resolve_device("cpu")
        assert device.type == "cpu"

    def test_auto_picks_cpu_when_no_cuda(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("auto picks cuda when available; covered separately")
        assert resolve_device("auto").type == "cpu"

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown device preference"):
            resolve_device("gpu")

    def test_malformed_indexed_cuda_falls_through_to_value_error(self) -> None:
        # "cuda:" with no digit is malformed; it does not match the
        # indexed regex and the bare resolver does not accept it.
        with pytest.raises(ValueError, match="Unknown device preference"):
            resolve_device("cuda:")

    def test_indexed_cuda_no_cuda_available_raises(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA is present; covered by the GPU-required path")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda:0")

    def test_bare_cuda_no_cuda_available_raises(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA is present")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")


class TestListCudaDevicesCPU:
    def test_returns_empty_when_no_cuda(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA is present")
        assert list_cuda_devices() == []


class TestDeviceLabel:
    def test_cpu_label(self) -> None:
        assert device_label(torch.device("cpu")) == "cpu"


# ----------------------------------------------------------- GPU-required


@pytest.mark.gpu_required
class TestResolveDeviceGPU:
    def test_bare_cuda_returns_default_device(self) -> None:
        device = resolve_device("cuda")
        assert device.type == "cuda"

    def test_index_zero_returns_index_zero(self) -> None:
        device = resolve_device("cuda:0")
        assert device.type == "cuda"
        assert device.index == 0

    def test_index_out_of_range_raises(self) -> None:
        n = torch.cuda.device_count()
        with pytest.raises(RuntimeError, match="only"):
            resolve_device(f"cuda:{n + 100}")

    def test_auto_picks_cuda_when_available(self) -> None:
        device = resolve_device("auto")
        assert device.type == "cuda"


@pytest.mark.gpu_required
class TestListCudaDevicesGPU:
    def test_returns_all_devices(self) -> None:
        devices = list_cuda_devices()
        assert len(devices) == torch.cuda.device_count()
        assert all(d.type == "cuda" for d in devices)

    def test_preserves_index_order(self) -> None:
        devices = list_cuda_devices()
        assert [d.index for d in devices] == list(range(len(devices)))


@pytest.mark.gpu_required
class TestDeviceLabelGPU:
    def test_includes_index_and_name(self) -> None:
        label = device_label(torch.device("cuda:0"))
        assert label.startswith("cuda:0:")
        # Whitespace must be replaced so the label can be embedded in
        # cell keys without breaking the parser.
        assert " " not in label
