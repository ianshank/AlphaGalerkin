"""Tests for the shared GPU-preferred device resolver."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from src.poc.device import resolve_device


class TestResolveDevice:
    def test_cpu_always_resolves(self) -> None:
        assert resolve_device("cpu") == torch.device("cpu")

    def test_auto_falls_back_to_cpu_without_cuda(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA available; cannot exercise the fallback path.")
        assert resolve_device("auto") == torch.device("cpu")

    def test_cuda_raises_when_unavailable(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA available; cannot exercise the failure path.")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")

    def test_cuda_message_includes_context(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA available.")
        with pytest.raises(RuntimeError, match="MyScenario"):
            resolve_device("cuda", context="MyScenario")

    def test_unknown_preference_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown device preference"):
            resolve_device("tpu")

    def test_cuda_resolves_when_available(self) -> None:
        # Patch torch.cuda.is_available so we cover the success branch even
        # without a GPU on the CI runner.
        with patch.object(torch.cuda, "is_available", return_value=True):
            assert resolve_device("cuda") == torch.device("cuda")
            assert resolve_device("auto") == torch.device("cuda")
