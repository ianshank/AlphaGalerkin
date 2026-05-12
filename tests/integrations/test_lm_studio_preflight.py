"""Preflight check tests — mocked SDK + monkeypatched torch.cuda."""

from __future__ import annotations

from typing import Any

import pytest

from src.integrations.lm_studio.config import LMStudioConfig
from src.integrations.lm_studio.preflight import (
    PreflightReport,
    check_lm_studio_server,
)


class _FailingModels:
    def list(self) -> Any:
        raise RuntimeError("server down")


class _OkModels:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def list(self) -> Any:
        from unittest.mock import MagicMock

        response = MagicMock()
        response.data = [MagicMock(id=mid) for mid in self._ids]
        return response


class _StubClient:
    def __init__(self, models: Any) -> None:
        self.models = models


def _patch_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _patch_cuda_with_free_bytes(monkeypatch: pytest.MonkeyPatch, free_bytes: int) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _idx=0: (free_bytes, free_bytes),
    )


def test_server_unreachable_marks_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_no_cuda(monkeypatch)
    config = LMStudioConfig()
    report = check_lm_studio_server(config, sdk_client=_StubClient(_FailingModels()))
    assert report.server_reachable is False
    assert report.model_available is False
    assert report.passed is False
    assert "server unreachable" in report.failure_reason


def test_model_not_listed_marks_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_no_cuda(monkeypatch)
    config = LMStudioConfig(model="qwen2.5-14b-instruct")
    report = check_lm_studio_server(
        config,
        sdk_client=_StubClient(_OkModels(ids=["llama-7b"])),
    )
    assert report.server_reachable is True
    assert report.model_available is False
    assert report.passed is False
    assert "qwen2.5-14b-instruct" in report.failure_reason


def test_insufficient_vram_marks_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cuda_with_free_bytes(monkeypatch, free_bytes=1 * 1024**3)  # 1 GiB
    config = LMStudioConfig(min_free_vram_gib=10.0)
    report = check_lm_studio_server(
        config,
        sdk_client=_StubClient(_OkModels(ids=[config.model])),
    )
    assert report.vram_sufficient is False
    assert report.passed is False
    assert "free VRAM" in report.failure_reason


def test_preflight_passes_when_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cuda_with_free_bytes(monkeypatch, free_bytes=16 * 1024**3)  # 16 GiB
    config = LMStudioConfig(min_free_vram_gib=10.0)
    report = check_lm_studio_server(
        config,
        sdk_client=_StubClient(_OkModels(ids=[config.model])),
    )
    assert isinstance(report, PreflightReport)
    assert report.passed is True
    assert report.free_vram_gib is not None and report.free_vram_gib >= 10.0


def test_preflight_skips_vram_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """No CUDA => VRAM check is treated as skipped (not blocking)."""
    _patch_no_cuda(monkeypatch)
    config = LMStudioConfig()
    report = check_lm_studio_server(
        config,
        sdk_client=_StubClient(_OkModels(ids=[config.model])),
    )
    assert report.free_vram_gib is None
    assert report.vram_sufficient is True
    assert report.passed is True
