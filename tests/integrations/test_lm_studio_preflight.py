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


def test_list_models_handles_dict_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_list_models` accepts both attribute-shaped and dict-shaped entries."""
    from unittest.mock import MagicMock

    _patch_no_cuda(monkeypatch)
    sdk = MagicMock()
    response = MagicMock()
    response.data = [{"id": "model-dict"}, MagicMock(id="model-attr")]
    sdk.models.list.return_value = response
    config = LMStudioConfig(model="model-dict")
    report = check_lm_studio_server(config, sdk_client=sdk)
    assert "model-dict" in report.available_models
    assert "model-attr" in report.available_models
    assert report.model_available is True


def test_list_models_accepts_bare_list_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_list_models` handles a bare-list response (no `.data` attribute)."""
    from unittest.mock import MagicMock

    _patch_no_cuda(monkeypatch)
    sdk = MagicMock()
    sdk.models.list.return_value = [MagicMock(id="model-a"), MagicMock(id="model-b")]
    config = LMStudioConfig(model="model-a")
    report = check_lm_studio_server(config, sdk_client=sdk)
    assert report.server_reachable is True
    assert report.available_models == ["model-a", "model-b"]


def test_list_models_rejects_unknown_response_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed responses surface as `LMStudioConnectionError` failure."""
    from unittest.mock import MagicMock

    _patch_no_cuda(monkeypatch)
    sdk = MagicMock()
    sdk.models.list.return_value = "not-a-list"  # noqa: ERA001
    config = LMStudioConfig()
    report = check_lm_studio_server(config, sdk_client=sdk)
    assert report.server_reachable is False
    assert "has no .data attribute" in report.failure_reason


def test_check_vram_with_zero_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `device_count()==0`, VRAM check is treated as skipped (sufficient=True)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    config = LMStudioConfig(min_free_vram_gib=10.0)
    report = check_lm_studio_server(config, sdk_client=_StubClient(_OkModels(ids=[config.model])))
    assert report.free_vram_gib is None
    assert report.vram_sufficient is True
    assert report.passed is True


def test_check_lm_studio_server_closes_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The function closes the SDK client it constructed when `sdk_client=None`."""
    _patch_no_cuda(monkeypatch)
    closed = {"flag": False}

    class _FakeSDKClient:
        def __init__(self, **_: object) -> None:
            self.models = _OkModels(ids=["qwen2.5-14b-instruct"])

        def close(self) -> None:
            closed["flag"] = True

    class _FakeOpenAIModule:
        def OpenAI(self, **kwargs: object) -> _FakeSDKClient:  # noqa: N802
            return _FakeSDKClient(**kwargs)

    import sys

    monkeypatch.setitem(sys.modules, "openai", _FakeOpenAIModule())
    report = check_lm_studio_server(LMStudioConfig())
    assert report.server_reachable is True
    assert closed["flag"] is True


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
