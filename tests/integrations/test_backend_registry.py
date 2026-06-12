"""Tests for the OpenAI-compatible backend-profile registry (WS1).

Covers:
    - built-in profile registration + lookup + duplicate/unknown errors,
    - ``apply_backend_defaults`` fill-unset semantics (explicit values win),
    - the ``vram_check_mode`` preflight conditioning for remote backends,
    - backwards compatibility of ``LMStudioConfig`` (old configs still parse,
      ``backend`` defaults to ``lm_studio``, behaviour is unchanged).
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from src.integrations.lm_studio.config import LMStudioConfig
from src.integrations.openai_compat.registry import (
    BACKEND_REGISTRY,
    BackendProfile,
    apply_backend_defaults,
    get_backend,
    list_backends,
    register_backend,
)


@pytest.fixture
def isolated_registry() -> Generator[None, None, None]:
    """Snapshot and restore the global registry so tests don't leak state."""
    snapshot = dict(BACKEND_REGISTRY)
    try:
        yield
    finally:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------


def test_builtin_backends_registered() -> None:
    assert set(list_backends()) >= {"lm_studio", "vllm", "llama_cpp"}


@pytest.mark.parametrize(
    ("name", "expected_mode"),
    [("lm_studio", "local"), ("vllm", "off"), ("llama_cpp", "off")],
)
def test_get_backend_returns_profile(name: str, expected_mode: str) -> None:
    profile = get_backend(name)
    assert profile.name == name
    assert profile.default_vram_check_mode == expected_mode
    assert profile.default_base_url.startswith("http")
    assert profile.default_model


def test_get_backend_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown LLM backend"):
        get_backend("does_not_exist")


def test_lm_studio_profile_matches_historical_defaults() -> None:
    """The lm_studio profile must equal LMStudioConfig's field defaults."""
    profile = get_backend("lm_studio")
    default_cfg = LMStudioConfig()
    assert profile.default_base_url == default_cfg.base_url
    assert profile.default_model == default_cfg.model
    assert profile.default_vram_check_mode == default_cfg.vram_check_mode


def test_register_backend_roundtrip(isolated_registry: None) -> None:
    profile = BackendProfile(
        name="custom_test_backend",
        default_base_url="http://127.0.0.1:9999/v1",
        default_model="custom-model",
        default_vram_check_mode="off",
    )
    returned = register_backend(profile)
    assert returned is profile
    assert get_backend("custom_test_backend") is profile


def test_register_backend_duplicate_raises(isolated_registry: None) -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_backend(get_backend("lm_studio"))


def test_backend_profile_is_frozen() -> None:
    profile = get_backend("vllm")
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen raises ValidationError
        profile.default_model = "mutated"  # type: ignore[misc]


def test_backend_profile_forbids_extra() -> None:
    with pytest.raises(Exception):  # noqa: B017
        BackendProfile(
            name="x",
            default_base_url="http://x/v1",
            default_model="m",
            default_vram_check_mode="off",
            unexpected_field=1,  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# apply_backend_defaults
# ---------------------------------------------------------------------------


def test_apply_defaults_lm_studio_is_noop() -> None:
    cfg = LMStudioConfig()
    resolved = apply_backend_defaults(cfg)
    assert resolved.base_url == cfg.base_url
    assert resolved.model == cfg.model
    assert resolved.vram_check_mode == "local"


def test_apply_defaults_vllm_fills_all_unset() -> None:
    cfg = LMStudioConfig(backend="vllm")
    resolved = apply_backend_defaults(cfg)
    profile = get_backend("vllm")
    assert resolved.base_url == profile.default_base_url
    assert resolved.model == profile.default_model
    assert resolved.vram_check_mode == "off"


def test_apply_defaults_llama_cpp_turns_vram_off() -> None:
    resolved = apply_backend_defaults(LMStudioConfig(backend="llama_cpp"))
    assert resolved.vram_check_mode == "off"
    assert resolved.base_url == get_backend("llama_cpp").default_base_url


def test_apply_defaults_explicit_values_win() -> None:
    cfg = LMStudioConfig(
        backend="vllm",
        base_url="http://gpu-box:1234/v1",
        vram_check_mode="local",
    )
    resolved = apply_backend_defaults(cfg)
    # User-set fields preserved...
    assert resolved.base_url == "http://gpu-box:1234/v1"
    assert resolved.vram_check_mode == "local"
    # ...unset field (model) filled from the vllm profile.
    assert resolved.model == get_backend("vllm").default_model


def test_apply_defaults_preserves_other_fields() -> None:
    cfg = LMStudioConfig(backend="vllm", temperature=0.7, max_retries=5)
    resolved = apply_backend_defaults(cfg)
    assert resolved.temperature == 0.7
    assert resolved.max_retries == 5


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def test_config_without_backend_defaults_to_lm_studio() -> None:
    """Old YAML/dicts (no backend/vram_check_mode keys) must still parse."""
    cfg = LMStudioConfig.model_validate(
        {
            "enabled": True,
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "qwen2.5-14b-instruct",
        }
    )
    assert cfg.backend == "lm_studio"
    assert cfg.vram_check_mode == "local"


def test_invalid_backend_rejected() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        LMStudioConfig(backend="openai_cloud")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Preflight vram_check_mode conditioning
# ---------------------------------------------------------------------------


def _fake_sdk(fake_openai: Any) -> Any:
    """Build a fake OpenAI SDK client so preflight skips its own import path."""
    return fake_openai.OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="k", timeout=1.0)


def test_preflight_off_mode_skips_local_vram_probe(
    monkeypatch: pytest.MonkeyPatch,
    fake_openai: Any,
) -> None:
    """With vram_check_mode='off', _check_vram must not be invoked."""
    from src.integrations.lm_studio import preflight

    def _boom(_min_free_gib: float) -> tuple[float | None, bool]:
        raise AssertionError("_check_vram must not be called when mode='off'")

    monkeypatch.setattr(preflight, "_check_vram", _boom)

    cfg = LMStudioConfig(backend="vllm", vram_check_mode="off", model="qwen2.5-14b-instruct")
    report = preflight.check_lm_studio_server(cfg, sdk_client=_fake_sdk(fake_openai))
    assert report.server_reachable is True
    assert report.free_vram_gib is None
    assert report.vram_sufficient is True


def test_preflight_local_mode_invokes_vram_probe(
    monkeypatch: pytest.MonkeyPatch,
    fake_openai: Any,
) -> None:
    from src.integrations.lm_studio import preflight

    called: dict[str, bool] = {"hit": False}

    def _probe(_min_free_gib: float) -> tuple[float | None, bool]:
        called["hit"] = True
        return 42.0, True

    monkeypatch.setattr(preflight, "_check_vram", _probe)

    cfg = LMStudioConfig(vram_check_mode="local", model="qwen2.5-14b-instruct")
    report = preflight.check_lm_studio_server(cfg, sdk_client=_fake_sdk(fake_openai))
    assert called["hit"] is True
    assert report.free_vram_gib == 42.0
