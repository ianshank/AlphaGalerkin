"""Pydantic validation tests for `LMStudioConfig`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.integrations.lm_studio.config import LMStudioConfig


def test_defaults_are_valid() -> None:
    config = LMStudioConfig()
    assert config.enabled is True
    assert config.base_url.endswith("/v1")
    assert config.model
    assert config.timeout_ms >= 100
    assert config.max_retries >= 0
    assert 0.0 <= config.temperature <= 2.0
    assert config.max_tokens >= 16
    assert config.min_free_vram_gib >= 0.0


def test_negative_timeout_rejected() -> None:
    with pytest.raises(ValidationError):
        LMStudioConfig(timeout_ms=-1)


def test_out_of_range_temperature_rejected() -> None:
    with pytest.raises(ValidationError):
        LMStudioConfig(temperature=2.5)


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        LMStudioConfig(unexpected_field=True)  # type: ignore[call-arg]


def test_max_tokens_lower_bound_enforced() -> None:
    with pytest.raises(ValidationError):
        LMStudioConfig(max_tokens=4)
