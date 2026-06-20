"""Response-schema and exception-hierarchy tests."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from src.integrations.lm_studio.schema import (
    LMStudioActionSpaceMismatchError,
    LMStudioConnectionError,
    LMStudioError,
    LMStudioParseError,
    LMStudioPolicyResponse,
    LMStudioPreflightError,
)


def test_response_happy_path() -> None:
    response = LMStudioPolicyResponse(logits=[0.1, -0.2, 0.3], value=0.5)
    assert len(response.logits) == 3
    assert response.value == 0.5
    assert response.reasoning == ""


def test_value_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        LMStudioPolicyResponse(logits=[0.0], value=1.5)


def test_nan_logits_rejected() -> None:
    with pytest.raises(ValidationError):
        LMStudioPolicyResponse(logits=[0.0, float("nan")], value=0.0)


def test_inf_logits_rejected() -> None:
    with pytest.raises(ValidationError):
        LMStudioPolicyResponse(logits=[0.0, math.inf], value=0.0)


def test_extra_key_rejected() -> None:
    with pytest.raises(ValidationError):
        LMStudioPolicyResponse(logits=[0.0], value=0.0, extra=True)  # type: ignore[call-arg]


def test_empty_logits_rejected() -> None:
    with pytest.raises(ValidationError):
        LMStudioPolicyResponse(logits=[], value=0.0)


def test_exception_hierarchy() -> None:
    assert issubclass(LMStudioParseError, LMStudioError)
    assert issubclass(LMStudioActionSpaceMismatchError, LMStudioError)
    assert issubclass(LMStudioConnectionError, LMStudioError)
    assert issubclass(LMStudioPreflightError, LMStudioError)
