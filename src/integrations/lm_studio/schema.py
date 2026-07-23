"""Pydantic response schema and typed exceptions for the LM Studio integration."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LMStudioError(Exception):
    """Base class for all LM Studio integration errors."""


class LMStudioParseError(LMStudioError):
    """The LLM returned a response that could not be parsed into ``LMStudioPolicyResponse``.

    Raised after all configured retries are exhausted. Carries the most
    recent raw response (truncated) so callers can log forensic detail.
    """


class LMStudioActionSpaceMismatchError(LMStudioError):
    """The LLM response has the wrong ``len(logits)`` for the action space.

    Treated as a retry-able failure inside ``LMStudioClient`` (the retry
    adds a corrective user-turn). Surfaces only when retries are exhausted.
    """


class LMStudioConnectionError(LMStudioError):
    """Could not reach the LM Studio endpoint (DNS, refused, timeout)."""


class LMStudioPreflightError(LMStudioError):
    """Preflight check failed (server unreachable, model missing, insufficient VRAM)."""


class LMStudioPolicyResponse(BaseModel):
    """Structured policy response from the LLM.

    The LLM must produce JSON matching this schema. Length validation
    against ``expected_action_size`` happens in the client after parsing.
    """

    model_config = ConfigDict(extra="forbid")

    logits: list[float] = Field(
        ...,
        description=(
            "One scalar per action in the basis-selection action space. "
            "Length must equal the action_space_size passed to the client. "
            "NaN/Inf are rejected; illegal-action masking happens later in "
            "the evaluator."
        ),
    )
    value: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Scalar value estimate in [-1, 1] (MCTS sign convention).",
    )
    reasoning: str = Field(
        default="",
        description="Optional free-text rationale logged with the call.",
    )

    @field_validator("logits")
    @classmethod
    def _no_nan_or_inf(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("logits must be non-empty")
        for i, x in enumerate(v):
            if not math.isfinite(x):
                raise ValueError(f"logits[{i}] is non-finite: {x!r}")
        return v
