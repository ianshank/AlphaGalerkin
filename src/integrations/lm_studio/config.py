"""Pydantic configuration for the LM Studio integration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LMStudioConfig(BaseModel):
    """Configuration for an OpenAI-compatible local LLM server.

    Every knob is a typed Pydantic field with a documented default. The
    integration imports the ``openai`` SDK lazily, so constructing an
    ``LMStudioConfig`` does not require the SDK to be installed.

    Defaults target a fresh LM Studio install serving Qwen-14B on the
    standard local endpoint.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    enabled: bool = Field(
        default=True,
        description=(
            "Master switch. When False, `LMStudioClient.complete_policy` "
            "raises immediately. Lets the LLM arm be skipped by config "
            "without restructuring the scenario."
        ),
    )
    base_url: str = Field(
        default="http://127.0.0.1:1234/v1",
        description="OpenAI-compatible endpoint (LM Studio default port is 1234).",
        min_length=1,
    )
    model: str = Field(
        default="qwen2.5-14b-instruct",
        description="Model identifier as reported by `/v1/models`.",
        min_length=1,
    )
    api_key: str = Field(
        default="lm-studio",
        description="Dummy key — LM Studio ignores it but the SDK requires one.",
        min_length=1,
    )
    timeout_ms: int = Field(
        default=30_000,
        ge=100,
        le=600_000,
        description=(
            "Per-request HTTP timeout. 30 s is a realistic floor for 14B Q4 "
            "generation at ~256 output tokens."
        ),
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=8,
        description=(
            "Retries on JSON-parse, action-size-mismatch, and "
            "openai.APIConnectionError / openai.APITimeoutError. Each "
            "attempt sleeps `backoff_base_s * 2 ** attempt` before retrying."
        ),
    )
    backoff_base_s: float = Field(
        default=0.25,
        gt=0.0,
        le=10.0,
        description="Exponential-backoff base in seconds.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (low => more deterministic).",
    )
    max_tokens: int = Field(
        default=256,
        ge=16,
        le=2048,
        description=(
            "Cap on LLM output tokens per call. For a 24-action basis-"
            "selection game the JSON body needs ~144 tokens for the logits "
            "list plus value and a short reasoning string. Lower for tighter "
            "p95 latency; higher to allow more reasoning."
        ),
    )
    fallback_to_uniform_on_parse_error: bool = Field(
        default=False,
        description=(
            "When True, the evaluator returns a uniform-over-legal policy "
            "and `value=0.0` if the client raises after exhausting retries. "
            "When False (default), the typed exception propagates and the "
            "scenario fails loud."
        ),
    )
    min_free_vram_gib: float = Field(
        default=10.0,
        ge=0.0,
        le=128.0,
        description=(
            "Preflight free-VRAM floor in GiB. Qwen-14B Q4_K_M needs "
            "roughly 10 GiB resident; below this the LLM will likely OOM."
        ),
    )
    preflight_on_construct: bool = Field(
        default=True,
        description=(
            "Run `check_lm_studio_server` inside `LMStudioClient.__init__`. "
            "Set False for unit tests that should not hit the network."
        ),
    )
