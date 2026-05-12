"""Synchronous LM Studio client wrapping the OpenAI SDK.

LM Studio exposes an OpenAI-compatible endpoint; we use the official
``openai`` SDK (>=1.40,<2.0) configured with the local ``base_url``. The
SDK ships its own connection/timeout error types and accepts both ``seed``
and ``response_format={"type": "json_object"}`` on ``chat.completions.create``.

The SDK import is lazy so that constructing an ``LMStudioConfig`` (or
running unit tests that mock the client) does not require ``openai`` to be
installed.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from src.integrations.lm_studio.config import LMStudioConfig
from src.integrations.lm_studio.prompt import prompt_hash
from src.integrations.lm_studio.schema import (
    LMStudioActionSpaceMismatchError,
    LMStudioConnectionError,
    LMStudioError,
    LMStudioParseError,
    LMStudioPolicyResponse,
    LMStudioPreflightError,
)

if TYPE_CHECKING:
    from src.poc.logging import ScenarioLogger

logger = structlog.get_logger(__name__)


_SYSTEM_INSTRUCTION = (
    "You return strict JSON objects matching the requested schema. "
    "No markdown fencing, no commentary outside the JSON."
)
"""System turn fixed across requests so the response_format guidance stays present."""

_RESPONSE_PREVIEW_CHARS = 240
"""How many characters of a malformed response are echoed back on parse failure."""


def _import_openai() -> Any:
    """Lazy import the ``openai`` package or raise a helpful error.

    Keeps the rest of the codebase importable on a base install that does
    not have the ``[lm-studio]`` extra.
    """
    try:
        import openai  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - import-time error path
        raise LMStudioError(
            "The 'openai' package is required for the LM Studio integration. "
            "Install with: pip install 'alphagalerkin[lm-studio]'"
        ) from e
    return openai


class LMStudioClient:
    """Synchronous client for an OpenAI-compatible LM Studio endpoint.

    Single public method: ``complete_policy``. Retries are bounded by
    ``config.max_retries`` and apply uniformly to:
        - JSON-parse failures (``LMStudioParseError``)
        - Action-space-size mismatches (``LMStudioActionSpaceMismatchError``;
          retry carries a corrective user-turn message)
        - Transient SDK errors (``openai.APIConnectionError``,
          ``openai.APITimeoutError``)

    Non-retryable errors (auth, model-not-found, malformed request) are
    wrapped in ``LMStudioConnectionError`` and surfaced immediately.

    Structured log events:
        - ``lm_studio_call`` — once per ``complete_policy`` call with
          ``prompt_hash``, ``latency_ms``, ``tokens_in``, ``tokens_out``,
          ``parse_ok``, ``retries_used``.
        - ``lm_studio_retry`` — once per retry attempt with the reason.
    """

    def __init__(
        self,
        config: LMStudioConfig,
        *,
        scenario_logger: ScenarioLogger | None = None,
    ) -> None:
        """Construct the client.

        Args:
            config: ``LMStudioConfig``. ``config.preflight_on_construct``
                controls whether the server is probed during init.
            scenario_logger: Optional ``ScenarioLogger`` whose context
                (scenario name, run_id, arm) is bound to each emitted
                event. When ``None`` the module-level structlog logger is
                used.

        """
        if not config.enabled:
            raise LMStudioError(
                "LMStudioClient constructed with config.enabled=False; no requests will be issued."
            )
        self._config = config
        self._scenario_logger = scenario_logger
        self._openai = _import_openai()
        self._sdk_client = self._openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_ms / 1000.0,
        )
        if config.preflight_on_construct:
            # Local import avoids a top-level cycle (preflight imports the
            # client indirectly via type-only references).
            from src.integrations.lm_studio.preflight import (  # noqa: PLC0415
                check_lm_studio_server,
            )

            report = check_lm_studio_server(
                config,
                sdk_client=self._sdk_client,
            )
            if not report.passed:
                raise LMStudioPreflightError(f"LM Studio preflight failed: {report.failure_reason}")

    @property
    def config(self) -> LMStudioConfig:
        """The configuration this client was constructed with."""
        return self._config

    @property
    def sdk_client(self) -> Any:
        """The underlying ``openai.OpenAI`` SDK client (for preflight reuse)."""
        return self._sdk_client

    def complete_policy(
        self,
        prompt: str,
        *,
        expected_action_size: int,
        seed: int,
    ) -> LMStudioPolicyResponse:
        """Issue one policy-prior request and return a validated response.

        Args:
            prompt: Rendered prompt (see ``prompt.build_policy_prompt``).
            expected_action_size: Length the response's ``logits`` list
                must equal. Mismatch triggers a retry with a corrective
                user-turn.
            seed: Sampling seed forwarded to the LLM (LM Studio + llama.cpp
                honour it best-effort).

        Returns:
            Validated ``LMStudioPolicyResponse``.

        Raises:
            LMStudioParseError: All retries exhausted on JSON-parse failures.
            LMStudioActionSpaceMismatchError: All retries exhausted on length
                mismatches.
            LMStudioConnectionError: Transport-level failure that exceeded
                retries, or any non-retryable SDK error.

        """
        if expected_action_size <= 0:
            raise ValueError(f"expected_action_size must be > 0, got {expected_action_size!r}")
        log = self._scenario_logger if self._scenario_logger is not None else logger

        attempt = 0
        prompt_hash_str = prompt_hash(prompt)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        start = time.perf_counter()
        tokens_in = 0
        tokens_out = 0
        last_error: LMStudioError | None = None

        while True:
            try:
                completion = self._sdk_client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    seed=seed,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                last_error = self._coerce_transport_error(exc)
                if not self._retryable(last_error) or attempt >= self._config.max_retries:
                    self._emit_call_log(
                        log,
                        prompt_hash_str,
                        start,
                        tokens_in,
                        tokens_out,
                        parse_ok=False,
                        retries_used=attempt,
                        error=type(last_error).__name__,
                    )
                    raise last_error
                self._sleep_backoff(attempt)
                self._emit_retry_log(
                    log, prompt_hash_str, attempt, reason=type(last_error).__name__
                )
                attempt += 1
                continue

            usage = getattr(completion, "usage", None)
            if usage is not None:
                tokens_in = int(getattr(usage, "prompt_tokens", tokens_in) or tokens_in)
                tokens_out = int(getattr(usage, "completion_tokens", tokens_out) or tokens_out)

            raw_content = self._extract_content(completion)
            try:
                response = self._parse_response(raw_content)
            except LMStudioParseError as exc:
                last_error = exc
                if attempt >= self._config.max_retries:
                    self._emit_call_log(
                        log,
                        prompt_hash_str,
                        start,
                        tokens_in,
                        tokens_out,
                        parse_ok=False,
                        retries_used=attempt,
                        error="LMStudioParseError",
                    )
                    raise
                self._emit_retry_log(log, prompt_hash_str, attempt, reason="LMStudioParseError")
                self._sleep_backoff(attempt)
                attempt += 1
                continue

            if len(response.logits) != expected_action_size:
                mismatch_error = LMStudioActionSpaceMismatchError(
                    f"logits length {len(response.logits)} != expected {expected_action_size}"
                )
                last_error = mismatch_error
                if attempt >= self._config.max_retries:
                    self._emit_call_log(
                        log,
                        prompt_hash_str,
                        start,
                        tokens_in,
                        tokens_out,
                        parse_ok=False,
                        retries_used=attempt,
                        error="LMStudioActionSpaceMismatchError",
                    )
                    raise mismatch_error
                self._emit_retry_log(
                    log, prompt_hash_str, attempt, reason="LMStudioActionSpaceMismatchError"
                )
                messages = self._messages_with_correction(
                    prompt,
                    actual_size=len(response.logits),
                    expected_size=expected_action_size,
                )
                self._sleep_backoff(attempt)
                attempt += 1
                continue

            self._emit_call_log(
                log,
                prompt_hash_str,
                start,
                tokens_in,
                tokens_out,
                parse_ok=True,
                retries_used=attempt,
            )
            return response

    def close(self) -> None:
        """Close the underlying SDK client if it exposes ``close``."""
        close = getattr(self._sdk_client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> LMStudioClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce_transport_error(self, exc: BaseException) -> LMStudioError:
        """Map SDK exceptions to typed ``LMStudioError`` subclasses."""
        connection_type = getattr(self._openai, "APIConnectionError", None)
        timeout_type = getattr(self._openai, "APITimeoutError", None)
        if connection_type is not None and isinstance(exc, connection_type):
            return LMStudioConnectionError(f"openai.APIConnectionError: {exc}")
        if timeout_type is not None and isinstance(exc, timeout_type):
            return LMStudioConnectionError(f"openai.APITimeoutError: {exc}")
        return LMStudioConnectionError(f"{type(exc).__name__}: {exc}")

    def _retryable(self, exc: LMStudioError) -> bool:
        """Connection/timeout errors are retryable; auth and 4xx are not.

        The coercer wraps both transient and permanent SDK errors in
        ``LMStudioConnectionError``; for now we retry any connection error
        and let ``max_retries`` cap the loop. Auth failures will exhaust
        retries quickly because LM Studio surfaces them on every attempt.
        """
        return isinstance(exc, LMStudioConnectionError)

    def _parse_response(self, raw: str) -> LMStudioPolicyResponse:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            preview = raw[:_RESPONSE_PREVIEW_CHARS]
            raise LMStudioParseError(
                f"response was not valid JSON ({exc}); preview={preview!r}"
            ) from exc
        try:
            return LMStudioPolicyResponse.model_validate(data)
        except Exception as exc:
            preview = raw[:_RESPONSE_PREVIEW_CHARS]
            raise LMStudioParseError(
                f"response did not match LMStudioPolicyResponse ({exc}); preview={preview!r}"
            ) from exc

    @staticmethod
    def _extract_content(completion: Any) -> str:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LMStudioParseError("completion has no choices")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise LMStudioParseError("completion choice has no message")
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LMStudioParseError("completion message content is empty")
        return content

    @staticmethod
    def _messages_with_correction(
        prompt: str,
        *,
        actual_size: int,
        expected_size: int,
    ) -> list[dict[str, str]]:
        """Add a corrective user turn after a length-mismatched response."""
        return [
            {"role": "system", "content": _SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"The previous response had logits of length {actual_size}; "
                    f"the action space size is {expected_size}. "
                    f"Return exactly {expected_size} floats in the logits list."
                ),
            },
        ]

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self._config.backoff_base_s * (2**attempt)
        if delay > 0:
            time.sleep(delay)

    def _emit_call_log(
        self,
        log: Any,
        prompt_hash_str: str,
        start: float,
        tokens_in: int,
        tokens_out: int,
        *,
        parse_ok: bool,
        retries_used: int,
        error: str | None = None,
    ) -> None:
        latency_ms = (time.perf_counter() - start) * 1000.0
        log.info(
            "lm_studio_call",
            prompt_hash=prompt_hash_str,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            parse_ok=parse_ok,
            retries_used=retries_used,
            error=error,
        )

    def _emit_retry_log(
        self,
        log: Any,
        prompt_hash_str: str,
        attempt: int,
        *,
        reason: str,
    ) -> None:
        log.warning(
            "lm_studio_retry",
            prompt_hash=prompt_hash_str,
            attempt=attempt,
            reason=reason,
        )
