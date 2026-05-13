"""Preflight check for the LM Studio integration.

Validates three independent conditions before the integration accepts
traffic:

1. **Server reachable** — ``GET /v1/models`` returns a list (via the
   ``openai`` SDK's ``models.list`` call).
2. **Model loaded** — the configured ``LMStudioConfig.model`` appears in
   the response.
3. **Free VRAM** — ``torch.cuda.mem_get_info`` reports at least
   ``LMStudioConfig.min_free_vram_gib`` GiB free on at least one CUDA
   device.

The check is intentionally synchronous and one-shot. It is invoked by
``LMStudioClient.__init__`` when ``preflight_on_construct=True`` and can
also be called directly from a scenario's ``setup()`` for explicit gating.
"""

from __future__ import annotations

import contextlib
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from src.integrations.lm_studio.config import LMStudioConfig
from src.integrations.lm_studio.schema import LMStudioConnectionError

logger = structlog.get_logger(__name__)


_BYTES_PER_GIB = 1024**3
"""Bytes in one GiB; used to convert ``torch.cuda.mem_get_info`` output."""


class PreflightReport(BaseModel):
    """Outcome of a preflight check.

    ``passed`` is True iff all three boolean fields are True. ``failure_reason``
    carries a human-readable description suitable for the
    ``LMStudioPreflightError`` message.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    server_reachable: bool = Field(..., description="LM Studio /v1/models responded.")
    model_available: bool = Field(
        ...,
        description="Configured model id was present in the /v1/models response.",
    )
    available_models: list[str] = Field(
        default_factory=list,
        description="Model ids returned by the server (empty when unreachable).",
    )
    free_vram_gib: float | None = Field(
        default=None,
        description=(
            "Max free VRAM across visible CUDA devices in GiB. None when "
            "CUDA is unavailable (then the VRAM check is skipped)."
        ),
    )
    vram_sufficient: bool = Field(
        ...,
        description="free_vram_gib >= config.min_free_vram_gib, or CUDA absent (skip).",
    )
    failure_reason: str = Field(
        default="",
        description="Empty when passed; otherwise a one-line failure description.",
    )

    @property
    def passed(self) -> bool:
        """True when every check succeeded (or was skipped because CUDA absent)."""
        return self.server_reachable and self.model_available and self.vram_sufficient


def _list_models(sdk_client: Any) -> list[str]:
    """Return model ids reported by the SDK's ``models.list`` call.

    Coerces every failure mode — transport exceptions raised by the SDK
    *and* malformed response shapes — into ``LMStudioConnectionError`` so
    the caller catches a single typed exception.
    """
    try:
        response = sdk_client.models.list()
    except LMStudioConnectionError:
        raise
    except Exception as exc:
        raise LMStudioConnectionError(f"models.list raised {type(exc).__name__}: {exc}") from exc
    data = getattr(response, "data", None)
    if data is None and isinstance(response, list):
        data = response
    if data is None:
        raise LMStudioConnectionError(
            "models.list response has no .data attribute and is not a list"
        )
    ids: list[str] = []
    for entry in data:
        entry_id = getattr(entry, "id", None)
        if isinstance(entry_id, str):
            ids.append(entry_id)
        elif isinstance(entry, dict) and isinstance(entry.get("id"), str):
            ids.append(entry["id"])
    return ids


def _check_vram(min_free_gib: float) -> tuple[float | None, bool]:
    """Inspect free VRAM across visible CUDA devices.

    Returns:
        Tuple ``(free_gib, sufficient)``. ``free_gib`` is ``None`` when
        CUDA is absent (or ``mem_get_info`` raises); in that case the
        VRAM check is treated as skipped and ``sufficient`` is True so it
        does not block scenarios that intentionally run on a server-side
        GPU we can't introspect.

    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - torch is a hard dep elsewhere
        return None, True
    if not torch.cuda.is_available():
        return None, True
    n_devices = torch.cuda.device_count()
    best_free_gib: float = 0.0
    inspected = False
    for idx in range(n_devices):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(idx)
        except Exception:  # pragma: no cover - per-device probe failure
            continue
        free_gib = float(free_bytes) / _BYTES_PER_GIB
        best_free_gib = max(best_free_gib, free_gib)
        inspected = True
    if not inspected:
        return None, True
    return best_free_gib, best_free_gib >= min_free_gib


def check_lm_studio_server(
    config: LMStudioConfig,
    *,
    sdk_client: Any | None = None,
) -> PreflightReport:
    """Run the LM Studio preflight check.

    Args:
        config: ``LMStudioConfig`` whose ``model`` and ``min_free_vram_gib``
            drive the checks.
        sdk_client: Optional ``openai.OpenAI`` client. When ``None`` (the
            usual call path from a scenario) the function imports
            ``openai`` lazily and builds its own one-shot client.

    Returns:
        A populated ``PreflightReport``. The caller decides whether to
        raise ``LMStudioPreflightError`` (``LMStudioClient`` does, scenario
        code may choose to skip the LLM arm and continue).

    """
    owns_client = sdk_client is None
    if owns_client:
        # Lazy import keeps the base install (no ``openai`` installed)
        # from breaking when this module is imported transitively.
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            return PreflightReport(
                server_reachable=False,
                model_available=False,
                available_models=[],
                free_vram_gib=None,
                vram_sufficient=True,
                failure_reason=f"openai package not installed: {exc}",
            )
        sdk_client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_ms / 1000.0,
        )

    try:
        available_models: list[str] = []
        server_reachable = False
        model_available = False
        failure_reasons: list[str] = []

        try:
            available_models = _list_models(sdk_client)
            server_reachable = True
            logger.debug(
                "lm_studio_preflight_check",
                check="server_reachable",
                outcome=True,
                model_count=len(available_models),
            )
        except LMStudioConnectionError as exc:
            failure_reasons.append(f"server unreachable at {config.base_url}: {exc}")
            logger.debug(
                "lm_studio_preflight_check",
                check="server_reachable",
                outcome=False,
                error=str(exc),
            )

        if server_reachable:
            model_available = config.model in available_models
            logger.debug(
                "lm_studio_preflight_check",
                check="model_available",
                outcome=model_available,
                requested_model=config.model,
            )
            if not model_available:
                failure_reasons.append(
                    f"model {config.model!r} not in /v1/models response "
                    f"(available: {available_models})"
                )

        free_vram_gib, vram_sufficient = _check_vram(config.min_free_vram_gib)
        logger.debug(
            "lm_studio_preflight_check",
            check="vram_sufficient",
            outcome=vram_sufficient,
            free_vram_gib=free_vram_gib,
            min_free_vram_gib=config.min_free_vram_gib,
        )
        if not vram_sufficient and free_vram_gib is not None:
            failure_reasons.append(
                f"free VRAM {free_vram_gib:.2f} GiB < required {config.min_free_vram_gib} GiB"
            )

        report = PreflightReport(
            server_reachable=server_reachable,
            model_available=model_available,
            available_models=available_models,
            free_vram_gib=free_vram_gib,
            vram_sufficient=vram_sufficient,
            failure_reason="; ".join(failure_reasons),
        )
        logger.info(
            "lm_studio_preflight",
            passed=report.passed,
            server_reachable=server_reachable,
            model_available=model_available,
            free_vram_gib=free_vram_gib,
            vram_sufficient=vram_sufficient,
        )
        return report
    finally:
        if owns_client:
            # Close the one-shot SDK client this function constructed so
            # repeated preflight calls don't leak HTTP connections. An
            # externally-supplied client is left untouched.
            close = getattr(sdk_client, "close", None)
            if callable(close):
                # Best-effort cleanup; the report has already been built so
                # a teardown failure here must not mask the real outcome.
                with contextlib.suppress(Exception):  # pragma: no cover
                    close()
