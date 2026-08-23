r"""Verifier registry and dispatch (WS1, spec §3).

Reuses :func:`src.templates.registry.create_typed_registry` — the standard
project registry pattern — rather than rolling a bespoke singleton. The
registry stores verifier **classes**; a call to :func:`get_verifier` returns
an instance.

Sentinel registrations (this module):
    ``autolirpa``, ``delta_crown``, ``jax_verify``, ``dreal`` map to
    :class:`_UnavailableVerifier` subclasses whose ``__init__`` raises
    :class:`~src.pde.certificate.types.VerifierUnavailableError`. This lets
    dispatch tests exercise the error path before the WS2 real backends
    ship. WS2 replaces the sentinel by calling :meth:`VerifierRegistry.replace`
    (via :func:`register_verifier`\'s ``replace=True`` flag).

Design rules honoured (§3):
    * **No hardcoded values** — the registry does not embed default budgets
      or thresholds; those live in :class:`CertificateConfig`.
    * **Backwards compatible** — WS2's real verifiers register with the same
      key strings, so a scenario built against the sentinels ports
      unchanged.
    * **No framework imports at load time** — this module imports only
      Pydantic / stdlib and the local ``types``/``interface`` modules;
      the Gate 5 sibling AST guard proves it.
"""

from __future__ import annotations

import platform
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

from src.pde.certificate.interface import ResidualVerifier
from src.pde.certificate.types import (
    CertificationBudget,
    CertifiedModel,
    CertifiedResidualBound,
    DomainSpec,
    HardwareMeta,
    VerifierBackend,
    VerifierUnavailableError,
)
from src.templates.registry import create_typed_registry

if TYPE_CHECKING:  # pragma: no cover — import-only, avoids runtime cycle
    pass

logger = structlog.get_logger(__name__)

# ``create_typed_registry`` returns (RegistryClass, decorator). The decorator
# it produces takes exactly one positional argument (the registry key), so
# we wrap it below to add the ``replace`` semantics WS2 needs.
_RawVerifierRegistry, _raw_register = create_typed_registry("Verifier")

# Public re-export under a friendly name. Instances are singleton per class.
VerifierRegistry = _RawVerifierRegistry

# ---------------------------------------------------------------------------
# Sentinel: "unavailable" verifier stub
# ---------------------------------------------------------------------------


class _UnavailableVerifier:
    """Base class for backends whose real implementation has not shipped.

    Subclasses set :attr:`backend_name` and :attr:`_extra` (name of the
    optional install extra). Instantiation raises
    :class:`VerifierUnavailableError` — that is the whole point.
    """

    backend_name: VerifierBackend
    _extra: str
    _detail: str = ""

    def __init__(self) -> None:
        # Structured telemetry: the raised exception is clean for the caller
        # but leaves no observable trace under the closed-set log vocabulary.
        # Emit ``certificate.verifier_unavailable`` first so operators can
        # correlate the failure with the WS2 install-extra it names. Event
        # membership is enforced by ``tests/pde/certificate/test_logging.py``.
        logger.warning(
            "certificate.verifier_unavailable",
            backend=str(self.backend_name),
            extra=self._extra,
            detail=self._detail,
        )
        raise VerifierUnavailableError(
            backend=str(self.backend_name),
            extra=self._extra,
            detail=self._detail,
        )

    def certify(
        self,
        *,
        model: CertifiedModel,
        domain: DomainSpec,
        budget: CertificationBudget,
    ) -> CertifiedResidualBound:  # pragma: no cover — unreachable
        # Present only so :class:`_UnavailableVerifier` is structurally a
        # :class:`ResidualVerifier`. ``__init__`` raises before this is ever
        # called.
        raise VerifierUnavailableError(
            backend=str(self.backend_name),
            extra=self._extra,
            detail=self._detail,
        )


def _make_unavailable_verifier(
    *,
    backend: VerifierBackend,
    extra: str,
    detail: str = "",
) -> type[_UnavailableVerifier]:
    """Build a :class:`_UnavailableVerifier` subclass with baked-in identity.

    Factored out so the sentinel registrations below are one-liners and the
    class-body noise stays here.
    """
    return type(
        f"_Unavailable_{backend}",
        (_UnavailableVerifier,),
        {
            "backend_name": backend,
            "_extra": extra,
            "_detail": detail,
        },
    )


# ---------------------------------------------------------------------------
# Decorator: register_verifier
# ---------------------------------------------------------------------------

_replace_lock = threading.Lock()


def register_verifier(
    name: VerifierBackend,
    *,
    replace: bool = False,
) -> Callable[[type[Any]], type[Any]]:
    r"""Decorator to register a verifier class under ``name``.

    Wraps :func:`create_typed_registry`\'s decorator with a ``replace`` flag
    so WS2 can drop-in-swap the sentinel entries without changing call
    sites. The default (``replace=False``) preserves the underlying
    registry's duplicate-raise behaviour.

    Args:
        name: One of :data:`VerifierBackend`. IDE autocomplete on the
            :data:`Literal` catches typos at edit time.
        replace: If True, ``unregister(name)`` first, then register. Used
            by WS2 sentinel overrides. Emits an INFO log line so a diff
            reviewer sees the intentional override.

    Returns:
        The unchanged class (register-and-return idiom).

    """

    def decorator(cls: type[Any]) -> type[Any]:
        registry = VerifierRegistry()
        if replace:
            with _replace_lock:
                existed = registry.unregister(name)
                logger.info(
                    "certificate.verifier_replaced",
                    backend=name,
                    cls=f"{cls.__module__}.{cls.__name__}",
                    existed=existed,
                )
        _raw_register(name)(cls)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Dispatch helper: get_verifier
# ---------------------------------------------------------------------------


def get_verifier(backend: VerifierBackend) -> ResidualVerifier:
    """Return a fresh :class:`ResidualVerifier` instance for ``backend``.

    Args:
        backend: Registry key. Raises :class:`KeyError` if unknown, or
            :class:`VerifierUnavailableError` if the backend maps to a
            sentinel (its real implementation is not installed).

    Returns:
        A fresh verifier instance. The registry stores classes, not
        instances, so each call yields a distinct object (important for
        verifiers that carry per-run scratch state).

    """
    cls = VerifierRegistry().get_or_raise(backend)
    logger.debug(
        "certificate.verifier_selected",
        backend=backend,
        cls=f"{cls.__module__}.{cls.__name__}",
    )
    # Instantiation of a sentinel raises VerifierUnavailableError, which
    # is exactly the fail-closed contract required by §4 AC1.
    instance = cls()
    if not isinstance(instance, ResidualVerifier):
        # Runtime protocol check — catches a registered class that
        # accidentally dropped ``certify``.
        raise TypeError(f"registered verifier {cls!r} does not implement ResidualVerifier")
    return instance


# ---------------------------------------------------------------------------
# Hardware-meta capture
# ---------------------------------------------------------------------------


def capture_hardware_meta(device: str, dtype: str) -> HardwareMeta:
    """Snapshot the current run-time versions for a certificate.

    Import order is deliberately lazy — importing ``torch`` here would put
    a full torch load on the module-import path of ``src/pde/certificate``,
    breaking the "no ML at load time" invariant.

    Args:
        device: One of :data:`~src.pde.certificate.types.Device`.
        dtype: One of :data:`~src.pde.certificate.types.Dtype`.

    Returns:
        A :class:`HardwareMeta` with as many version strings populated as
        the installed environment supports. Missing packages become
        ``None`` — a certificate on a base install still validates.

    """
    torch_version: str | None = None
    jax_version: str | None = None
    jax_verify_version: str | None = None
    cuda_capability: str | None = None

    try:  # pragma: no cover — env-dependent branch
        import torch  # type: ignore[import-not-found, unused-ignore]

        torch_version = str(torch.__version__)
        if device == "cuda" and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            cuda_capability = f"sm_{major}{minor}"
    except Exception:
        pass

    try:  # pragma: no cover — env-dependent branch
        import jax  # type: ignore[import-not-found, unused-ignore]

        jax_version = str(jax.__version__)
    except Exception:
        pass

    try:  # pragma: no cover — env-dependent branch
        import jax_verify  # type: ignore[import-not-found, unused-ignore]

        jax_verify_version = str(getattr(jax_verify, "__version__", "unknown"))
    except Exception:
        pass

    # ``device`` is validated by the outer Pydantic model; we cast here to
    # keep mypy happy without importing the Literal alias twice.
    logger.debug(
        "certificate.hardware_meta_captured",
        device=device,
        dtype=dtype,
        torch_version=torch_version,
        jax_version=jax_version,
        jax_verify_version=jax_verify_version,
        python=sys.version.split()[0],
        platform=platform.system(),
    )
    return HardwareMeta(
        device=device,  # type: ignore[arg-type]
        dtype=dtype,  # type: ignore[arg-type]
        torch_version=torch_version,
        jax_version=jax_version,
        jax_verify_version=jax_verify_version,
        cuda_capability=cuda_capability,
    )


# ---------------------------------------------------------------------------
# Built-in sentinel registrations — one line per not-yet-implemented backend
# ---------------------------------------------------------------------------


def _register_builtin_sentinels() -> None:
    """Populate the registry with placeholder rigorous-tier verifiers.

    Idempotent: early-returns if ``autolirpa`` is already known. WS2 will
    call :func:`register_verifier` with ``replace=True`` from the real
    implementation modules.
    """
    registry = VerifierRegistry()
    if "autolirpa" in registry:
        return

    for backend, extra, detail in (
        (
            "autolirpa",
            "certificate-rigorous",
            "auto_LiRPA (Torch LiRPA family) — WS2 replaces this sentinel.",
        ),
        (
            "delta_crown",
            "certificate-rigorous",
            "∂-CROWN optimisation-based bound — WS2 replaces this sentinel.",
        ),
        (
            "jax_verify",
            "jax",
            (
                "google-deepmind/jax_verify (unmaintained since 2023, ADR 0003 "
                "documents the risk) — WS2 replaces this sentinel."
            ),
        ),
        (
            "dreal",
            "certificate-rigorous",
            "dReal SMT bound — WS2 replaces this sentinel.",
        ),
    ):
        cls = _make_unavailable_verifier(
            backend=backend,  # type: ignore[arg-type]
            extra=extra,
            detail=detail,
        )
        _raw_register(backend)(cls)


_register_builtin_sentinels()


def _reset_for_tests() -> None:
    """Drop all registrations and re-populate the sentinels.

    Only for use from ``conftest.py``. Not part of the public API — the
    leading underscore is load-bearing.
    """
    VerifierRegistry().clear()
    _register_builtin_sentinels()


__all__ = [
    "VerifierRegistry",
    "capture_hardware_meta",
    "get_verifier",
    "register_verifier",
]
