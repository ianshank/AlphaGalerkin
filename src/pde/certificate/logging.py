"""Structured logging binder for the certificate subpackage.

The event namespace is ``certificate.*``. Every certificate-related log line
carries a stable ``certificate_id`` (16-char hex, matching the artifact field
of the same name) plus ``pde_type``, ``track``, and ``rigor``. That contract
is what makes cross-run traceability possible — grep by ``certificate_id`` and
you have the full lifecycle of one artifact.

Rationale for a shared binder (vs. one ``structlog.get_logger`` per module):
mixing free-form ``.info(...)`` calls with the artifact schema is exactly the
failure mode that produced the fabricated ``0.000209`` transfer-MSE headline
(CLAUDE.md 2026-07-22). The binder makes the event vocabulary explicit and
testable.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.pde.certificate.certificate import _fresh_certificate_id_hex

# Documented event names. Each is a plain string so grep works. New event names
# must be added here first — the ``test_logging.py`` guard asserts every
# ``structlog`` call in the subpackage uses a name from this set.
CERTIFICATE_LOG_EVENTS: frozenset[str] = frozenset(
    {
        "certificate.computed",  # emitted once per successful certificate
        "certificate.failed",  # emitted when a certificate cannot be built
        "certificate.stability_lookup",  # StabilityConstantRegistry.get(...)
        "certificate.stability_registered",  # register() success
        "certificate.stability_replaced",  # replace() escape-hatch fired
        "certificate.budget_overrun",  # Track B exceeded budget
        "certificate.track_routed",  # spec §4 AC3 provenance-based routing
        "certificate.unbounded_operator",  # UNBOUNDED render path taken
        # WS1 additions — verifier boundary lifecycle events.
        "certificate.verifier_selected",  # get_verifier(...) dispatch
        "certificate.verifier_replaced",  # register_verifier(replace=True)
        "certificate.verifier_start",  # ResidualVerifier.certify(...) enter
        "certificate.verifier_end",  # ResidualVerifier.certify(...) exit
        "certificate.compile_start",  # JIT/graph compile begin
        "certificate.compile_end",  # JIT/graph compile complete
        "certificate.parity_check",  # AC2 Torch↔JAX parity comparison
        "certificate.hardware_meta_captured",  # capture_hardware_meta(...)
        # Emitted from ``registry._UnavailableVerifier.__init__`` immediately
        # before it raises :class:`VerifierUnavailableError`. The exception is
        # a clean fail-closed signal but has no telemetry footprint on its own;
        # this event is what makes "operator asked for a backend that isn't
        # installed" observable across the closed-set log vocabulary.
        "certificate.verifier_unavailable",
    }
)


def new_certificate_id() -> str:
    """Return a fresh 16-char hex ID suitable for :class:`Certificate.certificate_id`.

    Shared with the artifact module so the ID space is identical whether the
    caller is the artifact factory or the logging binder (test guard).
    """
    return _fresh_certificate_id_hex()


def bind_certificate_logger(
    *,
    certificate_id: str,
    pde_type: str,
    track: str,
    rigor: str,
    **extra: Any,
) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to a certificate's identity.

    The four required kwargs mirror the artifact's four identity fields —
    if a caller cannot supply them, they should not be logging under this
    binder. ``**extra`` is preserved so downstream call sites can bind e.g.
    ``scenario_name`` or ``verifier_backend`` without a wrapper.
    """
    logger = structlog.get_logger("src.pde.certificate")
    return logger.bind(
        certificate_id=certificate_id,
        pde_type=pde_type,
        track=track,
        rigor=rigor,
        **extra,
    )


__all__ = [
    "CERTIFICATE_LOG_EVENTS",
    "bind_certificate_logger",
    "new_certificate_id",
]
