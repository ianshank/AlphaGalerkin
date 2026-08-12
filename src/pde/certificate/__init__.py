r"""Verified error certificate foundation for AlphaGalerkin PDE solutions.

This subpackage implements the *foundation* layer described in
``specs/verified_error_certificate.spec.md``:

* :class:`~src.pde.certificate.certificate.Certificate` — the Pydantic artifact
  every pinned scenario emits, schema-versioned and forward-compatible.
* :class:`~src.pde.certificate.config.CertificateConfig` — top-level knobs;
  thresholds reuse the canonical :class:`src.poc.config.MetricThreshold`
  (no parallel schema, per the spec-tree peer-review correction).
* :class:`~src.pde.certificate.stability.StabilityConstantRegistry` — declares
  the operator stability constant :math:`C_0` (or inf-sup :math:`\beta`) source
  per :class:`src.pde.config.PDEType`. Consumers include the Track A / Track B
  estimators (follow-on PRs) and ``specs/operator_gate.spec.md`` when it lands.
* :mod:`src.pde.certificate.logging` — structlog binder producing
  ``certificate.*`` events with a stable ``certificate_id`` per artifact.

Certificates are **batch artifacts**. This subpackage must never be imported
from :mod:`src.mcts` rollout paths — an AST guard in
``tests/pde/certificate/test_import_isolation.py`` enforces that invariant
(Gate 5 of ``.claude/skills/certificate-validation/SKILL.md``).
"""

from __future__ import annotations

from src.pde.certificate.certificate import (
    CERTIFICATE_DOCUMENT_SCHEMA_VERSION,
    Certificate,
    RigorKind,
    TrackKind,
    migrate_certificate_document,
)
from src.pde.certificate.config import CertificateConfig
from src.pde.certificate.logging import (
    CERTIFICATE_LOG_EVENTS,
    bind_certificate_logger,
    new_certificate_id,
)
from src.pde.certificate.stability import (
    StabilityConstantRegistry,
    StabilityEntry,
    StabilitySource,
    register_stability,
)

__all__ = [
    "CERTIFICATE_DOCUMENT_SCHEMA_VERSION",
    "CERTIFICATE_LOG_EVENTS",
    "Certificate",
    "CertificateConfig",
    "RigorKind",
    "StabilityConstantRegistry",
    "StabilityEntry",
    "StabilitySource",
    "TrackKind",
    "bind_certificate_logger",
    "migrate_certificate_document",
    "new_certificate_id",
    "register_stability",
]
