"""Backend-neutral :class:`ResidualVerifier` protocol (WS1, spec §3).

Mirrors :class:`src.backend.interface.BackendInterface` in spirit — a
``runtime_checkable`` :class:`typing.Protocol` that concrete verifiers
implement by shape, not inheritance. This lets the Torch and JAX verifiers
live in independent optional subpackages without importing each other.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.pde.certificate.types import (
    CertificationBudget,
    CertifiedModel,
    CertifiedResidualBound,
    DomainSpec,
    VerifierBackend,
)


@runtime_checkable
class ResidualVerifier(Protocol):
    """Contract every Track B residual verifier honours.

    Implementations are pure functions of ``(model, domain, budget)`` — no
    global state, no framework conversion, no side effects on ``model``.
    Failure returns a :class:`CertifiedResidualBound` with ``rigor='failed'``
    and a populated ``failure_reason`` (spec §4 AC1); implementations
    **must not** raise on budget overrun.
    """

    #: Registry key for this verifier — the same string used in
    #: :attr:`src.pde.certificate.config.CertificateConfig.verifier_backend`
    #: and :meth:`VerifierRegistry.get_or_raise`.
    backend_name: VerifierBackend

    def certify(
        self,
        *,
        model: CertifiedModel,
        domain: DomainSpec,
        budget: CertificationBudget,
    ) -> CertifiedResidualBound:
        """Compute a certified residual bound.

        Args:
            model: The neural operator handle. Verifiers reject
                ``model.backend`` values they cannot consume with
                :class:`~src.pde.certificate.types.VerifierUnavailableError`.
            domain: The certification domain. Verifiers that cannot handle
                the domain kind (e.g. rectangular-only) raise
                :class:`ValueError`.
            budget: Wall-clock / memory ceilings. Overrun returns
                ``rigor='failed'`` unless
                :attr:`CertificationBudget.allow_heuristic_fallback` is True.

        Returns:
            A :class:`CertifiedResidualBound`. Cost fields
            (``compile_wall_s``, ``cert_wall_s``, ``steady_state_wall_s``)
            are populated per §4 AC5.

        """
        ...


__all__ = ["ResidualVerifier"]
