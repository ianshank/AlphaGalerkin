"""Human-readable inspectors for certificate artifacts and residual bounds.

Pure formatting helpers — no framework imports, no side effects. Intended
for use from a debugger, a jupyter notebook, or a CI log dump.

Kept intentionally minimal (spec §3): the debug surface should grow as the
verifiers ship, not before.
"""

from __future__ import annotations

from src.pde.certificate.types import CertifiedResidualBound


def inspect_bound(bound: CertifiedResidualBound, *, verbose: bool = False) -> str:
    """Return a compact human-readable summary of ``bound``.

    Args:
        bound: The certified residual bound to describe.
        verbose: If True, include hardware provenance and cost accounting;
            if False (default), a single-line summary suitable for a CI log.

    Returns:
        Multi-line string in the ``verbose`` case, single line otherwise.

    """
    if not verbose:
        return (
            f"CertifiedResidualBound(upper={bound.upper_bound:g}, "
            f"rigor={bound.rigor!r}, backend={bound.backend!r}, "
            f"coverage={bound.domain_coverage!r})"
        )
    lines: list[str] = [
        "CertifiedResidualBound",
        f"  upper_bound        : {bound.upper_bound:g}",
        f"  rigor              : {bound.rigor}",
        f"  backend            : {bound.backend}",
        f"  domain_coverage    : {bound.domain_coverage}",
        f"  compile_wall_s     : {bound.compile_wall_s:.6f}",
        f"  cert_wall_s        : {bound.cert_wall_s:.6f}",
        f"  steady_state_wall_s: {bound.steady_state_wall_s:.6f}",
        f"  device             : {bound.hardware_meta.device}",
        f"  dtype              : {bound.hardware_meta.dtype}",
        f"  torch_version      : {bound.hardware_meta.torch_version}",
        f"  jax_version        : {bound.hardware_meta.jax_version}",
        f"  jax_verify_version : {bound.hardware_meta.jax_verify_version}",
    ]
    if bound.failure_reason is not None:
        lines.append(f"  failure_reason     : {bound.failure_reason}")
    if bound.notes:
        lines.append(f"  notes              : {bound.notes}")
    return "\n".join(lines)


__all__ = ["inspect_bound"]
