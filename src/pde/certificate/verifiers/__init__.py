"""Concrete Track B verifier implementations.

Only the always-available :class:`HeuristicGridResidualVerifier` ships in
WS1. WS2 lands ``TorchResidualVerifier`` and ``JaxVerifyResidualVerifier``
inside this subpackage, behind the ``[certificate-rigorous]`` optional
extra.
"""

from __future__ import annotations

from src.pde.certificate.verifiers.heuristic_grid import HeuristicGridResidualVerifier

__all__ = ["HeuristicGridResidualVerifier"]
