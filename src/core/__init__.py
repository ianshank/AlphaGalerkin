"""Core protocols and registry abstractions for AlphaGalerkin."""

from __future__ import annotations

from src.core.protocols import (
    EvaluatorProtocol,
    GameProtocol,
    OperatorProtocol,
    SolverProtocol,
)
from src.core.registry import Registry

__all__ = [
    "EvaluatorProtocol",
    "GameProtocol",
    "OperatorProtocol",
    "Registry",
    "SolverProtocol",
]
