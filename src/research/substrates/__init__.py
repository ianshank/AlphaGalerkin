"""Concrete ``RefinementSubstrate`` implementations and their shared config.

See ``src.refinement.substrate`` for the domain-free ``RefinementSubstrate``
Protocol these implement, and ``specs/refinement_substrate.spec.md`` for the
full data contract and acceptance criteria.
"""

from __future__ import annotations

from src.research.substrates.config import (
    AREA_FLOOR,
    RATE_FIT_MIN_POINTS,
    RATIO_FLOOR,
    SubstrateConfig,
)

__all__ = [
    "AREA_FLOOR",
    "RATE_FIT_MIN_POINTS",
    "RATIO_FLOOR",
    "SubstrateConfig",
]
