"""Concrete ``RefinementSubstrate`` implementations and their shared config.

See ``src.refinement.substrate`` for the domain-free ``RefinementSubstrate``
Protocol these implement, and ``specs/refinement_substrate.spec.md`` for the
full data contract and acceptance criteria.
"""

from __future__ import annotations

from src.research.substrates.config import (
    AREA_FLOOR,
    ERROR_METRIC_NODAL_RMS,
    ERROR_METRIC_QUADRATURE,
    RATE_FIT_MIN_POINTS,
    RATIO_FLOOR,
    SUBSTRATE_AREA_WEIGHTED_L2_KEY,
    SUBSTRATE_KIND_SKFEM_TRI,
    SUBSTRATE_KIND_TENSOR_GRID,
    SUBSTRATE_NODAL_RMS_L2_KEY,
    SUBSTRATE_PRIMARY_L2_KEY,
    SUBSTRATE_QUADRATURE_L2_KEY,
    SubstrateConfig,
    resolve_substrate_config,
)

__all__ = [
    "AREA_FLOOR",
    "ERROR_METRIC_NODAL_RMS",
    "ERROR_METRIC_QUADRATURE",
    "RATE_FIT_MIN_POINTS",
    "RATIO_FLOOR",
    "SUBSTRATE_AREA_WEIGHTED_L2_KEY",
    "SUBSTRATE_KIND_SKFEM_TRI",
    "SUBSTRATE_KIND_TENSOR_GRID",
    "SUBSTRATE_NODAL_RMS_L2_KEY",
    "SUBSTRATE_PRIMARY_L2_KEY",
    "SUBSTRATE_QUADRATURE_L2_KEY",
    "SubstrateConfig",
    "resolve_substrate_config",
]
