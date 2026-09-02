"""Data contract for ``RefinementSubstrate`` implementations.

``SubstrateConfig`` is the single Pydantic schema both ``TensorGridSubstrate``
(the legacy-behaviour control) and ``SkfemTriSubstrate`` (the element-local
substrate) construct against, per
``specs/refinement_substrate.spec.md``'s Data Contract. Building it ahead of
either concrete substrate avoids scattering the same eight knobs as ad hoc
constructor arguments.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from src.templates.config import BaseModuleConfig

#: Numerical-stability floor for any ratio computation over substrate
#: quantities (e.g. error ratios between two DOF counts), mirroring
#: ``DEFAULT_TRANSFER_RATIO_FLOOR`` in ``src/experiments/transfer_baseline_compare``-
#: adjacent code.
RATIO_FLOOR = 1e-15

#: Numerical-stability floor for element/cell area computations, guarding
#: against division by a degenerate (near-zero-area) element.
AREA_FLOOR = 1e-30

#: Minimum number of (DOF, error) points required before a log-log
#: convergence rate is fit; fewer points make the fitted slope meaningless.
RATE_FIT_MIN_POINTS = 3


class SubstrateConfig(BaseModuleConfig):
    """Typed configuration for a ``RefinementSubstrate`` implementation."""

    kind: Literal["tensor_grid", "skfem_tri"] = Field(
        default="skfem_tri",
        description=(
            "Which substrate to build. 'tensor_grid' reproduces today's "
            "behaviour and is the control."
        ),
    )
    element_type: Literal["P1", "P2", "P3"] = Field(
        default="P1",
        description="Lagrange order ('skfem_tri' only).",
    )
    initial_refinements: int = Field(
        default=2,
        ge=0,
        le=8,
        description="Uniform refinements applied to the coarse L-shape before the sweep.",
    )
    initial_side: int = Field(
        default=4,
        ge=2,
        le=64,
        description=(
            "Elements per axis ('tensor_grid' only); even so the reentrant corner is a node."
        ),
    )
    marking_variant: Literal["squared", "linear"] = Field(
        default="squared",
        description=(
            "Dörfler bulk quantity. 'squared' is the textbook form; 'linear' "
            "reproduces fem_baseline's existing behaviour."
        ),
    )
    error_metric: Literal["quadrature", "nodal_rms"] = Field(
        default="quadrature",
        description=(
            "Which L2 the substrate reports. 'nodal_rms' exists only to reproduce legacy numbers."
        ),
    )
    enforce_immutable_meshes: bool = Field(
        default=True,
        description="Clear numpy write flags on mesh arrays.",
    )
    solve_cache_max_entries: int = Field(
        default=4096,
        ge=1,
        le=1_000_000,
        description="Fingerprint-keyed solve cache bound.",
    )

    @field_validator("initial_side")
    @classmethod
    def _validate_initial_side_even(cls, v: int) -> int:
        """Reject an odd side count: the reentrant corner must land on a node."""
        if v % 2 != 0:
            raise ValueError(
                f"initial_side must be even so the reentrant corner is a node; got {v}"
            )
        return v
