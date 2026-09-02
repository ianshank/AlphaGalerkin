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

from pydantic import Field, field_validator, model_validator

from src.templates.config import BaseModuleConfig

#: Fields that only mean something for one substrate kind. Setting one away
#: from its default while building the *other* kind is rejected rather than
#: silently ignored -- a typed, validated, described field that does nothing
#: is a worse lie than a magic number, because it looks like a knob.
_KIND_SCOPED_FIELDS: dict[str, tuple[str, ...]] = {
    "skfem_tri": ("element_type", "initial_refinements"),
    "tensor_grid": ("initial_side",),
}

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
        description="Lagrange order. Scoped to kind='skfem_tri'.",
    )
    initial_refinements: int = Field(
        default=2,
        ge=0,
        le=8,
        description=(
            "Uniform refinements applied to the coarse L-shape before the sweep. "
            "Scoped to kind='skfem_tri'."
        ),
    )
    initial_side: int = Field(
        default=4,
        ge=2,
        le=64,
        description=(
            "Elements per axis; even so the reentrant corner is a node. "
            "Scoped to kind='tensor_grid'."
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
        description=(
            "Fingerprint-keyed solve cache bound. NOT YET WIRED: no substrate "
            "memoises solves today. Declared here alongside "
            "RefinementSubstrate.fingerprint, its only consumer, which lands "
            "with element-local-substrate Slice E (task 7.1) -- the same task "
            "that retires fingerprint's entry in "
            "scripts/audit_abstractions.py::_STAGED_FOR_UPCOMING_TASK."
        ),
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

    @model_validator(mode="after")
    def _reject_fields_scoped_to_the_other_kind(self) -> SubstrateConfig:
        """Refuse a knob that this ``kind`` would silently ignore.

        ``SubstrateConfig(kind="tensor_grid", element_type="P2")`` used to
        construct cleanly, validate cleanly, and then do nothing at all --
        the config equivalent of a dead abstraction. Rejecting it turns a
        silent no-op into an immediate, named error.

        Only fields set *away from their default* are rejected, so a caller
        that never mentions the field is unaffected and every existing
        construction keeps working.
        """
        other_kinds = [k for k in _KIND_SCOPED_FIELDS if k != self.kind]
        offending = [
            field
            for kind in other_kinds
            for field in _KIND_SCOPED_FIELDS[kind]
            if field in self.model_fields_set
            and getattr(self, field) != type(self).model_fields[field].default
        ]
        if offending:
            owners = sorted(
                kind
                for kind in other_kinds
                if any(field in _KIND_SCOPED_FIELDS[kind] for field in offending)
            )
            raise ValueError(
                f"kind={self.kind!r} ignores {sorted(offending)}, which only affect "
                f"{owners}. Setting a field the chosen substrate never reads is "
                f"silently a no-op, so it is rejected instead."
            )
        return self
