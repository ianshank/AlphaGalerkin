"""Data contract for ``RefinementSubstrate`` implementations.

``SubstrateConfig`` is the single Pydantic schema both ``TensorGridSubstrate``
(the legacy-behaviour control) and ``SkfemTriSubstrate`` (the element-local
substrate) construct against, per
``specs/refinement_substrate.spec.md``'s Data Contract. Building it ahead of
either concrete substrate avoids scattering the same eight knobs as ad hoc
constructor arguments.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field, field_validator, model_validator

from src.constants import DEFAULT_RATIO_FLOOR
from src.templates.config import BaseModuleConfig

#: The two substrate identities. Named because each was previously spelled as a
#: bare string at nine sites -- including inside both ``describe()`` bodies,
#: which is how ``describe()["kind"]`` came to be able to disagree with
#: ``SubstrateConfig.kind`` (D2).
SUBSTRATE_KIND_TENSOR_GRID: Final[str] = "tensor_grid"
SUBSTRATE_KIND_SKFEM_TRI: Final[str] = "skfem_tri"

#: The two ``error_metric`` members, named for the same reason: both substrates
#: compared ``self._config.error_metric == "quadrature"`` against a bare string
#: and left ``"nodal_rms"`` entirely unspelled as the implicit ``else``, so a
#: third metric would have silently fallen through to the nodal-RMS branch.
ERROR_METRIC_QUADRATURE: Final[str] = "quadrature"
ERROR_METRIC_NODAL_RMS: Final[str] = "nodal_rms"

#: ``SubstrateSolveResult.extra`` key carrying whichever metric ``error_metric``
#: selected, identically on every substrate. The metric-specific keys below stay
#: additive; this one exists because the two implementations previously published
#: the selected value under *different* names (``l2_error_area_weighted`` vs
#: ``l2_error_quadrature``), so no generic consumer could read it (D4).
SUBSTRATE_PRIMARY_L2_KEY: Final[str] = "l2_error_primary"

#: Metric-specific ``extra`` keys, unchanged and still always reported.
SUBSTRATE_NODAL_RMS_L2_KEY: Final[str] = "l2_error_nodal_rms"
SUBSTRATE_QUADRATURE_L2_KEY: Final[str] = "l2_error_quadrature"
SUBSTRATE_AREA_WEIGHTED_L2_KEY: Final[str] = "l2_error_area_weighted"

#: Fields that only mean something for one substrate kind. Setting one away
#: from its default while building the *other* kind is rejected rather than
#: silently ignored -- a typed, validated, described field that does nothing
#: is a worse lie than a magic number, because it looks like a knob.
_KIND_SCOPED_FIELDS: dict[str, tuple[str, ...]] = {
    SUBSTRATE_KIND_SKFEM_TRI: ("element_type", "initial_refinements"),
    SUBSTRATE_KIND_TENSOR_GRID: ("initial_side",),
}

#: Numerical-stability floor for any ratio computation over substrate
#: quantities (e.g. error ratios between two DOF counts).
#:
#: Re-exported from ``src.constants`` rather than redeclared: this was a third
#: independent ``1e-15`` literal, and its original provenance comment was wrong
#: on every count -- it cited ``src/experiments/transfer_baseline_compare`` (a
#: module that does not exist; the real one is under ``src/research/``) and
#: claimed to mirror ``DEFAULT_TRANSFER_RATIO_FLOOR``, which is ``1e-12``, a
#: thousandfold different. The two are the same *knob* with different *values*,
#: so they are named separately and deliberately not unified.
RATIO_FLOOR = DEFAULT_RATIO_FLOOR

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


def resolve_substrate_config(
    config: SubstrateConfig | None,
    *,
    kind: str,
    default_name: str,
) -> SubstrateConfig:
    """Return ``config``, or a default one, having checked it is for ``kind``.

    Both substrates previously wrote the same two lines by hand::

        self._config = config or SubstrateConfig(name="...", kind="...")

    and then hardcoded their kind a *second* time inside ``describe()``. The
    two spellings could disagree, and did: a ``SubstrateConfig(kind="skfem_tri")``
    handed to ``TensorGridSubstrate`` constructed cleanly and then reported
    ``describe()["kind"] == "tensor_grid"``, with the bound structlog logger
    emitting the same wrong value. ``kind`` had no production reader outside its
    own validator, so nothing could catch it -- and ``_KIND_SCOPED_FIELDS``
    structurally cannot, since it rejects a field scoped to the *other* kind,
    never a mismatched ``kind`` itself.

    Shared here, next to the config it validates, so a third substrate cannot
    reintroduce the divergence.

    Args:
        config: The caller's config, or ``None`` to build the default.
        kind: The concrete substrate's own kind. Must be a ``SubstrateConfig.kind``
            member.
        default_name: ``name`` for the default config when ``config`` is ``None``.

    Returns:
        A config whose ``kind`` is ``kind``.

    Raises:
        ValueError: If ``config.kind`` names a different substrate.

    """
    if config is None:
        return SubstrateConfig(name=default_name, kind=kind)  # type: ignore[arg-type]
    if config.kind != kind:
        raise ValueError(
            f"config.kind={config.kind!r} does not match the substrate being built "
            f"({kind!r}). Constructing one substrate with another's config used to "
            f"succeed and then misreport kind in describe() and in every log line."
        )
    return config


def select_primary_l2(config: SubstrateConfig, *, quadrature: float, nodal: float) -> float:
    """Pick the L2 value ``config.error_metric`` names -- and refuse an unknown one.

    Both substrates used to write ``x if error_metric == "quadrature" else y``
    with ``"nodal_rms"`` never spelled: it was the implicit ``else``, so a third
    metric would have silently fallen through to nodal RMS. ``ERROR_METRIC_NODAL_RMS``
    was then added to name that branch -- and read by nothing, because both
    ternaries still compared only against the quadrature member. This is where
    it is finally consumed, and where an unrecognised metric fails loudly instead
    of being absorbed. Pydantic's ``Literal`` makes that branch unreachable
    through a validated config; it exists for a config built some other way.

    Args:
        config: The substrate's config; ``error_metric`` decides.
        quadrature: The value to report under ``ERROR_METRIC_QUADRATURE`` (for
            the tensor grid this is the area-weighted analogue).
        nodal: The value to report under ``ERROR_METRIC_NODAL_RMS``.

    Returns:
        The selected value.

    Raises:
        ValueError: If ``config.error_metric`` is neither known member.

    """
    if config.error_metric == ERROR_METRIC_QUADRATURE:
        return quadrature
    if config.error_metric == ERROR_METRIC_NODAL_RMS:
        return nodal
    raise ValueError(
        f"unknown error_metric {config.error_metric!r}; expected "
        f"{ERROR_METRIC_QUADRATURE!r} or {ERROR_METRIC_NODAL_RMS!r}"
    )


class AdequacyGateConfig(BaseModuleConfig):
    """Thresholds for the substrate adequacy gate (``specs/refinement_substrate.spec.md`` AC7/AC8).

    Promoted out of ``tests/research/test_amr_arena_interpretability.py``, where
    these four values and the predicate reading them lived as module constants.
    That placement made the verdict the spec calls "the gate that makes any
    comparison meaningful" reachable only from pytest: a caller who ran a sweep
    through the public API had no way to ask whether the substrate was adequate.

    Every default is byte-identical to the constant it replaces -- this is a
    move, not a retune. The originating comments are preserved per field because
    each records a measurement, and a threshold whose provenance is lost is a
    threshold nobody dares change.
    """

    #: Uniform P1 L2 rate on the L-shape. Theory gives -2/3; the task-zero spike
    #: measured -0.710. A rate outside this band **on either side** is a defect:
    #: too *good* means the reentrant singularity is not actually in the domain
    #: (AC8's cautionary tale -- a translated mesh produced a confident, entirely
    #: wrong "adaptive loses on skfem too" result, caught only by this tripwire).
    uniform_rate_band: tuple[float, float] = Field(
        default=(-0.85, -0.55),
        description="Inclusive (low, high) band the uniform arm's log-log rate must land in.",
    )

    #: Adaptive must reach at least this steep a rate. Spike measured -1.256;
    #: measured -1.3109 through the production primitives, at the theta the gate
    #: actually passes (``ComparisonParams.marking_fraction = 0.5``) over
    #: ``rate_fit_dof_range``. Quote the theta with any rate: the same substrate
    #: reads -1.31 at theta=0.5 and -1.25 at theta=0.3, so a bare rate is not a
    #: fact.
    adaptive_rate_min: float = Field(
        default=-1.10,
        description="Adaptive log-log rate must be at least this steep (i.e. <= this value).",
    )

    #: Adaptive must beat uniform at matched DOF. Below 1.0 means adaptive wins.
    adaptive_vs_uniform_max_ratio: float = Field(
        default=1.0,
        gt=0.0,
        description="Adaptive/uniform error ratio at matched DOF must be strictly below this.",
    )

    #: The asymptotic window the rate is fitted over. Below it, neither arm has
    #: separated yet. Corrected from the spec's originally pinned ``(200, 2600)``,
    #: which is physically incapable of holding ``RATE_FIT_MIN_POINTS`` uniform
    #: points: a 2D uniform arm quadruples DOF per level, so a 13x window spans
    #: at most two of them.
    rate_fit_dof_range: tuple[float, float] = Field(
        default=(200.0, 4000.0),
        description="(low, high) DOF window the log-log rates are fitted over.",
    )

    #: Level ceilings (runaway guards). Uniform multiplies DOF each level so it
    #: needs very few; adaptive adds DOF slowly and needs many more.
    max_levels_uniform: int = Field(
        default=8, ge=1, description="Refinement levels the uniform control arm may take."
    )
    max_levels_adaptive: int = Field(
        default=40, ge=1, description="Refinement levels the adaptive arm may take."
    )

    @property
    def max_sweep_dof(self) -> int:
        """DOF budget for both arms.

        *Derived* from ``rate_fit_dof_range``'s upper bound rather than retyped,
        so a sweep stops as soon as it has spanned the fitting window. The
        coupling previously lived only in a comment, with ``4000`` written twice.

        Returns ``int``, not ``float``: a DOF budget is a count, and
        ``run_refinement_sweep(max_dof: int)`` types it that way. The fitting
        *window* stays float because it is interpolated over. This also preserves
        value identity with the module constant this replaced
        (``MAX_SWEEP_DOF = RATE_FIT_DOF_RANGE[1]`` over an int pair), which is
        the promotion's whole contract -- a move, not a retune.

        Returns:
            The upper bound of the fitting window, as a DOF count.

        """
        return int(self.rate_fit_dof_range[1])

    @field_validator("uniform_rate_band", "rate_fit_dof_range")
    @classmethod
    def _low_below_high(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Reject an inverted or degenerate interval.

        An inverted band admits nothing and would make the gate fail on every
        input; a degenerate one admits a single float. Both are configuration
        errors that would otherwise present as a mysterious gate result.

        Args:
            value: The (low, high) pair.

        Returns:
            The validated pair.

        Raises:
            ValueError: If ``low >= high``.

        """
        low, high = value
        if low >= high:
            raise ValueError(f"interval low must be < high, got {value!r}")
        return value
