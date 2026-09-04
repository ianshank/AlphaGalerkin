"""Validation tests for ``SubstrateConfig``.

The shared data contract every ``RefinementSubstrate`` implementation
(``TensorGridSubstrate``, ``SkfemTriSubstrate``) constructs against. See
``specs/refinement_substrate.spec.md``'s Data Contract table for the
field-by-field source of truth this file pins.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.research.substrates.config import (
    MIN_RATE_FIT_DOF,
    RATE_FIT_MIN_POINTS,
    RATIO_FLOOR,
    AdequacyGateConfig,
    SubstrateConfig,
    select_primary_l2,
)
from src.research.substrates.sweep import RateSeparation, gate_violations

#: A measurement that passes the pinned gate. Values are the AC7 skfem readings
#: recorded in CLAUDE.md (adaptive -1.3109, uniform -0.6710, ratio 0.0946),
#: rounded -- named so "the healthy case still passes" is a test about the
#: measured substrate rather than about arbitrary numbers.
HEALTHY_ADAPTIVE_RATE: float = -1.31
HEALTHY_UNIFORM_RATE: float = -0.67
HEALTHY_ERROR_RATIO: float = 0.095
HEALTHY_MATCHED_DOF: float = 1000.0

#: ``max_sweep_dof`` of the shipped gate. Named so the defaults-still-validate
#: test asserts the real value rather than merely "it constructed".
DEFAULT_MAX_SWEEP_DOF: int = 4000


class TestSubstrateConfigDefaults:
    def test_defaults_match_spec(self) -> None:
        config = SubstrateConfig(name="test")
        assert config.kind == "skfem_tri"
        assert config.element_type == "P1"
        assert config.initial_refinements == 2
        assert config.initial_side == 4
        assert config.marking_variant == "squared"
        assert config.error_metric == "quadrature"
        assert config.enforce_immutable_meshes is True
        assert config.solve_cache_max_entries == 4096


class TestSubstrateConfigValidation:
    @pytest.mark.parametrize("kind", ["tensor_grid", "skfem_tri"])
    def test_accepts_valid_kind(self, kind: str) -> None:
        assert SubstrateConfig(name="test", kind=kind).kind == kind

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", kind="quadtree")  # type: ignore[arg-type]

    @pytest.mark.parametrize("element_type", ["P1", "P2", "P3"])
    def test_accepts_valid_element_type(self, element_type: str) -> None:
        assert SubstrateConfig(name="test", element_type=element_type).element_type == (
            element_type
        )

    def test_rejects_unknown_element_type(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", element_type="P4")  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [-1, 9])
    def test_initial_refinements_bounds(self, value: int) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", initial_refinements=value)

    def test_initial_refinements_boundary_values_accepted(self) -> None:
        assert SubstrateConfig(name="test", initial_refinements=0).initial_refinements == 0
        assert SubstrateConfig(name="test", initial_refinements=8).initial_refinements == 8

    @pytest.mark.parametrize("value", [3, 5, 7])
    def test_rejects_odd_initial_side(self, value: int) -> None:
        with pytest.raises(ValidationError, match="must be even"):
            SubstrateConfig(name="test", initial_side=value)

    @pytest.mark.parametrize("value", [2, 4, 64])
    def test_accepts_even_initial_side_within_bounds(self, value: int) -> None:
        """``initial_side`` is tensor-grid-scoped, so this must name that kind.

        It reads as boilerplate but is not: on the default
        ``kind="skfem_tri"`` this construction is now *rejected*, because a
        tensor-grid knob on an skfem config is a silent no-op. Two of these
        parametrisations went red the moment the scope validator landed --
        which is the validator doing its job on real calling code, not just
        on a hypothetical.
        """
        config = SubstrateConfig(name="test", kind="tensor_grid", initial_side=value)
        assert config.initial_side == value

    @pytest.mark.parametrize("value", [0, 66])
    def test_initial_side_out_of_bounds_rejected(self, value: int) -> None:
        """Probed with EVEN out-of-range values on the RIGHT kind, deliberately.

        Two earlier versions of this test never reached the bounds at all. The
        first used ``[1, 65]`` -- both odd -- so the parity validator rejected
        them with ``ge``/``le`` deleted. The second switched to ``[0, 66]`` but
        left ``kind`` at its default ``"skfem_tri"``, and ``initial_side`` is
        scoped to ``"tensor_grid"``, so ``_reject_fields_scoped_to_the_other_kind``
        rejected *any* value before Pydantic looked at the number. A mutation
        check (delete the bounds; the test must fail) caught both. With the
        kind set and even values, only the bounds can reject these.
        """
        with pytest.raises(ValidationError, match="greater than or equal|less than or equal"):
            SubstrateConfig(name="test", kind="tensor_grid", initial_side=value)

    @pytest.mark.parametrize("variant", ["squared", "linear"])
    def test_accepts_valid_marking_variant(self, variant: str) -> None:
        assert SubstrateConfig(name="test", marking_variant=variant).marking_variant == variant

    def test_rejects_unknown_marking_variant(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", marking_variant="cubic")  # type: ignore[arg-type]

    @pytest.mark.parametrize("metric", ["quadrature", "nodal_rms"])
    def test_accepts_valid_error_metric(self, metric: str) -> None:
        assert SubstrateConfig(name="test", error_metric=metric).error_metric == metric

    def test_rejects_unknown_error_metric(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", error_metric="max_norm")  # type: ignore[arg-type]

    def test_solve_cache_max_entries_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", solve_cache_max_entries=0)
        assert SubstrateConfig(name="test", solve_cache_max_entries=1).solve_cache_max_entries == 1

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", not_a_real_field=1)  # type: ignore[call-arg]


class TestKindScopedFieldValidation:
    """A knob the chosen substrate never reads must be rejected, not ignored.

    ``SubstrateConfig(kind="tensor_grid", element_type="P2")`` previously
    constructed cleanly, validated cleanly, and did nothing — the config
    equivalent of a dead abstraction, and worse than a magic number because it
    looks like a working knob.
    """

    @pytest.mark.parametrize(
        ("kind", "field", "value"),
        [
            ("tensor_grid", "element_type", "P2"),
            ("tensor_grid", "initial_refinements", 5),
            ("skfem_tri", "initial_side", 8),
        ],
    )
    def test_rejects_field_scoped_to_the_other_kind(
        self, kind: str, field: str, value: object
    ) -> None:
        with pytest.raises(ValidationError, match="ignores"):
            SubstrateConfig(name="test", kind=kind, **{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kind", "field", "value"),
        [
            ("skfem_tri", "element_type", "P2"),
            ("skfem_tri", "initial_refinements", 5),
            ("tensor_grid", "initial_side", 8),
        ],
    )
    def test_accepts_field_scoped_to_its_own_kind(
        self, kind: str, field: str, value: object
    ) -> None:
        config = SubstrateConfig(name="test", kind=kind, **{field: value})  # type: ignore[arg-type]
        assert getattr(config, field) == value

    @pytest.mark.parametrize(
        ("kind", "field"),
        [
            ("tensor_grid", "element_type"),
            ("tensor_grid", "initial_refinements"),
            ("skfem_tri", "initial_side"),
        ],
    )
    def test_explicitly_setting_the_default_is_allowed(self, kind: str, field: str) -> None:
        """Only a value set *away from* its default is rejected.

        Restating the default is harmless and, more importantly, means no
        existing construction breaks just because it was explicit.
        """
        default = SubstrateConfig.model_fields[field].default
        config = SubstrateConfig(name="test", kind=kind, **{field: default})  # type: ignore[arg-type]
        assert getattr(config, field) == default

    @pytest.mark.parametrize(
        "field",
        ["marking_variant", "error_metric", "enforce_immutable_meshes", "solve_cache_max_entries"],
    )
    @pytest.mark.parametrize("kind", ["tensor_grid", "skfem_tri"])
    def test_shared_fields_are_accepted_by_both_kinds(self, kind: str, field: str) -> None:
        """Guards the validator's *scope*, not just its trigger.

        A rule that rejected everything would pass the two tests above and be
        useless. These four fields are read by both substrates, so neither kind
        may reject them — which is what makes ``_KIND_SCOPED_FIELDS`` a scope
        rather than a blanket.
        """
        non_default = {
            "marking_variant": "linear",
            "error_metric": "nodal_rms",
            "enforce_immutable_meshes": False,
            "solve_cache_max_entries": 7,
        }[field]
        config = SubstrateConfig(name="test", kind=kind, **{field: non_default})  # type: ignore[arg-type]
        assert getattr(config, field) == non_default


def test_named_constants_match_spec() -> None:
    """Pins the two constants against ``specs/refinement_substrate.spec.md``'s table.

    Be clear about what this is and is not. It guards **doc/code drift** — the
    spec names these values, so silently changing one here should fail. It is
    *not* a behavioural test, and on its own it would be close to a tautology
    (asserting a constant defined as ``1e-15`` equals ``1e-15``). The
    behavioural coverage lives in ``tests/research/test_substrates_sweep.py``,
    which drives both through their real consumers in
    ``src/research/substrates/sweep.py``: ``RATE_FIT_MIN_POINTS`` as the
    fit-refusal boundary and ``RATIO_FLOOR`` as the log-interpolation guard.

    There used to be a third, ``AREA_FLOOR``, whose "real consumer" was
    ``warn_on_degenerate_units`` -- a function with zero production callers,
    so the constant's entire consumption chain terminated in four tests written
    for it. Both were deleted on 2026-09-02; a dead-code audit, not this test,
    found it. A value-pinning test cannot tell a live constant from a dead one.
    """
    assert RATIO_FLOOR == 1e-15
    assert RATE_FIT_MIN_POINTS == 3


class TestSelectPrimaryL2:
    """The one shared metric-selection helper, incl. the branch a Literal makes unreachable."""

    def test_quadrature_selects_quadrature(self) -> None:
        cfg = SubstrateConfig(name="q", kind="tensor_grid", error_metric="quadrature")
        assert select_primary_l2(cfg, quadrature=1.0, nodal=2.0) == 1.0

    def test_nodal_rms_selects_nodal(self) -> None:
        """``ERROR_METRIC_NODAL_RMS`` finally has a reader; this is it."""
        cfg = SubstrateConfig(name="n", kind="tensor_grid", error_metric="nodal_rms")
        assert select_primary_l2(cfg, quadrature=1.0, nodal=2.0) == 2.0

    def test_unknown_metric_raises_instead_of_falling_through(self) -> None:
        """Bypasses validation on purpose.

        The ``Literal`` makes this unreachable through ``SubstrateConfig(...)``,
        and the old ternary would have silently returned the nodal value here.
        """
        cfg = SubstrateConfig.model_construct(name="x", kind="tensor_grid", error_metric="bogus")
        with pytest.raises(ValueError, match="unknown error_metric"):
            select_primary_l2(cfg, quadrature=1.0, nodal=2.0)


class TestAdequacyGateSweepBudget:
    """``AdequacyGateConfig.max_sweep_dof`` -- Copilot review, PR #144.

    The budget is *derived* from the top of the window the convergence rates are
    fitted over, and ``run_refinement_sweep`` halts once ``n_dof >= max_dof``. So
    the rounding direction is not cosmetic: the budget has to **span** the
    window. This was ``int(...)`` (truncating) until the review flagged it.
    """

    def test_rounds_up_so_the_budget_spans_the_fit_window(self) -> None:
        """A fractional upper bound must not shrink the sweep.

        Truncation stops the sweep just short of covering the window, leaving
        the fit with fewer points at the top -- a shorter lever arm, or
        ``InsufficientSweepPointsError``. Overshooting costs less than one DOF.
        """
        gate = AdequacyGateConfig(name="t", rate_fit_dof_range=(200.0, 4000.9))
        assert gate.max_sweep_dof == 4001

    def test_a_hair_under_an_integer_still_covers_it(self) -> None:
        """``3999.999`` must not become ``3999``."""
        gate = AdequacyGateConfig(name="t", rate_fit_dof_range=(200.0, 3999.999))
        assert gate.max_sweep_dof == 4000

    def test_an_integral_bound_is_unchanged(self) -> None:
        """Value identity with the module constant this replaced.

        The promotion out of the gate's test file was a move, not a retune, and
        the pinned default is an integral ``4000``; ``ceil`` leaves it exactly
        there. Without this, the fix above could silently shift the shipped gate.
        """
        assert AdequacyGateConfig(name="t").max_sweep_dof == 4000
        gate = AdequacyGateConfig(name="t", rate_fit_dof_range=(200.0, 4000.0))
        assert gate.max_sweep_dof == 4000

    def test_the_budget_is_an_int_for_the_sweep_signature(self) -> None:
        """``run_refinement_sweep(max_dof: int)`` types it as a count."""
        assert isinstance(AdequacyGateConfig(name="t").max_sweep_dof, int)


class TestTheAdequacyGateRejectsNonFiniteMeasurements:
    """A NaN measurement must not read as a passing substrate.

    ``gate_violations`` returns ``[]`` for "adequate". Two of its three clauses
    are bare ``>`` / ``>=`` comparisons, and **every** comparison against a NaN
    is False -- so a NaN rate produced no violation and the gate returned its
    pass verdict for a measurement that says nothing. Only the band clause
    caught it, by accident of being a chained comparison, which is precisely why
    this was invisible: one of three clauses handled the case and two did not.

    Reachable, not theoretical: ``fit_log_log_rate`` calls ``np.polyfit`` with no
    rank check, and ``RATE_FIT_MIN_POINTS`` guarantees three *points*, not three
    distinct x-values -- a degenerate fit emits a RankWarning and returns NaN.

    This is the gate the substrate spec calls "the gate that makes any
    comparison meaningful", so a false pass here silently validates an
    unmeasurable substrate.
    """

    @staticmethod
    def _separation(**overrides: float) -> RateSeparation:
        """A healthy measurement, with named fields overridable per test."""
        base: dict[str, float] = {
            "adaptive_rate": HEALTHY_ADAPTIVE_RATE,
            "uniform_rate": HEALTHY_UNIFORM_RATE,
            "error_ratio_at_matched_dof": HEALTHY_ERROR_RATIO,
        }
        base.update(overrides)
        return RateSeparation(
            adaptive_rate=base["adaptive_rate"],
            uniform_rate=base["uniform_rate"],
            error_ratio_at_matched_dof=base["error_ratio_at_matched_dof"],
            matched_dof=HEALTHY_MATCHED_DOF,
            n_adaptive_points=RATE_FIT_MIN_POINTS,
            n_uniform_points=RATE_FIT_MIN_POINTS,
        )

    def test_the_healthy_measurement_passes(self) -> None:
        """The conditional half: tightening must not reject a real pass.

        Without this, a gate that rejected *everything* would satisfy every
        assertion below.
        """
        assert gate_violations(self._separation()) == []

    @pytest.mark.parametrize(
        "field_name",
        ["adaptive_rate", "uniform_rate", "error_ratio_at_matched_dof"],
    )
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_field_is_a_violation(self, field_name: str, bad: float) -> None:
        """Each of the three fields, each of the three non-finite values."""
        violations = gate_violations(self._separation(**{field_name: bad}))
        assert violations, f"{field_name}={bad!r} produced the gate's PASS verdict"
        assert any(field_name in v for v in violations), (
            f"the violation must name {field_name}, not merely exist: got {violations}"
        )

    def test_the_non_finite_verdict_does_not_dilute_itself(self) -> None:
        """Only the non-finite diagnosis is reported, not threshold noise.

        Comparisons against a NaN silently pass, so continuing past the
        non-finite check would append meaningless numbers to a real diagnosis.
        """
        violations = gate_violations(self._separation(adaptive_rate=float("nan")))
        assert all("is nan" in v or "not a finite" in v for v in violations), violations


class TestTheGateConfigRejectsUnusableThresholds:
    """A gate whose own thresholds are non-finite cannot fire.

    ``_low_below_high`` checked ordering only, and ``nan >= x`` is False -- so a
    NaN bound passed the ordering check unnoticed. Same arithmetic that let a
    NaN *measurement* read as a passing substrate; this is the configuration
    side of it (Copilot review, PR #144).

    Measured on the pre-fix version, all of these constructed cleanly and failed
    late somewhere else:

    ============================  =========================================
    config                        consequence
    ============================  =========================================
    ``dof_range=(200, nan)``      ``max_sweep_dof`` raises ``ValueError``
    ``dof_range=(200, inf)``      ``max_sweep_dof`` raises ``OverflowError``
    ``dof_range=(-50, -10)``      ``max_sweep_dof = -10`` (negative budget)
    ``band=(nan, -0.55)``         band clause silently stops discriminating
    ``adaptive_rate_min=nan``     that clause can never fire
    ============================  =========================================
    """

    @staticmethod
    def _build(**overrides: object) -> AdequacyGateConfig:
        return AdequacyGateConfig(name="probe", description="probe", **overrides)  # type: ignore[arg-type]

    def test_the_shipped_defaults_still_validate(self) -> None:
        """The conditional half: validation that rejects everything is not validation."""
        gate = self._build()
        assert gate.max_sweep_dof == DEFAULT_MAX_SWEEP_DOF

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("position", [0, 1])
    def test_a_non_finite_interval_bound_is_rejected(self, bad: float, position: int) -> None:
        """Both fields, both positions, all three non-finite values."""
        for field, default in (
            ("uniform_rate_band", (-0.85, -0.55)),
            ("rate_fit_dof_range", (200.0, 4000.0)),
        ):
            bounds = list(default)
            bounds[position] = bad
            with pytest.raises(ValidationError, match="finite"):
                self._build(**{field: tuple(bounds)})

    @pytest.mark.parametrize("window", [(-50.0, -10.0), (0.0, 4000.0), (0.0, 0.5)])
    def test_a_non_positive_dof_window_is_rejected(self, window: tuple[float, float]) -> None:
        """A DOF count is a mesh size; zero degrees of freedom is not a mesh."""
        with pytest.raises(ValidationError, match=str(int(MIN_RATE_FIT_DOF))):
            self._build(rate_fit_dof_range=window)

    def test_a_negative_rate_band_is_still_accepted(self) -> None:
        """The positivity rule must not leak onto the rate fields.

        A converging method has a *negative* log-log slope, so a negative
        ``uniform_rate_band`` is correct and must survive. Folding the two checks
        into one validator would reject it -- which is why they are separate.
        """
        gate = self._build(uniform_rate_band=(-2.0, -0.1), adaptive_rate_min=-1.5)
        assert gate.uniform_rate_band == (-2.0, -0.1)
        assert gate.adaptive_rate_min == -1.5

    @pytest.mark.parametrize("field", ["adaptive_rate_min", "adaptive_vs_uniform_max_ratio"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_non_finite_scalar_threshold_is_rejected(self, field: str, bad: float) -> None:
        """A NaN threshold makes its clause False for every input, forever.

        Rejection is asserted, not the *message*: the two fields are caught by
        different layers. ``adaptive_vs_uniform_max_ratio`` carries ``gt=0``, and
        pydantic's own bound rejects a NaN before the finiteness validator runs;
        ``adaptive_rate_min`` is unbounded (a convergence rate is legitimately
        negative), so only the validator stands between it and a clause that can
        never fire. Asserting one shared message would either be false for one
        field or force a bound onto a field that must not have one.
        """
        with pytest.raises(ValidationError):
            self._build(**{field: bad})

    def test_the_unbounded_rate_threshold_is_caught_by_the_finiteness_validator(self) -> None:
        """Pin which layer protects ``adaptive_rate_min`` specifically.

        Without this, the parametrised test above would still pass if the
        finiteness validator were deleted and someone added a bound to the ratio
        field -- the field with no bound would silently lose its only guard.
        """
        with pytest.raises(ValidationError, match="finite"):
            self._build(adaptive_rate_min=float("nan"))

    def test_an_inverted_interval_is_still_rejected(self) -> None:
        """The original ordering check must survive the finiteness addition."""
        with pytest.raises(ValidationError, match="low must be < high"):
            self._build(uniform_rate_band=(-0.55, -0.85))
