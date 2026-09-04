"""AC7 — the adequacy gate: can this substrate show a marking win at all.

The charter records that the project's central claim (MCTS look-ahead beats
classical AMR) *cannot currently be measured*: on the only substrate that
existed, adaptive Dörfler marking converged **worse** than plain uniform
refinement, because refining one element inserts full tensor-product grid
lines. Any arena built on that substrate measures the substrate, not the
policy.

This module is the tripwire that keeps that from happening again. It asserts
a textbook AFEM result — adaptive beats uniform on the L-shaped Poisson
problem's reentrant-corner singularity — and, crucially, asserts the **same
measurement fails** on ``TensorGridSubstrate``. A gate that passes on both
substrates is not a gate; it is a formality. Both halves are required.

Scope note (from ``specs/refinement_substrate.spec.md`` AC7): this reproduces
a known result. It is an *implementation-correctness* gate, **not** a research
finding, and must never be quoted as one.

The skfem half carries ``fem_required`` (skipped visibly without the ``[fem]``
extra, a hard collection error under ``ALPHAGALERKIN_REQUIRE_EXTRAS=1``). The
tensor-grid half deliberately does **not** — the discriminating half of the
gate runs on every CPU CI job, with no optional dependency.
"""

from __future__ import annotations

import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import LShapedPoissonOperator
from src.research.lshape_amr_compare import ComparisonParams, lshape_inside_predicate
from src.research.substrates.config import SubstrateConfig
from src.research.substrates.sweep import (
    RateSeparation,
    default_adequacy_gate,
    gate_violations,
    measure_adequacy,
)
from src.research.substrates.tensor_grid import TensorGridSubstrate

# --------------------------------------------------------------------------
# Gate constants and predicate -- now RE-EXPORTS, not definitions.
#
# These four thresholds and `gate_violations` used to be defined here, which
# made the verdict the spec calls "the gate that makes any comparison
# meaningful" reachable only from pytest: a caller who ran a sweep through the
# public API had no way to ask whether the substrate was adequate. They now live
# in `src/research/substrates/` (AdequacyGateConfig + gate_violations +
# measure_adequacy) with byte-identical values -- a move, not a retune; the
# per-field provenance comments moved with them.
#
# The names are kept here so every assertion below still reads against a named
# threshold rather than a literal, and so `TestGatePredicate` exercises the
# PROMOTED function rather than a copy that could drift from it.
# --------------------------------------------------------------------------

#: The pinned gate, built once. Every constant below is read off it.
ADEQUACY_GATE = default_adequacy_gate()

UNIFORM_RATE_BAND = ADEQUACY_GATE.uniform_rate_band
ADAPTIVE_RATE_MIN = ADEQUACY_GATE.adaptive_rate_min
ADAPTIVE_VS_UNIFORM_MAX_RATIO = ADEQUACY_GATE.adaptive_vs_uniform_max_ratio
RATE_FIT_DOF_RANGE = ADEQUACY_GATE.rate_fit_dof_range
MAX_SWEEP_DOF = ADEQUACY_GATE.max_sweep_dof
MAX_LEVELS_UNIFORM = ADEQUACY_GATE.max_levels_uniform
MAX_LEVELS_ADAPTIVE = ADEQUACY_GATE.max_levels_adaptive


def measure(substrate: object, theta: float) -> RateSeparation:
    """Run both arms on ``substrate`` and fit their rates over the pinned window.

    Thin adapter onto the promoted ``measure_adequacy``; kept so the existing
    call sites read unchanged and so the ``object`` -> Protocol cast stays in
    one place.
    """
    return measure_adequacy(substrate, theta=theta, gate=ADEQUACY_GATE)  # type: ignore[arg-type]


def _lshaped_operator(scale: float) -> LShapedPoissonOperator:
    return LShapedPoissonOperator(
        PDEConfig(
            name="poisson_lshaped",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-scale, -scale],
            domain_max=[scale, scale],
        )
    )


def _separation(
    *,
    adaptive_rate: float,
    uniform_rate: float,
    ratio: float,
) -> RateSeparation:
    """Synthetic measurement, for testing ``gate_violations`` as a predicate."""
    return RateSeparation(
        adaptive_rate=adaptive_rate,
        uniform_rate=uniform_rate,
        error_ratio_at_matched_dof=ratio,
        matched_dof=1000.0,
        n_adaptive_points=5,
        n_uniform_points=3,
    )


class TestGatePredicate:
    """Unit-test ``gate_violations`` itself, on synthetic measurements.

    Exists because a mutation check found a real hole: widening
    ``UNIFORM_RATE_BAND`` to ``(-5.0, 0.0)`` left every solve-driven test in
    this module green. Both substrates' uniform arms sit comfortably inside
    the band, so *loosening* it removes a constraint nothing was exercising --
    the AC8 tripwire ("too good is a defect") was documented but not actually
    asserted anywhere. These cases pin the band's discriminating power without
    running a single PDE solve.
    """

    def test_impossibly_good_uniform_rate_is_rejected(self) -> None:
        """AC8's actual failure mode: a translated mesh moved the singularity out.

        The spike hit exactly this and produced a confident, entirely wrong
        "adaptive loses on skfem too" result -- both arms at ``N^-1.05``. A
        singular problem cannot give uniform P1 the smooth-problem rate, so a
        band that accepts -1.05 would have accepted that bug.
        """
        violations = gate_violations(
            _separation(adaptive_rate=-1.30, uniform_rate=-1.05, ratio=0.5)
        )
        assert any("uniform_rate" in v for v in violations)

    def test_stalled_uniform_rate_is_rejected(self) -> None:
        """The other side of the band: an arm that barely converges is also a defect."""
        violations = gate_violations(
            _separation(adaptive_rate=-1.30, uniform_rate=-0.20, ratio=0.5)
        )
        assert any("uniform_rate" in v for v in violations)

    def test_shallow_adaptive_rate_is_rejected(self) -> None:
        violations = gate_violations(
            _separation(adaptive_rate=-0.90, uniform_rate=-0.70, ratio=0.5)
        )
        assert any("adaptive_rate" in v for v in violations)

    def test_ratio_at_exactly_the_ceiling_is_rejected(self) -> None:
        """Boundary: "beats uniform" must be strict -- a tie is not a win."""
        violations = gate_violations(
            _separation(
                adaptive_rate=-1.30,
                uniform_rate=-0.70,
                ratio=ADAPTIVE_VS_UNIFORM_MAX_RATIO,
            )
        )
        assert any("error_ratio_at_matched_dof" in v for v in violations)

    def test_a_healthy_measurement_is_accepted(self) -> None:
        """Guards the opposite failure: a predicate that rejects everything."""
        assert (
            gate_violations(_separation(adaptive_rate=-1.25, uniform_rate=-0.67, ratio=0.25)) == []
        )


@pytest.fixture(scope="module")
def params() -> ComparisonParams:
    return ComparisonParams()


@pytest.fixture(scope="module")
def skfem_separation(params: ComparisonParams) -> RateSeparation:
    """Measured once per module: the sweep is ~1.5 s and every assertion reads it."""
    from src.research.substrates.skfem_tri import SkfemTriSubstrate

    substrate = SkfemTriSubstrate(
        _lshaped_operator(params.scale),
        config=SubstrateConfig(name="skfem_gate", kind="skfem_tri"),
    )
    return measure(substrate, params.marking_fraction)


@pytest.fixture(scope="module")
def tensor_grid_separation(params: ComparisonParams) -> RateSeparation:
    substrate = TensorGridSubstrate(
        _lshaped_operator(params.scale),
        inside=lshape_inside_predicate(params.scale),
        config=SubstrateConfig(
            name="tensor_grid_gate",
            kind="tensor_grid",
            initial_side=params.initial_side,
        ),
    )
    return measure(substrate, params.marking_fraction)


@pytest.mark.fem_required
class TestAdequacyGatePassesOnElementLocalSubstrate:
    """AC7, positive half: the element-local substrate can show a marking win."""

    @pytest.fixture(autouse=True)
    def separation(self, skfem_separation: RateSeparation) -> RateSeparation:
        return skfem_separation

    def test_gate_passes(self, separation: RateSeparation) -> None:
        assert gate_violations(separation) == [], (
            f"the element-local substrate must satisfy the adequacy gate; measured {separation}"
        )

    def test_uniform_rate_is_in_band(self, separation: RateSeparation) -> None:
        """Named separately so a band failure is legible without reading the list."""
        low, high = UNIFORM_RATE_BAND
        assert low <= separation.uniform_rate <= high

    def test_adaptive_rate_is_steep_enough(self, separation: RateSeparation) -> None:
        assert separation.adaptive_rate <= ADAPTIVE_RATE_MIN

    def test_adaptive_beats_uniform_at_matched_dof(self, separation: RateSeparation) -> None:
        assert separation.error_ratio_at_matched_dof < ADAPTIVE_VS_UNIFORM_MAX_RATIO

    def test_both_fits_used_enough_points(self, separation: RateSeparation) -> None:
        """A slope through two points always 'succeeds' and always means nothing."""
        from src.research.substrates.config import RATE_FIT_MIN_POINTS

        assert separation.n_adaptive_points >= RATE_FIT_MIN_POINTS
        assert separation.n_uniform_points >= RATE_FIT_MIN_POINTS


class TestAdequacyGateFailsOnTensorGridSubstrate:
    """AC7, discriminating half (tasks 3.3 / 6.2): the gate must reject the control.

    No ``fem_required`` marker: this half must run on every CPU CI job. If the
    gate ever silently stopped discriminating, a skfem-gated-only test would
    hide it on exactly the runs where the substrate work is most likely to
    regress.
    """

    @pytest.fixture(autouse=True)
    def separation(self, tensor_grid_separation: RateSeparation) -> RateSeparation:
        return tensor_grid_separation

    def test_gate_fails(self, separation: RateSeparation) -> None:
        assert gate_violations(separation) != [], (
            "the adequacy gate must REJECT the legacy tensor-product substrate -- "
            "a gate that passes on both substrates measures nothing; "
            f"measured {separation}"
        )

    def test_failure_is_the_adaptive_arm_not_a_broken_substrate(
        self, separation: RateSeparation
    ) -> None:
        """The *reason* matters: it must be marking, not a substrate that cannot converge.

        The tensor grid's **uniform** arm converges perfectly normally (measured
        -0.6489, inside the band). What fails is its **adaptive** arm: refining
        one element inserts whole grid lines, so bulk-chasing buys almost no
        error reduction per DOF. Pinning that distinction is what stops this
        test from passing for the wrong reason -- e.g. a substrate so broken
        that neither arm converges would also make ``gate_violations``
        non-empty, and would be a different (and much worse) defect.
        """
        low, high = UNIFORM_RATE_BAND
        assert low <= separation.uniform_rate <= high, (
            "tensor-grid uniform refinement is expected to converge normally; "
            f"got {separation.uniform_rate:.4f}"
        )
        assert separation.adaptive_rate > ADAPTIVE_RATE_MIN, (
            "the tensor grid's adaptive arm is expected to be too shallow; "
            f"got {separation.adaptive_rate:.4f}"
        )

    def test_adaptive_loses_to_uniform_at_matched_dof(self, separation: RateSeparation) -> None:
        """The inversion the charter records: adaptive is *worse* here."""
        assert separation.error_ratio_at_matched_dof > ADAPTIVE_VS_UNIFORM_MAX_RATIO
