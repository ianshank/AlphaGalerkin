"""Tests for ``SkfemTriSubstrate``, the element-local ``RefinementSubstrate``.

Covers the acceptance criteria this substrate exists to satisfy: AC2 (local,
conforming refinement -- element count grows by O(|M|), not O(N)), AC3 (mesh
immutability enforcement + opt-out), AC6 (quadrature L2 as the primary
metric, nodal RMS additive in ``extra``), AC8 (reentrant corner at the
origin), plus the Protocol contract and registry round-trip.

scikit-fem is required. ``fem_required`` marks every test in this module; the
root conftest.py hook skips them *visibly* (reporting a skip count) when
scikit-fem is not installed, and hard-fails collection instead when
``ALPHAGALERKIN_REQUIRE_EXTRAS=1`` (the test-extras CI job).
"""

from __future__ import annotations

import numpy as np
import pytest
from structlog.testing import capture_logs

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import LShapedPoissonOperator
from src.refinement.substrate import RefinementSubstrate
from src.refinement.substrate_registry import (
    RefinementSubstrateRegistry,
    register_refinement_substrate,
)
from src.research.substrates import skfem_tri as skfem_tri_module
from src.research.substrates.config import SUBSTRATE_PRIMARY_L2_KEY, SubstrateConfig
from src.research.substrates.skfem_tri import SkfemTriMesh, SkfemTriSubstrate

pytestmark = pytest.mark.fem_required


def _lshaped_operator() -> LShapedPoissonOperator:
    return LShapedPoissonOperator(
        PDEConfig(
            name="poisson_lshaped",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-1.0, -1.0],
            domain_max=[1.0, 1.0],
        )
    )


@pytest.fixture
def operator() -> LShapedPoissonOperator:
    return _lshaped_operator()


@pytest.fixture
def substrate(operator: LShapedPoissonOperator) -> SkfemTriSubstrate:
    return SkfemTriSubstrate(
        operator,
        config=SubstrateConfig(
            name="skfem_tri_test",
            kind="skfem_tri",
            initial_refinements=2,
            marking_variant="squared",
            error_metric="quadrature",
        ),
    )


class TestSkfemTriSubstrateProtocol:
    def test_satisfies_refinement_substrate(self, substrate: SkfemTriSubstrate) -> None:
        assert isinstance(substrate, RefinementSubstrate)


class TestSkfemTriSubstrateAC8ReentrantCorner:
    def test_origin_is_a_mesh_node(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh()
        pts = mesh.mesh.p.T
        assert np.any(np.all(np.isclose(pts, [0.0, 0.0], atol=1e-12), axis=1))


class TestSkfemTriSubstrateAC2LocalRefinement:
    def test_marking_one_element_grows_by_far_less_than_uniform(
        self, substrate: SkfemTriSubstrate
    ) -> None:
        mesh = substrate.initial_mesh()
        n0 = substrate.n_units(mesh)

        marked = np.zeros(n0, dtype=bool)
        marked[0] = True
        local = substrate.refine(mesh, marked)
        n_local = substrate.n_units(local)

        uniform_marked = np.ones(n0, dtype=bool)
        uniform = substrate.refine(mesh, uniform_marked)
        n_uniform = substrate.n_units(uniform)

        assert n0 < n_local < n_uniform
        # Local growth from a single marked element must be a small, bounded
        # constant, not proportional to the whole mesh.
        assert (n_local - n0) < 0.1 * (n_uniform - n0)

    def test_refine_does_not_mutate_input_mesh(self, substrate: SkfemTriSubstrate) -> None:
        """AC3 as written: the coordinate and connectivity **bytes** are unchanged.

        Previously this asserted only that ``n_units`` was unchanged, which a
        mesh whose vertices had been moved in place would also satisfy. AC3
        says "bytes", so compare bytes.
        """
        mesh = substrate.initial_mesh()
        n0 = substrate.n_units(mesh)
        p_before = mesh.mesh.p.tobytes()
        t_before = mesh.mesh.t.tobytes()

        marked = np.ones(substrate.n_units(mesh), dtype=bool)
        refined = substrate.refine(mesh, marked)

        assert substrate.n_units(mesh) == n0
        assert mesh.mesh.p.tobytes() == p_before
        assert mesh.mesh.t.tobytes() == t_before
        assert refined.mesh is not mesh.mesh


class TestSkfemTriSubstrateAC3Immutability:
    def test_enforced_by_default(self, operator: LShapedPoissonOperator) -> None:
        substrate = SkfemTriSubstrate(operator, config=SubstrateConfig(name="t", kind="skfem_tri"))
        mesh = substrate.initial_mesh()
        assert mesh.mesh.p.flags.writeable is False
        assert mesh.mesh.t.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            mesh.mesh.p[0, 0] = 999.0

    def test_opt_out(self, operator: LShapedPoissonOperator) -> None:
        substrate = SkfemTriSubstrate(
            operator,
            config=SubstrateConfig(name="t", kind="skfem_tri", enforce_immutable_meshes=False),
        )
        mesh = substrate.initial_mesh()
        assert mesh.mesh.p.flags.writeable is True

    def test_refined_mesh_is_also_frozen(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh()
        marked = np.ones(substrate.n_units(mesh), dtype=bool)
        refined = substrate.refine(mesh, marked)
        assert refined.mesh.p.flags.writeable is False


class TestSkfemTriSubstrateSolve:
    def test_solve_returns_sane_result(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh()
        result = substrate.solve(mesh)
        assert result.n_dof > 0
        assert 0 <= result.n_dof_free <= result.n_dof
        assert result.l2_error >= 0.0
        assert result.indicators.shape == (substrate.n_units(mesh),)
        assert np.all(result.indicators >= 0.0)

    def test_quadrature_is_primary_metric_by_default(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh()
        result = substrate.solve(mesh)
        assert result.l2_error == pytest.approx(result.extra["l2_error_quadrature"])
        assert "l2_error_nodal_rms" in result.extra

    def test_nodal_rms_selectable_as_primary_metric(self, operator: LShapedPoissonOperator) -> None:
        substrate = SkfemTriSubstrate(
            operator,
            config=SubstrateConfig(name="t", kind="skfem_tri", error_metric="nodal_rms"),
        )
        mesh = substrate.initial_mesh()
        result = substrate.solve(mesh)
        assert result.l2_error == pytest.approx(result.extra["l2_error_nodal_rms"])

    def test_quadrature_and_nodal_rms_differ(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh()
        result = substrate.solve(mesh)
        assert result.extra["l2_error_quadrature"] != pytest.approx(
            result.extra["l2_error_nodal_rms"]
        )

    def test_error_decreases_with_refinement(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh()
        first = substrate.solve(mesh)
        marked = np.ones(substrate.n_units(mesh), dtype=bool)
        mesh = substrate.refine(mesh, marked)
        second = substrate.solve(mesh)
        assert second.l2_error < first.l2_error


class TestSkfemTriSubstrateMarkAndDescribe:
    def test_mark_uses_configured_variant(self, substrate: SkfemTriSubstrate) -> None:
        indicators = np.zeros(8)
        marked = substrate.mark(indicators, theta=0.3)
        # variant="squared" marks exactly one element on an all-zero array (AC4).
        assert marked.sum() == 1

    def test_mark_linear_variant_marks_nothing_on_zeros(
        self, operator: LShapedPoissonOperator
    ) -> None:
        substrate = SkfemTriSubstrate(
            operator,
            config=SubstrateConfig(name="t", kind="skfem_tri", marking_variant="linear"),
        )
        marked = substrate.mark(np.zeros(8), theta=0.3)
        assert not marked.any()

    def test_refinable_mask_is_all_true(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh()
        mask = substrate.refinable_mask(mesh)
        assert mask.shape == (substrate.n_units(mesh),)
        assert mask.all()

    def test_fingerprint_changes_after_refine(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh()
        marked = np.ones(substrate.n_units(mesh), dtype=bool)
        refined = substrate.refine(mesh, marked)
        assert substrate.fingerprint(mesh) != substrate.fingerprint(refined)

    def test_fingerprint_deterministic(self, substrate: SkfemTriSubstrate) -> None:
        assert substrate.fingerprint(substrate.initial_mesh()) == substrate.fingerprint(
            substrate.initial_mesh()
        )

    def test_describe(self, substrate: SkfemTriSubstrate) -> None:
        info = substrate.describe()
        assert info["kind"] == "skfem_tri"
        assert info["element_type"] == "P1"


class TestSkfemTriMesh:
    def test_is_frozen(self) -> None:
        import dataclasses

        mesh = SkfemTriMesh(mesh=object())
        assert dataclasses.is_dataclass(mesh)
        with pytest.raises(dataclasses.FrozenInstanceError):
            mesh.mesh = object()  # type: ignore[misc]


class TestSkfemTriSubstrateRegistry:
    def setup_method(self) -> None:
        RefinementSubstrateRegistry().clear()

    def teardown_method(self) -> None:
        RefinementSubstrateRegistry().clear()

    def test_register_and_retrieve(self) -> None:
        register_refinement_substrate("skfem_tri")(SkfemTriSubstrate)
        cls = RefinementSubstrateRegistry().get_or_raise("skfem_tri")
        assert cls is SkfemTriSubstrate


class TestSkfemTriSubstrateAC2Conformity:
    """AC2's second clause, which had no test despite the module docstring claiming it.

    "Zero edges shared by more than two elements, after one local refinement
    and after four successive ones." Conformity is the entire justification for
    choosing skfem's RGB refinement over a quadtree backend (the spec's Out of
    Scope names exactly this), and it is the property most likely to break
    under a future scikit-fem major -- which is why ``pyproject.toml`` caps the
    dependency at ``<13``. Asserting it here means a version bump that
    introduces hanging nodes fails loudly instead of silently invalidating
    every error estimate downstream.
    """

    @staticmethod
    def _max_facet_incidence(mesh: object) -> int:
        """Largest number of elements sharing any one facet (edge). Conforming == 2."""
        t2f = np.asarray(mesh.t2f)  # type: ignore[attr-defined]
        counts = np.bincount(t2f.ravel())
        return int(counts.max()) if counts.size else 0

    def test_initial_mesh_is_conforming(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh()
        assert self._max_facet_incidence(mesh.mesh) <= 2

    def test_one_local_refinement_stays_conforming(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh()
        marked = np.zeros(substrate.n_units(mesh), dtype=bool)
        marked[0] = True
        refined = substrate.refine(mesh, marked)
        assert self._max_facet_incidence(refined.mesh) <= 2

    def test_four_successive_local_refinements_stay_conforming(
        self, substrate: SkfemTriSubstrate
    ) -> None:
        """The case a naive quadtree fails: repeated refinement of one region."""
        mesh = substrate.initial_mesh()
        for _ in range(4):
            result = substrate.solve(mesh)
            marked = substrate.mark(result.indicators, theta=0.3)
            mesh = substrate.refine(mesh, marked)
            assert self._max_facet_incidence(mesh.mesh) <= 2, (
                "a hanging node appeared -- RGB refinement is no longer conforming, "
                "which invalidates every error estimate computed on this mesh"
            )

    def test_the_incidence_helper_can_actually_detect_a_hanging_node(self) -> None:
        """Guards the guard: a helper that always returns <= 2 tests nothing.

        Feeds it a synthetic ``t2f`` where one facet is shared by three
        elements -- the exact condition the three tests above rule out -- and
        requires it to report 3.
        """

        class _FakeMesh:
            t2f = np.array([[0, 0, 0], [1, 2, 3]])

        assert TestSkfemTriSubstrateAC2Conformity._max_facet_incidence(_FakeMesh()) == 3


class TestSkfemTriSubstrateRequiresAnExactSolution:
    """A no-exact-solution operator must fail at construction, named.

    Before this, ``SkfemTriSubstrate`` accepted such an operator and crashed
    later with a ``TypeError`` from ``np.asarray(None, dtype=np.float64)``
    several frames inside a quadrature form -- the substrate had silently
    dropped ``BaseSolver._compute_l2_error``'s ``if exact is None`` guard while
    its docstring claimed the formula was "reproduced verbatim".
    """

    def test_construction_raises_with_an_actionable_message(self) -> None:
        operator = _lshaped_operator()
        object.__setattr__(operator, "exact_solution", lambda pts: None)
        with pytest.raises(ValueError, match="analytic exact solution"):
            SkfemTriSubstrate(operator)

    def test_a_real_operator_still_constructs(self) -> None:
        assert SkfemTriSubstrate(_lshaped_operator()) is not None


class TestSkfemTriMirrorsTheTensorGridContract:
    """The same D2/D3/D4 fixes, asserted on the other implementation.

    Two implementations of one Protocol that are only tested on one side are
    two implementations that will diverge -- which is exactly what happened to
    the ``extra`` key sets and the zero-marked refine warning.
    """

    @pytest.fixture
    def operator(self) -> LShapedPoissonOperator:
        return LShapedPoissonOperator(
            PDEConfig(
                name="lshaped_mirror",
                pde_type=PDEType.POISSON,
                domain_dim=2,
                domain_min=[-1.0, -1.0],
                domain_max=[1.0, 1.0],
            )
        )

    def test_a_mismatched_kind_is_rejected_at_construction(
        self, operator: LShapedPoissonOperator
    ) -> None:
        with pytest.raises(ValueError, match="kind"):
            SkfemTriSubstrate(operator, config=SubstrateConfig(name="mismatch", kind="tensor_grid"))

    def test_describe_derives_the_kind_rather_than_restating_it(
        self, operator: LShapedPoissonOperator
    ) -> None:
        """See the tensor-grid twin for why this rebinds ``_config``."""
        substrate = SkfemTriSubstrate(operator)
        substrate._config = SubstrateConfig(name="rebound", kind="tensor_grid")
        assert substrate.describe()["kind"] == "tensor_grid"

    @pytest.mark.parametrize("metric", ["quadrature", "nodal_rms"])
    def test_primary_key_present_and_equals_l2_error(
        self, operator: LShapedPoissonOperator, metric: str
    ) -> None:
        substrate = SkfemTriSubstrate(
            operator,
            config=SubstrateConfig(name="primary", kind="skfem_tri", error_metric=metric),
        )
        result = substrate.solve(substrate.initial_mesh())
        assert SUBSTRATE_PRIMARY_L2_KEY in result.extra
        assert result.extra[SUBSTRATE_PRIMARY_L2_KEY] == result.l2_error

    def test_solve_raises_when_the_nodal_rms_is_unmeasurable(
        self, operator: LShapedPoissonOperator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        substrate = SkfemTriSubstrate(operator)
        mesh = substrate.initial_mesh()
        monkeypatch.setattr(skfem_tri_module, "nodal_rms_l2_error", lambda *a, **k: None)
        with pytest.raises(ValueError, match="unmeasurable"):
            substrate.solve(mesh)


class TestSkfemZeroMarkedRefineWarns:
    """The pre-existing twin of the tensor-grid guard -- previously untested."""

    def test_empty_selection_emits_a_warning(self) -> None:
        operator = LShapedPoissonOperator(
            PDEConfig(
                name="lshaped_noop",
                pde_type=PDEType.POISSON,
                domain_dim=2,
                domain_min=[-1.0, -1.0],
                domain_max=[1.0, 1.0],
            )
        )
        substrate = SkfemTriSubstrate(operator)
        mesh = substrate.initial_mesh()
        empty = np.zeros(substrate.n_units(mesh), dtype=bool)
        with capture_logs() as events:
            substrate.refine(mesh, empty)
        assert any(e["event"] == "substrate_refine_noop" for e in events), events
