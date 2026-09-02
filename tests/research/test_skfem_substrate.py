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

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import LShapedPoissonOperator
from src.refinement.substrate import RefinementSubstrate
from src.refinement.substrate_registry import (
    RefinementSubstrateRegistry,
    register_refinement_substrate,
)
from src.research.substrates.config import SubstrateConfig
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
        mesh = substrate.initial_mesh()
        n0 = substrate.n_units(mesh)
        marked = np.ones(substrate.n_units(mesh), dtype=bool)
        substrate.refine(mesh, marked)
        assert substrate.n_units(mesh) == n0


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
