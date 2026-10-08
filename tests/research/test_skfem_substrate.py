"""Tests for ``SkfemTriSubstrate``, the element-local ``RefinementSubstrate``.

Covers the acceptance criteria this substrate exists to satisfy: AC2 (local,
conforming refinement -- element count grows by O(|M|), not O(N)), AC3 (mesh
immutability enforcement + opt-out), AC6 (quadrature L2 as the primary
metric, nodal RMS additive in ``extra``), AC8 (reentrant corner at the
origin), plus the Protocol contract and registry round-trip.

Also the polyomino path (the Z-shape second testbed, bottom of the file): the
L-shape mesh stays byte-identical, the generic builder reproduces it, the Z mesh
is conforming and covers exactly the domain, and the Z substrate converges at
the L-shape's singular rate before any policy comparison may use it.

scikit-fem is required. ``fem_required`` marks every test in this module; the
root conftest.py hook skips them *visibly* (reporting a skip count) when
scikit-fem is not installed, and hard-fails collection instead when
``ALPHAGALERKIN_REQUIRE_EXTRAS=1`` (the test-extras CI job).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.geometry_polyomino import (
    LSHAPE_POLYOMINO_CELLS,
    ZSHAPE_POLYOMINO_CELLS,
    PolyominoDomain,
)
from src.pde.operators import LShapedPoissonOperator, MultiCornerPoissonOperator
from src.pde.operators.multi_corner_poisson import (
    DEFAULT_ZSHAPE_PRIMARY_COEFFICIENT,
    DEFAULT_ZSHAPE_SECONDARY_COEFFICIENT,
    build_lshape_multi_corner_operator,
    build_zshape_poisson_operator,
)
from src.refinement.substrate import RefinementSubstrate
from src.refinement.substrate_registry import (
    RefinementSubstrateRegistry,
    register_refinement_substrate,
)
from src.research.fem_baseline import _require_skfem, build_lshaped_initial_mesh
from src.research.lshape_amr_compare import ComparisonParams
from src.research.substrates import skfem_tri as skfem_tri_module
from src.research.substrates.config import SUBSTRATE_PRIMARY_L2_KEY, SubstrateConfig
from src.research.substrates.factory import (
    adequacy_gate_for_operator,
    build_substrate_from_config,
)
from src.research.substrates.skfem_tri import (
    POLYOMINO_NODES_PER_CELL,
    SkfemTriMesh,
    SkfemTriSubstrate,
    build_polyomino_initial_mesh,
)
from src.research.substrates.sweep import SweepPoint, fit_log_log_rate, run_refinement_sweep

pytestmark = pytest.mark.fem_required

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        # Production lookups go through ``ensure_substrate_registrants``; restore
        # so a later test in this process is not left with ``Available: []``.
        from src.research.substrates.factory import ensure_substrate_registrants

        ensure_substrate_registrants()

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


class _RecordingLogger:
    """Double for a cached structlog logger; ``bind`` returns self.

    Same ``cache_logger_on_first_use`` trap as the tensor-grid twin: see
    ``tests/pde/test_mesh_refinement.py::test_degenerate_triangulation_is_logged``.
    """

    def __init__(self) -> None:
        self.events: list[str] = []

    def bind(self, **_kwargs: object) -> _RecordingLogger:
        return self

    def warning(self, event: str, **_kwargs: object) -> None:
        self.events.append(event)

    def info(self, event: str, **_kwargs: object) -> None:
        return None

    def debug(self, event: str, **_kwargs: object) -> None:
        return None


class TestSkfemZeroMarkedRefineWarns:
    """The pre-existing twin of the tensor-grid guard -- previously untested."""

    def test_empty_selection_emits_a_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _RecordingLogger()
        monkeypatch.setattr(skfem_tri_module, "logger", recorder)
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
        substrate.refine(mesh, empty)
        assert "substrate_refine_noop" in recorder.events, recorder.events


# ---------------------------------------------------------------------------
# The polyomino (multi-corner) path -- the second AMR testbed.
#
# These live in THIS file, not a new one, on purpose. The test-extras CI job is
# the only one with scikit-fem, and it selects a fixed list of files (this one,
# test_fem_baseline.py and the substrates coverage-gate list). A fem_required
# test in a new file would be skipped on every lane and gate nothing -- the
# CI-invisibility defect this repo has recorded eight times.
#
# Mutation kills (planted defect -> named test that went red):
#   * polyomino branch removed from SkfemTriSubstrate.initial_mesh (every
#     operator meshed as the L) -> every TestZShapeSubstrate / convergence test
#     (NotImplementedError from build_lshaped_initial_mesh's box check).
#   * builder cells translated by +10 in x (the singularities leave the mesh,
#     AC8's cautionary tale) -> TestZShapeSubstrate::
#     test_every_reentrant_corner_is_a_mesh_node and
#     TestZShapeConvergenceGate::test_uniform_rate_is_in_the_lshape_band.
#   * boundary_value returning zeros (wrong Dirichlet data) ->
#     TestZShapeConvergenceGate::test_error_decreases_monotonically_with_dof
#     (and both rate tests).
#   * the L-shape operator routed through the generic builder with the L cells
#     in another order (a plausible "unify the builders" refactor; the node
#     numbering moves) -> TestLShapePathIsByteIdentical::
#     test_lshape_operator_mesh_is_the_lshape_builders and
#     ::test_arena_mesh_bytes_match_the_committed_artifact.
# ---------------------------------------------------------------------------

#: sha256 of ``SkfemTriSubstrate.fingerprint(initial_mesh())`` for the L-shape
#: operator at the arena's ``initial_refinements``, captured from the code as it
#: stood *before* polyomino support was added (scikit-fem 12.0.2).
LSHAPE_ARENA_MESH_SHA256 = "21e4f2aed02f8e9c621ace0a185a1ef2473c8a4c401e25b47d9f11703a4057f5"

#: The committed arena artifact whose substrate this mesh must remain.
ARENA_RUN_JSON = REPO_ROOT / "results" / "mcts_classical_amr_arena.run.json"

#: Relative tolerance when the L preset and the L operator solve the same mesh:
#: they evaluate the same function in float64 and round to float32 at slightly
#: different points, so the quadrature L2 errors agree to ~float32 precision.
LSHAPE_PRESET_SOLVE_RTOL = 1e-5

#: The Z preset's uniform DOF ladder under the shared SubstrateConfig defaults,
#: from ``4(m+1)^2 - 3(m+1)`` with ``m = 2 * 2**k`` intervals per unit cell. The
#: Z gate's fitting window is derived from exactly these numbers
#: (``factory.ZSHAPE_ADEQUACY_RATE_FIT_DOF_RANGE``); if they move, re-derive it.
ZSHAPE_UNIFORM_LADDER = [297, 1105, 4257, 16705]


def _z_node_count(intervals_per_cell: int) -> int:
    """Nodes of the Z-tetromino meshed with ``m`` intervals per unit cell axis."""
    side = intervals_per_cell + 1
    return 4 * side**2 - 3 * side


def _zshape_substrate(**config: Any) -> SkfemTriSubstrate:
    substrate = build_substrate_from_config(
        SubstrateConfig(name="zshape_test", kind="skfem_tri", **config),
        operator_name="zshape_poisson",
    )
    assert isinstance(substrate, SkfemTriSubstrate)
    return substrate


def _triangle_areas(mesh: Any) -> np.ndarray:
    p = mesh.p[:, mesh.t]  # (2, 3, n)
    return 0.5 * np.abs(
        (p[0, 1] - p[0, 0]) * (p[1, 2] - p[1, 0]) - (p[0, 2] - p[0, 0]) * (p[1, 1] - p[1, 0])
    )


def _boundary_length(mesh: Any) -> float:
    facets = mesh.facets[:, mesh.boundary_facets()]
    return float(np.linalg.norm(mesh.p[:, facets[0]] - mesh.p[:, facets[1]], axis=0).sum())


class TestLShapePathIsByteIdentical:
    """The committed arena artifact was produced on the L mesh; it must not move."""

    @pytest.mark.parametrize("refinements", [0, 1, 2, 3])
    def test_lshape_operator_mesh_is_the_lshape_builders(self, refinements: int) -> None:
        substrate = SkfemTriSubstrate(
            _lshaped_operator(),
            config=SubstrateConfig(
                name="l_bytes", kind="skfem_tri", initial_refinements=refinements
            ),
        )
        mesh = substrate.initial_mesh().mesh
        reference = build_lshaped_initial_mesh(
            _lshaped_operator(), _require_skfem(), initial_mesh_refinements=refinements
        )
        assert (mesh.p.dtype, mesh.t.dtype) == (reference.p.dtype, reference.t.dtype)
        assert np.array_equal(mesh.p, reference.p)
        assert np.array_equal(mesh.t, reference.t)

    def test_arena_mesh_bytes_match_the_committed_artifact(self) -> None:
        run = json.loads(ARENA_RUN_JSON.read_text(encoding="utf-8"))
        recorded = run["packages"]["packages"]["scikit-fem"]
        installed = _require_skfem().__version__
        if installed != recorded:
            pytest.skip(
                f"scikit-fem {installed} is not the {recorded} the arena artifact was produced "
                f"with; its mesh bytes are only pinned for that version"
            )
        refinements = run["config"]["substrate"]["initial_refinements"]
        substrate = SkfemTriSubstrate(
            _lshaped_operator(),
            config=SubstrateConfig(name="arena", kind="skfem_tri", initial_refinements=refinements),
        )
        digest = hashlib.sha256(substrate.fingerprint(substrate.initial_mesh())).hexdigest()
        assert digest == LSHAPE_ARENA_MESH_SHA256

    @pytest.mark.parametrize("refinements", [0, 1, 2, 3])
    def test_the_lshape_polyomino_reproduces_the_lshape_mesh(self, refinements: int) -> None:
        """Same cells, same order, same node count: the generic builder is the L builder."""
        skfem = _require_skfem()
        polyomino = build_polyomino_initial_mesh(
            PolyominoDomain(LSHAPE_POLYOMINO_CELLS), skfem, initial_mesh_refinements=refinements
        )
        reference = build_lshaped_initial_mesh(
            _lshaped_operator(), skfem, initial_mesh_refinements=refinements
        )
        assert np.array_equal(polyomino.p, reference.p)
        assert np.array_equal(polyomino.t, reference.t)

    def test_the_lshape_preset_solves_like_the_lshape_operator(self) -> None:
        """Same mesh, same errors and indicators -- to float32 precision, not bitwise.

        Both meshes are driven by the *reference's* marking: the two operators
        round the float32 Dirichlet data at different points, which is enough to
        break exact indicator ties (measured: a 2-element swap at step two), so
        comparing the two arms' own markings would test tie-breaking, not the PDE.
        """
        reference = SkfemTriSubstrate(_lshaped_operator())
        preset = SkfemTriSubstrate(build_lshape_multi_corner_operator())
        ref_mesh, preset_mesh = reference.initial_mesh(), preset.initial_mesh()
        assert reference.fingerprint(ref_mesh) == preset.fingerprint(preset_mesh)
        for _ in range(3):
            ref_result, preset_result = reference.solve(ref_mesh), preset.solve(preset_mesh)
            assert ref_result.n_dof == preset_result.n_dof
            assert preset_result.l2_error == pytest.approx(
                ref_result.l2_error, rel=LSHAPE_PRESET_SOLVE_RTOL
            )
            np.testing.assert_allclose(
                preset_result.indicators,
                ref_result.indicators,
                rtol=LSHAPE_PRESET_SOLVE_RTOL,
                atol=LSHAPE_PRESET_SOLVE_RTOL * float(ref_result.indicators.max()),
            )
            marked = reference.mark(ref_result.indicators, theta=0.5)
            ref_mesh, preset_mesh = (
                reference.refine(ref_mesh, marked),
                preset.refine(preset_mesh, marked),
            )


class TestZShapeSubstrate:
    """The Z-tetromino on ``skfem_tri``, built through the production factory."""

    @pytest.fixture
    def substrate(self) -> SkfemTriSubstrate:
        return _zshape_substrate()

    def test_initial_mesh_covers_exactly_the_domain(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh().mesh
        domain = PolyominoDomain(ZSHAPE_POLYOMINO_CELLS)
        assert math.fsum(_triangle_areas(mesh)) == pytest.approx(domain.area, rel=1e-12)
        assert _boundary_length(mesh) == pytest.approx(domain.perimeter, rel=1e-12)

    def test_every_reentrant_corner_is_a_mesh_node(self, substrate: SkfemTriSubstrate) -> None:
        nodes = substrate.initial_mesh().mesh.p.T
        for corner in PolyominoDomain(ZSHAPE_POLYOMINO_CELLS).reentrant_corners:
            assert np.any(np.all(nodes == np.array(corner.position), axis=1)), corner

    def test_initial_mesh_is_conforming(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh().mesh
        assert TestSkfemTriSubstrateAC2Conformity._max_facet_incidence(mesh) <= 2

    def test_adaptive_refinement_stays_conforming_and_covering(
        self, substrate: SkfemTriSubstrate
    ) -> None:
        mesh = substrate.initial_mesh()
        for _ in range(4):
            result = substrate.solve(mesh)
            mesh = substrate.refine(mesh, substrate.mark(result.indicators, theta=0.5))
            assert TestSkfemTriSubstrateAC2Conformity._max_facet_incidence(mesh.mesh) <= 2
            assert _boundary_length(mesh.mesh) == pytest.approx(10.0, rel=1e-12)

    @pytest.mark.parametrize("refinements", [0, 1, 2])
    def test_node_count_follows_the_ladder_formula(self, refinements: int) -> None:
        substrate = _zshape_substrate(initial_refinements=refinements)
        intervals = (POLYOMINO_NODES_PER_CELL - 1) * 2**refinements
        assert substrate.initial_mesh().mesh.p.shape[1] == _z_node_count(intervals)

    def test_nodes_per_cell_is_honoured(self) -> None:
        mesh = build_polyomino_initial_mesh(
            PolyominoDomain(ZSHAPE_POLYOMINO_CELLS),
            _require_skfem(),
            initial_mesh_refinements=0,
            nodes_per_cell=4,
        )
        assert mesh.p.shape[1] == _z_node_count(3)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"initial_mesh_refinements": 0, "nodes_per_cell": 1}, "nodes_per_cell"),
            ({"initial_mesh_refinements": -1}, "initial_mesh_refinements"),
        ],
    )
    def test_builder_rejects_invalid_arguments(self, kwargs: dict[str, int], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            build_polyomino_initial_mesh(
                PolyominoDomain(ZSHAPE_POLYOMINO_CELLS), _require_skfem(), **kwargs
            )

    def test_solve_is_sane(self, substrate: SkfemTriSubstrate) -> None:
        mesh = substrate.initial_mesh()
        result = substrate.solve(mesh)
        assert result.n_dof == ZSHAPE_UNIFORM_LADDER[0]
        assert 0 < result.n_dof_free < result.n_dof
        assert result.indicators.shape == (substrate.n_units(mesh),)
        assert 0.0 < result.l2_error < 1.0
        assert substrate.refinable_mask(mesh).all()

    def test_the_default_strengths_reach_the_substrate(self) -> None:
        default = _zshape_substrate().solve(_zshape_substrate().initial_mesh())
        explicit = build_substrate_from_config(
            SubstrateConfig(name="z_explicit", kind="skfem_tri"),
            operator=build_zshape_poisson_operator(
                primary_coefficient=DEFAULT_ZSHAPE_PRIMARY_COEFFICIENT,
                secondary_coefficient=DEFAULT_ZSHAPE_SECONDARY_COEFFICIENT,
            ),
        )
        same = explicit.solve(explicit.initial_mesh())
        assert same.l2_error == default.l2_error

    def test_strengths_are_settable_through_a_prebuilt_operator(self) -> None:
        """The supported path for non-default coefficients: ``operator=`` on the factory."""
        operator = build_zshape_poisson_operator(secondary_coefficient=1.0)
        substrate = build_substrate_from_config(
            SubstrateConfig(name="z_custom", kind="skfem_tri"), operator=operator
        )
        custom = substrate.solve(substrate.initial_mesh())
        default = _zshape_substrate().solve(_zshape_substrate().initial_mesh())
        assert isinstance(operator, MultiCornerPoissonOperator)
        assert custom.l2_error != pytest.approx(default.l2_error, rel=1e-3)
        assert not np.allclose(custom.values, default.values)


@pytest.fixture(scope="module")
def zshape_uniform_sweep() -> list[SweepPoint]:
    """The uniform arm on the Z, with the Z gate's budget (one solve per level)."""
    gate = adequacy_gate_for_operator("zshape_poisson")
    return run_refinement_sweep(
        _zshape_substrate(),
        policy="uniform",
        theta=ComparisonParams().marking_fraction,
        max_levels=gate.max_levels_uniform,
        max_dof=gate.max_sweep_dof,
    )


class TestZShapeConvergenceGate:
    """The substrate must converge before any policy comparison on it means anything.

    Mirrors ``tests/research/test_lshape_convergence_gate.py``: uniform
    refinement alone, no marking policy. Both corners are 270 degrees, so the
    singular exponent (2/3) and the expected P1 L2 rate (``N^-2/3``) are the
    L-shape's, and the same band applies. Measured: 8.27e-3, 3.30e-3, 1.31e-3,
    5.18e-4 over the ladder below; fitted rate -0.692 over (200, 5000).
    """

    def test_uniform_dof_ladder_is_the_derived_one(
        self, zshape_uniform_sweep: list[SweepPoint]
    ) -> None:
        assert [p.n_dof for p in zshape_uniform_sweep] == ZSHAPE_UNIFORM_LADDER
        assert [p.n_dof for p in zshape_uniform_sweep] == [
            _z_node_count(2 * 2**k) for k in range(2, 2 + len(ZSHAPE_UNIFORM_LADDER))
        ]

    def test_error_decreases_monotonically_with_dof(
        self, zshape_uniform_sweep: list[SweepPoint]
    ) -> None:
        for coarse, fine in zip(zshape_uniform_sweep, zshape_uniform_sweep[1:]):
            assert fine.l2_error < coarse.l2_error, (
                f"L2 error rose from {coarse.l2_error:.6e} at {coarse.n_dof} DOF to "
                f"{fine.l2_error:.6e} at {fine.n_dof} DOF: the Z substrate is not converging"
            )

    def test_uniform_rate_is_in_the_lshape_band(
        self, zshape_uniform_sweep: list[SweepPoint]
    ) -> None:
        gate = adequacy_gate_for_operator("zshape_poisson")
        rate, n_points = fit_log_log_rate(
            zshape_uniform_sweep, gate.rate_fit_dof_range, arm="uniform"
        )
        low, high = gate.uniform_rate_band
        assert n_points >= 3
        assert low <= rate <= high, f"uniform rate {rate:.4f} outside {gate.uniform_rate_band}"

    def test_finest_pair_rate_is_also_in_band(self, zshape_uniform_sweep: list[SweepPoint]) -> None:
        """The asymptotic pair, not only the fitted average, carries the singular rate."""
        coarse, fine = zshape_uniform_sweep[-2], zshape_uniform_sweep[-1]
        rate = math.log(fine.l2_error / coarse.l2_error) / math.log(fine.n_dof / coarse.n_dof)
        low, high = adequacy_gate_for_operator("zshape_poisson").uniform_rate_band
        assert low <= rate <= high


class TestZShapeThroughTheRefinementGame:
    """The arena's own machinery accepts the new operator name, end to end.

    Mirrors the Slice E ``skfem_tri`` smoke (``tests/pde/games/
    test_substrate_refinement_game.py::TestSkfemTriMctsSmoke``), here because
    this file is the one the scikit-fem CI job actually runs. A mechanism
    check, not a result: one real ``MCTS.get_action`` step on the Z-shape.
    """

    def test_one_mcts_step_on_the_zshape(self) -> None:
        from src.mcts.search import MCTS
        from src.pde.games.substrate_refinement import SubstrateRefinementGame
        from src.pde.games.substrate_refinement_config import SubstrateRefinementConfig
        from src.refinement.adapter import RefinementGameAdapter
        from src.research.substrates.residual_evaluator import ResidualPriorErrorValueEvaluator

        game = SubstrateRefinementGame(
            SubstrateRefinementConfig(
                name="zshape-smoke",
                substrate=SubstrateConfig(
                    name="zshape-smoke-sub",
                    kind="skfem_tri",
                    initial_refinements=1,
                    solve_cache_max_entries=64,
                ),
                operator_name="zshape_poisson",
                max_action_space=512,
                max_steps=2,
                top_k_actions=4,
            )
        )
        assert isinstance(game.substrate, SkfemTriSubstrate)
        adapter = RefinementGameAdapter(game)
        initial_dof = adapter.state.dof
        assert initial_dof == _z_node_count((POLYOMINO_NODES_PER_CELL - 1) * 2)
        mcts = MCTS(
            evaluator=ResidualPriorErrorValueEvaluator(n_actions=adapter.action_space_size),
            n_simulations=2,
            search_mode=adapter.search_mode,
            use_intermediate_rewards=False,
        )
        action = mcts.get_action(adapter, temperature=0.0, add_noise=False)
        adapter.apply_action(action)
        assert adapter.state.step == 1
        assert adapter.state.dof > initial_dof
        assert math.isfinite(adapter.state.error_estimate)
        assert adapter.state.error_estimate > 0.0
