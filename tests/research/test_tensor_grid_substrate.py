"""Golden test for ``TensorGridSubstrate`` (AC1): the back-compat proof.

Drives ``TensorGridSubstrate`` through the exact ``initial_mesh -> solve ->
mark -> refine`` loop ``run_dorfler_arm`` (``src/research/lshape_amr_compare.py``)
implements, and asserts the two trajectories are bitwise identical -- proving
the ``mark()``/``refine()`` split (selection via the shared
``dorfler_mark``, then axis-projection + ``_refine_grid``) reproduces
``DorflerAMRSolver._dorfler_mark_2d``'s fused selection+projection exactly,
not merely by inspection. Also checked to float tolerance against the
committed ``results/lshape_mcts_vs_dorfler.csv`` per AC1's second clause.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import LShapedPoissonOperator, PoissonOperator
from src.research.lshape_amr_compare import (
    ComparisonParams,
    lshape_inside_predicate,
    make_solve_fn,
    run_dorfler_arm,
)
from src.research.substrates.config import SubstrateConfig
from src.research.substrates.tensor_grid import TensorGridSubstrate

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_CSV = REPO_ROOT / "results" / "lshape_mcts_vs_dorfler.csv"


def _build_operator(scale: float) -> LShapedPoissonOperator:
    return LShapedPoissonOperator(
        PDEConfig(
            name="poisson_lshaped",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-scale, -scale],
            domain_max=[scale, scale],
        )
    )


def _drive_substrate(
    substrate: TensorGridSubstrate,
    params: ComparisonParams,
) -> list[tuple[int, int, float]]:
    """Reproduce ``run_dorfler_arm``'s own loop shape, over the Protocol."""
    mesh = substrate.initial_mesh()
    rows: list[tuple[int, int, float]] = []
    for level in range(params.max_refinements + 1):
        result = substrate.solve(mesh)
        rows.append((level, result.n_dof, result.l2_error))
        if result.n_dof >= params.max_dof or result.l2_error < params.error_tolerance:
            break
        marked = substrate.mark(result.indicators, params.marking_fraction)
        mesh = substrate.refine(mesh, marked)
    return rows


def _read_committed_dorfler_rows(seed: int) -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    with COMMITTED_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "dorfler" and int(row["seed"]) == seed:
                rows.append(
                    (int(row["refinement_level"]), int(row["n_dof"]), float(row["l2_error"]))
                )
    rows.sort(key=lambda r: r[0])
    return rows


class TestTensorGridSubstrateGolden:
    """AC1: bitwise trajectory parity against the live legacy Dörfler arm."""

    @pytest.fixture
    def params(self) -> ComparisonParams:
        return ComparisonParams()

    @pytest.fixture
    def operator(self, params: ComparisonParams) -> LShapedPoissonOperator:
        return _build_operator(params.scale)

    def test_trajectory_matches_live_run_dorfler_arm_bitwise(
        self, operator: LShapedPoissonOperator, params: ComparisonParams
    ) -> None:
        inside = lshape_inside_predicate(params.scale)
        solve_fn = make_solve_fn(operator, inside)
        legacy_traj = run_dorfler_arm(operator, solve_fn, params)
        legacy_rows = [(p.level, p.n_dof, p.l2_error) for p in legacy_traj.points]

        substrate = TensorGridSubstrate(
            operator,
            inside=inside,
            config=SubstrateConfig(
                name="tensor_grid_golden",
                kind="tensor_grid",
                initial_side=params.initial_side,
            ),
        )
        substrate_rows = _drive_substrate(substrate, params)

        assert len(substrate_rows) == len(legacy_rows)
        for (s_level, s_dof, s_l2), (l_level, l_dof, l_l2) in zip(
            substrate_rows, legacy_rows, strict=True
        ):
            assert s_level == l_level
            assert s_dof == l_dof
            assert s_l2 == l_l2, f"level {s_level}: {s_l2!r} != {l_l2!r}"

    def test_trajectory_matches_committed_csv_to_float_tolerance(
        self, operator: LShapedPoissonOperator, params: ComparisonParams
    ) -> None:
        """AC1's second clause: float-tolerance against the committed artifact.

        The CSV's ``l2_error`` column is 8-significant-figure text
        (``f"{l2:.8e}"``), so comparing for text-exactness would be brittle.
        ``rel=1e-4`` is deliberately looser than that text precision: a
        sparse ``spsolve`` can differ in its low-order bits across BLAS/LAPACK
        builds (confirmed cross-runner -- CI's Linux BLAS build reproduced
        this CSV's own values to only ~8.6e-6 relative, not text-exact),
        which is a platform property, not a correctness one. This tolerance
        is still ~100x tighter than a genuine algorithmic divergence would
        produce (the mutation check on the *other* test in this class shows
        a wrong marking policy diverges by tens of percent within a few
        levels), so it still discriminates a real defect.
        """
        inside = lshape_inside_predicate(params.scale)
        substrate = TensorGridSubstrate(
            operator,
            inside=inside,
            config=SubstrateConfig(
                name="tensor_grid_golden_csv",
                kind="tensor_grid",
                initial_side=params.initial_side,
            ),
        )
        substrate_rows = _drive_substrate(substrate, params)
        committed_rows = _read_committed_dorfler_rows(seed=7961)

        committed_by_level = {level: (dof, l2) for level, dof, l2 in committed_rows}
        overlapping_levels = [
            level for level, _, _ in substrate_rows if level in committed_by_level
        ]
        assert overlapping_levels, "no overlapping refinement levels with the committed CSV"

        for level, dof, l2 in substrate_rows:
            if level not in committed_by_level:
                continue
            committed_dof, committed_l2 = committed_by_level[level]
            assert dof == committed_dof
            assert l2 == pytest.approx(committed_l2, rel=1e-4)


class TestTensorGridSubstratePrimitives:
    """Unit coverage for the primitive methods the golden test doesn't reach.

    Uses a plain rectangular ``PoissonOperator`` with ``inside=None`` -- the
    unmasked full-bounding-box path, the ``historical behaviour unchanged``
    branch every ``inside``-taking primitive documents.
    """

    @pytest.fixture
    def operator(self) -> PoissonOperator:
        return PoissonOperator(
            PDEConfig(
                name="poisson_rect",
                pde_type=PDEType.POISSON,
                domain_dim=2,
                domain_min=[0.0, 0.0],
                domain_max=[1.0, 1.0],
            )
        )

    @pytest.fixture
    def substrate(self, operator: PoissonOperator) -> TensorGridSubstrate:
        return TensorGridSubstrate(
            operator,
            inside=None,
            config=SubstrateConfig(name="tensor_grid_unit", kind="tensor_grid", initial_side=4),
        )

    def test_solve_without_inside_uses_full_bounding_box(
        self, substrate: TensorGridSubstrate
    ) -> None:
        mesh = substrate.initial_mesh()
        result = substrate.solve(mesh)
        assert result.n_dof == len(mesh.xs) * len(mesh.ys)
        assert result.n_dof_free == (len(mesh.xs) - 2) * (len(mesh.ys) - 2)
        assert result.indicators.shape == (substrate.n_units(mesh),)

    def test_n_units(self, substrate: TensorGridSubstrate) -> None:
        mesh = substrate.initial_mesh()
        assert substrate.n_units(mesh) == (len(mesh.xs) - 1) * (len(mesh.ys) - 1)

    def test_refinable_mask_is_all_true(self, substrate: TensorGridSubstrate) -> None:
        mesh = substrate.initial_mesh()
        mask = substrate.refinable_mask(mesh)
        assert mask.shape == (substrate.n_units(mesh),)
        assert mask.all()

    def test_fingerprint_changes_after_refine(self, substrate: TensorGridSubstrate) -> None:
        mesh = substrate.initial_mesh()
        result = substrate.solve(mesh)
        marked = substrate.mark(result.indicators, theta=0.3)
        refined = substrate.refine(mesh, marked)
        assert substrate.fingerprint(mesh) != substrate.fingerprint(refined)

    def test_fingerprint_deterministic_for_identical_mesh(
        self, substrate: TensorGridSubstrate
    ) -> None:
        mesh_a = substrate.initial_mesh()
        mesh_b = substrate.initial_mesh()
        assert substrate.fingerprint(mesh_a) == substrate.fingerprint(mesh_b)

    def test_describe(self, substrate: TensorGridSubstrate) -> None:
        info = substrate.describe()
        assert info["kind"] == "tensor_grid"
        assert info["initial_side"] == 4

    def test_missing_scipy_raises_import_error(
        self, operator: PoissonOperator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "scipy":
                raise ImportError("no scipy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ImportError, match="requires scipy"):
            TensorGridSubstrate(operator)


class TestTensorGridSubstrateConfigIsHonoured:
    """Every ``SubstrateConfig`` field this substrate accepts must actually do something.

    A gap-analysis review found four fields — ``marking_variant``,
    ``error_metric``, ``enforce_immutable_meshes`` and ``initial_refinements``
    — that this substrate declared, validated and then silently ignored. Two
    are now read (the first two below), one is now enforced
    (``enforce_immutable_meshes``), and the fourth is rejected outright by
    ``SubstrateConfig``'s kind-scope validator because it means nothing here.
    These tests are what stop that regressing to decoration again.
    """

    @pytest.fixture
    def operator(self) -> LShapedPoissonOperator:
        return _build_operator(ComparisonParams().scale)

    def _substrate(self, operator: LShapedPoissonOperator, **kwargs: object) -> TensorGridSubstrate:
        params = ComparisonParams()
        return TensorGridSubstrate(
            operator,
            inside=lshape_inside_predicate(params.scale),
            config=SubstrateConfig(
                name="tg_cfg",
                kind="tensor_grid",
                initial_side=params.initial_side,
                **kwargs,  # type: ignore[arg-type]
            ),
        )

    @pytest.mark.parametrize(
        "indicators",
        [
            # Squared bulk reaches theta with one element here; linear needs two.
            np.array([2.0, 1.0, 1.0, 1.0, 1.0]),
            # AC4's documented divergence: on an all-zero array the squared
            # variant still marks exactly one element, the linear variant none.
            np.zeros(5),
        ],
        ids=["weighted-vs-unweighted-bulk", "all-zeros"],
    )
    def test_marking_variant_is_read_from_config(
        self, operator: LShapedPoissonOperator, indicators: np.ndarray
    ) -> None:
        """The field must change the selection, not just be stored.

        Synthetic indicators on purpose: the two variants agree on this
        substrate's *real* coarse-mesh indicators (one dominant element), so a
        real-indicator test would pass whether or not the config were read --
        a non-test. These two inputs are chosen because they are exactly where
        the variants provably differ.
        """
        squared = self._substrate(operator, marking_variant="squared")
        linear = self._substrate(operator, marking_variant="linear")
        assert not np.array_equal(squared.mark(indicators, 0.5), linear.mark(indicators, 0.5))

    def test_error_metric_selects_which_l2_is_reported(
        self, operator: LShapedPoissonOperator
    ) -> None:
        quad = self._substrate(operator, error_metric="quadrature")
        nodal = self._substrate(operator, error_metric="nodal_rms")
        mesh = quad.initial_mesh()
        r_quad = quad.solve(mesh)
        r_nodal = nodal.solve(mesh)
        assert r_quad.l2_error != r_nodal.l2_error
        assert r_quad.l2_error == r_quad.extra["l2_error_area_weighted"]
        assert r_nodal.l2_error == r_nodal.extra["l2_error_nodal_rms"]

    def test_both_metrics_are_always_reported_in_extra(
        self, operator: LShapedPoissonOperator
    ) -> None:
        """AC6's shape: the unselected metric is additive, never dropped."""
        result = self._substrate(operator).solve(self._substrate(operator).initial_mesh())
        assert {"l2_error_area_weighted", "l2_error_nodal_rms"} <= set(result.extra)

    def test_enforce_immutable_meshes_clears_write_flags(
        self, operator: LShapedPoissonOperator
    ) -> None:
        """A frozen dataclass stops rebinding fields, not in-place array writes."""
        substrate = self._substrate(operator, enforce_immutable_meshes=True)
        mesh = substrate.initial_mesh()
        assert not mesh.xs.flags.writeable
        assert not mesh.ys.flags.writeable
        with pytest.raises(ValueError):
            mesh.xs[0] = 0.0

    def test_immutability_can_be_opted_out(self, operator: LShapedPoissonOperator) -> None:
        mesh = self._substrate(operator, enforce_immutable_meshes=False).initial_mesh()
        assert mesh.xs.flags.writeable

    def test_refined_meshes_are_frozen_too(self, operator: LShapedPoissonOperator) -> None:
        substrate = self._substrate(operator)
        mesh = substrate.initial_mesh()
        refined = substrate.refine(mesh, substrate.mark(substrate.solve(mesh).indicators, 0.5))
        assert not refined.xs.flags.writeable


class TestTensorGridRefinableMask:
    """``refinable_mask`` must agree with the estimator, not merely be permissive.

    It previously returned all-True unconditionally. With a geometry predicate
    that is wrong in a way that matters: ``_compute_indicators_2d`` forces
    out-of-domain elements to a **zero** indicator, so they can never be
    marked — calling them refinable is a claim the estimator contradicts. And
    the mask is read, not decorative: a uniform sweep marks exactly it
    (``src/research/substrates/sweep.py``).
    """

    @pytest.fixture
    def params(self) -> ComparisonParams:
        return ComparisonParams()

    def test_excludes_the_notch_when_a_geometry_predicate_is_given(
        self, params: ComparisonParams
    ) -> None:
        substrate = TensorGridSubstrate(
            _build_operator(params.scale),
            inside=lshape_inside_predicate(params.scale),
            config=SubstrateConfig(
                name="tg_mask", kind="tensor_grid", initial_side=params.initial_side
            ),
        )
        mesh = substrate.initial_mesh()
        mask = substrate.refinable_mask(mesh)
        assert mask.shape == (substrate.n_units(mesh),)
        assert not mask.all(), "the L-shape notch must be excluded"
        # The notch is one of four quadrants of the bounding box.
        assert mask.sum() == pytest.approx(0.75 * mask.size, rel=0.05)

    def test_agrees_with_the_zeroed_indicators(self, params: ComparisonParams) -> None:
        """The binding property: every non-refinable element has a zero indicator.

        This is what makes the mask *correct* rather than merely different --
        it is derived from the same ``element_inside_mask`` the estimator uses,
        so the two cannot drift apart.
        """
        substrate = TensorGridSubstrate(
            _build_operator(params.scale),
            inside=lshape_inside_predicate(params.scale),
            config=SubstrateConfig(
                name="tg_mask2", kind="tensor_grid", initial_side=params.initial_side
            ),
        )
        mesh = substrate.initial_mesh()
        indicators = substrate.solve(mesh).indicators
        mask = substrate.refinable_mask(mesh)
        assert np.all(indicators[~mask] == 0.0)

    def test_agrees_with_the_zeroed_indicators_after_refinement(
        self, params: ComparisonParams
    ) -> None:
        """The invariant must survive refinement, not just hold on the coarse mesh.

        `_refine_grid` inserts midpoints, so a refined cell's *centre* can land
        exactly on the slit (x=0 or y=0) — the one place the interior-unknown
        predicate and a closed-domain membership test disagree. Sharing
        `element_inside_mask` with the estimator is what keeps the mask and the
        indicators consistent there; this drives four levels to prove it,
        rather than trusting that the initial mesh is representative.
        """
        substrate = TensorGridSubstrate(
            _build_operator(params.scale),
            inside=lshape_inside_predicate(params.scale),
            config=SubstrateConfig(
                name="tg_mask3", kind="tensor_grid", initial_side=params.initial_side
            ),
        )
        mesh = substrate.initial_mesh()
        for level in range(4):
            result = substrate.solve(mesh)
            mask = substrate.refinable_mask(mesh)
            assert np.all(result.indicators[~mask] == 0.0), (
                f"level {level}: a non-refinable element has a non-zero indicator"
            )
            mesh = substrate.refine(mesh, substrate.mark(result.indicators, 0.5))

    def test_all_true_without_a_geometry_predicate(self) -> None:
        substrate = TensorGridSubstrate(
            PoissonOperator(
                PDEConfig(
                    name="poisson_rect",
                    pde_type=PDEType.POISSON,
                    domain_dim=2,
                    domain_min=[0.0, 0.0],
                    domain_max=[1.0, 1.0],
                )
            ),
            inside=None,
            config=SubstrateConfig(name="tg_full", kind="tensor_grid", initial_side=4),
        )
        mesh = substrate.initial_mesh()
        assert substrate.refinable_mask(mesh).all()


class TestExactSolutionGuard:
    """The construction-time guard, on the *other* caller.

    ``tests/research/test_skfem_substrate.py`` pins the same contract for
    ``SkfemTriSubstrate``. Both are needed: the guard now lives in one shared
    helper (``src.research.baselines.require_exact_solution``, whose own tests
    pin the ``operator.dim`` probe), but a substrate that simply stops *calling*
    it would leave that helper's tests green while regaining the several-frames-
    deep ``TypeError`` the guard exists to replace.
    """

    def test_construction_raises_with_an_actionable_message(self) -> None:
        operator = _build_operator(1.0)
        object.__setattr__(operator, "exact_solution", lambda pts: None)
        with pytest.raises(ValueError, match="analytic exact solution"):
            TensorGridSubstrate(operator, config=SubstrateConfig(name="t", kind="tensor_grid"))

    def test_message_names_this_substrate_not_the_helper(self) -> None:
        operator = _build_operator(1.0)
        object.__setattr__(operator, "exact_solution", lambda pts: None)
        with pytest.raises(ValueError, match="TensorGridSubstrate"):
            TensorGridSubstrate(operator, config=SubstrateConfig(name="t", kind="tensor_grid"))

    def test_a_real_operator_still_constructs(self) -> None:
        substrate = TensorGridSubstrate(
            _build_operator(1.0), config=SubstrateConfig(name="t", kind="tensor_grid")
        )
        assert substrate is not None
