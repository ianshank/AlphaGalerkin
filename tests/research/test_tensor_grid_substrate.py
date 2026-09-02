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
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from structlog.testing import capture_logs

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import LShapedPoissonOperator, PoissonOperator
from src.research.lshape_amr_compare import (
    ComparisonParams,
    lshape_inside_predicate,
    make_solve_fn,
    run_dorfler_arm,
)
from src.research.substrates import tensor_grid as tensor_grid_module
from src.research.substrates.config import (
    SUBSTRATE_PRIMARY_L2_KEY,
    SubstrateConfig,
)
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


class TestSubstrateIdentityIsNotDuplicated:
    """D2: ``describe()`` must report the config's ``kind``, not a hardcoded twin.

    Before the fix, ``SubstrateConfig(kind="skfem_tri")`` handed to
    ``TensorGridSubstrate`` constructed cleanly, passed
    ``_reject_fields_scoped_to_the_other_kind``, and then reported
    ``describe()["kind"] == "tensor_grid"`` -- and the bound structlog logger
    emitted the same wrong value, so the logs could not be trusted to reflect
    the config either. ``kind``'s only production readers were its own
    validator, making it the ``marking_fraction`` pattern with a twist: the
    field *is* load-bearing for validation, so the fix is to wire it, not
    delete it. The kind-scoped validator structurally cannot catch this -- it
    rejects a field scoped to the *other* kind, never a mismatched ``kind``.
    """

    @pytest.fixture
    def operator(self) -> PoissonOperator:
        return PoissonOperator(
            PDEConfig(
                name="poisson_kind",
                pde_type=PDEType.POISSON,
                domain_dim=2,
                domain_min=[0.0, 0.0],
                domain_max=[1.0, 1.0],
            )
        )

    def test_describe_derives_the_kind_rather_than_restating_it(
        self, operator: PoissonOperator
    ) -> None:
        """``describe()`` must *read* the config, not carry a second copy.

        Deliberately white-box, and the first version of this test was a
        tautology worth recording: asserting
        ``describe()["kind"] == substrate._config.kind`` on a normally-built
        substrate passes whether ``describe()`` reads the field or hardcodes
        the same string, because the constructor guard makes the two agree by
        construction. A mutation check caught it -- reverting ``describe()`` to
        a literal left the test green.

        Rebinding ``_config`` afterwards is the only way to separate the two
        spellings, so that is what this does. The state is unreachable through
        the public API precisely *because* of the constructor guard; the point
        is to stop a future edit reintroducing the second source of truth that
        let ``describe()`` disagree with its own config.
        """
        substrate = TensorGridSubstrate(operator)
        substrate._config = SubstrateConfig(name="rebound", kind="skfem_tri")
        assert substrate.describe()["kind"] == "skfem_tri"

    def test_a_mismatched_kind_is_rejected_at_construction(self, operator: PoissonOperator) -> None:
        """Fail loudly rather than silently disagreeing with your own config."""
        with pytest.raises(ValueError, match="kind"):
            TensorGridSubstrate(
                operator,
                config=SubstrateConfig(name="mismatch", kind="skfem_tri"),
            )


class TestPrimaryErrorKeyIsUniform:
    """D4: both substrates must publish the selected metric under one key.

    ``TensorGridSubstrate`` emitted ``l2_error_area_weighted`` while
    ``SkfemTriSubstrate`` emitted ``l2_error_quadrature`` -- two implementations
    of one Protocol shipping the *same slot* under different names, so no
    generic consumer could read it. Worse, the comment above the tensor-grid
    site asserted the opposite ("`extra` carries the same pair
    SkfemTriSubstrate reports"). Fixed additively: the metric-specific keys
    stay, and a shared ``l2_error_primary`` names the selected one.
    """

    @pytest.fixture
    def operator(self) -> LShapedPoissonOperator:
        return _build_operator(1.0)

    @pytest.mark.parametrize("metric", ["quadrature", "nodal_rms"])
    def test_primary_key_present_and_equals_l2_error(
        self, operator: LShapedPoissonOperator, metric: str
    ) -> None:
        substrate = TensorGridSubstrate(
            operator,
            inside=lshape_inside_predicate(1.0),
            config=SubstrateConfig(name="primary", kind="tensor_grid", error_metric=metric),
        )
        result = substrate.solve(substrate.initial_mesh())
        assert SUBSTRATE_PRIMARY_L2_KEY in result.extra
        assert result.extra[SUBSTRATE_PRIMARY_L2_KEY] == result.l2_error


class TestUnmeasurableSolveFailsLoudly:
    """D3: an unmeasurable error must not be published as a perfect score.

    ``nodal_rms_l2_error`` documents "Returns ``None`` -- not a crash, and not
    ``0.0``" and returns ``None`` on *two* triggers: no analytic exact solution
    (guarded at construction by ``require_exact_solution``) and an **empty diff
    array**, which nothing guarded. Both substrates then wrote
    ``float(nodal_rms or 0.0)`` -- exactly the value the helper refuses to
    return -- so an empty in-domain node set published ``l2_error = 0.0``.
    """

    def test_solve_raises_when_the_nodal_rms_is_unmeasurable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        substrate = TensorGridSubstrate(_build_operator(1.0), inside=lshape_inside_predicate(1.0))
        mesh = substrate.initial_mesh()
        monkeypatch.setattr(tensor_grid_module, "nodal_rms_l2_error", lambda *a, **k: None)
        with pytest.raises(ValueError, match="unmeasurable"):
            substrate.solve(mesh)


class TestSubstratesAreRegistered:
    """D5: the registry had no registrants, no exports, and no callers.

    ``src/refinement/substrate_registry.py`` promised lookup-by-key so callers
    "need not import the concrete module directly", but neither substrate
    carried the decorator, the pair was absent from ``src.refinement``'s
    ``__all__`` (its sibling game registry is present), and every registration
    in the tree lived inside a test.
    """

    def test_tensor_grid_is_registered_under_its_kind(self) -> None:
        """Read in a **subprocess**, deliberately.

        ``RefinementSubstrateRegistry`` is a process-global singleton and
        ``tests/refinement/test_substrate.py`` / ``test_skfem_substrate.py``
        both ``clear()`` it in setup *and* teardown, so an in-process assertion
        here passes or fails on collection order -- verified: it goes green
        alone and red after ``tests/refinement/test_substrate.py``. This is the
        hazard ``src/refinement/AGENT.md`` documents and the same reason
        ``tests/docs/test_charter_alignment.py`` reads ``ScenarioRegistry`` out
        of process. A fresh interpreter sees only the import-time registration,
        which is the property under test.
        """
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from src.refinement.substrate_registry import RefinementSubstrateRegistry;"
                "from src.research.substrates.tensor_grid import TensorGridSubstrate;"
                "assert RefinementSubstrateRegistry().get_or_raise('tensor_grid')"
                " is TensorGridSubstrate;"
                "print('registered')",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr
        assert "registered" in proc.stdout

    def test_the_registry_pair_is_exported_from_src_refinement(self) -> None:
        import src.refinement as refinement

        assert "RefinementSubstrateRegistry" in refinement.__all__
        assert "register_refinement_substrate" in refinement.__all__


class TestZeroMarkedRefineWarns:
    """Both substrates must warn on an empty selection, not spin silently.

    ``marking_variant="linear"`` returns all-False on an all-zero indicator
    array, and ``run_refinement_sweep``'s DOF-growth loop would then make no
    progress with nothing said. ``SkfemTriSubstrate`` warned; the tensor grid
    did not, despite reading the same ``marking_variant`` field and running in
    the same loop -- the identical failure observable on one substrate and
    invisible on the other. Neither warning had a test, so both were
    unexecuted code until now.
    """

    def test_empty_selection_emits_a_warning_and_leaves_the_mesh_unrefined(self) -> None:
        substrate = TensorGridSubstrate(_build_operator(1.0), inside=lshape_inside_predicate(1.0))
        mesh = substrate.initial_mesh()
        empty = np.zeros(substrate.n_units(mesh), dtype=bool)
        # structlog, not stdlib logging -- ``caplog`` sees nothing here, which is
        # itself worth recording: the first version of this test passed an empty
        # string into an ``in`` check and would have gone green on a deleted warning.
        with capture_logs() as events:
            refined = substrate.refine(mesh, empty)
        assert any(e["event"] == "substrate_refine_noop" for e in events), events
        assert substrate.n_units(refined) == substrate.n_units(mesh)
