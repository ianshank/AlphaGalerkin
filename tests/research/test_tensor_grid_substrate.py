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
        (``f"{l2:.8e}"``), so comparing for text-exactness would be brittle;
        this asserts agreement to that precision instead.
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
            assert l2 == pytest.approx(committed_l2, rel=1e-7)


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
