"""Tests for scripts/run_adaptive_vs_uniform.py.

The script exists to make the charter's *"L-shape adaptive Dörfler vs uniform at
matched DOF"* row cite an artifact that actually **contains both arms**. The
previously cited file (``results/lshape_mcts_vs_dorfler.csv``) has only
``{dorfler, mcts}`` in its ``method`` column, so the charter's evidence guard
passed on a file that could not support the claim.

The load-bearing property here is *shared substrate*: both arms must use one
solver, one geometry predicate and one refinement primitive, or the comparison
measures the plumbing rather than the marking.

The second property is *provenance*: the sidecar's ``config_hash`` must be a
real, deterministic hash of exactly the ``config`` it records, and
``--proposal-grade`` must refuse -- before computing or writing anything -- to
produce a sidecar the charter could not cite. Every provenance test below runs
into ``tmp_path`` with the git probe monkeypatched; none touches ``results/``.

``--output`` is a run-mode option: where a run writes is recorded under
``artifacts``, never in ``config`` or its hash. Planted defects, each killed by a
named test: ``output`` restored as a hashed config field (the pre-fix shape) ->
``TestConfigHash::test_the_output_path_changes_neither_config_nor_hash`` and
``TestDefaultInvocation::test_a_scratch_path_reproduction_records_the_same_config_and_hash``;
the path folded into ``config_hash`` while ``config`` stays clean -> the second
one (the parser-level test cannot see it); the path recorded in ``config`` but not
hashed -> the second one and ``::test_sidecar_keeps_its_shape_and_gains_a_real_hash``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.run_adaptive_vs_uniform import (
    CONFIG_HASH_HEX_CHARS,
    DEFAULT_OUTPUT,
    DORFLER_MAX_REFINEMENTS,
    ERROR_TOLERANCE,
    EXIT_NOT_PROPOSAL_GRADE,
    EXIT_OK,
    HARNESS,
    RUN_MODE_FLAGS,
    AdaptiveVsUniformConfig,
    build_operator,
    build_parser,
    compare,
    config_from_args,
    export_csv,
    main,
    run_uniform_arm,
)
from src.research.lshape_amr_compare import (
    ComparisonParams,
    lshape_inside_predicate,
    make_solve_fn,
    run_dorfler_arm,
)
from src.research.run_manifest import (
    UNKNOWN,
    GitProvenance,
    assert_proposal_grade,
    load_run_manifest,
    manifest_path_for,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed artifact's sidecar. Read-only here: no test regenerates it.
COMMITTED_SIDECAR = REPO_ROOT / manifest_path_for(DEFAULT_OUTPUT)

#: A deliberately small budget for the in-process ``main`` runs: fast, yet each arm
#: still produces enough levels for ``compare`` to fit a rate.
SMALL_BUDGET_ARGV: tuple[str, ...] = ("--max-dof", "120")

_CLEAN = GitProvenance(sha="c" * 40, branch="main", dirty=False)
_DIRTY = GitProvenance(sha="d" * 40, branch="main", dirty=True)


@pytest.fixture(scope="module")
def params() -> ComparisonParams:
    """A deliberately small budget: these are mechanism tests, not the headline."""
    return ComparisonParams(initial_side=4, max_dof=250, marking_fraction=0.5, max_refinements=8)


@pytest.fixture(scope="module")
def uniform_rows(params: ComparisonParams) -> list[tuple[int, int, float]]:
    operator = build_operator(params.scale)
    solve_fn = make_solve_fn(operator, lshape_inside_predicate(params.scale))
    return run_uniform_arm(solve_fn, params)


class TestParser:
    def test_defaults_are_declared_not_implied(self) -> None:
        args = build_parser().parse_args([])
        assert args.output.endswith(".csv")
        assert args.initial_side > 0 and args.max_dof > 0
        assert 0.0 < args.marking_fraction < 1.0

    def test_every_knob_is_overridable(self) -> None:
        args = build_parser().parse_args(
            ["--output", "/tmp/x.csv", "--initial-side", "8", "--max-dof", "99"]
        )
        assert (args.output, args.initial_side, args.max_dof) == ("/tmp/x.csv", 8, 99)


class TestUniformArm:
    def test_refines_every_element_so_dof_grows_geometrically(
        self, uniform_rows: list[tuple[int, int, float]]
    ) -> None:
        dofs = [row[1] for row in uniform_rows]
        assert len(dofs) >= 3, "need several levels to see the growth"
        ratios = [b / a for a, b in zip(dofs, dofs[1:], strict=False)]
        assert all(r > 2.0 for r in ratios), (
            f"uniform refinement must roughly quadruple DOF per level, got {ratios}"
        )

    def test_error_decreases_monotonically(
        self, uniform_rows: list[tuple[int, int, float]]
    ) -> None:
        errors = [row[2] for row in uniform_rows]
        assert all(b < a for a, b in zip(errors, errors[1:], strict=False)), (
            f"uniform refinement must converge; got {errors}"
        )

    def test_respects_the_dof_budget(
        self, uniform_rows: list[tuple[int, int, float]], params: ComparisonParams
    ) -> None:
        assert uniform_rows[-1][1] >= params.max_dof or len(uniform_rows) > 1

    def test_starts_from_the_same_grid_as_the_dorfler_arm(
        self, uniform_rows: list[tuple[int, int, float]], params: ComparisonParams
    ) -> None:
        """Level 0 must be the shared coarse solve, or the arms are not comparable."""
        from src.research.lshape_amr_compare import run_dorfler_arm

        operator = build_operator(params.scale)
        solve_fn = make_solve_fn(operator, lshape_inside_predicate(params.scale))
        dorfler = run_dorfler_arm(operator, solve_fn, params)
        assert uniform_rows[0][1] == dorfler.points[0].n_dof
        assert uniform_rows[0][2] == pytest.approx(dorfler.points[0].l2_error, rel=0)


class TestCompare:
    def test_reports_rates_and_a_ratio_band(self) -> None:
        uniform = [(0, 16, 1e-2), (1, 64, 2.5e-3), (2, 256, 6e-4)]
        dorfler_dofs = np.array([16.0, 64.0, 256.0])
        dorfler_errors = np.array([1e-2, 8e-3, 6e-3])
        metrics = compare(uniform, dorfler_dofs, dorfler_errors)
        assert metrics["uniform_convergence_exponent"] < -0.4
        assert metrics["dorfler_convergence_exponent"] > -0.3
        assert metrics["dorfler_over_uniform_max"] > 1.0

    def test_ratio_direction_is_dorfler_over_uniform(self) -> None:
        """Above 1 must mean adaptive is WORSE -- the direction is the claim."""
        uniform = [(0, 16, 1e-3), (1, 64, 1e-4)]
        metrics = compare(uniform, np.array([16.0, 64.0]), np.array([1e-3, 1e-2]))
        assert metrics["dorfler_over_uniform_max"] > 1.0

    def test_min_and_max_bound_the_reported_readings(self) -> None:
        uniform = [(0, 16, 1e-2), (1, 64, 2.5e-3), (2, 256, 6e-4)]
        metrics = compare(uniform, np.array([16.0, 64.0, 256.0]), np.array([1e-2, 8e-3, 6e-3]))
        readings = [
            value for key, value in metrics.items() if key.startswith("dorfler_over_uniform_at_")
        ]
        assert min(readings) == pytest.approx(metrics["dorfler_over_uniform_min"])
        assert max(readings) == pytest.approx(metrics["dorfler_over_uniform_max"])


class TestExportCsv:
    def test_artifact_contains_both_arms(self, tmp_path: Path) -> None:
        """The whole reason the script exists."""
        path = export_csv(tmp_path / "out.csv", [(0, 16, 1e-2)], [(0, 16, 1e-2), (1, 24, 9e-3)])
        with path.open(encoding="utf-8") as handle:
            methods = {row["method"] for row in csv.DictReader(handle)}
        assert methods == {"uniform", "dorfler"}, (
            "an artifact backing a comparison claim must contain the arms compared"
        )

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = export_csv(tmp_path / "a" / "b" / "out.csv", [(0, 16, 1e-2)], [(0, 16, 1e-2)])
        assert path.exists()


def _perturbed(value: Any) -> Any:
    """A different value of the same type, for the any-field-changes-the-hash test."""
    if isinstance(value, bool):  # pragma: no cover - no bool field today; guards the next one
        return not value
    if isinstance(value, int):
        return value + 2
    if isinstance(value, float):
        return value / 2
    if isinstance(value, str):
        return f"{value}.alt"
    raise TypeError(f"teach _perturbed about {type(value).__name__} config fields")


def _snapshot(
    monkeypatch: pytest.MonkeyPatch, git: GitProvenance, csv_path: Path | None = None
) -> list[str | None]:
    """Make the run's git snapshot return ``git``.

    Returns:
        One entry per probe: ``csv_path``'s content at probe time (``None`` if it
        did not exist), so a test can prove the snapshot preceded the write.

    """
    seen: list[str | None] = []

    def _probe(repo_root: Path | None = None) -> GitProvenance:
        exists = csv_path is not None and csv_path.exists()
        seen.append(csv_path.read_text(encoding="utf-8") if exists else None)
        return git

    monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", _probe)
    return seen


def _run(output_dir: Path, *extra: str) -> tuple[int, Path]:
    """``main`` into ``output_dir`` at the small budget; returns (exit code, CSV path)."""
    csv_path = output_dir / "out.csv"
    return main(["--output", str(csv_path), *SMALL_BUDGET_ARGV, *extra]), csv_path


class TestConfigHash:
    """``config_hash`` is real, deterministic, and covers exactly the recorded config.

    It must not depend on where the run wrote: ``--output`` is a run-mode option,
    recorded under ``artifacts``, or a scratch-path reproduction could never match
    the committed sidecar.
    """

    def test_is_deterministic(self) -> None:
        assert AdaptiveVsUniformConfig().compute_hash() == AdaptiveVsUniformConfig().compute_hash()

    def test_is_the_repo_scheme_over_exactly_the_recorded_config(self) -> None:
        """Recomputable from the sidecar alone, with nothing but hashlib."""
        config = AdaptiveVsUniformConfig()
        payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
        expected = hashlib.sha256(payload.encode()).hexdigest()[:CONFIG_HASH_HEX_CHARS]
        assert config.compute_hash() == expected
        assert config.compute_hash() != UNKNOWN

    @pytest.mark.parametrize("field", sorted(AdaptiveVsUniformConfig.model_fields))
    def test_changes_when_any_field_changes(self, field: str) -> None:
        base = AdaptiveVsUniformConfig()
        changed = base.model_copy(update={field: _perturbed(getattr(base, field))})
        assert getattr(changed, field) != getattr(base, field)
        assert changed.compute_hash() != base.compute_hash(), f"{field} escapes the hash"

    def test_run_mode_flags_change_neither_config_nor_hash(self) -> None:
        plain = config_from_args(build_parser().parse_args([]))
        graded = config_from_args(build_parser().parse_args(["--proposal-grade"]))
        assert graded == plain
        assert graded.compute_hash() == plain.compute_hash()

    def test_the_output_path_changes_neither_config_nor_hash(self) -> None:
        """A scratch-path reproduction hashes like the committed run.

        The reviewer's case: one computation hashed to ``865edce78a6a9453`` with the
        default output and to ``d0b9aa3758603076`` with ``--output /tmp/check.csv``.
        """
        default = config_from_args(build_parser().parse_args([]))
        scratch = config_from_args(build_parser().parse_args(["--output", "/tmp/check.csv"]))
        assert scratch == default
        assert scratch.compute_hash() == default.compute_hash()


class TestConfigFromArgs:
    """Every CLI knob reaches the config, so none can escape the hash."""

    def test_every_option_is_a_config_field_or_a_run_mode_flag(self) -> None:
        dests = set(vars(build_parser().parse_args([])))
        assert dests - RUN_MODE_FLAGS == set(AdaptiveVsUniformConfig.model_fields)
        assert dests >= RUN_MODE_FLAGS, "a RUN_MODE_FLAGS entry names no parser option"

    def test_parser_defaults_are_the_config_defaults(self) -> None:
        assert config_from_args(build_parser().parse_args([])) == AdaptiveVsUniformConfig()

    def test_every_computational_flag_reaches_the_config(self) -> None:
        """Every flag but the run-mode ones; ``--output`` goes to ``artifacts``."""
        argv = ["--output", "o.csv", "--initial-side", "8", "--max-dof", "99"]
        argv += ["--marking-fraction", "0.3", "--scale", "2.0"]
        assert config_from_args(build_parser().parse_args(argv)).model_dump() == {
            "initial_side": 8,
            "max_dof": 99,
            "marking_fraction": 0.3,
            "scale": 2.0,
        }

    def test_an_unaccounted_option_raises_rather_than_escaping_the_hash(self) -> None:
        args = build_parser().parse_args([])
        args.new_knob = 7
        with pytest.raises(ValueError, match="new_knob"):
            config_from_args(args)

    def test_a_non_config_namespace_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="neither"):
            config_from_args(argparse.Namespace(output="o.csv", proposal_grade=False, bogus=1))


class TestDefaultInvocation:
    """No flag: the historical behaviour, plus a real ``config_hash``."""

    def test_a_dirty_tree_is_recorded_not_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _snapshot(monkeypatch, _DIRTY)
        code, csv_path = _run(tmp_path)
        assert code == EXIT_OK
        assert csv_path.is_file()
        assert load_run_manifest(manifest_path_for(csv_path)).git == _DIRTY

    def test_sidecar_keeps_its_shape_and_gains_a_real_hash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _snapshot(monkeypatch, _CLEAN)
        code, csv_path = _run(tmp_path)
        assert code == EXIT_OK
        manifest = load_run_manifest(manifest_path_for(csv_path))
        assert set(manifest.config) == set(AdaptiveVsUniformConfig.model_fields)
        assert "output" not in manifest.config
        assert manifest.artifacts == {"csv": str(csv_path)}
        assert manifest.harness == HARNESS
        assert manifest.run_id == "adaptive-vs-uniform-4-120"
        assert [arm.name for arm in manifest.arms] == ["uniform", "dorfler"]
        assert manifest.config_hash == AdaptiveVsUniformConfig(**manifest.config).compute_hash()

    def test_the_artifact_is_exactly_what_the_arms_produce(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provenance changed; the numbers and their formatting did not."""
        _snapshot(monkeypatch, _CLEAN)
        code, csv_path = _run(tmp_path)
        assert code == EXIT_OK

        params = ComparisonParams(
            initial_side=4,
            max_dof=120,
            marking_fraction=0.5,
            max_refinements=DORFLER_MAX_REFINEMENTS,
            error_tolerance=ERROR_TOLERANCE,
        )
        operator = build_operator(params.scale)
        solve_fn = make_solve_fn(operator, lshape_inside_predicate(params.scale))
        dorfler = run_dorfler_arm(operator, solve_fn, params)
        expected = export_csv(
            tmp_path / "expected.csv",
            run_uniform_arm(solve_fn, params),
            [(p.level, p.n_dof, p.l2_error) for p in dorfler.points],
        )
        assert csv_path.read_bytes() == expected.read_bytes()

    def test_a_scratch_path_reproduction_records_the_same_config_and_hash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two runs of one computation into different paths differ only in ``artifacts``."""
        _snapshot(monkeypatch, _CLEAN)
        manifests = []
        for subdirectory in ("committed_path", "scratch_path"):
            code, csv_path = _run(tmp_path / subdirectory)
            assert code == EXIT_OK
            manifests.append(load_run_manifest(manifest_path_for(csv_path)))
        first, second = manifests
        expected = config_from_args(build_parser().parse_args(list(SMALL_BUDGET_ARGV)))
        assert first.config == second.config == expected.model_dump(mode="json")
        assert first.config_hash == second.config_hash == expected.compute_hash()
        assert first.artifacts != second.artifacts

    def test_git_is_snapshotted_before_the_artifact_is_rewritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regenerating a committed CSV must not count the regeneration as dirt."""
        csv_path = tmp_path / "out.csv"
        csv_path.write_text("stale artifact from a previous run\n", encoding="utf-8")
        seen = _snapshot(monkeypatch, _CLEAN, csv_path)
        assert main(["--output", str(csv_path), *SMALL_BUDGET_ARGV]) == EXIT_OK
        assert seen == ["stale artifact from a previous run\n"], (
            "exactly one probe, taken before the CSV was rewritten"
        )


class TestProposalGrade:
    """``--proposal-grade`` never produces a sidecar the charter could not cite."""

    @pytest.mark.parametrize(
        "git", [_DIRTY, GitProvenance(dirty=None)], ids=["dirty", "undetermined"]
    )
    def test_refuses_a_tree_it_cannot_prove_clean_before_computing_anything(
        self,
        git: GitProvenance,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _snapshot(monkeypatch, git)

        def _must_not_run(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the pre-flight must reject before any arm runs")

        monkeypatch.setattr("scripts.run_adaptive_vs_uniform.run_dorfler_arm", _must_not_run)
        code, csv_path = _run(tmp_path, "--proposal-grade")
        assert code == EXIT_NOT_PROPOSAL_GRADE
        assert f"git.dirty={git.dirty!r}" in capsys.readouterr().err
        assert not csv_path.exists()
        assert not manifest_path_for(csv_path).exists()

    def test_refuses_an_unhashed_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _snapshot(monkeypatch, _CLEAN)
        monkeypatch.setattr(AdaptiveVsUniformConfig, "compute_hash", lambda self: UNKNOWN)
        code, csv_path = _run(tmp_path, "--proposal-grade")
        assert code == EXIT_NOT_PROPOSAL_GRADE
        assert "config_hash='unknown'" in capsys.readouterr().err
        assert not csv_path.exists()

    def test_refuses_a_sidecar_that_does_not_read_back_proposal_grade(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The bytes on disk are what gets committed, so they are what is checked."""
        _snapshot(monkeypatch, _CLEAN)

        def _reads_back_dirty(path: Path) -> Any:
            return load_run_manifest(path).model_copy(update={"git": _DIRTY})

        monkeypatch.setattr("src.research.run_manifest.load_run_manifest", _reads_back_dirty)
        code, _ = _run(tmp_path, "--proposal-grade")
        assert code == EXIT_NOT_PROPOSAL_GRADE
        assert "git.dirty=True" in capsys.readouterr().err

    def test_a_clean_tree_yields_a_proposal_grade_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _snapshot(monkeypatch, _CLEAN)
        code, csv_path = _run(tmp_path, "--proposal-grade")
        assert code == EXIT_OK
        manifest = load_run_manifest(manifest_path_for(csv_path))
        assert_proposal_grade(manifest)
        assert manifest.git == _CLEAN

    def test_the_flag_changes_verification_not_what_is_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _snapshot(monkeypatch, _CLEAN)
        assert _run(tmp_path)[0] == EXIT_OK
        csv_path = tmp_path / "out.csv"
        plain_csv = csv_path.read_bytes()
        plain = load_run_manifest(manifest_path_for(csv_path)).stable_fields()

        assert _run(tmp_path, "--proposal-grade")[0] == EXIT_OK
        assert csv_path.read_bytes() == plain_csv
        assert load_run_manifest(manifest_path_for(csv_path)).stable_fields() == plain


class TestCommittedSidecar:
    """Hash-pin: the committed sidecar must still be reproducible by today's code.

    Proposal grade itself is ``tests/docs/test_proposal_grade_sidecars.py``'s job;
    these pin what that guard cannot see.
    """

    def test_its_config_hash_recomputes_from_its_recorded_config(self) -> None:
        """A ``compute_hash()`` change invalidates the sidecar: re-record it, do not edit it."""
        manifest = load_run_manifest(COMMITTED_SIDECAR)
        assert AdaptiveVsUniformConfig(**manifest.config).compute_hash() == manifest.config_hash

    def test_its_config_is_the_documented_default_invocation(self) -> None:
        """``python -m scripts.run_adaptive_vs_uniform`` is what produced it."""
        manifest = load_run_manifest(COMMITTED_SIDECAR)
        assert AdaptiveVsUniformConfig(**manifest.config) == AdaptiveVsUniformConfig()
        assert manifest.artifacts == {"csv": DEFAULT_OUTPUT}
        assert manifest.harness == HARNESS


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
