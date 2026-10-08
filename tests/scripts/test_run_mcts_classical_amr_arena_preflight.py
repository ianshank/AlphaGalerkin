"""``--proposal-grade`` on the arena CLI: the RunRecorder pre-flight and the read-back.

Defect class, in one sentence: **a proposal-grade arena run on a tree it cannot
prove clean overwrites the committed ``results/`` artifacts before it is
rejected** -- the shipped YAML writes into ``results/``, and the CLI used to
check cleanliness only on the sidecar the run had already written.

The pre-flight is ``RunRecorder.start``; the read-back is
``scripts.run_mcts_classical_amr_arena.verify_sidecar``, which also requires the
sidecar to carry the git snapshot the pre-flight approved. Every test drives
``main`` on a tiny ``tensor_grid`` config written to ``tmp_path``, so no
committed artifact is touched.

Mutation kills (``harden-a-guard``, 2026-10-08; each planted in
``scripts/run_mcts_classical_amr_arena.py`` and restored byte-for-byte; no
killer is ``gpu_required`` / ``fem_required``):

1. Pre-flight disabled (``proposal_grade=False`` passed to ``RunRecorder.start``)
   -> ``TestPreflight::test_refuses_a_tree_it_cannot_prove_clean_before_running``
   and ``test_refuses_an_unhashed_config``.
2. Snapshot comparison removed from ``verify_sidecar`` ->
   ``TestReadBack::test_a_sidecar_from_another_snapshot_is_rejected``.
3. ``assert_proposal_grade`` removed from the read-back ->
   ``TestReadBack::test_a_matching_record_must_still_be_proposal_grade``.
4. The read-back skipped entirely -> ``test_a_sidecar_from_another_snapshot_is_rejected``.
5. Config-hash equality reintroduced (the adjacent "tightening" that would
   reject every proposal-grade run) ->
   ``TestReadBack::test_the_post_setup_config_hash_is_not_compared``.

5/5 planted defects killed; the test count is larger and is not the number claimed.

Order dependence (fixed 2026-10-08): ``test_refuses_an_unhashed_config`` passed alone and
failed in CI's full run, because ``tests/poc/test_cli_commands.py`` purges
``sys.modules['src.poc.scenarios*']`` and ``main`` builds its config through
``load_config_from_dict``'s call-time import -- a *new* class, so a patch on the class bound
at this module's import never fired (``tests/poc/conftest.py``, rule 1). Reproduce the old
failure by running ``tests/poc/test_cli_commands.py`` first, then this module, in one
pytest invocation.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from scripts.run_mcts_classical_amr_arena import EXIT_NOT_PROPOSAL_GRADE, main, verify_sidecar
from src.poc.scenarios.mcts_classical_amr_arena import MCTSClassicalAMRArenaScenario
from src.poc.scenarios.mcts_classical_amr_arena_config import (
    SCENARIO_NAME,
    MCTSClassicalAMRArenaConfig,
)
from src.research.run_manifest import (
    UNKNOWN,
    GitProvenance,
    ProposalGradeError,
    RunManifest,
    RunRecorder,
    assert_proposal_grade,
    load_run_manifest,
    manifest_path_for,
    write_run_manifest,
)

pytest.importorskip("scipy", reason="scipy required when main() runs the scenario")

_CLEAN = GitProvenance(sha="c" * 40, branch="main", dirty=False)
_OTHER_CLEAN = GitProvenance(sha="e" * 40, branch="main", dirty=False)
_DIRTY = GitProvenance(sha="d" * 40, branch="main", dirty=True)

#: Exit codes a run that reached the gate may return: 0 pass, 1 honest FAIL.
_GATE_EXIT_CODES = (0, 1)

_BASENAME = "arena_preflight"


def _write_config(tmp_path: Path) -> Path:
    """A tiny tensor-grid arena config writing into ``tmp_path``."""
    entry: dict[str, object] = {
        "name": SCENARIO_NAME,
        "device": "cpu",
        "require_adequacy_precondition": False,
        "operator_name": "poisson",
        "substrate": {"name": "tg", "kind": "tensor_grid", "initial_side": 4},
        "max_dof": 60,
        "max_steps": 1,
        "max_refinements_classical": 1,
        "n_simulations": 1,
        "n_seeds": 1,
        "top_k_actions": 2,
        "max_action_space": 64,
        "output_dir": str(tmp_path),
        "artifact_basename": _BASENAME,
    }
    path = tmp_path / "arena.yaml"
    path.write_text(yaml.safe_dump({"scenarios": [entry]}), encoding="utf-8")
    return path


def _config_class_main_builds() -> type[MCTSClassicalAMRArenaConfig]:
    """The config class ``main`` will instantiate, resolved now, not at import.

    ``main`` builds its config through ``load_config_from_dict``, whose import runs at
    call time against the current ``sys.modules``; after another test purges the
    scenario modules, that is a different class object from the one this module bound
    at import (``tests/poc/conftest.py``, rule 1). Resolving it here the same way makes
    a patch land on the class ``main`` actually builds.
    """
    module = importlib.import_module(MCTSClassicalAMRArenaConfig.__module__)
    return cast("type[MCTSClassicalAMRArenaConfig]", module.MCTSClassicalAMRArenaConfig)


def _probe_returning(monkeypatch: pytest.MonkeyPatch, *states: GitProvenance) -> list[int]:
    """Make every git probe return ``states`` in turn (the last one repeats).

    Returns:
        A one-element counter of probes taken, so a test can prove the
        pre-flight probed before anything ran.

    """
    calls = [0]
    sequence: Iterator[GitProvenance] = iter(states)
    last = states[-1]

    def _probe(repo_root: Path | None = None) -> GitProvenance:
        calls[0] += 1
        return next(sequence, last)

    monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", _probe)
    return calls


def _forbid_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def _must_not_run(self: Any) -> Any:
        raise AssertionError("the pre-flight must reject before the scenario runs")

    monkeypatch.setattr(MCTSClassicalAMRArenaScenario, "run", _must_not_run)


class TestPreflight:
    """Nothing is computed or written when the run cannot be proposal-grade."""

    @pytest.mark.parametrize("git", [_DIRTY, GitProvenance(dirty=None)], ids=["dirty", "unknown"])
    def test_refuses_a_tree_it_cannot_prove_clean_before_running(
        self,
        git: GitProvenance,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _probe_returning(monkeypatch, git)
        _forbid_the_run(monkeypatch)
        code = main(["--config", str(_write_config(tmp_path)), "--proposal-grade"])
        assert code == EXIT_NOT_PROPOSAL_GRADE
        assert f"git.dirty={git.dirty!r}" in capsys.readouterr().err
        assert not list(tmp_path.glob(f"{_BASENAME}*")), "a rejected run wrote artifacts"

    def test_refuses_an_unhashed_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _probe_returning(monkeypatch, _CLEAN)
        _forbid_the_run(monkeypatch)
        monkeypatch.setattr(_config_class_main_builds(), "compute_hash", lambda self: UNKNOWN)
        code = main(["--config", str(_write_config(tmp_path)), "--proposal-grade"])
        assert code == EXIT_NOT_PROPOSAL_GRADE
        assert f"config_hash={UNKNOWN!r}" in capsys.readouterr().err

    def test_without_the_flag_a_dirty_tree_is_recorded_not_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Collectors never raise: a plain run documents the dirt and finishes."""
        _probe_returning(monkeypatch, _DIRTY)
        assert main(["--config", str(_write_config(tmp_path))]) in _GATE_EXIT_CODES
        sidecar = manifest_path_for(tmp_path / f"{_BASENAME}.csv")
        assert load_run_manifest(sidecar).git == _DIRTY


class TestReadBack:
    """The bytes on disk are what gets committed, so they are what is checked."""

    def test_a_clean_tree_yields_the_preflights_proposal_grade_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _probe_returning(monkeypatch, _CLEAN)
        code = main(["--config", str(_write_config(tmp_path)), "--proposal-grade"])
        assert code in _GATE_EXIT_CODES
        manifest = load_run_manifest(manifest_path_for(tmp_path / f"{_BASENAME}.csv"))
        assert_proposal_grade(manifest)
        assert manifest.git == _CLEAN
        assert calls[0] >= 2, "the pre-flight and the scenario each take a snapshot"

    def test_a_sidecar_from_another_snapshot_is_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Both snapshots clean, but not the same tree: the sidecar is not the approved run."""
        _probe_returning(monkeypatch, _CLEAN, _OTHER_CLEAN)
        code = main(["--config", str(_write_config(tmp_path)), "--proposal-grade"])
        assert code == EXIT_NOT_PROPOSAL_GRADE
        assert "not the pre-flight snapshot" in capsys.readouterr().err

    def test_a_sidecar_that_reads_back_differently_is_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The check reads the file, not the scenario's in-memory manifest."""
        _probe_returning(monkeypatch, _CLEAN)

        def _reads_back_dirty(path: Path) -> Any:
            return load_run_manifest(path).model_copy(update={"git": _DIRTY})

        monkeypatch.setattr(
            "scripts.run_mcts_classical_amr_arena.load_run_manifest", _reads_back_dirty
        )
        code = main(["--config", str(_write_config(tmp_path)), "--proposal-grade"])
        assert code == EXIT_NOT_PROPOSAL_GRADE
        assert "not the pre-flight snapshot" in capsys.readouterr().err

    def test_a_matching_record_must_still_be_proposal_grade(self, tmp_path: Path) -> None:
        """Defence in depth for a recorder started without the proposal-grade pre-flight."""
        sidecar = tmp_path / "x.run.json"
        write_run_manifest(
            RunManifest(run_id="r", harness="h", config_hash="h", git=_DIRTY), sidecar
        )
        with pytest.raises(ProposalGradeError, match="git.dirty=True"):
            verify_sidecar(RunRecorder(git=_DIRTY, config_hash="h"), sidecar)

    def test_the_post_setup_config_hash_is_not_compared(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``setup`` installs default thresholds after the pre-flight, moving the hash.

        Comparing the two hashes would reject every proposal-grade run; the
        recorded one only has to be real.
        """
        _probe_returning(monkeypatch, _CLEAN)
        config_path = _write_config(tmp_path)
        recorded: list[str] = []
        original = verify_sidecar

        def _spy(recorder: RunRecorder, sidecar: Path) -> None:
            recorded.extend([recorder.config_hash, load_run_manifest(sidecar).config_hash])
            original(recorder, sidecar)

        monkeypatch.setattr("scripts.run_mcts_classical_amr_arena.verify_sidecar", _spy)
        assert main(["--config", str(config_path), "--proposal-grade"]) in _GATE_EXIT_CODES
        preflight_hash, sidecar_hash = recorded
        assert preflight_hash != sidecar_hash, "the hashes agree; tighten verify_sidecar"
        assert UNKNOWN not in recorded

    def test_a_missing_csv_artifact_is_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _probe_returning(monkeypatch, _CLEAN)
        original = MCTSClassicalAMRArenaScenario.run

        def _drop_the_csv(self: MCTSClassicalAMRArenaScenario) -> Any:
            result = original(self)
            result.artifacts.pop("csv", None)
            return result

        monkeypatch.setattr(MCTSClassicalAMRArenaScenario, "run", _drop_the_csv)
        code = main(["--config", str(_write_config(tmp_path)), "--proposal-grade"])
        assert code == EXIT_NOT_PROPOSAL_GRADE
        assert "no CSV artifact" in capsys.readouterr().err
