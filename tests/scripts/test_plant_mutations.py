"""Deterministic tests of ``scripts/plant_mutations.py``, the harden-a-guard mutation runner.

The ledger cites this tool's verdicts ("16/16 planted defects killed"), so a
runner that reports a kill it did not see makes every such count false. Each
defect class below misled a real run in this repo before it was guarded:

- an anchor that silently fails to apply looks like a survival, or a kill
  elsewhere -- ``ANCHOR`` refuses it;
- a test already red before the mutation looks like a kill -- the unmutated
  baseline is subtracted;
- an import error naming no test looks like a kill -- ``BROKEN``;
- a mutation that makes a dependency *raise* fails a test without the guard's
  assertion firing (B52: 4-bit weights, then corrupted float initializers,
  both made onnxruntime refuse the model) -- ``reason`` shows the first ``E``
  line so the kill can be read for what it is;
- a hung run, and a hard kill of the runner, left a source file mutated --
  ``TIMEOUT``, and a journal restored on the next start. The journal lives
  in the repository and every path is contained, so a journal planted in a
  shared temp directory cannot become an arbitrary file write;
- an ``--only`` typo selected nothing and reported "all killed".

Every test but one drives ``main`` with an injected runner, so no subprocess
runs and the outcome depends only on the files under ``tmp_path``. The last
test drives the real pytest subprocess once, on a two-file repository.

Planted defects (harden-a-guard), recorded per the skill's Step 7. Each was
applied to ``scripts/plant_mutations.py`` and killed by the named test(s):

1. Baseline failures not subtracted (``killers = failed``):
   ``TestBaseline::test_a_test_red_before_the_mutation_is_not_a_kill``.
2. ``BROKEN`` folded into ``KILLED`` (errors counted as kills):
   ``TestVerdicts::test_errors_alone_are_broken_not_killed``.
3. The anchor count check removed (``count != 1`` -> ``count < 1``):
   ``TestVerdicts::test_an_ambiguous_anchor_is_refused``.
4. The zero-selection guard removed: ``TestSelection::test_an_only_typo
   _selects_nothing_and_fails``.
5. The journal written somewhere ``recover`` never looks: ``TestJournal::
   test_the_journal_holds_the_original_while_the_file_is_mutated``.
6. Restoration skipped on timeout (``finally`` -> ``else``):
   ``TestVerdicts::test_a_timeout_restores_the_file``.
7. ``reason`` dropped from the report: ``TestReport::test_a_kill_reports
   _its_failure_reason``.
8. ``recover`` writing wherever the journal says (the security review's
   arbitrary-write finding): ``TestJournal::test_a_journal_naming_a_path
   _outside_the_repository_is_refused``, all three forms.
9. The journal back in the shared temp directory: ``TestJournal::test_the
   _journal_lives_inside_the_repository``.
10. Recovery restoring over work done since the kill (final review of PR
    #165): ``TestJournal::test_a_file_edited_after_the_kill_is_not_overwritten``.
11. ``expect`` ignored, a wrong-reason failure counted as a kill: ``Test
    ExpectedReason::test_a_failure_for_another_reason_is_not_a_kill``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts import plant_mutations as plant

TARGET = "pkg.py"
ORIGINAL = "VALUE = 1\n"
SELECT = ["test_pkg.py"]
TEST_ID = "test_pkg.py::test_value"
OTHER_ID = "test_pkg.py::test_other"
ASSERTION_REASON = "E   AssertionError: assert 2 == 1"


def _mutation(old: str = "VALUE = 1", new: str = "VALUE = 2", label: str = "M1") -> dict:
    return {"label": label, "path": TARGET, "old": old, "new": new, "select": SELECT}


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A repository holding the one file the mutations change."""
    (tmp_path / TARGET).write_bytes(ORIGINAL.encode())
    return tmp_path


def _spec(tmp_path: Path, mutations: list[dict]) -> Path:
    path = tmp_path / "mutations.py"
    path.write_text(f"MUTATIONS = {mutations!r}\n", encoding="utf-8")
    return path


def _runner(
    when_mutated: plant.RunResult,
    unmutated: plant.RunResult | None = None,
) -> Callable[[list[str], argparse.Namespace, Path], plant.RunResult]:
    """A runner whose result depends only on whether the target file is mutated."""
    clean = unmutated or plant.RunResult((), (), "1 passed", "")

    def run(select: list[str], args: argparse.Namespace, root: Path) -> plant.RunResult:
        del select, args
        mutated = (root / TARGET).read_bytes() != ORIGINAL.encode()
        return when_mutated if mutated else clean

    return run


def _main(
    repo: Path, mutations: list[dict], runner: Callable[..., plant.RunResult], *argv: str
) -> int:
    return plant.main([str(_spec(repo, mutations)), *argv], repo_root=repo, runner=runner)


KILL = plant.RunResult((TEST_ID,), (), "1 failed", ASSERTION_REASON)


class TestVerdicts:
    def test_a_named_failure_is_a_kill(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _main(repo, [_mutation()], _runner(KILL)) == 0
        out = capsys.readouterr().out
        assert "M1: KILLED" in out and TEST_ID in out

    def test_a_green_run_is_a_survival(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        clean = plant.RunResult((), (), "1 passed", "")
        assert _main(repo, [_mutation()], _runner(clean)) == 1
        assert "M1: SURVIVED" in capsys.readouterr().out

    def test_errors_alone_are_broken_not_killed(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        broken = plant.RunResult((), ("test_pkg.py",), "1 error", "E   ImportError: boom")
        assert _main(repo, [_mutation()], _runner(broken)) == 1
        assert "M1: BROKEN" in capsys.readouterr().out

    def test_a_missing_anchor_is_refused_and_nothing_changes(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _main(repo, [_mutation(old="VALUE = 9")], _runner(KILL)) == 1
        assert "M1: ANCHOR" in capsys.readouterr().out
        assert (repo / TARGET).read_bytes() == ORIGINAL.encode()

    def test_an_ambiguous_anchor_is_refused(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (repo / TARGET).write_bytes(b"VALUE = 1\nVALUE = 1\n")
        assert _main(repo, [_mutation()], _runner(KILL)) == 1
        assert "anchor occurs 2 times" in capsys.readouterr().out

    def test_a_timeout_restores_the_file(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def hangs(select: list[str], args: argparse.Namespace, root: Path) -> plant.RunResult:
            if (root / TARGET).read_bytes() != ORIGINAL.encode():
                raise subprocess.TimeoutExpired(cmd="pytest", timeout=args.timeout)
            return plant.RunResult((), (), "1 passed", "")

        assert _main(repo, [_mutation()], hangs, "--timeout", "7") == 1
        assert "M1: TIMEOUT" in capsys.readouterr().out
        assert (repo / TARGET).read_bytes() == ORIGINAL.encode()

    def test_the_file_is_restored_byte_for_byte_with_crlf(self, repo: Path) -> None:
        crlf = b"VALUE = 1\r\nOTHER = 3\r\n"
        (repo / TARGET).write_bytes(crlf)

        def check(select: list[str], args: argparse.Namespace, root: Path) -> plant.RunResult:
            text = (root / TARGET).read_bytes()
            assert b"\r\n" in text, "the mutation flipped the file's line endings"
            return KILL if text != crlf else plant.RunResult((), (), "1 passed", "")

        assert _main(repo, [_mutation()], check) == 0
        assert (repo / TARGET).read_bytes() == crlf


class TestBaseline:
    def test_a_test_red_before_the_mutation_is_not_a_kill(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        already_red = plant.RunResult((OTHER_ID,), (), "1 failed", "E   boom")
        assert _main(repo, [_mutation()], _runner(already_red, unmutated=already_red)) == 1
        out = capsys.readouterr().out
        assert "BASELINE-RED" in out and "M1: SURVIVED" in out

    def test_only_failures_new_under_the_mutation_count(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        red = plant.RunResult((OTHER_ID,), (), "1 failed", "")
        both = plant.RunResult((OTHER_ID, TEST_ID), (), "2 failed", ASSERTION_REASON)
        assert _main(repo, [_mutation()], _runner(both, unmutated=red)) == 0
        killed_block = capsys.readouterr().out.split("M1: KILLED", 1)[1]
        assert TEST_ID in killed_block and OTHER_ID not in killed_block

    def test_each_selection_runs_its_baseline_once(self, repo: Path) -> None:
        calls: list[bool] = []

        def counting(select: list[str], args: argparse.Namespace, root: Path) -> plant.RunResult:
            mutated = (root / TARGET).read_bytes() != ORIGINAL.encode()
            calls.append(mutated)
            return KILL if mutated else plant.RunResult((), (), "1 passed", "")

        two = [_mutation(label="A"), _mutation(new="VALUE = 3", label="B")]
        assert _main(repo, two, counting) == 0
        assert calls.count(False) == 1, f"baseline ran {calls.count(False)} times for one selection"


class TestSelection:
    def test_an_only_typo_selects_nothing_and_fails(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _main(repo, [_mutation()], _runner(KILL), "--only", "M9-typo") == 1
        assert "no mutation matches" in capsys.readouterr().out

    def test_an_empty_spec_fails(self, repo: Path) -> None:
        assert _main(repo, [], _runner(KILL)) == 1

    def test_only_runs_the_matching_labels(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        two = [_mutation(label="A1"), _mutation(old="VALUE = 9", label="B1")]
        assert _main(repo, two, _runner(KILL), "--only", "A") == 0
        out = capsys.readouterr().out
        assert "A1: KILLED" in out and "B1" not in out


class TestJournal:
    def test_a_killed_run_is_restored_on_the_next_start(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seen: list[Path] = []

        def dies(select: list[str], args: argparse.Namespace, root: Path) -> plant.RunResult:
            if (root / TARGET).read_bytes() != ORIGINAL.encode():
                seen.append(plant.journal_path(root))
                raise KeyboardInterrupt  # a hard stop the runner cannot finish
            return plant.RunResult((), (), "1 passed", "")

        with pytest.raises(KeyboardInterrupt):
            _main(repo, [_mutation()], dies)
        # KeyboardInterrupt still unwinds the ``finally``; emulate a SIGKILL instead
        # by putting the mutated bytes and the journal back as a killed process leaves them.
        journal = seen[0]
        mutated = b"VALUE = 2\n"
        journal.write_text(
            json.dumps(plant.journal_entry(TARGET, ORIGINAL.encode(), mutated)), encoding="utf-8"
        )
        (repo / TARGET).write_bytes(mutated)

        assert _main(repo, [_mutation()], _runner(KILL)) == 0
        assert "recovered pkg.py" in capsys.readouterr().out
        assert (repo / TARGET).read_bytes() == ORIGINAL.encode()
        assert not journal.exists()

    def test_the_journal_holds_the_original_while_the_file_is_mutated(self, repo: Path) -> None:
        """The run journals the original before it mutates.

        Without this, the restart test above passes on a run that never
        journals: it writes the journal itself to emulate a SIGKILL.
        """
        journalled: list[bytes] = []

        def inspect(select: list[str], args: argparse.Namespace, root: Path) -> plant.RunResult:
            if (root / TARGET).read_bytes() != ORIGINAL.encode():
                entry = json.loads(plant.journal_path(root).read_text(encoding="utf-8"))
                assert entry["path"] == TARGET
                journalled.append(plant.base64.b64decode(entry["original"]))
                return KILL
            return plant.RunResult((), (), "1 passed", "")

        assert _main(repo, [_mutation()], inspect) == 0
        assert journalled == [ORIGINAL.encode()]

    def test_a_file_edited_after_the_kill_is_not_overwritten(self, repo: Path) -> None:
        """Recovery restores only the exact mutated bytes it left, never later work (review)."""
        journal = plant.journal_path(repo)
        journal.write_text(
            json.dumps(plant.journal_entry(TARGET, ORIGINAL.encode(), b"VALUE = 2\n")),
            encoding="utf-8",
        )
        edited = b"VALUE = 1\nNEW_USER_WORK = 42\n"
        (repo / TARGET).write_bytes(edited)
        with pytest.raises(RuntimeError, match="changed since"):
            plant.recover(repo)
        assert (repo / TARGET).read_bytes() == edited
        assert journal.exists(), "the journal is left for a person to inspect"

    def test_a_file_already_restored_just_clears_the_journal(self, repo: Path) -> None:
        journal = plant.journal_path(repo)
        journal.write_text(
            json.dumps(plant.journal_entry(TARGET, ORIGINAL.encode(), b"VALUE = 2\n")),
            encoding="utf-8",
        )
        assert plant.recover(repo) is None
        assert not journal.exists()
        assert (repo / TARGET).read_bytes() == ORIGINAL.encode()

    def test_the_journal_is_removed_after_a_normal_run(self, repo: Path) -> None:
        assert _main(repo, [_mutation()], _runner(KILL)) == 0
        assert not plant.journal_path(repo).exists()

    def test_a_mutation_path_outside_the_repository_is_refused(self, repo: Path) -> None:
        escaping = _mutation()
        escaping["path"] = "../pkg.py"
        (repo.parent / "pkg.py").write_bytes(ORIGINAL.encode())
        with pytest.raises(ValueError, match="outside the repository"):
            _main(repo, [escaping], _runner(KILL))
        assert (repo.parent / "pkg.py").read_bytes() == ORIGINAL.encode()

    def test_the_journal_is_per_repository(self, tmp_path: Path) -> None:
        assert plant.journal_path(tmp_path / "a") != plant.journal_path(tmp_path / "b")

    def test_the_journal_lives_inside_the_repository(self, repo: Path) -> None:
        """Not the shared temp dir, where another local user could plant one (security review)."""
        assert plant.journal_path(repo).resolve().parent == repo.resolve()

    @pytest.mark.parametrize("form", ["parent", "nested-parent", "absolute"])
    def test_a_journal_naming_a_path_outside_the_repository_is_refused(
        self, repo: Path, form: str
    ) -> None:
        """A planted journal must not become an arbitrary file write."""
        target = {
            "parent": "../outside.txt",
            "nested-parent": "sub/../../outside.txt",
            "absolute": str(repo.parent / "outside.txt"),
        }[form]
        journal = plant.journal_path(repo)
        journal.write_text(
            json.dumps({"path": target, "original": plant.encode_original(b"pwned")}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="outside the repository"):
            plant.recover(repo)
        assert not (repo.parent / "outside.txt").exists()
        assert journal.exists(), "a refused journal is left for a person to inspect"


class TestExpectedReason:
    """``expect`` makes "killed for the defect's reason" mechanical, not a reading exercise."""

    def test_a_kill_for_the_expected_reason_counts(self, repo: Path) -> None:
        mutation = {**_mutation(), "expect": "AssertionError"}
        assert _main(repo, [mutation], _runner(KILL)) == 0

    def test_a_failure_for_another_reason_is_not_a_kill(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        raised = plant.RunResult((TEST_ID,), (), "1 failed", "E   onnxruntime.InvalidGraph: bad")
        mutation = {**_mutation(), "expect": "AssertionError"}
        assert _main(repo, [mutation], _runner(raised)) == 1
        assert "M1: WRONG-REASON" in capsys.readouterr().out


class TestReport:
    def test_a_kill_reports_its_failure_reason(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _main(repo, [_mutation()], _runner(KILL)) == 0
        assert f"reason: {ASSERTION_REASON.removeprefix('E').strip()}" in capsys.readouterr().out


class TestParsePytestOutput:
    def test_reads_failed_errored_summary_and_first_reason(self) -> None:
        out = "\n".join(
            [
                "test_pkg.py F.E",
                "E   AssertionError: assert 2 == 1",
                "E   second line",
                f"FAILED {TEST_ID} - AssertionError",
                "ERROR test_pkg.py::test_fixture - RuntimeError",
                "======= 1 failed, 1 passed, 1 error in 0.10s =======",
            ]
        )
        result = plant.parse_pytest_output(out)
        assert result.failed == (TEST_ID,)
        assert result.errored == ("test_pkg.py::test_fixture",)
        assert result.summary == "1 failed, 1 passed, 1 error in 0.10s"
        assert result.reason == "E   AssertionError: assert 2 == 1"

    def test_a_clean_run_has_no_reason(self) -> None:
        result = plant.parse_pytest_output("===== 3 passed in 0.01s =====")
        assert result == plant.RunResult((), (), "3 passed in 0.01s", "")


def test_the_real_pytest_runner_reports_a_kill(tmp_path: Path) -> None:
    """One journey through the subprocess runner, on a two-file repository."""
    (tmp_path / TARGET).write_text(ORIGINAL, encoding="utf-8")
    (tmp_path / "test_pkg.py").write_text(
        "from pkg import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n", encoding="utf-8"
    )
    spec = _spec(tmp_path, [_mutation()])

    code = plant.main(
        [str(spec), "--pytest-arg=-p", "--pytest-arg=no:cacheprovider", "--timeout", "120"],
        repo_root=tmp_path,
    )
    assert code == 0
    assert (tmp_path / TARGET).read_text(encoding="utf-8") == ORIGINAL
