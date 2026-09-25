"""Every run sidecar the charter's evidence register cites must be proposal-grade.

Defect class, in one sentence: **a sidecar the evidence register cites as a
number's provenance fails the repo's own proposal-grade bar, and nothing
notices** -- every other guard checks that the sidecar *exists*, none what it
says.

It happened. ``results/lshape_adaptive_vs_uniform.run.json``, cited by the
register's adaptive-vs-uniform row, recorded ``git.dirty: true`` and
``config_hash: "unknown"`` from 2026-08-23 until its clean re-record on
2026-09-25, and :func:`~src.research.run_manifest.assert_proposal_grade`
rejected it the whole time (``docs/business/COMMERCIALIZATION_PEER_REVIEW.md``
§5, finding 1) while ``test_evidence_artifacts_carry_run_provenance`` stayed
green, because the file existed.

Scope: every ``.run.json`` beside a ``.csv`` the ``<!-- charter:evidence -->``
region cites, or cited directly -- brace-expanded, because the register writes
``results/x.{csv,run.json}``. A sidecar that does not exist is not this guard's
business: ``test_evidence_artifacts_carry_run_provenance`` owns existence, so
one root cause yields one failure. The region parser is the charter guard's own,
shared through ``tests/support/charter.py``.

Exemptions (:data:`PROPOSAL_GRADE_EXEMPTIONS`) map a sidecar to a reason of at
least :data:`MIN_EXEMPTION_REASON_CHARS` characters and expire in both
directions: an exemption for a sidecar that now passes fails, and so does one
for a sidecar the register no longer cites.

``TestPlantedDefects`` keeps the kills below running in CI: each plants a
defect in a ``tmp_path`` copy (or a monkeypatched exemption table / charter)
and asserts that a named guard in this module raises. No committed file is ever
edited to plant one.

Mutation kills (``harden-a-guard``, 2026-09-25; none of the killers is
``gpu_required`` / ``fem_required``). Kills 1-3 ran the named guard under a
throwaway pytest plugin that monkeypatched this module; kills 4-14 edited this
module or ``tests/support/charter.py`` and restored them byte-for-byte. No
committed artifact was edited for any of them.

1. **The literal historical defect, on the real tree**: the exemption table
   monkeypatched to ``{}`` -> ``test_every_cited_sidecar_is_proposal_grade``
   FAILED naming ``results/lshape_adaptive_vs_uniform.run.json`` with
   ``git.dirty=True`` and ``config_hash='unknown'``.
2. **A tmp copy of a cited sidecar with ``git.dirty: true``** (REPO_ROOT
   monkeypatched to the copy) -> ``test_every_cited_sidecar_is_proposal_grade``.
3. **A stale exemption** for a sidecar that passes ->
   ``test_proposal_grade_exemptions_are_still_needed``.
4. Brace expansion dropped from ``cited_paths`` -> ``test_cited_sidecar_scan_is_not_vacuous``
   (the register cites both sidecars only as ``results/x.{csv,run.json}``).
5. A cited ``.run.json`` pushed through ``manifest_path_for`` (``x.run.run.json``)
   -> ``TestScan::test_a_directly_cited_sidecar_is_itself`` -- the only killer,
   because the brace form still resolves through its ``.csv`` half.
6. The exemption filter inverted -> ``test_every_cited_sidecar_is_proposal_grade``.
7. ``proposal_grade_problem`` swallowing a rejection ->
   ``test_proposal_grade_exemptions_are_still_needed``.
8. ``MIN_CITED_SIDECARS`` lowered to 0 ->
   ``TestPlantedDefects::test_a_register_citing_no_sidecar_turns_the_scan_red``.
9. The still-needed check disabled ->
   ``TestPlantedDefects::test_an_exemption_for_a_passing_sidecar_is_stale``.
10. The still-cited check disabled ->
    ``TestPlantedDefects::test_an_exemption_for_an_uncited_sidecar_is_stale``.
11. The reason-length check disabled ->
    ``TestPlantedDefects::test_a_thin_exemption_reason_turns_the_guard_red``.
12. The moved parser's historical phantom-row bug (a lone header returned as data)
    -> ``test_row_lines_returns_empty_for_header_only_no_separator`` in
    ``tests/docs/test_charter_alignment.py``, the killer it had before the move.
13. URLs / absolute paths accepted as repo paths ->
    ``TestSharedCharterParser::test_looks_like_repo_path``.
14. ``_scan`` hard-wired to the real root instead of REPO_ROOT **survived the
    first draft**: the planted tests still went red, because loading honoured
    REPO_ROOT while existence did not. Killed by
    ``TestPlantedDefects::test_a_root_without_the_sidecars_turns_the_scan_red``,
    written for it.
15. **A real stale exemption.** This guard landed exempting the adaptive
    sidecar until its re-record; once the clean re-record was on disk,
    ``test_proposal_grade_exemptions_are_still_needed`` FAILED on the real tree
    naming it, and the entry was deleted in the same commit as the sidecar.

15 defects (13 planted, 2 observed on the real tree), each killed by a named
test; the test count is larger and is not the number being claimed.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest

from src.research.run_manifest import (
    ProposalGradeError,
    assert_proposal_grade,
    load_run_manifest,
    manifest_path_for,
)
from tests.support.charter import (
    cited_paths,
    expand_braces,
    looks_like_repo_path,
    row_lines,
)

#: Where cited paths resolve. Module state rather than a constant so the planted
#: tests can point the live guards at a ``tmp_path`` copy.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Cited sidecar (repo-relative) -> why it may stay below the proposal-grade bar.
#: Empty: every cited sidecar passes. An entry is a charter deviation and must also
#: be disclosed as one in the charter's deviations register.
PROPOSAL_GRADE_EXEMPTIONS: dict[str, str] = {}

#: An exemption reason shorter than this says nothing a reviewer can check.
MIN_EXEMPTION_REASON_CHARS: Final[int] = 40

#: Vacuity floor: the scan must examine at least this many cited sidecars.
MIN_CITED_SIDECARS: Final[int] = 1

#: The sidecar this guard was written for. The scan must see it -- vocabulary
#: coverage, not merely "something matched" -- because it is cited only through
#: the brace form a naive parser would miss.
ORIGIN_SIDECAR: Final[str] = "results/lshape_adaptive_vs_uniform.run.json"

#: Suffixes that identify an artifact whose sidecar sits beside it, and a sidecar.
ARTIFACT_SUFFIX: Final[str] = ".csv"
SIDECAR_SUFFIX: Final[str] = ".run.json"


@dataclass(frozen=True)
class CitedSidecar:
    """A sidecar the register cites, and the claim that cites it."""

    claim: str
    path: str


def sidecar_for(candidate: str) -> str | None:
    """The sidecar a cited path stands for, or ``None`` if it is neither artifact nor sidecar.

    A cited ``.run.json`` is its own sidecar: pushing it through
    :func:`~src.research.run_manifest.manifest_path_for` would yield
    ``x.run.run.json``, which exists nowhere.
    """
    if candidate.endswith(SIDECAR_SUFFIX):
        return candidate
    if candidate.endswith(ARTIFACT_SUFFIX):
        return manifest_path_for(candidate).as_posix()
    return None


def cited_sidecars(evidence_rows: list[list[str]], root: Path) -> list[CitedSidecar]:
    """Existing sidecars behind the register's citations, in register order, once each."""
    found: dict[str, CitedSidecar] = {}
    for cells in evidence_rows:
        claim = cells[0].strip("`")
        for candidate in cited_paths(cells[-1]):
            sidecar = sidecar_for(candidate)
            if sidecar is None or sidecar in found or not (root / sidecar).is_file():
                continue
            found[sidecar] = CitedSidecar(claim=claim, path=sidecar)
    return list(found.values())


def proposal_grade_problem(root: Path, sidecar: str) -> str | None:
    """Why ``root / sidecar`` is not proposal-grade, or ``None`` when it is."""
    try:
        assert_proposal_grade(load_run_manifest(root / sidecar))
    except ProposalGradeError as exc:
        return str(exc)
    except (OSError, ValueError) as exc:  # unreadable, not JSON, invalid, or newer schema
        return f"cannot be read as a run manifest: {exc}"
    return None


def _scan() -> list[CitedSidecar]:
    """The live scan: the real charter's evidence register, resolved under REPO_ROOT."""
    return cited_sidecars(row_lines("evidence"), REPO_ROOT)


# --------------------------------------------------------------------------------------
# the guards
# --------------------------------------------------------------------------------------


def test_cited_sidecar_scan_is_not_vacuous() -> None:
    """Without this, an empty scan passes every guard below."""
    scanned = _scan()
    assert len(scanned) >= MIN_CITED_SIDECARS, (
        f"the evidence register cites {len(scanned)} existing run sidecar(s); at least "
        f"{MIN_CITED_SIDECARS} expected. Either the citations moved or the parser stopped "
        "resolving them (the register cites sidecars as `results/x.{csv,run.json}`)."
    )
    checked = [s.path for s in scanned if s.path not in PROPOSAL_GRADE_EXEMPTIONS]
    assert checked, (
        "every cited sidecar is exempted, so test_every_cited_sidecar_is_proposal_grade "
        "checks nothing"
    )
    assert ORIGIN_SIDECAR in {s.path for s in scanned}, (
        f"{ORIGIN_SIDECAR} is no longer visible to the scan. If its register row was "
        "deliberately removed, update ORIGIN_SIDECAR; otherwise the parser drifted."
    )


def test_every_cited_sidecar_is_proposal_grade() -> None:
    """A register-cited sidecar must pass ``assert_proposal_grade``, or be exempted."""
    failures = [
        f"{s.claim!r}: {s.path}: {problem}"
        for s in _scan()
        if s.path not in PROPOSAL_GRADE_EXEMPTIONS
        and (problem := proposal_grade_problem(REPO_ROOT, s.path)) is not None
    ]
    assert not failures, (
        "run sidecars the charter's evidence register cites are not proposal-grade:\n  "
        + "\n  ".join(failures)
        + "\n\nRe-record the artifact from a clean tree with its harness's --proposal-grade "
        "(run-provenance + claims-ledger skills), or add it to PROPOSAL_GRADE_EXEMPTIONS "
        "with a reason -- and disclose it as a charter deviation."
    )


def test_proposal_grade_exemptions_state_a_reason() -> None:
    thin = [
        f"{path}: {reason!r}"
        for path, reason in PROPOSAL_GRADE_EXEMPTIONS.items()
        if len(reason.strip()) < MIN_EXEMPTION_REASON_CHARS
    ]
    assert not thin, (
        f"PROPOSAL_GRADE_EXEMPTIONS reasons must be at least {MIN_EXEMPTION_REASON_CHARS} "
        "characters -- say why the sidecar cannot be re-recorded:\n  " + "\n  ".join(thin)
    )


def test_proposal_grade_exemptions_are_still_cited() -> None:
    """An exemption for a sidecar the register no longer cites exempts nothing."""
    cited = {s.path for s in _scan()}
    stale = sorted(path for path in PROPOSAL_GRADE_EXEMPTIONS if path not in cited)
    assert not stale, (
        "PROPOSAL_GRADE_EXEMPTIONS names sidecars the evidence register no longer cites "
        "(or that no longer exist); delete them:\n  " + "\n  ".join(stale)
    )


def test_proposal_grade_exemptions_are_still_needed() -> None:
    """An exemption for a sidecar that now passes is a stale blind spot."""
    stale = sorted(
        path
        for path in PROPOSAL_GRADE_EXEMPTIONS
        if (REPO_ROOT / path).is_file() and proposal_grade_problem(REPO_ROOT, path) is None
    )
    assert not stale, (
        "these exempted sidecars are proposal-grade now; delete their exemptions so the "
        "guard covers them again:\n  " + "\n  ".join(stale)
    )


# --------------------------------------------------------------------------------------
# the scan, on synthetic rows
# --------------------------------------------------------------------------------------


def _touch(root: Path, *relative: str) -> None:
    for path in relative:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("{}", encoding="utf-8")


class TestScan:
    def test_a_brace_citation_resolves_to_its_sidecar_once(self, tmp_path: Path) -> None:
        _touch(tmp_path, "results/x.csv", "results/x.run.json")
        rows = [["claim", "value", "`results/x.{csv,run.json}`"]]
        assert cited_sidecars(rows, tmp_path) == [CitedSidecar("claim", "results/x.run.json")]

    def test_a_bare_csv_citation_reaches_its_sidecar(self, tmp_path: Path) -> None:
        _touch(tmp_path, "results/x.run.json")
        rows = [["claim", "value", "`results/x.csv`"]]
        assert [s.path for s in cited_sidecars(rows, tmp_path)] == ["results/x.run.json"]

    def test_a_directly_cited_sidecar_is_itself(self) -> None:
        assert sidecar_for("results/x.run.json") == "results/x.run.json"
        assert sidecar_for("results/x.csv") == "results/x.run.json"

    @pytest.mark.parametrize("path", ["results/x.png", "specs/x.spec.md", "pyproject.toml"])
    def test_other_artifacts_have_no_sidecar(self, path: str) -> None:
        assert sidecar_for(path) is None

    def test_a_missing_sidecar_is_left_to_the_existence_guard(self, tmp_path: Path) -> None:
        _touch(tmp_path, "results/x.csv")
        assert cited_sidecars([["claim", "value", "`results/x.csv`"]], tmp_path) == []

    def test_prose_and_urls_are_not_citations(self, tmp_path: Path) -> None:
        _touch(tmp_path, "results/x.run.json")
        cell = "`n_simulations=1` `https://x.org/results/x.csv` `/abs/results/x.csv`"
        assert cited_sidecars([["claim", "value", cell]], tmp_path) == []

    def test_a_sidecar_cited_twice_keeps_its_first_claim(self, tmp_path: Path) -> None:
        _touch(tmp_path, "results/x.run.json")
        rows = [["first", "v", "`results/x.csv`"], ["second", "v", "`results/x.run.json`"]]
        assert cited_sidecars(rows, tmp_path) == [CitedSidecar("first", "results/x.run.json")]

    def test_an_unreadable_sidecar_is_a_problem_not_a_crash(self, tmp_path: Path) -> None:
        (tmp_path / "x.run.json").write_text("not json", encoding="utf-8")
        problem = proposal_grade_problem(tmp_path, "x.run.json")
        assert problem is not None
        assert problem.startswith("cannot be read as a run manifest")


class TestSharedCharterParser:
    """Branches of ``tests/support/charter.py`` the live charter does not reach."""

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("results/x.csv", True),
            ("pyproject.toml", True),
            ("/abs/results/x.csv", False),
            ("https://example.org/x.csv", False),
            ("results/../x.csv", False),
            ("n_simulations", False),
        ],
    )
    def test_looks_like_repo_path(self, token: str, expected: bool) -> None:
        assert looks_like_repo_path(token) is expected

    def test_nested_braces_fail_loudly(self) -> None:
        with pytest.raises(AssertionError, match="nested or unbalanced"):
            expand_braces("results/x.{csv,{a,b}}")

    def test_cited_paths_expands_and_filters(self) -> None:
        cell = "`results/x.{csv, run.json}` and `a ratio of 1.5`"
        assert cited_paths(cell) == ["results/x.csv", "results/x.run.json"]

    def test_a_table_without_a_separator_drops_only_its_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = "\n| Claim | Value | Artifact |\n| a | 1 | `x.csv` |\n"
        text = f"<!-- charter:evidence:start -->{body}<!-- charter:evidence:end -->"
        monkeypatch.setattr("tests.support.charter.charter_text", lambda: text)
        assert row_lines("evidence") == [["a", "1", "`x.csv`"]]


# --------------------------------------------------------------------------------------
# planted defects -- each must turn a named guard above red
# --------------------------------------------------------------------------------------

_GUARD_MODULE = sys.modules[__name__]

#: A reason long enough to pass the reason guard, for exemptions planted below.
_PLANTED_REASON: Final[str] = "planted by TestPlantedDefects; never committed " * 2


def _checked_sidecar() -> str:
    """The first live-cited sidecar the guard actually checks (not exempted)."""
    return next(s.path for s in _scan() if s.path not in PROPOSAL_GRADE_EXEMPTIONS)


@pytest.fixture
def planted_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``tmp_path`` copy of every live-cited sidecar, with REPO_ROOT pointed at it."""
    for sidecar in _scan():
        destination = tmp_path / sidecar.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / sidecar.path, destination)
    monkeypatch.setattr(_GUARD_MODULE, "REPO_ROOT", tmp_path)
    return tmp_path


def _plant(path: Path, section: str | None, key: str, value: Any) -> None:
    """Rewrite one field of a copied sidecar."""
    document = json.loads(path.read_text(encoding="utf-8"))
    (document[section] if section else document)[key] = value
    path.write_text(json.dumps(document), encoding="utf-8")


class TestPlantedDefects:
    def test_the_unplanted_copy_passes_every_guard(self, planted_root: Path) -> None:
        """Control: the planted runs below fail because of the plant, not the copy."""
        test_cited_sidecar_scan_is_not_vacuous()
        test_every_cited_sidecar_is_proposal_grade()
        test_proposal_grade_exemptions_are_still_cited()
        test_proposal_grade_exemptions_are_still_needed()

    @pytest.mark.parametrize(
        ("section", "key", "value", "problem"),
        [
            ("git", "dirty", True, "git.dirty=True"),
            ("git", "dirty", None, "git.dirty=None"),
            (None, "config_hash", "unknown", "config_hash='unknown'"),
            (None, "config_hash", "", "config_hash=''"),
        ],
        ids=["dirty", "undetermined", "unknown-hash", "empty-hash"],
    )
    def test_a_planted_sidecar_turns_the_guard_red(
        self, planted_root: Path, section: str | None, key: str, value: Any, problem: str
    ) -> None:
        target = _checked_sidecar()
        _plant(planted_root / target, section, key, value)
        with pytest.raises(AssertionError, match=re.escape(f"{target}: {problem}")):
            test_every_cited_sidecar_is_proposal_grade()

    def test_an_unreadable_sidecar_turns_the_guard_red(self, planted_root: Path) -> None:
        target = _checked_sidecar()
        (planted_root / target).write_text("not json", encoding="utf-8")
        with pytest.raises(AssertionError, match=re.escape(f"{target}: cannot be read")):
            test_every_cited_sidecar_is_proposal_grade()

    def test_an_exemption_really_exempts(
        self, planted_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = _checked_sidecar()
        _plant(planted_root / target, "git", "dirty", True)
        exempt = {**PROPOSAL_GRADE_EXEMPTIONS, target: _PLANTED_REASON}
        monkeypatch.setattr(_GUARD_MODULE, "PROPOSAL_GRADE_EXEMPTIONS", exempt)
        test_every_cited_sidecar_is_proposal_grade()
        test_proposal_grade_exemptions_are_still_needed()

    def test_an_exemption_for_a_passing_sidecar_is_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = _checked_sidecar()
        exempt = {**PROPOSAL_GRADE_EXEMPTIONS, target: _PLANTED_REASON}
        monkeypatch.setattr(_GUARD_MODULE, "PROPOSAL_GRADE_EXEMPTIONS", exempt)
        with pytest.raises(AssertionError, match=re.escape(target)):
            test_proposal_grade_exemptions_are_still_needed()

    def test_an_exemption_for_an_uncited_sidecar_is_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exempt = {**PROPOSAL_GRADE_EXEMPTIONS, "results/never_cited.run.json": _PLANTED_REASON}
        monkeypatch.setattr(_GUARD_MODULE, "PROPOSAL_GRADE_EXEMPTIONS", exempt)
        with pytest.raises(AssertionError, match="never_cited"):
            test_proposal_grade_exemptions_are_still_cited()

    def test_a_thin_exemption_reason_turns_the_guard_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thin = "x" * (MIN_EXEMPTION_REASON_CHARS - 1)
        exempt = {**PROPOSAL_GRADE_EXEMPTIONS, _checked_sidecar(): thin}
        monkeypatch.setattr(_GUARD_MODULE, "PROPOSAL_GRADE_EXEMPTIONS", exempt)
        with pytest.raises(AssertionError, match="at least"):
            test_proposal_grade_exemptions_state_a_reason()

    def test_a_register_citing_no_sidecar_turns_the_scan_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = "\n| Claim | Value | Artifact |\n| --- | --- | --- |\n| c | 1 | `pyproject.toml` |\n"
        text = f"<!-- charter:evidence:start -->{body}<!-- charter:evidence:end -->"
        monkeypatch.setattr("tests.support.charter.charter_text", lambda: text)
        with pytest.raises(AssertionError, match="cites 0 existing run sidecar"):
            test_cited_sidecar_scan_is_not_vacuous()

    def test_a_root_without_the_sidecars_turns_the_scan_red(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scan resolves under REPO_ROOT -- which every planted test above relies on."""
        monkeypatch.setattr(_GUARD_MODULE, "REPO_ROOT", tmp_path)
        with pytest.raises(AssertionError, match="cites 0 existing run sidecar"):
            test_cited_sidecar_scan_is_not_vacuous()

    def test_exempting_every_sidecar_turns_the_scan_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exempt = {s.path: _PLANTED_REASON for s in _scan()}
        monkeypatch.setattr(_GUARD_MODULE, "PROPOSAL_GRADE_EXEMPTIONS", exempt)
        with pytest.raises(AssertionError, match="checks nothing"):
            test_cited_sidecar_scan_is_not_vacuous()
