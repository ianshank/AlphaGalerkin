"""Executable guards for the project charter.

The charter (``openspec/specs/project-charter/spec.md``) is the repository's supreme scope
document. A charter that is only prose is a wish: this module makes each of its Requirements
checkable, so documented scope and claims cannot silently diverge from code reality.

It exists because they did. The fabricated ``0.000209 / 240x`` transfer figure survived for
months with no artifact behind it; after that retraction the same failure mode recurred twice
more — a retracted AMR headline and an overstated transfer MSE sourced from an uncommitted
spike, both contradicted by artifacts committed in this very repository.

Design notes (see ``openspec/changes/project-charter-alignment/design.md``):

* **Referential, not duplicative.** The scope guard compares the charter to
  ``ARCHITECTURE.md``'s package map rather than to disk, because
  ``tests/docs/test_architecture_map.py`` already binds that map to disk. One root cause
  produces one failure, not two.
* **Registry truth is read in a subprocess.** ``ScenarioRegistry`` is a process-wide singleton
  and nine ``tests/poc/*`` modules ``clear()`` it in autouse fixtures without teardown, which
  makes an in-process read *order-dependent*. Measured, not assumed::

      pytest tests/poc tests/docs                      # in-process read sees 10 (recovers)
      pytest tests/poc/test_complexity_scenario.py \
             tests/poc/test_registry.py <probe>        # in-process read sees 0

  Whether it recovers depends on whether a later ``tests/poc`` module happens to re-import and
  re-register. That latent, selection-dependent behaviour is worse than a consistent failure —
  it would go green in CI and red for a contributor running a subset. A subprocess is hermetic
  and additionally keeps this module's import stdlib-only.
* **Non-goals are checked by directory existence only.** Substring-grepping ``src/`` for the cut
  module names produces false positives against live vocabulary (``vertex`` in
  ``mesh_refinement.py``, ``intercept`` in ``research/``).
* **Nothing consults git history.** CI checks out shallow with no tags.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.cut_modules import CUT_MODULES, FABRICATED_FIGURE, RETRACTED_BLANKET_CLAIM

REPO_ROOT = Path(__file__).resolve().parents[2]
CHARTER = REPO_ROOT / "openspec" / "specs" / "project-charter" / "spec.md"
ARCHITECTURE = REPO_ROOT / "ARCHITECTURE.md"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
SRC = REPO_ROOT / "src"

# Delimited regions the charter exposes for machine reading.
_REGIONS = ("scope", "non-goals", "evidence", "capabilities", "gates", "deviations")

# Fenced blocks are stripped before parsing so an illustrative table inside ``` fences is
# never mistaken for a real register. Same idiom as scripts/check_doc_links.py.
_FENCED = re.compile(r"```.*?```", re.DOTALL)

# A markdown table separator cell: ---, :---, ---:, :---:
_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")

# Words that mark a mention of a retracted claim as a *retraction* rather than a live claim.
# The charter must be able to describe the history that motivated it.
_RETRACTION_MARKERS = ("fabricat", "retract", "correct", "must not", "no longer", "inadmissible")

# ARCHITECTURE.md's own package-map region.
_ARCH_ROW_PACKAGE = re.compile(r"\|\s*`src/([a-z0-9_]+)/`\s*\|")

# ``### Requirement: Name`` — anchored so it cannot match ``#### Scenario:``.
_REQUIREMENT_HEADING = re.compile(r"^### Requirement:\s*(.+?)\s*$", re.MULTILINE)

# Every charter Requirement maps to the guard that enforces it. Both directions are checked,
# so renaming a Requirement without renaming its guard fails loudly rather than silently
# leaving the Requirement unenforced.
_GUARDED: dict[str, str] = {
    "Scope Integrity": "test_scope_register_matches_architecture_map",
    "Non-Goal Exclusion": "test_non_goal_packages_do_not_exist",
    "Evidence-Backed Claims": "test_evidence_claims_cite_existing_artifacts",
    "Novelty Claim Discipline": "test_charter_free_of_retracted_claims",
    "Capability Register Accuracy": "test_capability_register_matches_scenario_registry",
    "Quality Gate Fidelity": "test_documented_gates_are_enforced_in_ci",
    "Accepted Deviation Disclosure": "test_accepted_deviations_state_a_reason",
}

# Reason cells that are present but say nothing.
_EMPTY_REASONS = frozenset({"", "-", "--", "tbd", "n/a", "na", "?", "todo"})
_MIN_REASON_CHARS = 20

# R5's subprocess enumerates the scenario registry by importing src.poc.scenarios, which
# imports torch — generous but bounded, so a genuinely hung import fails the guard rather
# than the test session.
_REGISTRY_SUBPROCESS_TIMEOUT_S = 300


# --------------------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------------------


def _charter_text() -> str:
    assert CHARTER.exists(), f"the charter is missing: {CHARTER}"
    return CHARTER.read_text(encoding="utf-8")


def _region(name: str) -> str:
    """Return the body of a ``<!-- charter:<name>:start -->`` region, fences stripped."""
    text = _charter_text()
    start, end = f"<!-- charter:{name}:start -->", f"<!-- charter:{name}:end -->"
    assert text.count(start) == 1, f"{start} must appear exactly once in {CHARTER.name}"
    assert text.count(end) == 1, f"{end} must appear exactly once in {CHARTER.name}"
    body = text.split(start, 1)[1].split(end, 1)[0]
    return _FENCED.sub("", body)


def _row_lines(name: str) -> list[list[str]]:
    """Data rows of the markdown table in a region, as lists of cells.

    Header and `| --- |` separator rows are dropped structurally rather than by requiring a
    backticked first cell: two of the six registers (evidence, deviations) key on a prose
    claim/deviation label, and a parser that silently skipped them would make their guards
    vacuous — exactly the failure the meta-guard exists to catch.
    """
    table: list[list[str]] = []
    for line in _region(name).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        table.append([c.strip() for c in stripped.strip("|").split("|")])

    for index, cells in enumerate(table):
        if cells and all(_SEPARATOR_CELL.match(c) for c in cells if c):
            return table[index + 1 :]
    # No separator found: the first row is a header, never data. A table of only a header
    # (all real rows deleted) must return [] here, not `table` — returning the header row
    # itself would let the header text pass as a phantom data row. This matters most for the
    # evidence/deviations regions, which have no external cross-check to catch a phantom row
    # the way scope/non-goals/capabilities/gates would (each of those diffs against an
    # external source of truth and would flag a phantom row as an "extra" entry regardless of
    # its wording).
    return table[1:] if len(table) > 1 else []


def _rows(name: str) -> list[str]:
    """First-cell token of every data row in a region, backticks stripped."""
    return [cells[0].strip("`") for cells in _row_lines(name) if cells and cells[0]]


def _expand_braces(token: str) -> list[str]:
    """Expand a single ``{a,b}`` group — the form specs use for ``results/x.{csv,png}``."""
    match = re.search(r"\{([^{}]*)\}", token)
    if match is None:
        return [token]
    expanded = [
        token[: match.start()] + option.strip() + token[match.end() :]
        for option in match.group(1).split(",")
    ]
    # Single level only: anything left over is malformed and must fail loudly, never be skipped.
    for candidate in expanded:
        assert "{" not in candidate and "}" not in candidate, (
            f"nested or unbalanced brace expansion in charter citation {token!r}; "
            "only a single {a,b} group is supported"
        )
    return expanded


def _looks_like_repo_path(token: str) -> bool:
    """True for a token that names an in-repo path (not prose, not a URL, not a metric)."""
    if token.startswith(("/", "http://", "https://")) or ".." in token:
        return False
    return "/" in token or bool(re.search(r"\.[a-z]{2,5}$", token))


# --------------------------------------------------------------------------------------
# parser unit tests — exercise _row_lines directly, independent of the live charter content
# --------------------------------------------------------------------------------------


def _with_region(monkeypatch: pytest.MonkeyPatch, name: str, body: str) -> None:
    """Make ``_charter_text()`` return a synthetic single-region document for this test."""
    text = f"<!-- charter:{name}:start -->{body}<!-- charter:{name}:end -->"
    monkeypatch.setattr("tests.docs.test_charter_alignment._charter_text", lambda: text)


def test_row_lines_handles_normal_table(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_region(
        monkeypatch,
        "scope",
        "\n| Package | Domain |\n| --- | --- |\n| `src/mcts/` | shared |\n",
    )
    assert _rows("scope") == ["src/mcts/"]


def test_row_lines_returns_empty_for_header_only_no_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real bug caught in review.

    A table reduced to just its header line — every real row deleted, and the ``| --- |``
    separator deleted along with them — must parse to zero rows, not the header cells
    reinterpreted as a phantom data row. The header/separator/no-separator branches in
    ``_row_lines`` all funnel through the same ``len(table) <= 1`` case, and an earlier
    version returned ``table`` (the header itself) instead of ``[]`` there. That silently
    passed R7 (Accepted Deviation Disclosure) with every real deviation deleted, because R7
    has no external source of truth to cross-check against — unlike scope/non-goals/
    capabilities/gates, which would catch a phantom row as an unexpected "extra" regardless
    of its wording.
    """
    _with_region(monkeypatch, "deviations", "\n| Deviation | Explanation of divergence |\n")
    assert _rows("deviations") == []


def test_row_lines_returns_empty_for_a_wholly_empty_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_region(monkeypatch, "deviations", "\n")
    assert _rows("deviations") == []


# --------------------------------------------------------------------------------------
# meta-guards — a docs guard that parses nothing passes everything
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _REGIONS)
def test_charter_regions_parse_non_empty(name: str) -> None:
    assert _rows(name), (
        f"charter region 'charter:{name}' parsed to zero rows. Either the markers moved or the "
        "table format changed; a region that parses empty would make its guard vacuous."
    )


def test_every_requirement_has_a_guard() -> None:
    documented = set(_REQUIREMENT_HEADING.findall(_charter_text()))
    assert documented, "no '### Requirement:' headings parsed from the charter"

    unguarded = documented - set(_GUARDED)
    orphaned = set(_GUARDED) - documented
    assert not unguarded, (
        f"charter Requirements with no guard test: {sorted(unguarded)}. Add one to "
        "tests/docs/test_charter_alignment.py and register it in _GUARDED."
    )
    assert not orphaned, (
        f"_GUARDED names Requirements that are not in the charter: {sorted(orphaned)}. "
        "A Requirement was renamed or removed without updating its guard."
    )


def test_requirements_use_rfc2119_and_declare_scenarios() -> None:
    text = _charter_text()
    blocks = re.split(r"^### Requirement:", text, flags=re.MULTILINE)[1:]
    failures = [
        block.splitlines()[0].strip()
        for block in blocks
        if "SHALL" not in block or "#### Scenario:" not in block
    ]
    assert not failures, (
        f"charter Requirements missing a SHALL clause or a '#### Scenario:' block: {failures}"
    )


# --------------------------------------------------------------------------------------
# R1 — Scope Integrity
# --------------------------------------------------------------------------------------


def test_scope_register_matches_architecture_map() -> None:
    architecture = ARCHITECTURE.read_text(encoding="utf-8")
    region = architecture.split("<!-- package-map:start -->", 1)[1].split(
        "<!-- package-map:end -->", 1
    )[0]
    documented = set(_ARCH_ROW_PACKAGE.findall(region))

    charter_rows = [row.strip("/") for row in _rows("scope")]
    duplicates = {name for name in charter_rows if charter_rows.count(name) > 1}
    assert not duplicates, f"charter scope register lists packages more than once: {duplicates}"

    charter_packages = {name.removeprefix("src/") for name in charter_rows}

    missing = documented - charter_packages
    extra = charter_packages - documented
    assert not missing, (
        f"ARCHITECTURE.md documents src/ packages the charter's scope register omits: "
        f"{sorted(missing)}."
    )
    assert not extra, (
        f"the charter's scope register names packages ARCHITECTURE.md does not: {sorted(extra)}."
    )

    on_disk = {p.parent.name for p in SRC.glob("*/__init__.py")}
    phantom = charter_packages - on_disk
    assert not phantom, (
        f"the charter's scope register names packages with no src/<name>/__init__.py: "
        f"{sorted(phantom)}."
    )


# --------------------------------------------------------------------------------------
# R2 — Non-Goal Exclusion
# --------------------------------------------------------------------------------------


def test_non_goal_packages_do_not_exist() -> None:
    resurrected = sorted(name for name in CUT_MODULES if (SRC / name).is_dir())
    assert not resurrected, (
        f"packages cut on 2026-07-22 have reappeared under src/: {resurrected}. Re-adding one is "
        "a scope change — amend the charter's non-goal register first."
    )


def test_non_goal_register_covers_every_cut_module() -> None:
    registered = {row.strip("/").removeprefix("src/") for row in _rows("non-goals")}
    missing = set(CUT_MODULES) - registered
    assert not missing, (
        f"tests/support/cut_modules.py lists modules the charter's non-goal register omits: "
        f"{sorted(missing)}."
    )


# --------------------------------------------------------------------------------------
# R3 — Evidence-Backed Claims
# --------------------------------------------------------------------------------------


def test_evidence_claims_cite_existing_artifacts() -> None:
    failures: list[str] = []
    for cells in _row_lines("evidence"):
        claim = cells[0].strip("`")
        artifact_cell = cells[-1]
        citations = re.findall(r"`([^`]+)`", artifact_cell)
        if not citations:
            failures.append(f"{claim!r}: no artifact cited")
            continue
        for citation in citations:
            if not _looks_like_repo_path(citation):
                continue
            for candidate in _expand_braces(citation):
                if not (REPO_ROOT / candidate).exists():
                    failures.append(f"{claim!r}: cited artifact does not exist: {candidate}")
    assert not failures, "charter claims citing missing artifacts:\n  " + "\n  ".join(
        sorted(failures)
    )


# --------------------------------------------------------------------------------------
# R4 — Novelty Claim Discipline
# --------------------------------------------------------------------------------------


def _unretracted_mentions(needle: str) -> list[str]:
    """Lines citing ``needle`` without marking it as retracted.

    The charter must be free to *describe* the claims it retracts — that history is why it
    exists — so a bare substring ban would be unusable. What it may not do is state one as a
    live claim.
    """
    return [
        line.strip()
        for line in _charter_text().splitlines()
        if needle.lower() in line.lower()
        and not any(marker in line.lower() for marker in _RETRACTION_MARKERS)
    ]


def test_charter_free_of_retracted_claims() -> None:
    live_figure = _unretracted_mentions(FABRICATED_FIGURE)
    assert not live_figure, (
        f"the charter cites the fabricated transfer figure {FABRICATED_FIGURE!r} without marking "
        "it retracted; the committed result is ~2.3e-3 "
        "(results/transfer_baseline_compare.csv):\n  " + "\n  ".join(live_figure)
    )
    live_claim = _unretracted_mentions(RETRACTED_BLANKET_CLAIM)
    assert not live_claim, (
        "the charter states the retracted blanket 'no MCTS+Galerkin' claim as live; TreeMesh "
        "(arXiv:2111.07613) falsifies it. Use the narrow multi-step-look-ahead form:\n  "
        + "\n  ".join(live_claim)
    )


def test_retracted_figure_absent_from_the_evidence_register() -> None:
    """Belt and braces: whatever the prose says, no *claim row* may cite the fabricated figure."""
    offending = [
        cells[0] for cells in _row_lines("evidence") if FABRICATED_FIGURE in " | ".join(cells)
    ]
    assert not offending, (
        f"the charter's evidence register cites {FABRICATED_FIGURE!r} as a live claim: {offending}"
    )


# --------------------------------------------------------------------------------------
# R5 — Capability Register Accuracy
# --------------------------------------------------------------------------------------


def _registered_scenarios() -> set[str]:
    """Enumerate the scenario registry in a subprocess.

    In-process is not safe: ``tests/poc/*`` autouse fixtures ``clear()`` the singleton without
    restoring it and purge ``src.poc.scenarios*`` from ``sys.modules``. Whether a later
    in-process read sees 10 scenarios or 0 depends on which of those modules ran — see the
    measured orderings in this module's docstring.
    """
    code = (
        "import json, src.poc.scenarios;"
        "from src.poc.registry import ScenarioRegistry;"
        "print(json.dumps(sorted(ScenarioRegistry().list_scenarios())))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=_REGISTRY_SUBPROCESS_TIMEOUT_S,
    )
    if proc.returncode != 0:
        pytest.fail(f"could not enumerate the scenario registry:\n{proc.stderr[-2000:]}")
    import json

    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


def test_capability_register_matches_scenario_registry() -> None:
    documented = {row.strip("`") for row in _rows("capabilities")}
    registered = _registered_scenarios()

    missing = registered - documented
    extra = documented - registered
    assert not missing, (
        f"scenarios registered at runtime but absent from the charter's capability register: "
        f"{sorted(missing)}."
    )
    assert not extra, (
        f"the charter's capability register names scenarios that are not registered: "
        f"{sorted(extra)}."
    )


# --------------------------------------------------------------------------------------
# R6 — Quality Gate Fidelity
# --------------------------------------------------------------------------------------


def test_documented_gates_are_enforced_in_ci() -> None:
    """Charter gates must exist in CI at the stated value (charter subset of CI).

    The reverse direction is deliberately unchecked: adding a CI gate should not nag a charter
    edit. Matching is scoped to the CI *step* that mentions the target, because four gates use
    the native ``coverage report --fail-under=`` form with no ``--cov=`` flag at all, so a
    structural ``--cov=X`` -> ``--cov-fail-under=N`` pairing would silently miss them.
    """
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    steps = re.split(r"^      - name: ", workflow, flags=re.MULTILINE)

    failures: list[str] = []
    for cells in _row_lines("gates"):
        target = cells[0].strip("`")
        expected = cells[1].strip("`")
        relevant = [s for s in steps if f"--cov={target} " in s or f"--cov={target}\n" in s]
        if not relevant:
            failures.append(f"{target}: no CI step gates it")
            continue
        if not any(f"--cov-fail-under={expected}" in s for s in relevant):
            failures.append(f"{target}: charter says {expected}, no CI step gates at that value")
    assert not failures, "charter coverage gates not enforced in CI:\n  " + "\n  ".join(failures)


# --------------------------------------------------------------------------------------
# R7 — Accepted Deviation Disclosure
# --------------------------------------------------------------------------------------


def test_accepted_deviations_state_a_reason() -> None:
    failures: list[str] = []
    for cells in _row_lines("deviations"):
        deviation = cells[0].strip("`")
        reason = cells[-1] if len(cells) > 1 else ""
        if reason.strip().lower() in _EMPTY_REASONS or len(reason.strip()) < _MIN_REASON_CHARS:
            failures.append(f"{deviation!r}: reason is missing or too thin ({reason!r})")
    assert not failures, (
        "accepted deviations must state why the divergence is deliberate; an undisclosed "
        "deviation is indistinguishable from drift:\n  " + "\n  ".join(failures)
    )
