"""Hermetic, shared access to the project charter's machine-readable regions.

The charter (``openspec/specs/project-charter/spec.md``) exposes six delimited
regions -- ``<!-- charter:<name>:start -->`` ... ``<!-- charter:<name>:end -->``
-- each holding one Markdown table. More than one guard reads them:
``tests/docs/test_charter_alignment.py`` enforces every Requirement, and
``tests/docs/test_proposal_grade_sidecars.py`` holds the run sidecars the
evidence register cites to the proposal-grade bar. Two parsers that must agree
are two that will eventually disagree -- the lesson ``tests/support/workflows.py``
and ``tests/support/import_graph.py`` were extracted for -- so the parser lives
here once, moved out of the charter guard rather than copied from it.

The AMR policy-ratio *subject scan* (:func:`amr_policy_ratio_subjects`) lives here
for the same reason: the charter guard's manifest-pointer check and
``tests/docs/test_lookahead_attribution.py`` must read the same claims, from the
evidence register and from ``README.md``.

Stdlib-only and side-effect free: it reads Markdown files. The charter guard
relies on that to keep its own import surface stdlib-only.

The readers resolve :func:`charter_text` and :func:`readme_text` through this
module's namespace at call time, so a test can substitute a synthetic document
with ``monkeypatch.setattr("tests.support.charter.charter_text", ...)``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

#: Repository root, resolved from this file's location (``tests/support/``).
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: The supreme scope document.
CHARTER: Final[Path] = REPO_ROOT / "openspec" / "specs" / "project-charter" / "spec.md"

#: The other surface the AMR policy-ratio guards read.
README: Final[Path] = REPO_ROOT / "README.md"

#: Delimited regions the charter exposes for machine reading.
REGIONS: Final[tuple[str, ...]] = (
    "scope",
    "non-goals",
    "evidence",
    "capabilities",
    "gates",
    "deviations",
)

#: Fenced blocks are stripped before parsing so an illustrative table inside ```
#: fences is never mistaken for a real register. Same idiom as
#: ``scripts/check_doc_links.py``.
_FENCED: Final[re.Pattern[str]] = re.compile(r"```.*?```", re.DOTALL)

#: A Markdown table separator cell: ``---``, ``:---``, ``---:``, ``:---:``.
_SEPARATOR_CELL: Final[re.Pattern[str]] = re.compile(r"^:?-{3,}:?$")

#: An inline code span -- the form every artifact citation takes.
_CODE_SPAN: Final[re.Pattern[str]] = re.compile(r"`([^`]+)`")

#: A trailing file extension, the second way a token can name a path.
_EXTENSION: Final[re.Pattern[str]] = re.compile(r"\.[a-z]{2,5}$")

#: A single ``{a,b}`` brace group.
_BRACE_GROUP: Final[re.Pattern[str]] = re.compile(r"\{([^{}]*)\}")

#: Text that marks an MCTS-vs-Dörfler statement as quoting a policy *ratio*.
AMR_RATIO_HINT: Final[re.Pattern[str]] = re.compile(
    r"median\s+ratio\s+\d+\.\d+|ratio\s+\d+\.\d+|l2_error_ratio_at_matched_dof|"
    r"\b\d+\.\d{3,}\b",
    re.IGNORECASE,
)

#: Suffix of a cited tabular artifact.
CSV_SUFFIX: Final[str] = ".csv"


def charter_text() -> str:
    """The charter's full text."""
    assert CHARTER.exists(), f"the charter is missing: {CHARTER}"
    return CHARTER.read_text(encoding="utf-8")


def readme_text() -> str:
    """``README.md``'s full text."""
    return README.read_text(encoding="utf-8")


def region(name: str) -> str:
    """Return the body of a ``<!-- charter:<name>:start -->`` region, fences stripped."""
    text = charter_text()
    start, end = f"<!-- charter:{name}:start -->", f"<!-- charter:{name}:end -->"
    assert text.count(start) == 1, f"{start} must appear exactly once in {CHARTER.name}"
    assert text.count(end) == 1, f"{end} must appear exactly once in {CHARTER.name}"
    body = text.split(start, 1)[1].split(end, 1)[0]
    return _FENCED.sub("", body)


def row_lines(name: str) -> list[list[str]]:
    """Data rows of the Markdown table in a region, as lists of cells.

    Header and ``| --- |`` separator rows are dropped structurally rather than
    by requiring a backticked first cell: two of the six registers (evidence,
    deviations) key on a prose claim/deviation label, and a parser that silently
    skipped them would make their guards vacuous -- exactly the failure the
    meta-guard exists to catch.
    """
    table: list[list[str]] = []
    for line in region(name).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        table.append([c.strip() for c in stripped.strip("|").split("|")])

    for index, cells in enumerate(table):
        if cells and all(_SEPARATOR_CELL.match(c) for c in cells if c):
            return table[index + 1 :]
    # No separator found: the first row is a header, never data. A table of only
    # a header (all real rows deleted) must return [] here, not `table` --
    # returning the header row itself would let the header text pass as a phantom
    # data row. This matters most for the evidence/deviations regions, which have
    # no external cross-check to catch a phantom row the way scope/non-goals/
    # capabilities/gates would (each of those diffs against an external source of
    # truth and would flag a phantom row as an "extra" entry regardless of its
    # wording).
    return table[1:] if len(table) > 1 else []


def rows(name: str) -> list[str]:
    """First-cell token of every data row in a region, backticks stripped."""
    return [cells[0].strip("`") for cells in row_lines(name) if cells and cells[0]]


def expand_braces(token: str) -> list[str]:
    """Expand a single ``{a,b}`` group -- the form specs use for ``results/x.{csv,png}``."""
    match = _BRACE_GROUP.search(token)
    if match is None:
        return [token]
    expanded = [
        token[: match.start()] + option.strip() + token[match.end() :]
        for option in match.group(1).split(",")
    ]
    # Single level only: anything left over is malformed and must fail loudly,
    # never be skipped.
    for candidate in expanded:
        assert "{" not in candidate and "}" not in candidate, (
            f"nested or unbalanced brace expansion in charter citation {token!r}; "
            "only a single {a,b} group is supported"
        )
    return expanded


def looks_like_repo_path(token: str) -> bool:
    """True for a token that names an in-repo path (not prose, not a URL, not a metric)."""
    if token.startswith(("/", "http://", "https://")) or ".." in token:
        return False
    return "/" in token or bool(_EXTENSION.search(token))


def cited_paths(cell: str) -> list[str]:
    """Every in-repo path a table cell cites, brace-expanded, in citation order.

    Citations are inline code spans; spans that are prose, URLs or metrics are
    skipped, and ``results/x.{csv,run.json}`` yields both paths.
    """
    return [
        candidate
        for token in _CODE_SPAN.findall(cell)
        if looks_like_repo_path(token)
        for candidate in expand_braces(token)
    ]


def csv_citations_in(text: str) -> list[str]:
    """Every ``.csv`` path ``text`` cites, brace-expanded, in citation order."""
    return [path for path in cited_paths(text) if path.endswith(CSV_SUFFIX)]


def mentions_mcts_and_dorfler(text: str) -> bool:
    """True when ``text``'s prose names both MCTS and Dörfler.

    Inline code spans are stripped first, so a path such as
    ``results/lshape_mcts_vs_dorfler.csv`` cannot satisfy the Dörfler vocabulary
    by itself (the umlaut lives in the claim prose).
    """
    prose = _CODE_SPAN.sub(" ", text).lower()
    return "mcts" in prose and ("dörfler" in prose or "dorfler" in prose)


def amr_policy_ratio_subjects() -> list[tuple[str, str]]:
    """(source, body) pairs that look like an MCTS-vs-Dörfler policy ratio claim."""
    subjects: list[tuple[str, str]] = []
    for cells in row_lines("evidence"):
        body = " | ".join(cells)
        if mentions_mcts_and_dorfler(body) and AMR_RATIO_HINT.search(body):
            subjects.append((f"charter evidence:{cells[0]}", body))
    for line in readme_text().splitlines():
        if mentions_mcts_and_dorfler(line) and AMR_RATIO_HINT.search(line):
            subjects.append(("README.md", line))
    return subjects
