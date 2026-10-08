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

from tests.support.perf_claims import split_blocks

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

#: Suffix of the run-provenance sidecar written beside an artifact.
SIDECAR_SUFFIX: Final[str] = ".run.json"

#: The disclosure an AMR policy-ratio claim must carry when it cites a run whose
#: search made no decision single-element greedy marking would not have made
#: (``tests/docs/test_lookahead_attribution.py``). Defined once; the charter,
#: README and the guard all use this exact phrase.
NO_LOOKAHEAD_LABEL: Final[str] = "search contributed no decisions"

#: Source prefix of a subject read from the charter's evidence register.
EVIDENCE_SOURCE_PREFIX: Final[str] = "charter evidence:"

#: Source prefix of a subject read from ``README.md`` (followed by its first line).
README_SOURCE_PREFIX: Final[str] = f"{README.name}:"

#: The ``Block.kind`` ``tests/support/perf_claims.py::split_blocks`` gives a table.
TABLE_BLOCK_KIND: Final[str] = "table"

#: Markdown emphasis characters, ignored when looking for the label.
_EMPHASIS: Final[re.Pattern[str]] = re.compile(r"[*_]")

#: Any run of whitespace, collapsed when looking for the label.
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


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


def cited_sidecars_in(text: str) -> list[str]:
    """The run sidecars ``text`` cites, once each, in citation order.

    A cited ``.csv`` stands for its sibling ``<stem>.run.json``; a cited
    ``.run.json`` is itself. So ``results/x.{csv,run.json}``, ``results/x.csv``
    and ``results/x.run.json`` all reach the same sidecar.
    """
    found: list[str] = []
    for path in cited_paths(text):
        if path.endswith(SIDECAR_SUFFIX):
            sidecar = path
        elif path.endswith(CSV_SUFFIX):
            sidecar = path.removesuffix(CSV_SUFFIX) + SIDECAR_SUFFIX
        else:
            continue
        if sidecar not in found:
            found.append(sidecar)
    return found


def carries_no_lookahead_label(text: str) -> bool:
    """Whether ``text`` states :data:`NO_LOOKAHEAD_LABEL`.

    Case, emphasis and line wrapping are ignored: a label wrapped across two
    Markdown lines, or set in bold, is still the label.
    """
    flat = _WHITESPACE.sub(" ", _EMPHASIS.sub("", text)).lower()
    return NO_LOOKAHEAD_LABEL.lower() in flat


def mentions_mcts_and_dorfler(text: str) -> bool:
    """True when ``text``'s prose names both MCTS and Dörfler.

    Inline code spans are stripped first, so a path such as
    ``results/lshape_mcts_vs_dorfler.csv`` cannot satisfy the Dörfler vocabulary
    by itself (the umlaut lives in the claim prose).
    """
    prose = _CODE_SPAN.sub(" ", text).lower()
    return "mcts" in prose and ("dörfler" in prose or "dorfler" in prose)


def is_amr_policy_ratio_claim(text: str) -> bool:
    """Whether ``text`` states an MCTS-vs-Dörfler policy ratio."""
    return mentions_mcts_and_dorfler(text) and AMR_RATIO_HINT.search(text) is not None


def markdown_units(markdown: str) -> list[tuple[int, str]]:
    """``(first line number, text)`` units a Markdown claim is read in, in order.

    A paragraph or a list item is one unit, continuation lines included: a claim
    wrapped across lines is still one claim, and reading it line by line misses
    it when no single line names both arms and a ratio. A table row is its own
    unit, like a charter register row. Fenced code is not prose, but each fenced
    line stays a unit of its own so a claim cannot hide in a code block.
    """
    units: list[tuple[int, str]] = []
    covered: set[int] = set()
    for block in split_blocks(markdown):
        covered.update(line_no for line_no, _ in block.lines)
        if block.kind == TABLE_BLOCK_KIND:
            units.extend(block.lines)
        else:
            units.append((block.lines[0][0], block.text))
    units.extend(
        (line_no, line)
        for line_no, line in enumerate(markdown.splitlines(), start=1)
        if line_no not in covered and line.strip()
    )
    return sorted(units)


def amr_policy_ratio_subjects() -> list[tuple[str, str]]:
    """(source, body) pairs that look like an MCTS-vs-Dörfler policy ratio claim.

    Charter evidence rows are read whole (cells joined); ``README.md`` is read in
    :func:`markdown_units`, each source naming the unit's first line.
    """
    subjects: list[tuple[str, str]] = []
    for cells in row_lines("evidence"):
        body = " | ".join(cells)
        if is_amr_policy_ratio_claim(body):
            subjects.append((f"{EVIDENCE_SOURCE_PREFIX}{cells[0]}", body))
    for line_no, body in markdown_units(readme_text()):
        if is_amr_policy_ratio_claim(body):
            subjects.append((f"{README_SOURCE_PREFIX}{line_no}", body))
    return subjects
