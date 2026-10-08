"""Unit-bearing performance numbers in Markdown, and whether an artifact backs them.

The charter's *Evidence-Backed Claims* Requirement says every numeric headline claim cites a
committed artifact. The only guard over ``README.md`` used to cover MCTS-vs-Dörfler policy
ratios, so ``README.md`` could show inference latency and MCTS simulations per second for three
models "on NVIDIA RTX 3090" -- a card none of the documented rigs use -- with no artifact, run
output or sidecar behind a single figure (``docs/business/COMMERCIALIZATION_PEER_REVIEW.md``
§5.2). This module reads Markdown as data and finds such numbers; the guard that applies it to
the repository is ``tests/docs/test_performance_claims.py``.

A **performance number** is a hardware-dependent measurement:

* *latency* -- a number followed by ``ms``, ``µs``, ``us`` or ``ns``; or by ``s`` / ``sec`` /
  ``seconds`` when the same sentence names a measured quantity (:data:`MEASURED_QUANTITY_WORDS`
  or ``per <move|step|...>``), so "takes ~2 s" and a timeout are not claims;
* *throughput* -- a number followed by ``[noun]/s``, ``/sec``, ``per second`` or ``FPS``;
* *speedup* -- a number followed by ``×``, ``x`` or ``%`` within :data:`SPEEDUP_WINDOW_CHARS`
  of :data:`SPEED_WORDS` in the same sentence, so "~14× more accurate" is not a speed claim;
* *table column* -- a header cell naming a measurement ("Inference (ms)", "MCTS Sims/sec",
  "Speedup") makes every number under it a performance number. The removed README table
  needed exactly this: its cells were a bare ``45`` and ``670``, the units only in the header.

Fenced code blocks are skipped -- commands and their comments are instructional ("CPU smoke
test, ~10 s on a single core"). Inline code is **not** skipped: a figure in backticks is still a
claim, and skipping it would make backticks an evasion.

A block -- a paragraph, a list item, a heading or a whole table -- is *backed* when it cites a
committed artifact under ``results/`` or ``config/baselines/`` whose run provenance records a
real hardware tag: the sibling ``<stem>.run.json`` sidecar of a ``results/`` artifact, or a
``config/baselines/`` JSON document's own ``hardware_tag``. A latency figure that does not say
what it was measured on is the "honour-system loophole" the review found, so provenance that
says ``unknown`` backs nothing. One citation covers its whole block, a table included -- the
granularity at which a table's source is normally cited, and a disclosed limit: a row added to
an already-cited table is attributed to that table's artifact.
"""

from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

#: Sub-second latency units. Always a measurement when attached to a number.
SUBSECOND_UNITS: Final[tuple[str, ...]] = (
    "ms",
    "msec",
    "millisecond",
    "milliseconds",
    "µs",
    "μs",
    "us",
    "usec",
    "microsecond",
    "microseconds",
    "ns",
)

#: Second units. A measurement only in a sentence that names a measured quantity.
SECOND_UNITS: Final[tuple[str, ...]] = ("s", "sec", "secs", "second", "seconds")

#: Denominators that turn ``<number> [noun]/<denominator>`` into a throughput.
THROUGHPUT_DENOMINATORS: Final[tuple[str, ...]] = ("s", "sec", "second")

#: Frame-rate units, a throughput on their own.
FRAME_RATE_UNITS: Final[tuple[str, ...]] = ("FPS", "fps")

#: Symbols that express a ratio; a speedup only next to :data:`SPEED_WORDS`.
SPEEDUP_SYMBOLS: Final[tuple[str, ...]] = ("×", "x", "X", "%")

#: Words that make a nearby ratio a speed claim.
SPEED_WORDS: Final[tuple[str, ...]] = ("faster", "slower", "speedup", "speed-up", "slowdown")

#: Words naming a measured quantity; they make a nearby ``N s`` a latency.
MEASURED_QUANTITY_WORDS: Final[tuple[str, ...]] = (
    "latency",
    "inference",
    "throughput",
    "wall-clock",
    "wall clock",
    "wall time",
    "step time",
)

#: ``per <noun>`` rates that make a nearby ``N s`` a latency ("0.4 s per move").
PER_UNIT_NOUNS: Final[tuple[str, ...]] = (
    "move",
    "step",
    "iteration",
    "epoch",
    "sample",
    "call",
    "query",
    "solve",
    "frame",
    "batch",
    "token",
    "game",
    "search",
    "rollout",
    "evaluation",
    "position",
    "simulation",
)

#: How close a ratio must sit to a speed word to be a speedup claim. Wide enough for
#: "a speedup of 3.2×", narrow enough that "~14× more accurate ... faster" is not one.
SPEEDUP_WINDOW_CHARS: Final[int] = 40

#: Repository roots a backing citation must live under.
RESULTS_ROOT: Final[str] = "results/"
BASELINES_ROOT: Final[str] = "config/baselines/"
CITATION_ROOTS: Final[tuple[str, ...]] = (RESULTS_ROOT, BASELINES_ROOT)

#: ``hardware_tag`` values that record nothing: the run-manifest sentinel (``UNKNOWN`` in
#: ``src/research/run_manifest.py``) and the baseline schema's empty default
#: (``src/poc/baselines/schema.py``). Both are pinned by a test against those files.
PLACEHOLDER_HARDWARE_TAGS: Final[frozenset[str]] = frozenset({"", "unknown"})

#: A template is not evidence (``config/baselines/poc_headline.example.json``).
TEMPLATE_SUFFIX: Final[str] = ".example.json"

#: Suffix of a run-provenance sidecar written by ``src.research.run_manifest``.
SIDECAR_SUFFIX: Final[str] = ".run.json"

#: The provenance field that names the machine a number was measured on.
HARDWARE_TAG_FIELD: Final[str] = "hardware_tag"

#: The kind recorded for a number found under a measurement column header.
TABLE_COLUMN_KIND: Final[str] = "table-column"

#: Returned for a block that carries performance numbers and cites nothing at all.
NO_CITATION_REASON: Final[str] = "cites no artifact"


def _alternation(words: tuple[str, ...]) -> str:
    """Regex alternation over ``words``, longest first so ``secs`` is tried before ``s``."""
    return "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))


_NUMBER: Final[str] = r"(?<![\w.])\d[\d,]*(?:\.\d+)?"
_SUBSECOND_UNIT: Final[str] = rf"(?:{_alternation(SUBSECOND_UNITS)})(?!\w)"
_SECOND_UNIT: Final[str] = rf"(?:{_alternation(SECOND_UNITS)})(?!\w)"
_THROUGHPUT_UNIT: Final[str] = (
    rf"/\s*(?:{_alternation(THROUGHPUT_DENOMINATORS)})(?!\w)|per\s+second(?!\w)"
    rf"|(?:{_alternation(FRAME_RATE_UNITS)})(?!\w)"
)
_PER_UNIT: Final[str] = rf"\bper\s+(?:{_alternation(PER_UNIT_NOUNS)})\b"

LATENCY_PATTERN: Final[re.Pattern[str]] = re.compile(rf"{_NUMBER}\s*{_SUBSECOND_UNIT}")
SECONDS_PATTERN: Final[re.Pattern[str]] = re.compile(rf"{_NUMBER}\s*{_SECOND_UNIT}")
THROUGHPUT_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"{_NUMBER}\s*(?:[A-Za-z][\w-]*\s*){{0,2}}(?:{_THROUGHPUT_UNIT})"
)
SPEEDUP_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"{_NUMBER}\s*(?:{_alternation(SPEEDUP_SYMBOLS)})(?![\w×])"
)
SPEED_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(_alternation(SPEED_WORDS), re.IGNORECASE)
MEASURED_QUANTITY_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"{_alternation(MEASURED_QUANTITY_WORDS)}|{_PER_UNIT}", re.IGNORECASE
)

# Table headers. Unit symbols are matched case-sensitively -- an "MS-SSIM" column is not a
# millisecond column -- while words and rate spellings are not.
_HEADER_UNIT_PATTERN: Final[re.Pattern[str]] = re.compile(rf"(?<![\w.]){_SUBSECOND_UNIT}")
_HEADER_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"{_THROUGHPUT_UNIT}|{_alternation(SPEED_WORDS)}|{_alternation(MEASURED_QUANTITY_WORDS)}",
    re.IGNORECASE,
)
_HEADER_SECONDS: Final[str] = rf"\((?:{_alternation(SECOND_UNITS)})\)"
_HEADER_TIME_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"\btimes?\b.*{_HEADER_SECONDS}|{_HEADER_SECONDS}.*\btimes?\b", re.IGNORECASE
)
_ANY_NUMBER: Final[re.Pattern[str]] = re.compile(_NUMBER)

_FENCE: Final[re.Pattern[str]] = re.compile(r"^\s*(?:```|~~~)")
_LIST_ITEM: Final[re.Pattern[str]] = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TABLE_DELIMITER_CELL: Final[re.Pattern[str]] = re.compile(r"^:?-{3,}:?$")
_SENTENCE_BREAK: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?;])\s+")
_ESCAPED_PIPE: Final[str] = "\\|"
_PIPE_PLACEHOLDER: Final[str] = "\x00"

#: A repository path under a citation root. The look-behind admits ``/`` so a link written
#: ``../results/x.csv`` or ``blob/HEAD/results/x.csv`` from a docs page still resolves.
_CITATION: Final[re.Pattern[str]] = re.compile(
    rf"(?<![\w.-])((?:{_alternation(CITATION_ROOTS)})[^\s`'\"()\[\]<>|]+)"
)
_TRAILING_PUNCTUATION: Final[str] = ".,;:"


@dataclass(frozen=True)
class Block:
    """A paragraph, list item, heading or table: the unit a citation covers."""

    kind: str
    lines: tuple[tuple[int, str], ...]

    @property
    def text(self) -> str:
        """The block's source text, lines joined."""
        return "\n".join(line for _, line in self.lines)


@dataclass(frozen=True)
class PerformanceNumber:
    """One performance number, located in its source document."""

    line_no: int
    line: str
    kind: str
    token: str


@dataclass(frozen=True)
class UnbackedNumber:
    """A performance number whose block cites no hardware-tagged artifact."""

    number: PerformanceNumber
    reasons: tuple[str, ...]


def split_blocks(markdown: str) -> list[Block]:
    """Split Markdown into citation-scoped blocks, skipping fenced code.

    Args:
        markdown: A Markdown document.

    Returns:
        Paragraphs, list items (one block each, continuation lines included), headings and
        tables, in document order, with 1-based line numbers.

    """
    blocks: list[Block] = []
    current: list[tuple[int, str]] = []
    current_kind = ""
    in_fence = False

    def flush() -> None:
        nonlocal current, current_kind
        if current:
            blocks.append(Block(current_kind, tuple(current)))
        current, current_kind = [], ""

    for line_no, line in enumerate(markdown.splitlines(), start=1):
        if _FENCE.match(line):
            flush()
            in_fence = not in_fence
            continue
        stripped = line.strip()
        if in_fence or not stripped:
            flush()
            continue
        kind = "table" if stripped.startswith("|") else "text"
        is_heading = stripped.startswith("#")
        if current and (kind != current_kind or is_heading or _LIST_ITEM.match(line)):
            flush()
        current_kind = kind
        current.append((line_no, line))
        if is_heading:
            flush()
    flush()
    return blocks


def _table_cells(line: str) -> list[str]:
    """Cells of one table row, honouring GFM's backslash-escaped pipe."""
    row = line.strip().replace(_ESCAPED_PIPE, _PIPE_PLACEHOLDER)
    row = row.removeprefix("|").removesuffix("|")
    return [cell.replace(_PIPE_PLACEHOLDER, "|").strip() for cell in row.split("|")]


def is_measurement_header(header: str) -> bool:
    """Whether a table header cell names a measured, hardware-dependent quantity.

    Args:
        header: The text of one header cell.

    Returns:
        True for a latency or throughput unit ("Inference (ms)", "Sims/sec", "FPS"), a speed
        or measured-quantity word ("Speedup", "Latency"), or seconds next to "time".

    """
    return any(
        pattern.search(header)
        for pattern in (_HEADER_UNIT_PATTERN, _HEADER_WORD_PATTERN, _HEADER_TIME_PATTERN)
    )


def _sentence_spans(text: str) -> list[tuple[int, str]]:
    """``(offset, sentence)`` pairs covering ``text``."""
    spans: list[tuple[int, str]] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(text):
        spans.append((start, text[start : match.start()]))
        start = match.end()
    spans.append((start, text[start:]))
    return [(offset, sentence) for offset, sentence in spans if sentence]


def inline_numbers(text: str) -> list[tuple[int, str, str]]:
    """Inline performance numbers in ``text``, sentence by sentence.

    Args:
        text: Prose -- a paragraph, a list item or one table row.

    Returns:
        ``(offset, kind, token)`` per number, ``offset`` indexing into ``text``.

    """
    found: list[tuple[int, str, str]] = []
    for offset, sentence in _sentence_spans(text):
        hits = [("latency", m) for m in LATENCY_PATTERN.finditer(sentence)]
        hits += [("throughput", m) for m in THROUGHPUT_PATTERN.finditer(sentence)]
        if MEASURED_QUANTITY_PATTERN.search(sentence):
            hits += [("latency", m) for m in SECONDS_PATTERN.finditer(sentence)]
        for match in SPEEDUP_PATTERN.finditer(sentence):
            lo = max(0, match.start() - SPEEDUP_WINDOW_CHARS)
            if SPEED_WORD_PATTERN.search(sentence[lo : match.end() + SPEEDUP_WINDOW_CHARS]):
                hits.append(("speedup", match))
        found += [(offset + m.start(), kind, m.group(0)) for kind, m in hits]
    return found


def _column_numbers(block: Block) -> list[PerformanceNumber]:
    """Numbers under a measurement header of a table block.

    A cell that already carries an inline performance number ("45 ms") is left to the inline
    pass, so one figure is reported once, under its more specific kind.
    """
    header = _table_cells(block.lines[0][1])
    measured = {index for index, cell in enumerate(header) if is_measurement_header(cell)}
    found: list[PerformanceNumber] = []
    for line_no, line in block.lines[1:]:
        cells = _table_cells(line)
        if all(_TABLE_DELIMITER_CELL.match(cell) for cell in cells if cell):
            continue
        for index, cell in enumerate(cells):
            if index not in measured or inline_numbers(cell):
                continue
            if match := _ANY_NUMBER.search(cell):
                found.append(PerformanceNumber(line_no, line, TABLE_COLUMN_KIND, match.group(0)))
    return found


def performance_numbers(block: Block) -> list[PerformanceNumber]:
    """Every performance number in ``block``, located at its source line.

    Args:
        block: One block from :func:`split_blocks`.

    Returns:
        Prose is scanned as a whole so a sentence wrapped across lines keeps its measured
        quantity; a table row is its own sentence, and a table additionally contributes every
        number under a measurement header (:func:`is_measurement_header`).

    """
    if block.kind == "table":
        found = [
            PerformanceNumber(line_no, line, kind, token)
            for line_no, line in block.lines
            for _, kind, token in inline_numbers(line)
        ]
        found += _column_numbers(block)
    else:
        text = block.text
        line_starts = [0]
        for _, line in block.lines[:-1]:
            line_starts.append(line_starts[-1] + len(line) + 1)
        found = []
        for offset, kind, token in inline_numbers(text):
            line_no, line = block.lines[bisect.bisect_right(line_starts, offset) - 1]
            found.append(PerformanceNumber(line_no, line, kind, token))
    return found


def _expand_braces(token: str) -> list[str]:
    """Expand one ``{a,b}`` group, the form docs use for ``results/x.{csv,run.json}``."""
    match = re.search(r"\{([^{}]*)\}", token)
    if match is None:
        return [token]
    return [
        token[: match.start()] + option.strip() + token[match.end() :]
        for option in match.group(1).split(",")
    ]


def cited_artifacts(text: str) -> list[str]:
    """Repository paths under :data:`CITATION_ROOTS` that ``text`` cites, braces expanded.

    Args:
        text: Any Markdown text; citations may be bare, backticked or link targets.

    Returns:
        Unique cited paths in order of first appearance, trailing punctuation removed.

    """
    found: list[str] = []
    for match in _CITATION.finditer(text):
        token = match.group(1).rstrip(_TRAILING_PUNCTUATION)
        found += [path for path in _expand_braces(token) if path not in found]
    return found


def provenance_path(artifact: str) -> str:
    """The file that records ``artifact``'s run provenance.

    Args:
        artifact: A repository-relative path under :data:`CITATION_ROOTS`.

    Returns:
        ``artifact`` itself for a sidecar or a ``config/baselines/`` JSON document (both carry
        their own ``hardware_tag``), otherwise the sibling ``<stem>.run.json`` sidecar.

    """
    if artifact.endswith(SIDECAR_SUFFIX):
        return artifact
    if artifact.startswith(BASELINES_ROOT) and artifact.endswith(".json"):
        return artifact
    return str(PurePosixPath(artifact).with_suffix(SIDECAR_SUFFIX))


def citation_problem(repo_root: Path, artifact: str) -> str | None:
    """Why ``artifact`` cannot back a performance number, or ``None`` if it can.

    Args:
        repo_root: The repository root the path is relative to.
        artifact: A cited path from :func:`cited_artifacts`.

    Returns:
        A one-line reason, or ``None`` when the artifact exists and its provenance records a
        real hardware tag.

    """
    if artifact.endswith(TEMPLATE_SUFFIX):
        return f"{artifact} is a template, not evidence"
    if not (repo_root / artifact).is_file():
        return f"{artifact} does not exist"
    record = provenance_path(artifact)
    if not (repo_root / record).is_file():
        return f"{artifact} has no run-provenance sidecar {record}"
    try:
        document = json.loads((repo_root / record).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"{record} is not readable JSON ({exc.__class__.__name__})"
    tag = document.get(HARDWARE_TAG_FIELD) if isinstance(document, dict) else None
    if not isinstance(tag, str) or tag.strip().lower() in PLACEHOLDER_HARDWARE_TAGS:
        return f"{record} records no {HARDWARE_TAG_FIELD} (got {tag!r})"
    return None


def unbacked_performance_numbers(markdown: str, repo_root: Path) -> list[UnbackedNumber]:
    """Performance numbers in ``markdown`` whose block cites no hardware-tagged artifact.

    Args:
        markdown: A Markdown document.
        repo_root: The repository root citations are resolved against.

    Returns:
        One entry per unbacked number, with the reason each citation in its block (if any)
        failed to back it.

    """
    unbacked: list[UnbackedNumber] = []
    for block in split_blocks(markdown):
        numbers = performance_numbers(block)
        if not numbers:
            continue
        problems = [citation_problem(repo_root, path) for path in cited_artifacts(block.text)]
        if any(problem is None for problem in problems):
            continue
        reasons = tuple(p for p in problems if p is not None) or (NO_CITATION_REASON,)
        unbacked += [UnbackedNumber(number, reasons) for number in numbers]
    return unbacked
