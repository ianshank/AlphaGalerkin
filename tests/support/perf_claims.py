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

**What backs a number.** A number is backed when its block -- a paragraph, a list item, a
heading or a whole table -- cites a committed artifact under ``results/`` or
``config/baselines/`` that passes all three of these, in order:

1. *Provenance.* The artifact's run provenance records a real hardware tag: the sibling
   ``<stem>.run.json`` sidecar of a ``results/`` artifact, or a ``config/baselines/`` JSON
   document's own ``hardware_tag``. A timing that does not say what it was measured on is the
   "honour-system loophole" the review found, so provenance that says ``unknown`` backs nothing.
2. *Timing data.* The run records timing data: a column of ``results/<stem>.csv``, a metric or
   arm counter of its sidecar, or a baseline entry, whose **name** states a duration
   (``wall_time_seconds``, ``latency_ms``), a rate (``sims_per_second``, ``fps``) or a speed
   ratio (``fnet_speedup``, ``latency_reduction_pct``) -- see :func:`field_unit`. A hardware tag
   alone backs nothing: a run that recorded DOF and error cannot back a latency. A sidecar's
   ``config`` block is never read; a configured timeout is not a measurement.
3. *Traceability.* The figure is one of those recorded values: converted to the figure's unit
   and rounded to the figure's written precision, some value of the figure's own dimension
   equals it. "45 ms" needs a recorded duration in [44.5, 45.5] ms; "0.16 s" one in
   [0.155, 0.165] s. A figure whose unit is not stated (a "Throughput" column with no unit), or
   that is stated to fewer than :data:`MIN_TRACEABLE_SIGNIFICANT_DIGITS` significant digits
   ("2 s", "5×"), identifies no value and cannot be traced. Nothing is derived: a rate is not
   the reciprocal of a recorded duration and a speedup is not the ratio of two recorded
   durations, because with n recorded timings there are n² ratios to match by coincidence.

Disclosed limits. One citation covers its whole block, a table included -- the granularity at
which a table's source is normally cited -- but every figure in the block must trace on its
own. Traceability matches values, not meaning: a figure that happens to equal an unrelated
timing of the cited run is accepted. ``tests/docs/test_performance_claims.py`` records the
measured coincidence rate against the committed runs.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

#: Sub-second latency units, as seconds per unit. Always a measurement when attached to a number.
SUBSECOND_UNITS: Final[dict[str, float]] = {
    "ms": 1e-3,
    "msec": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "µs": 1e-6,
    "μs": 1e-6,
    "us": 1e-6,
    "usec": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
    "ns": 1e-9,
}

#: Second units, as seconds per unit. A measurement only in a sentence that names a measured
#: quantity.
SECOND_UNITS: Final[dict[str, float]] = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
}

#: Denominators that turn ``<number> [noun]/<denominator>`` into a throughput, as seconds.
THROUGHPUT_DENOMINATORS: Final[dict[str, float]] = {"s": 1.0, "sec": 1.0, "second": 1.0}

#: Frame-rate units, a throughput on their own, as events per second per unit.
FRAME_RATE_UNITS: Final[dict[str, float]] = {"FPS": 1.0, "fps": 1.0}

#: Symbols that express a ratio of two speeds or times; a speedup only next to SPEED_WORDS.
RATIO_SYMBOLS: Final[tuple[str, ...]] = ("×", "x", "X")

#: The symbol that expresses a speed change as a percentage; a speedup only next to SPEED_WORDS.
PERCENT_SYMBOL: Final[str] = "%"

#: Every symbol that can make a number a speedup.
SPEEDUP_SYMBOLS: Final[tuple[str, ...]] = (*RATIO_SYMBOLS, PERCENT_SYMBOL)

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

#: Suffix of the tabular data a ``results/`` run writes beside its sidecar.
DATA_SUFFIX: Final[str] = ".csv"

#: The provenance field that names the machine a number was measured on.
HARDWARE_TAG_FIELD: Final[str] = "hardware_tag"

#: Where a sidecar or baseline document records measured values -- never ``config``.
METRICS_FIELD: Final[str] = "metrics"
ARMS_FIELD: Final[str] = "arms"
COUNTERS_FIELD: Final[str] = "counters"
ENTRIES_FIELD: Final[str] = "entries"
ENTRY_NAME_FIELD: Final[str] = "metric_name"
ENTRY_VALUE_FIELD: Final[str] = "value"

#: The kind recorded for a number found under a measurement column header.
TABLE_COLUMN_KIND: Final[str] = "table-column"

#: Returned for a block that carries performance numbers and cites nothing at all.
NO_CITATION_REASON: Final[str] = "cites no artifact"

#: The dimensions a figure or a recorded value can have. Base units: the second (duration), one
#: event per second (rate), the plain number (ratio, percent).
DURATION: Final[str] = "duration"
RATE: Final[str] = "rate"
RATIO: Final[str] = "ratio"
PERCENT: Final[str] = "percent"

#: Field-name tokens that make a recorded value a speed ratio (``fnet_speedup``).
SPEED_FIELD_TOKENS: Final[frozenset[str]] = frozenset({"speedup", "slowdown"})

#: Field-name tokens naming a timing quantity; ended by a percent token, a percentage of it
#: (``latency_reduction_pct``).
TIMING_FIELD_TOKENS: Final[frozenset[str]] = frozenset(
    {"latency", "throughput", "time", "wall", "clock", "runtime", "duration", "elapsed"}
)

#: Last field-name tokens that make a speed or timing field a percentage.
PERCENT_FIELD_TOKENS: Final[frozenset[str]] = frozenset({"pct", "percent"})

#: The field-name token that, followed by a denominator, makes a rate (``sims_per_second``).
PER_FIELD_TOKEN: Final[str] = "per"

#: Fewest significant digits a figure may state and still identify one recorded value. "2 s"
#: matches anything in [1.5, 2.5] s, a quarter of its own size either way.
MIN_TRACEABLE_SIGNIFICANT_DIGITS: Final[int] = 2

#: Relative slack on the rounding window, so binary floating point cannot reject an exact match.
FLOAT_SLACK: Final[float] = 1e-9


def _alternation(words: tuple[str, ...] | dict[str, float]) -> str:
    """Regex alternation over ``words``, longest first so ``secs`` is tried before ``s``."""
    return "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))


_DURATION_UNITS: Final[dict[str, float]] = {**SUBSECOND_UNITS, **SECOND_UNITS}

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
_HEADER_UNIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"(?<![\w.])(?P<unit>{_alternation(SUBSECOND_UNITS)})(?!\w)"
)
_HEADER_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"{_THROUGHPUT_UNIT}|{_alternation(SPEED_WORDS)}|{_alternation(MEASURED_QUANTITY_WORDS)}",
    re.IGNORECASE,
)
_HEADER_SECONDS: Final[str] = rf"\((?P<unit>{_alternation(SECOND_UNITS)})\)"
_HEADER_SECONDS_PATTERN: Final[re.Pattern[str]] = re.compile(_HEADER_SECONDS, re.IGNORECASE)
_HEADER_TIME_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"\btimes?\b.*{_HEADER_SECONDS}|{_HEADER_SECONDS}.*\btimes?\b", re.IGNORECASE
)
_ANY_NUMBER: Final[re.Pattern[str]] = re.compile(_NUMBER)

# A unit written straight after a number, as every inline token and a unit-bearing table cell
# ends: the rate forms first, so the "s" of "sims/sec" is not read as seconds.
_RATE_TAIL: Final[str] = (
    rf"/\s*(?P<denominator>{_alternation(THROUGHPUT_DENOMINATORS)})(?!\w)"
    rf"|per\s+(?P<per>{_alternation(THROUGHPUT_DENOMINATORS)})(?!\w)"
    rf"|(?P<frequency>{_alternation(FRAME_RATE_UNITS)})(?!\w)"
)
_UNIT_TAIL: Final[str] = (
    rf"(?:{_RATE_TAIL}|(?P<percent>{re.escape(PERCENT_SYMBOL)})"
    rf"|(?P<ratio>{_alternation(RATIO_SYMBOLS)})(?![\w×])"
    rf"|(?P<duration>{_alternation(_DURATION_UNITS)})(?!\w))"
)
_TOKEN_UNIT_PATTERN: Final[re.Pattern[str]] = re.compile(rf"{_UNIT_TAIL}\s*$")
_CELL_FIGURE: Final[re.Pattern[str]] = re.compile(rf"{_NUMBER}(?:\s*{_UNIT_TAIL})?")
_HEADER_RATE_PATTERN: Final[re.Pattern[str]] = re.compile(_RATE_TAIL, re.IGNORECASE)
_FIELD_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[^\w]+|_")

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
    """One performance number, located in its source document.

    ``header`` is the column header a :data:`TABLE_COLUMN_KIND` number sits under -- where a
    bare cell's unit is written -- and empty for every other kind.
    """

    line_no: int
    line: str
    kind: str
    token: str
    header: str = ""


@dataclass(frozen=True)
class UnbackedNumber:
    """A performance number that no cited artifact backs, with why."""

    number: PerformanceNumber
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Unit:
    """A dimension, and the size of one unit in that dimension's base unit.

    ``Unit(DURATION, 1e-3)`` is a millisecond; ``Unit(RATE, 1 / 60)`` is "per minute".
    """

    dimension: str
    scale: float


@dataclass(frozen=True)
class Figure:
    """A performance number read as a quantity: its value, precision and unit."""

    value: float
    decimals: int
    significant_digits: int
    unit: Unit | None


@dataclass(frozen=True)
class RecordedValue:
    """One timing value an artifact records, and where."""

    source: str
    field: str
    value: float
    unit: Unit


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
    pass, so one figure is reported once, under its more specific kind. A unit written in the
    cell itself ("0.4 s") stays on the token, where it outranks the header's.
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
            if match := _CELL_FIGURE.search(cell):
                found.append(
                    PerformanceNumber(
                        line_no, line, TABLE_COLUMN_KIND, match.group(0), header=header[index]
                    )
                )
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


# --------------------------------------------------------------------------------------
# units: of a written figure, and of a recorded field
# --------------------------------------------------------------------------------------


def _rate_unit(match: re.Match[str]) -> Unit:
    """The rate a ``_RATE_TAIL`` match names."""
    if frequency := match.group("frequency"):
        scale = FRAME_RATE_UNITS.get(frequency, FRAME_RATE_UNITS[frequency.lower()])
        return Unit(RATE, scale)
    denominator = (match.group("denominator") or match.group("per")).lower()
    return Unit(RATE, 1.0 / THROUGHPUT_DENOMINATORS[denominator])


def _tail_unit(text: str) -> Unit | None:
    """The unit ``text`` ends with -- the way every detected token is written -- if any."""
    match = _TOKEN_UNIT_PATTERN.search(text)
    if match is None:
        return None
    if match.group("percent"):
        return Unit(PERCENT, 1.0)
    if match.group("ratio"):
        return Unit(RATIO, 1.0)
    if duration := match.group("duration"):
        return Unit(DURATION, _DURATION_UNITS[duration])
    return _rate_unit(match)


def header_unit(header: str) -> Unit | None:
    """The unit a measurement column header states for the bare numbers under it.

    Args:
        header: The text of one header cell.

    Returns:
        A rate for "Sims/sec" or "FPS", a percentage for a "%" column, a ratio for a speed
        word ("Speedup"), a duration for "(ms)" or "Time (s)"; ``None`` when the header names a
        quantity but no unit ("Throughput", "Latency").

    """
    if match := _HEADER_RATE_PATTERN.search(header):
        return _rate_unit(match)
    if PERCENT_SYMBOL in header:
        return Unit(PERCENT, 1.0)
    if SPEED_WORD_PATTERN.search(header):
        return Unit(RATIO, 1.0)
    if match := _HEADER_UNIT_PATTERN.search(header):
        return Unit(DURATION, SUBSECOND_UNITS[match.group("unit")])
    if match := _HEADER_SECONDS_PATTERN.search(header):
        return Unit(DURATION, SECOND_UNITS[match.group("unit").lower()])
    return None


def figure_of(number: PerformanceNumber) -> Figure:
    """Read a performance number as a quantity.

    Args:
        number: A number from :func:`performance_numbers`; every token starts with its digits.

    Returns:
        Its value; its precision, as decimal places and significant digits written (a trailing
        zero is written, so "180" is exact to the unit); and its unit -- the one written after
        the digits, else, under a table header, the header's.

    """
    match = _ANY_NUMBER.match(number.token)
    digits = match.group(0).replace(",", "") if match else ""
    unit = _tail_unit(number.token[len(match.group(0)) :]) if match else None
    if unit is None and number.kind == TABLE_COLUMN_KIND:
        unit = header_unit(number.header)
    return Figure(
        value=float(digits) if digits else math.nan,
        decimals=len(digits.partition(".")[2]),
        significant_digits=len(digits.replace(".", "").lstrip("0")),
        unit=unit,
    )


def field_unit(name: str) -> Unit | None:
    """What a recorded field measures, read from its snake- or kebab-case name.

    Args:
        name: A CSV column, sidecar metric or counter, or baseline ``metric_name``.

    Returns:
        A rate for ``<x>_per_<denominator>`` or a trailing frame-rate unit; a ratio for a
        ``speedup``/``slowdown`` field, or a percentage when such a field -- or one naming a
        timing quantity -- ends in ``pct``/``percent``; a duration for a trailing time unit
        (``wall_time_seconds``, ``train_time_s``); otherwise ``None``.

    """
    tokens = [token for token in _FIELD_TOKEN_SPLIT.split(name.lower()) if token]
    if not tokens:
        return None
    last = tokens[-1]
    if len(tokens) > 1 and tokens[-2] == PER_FIELD_TOKEN and last in THROUGHPUT_DENOMINATORS:
        return Unit(RATE, 1.0 / THROUGHPUT_DENOMINATORS[last])
    if last in FRAME_RATE_UNITS:
        return Unit(RATE, FRAME_RATE_UNITS[last])
    is_speed = not SPEED_FIELD_TOKENS.isdisjoint(tokens)
    if last in PERCENT_FIELD_TOKENS and (is_speed or not TIMING_FIELD_TOKENS.isdisjoint(tokens)):
        return Unit(PERCENT, 1.0)
    if is_speed:
        return Unit(RATIO, 1.0)
    if last in _DURATION_UNITS:
        return Unit(DURATION, _DURATION_UNITS[last])
    return None


# --------------------------------------------------------------------------------------
# citations, provenance and recorded values
# --------------------------------------------------------------------------------------


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


def _measured(value: object) -> float | None:
    """``value`` as a finite float, or ``None`` for text, booleans, NaN and infinities."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _as_dict(value: object) -> dict[object, object]:
    return value if isinstance(value, dict) else {}


def _csv_values(repo_root: Path, path: str) -> list[RecordedValue]:
    """Values under the timing columns of the CSV at ``path``; nothing if it cannot be read."""
    file = repo_root / path
    if not file.is_file():
        return []
    try:
        header, *rows = list(csv.reader(file.read_text(encoding="utf-8").splitlines()))
    except (UnicodeDecodeError, csv.Error, ValueError):
        return []
    units = {index: unit for index, name in enumerate(header) if (unit := field_unit(name))}
    return [
        RecordedValue(path, header[index], value, unit)
        for row in rows
        for index, unit in units.items()
        if index < len(row) and (value := _measured(row[index])) is not None
    ]


def _document_values(repo_root: Path, path: str) -> list[RecordedValue]:
    """Timing metrics, arm counters and baseline entries of the JSON document at ``path``."""
    try:
        document = _as_dict(json.loads((repo_root / path).read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    pairs = list(_as_dict(document.get(METRICS_FIELD)).items())
    for arm in _as_list(document.get(ARMS_FIELD)):
        pairs += _as_dict(_as_dict(arm).get(COUNTERS_FIELD)).items()
    for entry in map(_as_dict, _as_list(document.get(ENTRIES_FIELD))):
        pairs.append((entry.get(ENTRY_NAME_FIELD), entry.get(ENTRY_VALUE_FIELD)))
    return [
        RecordedValue(path, name, value, unit)
        for name, raw in pairs
        if isinstance(name, str)
        and (unit := field_unit(name)) is not None
        and (value := _measured(raw)) is not None
    ]


def recorded_values(repo_root: Path, artifact: str) -> list[RecordedValue]:
    """Every timing value the run behind ``artifact`` records.

    Args:
        repo_root: The repository root the path is relative to.
        artifact: A cited path from :func:`cited_artifacts`.

    Returns:
        For a ``results/`` artifact, the timing columns of ``<stem>.csv`` and the timing metrics
        and arm counters of ``<stem>.run.json``; for a ``config/baselines/`` document, its timing
        entries. Never a ``config`` value.

    """
    record = provenance_path(artifact)
    values: list[RecordedValue] = []
    if record.endswith(SIDECAR_SUFFIX):
        values += _csv_values(repo_root, record.removesuffix(SIDECAR_SUFFIX) + DATA_SUFFIX)
    return values + _document_values(repo_root, record)


def citation_problem(repo_root: Path, artifact: str) -> str | None:
    """Why ``artifact`` cannot back a performance number, or ``None`` if it can.

    Args:
        repo_root: The repository root the path is relative to.
        artifact: A cited path from :func:`cited_artifacts`.

    Returns:
        A one-line reason, or ``None`` when the artifact exists, its provenance records a real
        hardware tag and its run records timing data. Whether a given figure is one of those
        values is :func:`trace_problem`'s question.

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
    if not recorded_values(repo_root, artifact):
        return (
            f"{artifact} records no timing data: no column, metric or entry of its run "
            "names a duration, a rate or a speed ratio"
        )
    return None


def _in_figure_units(recorded: RecordedValue, unit: Unit) -> float:
    return recorded.value * recorded.unit.scale / unit.scale


def trace_problem(
    number: PerformanceNumber, evidence: list[RecordedValue], sources: list[str]
) -> str | None:
    """Why ``number`` is not one of the recorded values, or ``None`` if it is.

    Args:
        number: A performance number.
        evidence: Every timing value its block's backing citations record.
        sources: Those citations, for the message.

    Returns:
        ``None`` when some recorded value of the figure's dimension, converted to its unit,
        rounds to it at its written precision; otherwise a one-line reason.

    """
    figure = figure_of(number)
    cited = ", ".join(sources)
    if figure.unit is None:
        return f"{number.token!r} states no unit, so no recorded value can be matched to it"
    if figure.significant_digits < MIN_TRACEABLE_SIGNIFICANT_DIGITS:
        return (
            f"{number.token!r} states {figure.significant_digits} significant digit(s); "
            f"{MIN_TRACEABLE_SIGNIFICANT_DIGITS} are needed to identify a recorded value"
        )
    unit = figure.unit
    same = [recorded for recorded in evidence if recorded.unit.dimension == unit.dimension]
    if not same:
        return f"{cited} record(s) no {unit.dimension} value"
    window = 0.5 * 10.0**-figure.decimals + FLOAT_SLACK * max(1.0, abs(figure.value))
    nearest = min(same, key=lambda recorded: abs(_in_figure_units(recorded, unit) - figure.value))
    if abs(_in_figure_units(nearest, unit) - figure.value) <= window:
        return None
    return (
        f"{number.token!r} is not a {unit.dimension} that {cited} record(s) at its stated "
        f"precision (nearest: {_in_figure_units(nearest, unit):.6g} from "
        f"{nearest.source}:{nearest.field})"
    )


def unbacked_performance_numbers(markdown: str, repo_root: Path) -> list[UnbackedNumber]:
    """Performance numbers in ``markdown`` that no artifact cited in their block backs.

    Args:
        markdown: A Markdown document.
        repo_root: The repository root citations are resolved against.

    Returns:
        One entry per unbacked number, with the reason each citation in its block (if any)
        failed to back it and, where one did back its block, why the number is not among the
        values it records.

    """
    unbacked: list[UnbackedNumber] = []
    for block in split_blocks(markdown):
        numbers = performance_numbers(block)
        if not numbers:
            continue
        cited = cited_artifacts(block.text)
        problems = [citation_problem(repo_root, path) for path in cited]
        failed = tuple(problem for problem in problems if problem is not None)
        backing = [path for path, problem in zip(cited, problems, strict=True) if problem is None]
        if not backing:
            unbacked += [UnbackedNumber(n, failed or (NO_CITATION_REASON,)) for n in numbers]
            continue
        evidence = [value for path in backing for value in recorded_values(repo_root, path)]
        for number in numbers:
            if (problem := trace_problem(number, evidence, backing)) is not None:
                unbacked.append(UnbackedNumber(number, (*failed, problem)))
    return unbacked
