"""Unit-bearing performance numbers in Markdown, and whether an artifact backs them.

The charter's *Evidence-Backed Claims* Requirement says every numeric headline claim cites a
committed artifact. The only guard over ``README.md`` used to cover MCTS-vs-Dörfler policy
ratios, so ``README.md`` could show inference latency and MCTS simulations per second for three
models "on NVIDIA RTX 3090" -- a card none of the documented rigs use -- with no artifact, run
output or sidecar behind a single figure (``docs/business/COMMERCIALIZATION_PEER_REVIEW.md``
§5.2). This module reads Markdown as data and finds such numbers; the guard that applies it to
the repository is ``tests/docs/test_performance_claims.py``.

A **performance number** is a hardware-dependent measurement:

* *latency* -- a number followed by a sub-second unit (:data:`SUBSECOND_UNITS`); or by a second,
  minute or hour unit (:data:`CONTEXTUAL_DURATION_UNITS`) when its sentence names a measured
  quantity (:data:`MEASURED_QUANTITY_WORDS`, ``per <move|step|...>`` or ``s/move``), so "takes
  ~2 s" and "~5 min" are not claims;
* *throughput* -- a number followed by ``[noun]/<s|min|h>``, ``per <second|minute|hour>``,
  ``FPS`` or ``Hz`` (:data:`THROUGHPUT_DENOMINATORS`, :data:`RATE_UNITS`);
* *speedup* -- a number followed by ``×``, ``x``, ``-fold`` or ``%`` within
  :data:`SPEEDUP_WINDOW_CHARS` of a speed word ("faster", "speeds up", "accelerates") or a timing
  quantity ("latency") in its sentence, so "~14× more accurate" is not one;
* *table column* -- a header naming a measurement ("Inference (ms)", "MCTS Sims/sec", "Runtime
  (s)", "Time per move") makes every number under it one. The removed README table needed
  exactly this: its cells were a bare ``45`` and ``670``, the units only in the header.

A unit may be joined to its number by a hyphen ("a 12-ms pass", "10-fold"). A number whose noun
phrase is a configured setting (:data:`CONFIGURATION_NOUNS`: "a 900 s wall-clock timeout",
"timeout of 900 s", "a 0.5% tolerance") is not a measurement, nor is a header naming one
("Timeout (s)"). "Budget", "threshold", "limit" and "deadline" are deliberately not settings
here: each also frames a measured claim ("stays within a 100 ms budget"), and a missed claim
costs more than a reworded setting. Fenced code is instructional and skipped; inline code is
**not** -- a figure in backticks is still a claim, and skipping it would make backticks an evasion.

**What backs a number.** Its block -- a paragraph, list item, heading or whole table -- cites an
artifact under ``results/`` or ``config/baselines/`` that passes both checks:

1. *Provenance.* The run records a real hardware tag: the sibling ``<stem>.run.json`` sidecar of
   a ``results/`` artifact, or a ``config/baselines/`` document's own ``hardware_tag``. A timing
   that does not say what it was measured on backs nothing, and a template is not evidence.
2. *Traceability.* The figure is one of the run's **named timing measurements** -- a sidecar
   ``metrics`` or arm ``counters`` value, or a baseline ``entries`` value -- whose name states a
   unit of the figure's dimension (:func:`field_unit`: ``wall_time_seconds``,
   ``sims_per_second``, ``fnet_speedup``, ``latency_reduction_pct``) and which, converted to the
   figure's unit and rounded to its written precision, equals it: "45 ms" needs a recorded
   duration in [44.5, 45.5] ms. A figure with no unit, or written to fewer than
   :data:`MIN_TRACEABLE_SIGNIFICANT_DIGITS` significant digits ("2 s", "5×"), identifies no value.
   Nothing is derived (a rate is not the reciprocal of a duration, a percentage not a converted
   ratio), and a sidecar's ``config`` is never read: a configured timeout is not a measurement.

So a hardware tag alone backs nothing, and neither does timing data that is not the figure. Two
weaker rules were measured against the committed runs on 2026-10-08 and rejected:

* "The run records timing data" accepted the removed RTX 3090 table's latency column on a
  citation of ``results/mcts_classical_amr_arena.csv``, which records AMR solve wall times: the
  cheapest edit backing *any* latency was to cite that file.
* Tracing to *any CSV row* let a figure match by coincidence: of the 2-significant-digit
  durations from 10 ms to 99 s, 12% match a row of the arena table and 42% and 46% a row of the
  two Gate 1 tables (88% of the 10-99 s band of the Z-tetromino one). Against named measurements
  the rate is at most 3%, and 16% at one significant digit -- hence the precision floor.
  ``test_named_measurements_rarely_coincide`` keeps the 2-digit rate under its bound.

Disclosed limits. One citation covers its whole block, a table included, but every figure must
trace on its own. Traceability matches values, not meaning: a figure equal to an unrelated named
measurement of the cited run is accepted. Units are read from field names, so a name reusing a
unit token for something else (``ms_ssim``) is misread as a duration. A header naming a duration
with no unit or rate ("Epoch time") marks no column; a cell stating a sub-second unit still counts.
Resource usage -- GPU utilisation, peak memory -- is measured but is not a performance number here.
"""

from __future__ import annotations

import bisect
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

# --------------------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------------------

#: Sub-second duration units, as seconds per unit. A measurement wherever attached to a number.
SUBSECOND_UNITS: Final[dict[str, float]] = {
    **dict.fromkeys(("ms", "msec", "millisecond", "milliseconds"), 1e-3),
    **dict.fromkeys(("µs", "μs", "us", "usec", "microsecond", "microseconds"), 1e-6),
    **dict.fromkeys(("ns", "nanosecond", "nanoseconds"), 1e-9),
}

#: Second, minute and hour units, as seconds per unit. A measurement only in a sentence that
#: names a measured quantity.
CONTEXTUAL_DURATION_UNITS: Final[dict[str, float]] = {
    **dict.fromkeys(("s", "sec", "secs", "second", "seconds"), 1.0),
    **dict.fromkeys(("min", "mins", "minute", "minutes"), 60.0),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), 3600.0),
}

#: Duration abbreviations not read as units in a recorded field's name: there "min" means minimum
#: (``l2_ratio_seed_min``), and "h"/"hr" are too short to mean hours reliably.
FIELD_UNIT_EXCLUSIONS: Final[frozenset[str]] = frozenset({"min", "mins", "h", "hr", "hrs"})

#: Denominators that turn ``<number> [noun]/<denominator>`` or ``per <denominator>`` into a
#: throughput, as seconds per unit.
THROUGHPUT_DENOMINATORS: Final[dict[str, float]] = {
    **dict.fromkeys(("s", "sec", "second"), 1.0),
    **dict.fromkeys(("min", "minute"), 60.0),
    **dict.fromkeys(("h", "hr", "hour"), 3600.0),
}

#: Units that are a throughput on their own, as events per second per unit. ``MHz``/``GHz`` are
#: left out: on a front door they state a clock speed -- hardware, not a measurement of the code.
RATE_UNITS: Final[dict[str, float]] = {"FPS": 1.0, "fps": 1.0, "Hz": 1.0, "kHz": 1e3}

#: Symbols that express a ratio of two speeds or times; a speedup only near a speed word.
RATIO_SYMBOLS: Final[tuple[str, ...]] = ("×", "x", "X", "fold")

#: The symbol that expresses a speed change as a percentage; a speedup only near a speed word.
PERCENT_SYMBOL: Final[str] = "%"

#: Every symbol that can make a number a speedup.
SPEEDUP_SYMBOLS: Final[tuple[str, ...]] = (*RATIO_SYMBOLS, PERCENT_SYMBOL)

#: Words that state a change of speed. ``accelerate`` also matches "accelerates"/"accelerated".
SPEED_WORDS: Final[tuple[str, ...]] = (
    "faster",
    "slower",
    "quicker",
    "speedup",
    "speed-up",
    "speeds up",
    "speed up",
    "sped up",
    "slowdown",
    "accelerate",
)

#: Nouns naming a timing quantity: near a ratio or percentage they make it a speed claim ("cuts
#: latency by 80%"). "Runtime" is not one -- in this repository it names software as often.
TIMING_QUANTITY_WORDS: Final[tuple[str, ...]] = (
    "latency",
    "throughput",
    "wall-clock",
    "wall clock",
    "wall time",
    "step time",
)

#: Words naming a measured quantity; they make a nearby second, minute or hour figure a latency.
MEASURED_QUANTITY_WORDS: Final[tuple[str, ...]] = (*TIMING_QUANTITY_WORDS, "inference")

#: ``per <noun>`` / ``/<noun>`` rates that make a nearby second figure a latency ("0.4 s/move").
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

#: Header nouns naming a duration; a measurement header with a bracketed duration unit or a
#: ``per <noun>`` rate ("Runtime (s)", "Time per move").
DURATION_HEADER_NOUNS: Final[tuple[str, ...]] = ("time", "times", "runtime", "duration", "elapsed")

#: Nouns naming a configured setting, not a measurement of the code.
CONFIGURATION_NOUNS: Final[tuple[str, ...]] = (
    "timeout",
    "time-out",
    "time limit",
    "cap",
    "tolerance",
)

#: Words that may join a configuration noun to the value it is set to ("timeout of 900 s").
CONFIGURATION_LINKS: Final[tuple[str, ...]] = ("of", "is", "at", "to", "=", ":")

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

#: Where a sidecar or baseline document records named measurements -- never ``config``.
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

#: The dimensions of a figure or a recorded value. Base units: the second (duration), one event
#: per second (rate), the plain number (ratio, percent).
DURATION: Final[str] = "duration"
RATE: Final[str] = "rate"
RATIO: Final[str] = "ratio"
PERCENT: Final[str] = "percent"

#: How a value in each dimension's base unit is written in a reason message.
BASE_UNIT_LABELS: Final[dict[str, str]] = {DURATION: "s", RATE: "/s", RATIO: "×", PERCENT: "%"}

#: Field-name tokens that make a recorded value a speed ratio (``fnet_speedup``).
SPEED_FIELD_TOKENS: Final[frozenset[str]] = frozenset({"speedup", "slowdown"})

#: Field-name tokens naming a timing quantity; with a final percent token, a percentage of it
#: (``latency_reduction_pct``).
TIMING_FIELD_TOKENS: Final[frozenset[str]] = frozenset(
    {"latency", "throughput", "time", "wall", "clock", "runtime", "duration", "elapsed"}
)

#: Final field-name tokens that make a speed or timing field a percentage.
PERCENT_FIELD_TOKENS: Final[frozenset[str]] = frozenset({"pct", "percent"})

#: The field-name token that, followed by a denominator, makes a rate (``sims_per_second``).
PER_FIELD_TOKEN: Final[str] = "per"

#: Fewest significant digits a figure may state and still identify one recorded value. "2 s"
#: matches anything in [1.5, 2.5] s, a quarter of its own size either way.
MIN_TRACEABLE_SIGNIFICANT_DIGITS: Final[int] = 2

#: Relative slack on the rounding window, so binary floating point cannot reject an exact match.
FLOAT_SLACK: Final[float] = 1e-9

# --------------------------------------------------------------------------------------
# patterns
# --------------------------------------------------------------------------------------


def _alternation(words: Iterable[str]) -> str:
    """Regex alternation over ``words``, longest first so ``secs`` is tried before ``s``."""
    return "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))


_DURATION_UNITS: Final[dict[str, float]] = {**SUBSECOND_UNITS, **CONTEXTUAL_DURATION_UNITS}
_RATE_UNITS_BY_LOWER: Final[dict[str, float]] = {k.lower(): v for k, v in RATE_UNITS.items()}
_FIELD_DURATION_UNITS: Final[dict[str, float]] = {
    unit: seconds for unit, seconds in _DURATION_UNITS.items() if unit not in FIELD_UNIT_EXCLUSIONS
}

_NUMBER: Final[str] = r"(?<![\w.])\d[\d,]*(?:\.\d+)?"
#: Between a number and its unit: "45 ms", "45ms", "a 12-ms pass", "10-fold".
_GAP: Final[str] = r"\s*-?"
_SUBSECOND_UNIT: Final[str] = rf"(?:{_alternation(SUBSECOND_UNITS)})(?!\w)"
_CONTEXTUAL_UNIT: Final[str] = rf"(?:{_alternation(CONTEXTUAL_DURATION_UNITS)})(?!\w)"
_DENOMINATOR: Final[str] = rf"(?:{_alternation(THROUGHPUT_DENOMINATORS)})(?!\w)"
#: A rate unit is a whole word: the look-behind keeps "GHz" (a clock speed) from reading as "Hz".
_RATE_UNIT: Final[str] = rf"(?<![A-Za-z])(?:{_alternation(RATE_UNITS)})(?!\w)"
_THROUGHPUT_UNIT: Final[str] = rf"/\s*{_DENOMINATOR}|per\s+{_DENOMINATOR}|{_RATE_UNIT}"
_PER_UNIT: Final[str] = rf"(?:\bper\s+|/\s*)(?:{_alternation(PER_UNIT_NOUNS)})\b"
_CONFIGURATION: Final[str] = rf"(?:{_alternation(CONFIGURATION_NOUNS)})s?\b"

LATENCY_PATTERN: Final[re.Pattern[str]] = re.compile(rf"{_NUMBER}{_GAP}{_SUBSECOND_UNIT}")
SECONDS_PATTERN: Final[re.Pattern[str]] = re.compile(rf"{_NUMBER}{_GAP}{_CONTEXTUAL_UNIT}")
THROUGHPUT_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"{_NUMBER}\s*(?:[A-Za-z][\w-]*\s*){{0,2}}(?:{_THROUGHPUT_UNIT})"
)
SPEEDUP_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"{_NUMBER}{_GAP}(?:{_alternation(SPEEDUP_SYMBOLS)})(?![\w×])"
)
SPEED_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(_alternation(SPEED_WORDS), re.IGNORECASE)
SPEED_CONTEXT_PATTERN: Final[re.Pattern[str]] = re.compile(
    _alternation((*SPEED_WORDS, *TIMING_QUANTITY_WORDS)), re.IGNORECASE
)
MEASURED_QUANTITY_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"{_alternation(MEASURED_QUANTITY_WORDS)}|{_PER_UNIT}", re.IGNORECASE
)
# A setting written after its value ("900 s wall-clock timeout") or before it ("timeout of 900 s").
_SETTING_AFTER_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"(?:[\s-]+[A-Za-z][\w-]*)?[\s-]+{_CONFIGURATION}", re.IGNORECASE
)
_SETTING_BEFORE_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"\b{_CONFIGURATION}[`'\"]?\s*(?:(?:{_alternation(CONFIGURATION_LINKS)})\s*)?$",
    re.IGNORECASE,
)
_CONFIGURATION_PATTERN: Final[re.Pattern[str]] = re.compile(rf"\b{_CONFIGURATION}", re.IGNORECASE)

# Table headers. Sub-second unit symbols are matched case-sensitively -- an "MS-SSIM" column is
# not a millisecond column -- while words, rates and bracketed units are not.
_HEADER_SUBSECOND_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"(?<![\w.])(?P<unit>{_alternation(SUBSECOND_UNITS)})(?!\w)"
)
_HEADER_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"{_THROUGHPUT_UNIT}|{_alternation((*SPEED_WORDS, *MEASURED_QUANTITY_WORDS))}",
    re.IGNORECASE,
)
_HEADER_DURATION_UNIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"[(\[](?P<unit>{_alternation(CONTEXTUAL_DURATION_UNITS)})[)\]]", re.IGNORECASE
)
_HEADER_DURATION_NOUN_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:{_alternation(DURATION_HEADER_NOUNS)})\b", re.IGNORECASE
)
_PER_UNIT_PATTERN: Final[re.Pattern[str]] = re.compile(_PER_UNIT, re.IGNORECASE)

# The unit a detected token ends with -- rate forms first, so the "s" of "sims/sec" is not read as
# seconds. Each named group appears once per compiled pattern.
_RATE_TAIL: Final[str] = (
    rf"/\s*(?P<denominator>{_alternation(THROUGHPUT_DENOMINATORS)})(?!\w)"
    rf"|per\s+(?P<per>{_alternation(THROUGHPUT_DENOMINATORS)})(?!\w)"
    rf"|(?<![A-Za-z])(?P<rate>{_alternation(RATE_UNITS)})(?!\w)"
)
_UNIT_TAIL: Final[str] = (
    rf"-?(?:{_RATE_TAIL}|(?P<percent>{re.escape(PERCENT_SYMBOL)})"
    rf"|(?P<ratio>{_alternation(RATIO_SYMBOLS)})(?![\w×])"
    rf"|(?P<duration>{_alternation(_DURATION_UNITS)})(?!\w))"
)
_TOKEN_UNIT_PATTERN: Final[re.Pattern[str]] = re.compile(rf"{_UNIT_TAIL}\s*$")
_CELL_FIGURE_PATTERN: Final[re.Pattern[str]] = re.compile(rf"{_NUMBER}(?:\s*{_UNIT_TAIL})?")
_HEADER_RATE_PATTERN: Final[re.Pattern[str]] = re.compile(_RATE_TAIL, re.IGNORECASE)
_ANY_NUMBER: Final[re.Pattern[str]] = re.compile(_NUMBER)
_FIELD_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[\W_]+")

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

    ``header`` is the column header a :data:`TABLE_COLUMN_KIND` number sits under -- where a bare
    cell's unit is written -- and empty for every other kind.
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
    """A dimension, and one unit's size in its base unit: ``Unit(RATE, 1 / 60)`` is per minute."""

    dimension: str
    scale: float


@dataclass(frozen=True)
class Figure:
    """A performance number read as a quantity: its value, written precision and unit."""

    value: float
    decimals: int
    significant_digits: int
    unit: Unit | None


@dataclass(frozen=True)
class RecordedValue:
    """One named timing measurement a run records, and where."""

    source: str
    field: str
    value: float
    unit: Unit


# --------------------------------------------------------------------------------------
# finding performance numbers
# --------------------------------------------------------------------------------------


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
        True for a latency or throughput unit ("Inference (ms)", "Sims/sec", "FPS"), a speed or
        measured-quantity word ("Speedup", "Latency"), or a duration noun with a bracketed unit
        or a rate ("Runtime (s)", "Time per move"); False for any header naming a setting
        ("Timeout (s)").

    """
    if _CONFIGURATION_PATTERN.search(header):
        return False
    if _HEADER_SUBSECOND_PATTERN.search(header) or _HEADER_WORD_PATTERN.search(header):
        return True
    return bool(
        _HEADER_DURATION_NOUN_PATTERN.search(header)
        and (_HEADER_DURATION_UNIT_PATTERN.search(header) or _PER_UNIT_PATTERN.search(header))
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


def _is_setting(sentence: str, match: re.Match[str]) -> bool:
    """Whether ``match`` is the value of a configured setting named next to it."""
    return bool(
        _SETTING_AFTER_PATTERN.match(sentence, match.end())
        or _SETTING_BEFORE_PATTERN.search(sentence, 0, match.start())
    )


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
            if SPEED_CONTEXT_PATTERN.search(sentence[lo : match.end() + SPEEDUP_WINDOW_CHARS]):
                hits.append(("speedup", match))
        found += [
            (offset + m.start(), kind, m.group(0))
            for kind, m in hits
            if not _is_setting(sentence, m)
        ]
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
            if match := _CELL_FIGURE_PATTERN.search(cell):
                kind, token = TABLE_COLUMN_KIND, match.group(0)
                found.append(PerformanceNumber(line_no, line, kind, token, header[index]))
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
        return found + _column_numbers(block)
    text = block.text
    line_starts = [0]
    for _, line in block.lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)
    numbers = []
    for offset, kind, token in inline_numbers(text):
        line_no, line = block.lines[bisect.bisect_right(line_starts, offset) - 1]
        numbers.append(PerformanceNumber(line_no, line, kind, token))
    return numbers


# --------------------------------------------------------------------------------------
# units: of a written figure, and of a recorded field
# --------------------------------------------------------------------------------------


def _rate_unit(match: re.Match[str]) -> Unit:
    """The rate a ``_RATE_TAIL`` match names."""
    if rate := match.group("rate"):
        return Unit(RATE, _RATE_UNITS_BY_LOWER[rate.lower()])
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
        A rate for "Sims/sec" or "FPS", a percentage for a "%" column, a ratio for a speed word
        ("Speedup"), a duration for "(ms)" or "Runtime (s)"; ``None`` when the header names a
        quantity but no unit ("Throughput", "Time per move").

    """
    if match := _HEADER_RATE_PATTERN.search(header):
        return _rate_unit(match)
    if PERCENT_SYMBOL in header:
        return Unit(PERCENT, 1.0)
    if SPEED_WORD_PATTERN.search(header):
        return Unit(RATIO, 1.0)
    if match := _HEADER_SUBSECOND_PATTERN.search(header):
        return Unit(DURATION, SUBSECOND_UNITS[match.group("unit")])
    if match := _HEADER_DURATION_UNIT_PATTERN.search(header):
        return Unit(DURATION, CONTEXTUAL_DURATION_UNITS[match.group("unit").lower()])
    return None


def figure_of(number: PerformanceNumber) -> Figure:
    """Read a performance number as a quantity.

    Args:
        number: A number from :func:`performance_numbers`; every token starts with its digits.

    Returns:
        Its value; its precision, as decimal places and significant digits written (a trailing
        zero is written, so "180" is exact to the unit); and its unit -- the one written after
        the digits, else, under a table header, the header's.

    Raises:
        ValueError: If the token does not start with a number.

    """
    match = _ANY_NUMBER.match(number.token)
    if match is None:
        raise ValueError(f"not a performance-number token: {number.token!r}")
    digits = match.group(0).replace(",", "")
    unit = _tail_unit(number.token[match.end() :])
    if unit is None and number.kind == TABLE_COLUMN_KIND:
        unit = header_unit(number.header)
    return Figure(
        value=float(digits),
        decimals=len(digits.partition(".")[2]),
        significant_digits=len(digits.replace(".", "").lstrip("0")),
        unit=unit,
    )


def field_unit(name: str) -> Unit | None:
    """What a recorded field measures, read from its snake- or kebab-case name.

    Args:
        name: A sidecar metric or counter name, or a baseline ``metric_name``.

    Returns:
        A rate for ``<x>_per_<denominator>`` or a rate-unit token (``fps``, ``hz``); a ratio for
        a ``speedup``/``slowdown`` field, or a percentage when such a field -- or one naming a
        timing quantity -- ends in ``pct``/``percent``; a duration for a time-unit token
        (``wall_time_seconds``, ``wall_seconds_total``); otherwise ``None``.

    """
    tokens = [token for token in _FIELD_TOKEN_SPLIT.split(name.lower()) if token]
    for before, token in zip(tokens, tokens[1:]):
        if before == PER_FIELD_TOKEN and token in THROUGHPUT_DENOMINATORS:
            return Unit(RATE, 1.0 / THROUGHPUT_DENOMINATORS[token])
    rates = [_RATE_UNITS_BY_LOWER[token] for token in tokens if token in _RATE_UNITS_BY_LOWER]
    if rates:
        return Unit(RATE, rates[0])
    is_percent = bool(tokens) and tokens[-1] in PERCENT_FIELD_TOKENS
    if not SPEED_FIELD_TOKENS.isdisjoint(tokens):
        return Unit(PERCENT if is_percent else RATIO, 1.0)
    if is_percent and not TIMING_FIELD_TOKENS.isdisjoint(tokens):
        return Unit(PERCENT, 1.0)
    durations = [_FIELD_DURATION_UNITS[token] for token in tokens if token in _FIELD_DURATION_UNITS]
    return Unit(DURATION, durations[0]) if durations else None


# --------------------------------------------------------------------------------------
# citations, provenance and named measurements
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
    """The file that records ``artifact``'s run provenance and named measurements.

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
    """Why ``artifact``'s run provenance cannot back a performance number, or ``None``.

    Args:
        repo_root: The repository root the path is relative to.
        artifact: A cited path from :func:`cited_artifacts`.

    Returns:
        A one-line reason, or ``None`` when the artifact exists and its provenance records a
        real hardware tag. Whether a figure is one of its run's named measurements is
        :func:`trace_problem`'s question.

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


def _as_dict(value: object) -> dict[object, object]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _measured(value: object) -> float | None:
    """``value`` as a finite float, or ``None`` for text, booleans, NaN and infinities."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def recorded_values(repo_root: Path, artifact: str) -> list[RecordedValue]:
    """Every named timing measurement the run behind ``artifact`` records.

    Args:
        repo_root: The repository root the path is relative to.
        artifact: A citation :func:`citation_problem` accepts, so its record is a JSON object.

    Returns:
        The sidecar's ``metrics`` and arm ``counters``, or the baseline's ``entries``, whose
        names state a timing unit (:func:`field_unit`) and whose values are finite numbers.
        Never a ``config`` value, and never a CSV row.

    """
    record = provenance_path(artifact)
    document = _as_dict(json.loads((repo_root / record).read_text(encoding="utf-8")))
    pairs = list(_as_dict(document.get(METRICS_FIELD)).items())
    for arm in _as_list(document.get(ARMS_FIELD)):
        pairs += _as_dict(_as_dict(arm).get(COUNTERS_FIELD)).items()
    for entry in map(_as_dict, _as_list(document.get(ENTRIES_FIELD))):
        pairs.append((entry.get(ENTRY_NAME_FIELD), entry.get(ENTRY_VALUE_FIELD)))
    return [
        RecordedValue(record, name, value, unit)
        for name, raw in pairs
        if isinstance(name, str)
        and (unit := field_unit(name)) is not None
        and (value := _measured(raw)) is not None
    ]


def _in_units(recorded: RecordedValue, unit: Unit) -> float:
    return recorded.value * recorded.unit.scale / unit.scale


def trace_problem(
    number: PerformanceNumber, evidence: list[RecordedValue], sources: list[str]
) -> str | None:
    """Why ``number`` is not one of the named measurements in ``evidence``, or ``None``.

    Args:
        number: A performance number.
        evidence: Every named timing measurement of the runs its block's backing citations name.
        sources: Those citations, for the message.

    Returns:
        ``None`` when a measurement of the figure's dimension, converted to its unit, rounds to
        it at its written precision; otherwise a one-line reason.

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
        return f"the run(s) behind {cited} record no named {unit.dimension} measurement"
    window = 0.5 * 10.0**-figure.decimals + FLOAT_SLACK * max(1.0, abs(figure.value))
    nearest = min(same, key=lambda recorded: abs(_in_units(recorded, unit) - figure.value))
    if abs(_in_units(nearest, unit) - figure.value) <= window:
        return None
    base = f"{nearest.value * nearest.unit.scale:.6g} {BASE_UNIT_LABELS[unit.dimension]}"
    return (
        f"{number.token!r} is not a named {unit.dimension} measurement of the run(s) behind "
        f"{cited} at its stated precision (nearest: {base}, {nearest.source}:{nearest.field})"
    )


def unbacked_performance_numbers(markdown: str, repo_root: Path) -> list[UnbackedNumber]:
    """Performance numbers in ``markdown`` that no artifact cited in their block backs.

    Args:
        markdown: A Markdown document.
        repo_root: The repository root citations are resolved against.

    Returns:
        One entry per unbacked number, with why each citation in its block (if any) failed its
        provenance check and, when one passed, why the number is not among its run's named
        timing measurements.

    """
    unbacked: list[UnbackedNumber] = []
    for block in split_blocks(markdown):
        numbers = performance_numbers(block)
        if not numbers:
            continue
        cited = cited_artifacts(block.text)
        problems = [citation_problem(repo_root, path) for path in cited]
        failed = tuple(problem for problem in problems if problem is not None)
        backing = [path for path, problem in zip(cited, problems) if problem is None]
        if not backing:
            unbacked += [UnbackedNumber(n, failed or (NO_CITATION_REASON,)) for n in numbers]
            continue
        evidence = [value for path in backing for value in recorded_values(repo_root, path)]
        for number in numbers:
            if (problem := trace_problem(number, evidence, backing)) is not None:
                unbacked.append(UnbackedNumber(number, (*failed, problem)))
    return unbacked
