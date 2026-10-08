"""Unit tests for ``tests/support/perf_claims.py``, the performance-number scanner.

The scanner is the load-bearing half of ``tests/docs/test_performance_claims.py``: a unit it
cannot see is a claim that guard can never reject. So every unit alternative, speed word,
timing-quantity word and measured-quantity word has a literal positive below, and the literals
are written out rather than generated from the scanner's own vocabulary -- a case generated from
``SUBSECOND_UNITS`` would vanish together with the unit it exists to test, and the test would
pass on exactly the mutation it should kill. The reviewer's fourteen false negatives (2026-10-08:
five table headers, four speed verbs, five rates and percentages) are among them verbatim.

The negatives are the text the guard must never reject: "takes ~2 minutes", configured settings
(timeouts, tolerances -- one case per setting noun and link word, including the reviewer's two
false positives), asymptotic ``O(N)``, accuracy ratios, board sizes and clock speeds.

Every backing test runs on a miniature repository in ``tmp_path``, never on committed artifacts,
so no recorded mutation kill can stop working when a committed file is re-recorded -- which is
how kill #6 died once. Mutation kills are recorded in ``tests/docs/test_performance_claims.py``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Final

import pytest

from tests.support.perf_claims import (
    DURATION,
    NO_CITATION_REASON,
    PERCENT,
    PLACEHOLDER_HARDWARE_TAGS,
    RATE,
    RATIO,
    TABLE_COLUMN_KIND,
    Block,
    PerformanceNumber,
    RecordedValue,
    Unit,
    citation_problem,
    cited_artifacts,
    field_unit,
    figure_of,
    header_unit,
    inline_numbers,
    is_measurement_header,
    performance_numbers,
    provenance_path,
    recorded_values,
    split_blocks,
    trace_problem,
    unbacked_performance_numbers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: ``id -> (text, token the scanner must report)``. One case per unit alternative, speed word,
#: timing-quantity word, measured-quantity word and ``per <noun>``.
SYNTHETIC_POSITIVES: Final[dict[str, tuple[str, str]]] = {
    "unit-ms": ("A forward pass takes 45 ms.", "45 ms"),
    "unit-msec": ("A forward pass takes 45 msec.", "45 msec"),
    "unit-millisecond": ("Each call costs 1 millisecond.", "1 millisecond"),
    "unit-milliseconds": ("Each call costs 30 milliseconds.", "30 milliseconds"),
    "unit-micro-sign": ("A kernel launch costs 12 µs.", "12 µs"),
    "unit-greek-mu": ("A kernel launch costs 12 μs.", "12 μs"),
    "unit-us": ("A kernel launch costs 12 us.", "12 us"),
    "unit-usec": ("A kernel launch costs 12 usec.", "12 usec"),
    "unit-microsecond": ("A lookup costs 1 microsecond.", "1 microsecond"),
    "unit-microseconds": ("A lookup costs 9 microseconds.", "9 microseconds"),
    "unit-ns": ("A lookup costs 150 ns.", "150 ns"),
    "unit-nanosecond": ("A lookup costs 1 nanosecond.", "1 nanosecond"),
    "unit-nanoseconds": ("A lookup costs 90 nanoseconds.", "90 nanoseconds"),
    "unit-hyphenated": ("A forward pass is a 12-ms affair.", "12-ms"),
    "unit-s": ("Inference takes 0.4 s on one core.", "0.4 s"),
    "unit-sec": ("Inference takes 2 sec on one core.", "2 sec"),
    "unit-secs": ("Inference takes 2 secs on one core.", "2 secs"),
    "unit-second": ("Inference takes 1 second on one core.", "1 second"),
    "unit-seconds": ("Inference takes 3 seconds on one core.", "3 seconds"),
    "unit-min": ("The wall-clock cost was 3 min.", "3 min"),
    "unit-mins": ("The wall-clock cost was 3 mins.", "3 mins"),
    "unit-minute": ("Inference took 1 minute.", "1 minute"),
    "unit-minutes": ("Inference took 4 minutes.", "4 minutes"),
    "unit-h": ("The wall-clock cost was 2 h.", "2 h"),
    "unit-hr": ("The wall-clock cost was 2 hr.", "2 hr"),
    "unit-hrs": ("The wall-clock cost was 2 hrs.", "2 hrs"),
    "unit-hour": ("Inference took 1 hour.", "1 hour"),
    "unit-hours": ("Inference took 2 hours.", "2 hours"),
    "rate-slash-s": ("Training reaches 1,200 it/s.", "1,200 it/s"),
    "rate-slash-sec": ("MCTS runs 670 sims/sec.", "670 sims/sec"),
    "rate-slash-second": ("The loader streams 40 samples/second.", "40 samples/second"),
    "rate-slash-min": ("Self-play produces 40 games/min.", "40 games/min"),
    "rate-slash-minute": ("Self-play produces 40 games/minute.", "40 games/minute"),
    "rate-slash-h": ("Self-play produces 900 games/h.", "900 games/h"),
    "rate-slash-hr": ("Self-play produces 900 games/hr.", "900 games/hr"),
    "rate-slash-hour": ("Self-play generates 1,200 games/hour.", "1,200 games/hour"),
    "rate-per-second": ("It evaluates 900 positions per second.", "900 positions per second"),
    "rate-per-minute": ("Self-play runs 2,000 games per minute.", "2,000 games per minute"),
    "rate-per-hour": ("Self-play runs 50 games per hour.", "50 games per hour"),
    "rate-FPS": ("Rendering holds 60 FPS.", "60 FPS"),
    "rate-fps": ("Rendering holds 60fps.", "60fps"),
    "rate-Hz": ("The policy loop runs at 900 Hz.", "900 Hz"),
    "rate-kHz": ("Sampling hits 3 kHz.", "3 kHz"),
    "ratio-times-sign": ("FNet is 3.2× faster than softmax.", "3.2×"),
    "ratio-x": ("FNet is 3.2x faster than softmax.", "3.2x"),
    "ratio-X": ("FNet is 3.2X faster than softmax.", "3.2X"),
    "ratio-hyphen-fold": ("FNet gives a 10-fold speedup.", "10-fold"),
    "ratio-fold": ("Search is 4 fold faster.", "4 fold"),
    "ratio-percent": ("The cache made search 30% faster.", "30%"),
    "word-faster": ("Search got 2× faster.", "2×"),
    "word-slower": ("Search got 2× slower.", "2×"),
    "word-quicker": ("Search is 3× quicker.", "3×"),
    "word-speedup": ("We measured a 10× speedup.", "10×"),
    "word-speed-up": ("We measured a 10× speed-up.", "10×"),
    "word-speeds-up": ("FNet speeds up leaf evaluation by 10x.", "10x"),
    "word-speed-up-verb": ("Caching can speed up search 4×.", "4×"),
    "word-sped-up": ("MCTS was sped up 4×.", "4×"),
    "word-slowdown": ("We measured a 2× slowdown.", "2×"),
    "word-accelerate": ("The FFT path accelerates rollouts 5×.", "5×"),
    "timing-latency": ("Caching reduces inference latency by 80%.", "80%"),
    "timing-throughput": ("Batching raised throughput by 35%.", "35%"),
    "timing-wall-clock-hyphen": ("It cut the wall-clock cost by 40%.", "40%"),
    "timing-wall-clock-space": ("It cut the wall clock by 40%.", "40%"),
    "timing-wall-time": ("It cut the wall time by 40%.", "40%"),
    "timing-step-time": ("It cut the step time by 40%.", "40%"),
    "quantity-latency": ("The p95 latency is 2 s.", "2 s"),
    "quantity-inference": ("Inference is 2 s here.", "2 s"),
    "quantity-throughput": ("At that throughput a batch is 2 s.", "2 s"),
    "quantity-wall-clock-hyphen": ("The wall-clock cost was 2 s.", "2 s"),
    "quantity-wall-clock-space": ("The wall clock showed 2 s.", "2 s"),
    "quantity-wall-time": ("The wall time was 2 s.", "2 s"),
    "quantity-step-time": ("The step time was 2 s.", "2 s"),
    "per-move": ("Search spends 2 s per move.", "2 s"),
    "per-slash-move": ("Search takes 0.4 s/move.", "0.4 s"),
    "per-step": ("Training spends 2 s per step.", "2 s"),
    "per-iteration": ("Training spends 2 s per iteration.", "2 s"),
    "per-epoch": ("Training spends 2 s per epoch.", "2 s"),
    "per-sample": ("Generation spends 2 s per sample.", "2 s"),
    "per-call": ("The client spends 2 s per call.", "2 s"),
    "per-query": ("The index spends 2 s per query.", "2 s"),
    "per-solve": ("The solver spends 2 s per solve.", "2 s"),
    "per-frame": ("The codec spends 2 s per frame.", "2 s"),
    "per-batch": ("Training spends 2 s per batch.", "2 s"),
    "per-token": ("Decoding spends 2 s per token.", "2 s"),
    "per-game": ("Self-play spends 2 s per game.", "2 s"),
    "per-search": ("MCTS spends 2 s per search.", "2 s"),
    "per-rollout": ("MCTS spends 2 s per rollout.", "2 s"),
    "per-evaluation": ("The net spends 2 s per evaluation.", "2 s"),
    "per-position": ("The engine spends 2 s per position.", "2 s"),
    "per-simulation": ("MCTS spends 2 s per simulation.", "2 s"),
    # "Budget" is deliberately not a setting: it frames a measured claim as often as a setting.
    "budget-is-a-claim": ("Search stays within a 100 ms budget.", "100 ms"),
}

#: Instructional, configuration or non-performance text the scanner must not report.
SYNTHETIC_NEGATIVES: Final[dict[str, str]] = {
    "minutes": "Quick validation suite (~5 min, CPU); the full run takes ~2 minutes.",
    "bare-seconds": "Two hermetic suites run in ~2 s with no network.",
    "timeout": "The job has a 45-minute cap and a 30 s timeout.",
    "config-value": "Set `timeout_seconds: 3600` and `max_dof=600`.",
    "asymptotic": "Galerkin attention is O(N × d²), softmax O(N² × d), FNet O(N log N).",
    "accuracy-ratio": "A retrained CNN is ~14× more accurate than the zero-shot operator.",
    "far-speed-word": (
        "The operator is ~14× more accurate when retrained, although its inference is faster."
    ),
    "coverage": "An 85% branch coverage gate is enforced in CI.",
    "board-size": "Train on 9×9 and evaluate at 19×19 or 9x9.",
    "hex": "The flag is 0x1F.",
    "word-suffix": "It needs 5 sims and 3 seeds.",
    "rate-word-suffix": "See 3 files/sections and 2 pages/seconds-later.",
    "fold-symmetry": "Full Go rules with 8-fold symmetry.",
    "clock-speed": "A 3.5 GHz CPU runs the wall-clock suite.",
    # The reviewer's two false positives (2026-10-08), then one case per setting noun and link.
    "setting-wall-clock-timeout": "The E2E job has a 900 s wall-clock timeout.",
    "setting-tolerance": "Convergence is slower below a 0.5% tolerance.",
    "setting-timeout-of": "The wall-clock timeout of 900 s applies.",
    "setting-time-out-is": "The wall-clock time-out is 900 s.",
    "setting-time-limit-to": "Set the wall-clock time limit to 2 hours.",
    "setting-cap-equals": "The latency cap = 50 ms.",
    "setting-cap-at": "The latency cap at 50 ms is generous.",
    "setting-tolerance-colon": "The step time tolerance: 5%.",
}

REMOVED_ROW: Final[str] = "| Galerkin+FNet | 19×19 | 12 | 670 |"

#: The removed README table's header and row shape, for the backing tests below.
TABLE_HEADER: Final[str] = (
    "| Model | Board Size | Inference (ms) | MCTS Sims/sec |\n"
    "|-------|------------|----------------|---------------|"
)


@pytest.mark.parametrize(("text", "token"), SYNTHETIC_POSITIVES.values(), ids=SYNTHETIC_POSITIVES)
def test_synthetic_positive_is_detected(text: str, token: str) -> None:
    """Each unit, speed word and measured-quantity word is visible to the scanner."""
    tokens = [found for _, _, found in inline_numbers(text)]
    assert token in tokens, f"{text!r}: expected {token!r}, scanner reported {tokens}"


@pytest.mark.parametrize("text", SYNTHETIC_NEGATIVES.values(), ids=SYNTHETIC_NEGATIVES)
def test_instructional_text_is_not_a_performance_number(text: str) -> None:
    assert inline_numbers(text) == []


def test_seconds_need_a_measured_quantity_in_the_same_sentence() -> None:
    text = "Inference is O(N). The suite then takes 2 s."
    assert inline_numbers(text) == []


def test_a_sentence_wrapped_across_lines_keeps_its_measured_quantity() -> None:
    """Prose is scanned per block: the quantity and the number may sit on different lines."""
    wrapped = "Inference on the 19×19 board at batch size 1\ntakes 0.4 s on a laptop."
    (block,) = split_blocks(wrapped)
    (number,) = performance_numbers(block)
    assert (number.line_no, number.token) == (2, "0.4 s")


def test_speed_word_must_sit_within_the_window() -> None:
    near = "FNet leaf evaluation is 5× faster."
    far = "It is 5× cheaper to store, and that makes the whole evaluation pipeline much faster."
    assert [token for _, _, token in inline_numbers(near)] == ["5×"]
    assert inline_numbers(far) == []


def test_a_setting_phrase_does_not_hide_a_neighbouring_claim() -> None:
    """Only the figure the setting names is exempt; a comma ends the setting's noun phrase."""
    text = "Inference takes 12 ms, well inside the 900 s wall-clock timeout."
    assert [token for _, _, token in inline_numbers(text)] == ["12 ms"]


# --------------------------------------------------------------------------------------
# block splitting
# --------------------------------------------------------------------------------------


def test_split_blocks_separates_paragraphs_items_headings_and_tables() -> None:
    markdown = "\n".join(
        [
            "## Heading",
            "First paragraph,",
            "continued.",
            "",
            "- item one",
            "  continuation",
            "- item two",
            "| a | b |",
            "|---|---|",
            "| 1 | 2 |",
            "Trailing prose.",
        ]
    )
    blocks = split_blocks(markdown)
    assert [(block.kind, [n for n, _ in block.lines]) for block in blocks] == [
        ("text", [1]),
        ("text", [2, 3]),
        ("text", [5, 6]),
        ("text", [7]),
        ("table", [8, 9, 10]),
        ("text", [11]),
    ]


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_fenced_code_is_instructional_and_skipped(fence: str) -> None:
    markdown = f"{fence}bash\npytest  # ~10 s on one core, 45 ms per test\n{fence}\nAfter."
    assert [block.text for block in split_blocks(markdown)] == ["After."]


def test_inline_code_is_not_skipped() -> None:
    """Backticks must not be an evasion."""
    (block,) = split_blocks("The model answers in `45 ms`.")
    assert [number.token for number in performance_numbers(block)] == ["45 ms"]


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "Inference (ms)",
        "Latency p95 (µs)",
        "MCTS Sims/sec",
        "Throughput",
        "FPS",
        "Rate (Hz)",
        "Games/hour",
        "Speedup",
        "Wall time (s)",
        "Time (s)",
        "Rate (per second)",
        # The reviewer's five (2026-10-08), then the bracket form.
        "Runtime (s)",
        "Duration (s)",
        "Elapsed (s)",
        "Train time (min)",
        "Time per move",
        "Elapsed [s]",
    ],
)
def test_measurement_headers_are_recognised(header: str) -> None:
    assert is_measurement_header(header)


@pytest.mark.parametrize(
    "header",
    [
        "Board Size",
        "Model",
        "MS-SSIM",
        "Timeout (s)",
        "Time limit (s)",
        "Speedup tolerance (%)",
        "Coverage (%)",
        "Clock (GHz)",
        "Status",
        "Columns",
    ],
)
def test_non_measurement_headers_are_not(header: str) -> None:
    assert not is_measurement_header(header)


def test_numbers_under_a_measurement_header_are_reported() -> None:
    """The removed table's shape: bare cells, units only in the header."""
    (block,) = split_blocks(f"{TABLE_HEADER}\n{REMOVED_ROW}")
    numbers = performance_numbers(block)
    assert [(n.kind, n.token, n.line_no, n.header) for n in numbers] == [
        (TABLE_COLUMN_KIND, "12", 3, "Inference (ms)"),
        (TABLE_COLUMN_KIND, "670", 3, "MCTS Sims/sec"),
    ]


def test_a_measured_cell_without_a_number_is_skipped() -> None:
    markdown = "| Model | Inference (ms) |\n|---|---|\n| a | n/a |\n| b | — |\n| c | 9 |"
    (block,) = split_blocks(markdown)
    assert [(n.line_no, n.token) for n in performance_numbers(block)] == [(5, "9")]


def test_table_cells_honour_escaped_pipes() -> None:
    markdown = "| Name | Latency (ms) |\n|---|---|\n| a \\| b | 7 |"
    (block,) = split_blocks(markdown)
    assert [n.token for n in performance_numbers(block)] == ["7"]


def test_a_table_row_is_its_own_sentence() -> None:
    markdown = "| Step | Note |\n|---|---|\n| train | 2 s per epoch |\n| eval | 3 s |"
    (block,) = split_blocks(markdown)
    assert [(n.line_no, n.token) for n in performance_numbers(block)] == [(3, "2 s")]


def test_a_number_found_twice_is_reported_once() -> None:
    """Inline *and* by column: one report, keeping the specific inline kind."""
    markdown = "| Model | Latency (ms) |\n|---|---|\n| a | 45 ms |"
    (block,) = split_blocks(markdown)
    assert [(n.kind, n.token) for n in performance_numbers(block)] == [("latency", "45 ms")]


def test_a_cell_keeps_its_own_unit_under_a_header() -> None:
    markdown = "| Model | Time per move |\n|---|---|\n| a | 0.4 s |"
    (block,) = split_blocks(markdown)
    assert [(n.kind, n.token) for n in performance_numbers(block)] == [(TABLE_COLUMN_KIND, "0.4 s")]


def test_a_header_only_table_reports_nothing() -> None:
    (block,) = split_blocks("| Latency (ms) |\n|---|")
    assert performance_numbers(block) == []


# --------------------------------------------------------------------------------------
# units of a written figure and of a recorded field
# --------------------------------------------------------------------------------------


def _number(token: str, kind: str = "latency", header: str = "") -> PerformanceNumber:
    return PerformanceNumber(1, token, kind, token, header)


@pytest.mark.parametrize(
    ("token", "kind", "header", "expected"),
    [
        ("45 ms", "latency", "", (45.0, 0, 2, Unit(DURATION, 1e-3))),
        ("0.4 s", "latency", "", (0.4, 1, 1, Unit(DURATION, 1.0))),
        ("12-ms", "latency", "", (12.0, 0, 2, Unit(DURATION, 1e-3))),
        ("2 hours", "latency", "", (2.0, 0, 1, Unit(DURATION, 3600.0))),
        ("1,200 it/s", "throughput", "", (1200.0, 0, 4, Unit(RATE, 1.0))),
        ("2,000 games per minute", "throughput", "", (2000.0, 0, 4, Unit(RATE, 1 / 60))),
        ("900 Hz", "throughput", "", (900.0, 0, 3, Unit(RATE, 1.0))),
        ("3 kHz", "throughput", "", (3.0, 0, 1, Unit(RATE, 1e3))),
        ("3.2×", "speedup", "", (3.2, 1, 2, Unit(RATIO, 1.0))),
        ("10-fold", "speedup", "", (10.0, 0, 2, Unit(RATIO, 1.0))),
        ("80%", "speedup", "", (80.0, 0, 2, Unit(PERCENT, 1.0))),
        ("0.05", TABLE_COLUMN_KIND, "Inference (ms)", (0.05, 2, 1, Unit(DURATION, 1e-3))),
        ("670", TABLE_COLUMN_KIND, "MCTS Sims/sec", (670.0, 0, 3, Unit(RATE, 1.0))),
        ("30", TABLE_COLUMN_KIND, "Speedup (%)", (30.0, 0, 2, Unit(PERCENT, 1.0))),
        ("3.2", TABLE_COLUMN_KIND, "Speedup", (3.2, 1, 2, Unit(RATIO, 1.0))),
        ("1.5", TABLE_COLUMN_KIND, "Train time (min)", (1.5, 1, 2, Unit(DURATION, 60.0))),
        ("0.4 s", TABLE_COLUMN_KIND, "Time per move", (0.4, 1, 1, Unit(DURATION, 1.0))),
        ("7", TABLE_COLUMN_KIND, "Throughput", (7.0, 0, 1, None)),
    ],
)
def test_figure_of_reads_value_precision_and_unit(
    token: str, kind: str, header: str, expected: tuple[float, int, int, Unit | None]
) -> None:
    figure = figure_of(_number(token, kind, header))
    value, decimals, digits, unit = expected
    assert (figure.value, figure.decimals, figure.significant_digits) == (value, decimals, digits)
    if unit is None:
        assert figure.unit is None
    else:
        assert figure.unit is not None and figure.unit.dimension == unit.dimension
        assert figure.unit.scale == pytest.approx(unit.scale)


def test_figure_of_rejects_a_token_without_a_number() -> None:
    with pytest.raises(ValueError, match="not a performance-number token"):
        figure_of(_number("ms"))


def test_header_unit_is_none_when_the_header_states_no_unit() -> None:
    assert header_unit("Latency") is None


@pytest.mark.parametrize(
    ("name", "dimension", "scale"),
    [
        ("wall_time_seconds", DURATION, 1.0),
        ("matched_wall_time_seconds", DURATION, 1.0),
        ("wall_seconds_total", DURATION, 1.0),
        ("greedy_wall_seconds_at_equal_accuracy", DURATION, 1.0),
        ("stochastic_wall_clock_s", DURATION, 1.0),
        ("latency_ms", DURATION, 1e-3),
        ("train_minutes", DURATION, 60.0),
        ("sims_per_second", RATE, 1.0),
        ("games-per-hour", RATE, 1 / 3600),
        ("games_per_min", RATE, 1 / 60),
        ("render_fps", RATE, 1.0),
        ("update_hz", RATE, 1.0),
        ("fnet_speedup", RATIO, 1.0),
        ("speedup_pct", PERCENT, 1.0),
        ("latency_reduction_pct", PERCENT, 1.0),
    ],
)
def test_field_unit_reads_timing_names(name: str, dimension: str, scale: float) -> None:
    unit = field_unit(name)
    assert unit is not None and unit.dimension == dimension, (name, unit)
    assert unit.scale == pytest.approx(scale)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "l2_ratio_seed_min",
        "primary_tree_depth_min",
        "coverage_pct",
        "n_cache_misses",
        "error_per_dof",
        "matched_dof",
        "break_even_cost_coefficient_alpha1.5",
        "primary_error_per_dof_ratio_at_matched_wall_clock_vs_greedy",
    ],
)
def test_field_unit_ignores_names_that_state_no_timing_unit(name: str) -> None:
    """``min`` here means minimum, and a wall-clock *ratio* states no unit."""
    assert field_unit(name) is None


# --------------------------------------------------------------------------------------
# traceability: a figure must be one of the named measurements, at its written precision
# --------------------------------------------------------------------------------------


def _recorded(field: str, value: float) -> RecordedValue:
    unit = field_unit(field)
    assert unit is not None, field
    return RecordedValue("results/x.run.json", field, value, unit)


@pytest.mark.parametrize(
    ("token", "kind", "header", "field", "value"),
    [
        pytest.param("45 ms", "latency", "", "latency_ms", 45.0, id="exact"),
        pytest.param("45 ms", "latency", "", "latency_ms", 45.4, id="rounds-to-it"),
        pytest.param("45 ms", "latency", "", "wall_time_seconds", 0.045, id="unit-conversion"),
        pytest.param("0.35 s", "latency", "", "wall_seconds", 0.3459, id="decimal-precision"),
        pytest.param(
            "2,000 games per minute", "throughput", "", "games_per_second", 33.33, id="rate"
        ),
        pytest.param("3.2×", "speedup", "", "fnet_speedup", 3.24, id="ratio"),
        pytest.param("80%", "speedup", "", "latency_reduction_pct", 80.3, id="percent"),
        pytest.param("45", TABLE_COLUMN_KIND, "Inference (ms)", "latency_ms", 45.0, id="header"),
    ],
)
def test_a_figure_equal_to_a_named_measurement_traces(
    token: str, kind: str, header: str, field: str, value: float
) -> None:
    number = _number(token, kind, header)
    assert trace_problem(number, [_recorded(field, value)], ["results/x.csv"]) is None


@pytest.mark.parametrize(
    ("token", "kind", "header", "field", "value", "fragment"),
    [
        pytest.param("45 ms", "latency", "", "latency_ms", 45.6, "is not a named", id="outside"),
        pytest.param(
            "0.347 s", "latency", "", "wall_seconds", 0.3459, "is not a named", id="digits"
        ),
        pytest.param(
            "670 sims/sec", "throughput", "", "wall_seconds", 670.0, "no named rate", id="dim"
        ),
        pytest.param("30%", "speedup", "", "fnet_speedup", 1.3, "no named percent", id="derived"),
        pytest.param("5×", "speedup", "", "fnet_speedup", 5.0, "significant digit", id="one-digit"),
        pytest.param(
            "7", TABLE_COLUMN_KIND, "Throughput", "sims_per_second", 7.0, "no unit", id="unitless"
        ),
    ],
)
def test_a_figure_that_is_not_a_named_measurement_does_not_trace(
    token: str, kind: str, header: str, field: str, value: float, fragment: str
) -> None:
    """Nothing is derived, and one significant digit identifies no value -- even an exact one."""
    problem = trace_problem(_number(token, kind, header), [_recorded(field, value)], ["results/x"])
    assert problem is not None and fragment in problem, problem


def test_recorded_values_are_named_timing_measurements_and_nothing_else(tmp_path: Path) -> None:
    """Metrics and arm counters with a timing name and a finite number; never ``config``."""
    sidecar = {
        "hardware_tag": "x86_64-4cpu",
        "config": {"timeout_seconds": 900, "max_wall_seconds": 60},
        "metrics": {
            "matched_wall_time_seconds": 0.35,
            "l2_error_ratio": 0.95,
            "flag_seconds": True,
            "label_seconds": "12",
            "lost_seconds": float("nan"),
        },
        "arms": [{"counters": {"wall_seconds": 70.6, "n_levels": 12}}, "not-an-arm"],
    }
    _write(tmp_path, "results/run.run.json", json.dumps(sidecar))
    found = recorded_values(tmp_path, "results/run.csv")
    assert [(v.source, v.field, v.value) for v in found] == [
        ("results/run.run.json", "matched_wall_time_seconds", 0.35),
        ("results/run.run.json", "wall_seconds", 70.6),
    ]


def test_recorded_values_read_baseline_entries(tmp_path: Path) -> None:
    entries = [
        {"metric_name": "latency_ms", "value": 12.5},
        {"metric_name": "mse", "value": 1e-3},
        {"value": 3.0},
        "junk",
    ]
    _write(
        tmp_path, "config/baselines/b.json", json.dumps({"hardware_tag": "rig", "entries": entries})
    )
    assert [(v.field, v.value) for v in recorded_values(tmp_path, "config/baselines/b.json")] == [
        ("latency_ms", 12.5)
    ]


# --------------------------------------------------------------------------------------
# citations and provenance
# --------------------------------------------------------------------------------------


def test_cited_artifacts_reads_links_backticks_braces_and_relative_forms() -> None:
    text = (
        "See [`results/a.csv`](results/a.csv), `results/b.{csv,run.json}`, "
        "[x](../results/c.csv), https://github.com/o/r/blob/HEAD/config/baselines/d.json. "
        "Not `outputs/e.csv` nor `src/results.py`."
    )
    assert cited_artifacts(text) == [
        "results/a.csv",
        "results/b.csv",
        "results/b.run.json",
        "results/c.csv",
        "config/baselines/d.json",
    ]


@pytest.mark.parametrize(
    ("artifact", "record"),
    [
        ("results/a.csv", "results/a.run.json"),
        ("results/a.png", "results/a.run.json"),
        ("results/a.run.json", "results/a.run.json"),
        ("config/baselines/b.json", "config/baselines/b.json"),
    ],
)
def test_provenance_path(artifact: str, record: str) -> None:
    assert provenance_path(artifact) == record


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sidecar(tag: object, **metrics: float) -> str:
    return json.dumps({"hardware_tag": tag, "metrics": metrics})


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature repository with one artifact of each provenance and evidence state."""
    _write(tmp_path, "results/good.csv", "a,b\n")
    _write(tmp_path, "results/good.run.json", _sidecar("x86_64-4cpu", latency_ms=45.0, p95_ms=48))
    _write(tmp_path, "results/untagged.csv", "a,b\n")
    _write(tmp_path, "results/untagged.run.json", _sidecar("unknown", latency_ms=46.0))
    _write(tmp_path, "results/shouting.csv", "a,b\n")
    _write(tmp_path, "results/shouting.run.json", _sidecar(" UNKNOWN "))
    _write(tmp_path, "results/no_field.csv", "a,b\n")
    _write(tmp_path, "results/no_field.run.json", json.dumps({"seed": 1}))
    _write(tmp_path, "results/list.csv", "a,b\n")
    _write(tmp_path, "results/list.run.json", "[1, 2]")
    _write(tmp_path, "results/broken.csv", "a,b\n")
    _write(tmp_path, "results/broken.run.json", "{not json")
    _write(tmp_path, "results/orphan.csv", "a,b\n")
    _write(tmp_path, "config/baselines/tagged.json", _sidecar("RTX 5060 Ti 16GiB"))
    _write(tmp_path, "config/baselines/empty.json", _sidecar(""))
    _write(tmp_path, "config/baselines/poc.example.json", _sidecar("example"))
    return tmp_path


@pytest.mark.parametrize(
    "artifact", ["results/good.csv", "results/good.run.json", "config/baselines/tagged.json"]
)
def test_a_hardware_tagged_artifact_passes_the_provenance_check(repo: Path, artifact: str) -> None:
    assert citation_problem(repo, artifact) is None


@pytest.mark.parametrize(
    ("artifact", "fragment"),
    [
        ("results/missing.csv", "does not exist"),
        ("results/orphan.csv", "no run-provenance sidecar"),
        ("results/untagged.csv", "records no hardware_tag"),
        ("results/shouting.csv", "records no hardware_tag"),
        ("results/no_field.csv", "records no hardware_tag"),
        ("results/list.csv", "records no hardware_tag"),
        ("results/broken.csv", "not readable JSON"),
        ("config/baselines/empty.json", "records no hardware_tag"),
        ("config/baselines/poc.example.json", "is a template"),
    ],
)
def test_an_artifact_without_hardware_provenance_backs_nothing(
    repo: Path, artifact: str, fragment: str
) -> None:
    problem = citation_problem(repo, artifact)
    assert problem is not None and fragment in problem, problem


def test_unbacked_numbers_report_why(repo: Path) -> None:
    markdown = "\n\n".join(
        [
            "Backed: inference takes 45 ms (`results/good.csv`).",
            "Untagged: inference takes 46 ms (`results/untagged.csv`).",
            "Uncited: inference takes 47 ms.",
            "Mixed: inference takes 48 ms (`results/orphan.csv`, `results/good.csv`).",
            "No numbers here at all, only `results/missing.csv`.",
        ]
    )
    unbacked = unbacked_performance_numbers(markdown, repo)
    assert [(u.number.token, u.reasons) for u in unbacked] == [
        ("46 ms", ("results/untagged.run.json records no hardware_tag (got 'unknown')",)),
        ("47 ms", (NO_CITATION_REASON,)),
    ]


def test_a_citation_backs_only_its_own_block(repo: Path) -> None:
    markdown = "Inference takes 45 ms.\n\nSource: `results/good.csv`."
    assert [u.number.token for u in unbacked_performance_numbers(markdown, repo)] == ["45 ms"]


def test_a_block_without_performance_numbers_is_never_reported(repo: Path) -> None:
    block = Block("text", ((1, "MSE 2.3e-3 at 19×19, `results/missing.csv`."),))
    assert performance_numbers(block) == []
    assert unbacked_performance_numbers(block.text, repo) == []


# --------------------------------------------------------------------------------------
# the reviewed leaks, on synthetic runs: a hardware tag alone backs nothing
# --------------------------------------------------------------------------------------


def _tagged_run(root: Path, stem: str, csv: str = "a,b\n", **metrics: float) -> str:
    _write(root, f"results/{stem}.csv", csv)
    _write(root, f"results/{stem}.run.json", _sidecar("x86_64-4cpu", **metrics))
    return f"results/{stem}.csv"


def test_a_tagged_run_without_timing_backs_no_figure(tmp_path: Path) -> None:
    """Leak 1's shape: real provenance, DOF and error data, not one timing measurement."""
    cited = _tagged_run(tmp_path, "amr", l2_error_ratio=0.95, matched_dof=287.0)
    (unbacked,) = unbacked_performance_numbers(
        f"Inference takes 12 ms per move (`{cited}`).", tmp_path
    )
    assert unbacked.reasons == (f"the run(s) behind {cited} record no named duration measurement",)


def test_timing_that_is_not_the_figure_backs_nothing(tmp_path: Path) -> None:
    """Leak 2's shape: the removed table, one cell citing a run that records an AMR wall time."""
    cited = _tagged_run(tmp_path, "arena", matched_wall_time_seconds=0.3459)
    rows = "| Standard | 19×19 | 45 | 180 |\n| Galerkin | 19×19 | 28 | 290 |\n"
    table = f"{TABLE_HEADER}\n{rows}| Galerkin+FNet (`{cited}`) | 19×19 | 12 | 670 |\n"
    unbacked = unbacked_performance_numbers(table, tmp_path)
    assert [u.number.token for u in unbacked] == ["45", "180", "28", "290", "12", "670"]


def test_a_csv_timing_row_is_not_a_named_measurement(tmp_path: Path) -> None:
    """Rows are raw data: tracing to them let up to 46% of figures match by coincidence."""
    cited = _tagged_run(tmp_path, "rows", csv="level,wall_time_seconds\n1,0.012\n", n_seeds=3.0)
    claim = f"Inference takes 12 ms per move (`{cited}`)."
    assert [u.number.token for u in unbacked_performance_numbers(claim, tmp_path)] == ["12 ms"]


def test_a_figure_that_is_a_named_measurement_is_backed(tmp_path: Path) -> None:
    """The control for the three tests above: the same rule accepts real evidence."""
    cited = _tagged_run(tmp_path, "bench", inference_latency_ms=12.0, sims_per_second=670.0)
    claim = f"Inference takes 12 ms per move and MCTS runs 670 sims/sec (`{cited}`)."
    assert unbacked_performance_numbers(claim, tmp_path) == []


@pytest.mark.parametrize(
    ("artifact", "document"),
    [
        ("results/x.csv", _sidecar("unknown", inference_latency_ms=12.0)),
        ("results/x.csv", _sidecar("", inference_latency_ms=12.0)),
        ("results/x.csv", _sidecar(" Unknown ", inference_latency_ms=12.0)),
        ("config/baselines/x.example.json", None),
    ],
    ids=["unknown", "empty", "padded", "template"],
)
def test_a_placeholder_tag_backs_nothing_even_when_the_figure_matches(
    tmp_path: Path, artifact: str, document: str | None
) -> None:
    """Kill #6, re-planted on a synthetic run so no re-recorded committed file can disarm it."""
    if document is None:
        entries = [{"metric_name": "inference_latency_ms", "value": 12.0}]
        _write(tmp_path, artifact, json.dumps({"hardware_tag": "rig", "entries": entries}))
    else:
        _write(tmp_path, artifact, "a,b\n")
        _write(tmp_path, provenance_path(artifact), document)
    claim = f"Inference takes 12 ms per move (`{artifact}`)."
    (unbacked,) = unbacked_performance_numbers(claim, tmp_path)
    assert any("hardware_tag" in reason or "template" in reason for reason in unbacked.reasons)


# --------------------------------------------------------------------------------------
# the placeholder tags are the writers' defaults, not a guess
# --------------------------------------------------------------------------------------


def _module_constant(path: Path, name: str) -> object:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.AnnAssign | ast.Assign):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def _field_default(path: Path, field: str) -> object:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == field
            and isinstance(node.value, ast.Call)
        ):
            for keyword in node.value.keywords:
                if keyword.arg == "default":
                    return ast.literal_eval(keyword.value)
    raise AssertionError(f"{field} has no Field(default=...) in {path}")


def test_placeholder_tags_are_the_writers_defaults() -> None:
    """A sidecar written with no tag must not count as tagged, whatever the writer's default.

    Read by AST, not import: ``src.research`` imports torch, and this suite stays hermetic.
    """
    manifest_default = _module_constant(REPO_ROOT / "src/research/run_manifest.py", "UNKNOWN")
    baseline_default = _field_default(REPO_ROOT / "src/poc/baselines/schema.py", "hardware_tag")
    assert {manifest_default, baseline_default} <= PLACEHOLDER_HARDWARE_TAGS
