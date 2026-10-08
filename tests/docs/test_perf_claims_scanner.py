"""Unit tests for ``tests/support/perf_claims.py``, the performance-number scanner.

The scanner is the load-bearing half of ``tests/docs/test_performance_claims.py``: a unit it
cannot see is a claim that guard can never reject. So every unit alternative, speed word and
measured-quantity word has a literal positive below, and the literals are written out rather
than generated from the scanner's own vocabulary tuples -- a case generated from
``SUBSECOND_UNITS`` would vanish together with the unit it exists to test, and the test would
pass on exactly the mutation it should kill.

The negatives are the instructional text the guard must never reject: "takes ~2 minutes",
timeouts, config values, asymptotic ``O(N)`` and accuracy ratios.

Mutation kills are recorded in ``tests/docs/test_performance_claims.py``'s docstring.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Final

import pytest

from tests.support.perf_claims import (
    NO_CITATION_REASON,
    PLACEHOLDER_HARDWARE_TAGS,
    TABLE_COLUMN_KIND,
    Block,
    citation_problem,
    cited_artifacts,
    inline_numbers,
    is_measurement_header,
    performance_numbers,
    provenance_path,
    split_blocks,
    unbacked_performance_numbers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: ``id -> (text, token the scanner must report)``. One case per unit alternative, speed word,
#: measured-quantity word and ``per <noun>``.
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
    "unit-s": ("Inference takes 0.4 s on one core.", "0.4 s"),
    "unit-sec": ("Inference takes 2 sec on one core.", "2 sec"),
    "unit-secs": ("Inference takes 2 secs on one core.", "2 secs"),
    "unit-second": ("Inference takes 1 second on one core.", "1 second"),
    "unit-seconds": ("Inference takes 3 seconds on one core.", "3 seconds"),
    "rate-slash-s": ("Training reaches 1,200 it/s.", "1,200 it/s"),
    "rate-slash-sec": ("MCTS runs 670 sims/sec.", "670 sims/sec"),
    "rate-slash-second": ("The loader streams 40 samples/second.", "40 samples/second"),
    "rate-per-second": ("It evaluates 900 positions per second.", "900 positions per second"),
    "rate-FPS": ("Rendering holds 60 FPS.", "60 FPS"),
    "rate-fps": ("Rendering holds 60fps.", "60fps"),
    "ratio-times-sign": ("FNet is 3.2× faster than softmax.", "3.2×"),
    "ratio-x": ("FNet is 3.2x faster than softmax.", "3.2x"),
    "ratio-X": ("FNet is 3.2X faster than softmax.", "3.2X"),
    "ratio-percent": ("The cache made search 30% faster.", "30%"),
    "word-faster": ("Search got 2× faster.", "2×"),
    "word-slower": ("Search got 2× slower.", "2×"),
    "word-speedup": ("We measured a 10× speedup.", "10×"),
    "word-speed-up": ("We measured a 10× speed-up.", "10×"),
    "word-slowdown": ("We measured a 2× slowdown.", "2×"),
    "quantity-latency": ("The p95 latency is 2 s.", "2 s"),
    "quantity-inference": ("Inference is 2 s here.", "2 s"),
    "quantity-throughput": ("At that throughput a batch is 2 s.", "2 s"),
    "quantity-wall-clock-hyphen": ("The wall-clock cost was 2 s.", "2 s"),
    "quantity-wall-clock-space": ("The wall clock showed 2 s.", "2 s"),
    "quantity-wall-time": ("The wall time was 2 s.", "2 s"),
    "quantity-step-time": ("The step time was 2 s.", "2 s"),
    "per-move": ("Search spends 2 s per move.", "2 s"),
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
}

#: Instructional or non-performance text the scanner must not report.
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
}

REMOVED_ROW: Final[str] = "| Galerkin+FNet | 19×19 | 12 | 670 |"


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
        "Speedup",
        "Wall time (s)",
        "Time (s)",
        "Rate (per second)",
    ],
)
def test_measurement_headers_are_recognised(header: str) -> None:
    assert is_measurement_header(header)


@pytest.mark.parametrize(
    "header",
    ["Board Size", "Model", "MS-SSIM", "Timeout (s)", "Coverage (%)", "Status", "Columns"],
)
def test_non_measurement_headers_are_not(header: str) -> None:
    assert not is_measurement_header(header)


def test_numbers_under_a_measurement_header_are_reported() -> None:
    """The removed table's shape: bare cells, units only in the header."""
    markdown = "\n".join(
        [
            "| Model | Board Size | Inference (ms) | MCTS Sims/sec |",
            "|-------|------------|----------------|---------------|",
            REMOVED_ROW,
        ]
    )
    (block,) = split_blocks(markdown)
    numbers = performance_numbers(block)
    assert [(n.kind, n.token, n.line_no) for n in numbers] == [
        (TABLE_COLUMN_KIND, "12", 3),
        (TABLE_COLUMN_KIND, "670", 3),
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


def test_a_header_only_table_reports_nothing() -> None:
    (block,) = split_blocks("| Latency (ms) |\n|---|")
    assert performance_numbers(block) == []


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


def _tagged(tag: object) -> str:
    return json.dumps({"hardware_tag": tag})


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature repository with one artifact of each provenance state."""
    _write(tmp_path, "results/good.csv", "a,b\n")
    _write(tmp_path, "results/good.run.json", _tagged("x86_64-4cpu"))
    _write(tmp_path, "results/untagged.csv", "a,b\n")
    _write(tmp_path, "results/untagged.run.json", _tagged("unknown"))
    _write(tmp_path, "results/shouting.csv", "a,b\n")
    _write(tmp_path, "results/shouting.run.json", _tagged(" UNKNOWN "))
    _write(tmp_path, "results/no_field.csv", "a,b\n")
    _write(tmp_path, "results/no_field.run.json", json.dumps({"seed": 1}))
    _write(tmp_path, "results/list.csv", "a,b\n")
    _write(tmp_path, "results/list.run.json", "[1, 2]")
    _write(tmp_path, "results/broken.csv", "a,b\n")
    _write(tmp_path, "results/broken.run.json", "{not json")
    _write(tmp_path, "results/orphan.csv", "a,b\n")
    _write(tmp_path, "config/baselines/tagged.json", _tagged("RTX 5060 Ti 16GiB"))
    _write(tmp_path, "config/baselines/empty.json", _tagged(""))
    _write(tmp_path, "config/baselines/poc.example.json", _tagged("example"))
    return tmp_path


@pytest.mark.parametrize(
    "artifact", ["results/good.csv", "results/good.run.json", "config/baselines/tagged.json"]
)
def test_a_hardware_tagged_artifact_backs_a_claim(repo: Path, artifact: str) -> None:
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
