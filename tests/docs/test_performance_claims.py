"""Performance numbers on the project's front doors must be named measurements of tagged runs.

Defect class, in one sentence: a unit-bearing performance number -- a latency, a throughput or a
speedup -- published in ``README.md`` or on the docs site that is not one of the recorded
measurements of a committed run whose provenance names the hardware it ran on.

It shipped. ``README.md``'s "Benchmarks" table showed inference milliseconds and MCTS
simulations per second for three models, captioned "Benchmarks on NVIDIA RTX 3090, batch size
1". No artifact, run output or sidecar backed any of the six figures, the card matches none of
the rigs this repository documents, and no check read the table. A person found it
(``docs/business/COMMERCIALIZATION_PEER_REVIEW.md`` §5.2, Gate 0 item 0.3), as people found the
seven enforcement gaps before it. Then a reviewer found that this guard's first version backed
too much (2026-10-08): any cited run whose sidecar named real hardware backed *any* figure, so
"Inference takes 12 ms per move" citing ``results/lshape_adaptive_vs_uniform.csv`` (DOF and
error, no timing at all) passed, and so did the removed table with one cell citing
``results/mcts_classical_amr_arena.csv``. Both leaks are pinned below, verbatim.

The rule (``tests/support/perf_claims.py``, where it is argued and measured): a figure must be
one of the cited run's **named timing measurements** -- a sidecar ``metrics`` or arm
``counters`` value, or a ``config/baselines/`` entry -- at the precision it is written, to at
least two significant digits, and that run's provenance must record a real ``hardware_tag``.
A hardware tag alone backs nothing; neither does timing data that is not the figure.

Scope -- the front doors (``tests/support/docs_site.py``): ``README.md`` and every local page in
``mkdocs.yml``'s nav, read from the nav rather than listed, so moving a table onto a site page
is not an evasion. ``README.md``'s Performance section states the same rule and says that neither
documented measuring command writes the sidecar it needs, so the prose promises nothing the gate
does not enforce.

``ALLOWLIST`` holds the exceptions: each is a disclosed debt with a stated reason, bound to the
figure it was written for -- a number is exempt only when its whole token lies inside an
occurrence of the entry's fragment on its line, so a second figure on that line is reported like
any other. Each expires: an entry under whose fragment no *unbacked* number lies any more (the
claim was backed, reworded or deleted, even if the fragment still matches the line) fails
``test_allowlist_entries_are_still_needed``.

Mutation kills -- 32/32 planted defects, each failing a NAMED test, none of them
``gpu_required`` or ``fem_required``. Kills 6, 7, 8 and 10 were re-planted on 2026-10-08:
kill 6 had died silently when ``results/lshape_adaptive_vs_uniform.run.json`` was re-recorded
with a real tag, and 7, 8 and 10 leaned on committed state the same way (a CSV with no sidecar,
the live nav, the live allowlist). Every killer below now runs on a constant or on a synthetic
repository in ``tmp_path``; the committed-tree tests are kept as pins, never as the only killer.
``S::`` is ``tests/docs/test_perf_claims_scanner.py``.

1. README's RTX 3090 table restored verbatim (the literal historical defect) ->
   ``test_front_door_performance_numbers_are_backed``.
2. ``"ms"`` deleted from ``SUBSECOND_UNITS`` -> ``S::test_synthetic_positive_is_detected[unit-ms]``.
3. ``"sec"`` deleted from ``THROUGHPUT_DENOMINATORS`` -> ``S::...[rate-slash-sec]`` and
   ``test_the_removed_rtx_3090_table_is_caught`` (its "Sims/sec" column goes dark).
4. Header propagation removed -- every cell of the real table is a bare number ->
   ``test_the_removed_rtx_3090_table_is_caught``.
5. The hardware-tag check dropped ->
   ``S::test_an_artifact_without_hardware_provenance_backs_nothing``.
6. ``PLACEHOLDER_HARDWARE_TAGS`` emptied, so ``unknown`` / empty tags count as hardware ->
   ``S::test_a_placeholder_tag_backs_nothing_even_when_the_figure_matches[unknown]`` (and
   ``[empty]``, ``[padded]``); the run records the very figure, so only provenance can reject it.
7. The template check dropped -> ``S::...even_when_the_figure_matches[template]``.
8. The missing-sidecar check dropped, so a CSV with no sidecar is no longer rejected for it ->
   ``S::test_an_artifact_without_hardware_provenance_backs_nothing[results/orphan.csv]``.
9. Nav pages dropped from the front doors -> ``test_a_table_moved_onto_a_nav_page_is_caught``.
10. Inline code stripped before scanning, making backticks an evasion ->
    ``S::test_inline_code_is_not_skipped``.
11. Stale-entry detection disabled -> ``test_a_stale_allowlist_entry_is_reported``.
12. A 39-character allowlist reason -> ``test_allowlist_reasons_are_substantive``.
13. The pre-fix rule restored -- provenance alone backs a figure (the reviewer's finding) ->
    ``S::test_a_tagged_run_without_timing_backs_no_figure``,
    ``S::test_timing_that_is_not_the_figure_backs_nothing`` and
    ``test_the_reviewed_leaks_stay_closed[timing-free-run]``.
14. The presence rule -- any measurement of the figure's dimension backs it ->
    ``S::test_timing_that_is_not_the_figure_backs_nothing`` (its latency column passes).
15. The dimension filter dropped, so a duration can back a rate ->
    ``S::test_a_figure_that_is_not_a_named_measurement_does_not_trace[dim]``.
16. CSV rows read as measurements -> ``S::test_a_csv_timing_row_is_not_a_named_measurement``
    and ``test_named_measurements_rarely_coincide``.
17. ``config`` read as measurements ->
    ``S::test_recorded_values_are_named_timing_measurements_and_nothing_else``.
18. ``MIN_TRACEABLE_SIGNIFICANT_DIGITS`` lowered to 1 -> ``S::...does_not_trace[one-digit]``.
19. The rounding window widened tenfold -> ``S::...does_not_trace[outside]`` and ``[digits]``.
20. Settings no longer exempt -> ``S::test_instructional_text_is_not_a_performance_number``
    ``[setting-wall-clock-timeout]`` and ``[setting-tolerance]`` (the reviewer's two).
21. ``"budget"`` made a setting -> ``S::test_synthetic_positive_is_detected[budget-is-a-claim]``.
22. A setting phrase allowed to run past a comma ->
    ``S::test_a_setting_phrase_does_not_hide_a_neighbouring_claim``.
23. ``"fold"`` deleted from ``RATIO_SYMBOLS`` -> ``S::...[ratio-hyphen-fold]``.
24. Timing-quantity words dropped from the speed context -> ``S::...[timing-latency]``.
25. ``"speeds up"`` deleted from ``SPEED_WORDS`` -> ``S::...[word-speeds-up]``.
26. The ``/<noun>`` per-unit form dropped -> ``S::...[per-slash-move]``.
27. Minute and hour denominators dropped -> ``S::...[rate-per-minute]`` and
    ``[rate-slash-hour]``.
28. ``Hz``/``kHz`` deleted from ``RATE_UNITS`` -> ``S::...[rate-Hz]``.
29. The rate-unit look-behind removed, so "GHz" reads as "Hz" ->
    ``S::test_instructional_text_is_not_a_performance_number[clock-speed]``.
30. The duration-noun header rule removed ->
    ``S::test_measurement_headers_are_recognised[Runtime (s)]`` (and the reviewer's other four).
31. Settings no longer exempt headers -> ``S::test_non_measurement_headers_are_not[Time limit
    (s)]``.
32. ``FIELD_UNIT_EXCLUSIONS`` emptied, so ``min`` (minimum) reads as minutes ->
    ``S::test_field_unit_ignores_names_that_state_no_timing_unit[l2_ratio_seed_min]``.

Allowlist binding -- 12/12 further planted defects (COPILOT-3, 2026-10-10), numbered on from the
32 above, each confirmed applied by hash and killed under CI's own ``-m`` filter. Defect class,
in one sentence: an exemption whose scope is wider than the claim it was written for exempts
claims nobody reviewed. Copilot's review of PR #160 found it: ``_allowlisted`` tested
``fragment in line``, so every unbacked figure on an allowlisted line was exempt, and the stale
check passed while *any* unbacked figure sat on a matching line. Reproduced before the fix on a
synthetic copy of the live page: "and 40 ms per move" appended to either live line, or written
before its fragment, was reported by nothing, and an empty fragment exempted every figure on its
page without going stale. Below, ``E::`` is
``test_an_exemption_covers_only_the_figure_inside_its_fragment``, ``St::`` is
``test_an_entry_whose_fragment_holds_no_unbacked_figure_is_stale`` and ``L::`` is
``S::test_a_figure_is_located_at_its_own_occurrence``.

33. ``_allowlisted`` restored to whole-line matching (the reviewed defect) -> all four
    ``test_a_figure_planted_on_a_live_allowlisted_line_is_reported`` cases (both live entries,
    appended and prepended), ``E::[appended]``, ``[prepended]``, ``[same-figure-twice]`` and
    ``[fragment-cuts-a-figure]``, and every ``St::`` case.
34. The per-line stale check restored -> every ``St::`` case.
35. Containment by token text -- the fallback the binding refuses -> ``E::[same-figure-twice]``,
    ``St::[figure-no-longer-a-claim]`` and ``test_failures_name_the_column_of_the_reported_figure``.
36. A per-entry stale check judged by token text -> ``St::[figure-no-longer-a-claim]``.
37. An unlocated number exempted by its token text (fail open) ->
    ``test_a_number_without_a_column_is_never_exempt``.
38. Containment by the token's first character only -> ``E::[fragment-cuts-a-figure]`` and
    ``St::[figure-wrapped-across-lines]``.
39. Prose columns measured from the block, not the line -> ``L::[wrapped]``,
    ``L::[indented-continuation]`` and ``S::test_every_number_records_where_its_token_starts``.
40. Prose columns by the token's first occurrence -> ``L::[equal-figures]`` and
    ``E::[same-figure-twice]``.
41. An escaped pipe counted once in a cell's column -> ``L::[escaped-pipe]`` and the Hypothesis
    property ``S::test_a_figure_on_any_table_row_is_located``.
42. Table-cell columns by the token's first occurrence -> ``L::[digits-in-another-cell]``.
43. A table row's inline numbers left unlocated -> ``L::[table-inline]``.
44. An escaped pipe read as a cell separator -> ``L::[escaped-pipe]``,
    ``S::test_table_cells_honour_escaped_pipes`` and
    ``test_front_door_performance_numbers_are_backed``: live ``c4_mermaid.md`` tables escape the
    pipes of their LaTeX norms, and the strict cell alignment turns the misreading into a crash of
    the real scan rather than a silently misplaced column.

Kill 11 was re-planted against the rewritten stale check;
``test_a_stale_allowlist_entry_is_reported`` still kills it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from tests.support.docs_site import (
    DEFAULT_DOCS_DIR,
    MKDOCS_CONFIG,
    README,
    front_door_documents,
    mkdocs_nav_pages,
)
from tests.support.perf_claims import (
    BASELINES_ROOT,
    DURATION,
    RESULTS_ROOT,
    SIDECAR_SUFFIX,
    TABLE_COLUMN_KIND,
    PerformanceNumber,
    RecordedValue,
    UnbackedNumber,
    citation_problem,
    cited_artifacts,
    recorded_values,
    trace_problem,
    unbacked_performance_numbers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The docs-site landing page; like README.md it must always be scanned.
DOCS_INDEX: Final[str] = "docs/index.md"

#: Front doors expected at minimum -- README.md plus the site nav, which lists 16 pages today.
#: A floor rather than an equality, so adding a page does not fail this guard.
MIN_FRONT_DOORS: Final[int] = 10

#: An allowlist reason must say something; "TODO" or "see above" is not a disclosure.
MIN_ALLOWLIST_REASON_CHARS: Final[int] = 40

#: The largest share of 2-significant-digit durations from 10 ms to 99 s that may match a named
#: measurement of one committed run by coincidence. Measured 2026-10-08: 0% for the arena and 3%
#: for each Gate 1 run; tracing to their CSV rows instead measured 12%, 42% and 46%.
MAX_COINCIDENCE_RATE: Final[float] = 0.10

#: ``(document, literal line fragment) -> reason``. Exempts an unbacked performance number on a
#: line of ``document`` only when its whole figure lies inside an occurrence of the fragment --
#: never another figure on that line. A disclosed debt, not an endorsement.
ALLOWLIST: Final[dict[tuple[str, str], str]] = {
    ("docs/architecture/c4_mermaid.md", "N=361, d=32 → **10x speedup**"): (
        "An operation-count ratio (N/d = 361/32 for a 19x19 board) worded as a measured "
        "speedup; no committed artifact measures it. Reword it as an op-count ratio or back it."
    ),
    ("docs/architecture/c4_mermaid.md", "**Rationale**: 5× speedup for leaf evaluation"): (
        "Design rationale from the original FNet decision, never measured into a committed "
        "artifact (tests/benchmarks asserts only a relative-speedup floor). Back it with a "
        "hardware-tagged benchmark artifact or delete the figure."
    ),
}

#: The table removed from README.md's Performance section on 2026-09-25, verbatim.
REMOVED_RTX_3090_TABLE: Final[str] = """\
| Model | Board Size | Inference (ms) | MCTS Sims/sec |
|-------|------------|----------------|---------------|
| Standard | 19×19 | 45 | 180 |
| Galerkin | 19×19 | 28 | 290 |
| Galerkin+FNet | 19×19 | 12 | 670 |

*Benchmarks on NVIDIA RTX 3090, batch size 1*
"""

#: Its six figures, in reading order. Every one is a bare cell; only the headers carry units.
REMOVED_TABLE_FIGURES: Final[tuple[str, ...]] = ("45", "180", "28", "290", "12", "670")

#: The reviewer's two leaks (2026-10-08), verbatim, as ``id -> (text, figures it must report)``:
#: citations of committed runs with real hardware tags that this guard's first version accepted.
REVIEWED_LEAKS: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "timing-free-run": (
        "Inference takes 12 ms per move (`results/lshape_adaptive_vs_uniform.csv`).",
        ("12 ms",),
    ),
    "unrelated-timing": (
        REMOVED_RTX_3090_TABLE.replace(
            "| Galerkin+FNet |", "| Galerkin+FNet (`results/mcts_classical_amr_arena.csv`) |"
        ),
        REMOVED_TABLE_FIGURES,
    ),
}

#: Every 2-significant-digit duration from 10 ms to 99 s, written the way a page would.
COINCIDENCE_PROBES: Final[tuple[str, ...]] = tuple(
    token
    for mantissa in range(10, 100)
    for token in (f"{mantissa} ms", f"0.{mantissa} s", f"{mantissa / 10:.1f} s", f"{mantissa} s")
)

#: Planted on every live allowlisted line as ``id -> (text written just before the fragment, text
#: appended to the line, the planted figure)``: a claim nobody reviewed, beside one somebody did.
PLANTED_CLAIMS: Final[dict[str, tuple[str, str, str]]] = {
    "appended": ("", " and search is 3.5× faster", "3.5×"),
    "prepended": ("Inference takes 40 ms per move; ", "", "40 ms"),
}

#: Runs of characters a pytest id should not carry; an entry's id is its fragment's ASCII words.
_UNSAFE_ID_CHARACTERS: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9]+")
ALLOWLIST_IDS: Final[tuple[str, ...]] = tuple(
    _UNSAFE_ID_CHARACTERS.sub("-", fragment).strip("-") for _, fragment in ALLOWLIST
)

#: One README line, the fragment of the entry written for one of its figures, and the figures
#: that must still be reported: ``id -> (line, fragment, reported tokens)``.
NEIGHBOURING_FIGURES: Final[dict[str, tuple[str, str, tuple[str, ...]]]] = {
    "appended": (
        "Search is 5× faster, and inference takes 40 ms per move.",
        "Search is 5× faster",
        ("40 ms",),
    ),
    "prepended": (
        "Inference takes 40 ms per move, and search is 5× faster.",
        "search is 5× faster",
        ("40 ms",),
    ),
    # The second 5× is the entry's token, outside its fragment: "the token is in it" exempts both.
    "same-figure-twice": (
        "Search is 5× faster, and leaf evaluation is 5× faster too.",
        "Search is 5× faster",
        ("5×",),
    ),
    # The fragment ends inside "40 ms": a figure must lie wholly inside, not merely start there.
    "fragment-cuts-a-figure": (
        "Search is 5× faster, and inference takes 40 ms per move.",
        "Search is 5× faster, and inference takes 40",
        ("40 ms",),
    ),
}

#: A tagged run recording 10x as a named measurement, so a figure of 10x citing it is backed.
BENCH_RUN: Final[dict[str, str]] = {
    f"{RESULTS_ROOT}bench.csv": "a,b\n",
    f"{RESULTS_ROOT}bench{SIDECAR_SUFFIX}": json.dumps(
        {"hardware_tag": "x86_64-4cpu", "metrics": {"fnet_speedup": 10.0}}
    ),
}

#: Fragments that still match their README text but hold no unbacked figure, so their entry
#: exempts nothing and is stale: ``id -> (text, fragment, tokens reported)``.
EXEMPTS_NOTHING: Final[dict[str, tuple[str, str, tuple[str, ...]]]] = {
    "figure-outside-the-fragment": (
        "- **Rationale**: 5× speedup for leaf evaluation",
        "**Rationale**:",
        ("5×",),
    ),
    "empty-fragment": ("- **Rationale**: 5× speedup for leaf evaluation", "", ("5×",)),
    # The fragment's 5× is no longer read as a claim (no speed word in its sentence); the other is.
    "figure-no-longer-a-claim": (
        "The 5× case is an old design note; search is now 5× faster.",
        "The 5× case",
        ("5×",),
    ),
    # The fragment's 10x is backed by the run it cites; 3.5× on the same line is not.
    "figure-now-backed": (
        f"FNet gives a 10x speedup and search is 3.5× faster (`{RESULTS_ROOT}bench.csv`).",
        "10x speedup",
        ("3.5×",),
    ),
    # "45\nms" starts inside the fragment and ends on the next line, past any one-line fragment.
    "figure-wrapped-across-lines": (
        "Inference takes 45\nms per move.",
        "Inference takes 45",
        ("45\nms",),
    ),
}


def _unbacked(repo_root: Path, document: str) -> list[UnbackedNumber]:
    text = (repo_root / document).read_text(encoding="utf-8")
    return unbacked_performance_numbers(text, repo_root)


def _occurrences(line: str, fragment: str) -> list[tuple[int, int]]:
    """``[start, end)`` of every occurrence of ``fragment`` in ``line``, overlapping ones too.

    An empty fragment occurs nowhere: it names no claim.
    """
    spans: list[tuple[int, int]] = []
    start = line.find(fragment) if fragment else -1
    while start >= 0:
        spans.append((start, start + len(fragment)))
        start = line.find(fragment, start + 1)
    return spans


def _exempting_entries(
    document: str, number: PerformanceNumber, allowlist: Mapping[tuple[str, str], str]
) -> list[tuple[str, str]]:
    """The entries with an occurrence of their fragment, on its line, that holds its whole token.

    An entry exempts the figure it was written for, never a neighbour on its line. There is no
    fallback to "the token occurs in the fragment", which exempts a second, identical figure
    anywhere on the line; a number without a column -- the scanner returns none -- cannot be
    located, so nothing exempts it. A token wrapped onto the next line ends past every fragment.
    """
    if number.column is None:
        return []
    start, end = number.column, number.column + len(number.token)
    return [
        (doc, fragment)
        for doc, fragment in allowlist
        if doc == document
        and any(lo <= start and end <= hi for lo, hi in _occurrences(number.line, fragment))
    ]


def _allowlisted(
    document: str, number: PerformanceNumber, allowlist: Mapping[tuple[str, str], str]
) -> bool:
    return bool(_exempting_entries(document, number, allowlist))


def unexempted_numbers(
    repo_root: Path, allowlist: Mapping[tuple[str, str], str]
) -> list[tuple[str, UnbackedNumber]]:
    """``(document, number)`` per unbacked front-door number that no ``allowlist`` entry exempts."""
    return [
        (document, u)
        for document in front_door_documents(repo_root)
        for u in _unbacked(repo_root, document)
        if not _allowlisted(document, u.number, allowlist)
    ]


def front_door_failures(repo_root: Path, allowlist: Mapping[tuple[str, str], str]) -> list[str]:
    """One line per unbacked, non-allowlisted performance number on ``repo_root``'s front doors.

    Located ``document:line:column`` (1-based column, as editors read it): an entry may exempt
    one figure on a line and not another, so the line alone does not say which was reported.
    """
    failures = []
    for document, u in unexempted_numbers(repo_root, allowlist):
        column = "" if u.number.column is None else f":{u.number.column + 1}"
        failures.append(
            f"{document}:{u.number.line_no}{column}: {u.number.kind} {u.number.token!r} -- "
            f"{'; '.join(u.reasons)}\n      {u.number.line.strip()}"
        )
    return failures


def stale_allowlist_entries(repo_root: Path, allowlist: Mapping[tuple[str, str], str]) -> list[str]:
    """Entries that exempt no unbacked front-door number: none lies inside their fragment.

    Judged per entry, not per line: a fragment that still matches a line whose own figure was
    backed, reworded or deleted is stale even while another unbacked figure sits on that line --
    a figure the entry never exempted.
    """
    scanned = {document for document, _ in allowlist} & set(front_door_documents(repo_root))
    live = {
        entry
        for document in scanned
        for u in _unbacked(repo_root, document)
        for entry in _exempting_entries(document, u.number, allowlist)
    }
    return [f"{doc}: {fragment!r}" for doc, fragment in allowlist if (doc, fragment) not in live]


def _committed_runs() -> dict[str, list[RecordedValue]]:
    """Every committed run whose provenance can back a figure, with its named measurements."""
    records = [
        *(REPO_ROOT / RESULTS_ROOT).glob(f"*{SIDECAR_SUFFIX}"),
        *(REPO_ROOT / BASELINES_ROOT).glob("*.json"),
    ]
    citations = sorted(path.relative_to(REPO_ROOT).as_posix() for path in records)
    return {
        citation: recorded_values(REPO_ROOT, citation)
        for citation in citations
        if citation_problem(REPO_ROOT, citation) is None
    }


# --------------------------------------------------------------------------------------
# vacuity: the scan reads real documents, and the scanner sees the defects it exists for
# --------------------------------------------------------------------------------------


def test_front_door_scan_is_not_vacuous() -> None:
    """Without this, an empty nav or a renamed README would make every assertion below pass."""
    documents = front_door_documents(REPO_ROOT)
    assert README in documents and DOCS_INDEX in documents, documents
    assert len(documents) >= MIN_FRONT_DOORS, documents
    empty = [doc for doc in documents if not (REPO_ROOT / doc).read_text(encoding="utf-8").strip()]
    assert not empty, f"front-door documents that are missing content: {empty}"


def test_the_removed_rtx_3090_table_is_caught() -> None:
    """The literal historical defect must stay visible to the scanner, figure by figure."""
    unbacked = unbacked_performance_numbers(REMOVED_RTX_3090_TABLE, REPO_ROOT)
    assert tuple(u.number.token for u in unbacked) == REMOVED_TABLE_FIGURES
    assert {u.number.kind for u in unbacked} == {TABLE_COLUMN_KIND}


@pytest.mark.parametrize(("text", "figures"), REVIEWED_LEAKS.values(), ids=REVIEWED_LEAKS)
def test_the_reviewed_leaks_stay_closed(text: str, figures: tuple[str, ...]) -> None:
    """Against the committed runs they cited -- which must still pass the provenance check.

    If a cited sidecar lost its tag, the leak would be rejected for provenance and this pin
    would stop testing traceability without a sound; the first assertion makes that loud.
    """
    for cited in cited_artifacts(text):
        assert citation_problem(REPO_ROOT, cited) is None, f"{cited} no longer pins traceability"
    unbacked = unbacked_performance_numbers(text, REPO_ROOT)
    assert tuple(u.number.token for u in unbacked) == figures


def test_the_rule_is_satisfiable_with_committed_evidence() -> None:
    """A rule no committed artifact can meet would be a ban wearing a guard's clothes."""
    durations = [
        (citation, value)
        for citation, values in _committed_runs().items()
        for value in values
        if value.unit.dimension == DURATION and "e" not in f"{value.value * value.unit.scale:.4g}"
    ]
    assert durations, "no committed tagged run records a named duration measurement"
    citation, value = durations[0]
    claim = f"The wall-clock time was {value.value * value.unit.scale:.4g} s (`{citation}`)."
    assert unbacked_performance_numbers(claim, REPO_ROOT) == [], claim


def test_named_measurements_rarely_coincide() -> None:
    """Traceability is a check only while an arbitrary figure rarely matches a value by chance.

    Every 2-significant-digit duration from 10 ms to 99 s is traced against each committed run
    that can back a figure. Tracing to CSV rows instead of named measurements fails this.
    """
    runs = _committed_runs()
    assert runs, "no committed run passes the provenance check"
    probes = [PerformanceNumber(0, token, "latency", token) for token in COINCIDENCE_PROBES]
    rates = {
        citation: sum(trace_problem(probe, evidence, [citation]) is None for probe in probes)
        / len(probes)
        for citation, evidence in runs.items()
    }
    assert max(rates.values()) <= MAX_COINCIDENCE_RATE, rates


# --------------------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------------------


def test_front_door_performance_numbers_are_backed() -> None:
    """Every front-door latency, throughput or speedup is a named measurement of a tagged run."""
    failures = front_door_failures(REPO_ROOT, ALLOWLIST)
    assert not failures, (
        "front-door performance numbers that are not a named timing measurement of a "
        "hardware-tagged run cited in their paragraph, list item or table:\n  "
        + "\n  ".join(failures)
        + "\n\nCite a committed results/ or config/baselines/ artifact whose run records this "
        "figure as a named metric (src/research/run_manifest.py writes the sidecar) and states "
        "its hardware_tag, or delete the figure. ALLOWLIST is for disclosed debt, not new claims."
    )


def test_allowlist_entries_are_still_needed() -> None:
    """An exemption that matches no unbacked number silently shrinks the guard's scope."""
    stale = stale_allowlist_entries(REPO_ROOT, ALLOWLIST)
    assert not stale, (
        "ALLOWLIST entries that no longer match an unbacked performance number:\n  "
        + "\n  ".join(stale)
        + "\n\nThe figure was backed, reworded or deleted (or its page left the nav): "
        "remove the entry so the line is guarded again."
    )


def test_allowlist_reasons_are_substantive() -> None:
    thin = [key for key, reason in ALLOWLIST.items() if len(reason) < MIN_ALLOWLIST_REASON_CHARS]
    assert not thin, f"ALLOWLIST reasons shorter than {MIN_ALLOWLIST_REASON_CHARS} chars: {thin}"


# --------------------------------------------------------------------------------------
# the guard and tests/support/docs_site.py on a synthetic site
# --------------------------------------------------------------------------------------


def _site(tmp_path: Path, config: str, pages: Mapping[str, str] | None = None) -> Path:
    (tmp_path / MKDOCS_CONFIG).write_text(config, encoding="utf-8")
    for relative, text in (pages or {}).items():
        page = tmp_path / relative
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(text, encoding="utf-8")
    return tmp_path


def test_a_table_moved_onto_a_nav_page_is_caught(tmp_path: Path) -> None:
    """Moving the removed table off README onto a site page is not an evasion."""
    pages = {
        README: "# Project\n",
        f"{DEFAULT_DOCS_DIR}/index.md": "Welcome.\n",
        f"{DEFAULT_DOCS_DIR}/use-cases.md": REMOVED_RTX_3090_TABLE,
    }
    root = _site(tmp_path, "nav:\n  - Home: index.md\n  - Uses: use-cases.md\n", pages)
    failures = front_door_failures(root, {})
    assert [failure.split(":")[0] for failure in failures] == [
        f"{DEFAULT_DOCS_DIR}/use-cases.md"
    ] * 6


def test_a_stale_allowlist_entry_is_reported(tmp_path: Path) -> None:
    """An exemption expires when its figure goes, and an exemption for a non-page never applied."""
    pages = {README: "Search is 5× faster.\n", f"{DEFAULT_DOCS_DIR}/index.md": "Welcome.\n"}
    root = _site(tmp_path, "nav:\n  - index.md\n", pages)
    allowlist = {
        (README, "5× faster"): "live: the figure is on the page and unbacked",
        (README, "a line that was reworded"): "stale: matches nothing any more",
        ("docs/not-in-nav.md", "5× faster"): "stale: not a front door",
    }
    assert front_door_failures(root, allowlist) == []
    assert stale_allowlist_entries(root, allowlist) == [
        f"{README}: 'a line that was reworded'",
        "docs/not-in-nav.md: '5× faster'",
    ]


def test_nav_pages_come_from_the_nested_nav_and_skip_links(tmp_path: Path) -> None:
    root = _site(
        tmp_path,
        "docs_dir: site_src/\n"
        "markdown_extensions:\n"
        "  - pymdownx.superfences:\n"
        "      custom_fences:\n"
        "        - format: !!python/name:pymdownx.superfences.fence_code_format\n"
        "nav:\n"
        "  - Home: index.md\n"
        "  - Guide:\n"
        "      - Start: guide/start.md\n"
        "      - Repo (GitHub): https://example.org/blob/HEAD/ARCHITECTURE.md\n"
        "  - Assets: logo.png\n"
        "  - Placeholder: null\n",
    )
    assert mkdocs_nav_pages(root) == ["site_src/index.md", "site_src/guide/start.md"]


def test_front_doors_put_readme_first_once_with_the_default_docs_dir(tmp_path: Path) -> None:
    root = _site(tmp_path, "nav:\n  - index.md\n  - Again: index.md\n")
    assert front_door_documents(root) == [README, f"{DEFAULT_DOCS_DIR}/index.md"]


@pytest.mark.parametrize("config", ["site_name: x\n", "nav: []\n"])
def test_a_site_without_a_nav_has_only_the_readme(tmp_path: Path, config: str) -> None:
    assert front_door_documents(_site(tmp_path, config)) == [README]


# --------------------------------------------------------------------------------------
# an allowlist entry exempts the figure inside its fragment, and nothing beside it
# --------------------------------------------------------------------------------------


def _readme_site(tmp_path: Path, text: str) -> Path:
    pages = {README: f"{text}\n", f"{DEFAULT_DOCS_DIR}/index.md": "Welcome.\n", **BENCH_RUN}
    return _site(tmp_path, "nav:\n  - index.md\n", pages)


def _copy_of_front_door(tmp_path: Path, document: str, text: str) -> Path:
    """A site whose front doors are README and ``document``, holding ``text``.

    ``results/`` and ``config/baselines/`` link to this repository's, so the copy's citations
    resolve exactly as the live page's do.
    """
    if document == README:
        root = _site(tmp_path, "nav: []\n", {README: text})
    else:
        assert document.startswith(f"{DEFAULT_DOCS_DIR}/"), f"{document} is not a docs page"
        nav = f"nav:\n  - {document.removeprefix(f'{DEFAULT_DOCS_DIR}/')}\n"
        root = _site(tmp_path, nav, {README: "# Project\n", document: text})
    for evidence in (RESULTS_ROOT, BASELINES_ROOT):
        link = root / evidence
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(REPO_ROOT / evidence, target_is_directory=True)
    return root


@pytest.mark.parametrize("planted", PLANTED_CLAIMS)
@pytest.mark.parametrize(("document", "fragment"), ALLOWLIST, ids=ALLOWLIST_IDS)
def test_a_figure_planted_on_a_live_allowlisted_line_is_reported(
    tmp_path: Path, document: str, fragment: str, planted: str
) -> None:
    """The reviewed defect, on every live entry: an unreviewed figure on an allowlisted line.

    On a copy of the live page, a figure appended to the line or written before the fragment is
    reported, and the figure the entry was written for stays exempt.
    """
    before, after, token = PLANTED_CLAIMS[planted]
    own = {key: reason for key, reason in ALLOWLIST.items() if key[0] == document}
    lines = (REPO_ROOT / document).read_text(encoding="utf-8").splitlines()
    hits = [index for index, line in enumerate(lines) if fragment in line]
    assert hits, f"{fragment!r} is on no line of {document}"
    root = _copy_of_front_door(tmp_path, document, "\n".join(lines) + "\n")
    assert unexempted_numbers(root, own) == [], "the copy must start as clean as the live page"
    for index in hits:
        lines[index] = lines[index].replace(fragment, before + fragment, 1) + after
    (root / document).write_text("\n".join(lines) + "\n", encoding="utf-8")
    reported = [(u.number.line_no, u.number.token) for _, u in unexempted_numbers(root, own)]
    assert reported == [(index + 1, token) for index in hits]
    assert stale_allowlist_entries(root, own) == []


@pytest.mark.parametrize(
    ("line", "fragment", "tokens"), NEIGHBOURING_FIGURES.values(), ids=NEIGHBOURING_FIGURES
)
def test_an_exemption_covers_only_the_figure_inside_its_fragment(
    tmp_path: Path, line: str, fragment: str, tokens: tuple[str, ...]
) -> None:
    """The defect class on a constant: an entry exempts its own figure and nothing beside it."""
    allowlist = {(README, fragment): "live: exempts the one figure inside its fragment"}
    root = _readme_site(tmp_path, line)
    reported = [(u.number.token, u.number.column) for _, u in unexempted_numbers(root, allowlist)]
    assert reported == [(token, line.rindex(token)) for token in tokens]
    assert stale_allowlist_entries(root, allowlist) == []


@pytest.mark.parametrize(
    ("text", "fragment", "tokens"), EXEMPTS_NOTHING.values(), ids=EXEMPTS_NOTHING
)
def test_an_entry_whose_fragment_holds_no_unbacked_figure_is_stale(
    tmp_path: Path, text: str, fragment: str, tokens: tuple[str, ...]
) -> None:
    """Judged per entry: an unbacked figure elsewhere on the line keeps no entry alive."""
    allowlist = {(README, fragment): "stale: no unbacked figure lies inside the fragment"}
    root = _readme_site(tmp_path, text)
    assert stale_allowlist_entries(root, allowlist) == [f"{README}: {fragment!r}"]
    assert [u.number.token for _, u in unexempted_numbers(root, allowlist)] == list(tokens)


def test_a_number_without_a_column_is_never_exempt() -> None:
    """Unlocated, a figure cannot be shown to lie inside a fragment, so no entry exempts it.

    The refused fallback -- "the token occurs in the fragment" -- would exempt a second,
    identical figure anywhere on the line (``NEIGHBOURING_FIGURES["same-figure-twice"]``).
    """
    line = "Search is 5× faster."
    allowlist = {(README, "Search is 5× faster"): "live: exempts the one figure inside it"}
    located = PerformanceNumber(1, line, "speedup", "5×", column=line.index("5×"))
    assert _allowlisted(README, located, allowlist)
    assert not _allowlisted(README, replace(located, column=None), allowlist)


def test_failures_name_the_column_of_the_reported_figure(tmp_path: Path) -> None:
    """Two equal figures on one line, one exempt: the report must say which was not."""
    line, fragment, _ = NEIGHBOURING_FIGURES["same-figure-twice"]
    root = _readme_site(tmp_path, line)
    (failure,) = front_door_failures(root, {(README, fragment): "live: exempts the first 5×"})
    assert failure.startswith(f"{README}:1:{line.rindex('5×') + 1}: speedup '5×' -- ")
