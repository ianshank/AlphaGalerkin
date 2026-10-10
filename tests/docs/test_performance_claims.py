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

``ALLOWLIST`` holds the exceptions: each is a disclosed debt with a stated reason, and each
expires -- an entry that no longer matches an *unbacked* number (the claim was backed, reworded
or deleted) fails ``test_allowlist_entries_are_still_needed``.

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
"""

from __future__ import annotations

from collections.abc import Mapping
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
