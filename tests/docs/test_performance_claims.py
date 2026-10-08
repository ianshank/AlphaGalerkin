"""Performance numbers on the project's front doors must cite hardware-tagged evidence.

Defect class, in one sentence: a unit-bearing performance number -- a latency, a throughput or
a speedup -- published in ``README.md`` or on the docs site with no committed artifact that
records the hardware it was measured on.

It shipped. ``README.md``'s "Benchmarks" table showed inference milliseconds and MCTS
simulations per second for three models, captioned "Benchmarks on NVIDIA RTX 3090, batch size
1". No artifact, run output or sidecar backed any of the six figures, the card matches none of
the rigs this repository documents, and no check read the table: the only README claims guard
(``test_charter_alignment.py::test_amr_policy_ratios_cite_a_manifest``) matches MCTS-vs-Dörfler
policy ratios. A person found it (``docs/business/COMMERCIALIZATION_PEER_REVIEW.md`` §5.2,
Gate 0 item 0.3), as people found the seven enforcement gaps before it.

Scope -- the front doors (``tests/support/docs_site.py``): ``README.md`` and every local page in
``mkdocs.yml``'s nav, read from the nav rather than listed, so moving a table onto a site page
is not an evasion. What a *performance number* is, and what backs one, live in
``tests/support/perf_claims.py``: a number must share its paragraph, list item or table with a
cited ``results/`` or ``config/baselines/`` artifact whose run provenance records a real
``hardware_tag`` -- a timing with no machine attached is not evidence. ``README.md``'s
Performance section states the same rule (a ``.run.json`` sidecar recording the tag), so the
prose promises nothing the gate does not enforce; the gate additionally accepts a
``config/baselines/`` document, which records its own ``hardware_tag``.

``ALLOWLIST`` holds the exceptions: each is a disclosed debt with a stated reason, and each
expires -- an entry that no longer matches an *unbacked* number (the claim was backed,
reworded or deleted) fails ``test_allowlist_entries_are_still_needed``.

Mutation kills -- 11/11 planted defects, each failing a NAMED test, none of them
``gpu_required`` or ``fem_required``:

1. README's RTX 3090 table restored verbatim (the literal historical defect) ->
   ``test_front_door_performance_numbers_cite_hardware_tagged_artifacts``.
2. ``"ms"`` deleted from ``SUBSECOND_UNITS`` ->
   ``test_perf_claims_scanner.py::test_synthetic_positive_is_detected[unit-ms]``.
3. ``"sec"`` deleted from ``THROUGHPUT_DENOMINATORS`` ->
   ``test_synthetic_positive_is_detected[rate-slash-sec]`` and
   ``test_the_removed_rtx_3090_table_is_caught`` (its "Sims/sec" column goes dark).
4. Header propagation removed -- the adjacent weaker defect, since every cell of the real
   table is a bare number -> ``test_the_removed_rtx_3090_table_is_caught``.
5. The hardware-tag check dropped, so any existing artifact backs a claim (the letter of
   "cite a committed artifact" without its point) ->
   ``test_perf_claims_scanner.py::test_an_artifact_without_hardware_provenance_backs_nothing``.
6. A README paragraph "Inference takes 12 ms per move" citing
   ``results/lshape_adaptive_vs_uniform.csv``, whose sidecar records ``hardware_tag``
   ``unknown`` -> ``test_front_door_performance_numbers_cite_hardware_tagged_artifacts``.
7. The same claim citing ``results/transfer_baseline_compare.csv`` (committed, no sidecar) ->
   ``test_front_door_performance_numbers_cite_hardware_tagged_artifacts``.
8. The removed table moved onto ``docs/use-cases.md`` ->
   ``test_front_door_performance_numbers_cite_hardware_tagged_artifacts``.
9. Inline code stripped before scanning, making backticks an evasion ->
   ``test_perf_claims_scanner.py::test_inline_code_is_not_skipped``.
10. An allowlisted ``c4_mermaid.md`` line reworded without dropping its entry ->
    ``test_allowlist_entries_are_still_needed``.
11. A 39-character allowlist reason -> ``test_allowlist_reasons_are_substantive``.
"""

from __future__ import annotations

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
    RESULTS_ROOT,
    SIDECAR_SUFFIX,
    TABLE_COLUMN_KIND,
    UnbackedNumber,
    citation_problem,
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

#: ``(document, literal line fragment) -> reason``. Exempts the unbacked performance numbers on
#: lines of ``document`` that contain the fragment. A disclosed debt, not an endorsement.
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


def _front_doors() -> list[str]:
    return front_door_documents(REPO_ROOT)


def _unbacked(document: str) -> list[UnbackedNumber]:
    text = (REPO_ROOT / document).read_text(encoding="utf-8")
    return unbacked_performance_numbers(text, REPO_ROOT)


def _allowlisted(document: str, line: str) -> bool:
    return any(doc == document and fragment in line for doc, fragment in ALLOWLIST)


# --------------------------------------------------------------------------------------
# vacuity: the scan reads real documents, and the scanner sees the defect it exists for
# --------------------------------------------------------------------------------------


def test_front_door_scan_is_not_vacuous() -> None:
    """Without this, an empty nav or a renamed README would make every assertion below pass."""
    documents = _front_doors()
    assert README in documents and DOCS_INDEX in documents, documents
    assert len(documents) >= MIN_FRONT_DOORS, documents
    empty = [doc for doc in documents if not (REPO_ROOT / doc).read_text(encoding="utf-8").strip()]
    assert not empty, f"front-door documents that are missing content: {empty}"


def test_the_removed_rtx_3090_table_is_caught() -> None:
    """The literal historical defect must stay visible to the scanner, figure by figure."""
    unbacked = unbacked_performance_numbers(REMOVED_RTX_3090_TABLE, REPO_ROOT)
    assert tuple(u.number.token for u in unbacked) == REMOVED_TABLE_FIGURES
    assert {u.number.kind for u in unbacked} == {TABLE_COLUMN_KIND}


def test_the_rule_is_satisfiable_with_committed_evidence() -> None:
    """A rule no committed artifact can meet would be a ban wearing a guard's clothes."""
    sidecars = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / RESULTS_ROOT).glob(f"*{SIDECAR_SUFFIX}")
    )
    tagged = [sidecar for sidecar in sidecars if citation_problem(REPO_ROOT, sidecar) is None]
    assert tagged, f"no committed sidecar records a hardware_tag: {sidecars}"
    claim = f"Inference takes 12 ms on the tagged host (`{tagged[0]}`)."
    assert unbacked_performance_numbers(claim, REPO_ROOT) == []


# --------------------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------------------


def test_front_door_performance_numbers_cite_hardware_tagged_artifacts() -> None:
    """Every front-door latency, throughput or speedup cites hardware-tagged evidence."""
    failures = [
        f"{document}:{u.number.line_no}: {u.number.kind} {u.number.token!r} -- "
        f"{'; '.join(u.reasons)}\n      {u.number.line.strip()}"
        for document in _front_doors()
        for u in _unbacked(document)
        if not _allowlisted(document, u.number.line)
    ]
    assert not failures, (
        "unit-bearing performance numbers with no hardware-tagged artifact in their "
        "paragraph, list item or table:\n  "
        + "\n  ".join(failures)
        + "\n\nCite a committed results/ or config/baselines/ artifact whose run provenance "
        "records a hardware_tag (src/research/run_manifest.py writes one), or delete the "
        "figure. ALLOWLIST is for disclosed debt, not for new claims."
    )


def test_allowlist_entries_are_still_needed() -> None:
    """An exemption that matches no unbacked number silently shrinks the guard's scope."""
    documents = set(_front_doors())
    stale = [
        f"{document}: {fragment!r}"
        for document, fragment in ALLOWLIST
        if document not in documents
        or not any(fragment in u.number.line for u in _unbacked(document))
    ]
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
# tests/support/docs_site.py on a synthetic site
# --------------------------------------------------------------------------------------


def _site(tmp_path: Path, config: str) -> Path:
    (tmp_path / MKDOCS_CONFIG).write_text(config, encoding="utf-8")
    return tmp_path


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
