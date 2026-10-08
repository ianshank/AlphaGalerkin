"""A claim citing an arena run whose search never left greedy marking must say so.

Defect class, in one sentence: **an AMR policy-ratio claim attributes to look-ahead a
result whose own run record shows the search made no decision that single-element
greedy marking would not have made.**

It happened. ``results/mcts_classical_amr_arena.csv``'s median
``l2_error_ratio_at_matched_dof`` 0.9532 was stated as "MCTS ~4.7% better at matched DOF"
in the charter's evidence register and Novelty text, ``README.md``, ``docs/FOCUS.md`` and
``CLAUDE.md``, while the MCTS trajectory was greedy marking decision for decision
(``docs/business/COMMERCIALIZATION_PEER_REVIEW.md`` §1). Re-recorded with the greedy
control, the sidecar says so itself: ``decisions_diverging_from_greedy_max`` 0, MCTS/greedy
1.0. Every check that existed asked whether a claim *cited* an artifact; none asked what the
artifact said about the claim.

The trigger is the artifact, not the prose, so no paraphrase escapes it:

* **(a)** an MCTS-vs-Dörfler policy-ratio subject -- a charter evidence row or a ``README.md``
  paragraph, list item or table row, read by ``tests/support/charter.py``'s shared
  :func:`~tests.support.charter.amr_policy_ratio_subjects` -- that cites a run (a ``.csv``,
  brace form included, or its ``.run.json``) whose sidecar records
  ``decisions_diverging_from_greedy_max == 0`` must carry
  :data:`~tests.support.charter.NO_LOOKAHEAD_LABEL`; and the label may not outlive its fact
  (a claim whose cited arena runs all record divergence above zero may not carry it);
* **(b)** every sidecar written by the arena harness records both divergence metrics as
  finite, non-negative numbers -- otherwise an old-schema artifact needs no label and dodges
  (a) exactly when it matters;
* **(c)** vacuity: the scan finds an arena sidecar; it examines an arena claim on *both*
  surfaces; no README paragraph or list item that states an arena ratio goes unexamined (the
  README's main arena bullet wraps over seven lines and was invisible to the old per-line
  scan); and the label is stated in the charter's evidence register.

Divergence above zero is not evidence of look-ahead either (``src/research/greedy_control.py``:
at ``top_k_actions=0`` one simulation diverges by tie-break alone), so nothing is *required*
when it is positive. ``TestPlantedDefects`` keeps the kills below running in CI; every plant is
a monkeypatched document or a ``tmp_path`` sidecar copy -- no committed file is edited.

Mutation kills (``harden-a-guard``, 2026-10-08; no killer is ``gpu_required`` /
``fem_required``). Plants 1-5 substitute a document in memory or rewrite a ``tmp_path``
sidecar copy, through a throwaway pytest plugin, and each is also pinned below as a
``TestPlantedDefects`` test; mutations 6-16 edited this module or
``tests/support/charter.py`` and restored them byte-for-byte.

1. **The literal historical defect** -- the pre-correction evidence row ("MCTS **wins**
   ~4.7%", no label) restored -> ``test_claims_citing_a_no_divergence_run_carry_the_label``
   and ``test_the_label_is_stated_in_the_evidence_register``.
2. The label removed from the charter's arena row -> the same two tests.
3. The label removed from ``README.md`` ->
   ``test_claims_citing_a_no_divergence_run_carry_the_label``, naming both README arena claims.
4. A sidecar copy with both divergence metrics set to 1, label kept ->
   ``test_the_label_does_not_outlive_its_fact``; with the label also removed every guard
   passes (the positive direction, ``TestPlantedDefects::test_a_diverging_run_requires_no_label``).
5. ``decisions_diverging_from_greedy_max`` deleted from a sidecar copy ->
   ``test_arena_sidecars_record_the_divergence_metrics``.
6. The label required whatever the divergence ->
   ``TestPlantedDefects::test_a_diverging_run_requires_no_label``.
7. The stale-label check disabled ->
   ``TestPlantedDefects::test_a_label_on_a_diverging_run_is_stale``.
8. NaN, negative or infinite counts accepted ->
   ``TestPlantedDefects::test_a_non_count_divergence_turns_the_schema_check_red``.
9. The harness filter matching nothing -> ``test_the_arena_sidecar_scan_is_not_vacuous`` and
   ``test_an_arena_claim_is_examined_on_each_surface``.
10. An unreadable sidecar skipped silently ->
    ``TestPlantedDefects::test_an_unreadable_cited_sidecar_turns_the_guard_red``.
11. Sidecars resolved against the real root instead of ``REPO_ROOT`` ->
    ``TestPlantedDefects::test_a_root_without_sidecars_turns_the_vacuity_check_red``.
12. **The subject scan emptied** -> ``test_an_arena_claim_is_examined_on_each_surface`` and
    ``test_no_readme_arena_claim_goes_unexamined`` (and the charter guard's
    ``test_amr_policy_ratio_scan_is_not_vacuous``).
13. **The per-line README scan restored** -- the adjacent weaker defect, which kept a
    single-line README claim in view and so passed the per-surface check ->
    ``test_no_readme_arena_claim_goes_unexamined`` and
    ``TestSharedHelpers::test_a_wrapped_list_item_is_one_unit``.
14. Brace expansion dropped from ``cited_sidecars_in`` ->
    ``test_an_arena_claim_is_examined_on_each_surface`` (the charter cites the arena only as
    ``results/x.{csv,run.json}``).
15. The label matched verbatim (case, emphasis and wrapping significant) ->
    ``TestSharedHelpers::test_the_label_survives_formatting``.
16. ``TABLE_BLOCK_KIND`` drifting from ``split_blocks``' own literal ->
    ``TestSharedHelpers::test_the_splitter_still_names_tables_the_way_the_units_expect``.

16/16 planted defects killed; the test count is larger and is not the number claimed. Plant
3 also exposed a defect in the first draft: one check read ``README.md`` through a name bound
at import, so a substituted README reached the subject scan and not that check (a spurious
second failure). Every document is now read through ``tests.support.charter`` at call time.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Final

import pytest

from src.research.amr_arena_types import DIVERGENCE_MAX_METRIC, DIVERGENCE_METRIC
from src.research.mcts_classical_amr_arena import HARNESS_NAME
from src.research.run_manifest import RunManifest, load_run_manifest
from tests.support import charter as charter_support
from tests.support.charter import (
    EVIDENCE_SOURCE_PREFIX,
    NO_LOOKAHEAD_LABEL,
    README_SOURCE_PREFIX,
    SIDECAR_SUFFIX,
    TABLE_BLOCK_KIND,
    amr_policy_ratio_subjects,
    carries_no_lookahead_label,
    cited_sidecars_in,
    is_amr_policy_ratio_claim,
    markdown_units,
    region,
)
from tests.support.perf_claims import split_blocks

# Documents are always read through ``charter_support`` (``charter_text`` /
# ``readme_text``), never through names bound at import, so a document a test
# substitutes is the one every check sees.

#: Where sidecar paths resolve. Module state, not a constant, so the planted tests
#: can point every guard at a ``tmp_path`` copy.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Committed sidecars, scanned for the arena harness whether or not a claim cites them.
COMMITTED_SIDECARS_GLOB: Final[str] = f"results/**/*{SIDECAR_SUFFIX}"

#: The divergence count that means the search changed no decision.
NO_DIVERGENCE: Final[float] = 0.0

#: Metrics every arena sidecar must record, so a claim citing it can be checked.
DIVERGENCE_METRICS: Final[tuple[str, ...]] = (DIVERGENCE_METRIC, DIVERGENCE_MAX_METRIC)

#: Vacuity floor: arena sidecars the scan must find.
MIN_ARENA_SIDECARS: Final[int] = 1


def _read_sidecar(relative: str) -> tuple[RunManifest | None, str | None]:
    """``(manifest, None)``; ``(None, why)`` if unreadable; ``(None, None)`` if absent.

    A missing sidecar is not this guard's business: the charter guard's
    manifest-pointer check owns existence, so one root cause yields one failure.
    """
    path = REPO_ROOT / relative
    if not path.is_file():
        return None, None
    try:
        return load_run_manifest(path), None
    except (OSError, ValueError) as exc:  # unreadable, not JSON, invalid, or newer schema
        return None, f"{relative} cannot be read as a run manifest: {exc}"


def _arena_sidecars() -> tuple[list[tuple[str, RunManifest]], list[str]]:
    """Arena-harness sidecars among the committed and the cited ones, and read problems."""
    candidates = {
        path.relative_to(REPO_ROOT).as_posix() for path in REPO_ROOT.glob(COMMITTED_SIDECARS_GLOB)
    }
    candidates |= {
        sidecar for _, body in amr_policy_ratio_subjects() for sidecar in cited_sidecars_in(body)
    }
    arena: list[tuple[str, RunManifest]] = []
    problems: list[str] = []
    for relative in sorted(candidates):
        manifest, problem = _read_sidecar(relative)
        if problem is not None:
            problems.append(problem)
        elif manifest is not None and manifest.harness == HARNESS_NAME:
            arena.append((relative, manifest))
    return arena, problems


def _is_count(value: float | None) -> bool:
    """A recorded decision count: finite and not negative."""
    return value is not None and math.isfinite(value) and value >= NO_DIVERGENCE


# --------------------------------------------------------------------------------------
# the guards
# --------------------------------------------------------------------------------------


def test_claims_citing_a_no_divergence_run_carry_the_label() -> None:
    """(a) A claim citing a run whose search never left greedy must say so."""
    failures: list[str] = []
    for source, body in amr_policy_ratio_subjects():
        no_divergence: list[str] = []
        for sidecar in cited_sidecars_in(body):
            manifest, problem = _read_sidecar(sidecar)
            if problem is not None:
                failures.append(f"{source}: {problem}")
            elif manifest is not None:
                if manifest.metrics.get(DIVERGENCE_MAX_METRIC) == NO_DIVERGENCE:
                    no_divergence.append(sidecar)
        if no_divergence and not carries_no_lookahead_label(body):
            failures.append(
                f"{source}: cites {no_divergence}, which records {DIVERGENCE_MAX_METRIC} == 0, "
                f"but does not say {NO_LOOKAHEAD_LABEL!r}"
            )
    assert not failures, (
        "AMR policy-ratio claims attribute to look-ahead a run whose search made no decision "
        "greedy marking would not have made:\n  "
        + "\n  ".join(failures)
        + f"\n\nState {NO_LOOKAHEAD_LABEL!r} in the claim (the ratio is single-element greedy "
        "marking, not search), or cite a run in which the search diverged."
    )


def test_the_label_does_not_outlive_its_fact() -> None:
    """(a) A label on a claim whose cited arena runs all diverged from greedy is stale."""
    stale: list[str] = []
    for source, body in amr_policy_ratio_subjects():
        if not carries_no_lookahead_label(body):
            continue
        recorded = {
            sidecar: manifest.metrics[DIVERGENCE_MAX_METRIC]
            for sidecar in cited_sidecars_in(body)
            if (manifest := _read_sidecar(sidecar)[0]) is not None
            and DIVERGENCE_MAX_METRIC in manifest.metrics
        }
        if recorded and NO_DIVERGENCE not in recorded.values():
            stale.append(f"{source}: says {NO_LOOKAHEAD_LABEL!r}, but its runs record {recorded}")
    assert not stale, (
        "claims carry the no-look-ahead label although every arena run they cite records "
        f"{DIVERGENCE_MAX_METRIC} above zero -- the disclosure outlived its fact:\n  "
        + "\n  ".join(stale)
    )


def test_arena_sidecars_record_the_divergence_metrics() -> None:
    """(b) An arena sidecar without the metrics would let a claim citing it dodge (a)."""
    arena, problems = _arena_sidecars()
    failures = list(problems)
    for relative, manifest in arena:
        for key in DIVERGENCE_METRICS:
            value = manifest.metrics.get(key)
            if not _is_count(value):
                failures.append(f"{relative}: {key}={value!r}")
    assert not failures, (
        f"sidecars written by {HARNESS_NAME} must record {list(DIVERGENCE_METRICS)} as finite, "
        "non-negative counts:\n  "
        + "\n  ".join(failures)
        + "\n\nRe-record the artifact with the greedy control (include_greedy_control, the "
        "default) -- the claims-ledger / run-provenance skills."
    )


def test_the_arena_sidecar_scan_is_not_vacuous() -> None:
    """(c) Without an arena sidecar in view, (a) and (b) check nothing."""
    arena, _ = _arena_sidecars()
    assert len(arena) >= MIN_ARENA_SIDECARS, (
        f"found {len(arena)} sidecar(s) written by {HARNESS_NAME}; at least "
        f"{MIN_ARENA_SIDECARS} expected under {COMMITTED_SIDECARS_GLOB} or cited by a claim"
    )


def test_an_arena_claim_is_examined_on_each_surface() -> None:
    """(c) Coverage, not "something matched": one surface still matching can mask the other."""
    arena = {relative for relative, _ in _arena_sidecars()[0]}
    sources = [
        source
        for source, body in amr_policy_ratio_subjects()
        if arena & set(cited_sidecars_in(body))
    ]
    for prefix in (EVIDENCE_SOURCE_PREFIX, README_SOURCE_PREFIX):
        assert any(source.startswith(prefix) for source in sources), (
            f"no claim from {prefix!r} citing an arena run was examined (examined: {sources}); "
            "the subject scan or its vocabulary drifted"
        )


def test_no_readme_arena_claim_goes_unexamined() -> None:
    """(c) A paragraph or list item stating an arena ratio is examined whole.

    Read independently of the subject scan, through the block splitter itself: a scan
    that fell back to single lines would lose a claim wrapped across lines, exactly as
    the per-line scan lost README's main arena bullet.
    """
    arena = {relative for relative, _ in _arena_sidecars()[0]}
    examined = {
        body
        for source, body in amr_policy_ratio_subjects()
        if source.startswith(README_SOURCE_PREFIX)
    }
    missed = [
        f"{README_SOURCE_PREFIX}{block.lines[0][0]}"
        for block in split_blocks(charter_support.readme_text())
        if block.kind != TABLE_BLOCK_KIND
        and is_amr_policy_ratio_claim(block.text)
        and arena & set(cited_sidecars_in(block.text))
        and block.text not in examined
    ]
    assert not missed, f"README arena claims the subject scan never examined: {missed}"


def test_the_label_is_stated_in_the_evidence_register() -> None:
    """(c) The charter's own arena row carries the disclosure."""
    assert carries_no_lookahead_label(region("evidence")), (
        f"the charter's evidence register does not state {NO_LOOKAHEAD_LABEL!r}; the arena "
        "row must say the 0.9532 is greedy marking, not look-ahead"
    )


# --------------------------------------------------------------------------------------
# the shared helpers, on synthetic input
# --------------------------------------------------------------------------------------


class TestSharedHelpers:
    """Branches of ``tests/support/charter.py`` the live documents do not reach."""

    def test_a_wrapped_list_item_is_one_unit(self) -> None:
        markdown = "- MCTS vs Dörfler\n  median ratio 0.95\n- next item\n"
        assert markdown_units(markdown) == [
            (1, "- MCTS vs Dörfler\n  median ratio 0.95"),
            (3, "- next item"),
        ]

    def test_table_rows_and_fenced_lines_are_units_of_their_own(self) -> None:
        markdown = "| a | b |\n| --- | --- |\n| 1 | 2 |\n\n```\nx = 1\n```\n"
        assert markdown_units(markdown) == [
            (1, "| a | b |"),
            (2, "| --- | --- |"),
            (3, "| 1 | 2 |"),
            (5, "```"),
            (6, "x = 1"),
            (7, "```"),
        ]

    def test_the_splitter_still_names_tables_the_way_the_units_expect(self) -> None:
        """``TABLE_BLOCK_KIND`` mirrors a literal inside ``split_blocks``; pin the coupling."""
        assert [block.kind for block in split_blocks("| a |\n| --- |\n")] == [TABLE_BLOCK_KIND]

    @pytest.mark.parametrize(
        "text",
        [
            f"so **{NO_LOOKAHEAD_LABEL}**, and",
            NO_LOOKAHEAD_LABEL.upper(),
            NO_LOOKAHEAD_LABEL.replace(" ", "\n  ", 1),
            NO_LOOKAHEAD_LABEL.replace("no", "*no*", 1),
        ],
        ids=["bold", "upper-case", "wrapped", "emphasis-inside"],
    )
    def test_the_label_survives_formatting(self, text: str) -> None:
        assert carries_no_lookahead_label(text)

    def test_a_paraphrase_is_not_the_label(self) -> None:
        assert not carries_no_lookahead_label("the search made no difference")

    @pytest.mark.parametrize(
        "cell",
        [
            "`results/x.{csv,run.json}`",
            "`results/x.csv` and again `results/x.run.json`",
            "`results/x.run.json`",
        ],
        ids=["brace", "csv-then-sidecar", "sidecar-only"],
    )
    def test_every_citation_form_reaches_the_sidecar_once(self, cell: str) -> None:
        assert cited_sidecars_in(cell) == ["results/x.run.json"]

    def test_other_artifacts_have_no_sidecar(self) -> None:
        assert cited_sidecars_in("`results/x.png` `specs/x.spec.md` `n_simulations=1`") == []


# --------------------------------------------------------------------------------------
# planted defects -- each must turn a named guard above red
# --------------------------------------------------------------------------------------

_GUARD_MODULE = sys.modules[__name__]

#: The arena evidence row as it stood before the 2026-10-08 correction, verbatim: the
#: literal historical defect -- 0.9532 read as an MCTS win, with no disclosure.
HISTORICAL_EVIDENCE_ROW: Final[str] = (
    "| Element-local AMR, MCTS vs Dörfler at matched DOF (θ=0.5, policy max_dof=600, "
    "matched_dof=287) | median ratio 0.9532 (MCTS **wins** ~4.7%; 3 identical seeds; evaporates "
    "at matched solves, ratio 9.23 ungated). Adequacy rates over (200, 4000) are **not** this "
    "result. | `results/mcts_classical_amr_arena.{csv,run.json}` |"
)

#: What a plant writes in place of the label.
_NOT_THE_LABEL: Final[str] = "look-ahead won"

_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(re.escape(NO_LOOKAHEAD_LABEL), re.IGNORECASE)


def _arena_sidecar() -> str:
    """The arena sidecar the live charter evidence register cites."""
    arena = {relative for relative, _ in _arena_sidecars()[0]}
    cited = [
        sidecar
        for source, body in amr_policy_ratio_subjects()
        if source.startswith(EVIDENCE_SOURCE_PREFIX)
        for sidecar in cited_sidecars_in(body)
        if sidecar in arena
    ]
    assert cited, "no charter evidence row cites an arena sidecar"
    return cited[0]


def _arena_evidence_row() -> str:
    """The live evidence-register line citing the arena sidecar."""
    sidecar = _arena_sidecar()
    rows = [line for line in region("evidence").splitlines() if sidecar in cited_sidecars_in(line)]
    assert len(rows) == 1, rows
    return rows[0]


def _unlabelled(text: str) -> str:
    """``text`` with every statement of the label replaced; asserts it had one."""
    planted, count = _LABEL_PATTERN.subn(_NOT_THE_LABEL, text)
    assert count, "the plant found no label to remove -- it would silently no-op"
    return planted


def _plant_documents(
    monkeypatch: pytest.MonkeyPatch, *, charter: str | None = None, readme: str | None = None
) -> None:
    if charter is not None:
        monkeypatch.setattr("tests.support.charter.charter_text", lambda: charter)
    if readme is not None:
        monkeypatch.setattr("tests.support.charter.readme_text", lambda: readme)


@pytest.fixture
def planted_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``tmp_path`` copy of every sidecar the guards read, with REPO_ROOT pointed at it."""
    arena, _ = _arena_sidecars()
    committed = [
        p.relative_to(REPO_ROOT).as_posix() for p in REPO_ROOT.glob(COMMITTED_SIDECARS_GLOB)
    ]
    for relative in {*committed, *(relative for relative, _ in arena)}:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    monkeypatch.setattr(_GUARD_MODULE, "REPO_ROOT", tmp_path)
    return tmp_path


def _set_metrics(path: Path, **values: float | None) -> None:
    """Rewrite metrics of a copied sidecar; ``None`` deletes the key."""
    document = json.loads(path.read_text(encoding="utf-8"))
    for key, value in values.items():
        if value is None:
            document["metrics"].pop(key, None)
        else:
            document["metrics"][key] = value
    path.write_text(json.dumps(document), encoding="utf-8")


class TestPlantedDefects:
    def test_the_unplanted_copy_passes_every_guard(self, planted_root: Path) -> None:
        """Control: the planted runs below fail because of the plant, not the copy."""
        test_claims_citing_a_no_divergence_run_carry_the_label()
        test_the_label_does_not_outlive_its_fact()
        test_arena_sidecars_record_the_divergence_metrics()
        test_the_arena_sidecar_scan_is_not_vacuous()
        test_an_arena_claim_is_examined_on_each_surface()
        test_no_readme_arena_claim_goes_unexamined()

    def test_the_pre_correction_evidence_row_turns_the_guard_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Peer review §6, Gate 0.1 exit: a row attributing 0.9532 to look-ahead fails by name."""
        text = charter_support.charter_text()
        planted = text.replace(_arena_evidence_row(), HISTORICAL_EVIDENCE_ROW, 1)
        assert planted != text
        _plant_documents(monkeypatch, charter=planted)
        claim = HISTORICAL_EVIDENCE_ROW.split("|")[1].strip()
        with pytest.raises(AssertionError, match=re.escape(f"{EVIDENCE_SOURCE_PREFIX}{claim}")):
            test_claims_citing_a_no_divergence_run_carry_the_label()

    def test_removing_the_label_from_the_charter_row_turns_the_guard_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = _arena_evidence_row()
        _plant_documents(
            monkeypatch, charter=charter_support.charter_text().replace(row, _unlabelled(row), 1)
        )
        with pytest.raises(AssertionError, match=re.escape(EVIDENCE_SOURCE_PREFIX)):
            test_claims_citing_a_no_divergence_run_carry_the_label()
        with pytest.raises(AssertionError, match="evidence register does not state"):
            test_the_label_is_stated_in_the_evidence_register()

    def test_removing_the_label_from_the_readme_turns_the_guard_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _plant_documents(monkeypatch, readme=_unlabelled(charter_support.readme_text()))
        with pytest.raises(AssertionError, match=re.escape(README_SOURCE_PREFIX)):
            test_claims_citing_a_no_divergence_run_carry_the_label()

    def test_a_diverging_run_requires_no_label(
        self, planted_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive direction: with divergence above zero, unlabelled claims pass (a)."""
        _set_metrics(planted_root / _arena_sidecar(), **dict.fromkeys(DIVERGENCE_METRICS, 1.0))
        _plant_documents(
            monkeypatch,
            charter=_unlabelled(charter_support.charter_text()),
            readme=_unlabelled(charter_support.readme_text()),
        )
        test_claims_citing_a_no_divergence_run_carry_the_label()
        test_the_label_does_not_outlive_its_fact()

    def test_a_label_on_a_diverging_run_is_stale(self, planted_root: Path) -> None:
        _set_metrics(planted_root / _arena_sidecar(), **dict.fromkeys(DIVERGENCE_METRICS, 1.0))
        with pytest.raises(AssertionError, match="outlived its fact"):
            test_the_label_does_not_outlive_its_fact()

    @pytest.mark.parametrize("key", DIVERGENCE_METRICS)
    def test_a_missing_divergence_metric_turns_the_schema_check_red(
        self, planted_root: Path, key: str
    ) -> None:
        sidecar = _arena_sidecar()
        _set_metrics(planted_root / sidecar, **{key: None})
        with pytest.raises(AssertionError, match=re.escape(f"{sidecar}: {key}=None")):
            test_arena_sidecars_record_the_divergence_metrics()

    def test_an_old_schema_sidecar_requires_no_label_without_the_schema_check(
        self, planted_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Why (b) exists: with the metrics gone, (a) alone passes an unlabelled claim."""
        _set_metrics(planted_root / _arena_sidecar(), **dict.fromkeys(DIVERGENCE_METRICS))
        _plant_documents(
            monkeypatch,
            charter=_unlabelled(charter_support.charter_text()),
            readme=_unlabelled(charter_support.readme_text()),
        )
        test_claims_citing_a_no_divergence_run_carry_the_label()
        with pytest.raises(AssertionError, match="must record"):
            test_arena_sidecars_record_the_divergence_metrics()

    @pytest.mark.parametrize("value", [math.nan, -1.0, math.inf], ids=["nan", "negative", "inf"])
    def test_a_non_count_divergence_turns_the_schema_check_red(
        self, planted_root: Path, value: float
    ) -> None:
        sidecar = _arena_sidecar()
        _set_metrics(planted_root / sidecar, **{DIVERGENCE_MAX_METRIC: value})
        with pytest.raises(AssertionError, match=re.escape(f"{sidecar}: {DIVERGENCE_MAX_METRIC}")):
            test_arena_sidecars_record_the_divergence_metrics()

    def test_an_unreadable_cited_sidecar_turns_the_guard_red(self, planted_root: Path) -> None:
        sidecar = _arena_sidecar()
        (planted_root / sidecar).write_text("not json", encoding="utf-8")
        with pytest.raises(AssertionError, match="cannot be read as a run manifest"):
            test_claims_citing_a_no_divergence_run_carry_the_label()

    def test_an_empty_subject_scan_turns_the_vacuity_checks_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_GUARD_MODULE, "amr_policy_ratio_subjects", list)
        with pytest.raises(AssertionError, match="no claim from"):
            test_an_arena_claim_is_examined_on_each_surface()

    def test_a_root_without_sidecars_turns_the_vacuity_check_red(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_GUARD_MODULE, "REPO_ROOT", tmp_path)
        with pytest.raises(AssertionError, match=f"at least {MIN_ARENA_SIDECARS} expected"):
            test_the_arena_sidecar_scan_is_not_vacuous()

    def test_a_per_line_readme_scan_turns_the_coverage_check_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The adjacent weaker defect: the scan this module replaced, restored."""

        def per_line(markdown: str) -> list[tuple[int, str]]:
            return list(enumerate(markdown.splitlines(), start=1))

        monkeypatch.setattr("tests.support.charter.markdown_units", per_line)
        with pytest.raises(AssertionError, match="never examined"):
            test_no_readme_arena_claim_goes_unexamined()
