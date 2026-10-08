"""A claim citing an arena-family run whose search never left greedy marking must say so.

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

The same held for Gate 1 (``specs/lookahead_vs_greedy.spec.md``): both pre-registered
testbeds came out NO-GO with the deterministic search making greedy's 30 decisions, and its
sidecars record that as ``primary_decisions_diverging_from_greedy`` -- a metric the guard's
first version, which read only the arena's ``decisions_diverging_from_greedy_max``, could not
see. The *arena family* is therefore declared in :data:`DIVERGENCE_SCHEMAS`: each harness whose
sidecars record a greedy-divergence count, with the metric that decides the label (the
arena's maximum over seeds; Gate 1's deterministic primary arm -- its root-noise arm departs
from greedy through noise, not search, and neither requires nor excuses the label).

The trigger is the artifact, not the prose, so no paraphrase escapes it:

* **(a)** an MCTS-vs-Dörfler policy-ratio subject -- a charter evidence row or a ``README.md``
  paragraph, list item or table row, read by ``tests/support/charter.py``'s shared
  :func:`~tests.support.charter.amr_policy_ratio_subjects` -- that cites a run (a ``.csv``,
  brace form included, or its ``.run.json``) whose sidecar records its harness's label metric
  as 0 must carry :data:`~tests.support.charter.NO_LOOKAHEAD_LABEL`; and the label may not
  outlive its fact (a claim whose cited arena-family runs all diverged may not carry it);
* **(b)** every sidecar written by an arena-family harness -- cited or not -- records its
  schema's metrics as finite, non-negative numbers; otherwise an old-schema artifact needs no
  label and dodges (a) exactly when it matters;
* **(c)** vacuity: the scan finds a sidecar of *each* declared harness; every committed
  sidecar that records a divergence count belongs to a declared harness (an undeclared one is
  invisible to (a)); each harness has a claim the subject scan examines in the evidence
  register; an arena claim is examined on *both* surfaces; no README paragraph or list item
  that states an arena-family ratio goes unexamined (the README's main arena bullet wraps over
  seven lines and was invisible to the old per-line scan); and the label is stated in the
  charter's evidence register.

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
   (and, while it was the only labelled row, ``test_the_label_is_stated_in_the_evidence_register``).
2. The label removed from the charter's arena row -> the first of those tests, naming the row.
   Since the Gate 1 rows also carry the label, the register-wide check fires only when no row
   does: ``TestPlantedDefects::``
   ``test_an_evidence_register_without_the_label_turns_the_vacuity_check_red``.
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

The Gate 1 extension (2026-10-08) was mutation-tested the same way; each defect edited this
module and was restored byte-for-byte, and the planted twin keeps it running in CI:

17. **The label read only from the arena's maximum** (the guard before the extension) ->
    ``TestPlantedDefects::test_removing_the_label_from_a_gate1_row_turns_the_guard_red``.
18. Gate 1's label read from the root-noise arm (23 departures) ->
    ``test_the_label_does_not_outlive_its_fact`` (and
    ``TestPlantedDefects::test_a_gate1_label_read_from_the_noise_arm_is_stale``).
19. Gate 1's schema deleted -> ``test_every_harness_recording_divergence_is_declared``.
20. The divergence stem matching nothing ->
    ``TestPlantedDefects::test_an_undeclared_divergence_harness_turns_the_vocabulary_check_red``.
21. Vacuity counted in total rather than per harness ->
    ``TestPlantedDefects::test_a_harness_with_no_sidecar_in_view_turns_the_vacuity_check_red``.
22. The ledger check not filtered by harness ->
    ``TestPlantedDefects::test_gate1_rows_the_scan_cannot_see_turn_the_ledger_check_red``.
23. The label-metric meta-check emptied ->
    ``TestPlantedDefects::test_a_label_metric_outside_the_required_set_turns_the_meta_check_red``.
24. (b) reading the arena's metrics for every harness ->
    ``test_arena_sidecars_record_the_divergence_metrics``.
25. "Stale" read as *any* cited run diverging, not *every* one ->
    ``TestPlantedDefects::test_a_label_stays_while_any_cited_run_stayed_greedy``.
26. The plants' label pattern matching only an unwrapped label ->
    ``TestPlantedDefects::test_a_diverging_run_requires_no_label`` (README's Gate 1 bullet
    wraps the label across two lines; ``_unlabelled`` now also asserts no statement survives).
27. The harness filter admitting every sidecar ->
    ``test_arena_sidecars_record_the_divergence_metrics``.
28. The committed-sidecar scan globbing the real root instead of ``REPO_ROOT`` ->
    ``TestPlantedDefects::test_an_uncited_committed_sidecar_is_held_to_the_schema``. Before
    that plant it survived: every committed arena-family sidecar is also cited, so the
    citation half of the scan masked the committed half.

12/12 killed.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from src.research.amr_arena_types import DIVERGENCE_MAX_METRIC, DIVERGENCE_METRIC
from src.research.lookahead_vs_greedy_artifacts import HARNESS_NAME as GATE1_HARNESS_NAME
from src.research.lookahead_vs_greedy_metrics import (
    PRIMARY_DIVERGENCE_METRIC,
    ROBUST_DIVERGENCE_MAX_METRIC,
)
from src.research.mcts_classical_amr_arena import HARNESS_NAME as ARENA_HARNESS_NAME
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

#: Vacuity floor: sidecars the scan must find for each declared harness.
MIN_SIDECARS_PER_HARNESS: Final[int] = 1

#: Every greedy-divergence metric name carries this stem, whatever its prefix or
#: suffix; a sidecar recording one under an undeclared harness is one (a) cannot read.
DIVERGENCE_METRIC_STEM: Final[str] = DIVERGENCE_METRIC


@dataclass(frozen=True)
class DivergenceSchema:
    """Where one harness of the arena family records whether its search left greedy.

    ``label_metric`` decides the label: a claim citing a run that recorded it as zero
    must say :data:`NO_LOOKAHEAD_LABEL`. Each of ``required_metrics`` must be a finite,
    non-negative count in every sidecar the harness writes, so a claim cannot dodge (a)
    by citing a run that recorded nothing.
    """

    label_metric: str
    required_metrics: tuple[str, ...]


#: The arena family, keyed by the ``harness`` a sidecar names. A harness that records a
#: divergence count and is missing here is caught by
#: ``test_every_harness_recording_divergence_is_declared``.
DIVERGENCE_SCHEMAS: Final[dict[str, DivergenceSchema]] = {
    # Per-seed counts; the label reads the maximum, so one seed that diverged drops it.
    ARENA_HARNESS_NAME: DivergenceSchema(
        label_metric=DIVERGENCE_MAX_METRIC, required_metrics=DIVERGENCE_METRICS
    ),
    # Gate 1: the deterministic primary arm is the search its verdict reads (criterion
    # C2). The root-noise arm departs from greedy through noise in its decision rule,
    # not through search, so its count neither requires nor excuses the label.
    GATE1_HARNESS_NAME: DivergenceSchema(
        label_metric=PRIMARY_DIVERGENCE_METRIC, required_metrics=(PRIMARY_DIVERGENCE_METRIC,)
    ),
}


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
    """Arena-family sidecars among the committed and the cited ones, and read problems."""
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
        elif manifest is not None and manifest.harness in DIVERGENCE_SCHEMAS:
            arena.append((relative, manifest))
    return arena, problems


def _sidecars_of(harness: str) -> set[str]:
    """The arena-family sidecars in view that one harness wrote."""
    return {relative for relative, manifest in _arena_sidecars()[0] if manifest.harness == harness}


def _label_reading(manifest: RunManifest) -> tuple[str, float] | None:
    """``(label metric, recorded value)`` for an arena-family run; ``None`` otherwise."""
    schema = DIVERGENCE_SCHEMAS.get(manifest.harness)
    if schema is None or schema.label_metric not in manifest.metrics:
        return None
    return schema.label_metric, manifest.metrics[schema.label_metric]


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
            elif manifest is not None and (reading := _label_reading(manifest)) is not None:
                metric, value = reading
                if value == NO_DIVERGENCE:
                    no_divergence.append(f"{sidecar} ({metric} == 0)")
        if no_divergence and not carries_no_lookahead_label(body):
            failures.append(
                f"{source}: cites {no_divergence}, but does not say {NO_LOOKAHEAD_LABEL!r}"
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
            sidecar: reading
            for sidecar in cited_sidecars_in(body)
            if (manifest := _read_sidecar(sidecar)[0]) is not None
            and (reading := _label_reading(manifest)) is not None
        }
        if recorded and all(value != NO_DIVERGENCE for _, value in recorded.values()):
            stale.append(f"{source}: says {NO_LOOKAHEAD_LABEL!r}, but its runs record {recorded}")
    assert not stale, (
        "claims carry the no-look-ahead label although every arena-family run they cite "
        "records its search diverging from greedy -- the disclosure outlived its fact:\n  "
        + "\n  ".join(stale)
    )


def test_arena_sidecars_record_the_divergence_metrics() -> None:
    """(b) An arena sidecar without the metrics would let a claim citing it dodge (a)."""
    arena, problems = _arena_sidecars()
    failures = list(problems)
    for relative, manifest in arena:
        for key in DIVERGENCE_SCHEMAS[manifest.harness].required_metrics:
            value = manifest.metrics.get(key)
            if not _is_count(value):
                failures.append(f"{relative}: {key}={value!r}")
    assert not failures, (
        "sidecars written by an arena-family harness must record their schema's divergence "
        "metrics as finite, non-negative counts:\n  "
        + "\n  ".join(failures)
        + "\n\nRe-record the artifact with the greedy control (include_greedy_control, the "
        "default) -- the claims-ledger / run-provenance skills."
    )


@pytest.mark.parametrize("harness", sorted(DIVERGENCE_SCHEMAS))
def test_the_arena_sidecar_scan_is_not_vacuous(harness: str) -> None:
    """(c) Without a sidecar of each declared harness in view, (a) and (b) check nothing.

    Per harness, not in total: one harness still matching would mask a renamed other.
    """
    found = _sidecars_of(harness)
    assert len(found) >= MIN_SIDECARS_PER_HARNESS, (
        f"found {len(found)} sidecar(s) written by {harness}; at least "
        f"{MIN_SIDECARS_PER_HARNESS} expected under {COMMITTED_SIDECARS_GLOB} or cited by a claim"
    )


def test_every_harness_recording_divergence_is_declared() -> None:
    """(c) A sidecar counting greedy divergence under an undeclared harness escapes (a)."""
    undeclared: list[str] = []
    for path in sorted(REPO_ROOT.glob(COMMITTED_SIDECARS_GLOB)):
        relative = path.relative_to(REPO_ROOT).as_posix()
        manifest, problem = _read_sidecar(relative)
        if problem is not None:
            undeclared.append(problem)
        elif (
            manifest is not None
            and manifest.harness not in DIVERGENCE_SCHEMAS
            and any(DIVERGENCE_METRIC_STEM in key for key in manifest.metrics)
        ):
            undeclared.append(f"{relative} (harness {manifest.harness!r})")
    assert not undeclared, (
        "sidecars record a greedy-divergence count under a harness DIVERGENCE_SCHEMAS does "
        "not declare, so no claim citing them is held to the label:\n  " + "\n  ".join(undeclared)
    )


@pytest.mark.parametrize("harness", sorted(DIVERGENCE_SCHEMAS))
def test_each_harness_has_a_claim_examined_in_the_evidence_register(harness: str) -> None:
    """(c) The ledger states every arena-family result, and the subject scan reads it.

    A row the scan cannot see -- one that drops the Dörfler vocabulary, say -- is a row
    (a) never holds to the label.
    """
    sidecars = _sidecars_of(harness)
    examined = [
        source
        for source, body in amr_policy_ratio_subjects()
        if source.startswith(EVIDENCE_SOURCE_PREFIX) and sidecars & set(cited_sidecars_in(body))
    ]
    assert examined, (
        f"no evidence-register claim citing a {harness} run was examined (its sidecars: "
        f"{sorted(sidecars)}); state the result in the register, in words the subject scan "
        "recognises"
    )


@pytest.mark.parametrize("harness", sorted(DIVERGENCE_SCHEMAS))
def test_each_schema_requires_its_own_label_metric(harness: str) -> None:
    """A label metric outside (b)'s required set could be misspelled and read as absent."""
    schema = DIVERGENCE_SCHEMAS[harness]
    assert schema.label_metric in schema.required_metrics


def test_an_arena_claim_is_examined_on_each_surface() -> None:
    """(c) Coverage, not "something matched": one surface still matching can mask the other."""
    arena = _sidecars_of(ARENA_HARNESS_NAME)
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
    """(c) The register states the disclosure at all; (a) holds each row to it."""
    assert carries_no_lookahead_label(region("evidence")), (
        f"the charter's evidence register does not state {NO_LOOKAHEAD_LABEL!r}; the arena "
        "row must say the 0.9532 is greedy marking, and the Gate 1 rows that the search made "
        "greedy's decisions"
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

#: The label as a plant must find it: any whitespace, line breaks included, between
#: its words -- a label wrapped across two Markdown lines is still the label.
_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\s+".join(map(re.escape, NO_LOOKAHEAD_LABEL.split())), re.IGNORECASE
)


def _cited_in_evidence(harness: str) -> list[str]:
    """Sidecars of ``harness`` the live charter evidence register cites, in row order."""
    written = _sidecars_of(harness)
    return [
        sidecar
        for source, body in amr_policy_ratio_subjects()
        if source.startswith(EVIDENCE_SOURCE_PREFIX)
        for sidecar in cited_sidecars_in(body)
        if sidecar in written
    ]


def _arena_sidecar() -> str:
    """The arena sidecar the live charter evidence register cites."""
    arena = _sidecars_of(ARENA_HARNESS_NAME)
    cited = [
        sidecar
        for source, body in amr_policy_ratio_subjects()
        if source.startswith(EVIDENCE_SOURCE_PREFIX)
        for sidecar in cited_sidecars_in(body)
        if sidecar in arena
    ]
    assert cited, "no charter evidence row cites an arena sidecar"
    return cited[0]


def _evidence_row_citing(sidecar: str) -> str:
    """The one live evidence-register line citing ``sidecar``."""
    rows = [line for line in region("evidence").splitlines() if sidecar in cited_sidecars_in(line)]
    assert len(rows) == 1, rows
    return rows[0]


def _arena_evidence_row() -> str:
    """The live evidence-register line citing the arena sidecar."""
    return _evidence_row_citing(_arena_sidecar())


def _gate1_sidecar() -> str:
    """The first Gate 1 sidecar the live charter evidence register cites."""
    cited = _cited_in_evidence(GATE1_HARNESS_NAME)
    assert cited, "no charter evidence row cites a Gate 1 sidecar"
    return cited[0]


def _claim_of(row: str) -> str:
    """An evidence row's claim cell: what a subject's source names after the prefix."""
    return row.split("|")[1].strip()


def _unlabelled(text: str) -> str:
    """``text`` with every statement of the label replaced; asserts it had one.

    And asserts none survives as the guard reads it: a plant that misses one statement
    (a wrapped one, say) leaves a claim labelled and silently weakens the test using it.
    """
    planted, count = _LABEL_PATTERN.subn(_NOT_THE_LABEL, text)
    assert count, "the plant found no label to remove -- it would silently no-op"
    assert not carries_no_lookahead_label(planted), "the plant left a statement of the label"
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


def _set_family_divergence(root: Path, value: float | None) -> None:
    """Set (``None``: delete) every schema metric in every arena-family sidecar under ``root``."""
    for relative, manifest in _arena_sidecars()[0]:
        schema = DIVERGENCE_SCHEMAS[manifest.harness]
        _set_metrics(root / relative, **dict.fromkeys(schema.required_metrics, value))


def _run_every_guard() -> None:
    """Every guard above, each parametrised one over every declared harness."""
    test_claims_citing_a_no_divergence_run_carry_the_label()
    test_the_label_does_not_outlive_its_fact()
    test_arena_sidecars_record_the_divergence_metrics()
    test_every_harness_recording_divergence_is_declared()
    for harness in DIVERGENCE_SCHEMAS:
        test_the_arena_sidecar_scan_is_not_vacuous(harness)
        test_each_harness_has_a_claim_examined_in_the_evidence_register(harness)
        test_each_schema_requires_its_own_label_metric(harness)
    test_an_arena_claim_is_examined_on_each_surface()
    test_no_readme_arena_claim_goes_unexamined()
    test_the_label_is_stated_in_the_evidence_register()


class TestPlantedDefects:
    def test_the_unplanted_copy_passes_every_guard(self, planted_root: Path) -> None:
        """Control: the planted runs below fail because of the plant, not the copy."""
        _run_every_guard()

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
        source = f"{EVIDENCE_SOURCE_PREFIX}{_claim_of(row)}"
        with pytest.raises(AssertionError, match=re.escape(source)):
            test_claims_citing_a_no_divergence_run_carry_the_label()

    def test_an_evidence_register_without_the_label_turns_the_vacuity_check_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every row unlabelled: the register no longer states the disclosure at all."""
        text = charter_support.charter_text()
        evidence = region("evidence")
        planted = text.replace(evidence, _unlabelled(evidence), 1)
        assert planted != text
        _plant_documents(monkeypatch, charter=planted)
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
        _set_family_divergence(planted_root, 1.0)
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
        _set_family_divergence(planted_root, None)
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
        expected = f"at least {MIN_SIDECARS_PER_HARNESS} expected"
        for harness in DIVERGENCE_SCHEMAS:
            with pytest.raises(AssertionError, match=expected):
                test_the_arena_sidecar_scan_is_not_vacuous(harness)

    def test_a_per_line_readme_scan_turns_the_coverage_check_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The adjacent weaker defect: the scan this module replaced, restored."""

        def per_line(markdown: str) -> list[tuple[int, str]]:
            return list(enumerate(markdown.splitlines(), start=1))

        monkeypatch.setattr("tests.support.charter.markdown_units", per_line)
        with pytest.raises(AssertionError, match="never examined"):
            test_no_readme_arena_claim_goes_unexamined()

    # -- Gate 1: the second harness of the arena family -----------------------------

    def test_removing_the_label_from_a_gate1_row_turns_the_guard_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Gate 1 row is read through its own label metric, not the arena's.

        Before ``DIVERGENCE_SCHEMAS`` the guard read only the arena's maximum, which a
        Gate 1 sidecar does not record, so this row would pass unlabelled.
        """
        row = _evidence_row_citing(_gate1_sidecar())
        _plant_documents(
            monkeypatch, charter=charter_support.charter_text().replace(row, _unlabelled(row), 1)
        )
        source = f"{EVIDENCE_SOURCE_PREFIX}{_claim_of(row)}"
        with pytest.raises(AssertionError, match=re.escape(source)):
            test_claims_citing_a_no_divergence_run_carry_the_label()
        test_the_label_is_stated_in_the_evidence_register()  # the arena row still says it

    def test_a_gate1_label_read_from_the_noise_arm_is_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Root noise departs from greedy without search; reading it would flip the label."""
        noise = DivergenceSchema(
            label_metric=ROBUST_DIVERGENCE_MAX_METRIC,
            required_metrics=(ROBUST_DIVERGENCE_MAX_METRIC,),
        )
        monkeypatch.setitem(DIVERGENCE_SCHEMAS, GATE1_HARNESS_NAME, noise)
        with pytest.raises(AssertionError, match="outlived its fact"):
            test_the_label_does_not_outlive_its_fact()

    def test_an_undeclared_divergence_harness_turns_the_vocabulary_check_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sidecar = _gate1_sidecar()
        monkeypatch.delitem(DIVERGENCE_SCHEMAS, GATE1_HARNESS_NAME)
        with pytest.raises(AssertionError, match=re.escape(sidecar)):
            test_every_harness_recording_divergence_is_declared()

    def test_a_harness_with_no_sidecar_in_view_turns_the_vacuity_check_red(self) -> None:
        """Per harness: the arena's sidecars must not stand in for a renamed Gate 1."""
        renamed = f"{GATE1_HARNESS_NAME}_renamed"
        with pytest.raises(AssertionError, match=re.escape(f"written by {renamed}")):
            test_the_arena_sidecar_scan_is_not_vacuous(renamed)

    def test_a_gate1_sidecar_without_its_metric_turns_the_schema_check_red(
        self, planted_root: Path
    ) -> None:
        sidecar = _gate1_sidecar()
        _set_metrics(planted_root / sidecar, **{PRIMARY_DIVERGENCE_METRIC: None})
        with pytest.raises(
            AssertionError, match=re.escape(f"{sidecar}: {PRIMARY_DIVERGENCE_METRIC}=None")
        ):
            test_arena_sidecars_record_the_divergence_metrics()

    def test_gate1_rows_the_scan_cannot_see_turn_the_ledger_check_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rows without the Dörfler vocabulary are not subjects, so (a) never reads them."""
        text = charter_support.charter_text()
        planted = text
        for sidecar in _cited_in_evidence(GATE1_HARNESS_NAME):
            row = _evidence_row_citing(sidecar)
            planted = planted.replace(row, row.replace("Dörfler", "classical"), 1)
        assert planted != text
        _plant_documents(monkeypatch, charter=planted)
        with pytest.raises(AssertionError, match=re.escape(f"citing a {GATE1_HARNESS_NAME} run")):
            test_each_harness_has_a_claim_examined_in_the_evidence_register(GATE1_HARNESS_NAME)

    def test_a_label_metric_outside_the_required_set_turns_the_meta_check_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        typo = DivergenceSchema(
            label_metric=f"{PRIMARY_DIVERGENCE_METRIC}_typo",
            required_metrics=(PRIMARY_DIVERGENCE_METRIC,),
        )
        monkeypatch.setitem(DIVERGENCE_SCHEMAS, GATE1_HARNESS_NAME, typo)
        with pytest.raises(AssertionError):
            test_each_schema_requires_its_own_label_metric(GATE1_HARNESS_NAME)

    def test_a_label_stays_while_any_cited_run_stayed_greedy(self, planted_root: Path) -> None:
        """Only one Gate 1 testbed diverges: its own row's label is stale, no other is.

        README's Gate 1 claims cite both testbeds and still need the label for the one
        that stayed greedy; reading "stale" as "any cited run diverged" would flag them.
        """
        sidecar = _gate1_sidecar()
        _set_metrics(planted_root / sidecar, **{PRIMARY_DIVERGENCE_METRIC: 1.0})
        test_claims_citing_a_no_divergence_run_carry_the_label()
        with pytest.raises(AssertionError) as raised:
            test_the_label_does_not_outlive_its_fact()
        message = str(raised.value)
        assert _claim_of(_evidence_row_citing(sidecar)) in message
        assert README_SOURCE_PREFIX not in message, message

    def test_an_uncited_committed_sidecar_is_held_to_the_schema(self, planted_root: Path) -> None:
        """(b) reads every committed arena-family sidecar, not only the ones a claim cites.

        Every committed one is cited today, so without this plant the committed-file
        scan could read the wrong root and nothing would notice.
        """
        uncited = Path(_arena_sidecar()).with_name(f"uncited{SIDECAR_SUFFIX}").as_posix()
        shutil.copyfile(planted_root / _arena_sidecar(), planted_root / uncited)
        _set_metrics(planted_root / uncited, **{DIVERGENCE_MAX_METRIC: None})
        with pytest.raises(AssertionError, match=re.escape(f"{uncited}: {DIVERGENCE_MAX_METRIC}")):
            test_arena_sidecars_record_the_divergence_metrics()
