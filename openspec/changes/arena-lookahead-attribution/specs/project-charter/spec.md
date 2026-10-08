# Delta: `project-charter` — arena-lookahead-attribution

This change modifies three Requirements: **Evidence-Backed Claims** (one register row, one new
scenario), **Novelty Claim Discipline** (the arena paragraph) and **Accepted Deviation
Disclosure** (the frozen-tracks row). The other Requirements are untouched: no package is added
or removed (Scope Integrity), no cut module returns (Non-Goal Exclusion), no interactive surface
renders the arena figure (UI Claim Fidelity — `dashboard/` and `hf_space/` carry neither 0.9532
nor the 4.7% framing), no scenario is registered or renamed (Capability Register Accuracy), and
no coverage gate moves (Quality Gate Fidelity).

## MODIFIED Requirements

### Requirement: Evidence-Backed Claims

The Requirement text is unchanged. One register row is amended and one scenario is added.

**Amended.** *"Element-local AMR, MCTS vs Dörfler at matched DOF"* read 0.9532 as "MCTS
**wins** ~4.7%" and named "policy max_dof=600". The re-recorded artifact
(`results/mcts_classical_amr_arena.{csv,run.json}`, proposal-grade, SHA `f2c65c4`; re-recorded on `4574475` after the
config-hash fix, every row unchanged) carries a
single-element greedy control: MCTS/greedy 1.0, greedy/Dörfler 0.9532,
`decisions_diverging_from_greedy_max` 0 over three seeds. Every policy row is bit-identical to
the 2026-09-08 record. The row now attributes the ratio to greedy marking, carries the label
**"search contributed no decisions"**, says look-ahead is untested (Gate 1 pending) and names
`max_steps=12` as the binding limit.

Live table row (applied when this change lands):

| Claim | Value | Artifact |
| --- | --- | --- |
| Element-local AMR, MCTS vs Dörfler at matched DOF, with a single-element greedy control (θ=0.5, matched_dof=287; the binding limit is `max_steps=12`, not `max_dof=600`) | median ratio 0.9532 (3 identical seeds) is **single-element greedy marking** vs Dörfler bulk marking, not look-ahead: MCTS/greedy 1.0 and greedy/Dörfler 0.9532, because the MCTS arm makes the greedy control's identical 12 decisions on every seed (`decisions_diverging_from_greedy_max` 0) — **search contributed no decisions**, and look-ahead is untested by this artifact (Gate 1 pending). MCTS at matched solves 9.23 (ungated). Adequacy rates over (200, 4000) are **not** this result. | `results/mcts_classical_amr_arena.{csv,run.json}` |

#### Scenario: A greedy-marking result is attributed to look-ahead
- GIVEN an MCTS-vs-Dörfler AMR policy-ratio claim in README or the charter's evidence register
- AND it cites an artifact whose `.run.json` records `decisions_diverging_from_greedy_max == 0`
- WHEN `tests/docs/test_lookahead_attribution.py` runs
- THEN the claim SHALL carry the label "search contributed no decisions"
- AND `test_claims_citing_a_no_divergence_run_carry_the_label` SHALL fail naming the claim
  otherwise

#### Scenario: An old-schema arena artifact dodges the label
- GIVEN a `.run.json` whose `harness` is `scripts.run_mcts_classical_amr_arena`
- AND its metrics lack `decisions_diverging_from_greedy` or
  `decisions_diverging_from_greedy_max`, or record either as non-finite or negative
- WHEN the guard runs
- THEN `test_arena_sidecars_record_the_divergence_metrics` SHALL fail naming the sidecar

### Requirement: Novelty Claim Discipline

The arena paragraph changes. The rest of the Requirement — the narrow novelty form, the
TreeMesh retraction, the 2026-08-16 L-shape retraction and the tensor-product disclosure — is
unchanged.

**Amended.** The paragraph said the cycle thesis "is measured by `mcts_classical_amr_arena`" and
quoted 0.9532 as "MCTS ~4.7% better at matched DOF". It now says the committed arena result does
not measure the thesis, that 0.9532 is a marking-granularity effect, that **no text SHALL
attribute it to look-ahead**, and records the earlier framing as corrected on 2026-10-08.

Live paragraph (applied when this change lands):

> Novelty is a *method* delta, not a demonstrated win. The honest `lshape_amr_compare` result —
> MCTS **losing** at matched DOF (ratio 1.0996, wins 1/5 seeds) and losing further at matched
> compute (ratio 2.04, 0/5 seeds) — SHALL be reported alongside any favourable framing. Those
> figures are **non-informative for element-local policy** (tensor-product substrate / legacy
> harness). The cycle thesis is meant to be measured by `mcts_classical_amr_arena` on
> `SkfemTriSubstrate`, and its committed result does **not** measure it
> (`results/mcts_classical_amr_arena.{csv,run.json}`, θ=0.5, matched DOF 287; the binding limit
> is `max_steps=12`, not `max_dof=600`). Its median `l2_error_ratio_at_matched_dof` **0.9532**
> (3 identical seeds) is **single-element greedy marking** against Dörfler bulk marking, a
> marking-granularity effect: the artifact's greedy control makes the same 12 decisions as the
> MCTS arm on every seed (`decisions_diverging_from_greedy_max` 0, MCTS/greedy 1.0), so **search
> contributed no decisions**. Look-ahead is untested by this artifact; whether it beats greedy
> anywhere is Gate 1 of `docs/business/COMMERCIALIZATION_PEER_REVIEW.md`. At matched solves MCTS
> is at 9.23 (ungated). The earlier reading of 0.9532 as "MCTS ~4.7% better at matched DOF" is
> **corrected (2026-10-08)**, and no text SHALL attribute this result to look-ahead. Do **not**
> quote adequacy `N^-1.31` / ~10× rates as this result — those are gate evidence on
> `(200, 4000)`.

#### Scenario: The arena ratio is restated as a look-ahead win
- GIVEN a charter or README statement that attributes the arena's 0.9532 to MCTS look-ahead
- WHEN it cites `results/mcts_classical_amr_arena.{csv,run.json}`
- THEN the Evidence-Backed Claims scenario above SHALL fail it unless it carries the label
- AND an uncited restatement elsewhere in the prose is a **review** obligation: no phrase ban
  is added (see `design.md` §4)

### Requirement: Accepted Deviation Disclosure

One deviation row changes. No Requirement text outside the deviations table changes.

**Amended.** *"Two tracks are frozen rather than active or removed"* opened with "The refinement
thesis now has a committed interpretable answer". The arena result answers the pre-registered
gate, not the thesis. The row now says the lift rests on the pre-registered verdict (which still
holds), that the result is greedy marking with the label stated, and that the thesis question
is open until Gate 1.

**Retirement condition (unchanged):** rewrite or remove the row when `config/focus.yaml` is
re-scoped.

Live table row (applied when this change lands):

| Deviation | Reason |
| --- | --- |
| Two tracks are frozen rather than active or removed | The thesis freeze lifted on the committed arena result (`results/mcts_classical_amr_arena.{csv,run.json}`, median `l2_error_ratio_at_matched_dof` **0.9532**, the pre-registered gate). **Corrected 2026-10-08:** that result does not answer the refinement thesis. It is single-element greedy marking against Dörfler — the MCTS arm makes the greedy control's identical decisions (`decisions_diverging_from_greedy_max` 0, MCTS/greedy 1.0), so **search contributed no decisions** — and the thesis question is open until Gate 1. The lift stands on the pre-registered verdict, which still holds; re-freezing is an owner decision this correction does not make. `codec` and `interactive-surfaces` remain paused so the split-attention gate keeps working until a follow-up edits `config/focus.yaml` (empty `frozen_tracks` is rejected). Frozen code stays in the tree, green in CI, and keeps its coverage gate. Recorded in `docs/FOCUS.md`, enforced by `scripts/check_focus.py` through CI's `focus` job, which is a **hard merge gate** in `ci-success` since 2026-09-11 (pull-request runs only; `ci-success` accepts `skipped` from it exactly when the event is not a pull request or the visible `focus-override` label is present, and fails the build on any other skip — `tests/docs/test_ci_success_hard_gates.py` guards both clauses). **Retirement:** rewrite or remove this row when `config/focus.yaml` is re-scoped. |

#### Scenario: The frozen-tracks row reads the arena result as a thesis answer
- GIVEN the frozen-tracks row says the refinement thesis has an answer
- AND the cited arena sidecar records `decisions_diverging_from_greedy_max == 0`
- WHEN a reviewer reads Accepted Deviation Disclosure
- THEN the row SHALL say the thesis question is open and carry the label

> **Note on guard coverage.** The two Evidence-Backed Claims scenarios are mechanical
> (`tests/docs/test_lookahead_attribution.py`; its docstring records the planted defects each
> test killed). The guard reads the evidence register and README, not the Novelty prose, the
> deviations table, `docs/FOCUS.md`, the arena spec or `CLAUDE.md`: those corrections are a
> **review** obligation, and claiming a mechanical guard over them would be the unguarded
> assertion the charter exists to stop. The existing deviation guard still checks only that a
> reason of at least 20 characters is present.
