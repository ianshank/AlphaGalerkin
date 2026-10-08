# Design: `arena-lookahead-attribution`

Only the decisions a reviewer could reasonably push back on.

## 1. Correct the attribution, keep the number and the gate

0.9532 is a true number: the MCTS arm's error at 287 DOF really is 4.7% below Dörfler's, and the
pre-registered gate (`l2_error_ratio_at_matched_dof < 1`) really passes. What was wrong is the
reading of it as a look-ahead effect. So nothing is retracted as false; the attribution is
corrected, the gate and verdict table stay, and the spec's "Win" row is annotated rather than
rewritten -- rewriting a pre-registration after the result is the failure mode
pre-registration exists to prevent. The annotation says what the verdict is evidence of:
marking granularity (single-element vs bulk), which the greedy control isolates.

## 2. A required label keyed on the artifact's own record, not a phrase ban

The guard does not look for wrong wording; it looks for missing disclosure. A claim that cites
an artifact whose sidecar records `decisions_diverging_from_greedy_max == 0` must carry the
canonical label **"search contributed no decisions"** (`NO_LOOKAHEAD_LABEL`, defined once in
`tests/support/charter.py`). That is spelling-independent in the direction that matters: no
paraphrase of "MCTS wins" escapes it, because the trigger is the citation plus the sidecar's
metric, not the prose. Matching is case-insensitive over whitespace-collapsed text, so a label
wrapped across Markdown lines still counts.

Both directions are checked. A cited arena sidecar recording divergence above zero -- and none
recording zero -- makes the label stale, and the guard fails: a disclosure that stops being
true is a claim the artifact no longer supports. Divergence above zero is still not evidence of
look-ahead (at `top_k_actions=0` one simulation diverges by tie-break alone; see
`src/research/greedy_control.py`), so the guard never *requires* any wording when divergence is
positive -- it only stops the zero-divergence label from surviving its fact.

## 3. Schema completeness, keyed on the harness

Requirement (a) fires only when the metric is present and zero. An arena sidecar written
before the greedy control existed carries no divergence metric at all, so a claim citing it
would need no label -- the old artifact would dodge the rule exactly when it matters most.
Every sidecar whose `harness` is `scripts.run_mcts_classical_amr_arena` (the harness module's
own `HARNESS_NAME`) must therefore record `decisions_diverging_from_greedy` and
`decisions_diverging_from_greedy_max` (the keys are `amr_arena_types`' constants) as finite,
non-negative numbers -- a NaN would compare unequal to zero and dodge (a) the same way. The
scan covers every `results/**/*.run.json` and every sidecar a claim cites.

## 4. No retracted-phrase constant in `tests/support/cut_modules.py`

Considered and rejected, measured rather than assumed. The candidate needles were the three
live spellings (`"MCTS ~4.7% better"`, `"MCTS wins ~4.7%"`, `"MCTS **wins** ~4.7%"`). Run through
the existing consumers' own logic before this change:

- `tests/regression/test_retracted_claims_guard.py` (block-level markers over `docs/business/`,
  `docs/doe_genesis/`, `README.md`, `CLAUDE.md`) would report
  `docs/business/COMMERCIALIZATION_PEER_REVIEW.md:23` as live -- the review quoting the claim
  it overturns, in a block with no marker word. That file is outside this change and must not
  be edited to placate a guard.
- On `CLAUDE.md` the same guard is vacuous for this phrase: the milestone list is one
  contiguous block full of marker words ("CORRECTED", "retract"), so the struck 2026-09-08
  milestone would be excused whatever it said. The same holds for the README roadmap list.
- `CHANGELOG.md` carries "MCTS wins ~4.7%" and is append-only and unscanned.

A phrase list is also the brittle form: it catches three spellings of one sentence, while the
label rule (§2) catches every claim that cites the artifact. The constant would add a false
positive, a vacuous scan and a vocabulary to maintain, for coverage §2 already provides.

## 5. README claims are read block by block

`amr_policy_ratio_subjects()` moved from `tests/docs/test_charter_alignment.py` to
`tests/support/charter.py` (both guards read it; a second copy is how parsers drift). Moving it
exposed a blind spot: README subjects were single lines, and the README's main arena bullet
wraps over seven lines with no single line naming MCTS, Dörfler and a ratio, so neither the
manifest-pointer guard nor this one ever examined it. README is now split with the Markdown
block splitter `tests/support/perf_claims.py` already uses: a paragraph or list item is one
subject with its continuation lines, a table row is one subject, and a line inside a fenced
block stays one subject, as before. On the tree this change starts from, the per-line scan saw
one README subject; the block scan sees two, and the manifest-pointer guard passes on both.

## 6. Charter amended in place, delta recorded here

Following `openspec/changes/focus-secrets-merge-gate/` (design §6) and
`hygiene-hardening-cycle`, both of which landed their charter edit in the same change as the
delta: the live charter is edited here and the delta carries the applied text, so the
correction and the guard that enforces it cannot land apart.

## 7. The freeze lift stands

The FOCUS freeze lifted on the pre-registered verdict, and the verdict still holds. Re-freezing
because the verdict turned out to measure something narrower would be a scope decision, and
`config/focus.yaml` is the owner's to change. The deviation row and `docs/FOCUS.md` now say
what the lift rests on and that the thesis question is open.

## 8. The recorder pre-flight lives in the CLI

`RunRecorder` was adopted in `scripts/run_mcts_classical_amr_arena.py`, not in
`write_arena_manifest`: the sidecar is written by the scenario
(`src/poc/scenarios/mcts_classical_amr_arena.py`, outside this change), which always passes
`proposal_grade=False`; the CLI is what owns the proposal-grade decision. The CLI pre-flights
with `RunRecorder.start` and, after the run, requires the sidecar read back from disk to carry
the pre-flight's git snapshot. It does not compare config hashes: the scenario's `setup`
installs default thresholds after the pre-flight, so the recorded hash is legitimately the
post-setup one (a test pins this, so a later "tightening" that would reject every run fails
by name). `src/research/mcts_classical_amr_arena.py` is untouched and outputs are identical.
