---
name: claims-ledger
description: Add, correct or retract a numeric claim in the charter's evidence register end-to-end — artifact, provenance sidecar, register row, and the guard that enforces it. Use whenever a headline number is introduced, changed or withdrawn.
---

# Add, correct or retract a claim

The charter's evidence register is this project's claims ledger. It already exists; do **not**
build a parallel `CLAIMS.yaml`. Extend the register and its guard in
`tests/docs/test_charter_alignment.py`.

Three headline numbers have been retracted here. In each case the number was written down
before, or instead of, the artifact that would support it.

## Adding or correcting a claim

1. **Produce the artifact first.** A committed `results/*.csv` from a script under `scripts/`,
   not a notebook cell and not a local run. The number follows the artifact.
2. **Write the provenance sidecar** — see the `run-provenance` skill.
3. **Check the artifact contains what the claim asserts.** A comparison claim needs an artifact
   holding *both arms*. The charter's adaptive-vs-uniform row cited a CSV whose `method` column
   held only `{dorfler, mcts}`; a correct number traced to prose, and the existence guard could
   not see it because the file existed.
4. **Add the register row** between the `<!-- charter:evidence -->` markers. State the measured
   range, not a remembered band — "1.5× at 56 DOF rising to 10.5× at 2847" beats "5–9×" and is
   harder to drift.
5. **Prefer a rate to a ratio** where one exists. A convergence exponent does not depend on
   where the reader takes the reading; a ratio does.
6. **Verify**: `pytest tests/docs/test_charter_alignment.py -v`

## Retracting a claim

1. **Add the constant** to `tests/support/cut_modules.py` — one definition, several guards.
2. **Correct every live statement.** Guards scan different surfaces: the charter, `README.md`
   + `docs/related-work.md`, `dashboard/` + `hf_space/`, and `docs/business/` +
   `docs/doe_genesis/`. A retraction that propagates to some and not others is how one survived
   in `PRIOR_ART_REVIEW.md` after the charter had already corrected it.
3. **Leave the retraction visible.** House style is to record it *in place* — see
   `specs/lshape_amr_compare.spec.md` — not to edit history away. Guards allow a retracted
   string when the surrounding block carries a marker word.
4. **Add the guard**, then **mutation-test it**: restore the defect and confirm a *named* test
   fails.

## Two ways a guard silently does nothing

Both have happened here; check for both.

- **It scans an empty set.** A guard whose roots or vocabulary match nothing passes everything.
  Add a meta-test asserting it examines at least one real subject — and prefer *coverage* of the
  vocabulary over "something matched", because one arm still matching can mask another going
  unexamined.
- **The mutation never applied.** A mutation lost to shell quoting (an umlaut, say) produces a
  false negative indistinguishable from a passing guard. Assert the file actually changed before
  trusting the result.

## Exemptions

Any allowlist entry needs a reason **and** a meta-test asserting it is still required, so a stale
exemption fails rather than rotting into a permanent blind spot. When the drafted exemption turns
out to be unnecessary, that is the meta-test working — delete it.
