# Proposal: `arena-lookahead-attribution`

## Why

The charter's evidence register, its Novelty requirement, its frozen-tracks deviation row,
`docs/FOCUS.md`, `README.md`, the 2026-09-08 `CLAUDE.md` milestone and the arena spec's
"Win" row all read the committed arena result as an MCTS / look-ahead win:
median `l2_error_ratio_at_matched_dof` **0.9532**, "MCTS ~4.7% better at matched DOF".

That attribution is wrong, and the committed artifact now says so itself.
`docs/business/COMMERCIALIZATION_PEER_REVIEW.md` §1 found the published MCTS trajectory
bit-for-bit identical to single-element greedy marking at every simulation budget it tried.
Gate 0.1 (code half, merged 2026-10-08) added the control that separates the two effects --
a single-element greedy arm on the MCTS arm's exact game (`src/research/greedy_control.py`)
and a `decisions_diverging_from_greedy` counter -- and this change re-records the artifact
with it, from a clean tree under `--proposal-grade`
(`results/mcts_classical_amr_arena.{csv,run.json}`, sidecar `dirty: false`, SHA `f2c65c4`):

| Metric | Value |
| --- | --- |
| `l2_error_ratio_at_matched_dof` (MCTS / Dörfler θ=0.5, matched DOF 287) | 0.9531782126653989 -- bit-identical to the 2026-09-08 record |
| `l2_error_ratio_greedy_over_dorfler_at_matched_dof` | 0.9531782126653989 |
| `l2_error_ratio_mcts_over_greedy_at_matched_dof` | 1.0 |
| `decisions_diverging_from_greedy` / `_max` (over 3 seeds) | 0 / 0 |
| `l2_error_ratio_at_matched_solves` (MCTS / Dörfler, ungated) | 9.231951512014659 -- bit-identical |

Every `uniform` / `dorfler` / `mcts` row of the CSV is identical to the previous record in every
column except `wall_time_seconds`, and the 13 new `greedy` rows equal each MCTS seed's
trajectory point for point. So 0.9532 is **single-element greedy marking against Dörfler bulk
marking**: the MCTS arm makes the greedy control's identical 12 decisions on every seed, and
**search contributed no decisions**. The cycle thesis -- *MCTS multi-step look-ahead beats
classical greedy marking* (`specs/mcts_classical_amr_arena.spec.md`) -- is untested by this
artifact. Whether look-ahead beats greedy anywhere is Gate 1 of the peer review.

Two further facts the old text got wrong or left out:

- **The binding limit is `max_steps=12`, not `max_dof=600`.** Both game-driven arms stop after
  12 single-element refinements at 287 DOF; the 600-DOF policy budget never binds. Every
  citation quoted "policy `max_dof=600`".
- **The 9.23 at matched solves is not the price of a quality edge.** The greedy control reaches
  the same mesh in 13 solves; the MCTS arm spends 43.

The owner accepted this correction (decision 1, peer review §7).

## What Changes

### Evidence-Backed Claims

- **Amend** the element-local AMR row: 0.9532 is attributed to single-element greedy marking
  vs Dörfler θ=0.5 at matched DOF 287; MCTS/greedy 1.0 and divergence 0 are stated; the row
  carries the label **"search contributed no decisions"**, states that look-ahead is untested
  (Gate 1 pending), names `max_steps=12` as the binding limit, and keeps MCTS at matched solves
  9.23 (ungated) and the adequacy-rate caveat.
- **Add** a scenario and its guard, `tests/docs/test_lookahead_attribution.py`: an AMR
  policy-ratio claim (charter evidence row or README) citing an artifact whose sidecar records
  `decisions_diverging_from_greedy_max == 0` must carry the label; the label may not outlive
  the fact (a cited arena sidecar that records divergence above zero, and none at zero, makes
  the label stale); and every sidecar written by `scripts.run_mcts_classical_amr_arena` must
  record both divergence metrics, so an old-schema arena artifact cannot dodge the check.

### Novelty Claim Discipline

- **Amend** the arena paragraph: the committed arena result does not measure the cycle thesis;
  0.9532 is a marking-granularity effect; no text SHALL attribute it to look-ahead. The
  earlier "MCTS ~4.7% better at matched DOF" is recorded as corrected (2026-10-08).

### Accepted Deviation Disclosure

- **Amend** the frozen-tracks row, which said "The refinement thesis now has a committed
  interpretable answer". The freeze lift stands on the pre-registered verdict
  (`l2_error_ratio_at_matched_dof < 1`, which still holds); what changes is what that verdict
  is evidence of. The thesis question is open until Gate 1. Retirement condition unchanged.

### Live statements outside the charter (house style: corrections stay visible)

- `README.md`: the "Key features" arena bullet and the roadmap's scored-arena item.
- `docs/FOCUS.md`: "The current focus" and "When the freeze lifts".
- `specs/mcts_classical_amr_arena.spec.md`: the pre-registered verdict and the "Win" row are
  annotated in place, not rewritten -- it is a pre-registration.
- `CLAUDE.md`: the 2026-09-08 milestone's attribution is struck through and marked
  **CORRECTED (2026-10-08)**; a 2026-10-08 Gate 0 milestone is appended.

### Recording path

- `scripts/run_mcts_classical_amr_arena.py --proposal-grade` now pre-flights with
  `RunRecorder.start` before anything runs, and re-checks the sidecar as read back from disk.
  The shipped YAML writes into `results/`, so a dirty-tree run used to overwrite the committed
  artifacts before it was rejected.

## Impact

- The charter no longer claims the project has shown a look-ahead effect. Its only
  element-local AMR claim is a marking-granularity result, labelled as such.
- A README or charter AMR policy-ratio claim that cites a no-divergence arena run without the
  label fails CI, naming the claim. Restoring the old evidence row turns
  `test_claims_citing_a_no_divergence_run_carry_the_label` red.
- README claims are now read block by block (a paragraph or list item with its continuation
  lines; a table row; a fenced line) instead of line by line, for both this guard and the
  existing manifest-pointer guard. The README's main arena bullet wraps over seven lines and
  no single line names MCTS, Dörfler and a ratio, so the per-line scan never examined it.
- The scenario's gate (`l2_error_ratio_at_matched_dof < 1`, MCTS / Dörfler) is unchanged and
  still passes; no metric key, CSV column, threshold or config default changes.

## What This Change Does NOT Do

- **Does not change the arena's gate, verdict table or config.** The pre-registered
  `l2_error_ratio_at_matched_dof < 1` stays; `src/poc/scenarios/mcts_classical_amr_arena_config.py`
  is untouched. A Win on that metric is now documented as not, by itself, evidence of
  look-ahead.
- **Does not test look-ahead.** That is Gate 1 (a problem where greedy has a structural reason
  to be myopic), pre-registered separately.
- **Does not re-freeze anything** or edit `config/focus.yaml`; re-freezing is an owner decision.
- **Does not ban a phrase.** No retracted-phrase constant is added to
  `tests/support/cut_modules.py`; `design.md` §4 says why.
- **Does not edit history**: `CHANGELOG.md` (append-only; the correction gets its own
  `[Unreleased]` entry), `docs/business/**`, earlier change packages
  (`hygiene-hardening-cycle`, `focus-secrets-merge-gate`), and the `CLAUDE.md` Regression
  Surface table are left as they are.
- **Does not touch the capability register** or any other Requirement.
