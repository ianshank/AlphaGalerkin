# Delta: `project-charter` — hygiene-hardening-cycle

This change modifies the **Accepted Deviation Disclosure** Requirement.
**Scope Integrity** is acknowledged only to record that four in-register
packages stay (no row added or removed). The other Requirements are
untouched. `src/device.py` is a root module, not a package, and is **not**
a Scope Integrity edit.

## MODIFIED Requirements

### Requirement: Accepted Deviation Disclosure

Two deviation rows change. No Requirement text outside the deviations table
changes.

**Amended.** *“Two tracks are frozen rather than active or removed”* still
described the freeze as lifting when the refinement experiment has an
interpretable answer. That answer is committed
(`results/mcts_classical_amr_arena.csv`, median
`l2_error_ratio_at_matched_dof` **0.9532**, θ=0.5, policy `max_dof=600`,
matched DOF 287). The row’s reason is rewritten: the thesis freeze lifted;
`codec` and `interactive-surfaces` remain paused until a follow-up edits
`config/focus.yaml`. Empty `frozen_tracks` is rejected by the focus config
validator; promoting the `focus` job into `ci-success` is a separate PR.

**Retirement condition (amended row):** rewrite or remove the row when
`config/focus.yaml` is re-scoped (new frozen set, or a successor focus
document).

**Added.** Four B10 packages (`src/prototyping/`, `src/analysis/`,
`src/curriculum/`, `src/tournament/`) stay in the scope register. They are
in-tree, test-held, and not production-wired. This is an owner decision,
not a 2026-07-22-style cut. AGENT.md in each package carries the same
keep-reason.

**Retirement condition (added row):** remove the row when a dedicated
change either wires a production caller in `src/` outside each package’s
own tests, or cuts a package through Non-Goal Exclusion with a
`CUT_MODULES` entry. Silent deletion is the failure mode.

Live table rows (applied when this change lands):

| Deviation | Reason |
| --- | --- |
| Two tracks are frozen rather than active or removed | The refinement thesis now has a committed interpretable answer (`results/mcts_classical_amr_arena.csv`, median `l2_error_ratio_at_matched_dof` **0.9532**). The thesis freeze lifted on that signed result. `codec` and `interactive-surfaces` remain paused so the split-attention gate keeps working until a follow-up edits `config/focus.yaml` (empty `frozen_tracks` is rejected). Frozen code stays in the tree, green in CI, and keeps its coverage gate. Recorded in `docs/FOCUS.md`, enforced by `scripts/check_focus.py`. **Retirement:** rewrite or remove this row when `config/focus.yaml` is re-scoped. |
| Four B10 packages stay in scope without a production caller | `src/prototyping/`, `src/analysis/`, `src/curriculum/`, and `src/tournament/` are in-tree, test-held, and not production-wired. They remain in the scope register. This is not a 2026-07-22-style cut (`video_compression` was restored the next day). **Retirement:** remove this row when a dedicated change wires a production `src/` caller outside each package’s tests, or cuts a package through Non-Goal Exclusion with a `CUT_MODULES` entry. |

#### Scenario: The frozen-tracks row still claims the thesis has no interpretable answer
- GIVEN `results/mcts_classical_amr_arena.csv` is a committed artifact
- AND the charter frozen-tracks reason still says the freeze lifts when the
  refinement experiment has an interpretable answer, as a future event
- WHEN a reviewer reads Accepted Deviation Disclosure
- THEN the row SHALL be rewritten to the lift-already-fired form above

> **Note on guard coverage.** The existing guard checks that every deviation
> states a *reason* of at least 20 characters. It does not check that the
> frozen-tracks row matches the committed arena CSV, and this change does
> not add that check — “the reason is in the present tense” is a review
> obligation. Claiming a mechanical guard here would be the unguarded
> assertion the charter exists to stop.

#### Scenario: A B10 package is deleted without a charter cut
- GIVEN `src/prototyping/` (or analysis / curriculum / tournament) is removed
  from disk while still listed in the scope register
- WHEN the scope guard runs
- THEN it SHALL fail (existing Scope Integrity phantom check)
- AND the keep-reasons deviation SHALL still be present until a dedicated
  cut change retires it via Non-Goal Exclusion

### Requirement: Scope Integrity

Unchanged as a Requirement. No package is added or removed. The four B10
packages stay in the register; the keep-reason lives under Accepted
Deviation Disclosure (above), not as a new register row.

`src/device.py` is a root module (`src/seeding.py` shape). It MUST NOT appear
in this register: `tests/docs/test_architecture_map.py` enumerates
`src/*/__init__.py` only, and a module-shaped row fails both directions of
the scope guard.

#### Scenario: A root module is added to the package map
- GIVEN `src/device.py` exists and has no `src/device/__init__.py`
- WHEN the scope register lists `src/device/`
- THEN the architecture-map / charter scope guards SHALL fail
- AND the correct documentation is unguarded prose in `ARCHITECTURE.md`
  (“four root-level modules”), not a register row
