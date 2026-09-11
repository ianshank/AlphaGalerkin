# Delta: `project-charter` — focus-secrets-merge-gate

This change modifies one row under the **Accepted Deviation Disclosure**
Requirement. No other Requirement changes: no package is added or removed
(Scope Integrity), no numeric claim moves (Evidence-Backed Claims), no
scenario is registered (Capability Register Accuracy), and no coverage gate
is added, dropped or re-thresholded (Quality Gate Fidelity — `ci-success`
membership is not a coverage gate and is guarded separately).

## MODIFIED Requirements

### Requirement: Accepted Deviation Disclosure

One deviation row changes. No Requirement text outside the deviations table
changes.

**Amended.** *"Two tracks are frozen rather than active or removed"* said the
freeze was enforced by `scripts/check_focus.py`, and `docs/FOCUS.md` (which
the row cites) said promoting the `focus` job into `ci-success` was "a
separate PR". This change is that PR. The row now states that the check is
enforced through CI's `focus` job, a hard merge gate in `ci-success` since
2026-09-11, and names the one legitimate skip and the guard.

**Retirement condition (unchanged):** rewrite or remove the row when
`config/focus.yaml` is re-scoped.

Live table row (applied when this change lands):

| Deviation | Reason |
| --- | --- |
| Two tracks are frozen rather than active or removed | The refinement thesis now has a committed interpretable answer (`results/mcts_classical_amr_arena.csv`, median `l2_error_ratio_at_matched_dof` **0.9532**). The thesis freeze lifted on that signed result. `codec` and `interactive-surfaces` remain paused so the split-attention gate keeps working until a follow-up edits `config/focus.yaml` (empty `frozen_tracks` is rejected). Frozen code stays in the tree, green in CI, and keeps its coverage gate. Recorded in `docs/FOCUS.md`, enforced by `scripts/check_focus.py` through CI's `focus` job, which is a **hard merge gate** in `ci-success` since 2026-09-11 (pull-request runs only; `ci-success` accepts `skipped` from it exactly when the event is not a pull request or the visible `focus-override` label is present, and fails the build on any other skip — `tests/docs/test_ci_success_hard_gates.py` guards both clauses). **Retirement:** rewrite or remove this row when `config/focus.yaml` is re-scoped. |

#### Scenario: The scope-containment job reports without blocking
- GIVEN `focus` is in `ci.yml` with a `pull_request`-only `if:`
- AND it is absent from `ci-success.needs` or named only in an `echo`
- WHEN `tests/docs/test_ci_success_hard_gates.py` runs
- THEN it SHALL fail on `test_promoted_job_is_in_ci_success_needs[focus]` or
  `test_promoted_job_is_hard_gated[focus]`

#### Scenario: A skipped scope check is excused on an unlabelled pull request
- GIVEN the `focus` gate in `ci-success` accepts `skipped` without consulting
  `github.event_name` and the `focus-override` label
- WHEN the guard runs
- THEN `test_focus_gate_accepts_skipped_only_when_explained` SHALL fail

#### Scenario: A new job lands as a silent report
- GIVEN a job is added to `ci.yml` that is neither in `ci-success.needs ∩ hard`
  nor listed in the guard's `DISCLOSED_NOT_HARD` with a reason
- WHEN the guard runs
- THEN `test_every_job_is_a_hard_gate_or_a_disclosed_exception` SHALL fail

> **Note on guard coverage.** The three scenarios above are mechanical
> (`tests/docs/test_ci_success_hard_gates.py`, four planted mutations killed).
> Whether the *prose* of the row still matches the workflow is a review
> obligation; the existing deviation guard checks only that a reason of at
> least 20 characters is present.
