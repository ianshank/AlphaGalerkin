# Proposal: `focus-secrets-merge-gate`

## Why

Two jobs in `.github/workflows/ci.yml` ran on every pull request and could
not fail the build:

- `focus` (Scope Containment, `scripts/check_focus.py`) landed on 2026-09-08
  with a comment saying it was "NOT in `ci-success`'s needs list, deliberately
  and temporarily" and would be promoted "once it has been green across a few
  pull requests".
- `secrets` (gitleaks) landed on 2026-08-21 with the same comment, because its
  first attempt failed for a harness reason (a shallow clone) rather than a
  finding.

Both conditions have been met: `focus` was green on every pull-request run
since it landed, and `secrets` has been green on every push since the
`fetch-depth: 0` fix. The promotion never happened, because the promise lived
in a comment and nothing read it. `docs/ENGINEERING_REFLECTION_2026-09-11.md`
§1 measured the merge gate at 9 hard + 1 soft of 14 jobs, with `focus`,
`secrets` and `test-slow` outside `needs` entirely (ticket R-09).

The charter's frozen-tracks deviation row (Accepted Deviation Disclosure)
still describes the promotion as "a separate PR". This change is that PR, so
the row must stop saying so.

## What Changes

### Accepted Deviation Disclosure

- **Amend** the frozen-tracks row: `scripts/check_focus.py` is enforced through
  CI's `focus` job, which is a hard merge gate in `ci-success` since
  2026-09-11. The row states the one legitimate skip (a non-pull-request event
  or the visible `focus-override` label) and names the guard. The retirement
  condition is unchanged.

### Workflow (mechanical, guarded)

- `focus` and `secrets` join `ci-success.needs` and its `exit 1` block.
- The `focus` gate is event-aware: it accepts `skipped` exactly when the event
  is not a pull request or the `focus-override` label is present, and fails
  the build on any other skip. `secrets` shares `ci-success`'s own `if:` and
  needs no skip handling.
- New guard `tests/docs/test_ci_success_hard_gates.py`: the promoted jobs are
  in `needs` and hard-gated; the `focus` condition carries `skipped`,
  `github.event_name` and the label; the label is spelled identically in the
  job's own `if:`; and every job in `ci.yml` is either a hard gate or a
  disclosed, self-expiring exception (`test-slow`,
  `transfer-baseline-regression`).
- Stale prose corrected: the two job comments, the `transfer-baseline`
  comment, the B38 comment inside `ci-success`, `config/focus.yaml`'s header
  (which said the check ran inside the `lint` job) and `docs/FOCUS.md`.

## Impact

- A pull request that makes a substantive change to a frozen track in the same
  changeset as core solver work is now blocked, not reported. The
  `focus-override` label remains the visible escape hatch.
- A gitleaks finding in `src/`, `scripts/` or `.github/` now blocks a merge.
- `tests/docs/test_e2e_visibility.py::ci_success_blocking_jobs` grows by two;
  its existing assertions (`{lint, test-fast} ⊆ blocking`,
  `transfer-baseline-regression ∉ blocking`) are unaffected.

## What This Change Does NOT Do

- **Does not edit `config/focus.yaml`'s tracks.** Both frozen tracks stay;
  `frozen_tracks` may not be empty. Decision D9 of the reflection plan.
- **Does not flip `transfer-baseline-regression`** to hard. That is R-07 and
  needs three `workflow_dispatch` runs first.
- **Does not narrow the `.gitleaks.toml` allowlist.** Widening the scan and
  making it blocking in one change would make a first red impossible to
  triage; the allowlist is a separate change (CLAUDE.md Next Steps).
- **Does not touch `test-slow`.** It is PR-invisible by design and is listed
  as a disclosed exception in the guard, with that reason.
