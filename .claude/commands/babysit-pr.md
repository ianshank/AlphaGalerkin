---
description: Watch a PR to green — subscribe to its activity, then triage each CI failure against this repo's known failure classes rather than re-deriving them.
argument-hint: "[PR number, default: the PR for the current branch]"
---

Drive PR `$ARGUMENTS` to a green, mergeable state. This encodes the triage that has
been re-improvised every session; the backlog items B22 and B23 were both "observed
while driving PR #140 to green", i.e. this loop, run again, with the lessons landing
in a table instead of a command.

## Mechanism, not polling

Call `subscribe_pr_activity` for the PR, then **end the turn**. Events arrive as
`<wake reason="external-event">` envelopes. Never `sleep`, never poll in a loop —
the subscription is server-side.

`gh` is **not available** in every environment this repo is worked from. Use the
GitHub MCP tools (`mcp__github__*`); `mcp__github__get_job_logs` with
`failed_only: true` is the fast path to a failure's cause. This constraint has been
rediscovered by hand more than once — it is written here so it is not rediscovered again.

## Triage: match the failure to a known class before investigating

| Symptom | Class | Action |
|---|---|---|
| `test-e2e` stalls then "runner has received a shutdown signal" | the pre-existing chess/MCTS leak (`docs/E2E_TEST_PLAN.md` §12.4) — measured 13,649 MB whole-tier vs 1,451/3,614 MB split | a **finer** split, never a bigger runner or a deleted test |
| a `tests/benchmarks/` ratio assertion, red in a full run and green alone | load sensitivity, recorded in CLAUDE.md Next Steps (measured 2026-08-21: failed under load, passed 2 s later at load 0.96) | re-run once to confirm; **never** widen the threshold |
| a `[fem]`-gated step fails at import | the optional extra did not install | fix the install, not the gate |
| a coverage gate reports `0.00%` with `CoverageWarning: No data was collected` | `omit` collision, or a `--cov=<file>.py` spec coverage 7.x drops | see `add-coverage-gate`; the gate was measuring nothing |
| a gate passes but a module inside it is near-0% | the step's *test selection* excludes that module's tests | add the test file to the step |
| a torchvision/HF download 403s | sandbox proxy, environmental | not a code defect; say so |

Anything not in this table is this PR's to root-cause. "Flake" is not a root cause.

## Hard rules

- **Never** widen a threshold, timeout, or marker filter to get green. That is how a
  `0.5` speedup threshold came to pass on a 2x *slowdown*.
- **Never** skip, disable, or quarantine a test; never push an empty commit or
  close/reopen the PR to kick CI.
- One re-run maximum, and only to confirm a failure that is not this PR's.
- Before pushing: run the repo's own fast checks, reproduce the original failure,
  then show the same check passing. One validated push beats three speculative ones.
- Verify every review finding against the code **before** fixing it — and equally,
  do not dismiss one without checking. On this repo's recent PRs every external
  finding was real.

## Closing the loop

A PR is done when CI is green on the current head, there is no merge conflict, and
(where the repo runs it) the Claude Approvals check passes. Until then keep a
check-in scheduled (`send_later`, ~1 h) and re-arm it silently when nothing changed.
Stop on merged, closed, or a request to stop — then `unsubscribe_pr_activity`.
