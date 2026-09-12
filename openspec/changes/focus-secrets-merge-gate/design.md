# Design: `focus-secrets-merge-gate`

Only the decisions a reviewer could reasonably push back on.

## 1. Event-aware skip handling for `focus`, not a bare `!= success`

`focus` runs under `if: github.event_name == 'pull_request' &&
!contains(labels, 'focus-override')`, so on a plain push it is `skipped`, and
`ci-success` (`!cancelled()`) still runs. A bare `!= "success"` would fail
every push run. The tempting alternative, `== "failure"`, would let a
pull request whose `focus` job was skipped by accident (a workflow edit that
breaks the `if:`) merge silently. Two clauses are needed:

```bash
if [[ "${{ needs.focus.result }}" != "success" && "${{ needs.focus.result }}" != "skipped" ]]; then exit 1; fi
if [[ "${{ needs.focus.result }}" == "skipped" && "${{ github.event_name }}" == "pull_request" \
      && "${{ contains(github.event.pull_request.labels.*.name, 'focus-override') }}" != "true" ]]; then exit 1; fi
```

The guard asserts the *shape* (`skipped`, `github.event_name` and the label
all appear in an exit-1 condition naming `needs.focus.result`), because a
membership check alone would pass the bare form.

## 2. Inline `[[ ]]` blocks, not a precomputed shell variable

`tests/support/workflows.py::hard_gate_jobs` recognises `if [[ … needs.X.result
… ]]; then … exit 1 … fi` blocks and nothing else. A form such as
`FOCUS_OK=$(...)` followed by `if [[ "$FOCUS_OK" != "true" ]]` is invisible to
it, so the promotion would be real and the guard blind. The script was
verified against the parser before landing (the plan's §5 card records this).

## 3. `secrets` gets a bare check

`secrets` and `ci-success` share `if: github.event_name != 'schedule'`, so
`secrets` is never `skipped` when `ci-success` runs. Adding skip handling
"for symmetry" would accept a skip that can only arise from a future edit to
one of the two `if:`s, which is precisely the case that should fail.

## 4. Close the class with a disclosed-exception list

The plan's assertion (4) was "every job with a PR-firing `if:` is a hard
gate". Deciding "PR-firing" from the `if:` text needs an evaluator for
GitHub's expression language (`test-slow`'s condition is eleven lines of
nested `||`). The guard instead asserts the stronger, simpler property: every
job in `ci.yml` other than `ci-success` is either in `needs ∩ hard` or in
`DISCLOSED_NOT_HARD` with a reason. Each entry is checked to still be needed
(the job exists and is not a hard gate), so the list expires itself in both
directions. Two entries today: `test-slow` (PR-invisible by design) and
`transfer-baseline-regression` (disclosed soft, R-07 decides).

## 5. A shared parser helper rather than a private-regex import

The shape assertion needs the condition text, which `hard_gate_jobs` discards.
`tests/support/workflows.py` gains `hard_gate_conditions(script) -> list[str]`
and `hard_gate_jobs` is re-expressed over it, so both guards read one regex.
This is the "second workflow parser" lesson from CLAUDE.md Next Steps applied
before a second copy could form.

## 6. Charter row amended in place, delta recorded here

The frozen-tracks deviation row named the promotion as future work. Following
`openspec/changes/hygiene-hardening-cycle/` (which amended the same row), the
live row is edited in the same change that ships the delta, and the delta
carries the applied text. No Requirement text outside the deviations table
changes.
