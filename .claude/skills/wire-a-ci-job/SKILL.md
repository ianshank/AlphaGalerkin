---
name: wire-a-ci-job
description: Bring a new CI job or a whole test tier into enforcement end-to-end — the ci.yml job, the ci-success exit 1 gate, the Makefile mirror with correct status propagation, the pre-pr chain and its guard, and the CLAUDE.md Regression Surface row copied verbatim from CI. Use when adding a CI job, or when a tests/ directory runs nowhere.
---

# wire-a-ci-job — six edits, and four of them are the ones people miss

`add-coverage-gate` codifies this motion for a **coverage gate**. There was no
equivalent for a **test job**, and the branch that added `test-e2e` performed the
motion by hand and got two of the six wrong: the `pre-pr` chain was unguarded, and
the Regression Surface row documented a command CI does not run.

The failure mode is always the same and always looks fine in a diff: the job
exists, the tests pass, and nothing about it can fail the build.

## Step 1 — The job

```yaml
  test-<tier>:
    name: <Human Name>
    runs-on: ubuntu-latest
    timeout-minutes: <measured, not guessed>
    needs: test-fast
```

Budgets: measure the tier, then set `timeout-minutes` to roughly 3x. If the tier
runs subprocesses with their own budgets, those were probably measured on an idle
box — a shared runner is not idle. Scale them through an env var rather than
widening each one (`E2E_TIMEOUT_SCALE`), so a genuinely hung child is still caught.

## Step 2 — The `-m` expression, checked against registered markers

`--strict-markers` rejects an unknown marker on a **test** but **not** an unknown
identifier inside a `-m` string. Verified: `-m "not gpu_requried"` selects
everything and exits 0. Every identifier must exist in `pyproject.toml`'s
`markers`; `tests/docs/test_marker_vocabulary.py` enforces this — run it.

## Step 3 — `ci-success` needs it **and** gates on it

Two separate edits. Adding to `needs:` alone blocks nothing:

```yaml
    needs: [..., test-<tier>]
```
```bash
          if [[ "${{ needs.test-<tier>.result }}" != "success" ]]; then
            echo "::error::test-<tier> failed"; exit 1
          fi
```

An `echo` without the `exit 1` reads identically in review and gates nothing.
`tests/support/workflows.py::hard_gate_jobs` returns only jobs inside an `if`
whose body reaches `exit 1` — use it in the guard, not a substring search.

## Step 4 — The Makefile mirror, with status that propagates

The target must run the **same selection** as CI, not a glob subset. (`make
test-e2e` once ran a 3-test glob of an 81-test tier, so `make pre-pr` certified
every PR against three.)

If the job runs more than one invocation, accumulate status explicitly. Make's
`-` prefix discards a line's exit code, and a target's status is its **last**
line's — so both of these exit 0 with the first half red:

```make
	-$(PYTEST) ... half-one          # `-` discards it
	$(PYTEST) ... half-two
```
```make
	$(PYTEST) ... half-one; $(PYTEST) ... half-two   # `sh` returns only the last
```

Correct:

```make
test-<tier>:
	status=0; \
	$(PYTEST) ... half-one || status=$$?; \
	$(PYTEST) ... half-two || status=$$?; \
	exit $$status
```

Verify it, do not reason about it — this takes 50 ms and needs no pytest:

```bash
printf '#!/bin/sh\nexit 3\n' > /tmp/stub && chmod +x /tmp/stub
make test-<tier> PYTEST=/tmp/stub; echo "exit=$?"   # must be non-zero
```

## Step 5 — Chain into `pre-pr` **and** add it to the guard

`Makefile`'s `pre-pr:` line, **and**
`tests/claude/test_harness_validation.py::test_make_pre_pr_chains_the_local_equivalents`'s
parametrize list. Doing only the first means deleting the chain later fails
nothing — which is the "pre-pr is narrower than CI" defect the target exists to
prevent.

## Step 6 — The Regression Surface row, copied not paraphrased

The command cell must be the **literal string** `ci.yml` runs, including every
`-m`, `-k`, and any env var the step sets. Paraphrasing is how a row came to
prescribe an invocation measured to exhaust the runner's memory.

If the job runs several invocations, document all of them. If an env var changes
behaviour (`ALPHAGALERKIN_REQUIRE_EXTRAS=1` turns a silent skip into an error),
it is part of the command.

## Step 7 — Guard the wiring, then mutate it

Follow `harden-a-guard`. The mutations that matter here, all of which have
survived a first-draft guard in this repo:

| Mutation | What it defeats |
|---|---|
| replace `exit 1` with `echo` | a `needs`-only check |
| delete the job, keep a sibling step with the same `-m` | a clause satisfied by another step |
| delete the target from `pre-pr` | an unguarded chain |
| re-add Make's `-` prefix, or rewrite as `A; B` | a syntactic status check |
| paraphrase the Regression Surface command | a drift guard that only greps for a keyword |

## The completion bar

- [ ] Job added, `timeout-minutes` measured
- [ ] Every `-m` identifier is a registered marker (marker-vocabulary suite green)
- [ ] In `ci-success.needs` **and** in an `exit 1` block
- [ ] Makefile target runs CI's selection; status verified with a stub `PYTEST`
- [ ] Chained into `pre-pr` **and** added to the pre-pr guard's parametrize list
- [ ] Regression Surface row carries the literal CI command(s) and env vars
- [ ] Wiring guard added and mutation-killed per `harden-a-guard`
