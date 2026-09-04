---
name: harden-a-guard
description: Prove a new guard test can actually fail — plant the defect it claims to catch, confirm a NAMED test goes red, confirm the mutation applied, and record the kill. Use whenever adding a test that reads config, docs or CI as data, a vacuity check, or any assertion whose job is to prevent a class of defect rather than to test a function.
---

# harden-a-guard — a check that cannot fail is not a check

This repo has shipped the same defect at least seven times: a guard, a gate, or a
tier that looked enforced and enforced nothing. `video_compression`, `demos`,
`skfem_tri.py`, `src/backend`, `tests/e2e/` — **every one was found by a person
reading a config file, none by a check.** The convention that prevents it is
stated as a *value* in `.claude/agents/reviewer.md` and `.claude/agents/sqe.md`
and is enforced by nothing. This skill is the procedure.

Use it for any test whose subject is *a class of defect* rather than a function:
guards over `ci.yml` / `Makefile` / `pyproject.toml` / Markdown, coverage-gate
integrity checks, marker-vocabulary checks, "this abstraction has a caller"
checks, and any assertion added in response to a review finding.

Related: `add-coverage-gate` (creating a gate), `coverage-gate` (running one),
`claims-ledger` (retracting a *numeric* claim).

## Step 0 — Name the defect class in one sentence

Write it in the test's docstring before writing the assertion. If you cannot
name it, you are testing an implementation detail, not guarding a class.

> "A job listed in `ci-success.needs` but with no `exit 1` block blocks nothing."

## Step 1 — Vacuity first: assert your inputs are non-empty

A parametrised guard over an empty list passes forever, and so does a scan whose
roots match nothing. Add the emptiness assertion **before** the real one.

```python
def test_the_e2e_directory_is_not_empty() -> None:
    """Without this, every parametrised assertion below iterates nothing."""
    assert len(list(E2E_DIR.glob("test_*.py"))) >= MIN_E2E_TEST_FILES
```

Real instance: an STL guard asserted `len(data) >= 84` and `n_triangles == 0`,
both of which hold for a **zero-triangle** file — the exact output the export was
meant to make impossible.

## Step 2 — Plant the literal historical defect

Not a synthetic near-miss. Where the defect is in git, restore that exact line:

```bash
cp target.py /tmp/target.bak
python - <<'PY'
import pathlib
p = pathlib.Path("target.py"); s = p.read_text()
old = "the fixed line"
assert old in s, "anchor not found -- the mutation would silently no-op"
p.write_text(s.replace(old, "the pre-fix line", 1))
print("MUTATION APPLIED")
PY
```

The `assert old in s` is not optional. A mutation that fails to apply produces a
green run that looks like a kill.

## Step 3 — Run the narrowest selection and record the NAMED test

"The suite went red" is not a kill — an unrelated failure looks identical.

```bash
pytest path/to/test_guard.py -q -p no:randomly 2>&1 | grep -E "^FAILED|passed|failed"
cp /tmp/target.bak target.py
pytest path/to/test_guard.py -q -p no:randomly | tail -2   # green again
```

Record the test *name*, not the count.

## Step 4 — Check the killing test's markers

A killer carrying `gpu_required` or `fem_required` **skips on CI**, so the kill is
fictional on every real run. Every workflow here is `ubuntu-latest`. Verify with
the filter CI uses:

```bash
pytest path/to/test_guard.py -m "not gpu_required and not fem_required" -q
```

Real instance: a mutation was recorded as killed by a `fem_required` test, which
had never run in CI.

## Step 5 — Plant the adjacent weaker defect

The dangerous mutation is not the obvious one; it is the thing that satisfies
your assertion *literally* while defeating its purpose. Ask: what is the cheapest
edit that keeps this test green and removes the protection?

Real instances from this repo, all of which passed the first draft of their guard:
- an `echo` of `needs.<job>.result` where an `exit 1` was meant;
- a *sibling* CI step that satisfied the clause alone, so deleting the whole job
  under test left the guard green;
- Make's `-` prefix removed but rewritten as `A; \ B`, where `sh` returns only
  `B`'s status;
- a `--cov=<path>.py` spec, silently dropped by coverage 7.x;
- a guard comparing a production default against itself across a process
  boundary — retuning the default moves both sides together.

## Step 6 — Prefer a semantic mutation over a syntactic assertion

If your guard greps for a *character* (`-`, `@`, a flag name), a rewrite that
preserves the defect and changes the spelling defeats it. Where the behaviour can
be driven cheaply, drive it:

```bash
make test-e2e PYTEST=/tmp/stub-that-exits-3   # ~50 ms, no pytest at all
```

## Step 7 — Record the kill where it can go stale visibly

In the test docstring **and** the CLAUDE.md Regression Surface row, name each
mutation and the test that killed it — `N/N mutation-killed`, with N being the
number of *planted defects*, not the number of tests. State both if you cite a
test count too; conflating them is how "9/9 mutation-killed" came to be read as a
test total.

Prefer a self-maintaining claim over a hand-counted one: a count in prose drifts
(a Regression Surface row said "Six clauses" after eight existed). Where the
number is derivable, assert it.

## Step 8 — Make exemptions self-expiring

Any allowlist (`FORWARD_REFERENCES`, `_STAGED_FOR_UPCOMING_TASK`,
`_OMITTED_BUT_GATED_ELSEWHERE`) must fail in **both** directions: when the
exemption is no longer needed, and when the thing it exempts no longer exists. A
stale exemption silently shrinks a guard's scope, which is the defect one layer up.

## The completion bar

- [ ] Defect class named in one sentence in the docstring
- [ ] Vacuity assertion on every scanned input
- [ ] Literal historical defect planted, with an anchor assertion
- [ ] A **named** test failed, and passes again after restore
- [ ] The killing test is not `gpu_required` / `fem_required`
- [ ] The adjacent weaker defect planted and killed too
- [ ] Kill recorded in the docstring and the CLAUDE.md row, mutations != tests
- [ ] Exemptions expire in both directions
