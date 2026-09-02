---
name: pr-preflight
description: Run the exact lint / format / type / regression / coverage invocations CI runs, in CI's order and with CI's flag set, before opening or updating a PR. Use at the end of any change to src/, tests/, dashboard/, scripts/ or config/ — the flags are load-bearing and the shorthand forms in prose docs are not the CI forms.
---

# pr-preflight — run what CI runs, not what the docs abbreviate

**Derive every command from `.github/workflows/ci.yml`, never from prose.** Convenience forms in
documentation have historically diverged from CI in ways that flip the result: `mypy src/ --strict`
reports errors on a clean tree while CI's `--strict --ignore-missing-imports` is clean, and a
`ruff check src/` that passes says nothing about `dashboard/`, `scripts/`, `config/`,
`conftest.py`, or `deploy_space.py`, which CI also lints.

## Step 0 — Re-derive the commands (do this, don't trust this file's copies)

```bash
grep -nE "ruff (check|format)|mypy |cov-fail-under=|coverage report" .github/workflows/ci.yml
```

The blocks below are correct as of the last edit to this skill; the grep is correct always.

## Step 1 — Lint + format (CI job: `lint`, hard gate)

```bash
ruff check src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py
ruff format --check src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py
```

The path list must match CI's exactly. `hf_space/`, `notebooks/` and `claude-code-platform/` are
excluded on both sides (CI by omission, pre-commit by per-hook `exclude:`), so every tracked
`*.py` is linted by exactly one of the two.

## Step 2 — Types (CI job: `lint`, currently `continue-on-error`)

```bash
mypy src/ --strict --ignore-missing-imports
```

`--ignore-missing-imports` is **load-bearing**: without it you get import-not-found noise for
optional deps that are not installed in the lint env (PicoGK, openai, skfem, pyamg, pettingzoo,
gymnasium, pandas). CI does not block on mypy today, so **this is local discipline** — a
regression here will not be caught for you.

## Step 3 — Regression surfaces for the paths you changed

Use the `regression-surface` skill: it maps changed paths to the exact command blocks in
`CLAUDE.md`'s Regression Surface table. Run those blocks, not a suite you guessed at.

## Step 4 — Coverage gates for the packages you changed

Use the `coverage-gate` skill (and `add-coverage-gate` if the package has no gate yet). Do **not**
set `COVERAGE_CORE`: this step used to require `pytrace`, but the pin was retired repo-wide on
2026-09-02 after the C-tracer crash it guarded against failed to reproduce, and no CI job sets it
now.

## Step 5 — One combined invocation (the step people skip)

Per-file runs cannot see the failure mode that has bitten this repo most: **global-singleton state
crossed with collection order** (registry singletons, `sys.modules` purges, structlog config). Run
the affected suites together, in one pytest process, mirroring CI's ignore/deselect flags:

```bash
pytest <all affected test dirs> \
  -m "not slow and not e2e and not gpu_required" -q \
  <the --ignore / --deselect flags from ci.yml's test-fast step, verbatim>
```

Mirroring CI's flags matters in both directions: omitting them can red the run on a
pre-existing, already-excluded failure (a red herring), while adding flags CI does not have hides
a real regression.

## Step 6 — Capture exit codes honestly

Piping pytest into `tail`/`grep` loses the exit status, and a trailing library log line can bury
the summary. Either check `${PIPESTATUS[0]}` or redirect to a file and grep it afterwards:

```bash
pytest ... > /tmp/out.txt 2>&1; echo "exit=$?"; grep -E "passed|failed|error" /tmp/out.txt | tail -3
```

Note: the full suite in one process can exhaust memory in constrained environments (SIGKILL,
exit 137) — that is an environment limit, not a test failure. Chunk by directory and say so.

## Step 7 — Report

State what you ran, what passed, and what you did not run. Never report a suite as green that you
did not execute, and never let a truncated pipe stand in for an exit code.

## Environment

```bash
pip install -e '.[dev,fem]'
```

The `fem` extra matters: without `scikit-fem`, `tests/research/test_fem_baseline.py` skips at
import and reports green while testing nothing. That suite went unexecuted in this environment
for its entire existence. CI's `test-extras` job installs it; a local preflight that does not is
narrower than CI.

Note CI also runs a **Python 3.10** job (`requires-python = ">=3.10"`). Stdlib newer than that
floor — `tomllib`, `datetime.UTC` — is a *collection* error, which takes the whole run down
rather than failing one test. `tests/docs/test_python_floor_compatibility.py` catches it locally.
