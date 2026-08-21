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
gymnasium, pandas).

## Step 3 — Regression surfaces for the paths you changed

Use the `regression-surface` skill: it maps changed paths to the exact command blocks in
`CLAUDE.md`'s Regression Surface table. Run those blocks, not a suite you guessed at.

## Step 4 — Coverage gates for the packages you changed

Use the `coverage-gate` skill (and `add-coverage-gate` if the package has no gate yet). Set
`COVERAGE_CORE=pytrace` — the installed torch wheel crashes coverage's C tracer, and the CI
`coverage` job sets it at job level for exactly this reason.

## Step 5 — One combined invocation (the step people skip)

Run the affected suites together, in one pytest process, mirroring CI's ignore/deselect flags:

```bash
COVERAGE_CORE=pytrace pytest <all affected test dirs> \
  -m "not slow and not e2e and not gpu_required" -q
```
