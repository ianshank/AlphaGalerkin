---
name: coverage-gate
description: Run the per-module coverage gate for an AlphaGalerkin package the way CI enforces it. Use before opening a PR to confirm a changed package still meets its --cov-fail-under threshold (branch coverage), matching the gates in .github/workflows/ci.yml and the CLAUDE.md Regression Surface.
---

# coverage-gate — enforce a package's coverage threshold

AlphaGalerkin gates coverage globally (85%) and per-module. This skill runs the correct gate for
the package you changed.

## Per-module thresholds — single source of truth is `.github/workflows/ci.yml`

There is **no threshold table here** on purpose: two copies of the same numbers drift (that
duplication is the mechanism by which `src/pde/game.py`'s docstring became a lie). Read the gate
straight from CI:

```bash
# List every per-module coverage gate and its threshold, from ci.yml.
# BOTH invocation forms must be matched — see below.
grep -nE "cov=src/|cov-fail-under=|--include=|--fail-under=" .github/workflows/ci.yml
```

**There are three `--cov` spec forms and only two of them work. Picking the wrong one produces a
gate that passes while measuring nothing — this is not hypothetical, it silently degraded the
llm_prior gate until 2026-08 (see `docs/CODE_HYGIENE_AUDIT.md` §7.1).**

| Spec form | Example | Verdict |
|---|---|---|
| **Directory** | `--cov=src/pde` | ✅ Works with pytest-cov. Use for whole-package gates. |
| **File path** | `--cov=src/poc/scenarios/x.py` | ❌ **BANNED.** Under coverage 7.x / pytest-cov 7.x these are silently dropped with only a `CovReportWarning`. If any directory spec is present in the same command, the gate passes on that directory alone and the files enforce nothing; if it is the only spec, coverage reports `0` and the gate fails for the wrong reason. |
| **Dotted module** | `--cov=src.poc.scenarios.x` | ❌ **BANNED.** Makes coverage import the target during tracer setup, colliding with the torch C extension (`SystemError: bad call flags`) under every tracer, pure-Python included — so no `COVERAGE_CORE` value rescues it. |

**To gate individual files, use the native runner** (`python -m coverage run --branch
--include=<path globs>` + `python -m coverage report --include=<same globs> --fail-under=<N>`).
Path globs are matched against collected data, not imported for discovery, so they dodge both
failure modes.

Do not hardcode which gates use which form — that list has gone stale twice. Derive it:

```bash
grep -c "coverage report" .github/workflows/ci.yml   # native-runner gates (minus 1 for the artifact-upload step name)
grep -cE "cov-fail-under=" .github/workflows/ci.yml  # pytest-cov gates
```

Scenario / integration packages (`src/poc/scenarios/*`, `src/integrations/*`, `src/agents/*`) are
gated at 85 branch. A **new** package's gate is added to `ci.yml` in the same PR as the package.

Note that branch coverage is already global via `pyproject.toml [tool.coverage.run] branch = true`,
so passing `--cov-branch` explicitly (as the template below does, mirroring CI) is belt-and-braces
rather than the thing that turns branch measurement on.

## Steps

1. Pick the package and read its threshold from `ci.yml` (command above). For a new scenario or
   agent, use 85 (branch coverage).
2. Run the gate, mirroring CI:
   ```bash
   pytest tests/<pkg>/ -m "not gpu_required" \
     --cov=src/<pkg> --cov-branch --cov-fail-under=<N>
   ```
3. If below threshold, add tests for the uncovered branches (gating/error paths are the usual
   gaps — mirror the synthetic-harness pattern in `tests/poc/test_scaling_law_scenario.py`).
4. Report the actual percentage; never claim a gate passed without running it.

## Environment note — do NOT set `COVERAGE_CORE` (retired 2026-09-02)

This section used to instruct setting `COVERAGE_CORE=pytrace`, on the claim that the installed
PyTorch wheel crashes coverage's default **C tracer** on `import torch` (`SystemError: ... bad
call flags`) so `pytest --cov` collects no data, and that CI's `coverage` job pinned it at job
level for the same reason. Both halves were re-verified in 2026-09 and **neither reproduces**: a
minimal `coverage run --branch` over a script importing `torch._C` exits 0 under both cores, the
`src/training` gate measures 88.25% under both with a byte-identical per-file breakdown, and
removing the pin from CI cut that job's pytest execution 1967.30s → 604.32s (3.26×) with
byte-identical coverage totals. No CI job sets it now, and
`tests/claude/test_harness_validation.py::test_no_coverage_core_tracer_pin` asserts its absence.

```bash
python -m coverage run --branch \
  --include="*/<pkg>/*.py" -m pytest tests/<pkg>/ -m "not gpu_required" -q -p no:cov
python -m coverage report \
  --include="*/<pkg>/*.py" --fail-under=<N>
```

Note the second command carries `--fail-under=<N>` and repeats `--include`. Both are required:
without `--fail-under` you get a report, not a gate; without the repeated `--include` the report
covers everything coverage happened to record. `coverage run` erases prior data by default, so
sequential gate steps in one job cannot contaminate each other.
