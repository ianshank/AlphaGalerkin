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

**There are two gate forms, and a grep for only the first silently misses four gates.**

1. **pytest-cov form** (most gates): a `Per-module coverage gate` step pairing `--cov=src/<pkg>`
   with `--cov-fail-under=<N>`.
2. **Native-runner form** (`python -m coverage run --include=… ` + `python -m coverage report
   --fail-under=<N>`): used by the `scaling_law` + `_centaur_common`, `agents/research_loop` +
   `agents/config`, `transfer_baseline_compare` + `cnn_baseline`, and stochastic-Galerkin/NKE
   gates. These exist because the dotted `--cov=module.path` form collides with the torch C
   extension during source discovery (ci.yml documents this at the first such step). Grepping only
   for `cov=src/` reports these four packages as ungated, which is wrong — they gate at 85.

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

## Environment note — coverage tracer vs. some PyTorch wheels

Certain nightly PyTorch CPU wheels crash on `import torch` while coverage's default **C tracer**
is active (`SystemError: ... bad call flags`), so `pytest --cov` collects no data. When that
happens, run coverage with the **pure-Python tracer** via the native runner (not the pytest-cov
plugin):

```bash
COVERAGE_CORE=pytrace python -m coverage run --branch \
  --include="*/<pkg>/*.py" -m pytest tests/<pkg>/ -m "not gpu_required" -q -p no:cov
COVERAGE_CORE=pytrace python -m coverage report -m
```

CI uses a stable wheel where the C tracer works, so this fallback is only needed locally.
