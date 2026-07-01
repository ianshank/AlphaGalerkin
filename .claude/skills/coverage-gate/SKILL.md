---
name: coverage-gate
description: Run the per-module coverage gate for an AlphaGalerkin package the way CI enforces it. Use before opening a PR to confirm a changed package still meets its --cov-fail-under threshold (branch coverage), matching the gates in .github/workflows/ci.yml and the CLAUDE.md Regression Surface.
---

# coverage-gate — enforce a package's coverage threshold

AlphaGalerkin gates coverage globally (85%) and per-module. This skill runs the correct gate for
the package you changed.

## Per-module thresholds (from `.github/workflows/ci.yml`)

| Package | `--cov-fail-under` |
|---|---|
| `src/pde/` | 75 |
| `src/physics/` | 75 |
| `src/modeling/` | 85 |
| `src/training/` | 85 |
| `src/research/` | 85 |
| `src/games/` | 80 |
| `src/distributed/` | 60 |
| Global (`src/`) | 85 |

Scenario / integration packages use 85 via the CLAUDE.md Regression-Surface rows
(e.g. `src/poc/scenarios/*`, `src/integrations/lm_studio`, `src/agents/*`).

## Steps

1. Pick the package and its threshold. For a new scenario or agent, use 85 (branch coverage).
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
