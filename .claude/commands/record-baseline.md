---
description: Record a PoC run as a baseline and regression-gate a later run against it.
argument-hint: <run-id> [tolerance-pct]
---

Record a completed PoC run as a baseline, then diff a later run against it (exits non-zero on
regression — CI-usable).

```bash
python -m src.poc.cli record-baseline --run-id "$ARGUMENTS" --out /tmp/base.json --tolerance-pct 10
python -m src.poc.cli diff --baseline /tmp/base.json --run-id "$ARGUMENTS"   # 0 = ok, 1 = regression
```

The baseline harness is direction-aware (per-metric higher/lower-better + tolerance), so one path
gates residual, solved_fraction, and latency. Report the diff outcome.
