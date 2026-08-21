---
name: add-coverage-gate
description: Add or ratchet a per-module coverage gate end-to-end — measure the actual branch %, pick the threshold, wire the ci.yml step in the correct --cov form, and mirror it into CLAUDE.md's Regression Surface and the charter gates register. Use when a new package lands, when raising a threshold, or when docs/CODE_HYGIENE_AUDIT.md §7.4 lists a package as ungated.
---

# add-coverage-gate — wire a per-module gate that actually enforces

Adding a gate is five coupled edits. Doing four of them produces a gate that passes while
measuring nothing (this has happened: see `docs/CODE_HYGIENE_AUDIT.md` §7.1) or a red
`tests/docs/test_charter_alignment.py`.

Related: `coverage-gate` runs an *existing* gate. This skill *creates* one.

## Step 1 — Measure first, with the exact test selection the gate will use

Never pick a threshold before measuring; never measure with a different test selection than the
gate will run (a wider selection inflates the number and the gate goes red in CI).

```bash
COVERAGE_CORE=pytrace python -m coverage run --branch \
  --include="*/src/<pkg>/*.py" \
  -m pytest tests/<pkg>/ -m "not gpu_required" -q -p no:cov
COVERAGE_CORE=pytrace python -m coverage report --include="*/src/<pkg>/*.py"
```

`COVERAGE_CORE=pytrace` is required — the installed torch wheel crashes coverage's C tracer, and
the CI `coverage` job sets it at job level for the same reason.

## Step 2 — Pick the threshold: `floor(measured) - 2`

- **Never** the aspirational number. A gate you cannot pass today is not a gate, it is a red build.
- **Never** above 85 on a first landing, even if measured higher — leave headroom for legitimate
  churn. 85 is the project ceiling for a new gate; the global gate is also 85.
- Ratcheting up later is cheap and is tracked per owner-decision #4 in
  `docs/CODE_HYGIENE_AUDIT.md` §7.6 (+2/quarter toward 85).
- If measured coverage is far below (< 75), prefer **triage before gating** — landing a gate at 59
  institutionalises a bad number. Say so instead of gating.

## Step 3 — Choose the `--cov` form (this is where gates silently break)

| Gating | Form |
|---|---|
| A whole package | pytest-cov, directory spec: `pytest tests/<pkg>/ --cov=src/<pkg> --cov-branch --cov-fail-under=<N> -q --no-header` |
| Individual files | **Native runner only**: `python -m coverage run --branch --include="<globs>" -m pytest <tests> -q -p no:cov` then `python -m coverage report --include="<globs>" --fail-under=<N>` |

**Never** put a file-path spec (`--cov=src/x/y.py`) or a dotted spec (`--cov=src.x.y`) in a
pytest-cov command. The first is silently dropped under coverage 7.x; the second collides with the
torch C extension. See the decision table in the `coverage-gate` skill.

## Step 4 — Wire the `ci.yml` step

Add to the `coverage` job, mirroring its neighbours exactly:

```yaml
      - name: Per-module coverage gate (<pkg>)
        if: always() && hashFiles('tests/<pkg>/') != ''
        run: |
          pytest tests/<pkg>/ --cov=src/<pkg> --cov-branch --cov-fail-under=<N> -q --no-header
```

## Step 5 — Mirror to documentation + charter alignment

1. Add a row to `CLAUDE.md`'s *Regression Surface* table under "Per-module coverage gates".
2. Run `pytest tests/docs/test_charter_alignment.py` to confirm alignment.
