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
python -m coverage run --branch \
  --include="*/src/<pkg>/*.py" \
  -m pytest tests/<pkg>/ -m "not gpu_required" -q -p no:cov
python -m coverage report --include="*/src/<pkg>/*.py"
```

**Do not set `COVERAGE_CORE`.** This step used to require `COVERAGE_CORE=pytrace`; the pin was
retired repo-wide on 2026-09-02 after both claims justifying it (a C-tracer crash on `import
torch`, and silent under-measurement) were re-verified and neither reproduced. The default C
tracer measures identically and runs ~3× faster —
`tests/claude/test_harness_validation.py::test_no_coverage_core_tracer_pin` asserts its absence.

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

## Step 3.5 — If the package is in the global coverage `omit`, override it *for this step only*

Check first, because this is the failure mode most likely to bite:

```bash
python - <<'PY'
import tomllib, pathlib
print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["tool"]["coverage"]["run"]["omit"])
PY
```

If any entry is an **ancestor or descendant** of your target, a bare `--cov=<target>` reports
`0.00%` with `CoverageWarning: No data was collected` — and `--cov-fail-under` can then never
fail. That is a gate that measures nothing while looking green; it has shipped three times here
(`video_compression`, `demos`, `skfem_tri.py`). `--include` alone does **not** rescue it: the
native `coverage run` reads pyproject's `omit` too.

The fix is an inline `.coveragerc` generated in the step, overriding the omit without touching
pyproject or the global `--cov=src` gate's number (used by 5 gates today: `backend`, `demos`,
`video_compression`, `research/substrates`, `research/fem_baseline.py`):

```yaml
        run: |
          cat > .coveragerc.<pkg> <<'COVEOF'
          [run]
          branch = true

          [report]
          show_missing = true
          COVEOF
          pytest tests/<pkg>/ --cov=src/<pkg> --cov-config=.coveragerc.<pkg> \
            --cov-branch --cov-fail-under=<N> -q --no-header
```

For a **native-runner** gate use `--rcfile=` on both `coverage run` and `coverage report` instead
of `--cov-config=`. `.gitignore`'s `.coveragerc.*` glob already covers the generated file.

`tests/docs/test_coverage_gate_integrity.py` enforces both directions of this and will fail your
PR: an omitted target with no overriding gate needs an entry in `_OMIT_WITHOUT_A_CI_GATE` with a
reason, and `_OMITTED_BUT_GATED_ELSEWHERE` entries must name a step that *both* selects the module
**and** passes `--cov-config=`/`--rcfile=`. An ancestor `--cov` is explicitly not accepted as
proof — it inherits the same omit.

## Step 4 — Wire the `ci.yml` step

Add to the **`coverage-gates`** job (split out of `coverage` on 2026-09-02 when the combined job
blew its 45-minute cap), mirroring its neighbours exactly:

```yaml
      - name: Per-module coverage gate (<pkg>)
        if: always() && hashFiles('tests/<pkg>/') != ''
        run: |
          pytest tests/<pkg>/ --cov=src/<pkg> --cov-branch --cov-fail-under=<N> -q --no-header
```

- `if: always() && hashFiles(...)` so the step still runs after an earlier gate fails (you want
  every gate's number in one CI run, not one per push) and skips cleanly if the path is absent.
- Verify the `hashFiles` path exists — a typo makes the step skip forever and look green.
- Do **not** declare `COVERAGE_CORE` anywhere. This bullet used to read "it is already job-level";
  no job sets it as of 2026-09-02 (see Step 1).

## Step 5 — Mirror it into the docs, in the right order

The charter guard (`tests/docs/test_charter_alignment.py`) enforces **charter ⊆ CI**, one
directionally: a documented gate whose CI step is missing fails; a CI gate that no doc mentions is
fine. So **land the `ci.yml` step first**, then document it — never the reverse.

1. `CLAUDE.md` Regression Surface: add/extend the row with the **same command bytes** as `ci.yml`.
2. The charter gates register (`openspec/specs/project-charter/spec.md`) — only for gates expressed
   as `--cov=<target>` with `--cov-fail-under=<N>`. The guard matches per CI *step* by that literal
   pair, so a native-runner gate (no `--cov=` at all) **cannot** be charter-registered; leave it out
   rather than inventing a row the matcher can't verify. Edit the charter file directly for this —
   it's the documented carve-out from the `openspec-change` skill's usual proposal/design/tasks
   process, since this is a mechanical, guard-verified row under an unchanged Requirement, not a
   policy decision. If your case involves a genuine policy call instead (e.g. the threshold needed
   a real tradeoff, not `floor(measured)-2`), use `openspec-change` instead of this step.

## Step 6 — Verify

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
pytest tests/docs/test_charter_alignment.py -q
# and run the gate command itself, verbatim, confirming it exits 0
```

Report the measured percentage and the chosen threshold. Never claim a gate passes without
running the exact command you wired.
