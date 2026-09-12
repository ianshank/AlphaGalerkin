# Tasks: `focus-secrets-merge-gate`

Critical path: 1.1 → 1.2 → 2.1 (the guard must exist before the charter row
can cite it) → 3.x docs → 4.1 verify.

## 1 — Workflow

- [x] 1.1 Add `focus` and `secrets` to `ci-success.needs`
- [x] 1.2 Add the exit-1 blocks: two-clause event-aware `focus` gate, bare
      `secrets` gate; extend the echo table
- [x] 1.3 Rewrite the stale "NOT in `ci-success`" comments on both jobs, the
      `transfer-baseline` comment, and the B38 comment inside `ci-success`

## 2 — Guard

- [x] 2.1 `tests/support/workflows.py::hard_gate_conditions` (shared helper;
      `hard_gate_jobs` re-expressed over it)
- [x] 2.2 `tests/docs/test_ci_success_hard_gates.py`: promoted jobs in `needs`
      and hard; `focus` condition shape; label spelled identically in both
      places; every job hard or disclosed; disclosed exceptions self-expire
- [x] 2.3 Plant and revert the four mutations named in the module docstring

## 3 — Prose

- [x] 3.1 `config/focus.yaml` header: the check runs in its own `focus` job,
      a hard gate (it said "wired into CI's `lint` job")
- [x] 3.2 `docs/FOCUS.md`: the promotion has happened; skip semantics stated
- [x] 3.3 Charter frozen-tracks deviation row amended (delta in
      `specs/project-charter/spec.md` here; live row applied)
- [x] 3.4 `CHANGELOG.md` `[Unreleased]` entry; CLAUDE.md Regression Surface row

## 4 — Verify

- [x] 4.1 `pytest tests/docs/ tests/scripts/test_check_focus.py -q` and
      `python scripts/check_doc_links.py`

## Deferred (out of this change)

- `transfer-baseline-regression` hard-or-deviation — R-07, after three
  `workflow_dispatch` runs
- `.gitleaks.toml` allowlist narrowing — separate change (CLAUDE.md Next Steps)
- `config/focus.yaml` track re-scope — D9: nothing this cycle
- R-12's `typecheck` job joins `PROMOTED_JOBS`' expected set when it lands
