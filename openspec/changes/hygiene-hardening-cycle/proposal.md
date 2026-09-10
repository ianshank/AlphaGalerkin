# Proposal: `hygiene-hardening-cycle`

## Why

Default-branch CI is already green
([run 34309501940](https://github.com/ianshank/AlphaGalerkin/actions/runs/34309501940)
on `eceb1ec`). This cycle is not a red-build rescue. It consumes the existing
hygiene ledger (`docs/CODE_HYGIENE_AUDIT.md` B1–B40) under two owner
constraints that the live charter still mis-states:

1. **The FOCUS freeze-lift condition has already fired.** The frozen-tracks
   deviation still says the freeze lifts when the refinement experiment has an
   interpretable answer. That answer is committed:
   `results/mcts_classical_amr_arena.csv`, median
   `l2_error_ratio_at_matched_dof` **0.9532** (θ=0.5, policy `max_dof=600`,
   matched DOF 287). `docs/FOCUS.md` already records the lift;
   `config/focus.yaml` still lists `codec` and `interactive-surfaces` so the
   split-attention gate keeps working until a follow-up re-scopes tracks.
2. **B10 packages stay.** `src/prototyping/`, `src/analysis/`,
   `src/curriculum/`, and `src/tournament/` are in-tree, test-held, and not
   production-wired. They are **not** a 2026-07-22-style cut. Deleting them
   would repeat the `video_compression` cut-and-restore. The charter currently
   lists them in the scope register with no keep-reason, which makes the next
   “cut to the core” look like an uncontested cleanup.

Soft CI gates and remaining god files are in-cycle work *after* this
disclosure lands. They do not change a Requirement on their own (mechanical
coverage rows use `add-coverage-gate`; god-file splits are import-compatible
re-exports).

## What Changes

### Accepted Deviation Disclosure

- **Amend** the frozen-tracks row: thesis freeze lifted on the committed arena
  CSV; `codec` / `interactive-surfaces` remain paused until a follow-up edits
  `config/focus.yaml`. Retirement condition stated.
- **Add** a B10 keep-reasons row for the four unused-but-in-scope packages.
  Retirement condition stated. Packages stay in the scope register.

### Scope Integrity

No package is added or removed. `src/device.py` (Wave B1) is a root *module*
like `src/seeding.py`, not a package, and does **not** get a scope-register
row (`tests/docs/test_architecture_map.py` enumerates `src/*/__init__.py`
only).

### Quality Gate Fidelity

Untouched in this change. Flipping mypy’s `continue-on-error` is a later
charter PR after the lint-job torch wheel is pinned. Mechanical
`floor(measured)-2` rows use `add-coverage-gate`.

## What This Change Does NOT Do

- **Does not delete** `prototyping` / `analysis` / `curriculum` / `tournament`.
- **Does not unfreeze** codec or interactive surfaces in `config/focus.yaml`.
- **Does not add** a `src/device.py` scope-register row.
- **Does not flip** mypy, ONNX, backend audit, or `transfer-baseline-regression`
  into hard merge blockers.
- **Does not unify** seed strides (1009 / 7919 / 9973).
- **Does not** rewrite `Trainer.__init__` to call `super()`.
- **Does not** touch frozen tracks substantively (`src/video_compression/`,
  `dashboard/`, `hf_space/`).
- **Does not** gate `src/templates` / `src/math_kernel`, raise `src/backend`
  54, or raise `src/deployment` 25.
- **Does not** land B37 (`eval_harness`) or B7 (args-file extract).

## Impact

- Live charter deviations table and this delta.
- Subsequent hygiene PRs (device promotion, god-file splits, CI hardness)
  may proceed once the disclosure is reviewable.
- `docs/FOCUS.md` already matches the amended freeze story; keep it in step
  with `config/focus.yaml` (`tests/scripts/test_check_focus.py`).

## Risks

| Risk | Mitigation |
| --- | --- |
| Keep-reason read as a cut permission | Row states the packages stay in the register; retirement is “wire into production or a dedicated cut change”, not silent deletion |
| Freeze-lift read as “edit codec now” | `config/focus.yaml` still lists both tracks; empty `frozen_tracks` is rejected |
| Scope-register row for `src/device.py` | Explicitly out of this change; architecture-map guard is package-granular |
