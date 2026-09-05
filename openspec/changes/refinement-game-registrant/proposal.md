# Proposal: `refinement-game-registrant`

## Why

`openspec/changes/element-local-substrate/` delivered the measurement foundation the cycle
needed: a `RefinementSubstrate` Protocol, `TensorGridSubstrate` (defective-by-design control),
`SkfemTriSubstrate` (element-local), shared `dorfler_mark`, and an adequacy gate that passes on
skfem and fails on the tensor-product control (tasks 0–6).

What it did **not** deliver — and what `docs/FOCUS.md` still blocks the freeze-lift on — is a
path by which a non-test process constructs a substrate by key and drives it through a
`RefinementGame` into MCTS. Today:

- `RefinementGameRegistry` has **zero** production registrants (toy only in tests).
- `RefinementSubstrateRegistry` has two registrants and **zero** runtime lookups.
- `RefinementSubstrate.fingerprint` has no reader (`_STAGED_FOR_UPCOMING_TASK`).
- Live PDE games (`MeshRefinementGame`, `BasisSelectionGame`, `LShapeAMRGame`) register in
  `GameRegistry` via `PDEGame`, not `RefinementGame` — a parallel abstraction that never got its
  first real consumer.

Without a registrant, the MCTS-vs-classical arena cannot run on the gated substrate. Without
governance closeout, `specs/lshape_amr_compare.spec.md` stays `Implemented` and readers can still
treat the defective-substrate MCTS-lose CSV as the live competitive baseline.

This change **is Slice E**: the remaining work previously listed as tasks 7.1–8.5 in
`element-local-substrate`. It is split into its own package so it cannot be bundled with the
arena experiment (the failure mode that change's proposal explicitly forbids).

## What Changes

1. **Minimal pure `RefinementGame`** over `RefinementSubstrate` (single-element / top-1 mark →
   refine; map `SubstrateSolveResult` → `RefinementState`).
2. **Registration only via an explicit `register_*` module** — never package `__init__.py`
   (documented SIGSEGV / coverage-tracer class in `src/pde/games/__init__.py`).
3. **First non-test `RefinementSubstrateRegistry` lookup** (config-driven `kind`).
4. **Fingerprint-keyed solve cache** consuming `SubstrateConfig.solve_cache_max_entries`; retire
   the staged `fingerprint` audit exemption.
5. **Governance:** retire empty-`RefinementGameRegistry` deviation; add time-boxed two-path
   deviation; mark `lshape_amr_compare` superseded; CLAUDE / CHANGELOG / run-manifest hygiene.
6. **Doc hygiene:** update `docs/FOCUS.md` / proposal present-tense claims — substrate adequacy
   for classical marking is gated; what remains untested is MCTS-on-element-local, not "substrate
   still wrong."

## What This Change Does NOT Do

- **No MCTS-vs-classical experiment.** Arms, budget matching, statistics, and pre-registration
  live in `mcts-classical-amr-arena` (sibling change).
- **No trained evaluator.**
- **No deletion** of `LShapeAMRGame` / `lshape_amr_compare.py` (golden back-compat).
- **No work on frozen tracks** (`src/video_compression/`, `dashboard/`, `hf_space/`).
- **No multi-field PDE / PETSc / MFEM.**
- **No estimator vectorisation** (separate tolerance'd change after goldens).

## Risks

| Risk | Mitigation |
| --- | --- |
| Impure `apply_action` copies `LShapeAMRGame` mutation | Design law: mesh on immutable state / fingerprint; property + unit tests; peer review gate |
| Registration from `__init__` re-triggers SIGSEGV class | Explicit `register_*` module + import-graph test in the same PR |
| Layering: `src/refinement` imports research/PDE | Game + cache live **outside** `src/refinement/` (domain-free package) |
| "Registered but unused" abstraction | Same PR must include ≥1 non-test lookup by key and a CI/guard that registrants ≥ 1 |
| Calling Slice E "thesis progress" | Adequacy rates stay gate-only; no README headline from this change |

## Impact

- **Depends on:** `element-local-substrate` tasks 0–6 (landed).
- **Unblocks:** `mcts-classical-amr-arena` MCTS arm.
- **Charter:** retires one deviation; adds one time-boxed deviation; supersession pointer.
- **Focus gate:** may touch only core paths (`src/refinement/`, `src/pde/`, `src/research/`,
  `src/mcts/` as needed for adapter wiring) — not frozen tracks.
- **When this lands:** check off `element-local-substrate` tasks 7.1–8.5 and treat that change as
  substrate-complete (archive when both packages agree).

## Relationship to sibling changes

| Change | Role |
| --- | --- |
| `element-local-substrate` | Substrates + adequacy gate (done); Slice E tasks move here |
| **`refinement-game-registrant` (this)** | First real `RefinementGame` + governance |
| `mcts-classical-amr-arena` | Pre-registration, then falsifiable MCTS vs classical experiment |
