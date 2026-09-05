# Proposal: `mcts-classical-amr-arena`

## Why

The cycle thesis — **MCTS multi-step look-ahead beats classical greedy marking for error-driven
AMR** — is still **untested on an interpretable substrate**.

What is already true (do not re-litigate):

- On the tensor-product harness, adaptive Dörfler is *worse* than uniform at matched DOF
  (`results/lshape_adaptive_vs_uniform.*`); MCTS vs Dörfler there showed MCTS **losing**
  (`results/lshape_mcts_vs_dorfler.csv`). Those numbers measure a defective substrate / legacy
  harness, not planning depth on element-local meshes.
- `element-local-substrate` closed the classical adequacy gate: on `SkfemTriSubstrate`, adaptive
  beats uniform under pinned θ/DOF windows, and the same gate fails on `TensorGridSubstrate`.
- Adequacy rates are **gate evidence**, not a research win (`specs/refinement_substrate.spec.md`).

What is missing: a pre-registered, shared-substrate comparison of **MCTS vs classical marking**
on `skfem_tri`, with honest artifacts. No `openspec/changes/*arena*` exists today — that absence
is a process defect relative to the charter's evidence standard.

## What Changes

### Phase 0 — Pre-registration only (first commit; no numbers)

Lock, in-repo, before any multi-seed run:

1. Hypothesis + falsifiers (win / null / inconclusive / abort).
2. Pinned config: `kind=skfem_tri`, element order, θ, DOF window/budgets, `n_seeds`, search
   settings, metric = quadrature L2 for headlines.
3. Shared-substrate invariant + `describe()["dof_convention"]` in manifests.
4. Adequacy precondition: gate must pass on that exact `SubstrateConfig` or abort.
5. Metric hierarchy: matched-DOF primary; matched-solves / wall-clock recorded but ungated
   (unless pre-declared otherwise).
6. Artifact contract: CSV columns, `*.run.json` fields (config hash, git SHA, dirty policy —
   no `"dirty": true` / `"config_hash": "unknown"` as proposal-grade).
7. Explicit out-of-scope: no trained evaluator in v1.

### Phase 1 — Classical arms on shared substrate

Dörfler vs uniform (and any other pre-registered classical baselines) via
`src/research/substrates/sweep.py` — may proceed once configs exist; does not require MCTS.

### Phase 2 — MCTS arm (blocked on `refinement-game-registrant`)

Drive the registered `RefinementGame` + `RefinementGameAdapter` under the Phase 0 contract.
Publish results only to committed artifacts that match the pre-registration.

### Phase 3 — Reporting & supersession

README/charter updates only from committed artifacts; quote rates **with θ and DOF window**;
keep legacy tensor-grid MCTS-lose rows labeled non-informative for element-local policy.

## What This Change Does NOT Do

- **Does not implement Slice E** — that is `refinement-game-registrant`.
- **Does not claim a win from the adequacy gate.**
- **Does not unfreeze** codec or interactive surfaces.
- **Does not add** multi-field PDE, PETSc/MFEM, or trained evaluators.
- **Does not delete** the legacy L-shape golden until its retirement condition fires.

## Risks

| Risk | Mitigation |
| --- | --- |
| Opportunistic headlines before pre-reg | Phase 0 is mergeable alone; CI/doc guard rejects new AMR ratios without manifest pointer |
| Impure game / clone bugs | Hard dependency on registrant purity tests |
| Budget mismatch masquerading as policy quality | Pre-register matched-DOF primary; document compute secondary |
| Misreading legacy CSV as live falsification | Supersession prose + labels in charter evidence notes |
| FOCUS language still says "cannot measure" | Patch present tense when Phase 0 lands |

## Impact

- **Prerequisite:** `refinement-game-registrant` for Phase 2; Phase 0–1 can start in parallel after
  adequacy gate (already green).
- **New specs:** arena feature spec under `specs/` (repo format) + openspec delta as needed.
- **Focus:** core paths only (`src/research/`, `src/refinement/`, `src/pde/`, `src/mcts/`,
  `scripts/`, `tests/`, `results/`, `specs/`, `openspec/`).
- **Freeze lift:** when Phase 2 produces an interpretable answer either way (per `docs/FOCUS.md`).

## Relationship to sibling changes

| Change | Role |
| --- | --- |
| `element-local-substrate` | Substrates + adequacy gate |
| `refinement-game-registrant` | First `RefinementGame` + governance |
| **`mcts-classical-amr-arena` (this)** | Pre-reg → falsifiable experiment |
