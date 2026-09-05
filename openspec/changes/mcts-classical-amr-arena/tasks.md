# Tasks: `mcts-classical-amr-arena`

Critical path: Phase 0 → (Phase 1 ∥ registrant) → Phase 2 → Phase 3.
Phase 2 is blocked on `refinement-game-registrant` exit criteria.

## Phase 0 — Pre-registration (no numbers)

- [ ] 0.1 Add `specs/mcts_classical_amr_arena.spec.md` with hypothesis, falsifiers, metric
      hierarchy, artifact contract, and AQA scenarios
- [ ] 0.2 Add pinned config YAML (kind, θ, DOF window, seeds, search, estimator/metric)
- [ ] 0.3 Manifest schema: config hash, git SHA, dirty flag policy, `dof_convention`, substrate
      `describe()` dump
- [ ] 0.4 Document adequacy precondition (abort if gate fails on this config)
- [ ] 0.5 Explicit out-of-scope block: trained evaluator, frozen tracks, PETSc/MFEM
- [ ] 0.6 Guard or checklist: reject new README/charter AMR ratios that lack a manifest pointer
- [ ] 0.7 Update `docs/FOCUS.md` present tense (measurement blocker ≠ "substrate still wrong")

## Phase 1 — Classical arms (shared substrate)

- [ ] 1.1 Dörfler arm on `skfem_tri` via `sweep.py` under Phase 0 config
- [ ] 1.2 Uniform arm on the same substrate/config
- [ ] 1.3 Commit raw artifacts under `results/` with valid manifests (no unknown hashes)
- [ ] 1.4 Verify adequacy precondition still passes for the locked config

## Phase 2 — MCTS arm (blocked on registrant)

- [ ] 2.1 Confirm `refinement-game-registrant` exit criteria (pure game, registry, cache, governance)
- [ ] 2.2 MCTS arm via registered `RefinementGame` + `RefinementGameAdapter`
- [ ] 2.3 Matched-DOF primary comparison vs classical; record compute secondary ungated
- [ ] 2.4 Multi-seed run per pre-reg; commit CSV + run.json
- [ ] 2.5 Human number-match review against committed artifacts before any doc claim

## Phase 3 — Reporting

- [ ] 3.1 README / charter evidence updates **only** from Phase 2 artifacts
- [ ] 3.2 Quote all rates with θ + DOF window
- [ ] 3.3 Label legacy `lshape_mcts_vs_dorfler` / tensor-grid rows as non-informative for
      element-local policy (if not already done in registrant supersession)
- [ ] 3.4 FOCUS freeze-lift note: interpretable answer recorded (win **or** honest negative)
- [ ] 3.5 `CHANGELOG.md` + CLAUDE milestone

## Explicit non-tasks

- Estimator vectorisation
- Trained evaluator
- Deleting `LShapeAMRGame` before golden retirement condition
- Codec / dashboard / multi-field / PETSc work
