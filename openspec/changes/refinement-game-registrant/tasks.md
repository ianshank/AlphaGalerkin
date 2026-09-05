# Tasks: `refinement-game-registrant`

Prerequisite: `element-local-substrate` tasks 0–6 complete (substrates + adequacy gate).
Critical path: 1 → 2 → 3 → 4 → 5.

## 0. Alignment with parent change

- [ ] 0.1 Add a pointer in `openspec/changes/element-local-substrate/tasks.md` §7–8 that Slice E
      execution is owned by this change (avoid two sources of unchecked truth)
- [ ] 0.2 Confirm `tests/research/test_amr_arena_interpretability.py` still green on current main

## 1. Minimal `RefinementGame`

- [ ] 1.1 Add concrete game module **outside** `src/refinement/` (e.g.
      `src/pde/games/substrate_refinement.py`) implementing `RefinementGame` over a
      `RefinementSubstrate`
- [ ] 1.2 Pure `apply_action(state, action)` — no instance mutation; mesh via fingerprint /
      immutable handle
- [ ] 1.3 Map `SubstrateSolveResult` → `RefinementState` (error, dofs, indicators, budget/step)
- [ ] 1.4 Action space: per-element (or documented top-1) mark → `refine`; reject invalid ids
- [ ] 1.5 Unit tests: purity, terminal/reward smoke, tensor export shape

## 2. Registration (never `__init__`)

- [ ] 2.1 `@register_refinement_game("…")` on the concrete class
- [ ] 2.2 Explicit `register_*` module (e.g. `src/pde/register_refinement_games.py`); document
      import side-effect pattern
- [ ] 2.3 Import-graph test: package `__init__` files do not register refinement games
- [ ] 2.4 Registry clear setup/teardown (or subprocess) per `src/refinement/AGENT.md`

## 3. Substrate lookup + fingerprint cache

- [ ] 3.1 First non-test `RefinementSubstrateRegistry` lookup by config `kind`
- [ ] 3.2 Fingerprint-keyed solve cache bounded by `solve_cache_max_entries`
- [ ] 3.3 Drop `("RefinementSubstrate", "fingerprint")` from
      `scripts/audit_abstractions.py::_STAGED_FOR_UPCOMING_TASK`
- [ ] 3.4 Ensure `src/research/substrates/` concretes are imported from the register module (not
      only via direct test construction)

## 4. Adapter smoke (not the arena)

- [ ] 4.1 Wire `RefinementGameAdapter` for a single short search smoke (CPU tensor_grid and/or
      `fem_required` skfem)
- [ ] 4.2 Assert smoke does **not** publish README/charter headline numbers

## 5. Governance / charter / docs

- [ ] 5.1 Retire `RefinementGameRegistry`-has-no-registrants deviation (was parent task 8.1b)
- [ ] 5.2 Add time-boxed two-path deviation (legacy L-shape harness vs substrate game) with
      retirement condition = golden is sole remaining consumer
- [ ] 5.3 Mark `specs/lshape_amr_compare.spec.md` superseded → point at substrate + this change /
      arena successor
- [ ] 5.4 `CLAUDE.md` Regression Surface rows + `CHANGELOG.md` entry
- [ ] 5.5 Run manifests for any new committed artifact (no dirty/`config_hash=unknown` proposal-grade claims)
- [ ] 5.6 Fix stale FOCUS/proposal present tense: adequacy gated; MCTS-on-skfem still untested
- [ ] 5.7 Soft-fix `ARCHITECTURE.md` if it still claims `src/pde/` "implements" `RefinementGame`
      without distinguishing `PDEGame` / `GameRegistry`

## 6. Exit criteria

- [ ] 6.1 Charter guards green; focus check green; adequacy gate still passes/fails correctly
- [ ] 6.2 Parent `element-local-substrate` tasks 7.1–8.5 checked off with pointer here
- [ ] 6.3 Sibling `mcts-classical-amr-arena` unblocked for Phase 0 pre-registration commit
