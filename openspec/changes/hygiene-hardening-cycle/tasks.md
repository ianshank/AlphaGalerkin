# Tasks: `hygiene-hardening-cycle`

Critical path: Wave 0 (this delta + live charter) → B21 skill → B1 device →
C1 baselines package. CI hardness (Wave A) is parallel with B1 after Wave 0.
B2 waits on C1. Trainer/physics splits wait on C1. Coverage / literals / docs
are later waves.

## Wave 0 — Charter disclosure (this change)

- [x] 0.1 Scaffold `openspec/changes/hygiene-hardening-cycle/` (proposal,
      design, tasks, charter delta)
- [x] 0.2 Amend live frozen-tracks deviation: thesis freeze lifted on
      `results/mcts_classical_amr_arena.csv` (median 0.9532); codec /
      interactive-surfaces remain paused until `config/focus.yaml` is edited
- [x] 0.3 Add live B10 keep-reasons deviation (four packages stay in the
      scope register; test-held, not production-wired, not a 2026-07-22 cut)
- [x] 0.4 `pytest tests/docs/test_charter_alignment.py` and `python scripts/check_doc_links.py`

## Wave B21 — skill before any split

- [x] 1.1 Add `.claude/skills/god-file-split/SKILL.md` (public-name freeze,
      `__all__` one-directional, `del` submodules, mypy `<pkg>.*`, grep old
      path ± directory prefix, coverage `--include` package + `pkg/*`)
- [x] 1.2 Update CLAUDE.md Agentic-harness row 14 skills → 15

## Wave A — CI hardness

- [x] 2.1 Document `transfer-baseline-regression` as a disclosed soft job
      next to `focus`/`secrets` (no `exit 1`; do not mix with exclusion edits)
- [x] 2.2 B23: skip non-`test-slow` jobs on `schedule`; keep `test-slow`
      runnable when `test-fast` is skipped
- [x] 2.3 Reopen `covered_by: ~` clusters one at a time; if still red, keep
      the deselect and write the real exception into the ledger
- [x] 2.4 Do **not** flip mypy / ONNX / backend audit; do **not** land B37
      or B7; do **not** collapse the e2e chess `-k` split

## Wave B1 — `src/device.py`

- [x] 3.1 Move body from `src/poc/device.py`; shim is identity re-export
- [x] 3.2 Migrate only `src/research/baselines.py` import
- [x] 3.3 SBIR `--include` adds `*/src/device.py` and keeps `*/src/poc/device.py`
- [x] 3.4 Identity test: `src.poc.device.resolve_device is src.device.resolve_device`
- [x] 3.5 ARCHITECTURE.md “three root-level modules” → four. No charter
      scope row. No `video_compression` / `hf_space` / solver.py edits

## Wave C1 — `src/research/baselines.py` → package

- [x] 4.1 Freeze public `dir()` names before the split
- [x] 4.2 Split at documented line bands; `del` submodule names
- [x] 4.3 Bind `SOLVER_REGISTRY` before any extra_solvers import; do not
      import extra_solvers from baselines `__init__`
- [x] 4.4 mypy override `"src.research.baselines.*"`
- [x] 4.5 SBIR `--include` adds `*/src/research/baselines/*`
- [x] 4.6 Combined pytest block from the cycle plan, one process

## Wave C2 / C3 — trainer then physics

- [x] 5.1 Trainer facade without `super().__init__()`; do not reorder `_log`
- [x] 5.2 `src/training/losses/physics.py` → package; one file per
      `@register_loss`; public-name freeze + `del` submodules
- [x] 5.3 Skip `chess.py`, `agents/config.py`, codec, `lshape_amr_compare.py`

## Wave B2 — CompareScenarioBase

- [x] 6.1 B2a: `_compare_lock.py` + `_compare_common.py` teardown +
      `lock_scenario_name` function; stochastic `_name_locked` alias
- [x] 6.2 B2b: `setup` / `_record_metrics` / `_write_csv_png_artifacts`;
      arena keeps `write_arena_manifest`; hashes byte-stable
- [x] 6.3 Run all four scenario surfaces in one pytest process (161 passed)
- [ ] 6.4 B15 script CLI dedup stays deferred

## Wave D — Coverage (core-only)

- [x] 7.1 B40: parametrize `fem_baseline.py` `p_adaptive` / `hp_adaptive`
- [x] 7.2 B39: thread raw skfem Dof through `dirichlet_dof_indices`; do not
      delete `assemble_and_solve` `basis.get_dofs()` (needed for `condense(D=)`)
- [x] 7.3 `tests/support`: gated at 85 from docs + import-graph consumers
- [x] 7.4 Park templates / math_kernel / backend 54 / deployment 25 / B37

## Wave E — Hardcoded / unread (zero numeric change)

- [x] 8.1 Name policy CE `clamp(min=-100.0)` once; three call sites
- [x] 8.2 Trainer start-buffer `min(batch_size * 10, replay_buffer_size // 10)`
      → typed fields / constants, same defaults
- [x] 8.3 Cole-Hopf already named — skip
- [ ] 8.4 Self-play fallback `8`: fail-loud preferred (own behavior-change)
- [x] 8.5 Reject or document-as-reserved: `rbf_kernel`, `success_metrics`,
      `StrangTrainerConfig.n_particles`, `dt_min`/`dt_max` with `adaptive_dt`

## Wave F — Enterprise docs

- [ ] 9.1 `src/research/AGENT.md` (and other in-scope missing AGENT.md)
- [ ] 9.2 Four B10 keep-reason AGENT.md files
- [ ] 9.3 Fix stale `operators.py` sub-agent row in `src/pde/AGENT.md`
- [ ] 9.4 B19 extras in README / getting-started (skip dashboard extra)
- [ ] 9.5 B8 CLAUDE.md coverage-gate rows ⊆ CI; mutation-kill a planted row
- [ ] 9.6 CHANGELOG release cut after several hygiene commits, not first

## Explicit non-tasks / deferred

- mypy torch-pin + hard-gate (own charter PR)
- B7 args-file extract
- B37 eval_harness gate
- B3 registry consolidation, B5 config base unification
- B9 seed-stride unification
- Modeling LBB `* 10`, FNO `128`
- Manufactured Heat solution
- Promote `focus` / `secrets` into `ci-success`
- Collapse e2e chess `-k` split
- Context7 (quota exceeded)
