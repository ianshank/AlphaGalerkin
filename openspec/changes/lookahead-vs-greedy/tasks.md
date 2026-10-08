# Tasks: `lookahead-vs-greedy`

Critical path: 1 (pre-registration, committed alone and first) → 2 → 3 → 4.
No comparison run on either testbed happens before task 1.3 is committed.

## 1 — Pre-registration (first commit; no numbers)

- [x] 1.1 `specs/lookahead_vs_greedy.spec.md`: question, testbeds, arms, budgets, metric, break-even
      formula, depth definition, GO criteria, abort/invalid conditions, interpretation
- [x] 1.2 `specs/README.md` index row
- [x] 1.3 This change package (proposal, design, tasks, charter delta)

## 2 — Implementation

- [x] 2.1 `MCTS.root` (read-only) + `subtree_depth(node)` with tests; F0/F1 surfaces green;
      `src/mcts/search.py` within its size budget
- [x] 2.2 Arena config: `zshape_poisson`, operator-aware `adequacy_gate()`, relaxed adequacy
      validator; arena artifact still reproduces
- [x] 2.3 `src/research/lookahead_vs_greedy_metrics.py`: matched DOF, best classical, first passage,
      break-even, verdict
- [x] 2.4 `src/research/lookahead_vs_greedy.py`: adequacy abort, classical arms, greedy, MCTS
      decision rule (depth + divergence per step), span check, artifacts via `RunRecorder`
- [x] 2.5 `LookaheadVsGreedyConfig` + `@scenario("lookahead_vs_greedy")` + `load_config_from_dict`
      dispatch; YAMLs for T1 and T2
- [x] 2.6 `scripts/run_lookahead_vs_greedy.py` (exit code ≠ verdict)
- [x] 2.7 Charter capability row (capability region only)
- [x] 2.8 Tests: unit (config, verdict incl. each criterion failing alone, K* incl. ∞, depth),
      Hypothesis (verdict monotone), tensor_grid integration, E2E `--help` + one `fem_required`
      skfem smoke under `tests/e2e/`; coverage gates; shape ratchet; import contracts;
      abstraction audit

## 3 — Runs (clean tree, after task 2 is committed)

- [x] 3.1 T1 with `--proposal-grade` → `results/lookahead_vs_greedy_lshape.*`
- [x] 3.2 T2 with `--proposal-grade` → `results/lookahead_vs_greedy_zshape.*`
- [x] 3.3 `python -m scripts.artifact_manifest write` then `check`; commit artifacts + manifest
- [x] 3.4 `assert_proposal_grade` on both sidecars

## 4 — Reporting (through `claims-ledger`, after the runs)

- [x] 4.1 Charter: two evidence-register rows (NO-GO on T1 and T2), the arena row's pointer, the
      Novelty paragraph and the frozen-tracks deviation row (delta here; live text applied);
      `README.md` (feature bullet and roadmap), `docs/FOCUS.md`, `specs/mcts_classical_amr_arena.spec.md`
      and the peer review's status table follow
- [x] 4.2 `CHANGELOG.md` bullets and the `CLAUDE.md` Regression Surface row and milestone
- [x] 4.4 `tests/docs/test_lookahead_attribution.py` reads Gate 1 sidecars through
      `primary_decisions_diverging_from_greedy` (`DIVERGENCE_SCHEMAS`); every committed sidecar
      recording a divergence count must belong to a declared harness; 12/12 planted defects killed
- [ ] 4.3 Archive this change. Deferred: no change package in this repository has been archived
      yet (`openspec/changes/archive/` holds only `.gitkeep`), and the specs and docs that cite
      `openspec/changes/lookahead-vs-greedy/` would need repointing in the same step

## Deferred (out of this change)

- T3, the moving-front testbed — needs a time-dependent substrate; the decisive test.
- Goal-oriented refinement with a dual-weighted greedy baseline.
- Any trained evaluator or post-result tuning.
