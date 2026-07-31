# Tasks: `project-charter-alignment`

## 1. P0 — claims contradicted by their own artifacts

- [x] 1.1 `CLAUDE.md` — replace the retracted AMR headline with the committed figures (median
      0.9605 / ~4% win, spread 0.8166–1.1157, matched-compute ratio 1.26, 0/5 seeds, ~350×) and
      add the F0-defect retraction note
- [x] 1.2 `CLAUDE.md` — "two honest comparisons" → three (`l2_error_ratio_at_matched_solves`)
- [x] 1.3 `README.md` + `CLAUDE.md` (×4 sites) — transfer MSE ≈4e-4 → committed ≈2.3e-3, with the
      retrained-CNN ≈1.6e-4 and ~14× ratio, citing the artifacts
- [x] 1.4 Propagate the corrected transfer figure through the nine outward-facing documents
      (`IP_STRATEGY`, `sbir_phase1`, `DIFFERENTIATION_MATRIX`, `COMPETITIVE_LANDSCAPE`,
      `VALUATION_FRAMEWORK`, `proposal/outline`, `doe_genesis/theory`,
      `CHESS_ENGINE_BENCHMARKING`, `GALERKIN_FUSION_HEAD_PLAN`)
- [x] 1.5 `docs/demos/transfer_results.md` — mark the table as spike numbers and name the
      committed benchmark, so it cannot be mis-quoted again
- [x] 1.6 `concept_note.md` + `outline.md` — mark the Pareto artifact `[PENDING]`, matching
      `outreach_template.md`'s existing honest framing
- [x] 1.7 `CLAUDE.md` — banner the four `video_compression` milestones as REMOVED-2026-07-22
- [x] 1.8 `docs/TRAINING_DATA_SOURCES.md` — scope out the removed video-compression domain

## 2. P1/P2 — stale scope, broken commands, hygiene

- [x] 2.1 Remove the three never-runnable `scripts/train_distributed.py` command blocks
      (`CLAUDE.md`, `src/distributed/AGENT.md`, `docs/IMPLEMENTATION_PLAN.md`) and state the gap
- [x] 2.2 `specs/README.md` — add the missing `lshape_amr_compare` row; correct the
      `llm_prior_ood` status
- [x] 2.3 `AGENT.md` — drop the "Neural Video Compression" domain
- [x] 2.4 `README.md` + `docs/getting-started.md` — document the undocumented `fem` extra
- [x] 2.5 `CLAUDE.md` — `loss.py` → `losses/` package
- [x] 2.6 `CLAUDE.md` + `specs/headline_runs.spec.md` — repoint the moved `PR86_HEADLINE_RUNS.md`
- [x] 2.7 `specs/transfer_baseline_compare.spec.md` — drop the dangling `lambda_scheduling.spec.md`
      reference
- [x] 2.8 `docs/NEXT_STEPS_PLAN.md` — banner as historical; correct the inverted mypy claim
- [x] 2.9 `docs/ROI_IMPLEMENTATION_PLAN.md` — version label + two dead module paths
- [x] 2.10 `docs/architecture/components.md` — refresh the stale operator and game snapshots
- [x] 2.11 `CONTRIBUTING.md` — "chess ≥80" is really the whole `src/games` package
- [x] 2.12 `docs/README.md` — move the COMPLETE plan out of "Roadmap" into History

## 3. OpenSpec scaffold

- [x] 3.1 `openspec/project.md` — conventions + precedence (charter > ARCHITECTURE > CLAUDE > specs)
- [x] 3.2 `proposal.md`, `design.md`, `tasks.md`
- [x] 3.3 Delta spec under `changes/.../specs/project-charter/spec.md`

## 4. Charter

- [x] 4.1 `openspec/specs/project-charter/spec.md` — R1–R7, thin and referential
- [x] 4.2 Six delimited registers; seven accepted deviations, each with a reason
- [x] 4.3 Add the charter row to `ARCHITECTURE.md`'s documentation-hierarchy table
- [x] 4.4 Point `CLAUDE.md`, `README.md`, `CONTRIBUTING.md`, `specs/README.md`, `docs/README.md`
      at the charter

## 5. Guards

- [x] 5.1 `tests/docs/test_charter_alignment.py` — 7 Requirement guards + 2 meta-guards
- [x] 5.2 Promote `CUT_MODULES` to a shared constant; widen enforcement beyond `hf_space/`
- [~] 5.3 **DROPPED after prototyping** — resolving repo-path-shaped inline code spans in
      `scripts/check_doc_links.py`. Measured: 144 unresolved spans across 30 files; adding
      glob expansion and file-relative/ancestor-relative base resolution (matching the checker's
      existing link semantics) brings it to **105 across 21 files**, and the residue is
      overwhelmingly legitimate:
      * bare sub-paths inside package-scoped sections (`games/basis_selection.py` under a
        `src/pde/` heading; 19 such rows in `ARCHITECTURE.md` alone)
      * deliberate placeholders (`src/games/my_game.py`, `src/poc/scenarios/my_scenario.py`,
        `docs/adr/NNNN-post-fusion-direction.md`)
      * a deliberately-*rejected* path (`src/operators/` in
        `specs/stochastic_galerkin_nke.spec.md`, which exists to explain why it was not used)
      * gitignored runtime outputs (`outputs/**`, `checkpoints/**`)
      * documented future artifacts (`config/baselines/transfer_headline.json`, `docs/paper/`)
      * cross-repo references to Mouse-Droid-AGI (`sensing/**`, `scripts/bench_fusion.py`)
      Separating those from real defects needs a fuzzy relative-path heuristic plus a ~15-entry
      allowlist — the "turns `docs.yml` red and gets reverted" outcome the design warned about.
      The genuine catches were harvested by hand instead (task 5.3a). Revisit only with a
      convention for marking intentionally-nonexistent paths.
- [x] 5.3a Harvest the prototype's genuine catches: `docs/TRAINING_IMPLEMENTATION_TEMPLATE.md`
      `src/training/loss.py` → `losses/alphagalerkin.py` (the rest were already fixed in §1–§2 or
      live in banner-stamped historical documents)
- [x] 5.4 `.github/workflows/docs.yml` — widen `paths:` so docs-only PRs run the link checker
- [x] 5.5 Register the guard command in `CLAUDE.md`'s Regression Surface table

## 6. Verification

- [x] 6.1 Prove every guard bites (one mutation each, then revert)
- [x] 6.2 Order-independence: `pytest tests/poc tests/docs -q`
- [ ] 6.3 Full CPU surface + `ruff check` / `ruff format --check`
