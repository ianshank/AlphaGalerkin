# Tasks: `arena-lookahead-attribution`

Critical path: 1.1 → 2.1 → 2.2 (the artifact must carry the greedy control before any
statement can cite it) → 3.x → 4.1 (the guard reads the corrected text) → 5.1.

## 1 — Recording path

- [x] 1.1 `scripts/run_mcts_classical_amr_arena.py --proposal-grade`: `RunRecorder.start`
      pre-flight before the scenario runs; `verify_sidecar` re-reads the sidecar and requires
      the pre-flight's git snapshot plus `assert_proposal_grade`
      (`tests/scripts/test_run_mcts_classical_amr_arena_preflight.py`, 5/5 planted defects)

## 2 — Artifact (claims-ledger / run-provenance)

- [x] 2.1 Re-record from a clean tree:
      `python -m scripts.run_mcts_classical_amr_arena --proposal-grade`
- [x] 2.2 Verify before committing: every `uniform` / `dorfler` / `mcts` row identical in every
      column but `wall_time_seconds`; the matched-DOF and matched-solves ratios and both anchors
      bit-identical; divergence max 0, MCTS/greedy 1.0, greedy/Dörfler 0.9531782126653989;
      `assert_proposal_grade` passes
- [x] 2.3 `python -m scripts.artifact_manifest write` and `check`; commit CSV, PNG, sidecar and
      manifest together

## 3 — Correct every live statement

- [x] 3.1 Charter: evidence row, Novelty arena paragraph, frozen-tracks deviation row, new
      Evidence-Backed Claims scenario (delta in `specs/project-charter/spec.md` here; live
      text applied)
- [x] 3.2 `README.md`: the "Key features" arena bullet and the roadmap's scored-arena item
- [x] 3.3 `docs/FOCUS.md`: "The current focus" and "When the freeze lifts"
- [x] 3.4 `specs/mcts_classical_amr_arena.spec.md`: verdict table and "Win" row annotated in
      place; measured-result section records the re-record
- [x] 3.5 `CLAUDE.md`: strike the 2026-09-08 attribution (**CORRECTED (2026-10-08)**); append
      the 2026-10-08 Gate 0 milestone

## 4 — Guard

- [ ] 4.1 Move `_amr_policy_ratio_subjects` / `_csv_citations_in` (and their helpers) into
      `tests/support/charter.py`; import them back into `test_charter_alignment.py`
- [ ] 4.2 README subjects read block by block (`tests/support/perf_claims.py::split_blocks`)
- [ ] 4.3 `tests/docs/test_lookahead_attribution.py`: label required on a no-divergence
      citation; label not stale; arena sidecars carry both divergence metrics; vacuity on
      both surfaces and on the label's presence in the evidence register
- [ ] 4.4 Plant and revert the mutations named in the module docstring

## 5 — Verify

- [ ] 5.1 `pytest tests/docs tests/claude tests/regression -q`; the arena and greedy-control
      Regression Surface rows; `TestCommittedArtifacts`; `test_proposal_grade_sidecars.py`;
      `python -m scripts.check_doc_links`; `python -m scripts.artifact_manifest check`;
      `python -m scripts.measure_shape check`; the `tests/support` coverage gate

## Deferred (out of this change)

- Gate 1 — the look-ahead experiment itself, on a problem where greedy has a structural reason
  to be myopic (peer review §6); pre-registered with `spec-new`
- `CHANGELOG.md` `[Unreleased]` entry and the `CLAUDE.md` Regression Surface row — the
  orchestrator owns both; proposed text is in the change's hand-off report
- Archiving this package to `openspec/changes/archive/` once it has landed (the 2026-09
  packages `focus-secrets-merge-gate` and `hygiene-hardening-cycle` were left in place)
- Re-freezing, or re-scoping `config/focus.yaml` — owner decision
