# Implementation sequence (OpenSpec)

Peer-reviewed plan (Product / Architect / SQE / codebase reality check + Cody).

```
element-local-substrate (tasks 0–6) ✅
        │
        ▼
refinement-game-registrant   ✅ (Slice E: real MCTS smoke, not adapter-only)
        │
        ▼
mcts-classical-amr-arena Phase 0 ✅ → Phase 1 (classical in harness) → Phase 2 scored run
        │
        ▼
Phase 3 reporting + freeze lift (from committed artifacts only)
```

## Package map

| Directory | Purpose |
| --- | --- |
| `openspec/changes/refinement-game-registrant/` | Slice E: pure `RefinementGame`, safe registration, fingerprint cache, governance |
| `openspec/changes/mcts-classical-amr-arena/` | Pre-registration first, then falsifiable MCTS vs classical experiment |

## Explicit non-goals this cycle

Frozen: `codec` (`src/video_compression/`), `interactive-surfaces` (`dashboard/`, `hf_space/`).
Deferred: multi-field PDE, PETSc/MFEM, trained evaluator, SBIR narrative as a substitute for evidence.

## Kill criteria

- Adequacy gate regresses on `skfem_tri` → stop arena; diagnose.
- Arena negative with honest stats → report per charter; freeze lifts.
- No unearned MCTS win claims from adequacy rates alone.
