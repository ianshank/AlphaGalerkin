# Implementation sequence (OpenSpec)

Peer-reviewed plan (Product / Architect / SQE / codebase reality check + Cody).

```
element-local-substrate (tasks 0–6) ✅
        │
        ▼
refinement-game-registrant   ←── implement NEXT (Slice E)
        │
        │     ┌── mcts-classical-amr-arena Phase 0–1 (parallel OK)
        ▼     ▼
mcts-classical-amr-arena Phase 2 (MCTS arm) → Phase 3 reporting
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
