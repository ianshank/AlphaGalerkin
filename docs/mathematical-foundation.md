# Mathematical Foundation

AlphaGalerkin reinterprets attention as a **Petrov-Galerkin projection** and uses
that structure to get linear-complexity global mixing with a stability guarantee.

## Galerkin projection

The weak form of an operator equation is:

```
Find u ∈ U:  ⟨Lu, v⟩ = ⟨f, v⟩   ∀ v ∈ V
```

Mapped onto attention:

- **Q** (Query) — test-function basis
- **K** (Key)   — trial-function basis
- **V** (Value) — the function being projected

The projection becomes:

```
Context = Kᵀ V / n      (Monte-Carlo integral, normalization 1/n)
Output  = Q · Context    (reconstruction)
```

This is O(N) in sequence length N, versus O(N²) for softmax attention.

## LBB / inf-sup stability

Convergence requires the Ladyzhenskaya–Babuška–Brezzi (inf-sup) condition:

```
inf_u sup_v  ⟨Lu, v⟩ / (‖u‖ ‖v‖)  ≥  β > 0
```

In practice this is ensured by `dim(Key) ≥ dim(Query)` and monitored during
training by the `StabilityGuard`, which tracks the smallest singular value of the
Key→Value projection.

## Resolution transfer

Zero-shot transfer across resolutions relies on spectral methods:

1. **Fourier encoding** — position → frequency representation.
2. **Spectral filtering** — anti-alias when changing resolution.
3. **Normalization** — adjust the Monte-Carlo integral factor for the new N.

## Complexity

| Operation | Standard attention | Galerkin attention |
| --- | --- | --- |
| 9×9 board | O(81² · d) | O(81 · d²) |
| 19×19 board | O(361² · d) | O(361 · d²) |
| Scaling in N | Quadratic | Linear |

FNet mixing replaces attention with FFT operations (O(N log N)) for fast MCTS
leaf evaluation. The complexity advantage is the theoretical claim; concrete
wall-clock numbers depend on hardware and model size and should be reproduced
with `python -m src.experiments.benchmark_fnet` rather than quoted from a fixed rig.

## Further reading

- [Glossary](GLOSSARY.md) — term-by-term definitions.
- [C4 architecture (Mermaid)](architecture/c4_mermaid.md) — includes the
  attention / Monte-Carlo-integral math in diagram form.
- [Related work](related-work.md) — the novelty-boundary register.
