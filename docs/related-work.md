# Related Work — Novelty-Boundary Register

This document holds AlphaGalerkin's per-paper related-work entries with an explicit
**novelty boundary** for each: what the cited work does, what it does NOT do relative to
AlphaGalerkin's claims, and how it complements this project.

**Structure rule (enforced):** every entry claiming method overlap with AlphaGalerkin
MUST carry a **"What it does NOT do"** clause covering MCTS/planning, adaptive
mesh/basis selection, and LBB stability — unless the source paper demonstrably includes
them. This rule is executable: `tests/regression/test_related_work_guard.py` parses the
delimited entries region below and fails on any entry missing the clause. It exists to
prevent inadvertent over- OR under-claiming of novelty (see
`docs/proposals/PRIOR_ART_REVIEW.md` for the two standing honesty constraints: never the
blanket "no MCTS+FEM" claim — TreeMesh exists — and novelty ≠ superiority).

Entries live between the markers below; sections outside the markers (this preamble, the
summary notes at the end) are exempt from the guard.

<!-- entries:start -->

## 1. Neural Kolmogorov Equations (NKE) — arXiv:2607.19173

*"Neural Kolmogorov Equations: Parallelizable Learning of Stochastic Dynamics under
General Noise."*

**Provenance caveat (read first):** the paper was unreachable from the implementation
environment (arXiv rejects datacenter IPs; the paper postdates the implementing model's
training data). AlphaGalerkin's stochastic layer (`src/pde/stochastic/`) is therefore
implemented from the **standard, independently derivable formulation** — moment-matching
Galerkin projection of the Kolmogorov forward equation onto Gaussian mixtures, symmetric
Strang composition, compound-Poisson jump semigroup — validated against closed-form
OU/jump-OU references rather than against the paper's own tables. A paper-exact
cross-check of the projection ansatz (especially mixture-weight dynamics and the loss
weighting) is an **open reviewer follow-up** (`specs/stochastic_galerkin_nke.spec.md`,
Out of Scope).

**What it does:** derives a Lagrangian Galerkin projection of the Kolmogorov Forward
Equation onto a Gaussian-mixture manifold and trains it via operator splitting
(Strang/Lie-Trotter) with parallel-in-time losses over precomputed particle data,
handling general (including Lévy/jump) noise via a mixture-density jump model. This is
the citable basis for AlphaGalerkin's additive stochastic-operator layer.

**What it does NOT do (and we do not claim it does):** no MCTS or planning of any kind;
no adaptive basis or mesh selection (the Gaussian-mixture basis is fixed, not searched);
no LBB/inf-sup stability claims — and AlphaGalerkin makes **no LBB claims for the
stochastic layer** either. NKE therefore does not overlap with, compete with, or
undercut AlphaGalerkin's narrow novelty claim (MCTS multi-step look-ahead for
error-driven refinement and Galerkin basis selection).

**Complementarity:** NKE-style stochastic Galerkin projection gives AlphaGalerkin a
density-propagation path alongside its deterministic operator learning; a future
research item is MCTS-guided selection *over* the stochastic layer's mixture basis
(K, component placement), which would combine the two without merging their code paths.

<!-- entries:end -->

## Notes

- The comprehensive per-family comparison (PINNs, FNO/DeepONet, RL-for-AMR canon,
  TreeMesh) lives in `docs/doe_genesis/related_work.md` and
  `docs/proposals/PRIOR_ART_REVIEW.md`; new entries here should cross-link rather than
  duplicate.
- When adding an entry, copy the NKE entry's section skeleton (**What it does** /
  **What it does NOT do** / **Complementarity**) and keep it inside the markers so the
  guard covers it.
