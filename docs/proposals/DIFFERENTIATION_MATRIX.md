# Technical Differentiation Matrix

## Overview
Structured comparison of AlphaGalerkin against competing approaches for PDE solving and mesh optimization. Data sourced from benchmarks (`config/benchmarks/sbir_suite.yaml`), published literature, and competitor documentation.

## Method Comparison

| Capability | AlphaGalerkin | PINNs | FNO | Classical AMR (Dorfler/ZZ) | PhysicsNeMo | PhysicsX |
|-----------|--------------|-------|-----|---------------------------|-------------|----------|
| **Computational Complexity** | O(N log N) | O(N) per iteration, O(N × epochs) total | O(N log N) | O(N) per refinement step | O(N log N) to O(N²) | O(1) inference |
| **Accuracy Guarantee** | LBB inf-sup stability (provable) | None (empirical convergence) | None (empirical) | A posteriori error bounds | None (empirical) | None (surrogate fidelity) |
| **Zero-Shot Resolution Transfer** | Yes (measured MSE ~4e-4, no retraining; a retrained CNN is more accurate) | No (retrain per domain) | Limited (fixed architecture) | N/A (no neural component) | No (retrain per resolution) | No (retrain per geometry) |
| **Training Data Required** | None (operates directly on PDE) | None (self-supervised) | Large dataset from solver | N/A | Large dataset from solver | Large dataset from solver |
| **Multi-Step Planning** | MCTS with configurable search depth | None (gradient descent) | None (single forward pass) | Myopic (single-step error indicator) | None (single forward pass) | None (single forward pass) |
| **Convergence Guarantee** | Provable via LBB condition | Empirical (may diverge) | Empirical (spectral bias) | Provable (a posteriori) | Empirical | Empirical |
| **Mesh Adaptivity** | MCTS-guided h/p/hp-refinement | No mesh (collocation) | No mesh (spectral) | Error-indicator driven | No mesh (neural operator) | No mesh (surrogate) |
| **Technology Readiness** | TRL 3-4 | TRL 4-5 | TRL 4 | TRL 9 (industry standard) | TRL 5-6 | TRL 5-6 |
| **Published Prior Art** | None (verified novelty gap) | Extensive (Raissi et al. 2019+) | Extensive (Li et al. 2020+) | Extensive (Dorfler 1996, ZZ 1987) | Growing (NVIDIA 2023+) | Limited (proprietary) |
| **Open Source** | Partial (MIT core, proprietary MCTS) | Many implementations | PyTorch, JAX versions | Many implementations | Apache 2.0 | Proprietary |

## Detailed Capability Analysis

### Where AlphaGalerkin Wins

| Advantage | vs. PINNs | vs. FNO | vs. Classical AMR | vs. PhysicsNeMo |
|-----------|-----------|---------|-------------------|-----------------|
| **Multi-step look-ahead** | MCTS plans 10-100 steps ahead vs. single gradient step | MCTS vs. single forward pass | MCTS vs. myopic error indicator | MCTS vs. no planning |
| **Resolution independence** | Transfer across resolutions without retraining | Better transfer than fixed architecture | N/A (different paradigm) | No retraining needed |
| **Spectral bias mitigation** | Multi-scale Fourier features | Addresses FNO's Gibbs phenomenon | N/A | Not available in toolkit |
| **Convergence proof** | LBB stability with log-barrier regularization | No convergence guarantee | Both have guarantees | No mathematical guarantee |

### Where AlphaGalerkin Has Gaps (Honest Assessment)

| Gap | Competitor Advantage | Mitigation Strategy |
|-----|---------------------|---------------------|
| **TRL maturity** | Classical AMR is TRL 9 (production) | SBIR Phase I validates to TRL 5 |
| **Inference speed** | PhysicsX O(1) surrogate is faster at inference | MCTS depth is configurable (trade accuracy for speed) |
| **3D validation** | Classical AMR has extensive 3D benchmarks | 3D domain support exists, needs benchmarks |
| **Production readiness** | PhysicsNeMo has NVIDIA GPU optimization | ONNX export ready, PyTorch native |
| **Customer validation** | PhysicsX has Rio Tinto, Siemens | SBIR provides government customer validation |

## Benchmark Evidence

Results from `config/benchmarks/sbir_suite.yaml`:

### L-Shaped Poisson (AMR Benchmark)
- **Exact solution**: r^(2/3) * sin(2*theta/3) — singular gradient at reentrant corner
- **AlphaGalerkin advantage**: MCTS concentrates DOF at singularity (multi-step planning vs. myopic marking)
- **Baseline comparison**: Dorfler AMR with marking_fraction=0.3

### Burgers Shock (Nonlinear PDE)
- **Exact solution**: Cole-Hopf transform
- **AlphaGalerkin advantage**: MCTS anticipates shock location and pre-refines (look-ahead)
- **Baseline comparison**: Uniform FDM, PINN (autograd Laplacian)

### Navier-Stokes Taylor-Green (CFD Benchmark)
- **Exact solution**: Analytical Taylor-Green vortex decay
- **AlphaGalerkin advantage**: Resolution-independent operator handles multi-scale vortex dynamics
- **Baseline comparison**: Uniform FDM at Re=100, 200, 500

## Key Takeaway for SBIR Proposals

> AlphaGalerkin is the **only approach** that combines:
> 1. Multi-step planning (MCTS) for mesh/basis selection
> 2. Mathematical convergence guarantees (LBB stability)
> 3. Zero-shot resolution transfer (no retraining)
> 4. No training data requirement (operates directly on PDE)
>
> This combination (MCTS multi-step look-ahead + Galerkin basis / error-driven refinement) has **no published precedent** — a *narrow* gap. Note: TreeMesh (arXiv:2111.07613) applies MCTS to FE mesh *generation*, a distinct problem, so the blanket "no MCTS+FEM" claim is false. See `docs/proposals/PRIOR_ART_REVIEW.md`.

## References
- Benchmark config: `config/benchmarks/sbir_suite.yaml`
- Baseline solvers: `src/research/baselines.py`
- Benchmark runner: `src/research/pde_benchmarks.py`
- SBIR demo: `scripts/run_sbir_demo.py`
- Competitive landscape: `docs/proposals/COMPETITIVE_LANDSCAPE.md`
- IP strategy: `docs/proposals/IP_STRATEGY.md`
