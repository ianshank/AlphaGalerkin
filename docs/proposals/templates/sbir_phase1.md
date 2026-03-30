# SBIR Phase I Proposal Template - AlphaGalerkin

## A. Cover Page
- **Title**: MCTS-Guided Adaptive Mesh Refinement for [Domain-Specific] Applications
- **Topic Number**: [e.g., N252-088, C59-01]
- **Company**: [Company Name]
- **PI**: [Name, >50% effort commitment]
- **Duration**: [6-12 months per agency]
- **Cost**: [Per agency Phase I limits]
- **NAICS**: 541715 (R&D in Physical Sciences)
- **UEI**: [From SAM.gov]

## B. Technical Volume

### 1. Identification and Significance of the Problem
The computational engineering community faces a fundamental bottleneck: adaptive mesh
refinement (AMR) for PDE-based simulations relies on myopic, heuristic error indicators
that cannot anticipate future solution behavior. This results in:
- 50-80% of simulation time consumed by mesh generation (industry estimate)
- Suboptimal mesh configurations requiring human expert intervention
- No mathematical framework for optimal multi-step refinement planning

**Key insight**: AlphaGalerkin treats mesh refinement as a sequential decision-making
problem and applies Monte Carlo Tree Search (MCTS) - the same planning algorithm behind
AlphaZero - to discover optimal refinement strategies with multi-step look-ahead.

### 2. Technical Approach (Phase I Objectives)
**Objective 1**: Validate MCTS-guided AMR on standard benchmark problems
- L-shaped domain Poisson (singular solution, known optimal convergence rate)
- Viscous Burgers equation with shock formation
- 2D Navier-Stokes Taylor-Green vortex

**Objective 2**: Demonstrate superiority over classical AMR baselines
- Dörfler marking with residual-based indicators
- Zienkiewicz-Zhu error estimation
- Uniform refinement

**Objective 3**: Quantify computational complexity advantage
- O(N) Galerkin attention vs O(N²) standard attention
- FFT-based mixing for O(N log N) evaluation
- Wall-clock comparison at equal error thresholds

### 3. Key Innovation (Verified Novelty)
No published papers combine MCTS with Galerkin methods for PDE solving or mesh
refinement (verified via exhaustive literature search, March 2026). Existing RL-for-AMR
approaches (Yang et al. AISTATS 2023, Foucart et al. JCP 2023, Freymuth et al. 2024)
use standard policy gradient methods. MCTS provides:
- Multi-step look-ahead planning (vs. myopic RL policies)
- Provable exploration guarantees (UCB bounds)
- No training data requirement (operates directly on the PDE)
- Integration with mathematical convergence theory (LBB stability)

### 4. Technical Merit (Prior Results)
**Zero-shot transfer**: Trained on 9x9 grid, achieved MSE = 0.000209 on 19x19 grid
(240x better than 0.05 threshold) without retraining.

**LBB stability**: Galerkin attention satisfies the inf-sup condition, providing
mathematical convergence guarantees absent in PhysicsNeMo, PINNs, and FNO.

### 5. Phase I Work Plan
| Month | Milestone | Deliverable |
|-------|-----------|-------------|
| 1-2 | Benchmark implementation | L-shaped Poisson, Burgers, NS operators |
| 2-4 | MCTS-AMR integration | Full pipeline: PDE → MCTS → refined mesh |
| 4-5 | Baseline comparison | Error-DOF-time comparison tables |
| 5-6 | Documentation | Technical report, Phase II proposal draft |

### 6. Related Work / Principal Investigator
[PI qualifications, relevant publications, company capabilities]

## C. Cost Volume
[Standard SBIR cost breakdown: labor, materials, travel, overhead]

## D. Supporting Documentation
- Company registration (SAM.gov, SBIR.gov)
- PI commitment letter (>50% effort)
- Subcontractor agreements (if any)
- Data management plan

## Notes
- DoD Phase I: 10 pages, $150K-$250K, 6 months
- NSF Phase I: 15 pages, $305K, 12 months
- AFWERX Open Topic: 5 pages, $75K, 3 months
- DARPA Direct-to-Phase-II: 20 pages + 10 page feasibility, $750K-$1.5M
