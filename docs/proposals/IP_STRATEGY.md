# AlphaGalerkin Intellectual Property Strategy

## Overview
Hybrid IP protection: trade secrets (immediate) + provisional patents (12-month window) + strategic publications (priority establishment).

## Trade Secrets (Protect Now)
The following are protected as trade secrets and should NOT be published:
1. **MCTS reward engineering** - specific reward functions for mesh refinement
2. **Training hyperparameters** - the specific configurations that achieve optimal performance
3. **MCTS-Galerkin integration methodology** - implementation details of the bridge
4. **Data augmentation strategies** - symmetry exploitation specifics

## Provisional Patent Claims (File Within 12 Months)

### Claim 1: MCTS-Guided Adaptive Mesh Refinement
**Title**: "System and Method for Adaptive Mesh Refinement Using Monte Carlo Tree Search-Guided Galerkin Discretization"
- **What**: Computational pipeline where MCTS selects refinement actions on a Galerkin FEM mesh
- **Key files**: `src/pde/games/mesh_refinement.py`, `src/mcts/search.py`, `src/pde/mcts_adapter.py`
- **Claims focus**: The integration interface, MCTS-derived adaptation criteria, multi-step planning
- **Post-Alice framing**: Technical improvement to specific FEM computation (not abstract algorithm)

### Claim 2: Resolution-Independent Neural Operator Learning
**Title**: "Neural Network Architecture with Galerkin Attention for Resolution-Independent Function Approximation"
- **What**: Galerkin attention mechanism enabling zero-shot transfer between grid resolutions
- **Key files**: `src/modeling/model.py`, `src/modeling/attention.py`
- **Claims focus**: Q(K^T V) formulation with Monte Carlo normalization, LBB stability enforcement
- **Supporting data**: MSE 0.000209 on 19x19 (trained on 9x9)

### Claim 3: LBB-Stabilized Neural Attention Training
**Title**: "Training Method for Neural Attention Networks with Inf-Sup Stability Guarantees"
- **What**: Loss function incorporating LBB regularization to ensure mathematical stability
- **Key files**: `src/training/losses/alphagalerkin.py`, `src/modeling/stability.py`
- **Claims focus**: Log-barrier + threshold penalty for singular value enforcement during training

### Cost Estimate
- Micro entity provisional: $800-$1,600 per application
- Non-provisional conversion: $3,000-$8,000 per patent
- Total budget (2 provisionals + 1 non-provisional): $5K-$15K

## Strategic Publications (Establish Priority)

### Paper 1: "MCTS-Guided Adaptive Mesh Refinement for Galerkin Methods"
- **Publish**: Results from L-shaped domain, Burgers shock, NS benchmarks
- **Reveal**: Architecture overview, comparison results, convergence proofs
- **Protect**: Training details, reward functions, hyperparameters
- **Target**: NeurIPS ML4PhysicalSciences workshop (deadline ~Sep 2026)

### Paper 2: "Resolution-Independent Neural Operators via Galerkin Attention"
- **Publish**: Zero-shot transfer results, spectral bias comparison
- **Reveal**: Q(K^T V) formulation, LBB monitoring methodology
- **Protect**: Specific training recipes, data augmentation
- **Target**: Journal of Computational Physics (6-month review)

## Licensing Strategy
- **Core framework** (`src/templates/`, `src/games/interface.py`): MIT license (ecosystem building)
- **SBIR-funded work** (`src/pde/games/`, `src/mcts/`): Proprietary / dual-license
- **Training infrastructure** (`src/training/`): MIT license (attract contributors)
- **Benchmark results**: CC-BY (encourage citation)

## Timeline
| Action | Deadline | Cost |
|--------|----------|------|
| SAM.gov registration | Immediate | $0 |
| Provisional patent #1 (MCTS-AMR) | Within 12 months of first disclosure | ~$1,200 |
| Provisional patent #2 (Galerkin attention) | Within 12 months | ~$1,200 |
| NeurIPS workshop submission | Sep 2026 | $0 |
| AFWERX Phase I submission | Upon reopening | $0 |
| Non-provisional conversion | 12 months after provisional | ~$5,000-$8,000 |
