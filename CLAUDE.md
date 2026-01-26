# CLAUDE.md - AlphaGalerkin Context

## Project Overview
AlphaGalerkin is a resolution-independent Go AI that uses Continuous Operator Learning
(Galerkin Transformers & FNet) instead of discrete CNNs, enabling zero-shot transfer
between board sizes (e.g., 9x9 to 19x19) and accelerating MCTS rollouts via FFT mixing.

## Mathematical Decisions
- [2026-01-26]: Chosen Kernel: Fredholm integral equation with Green's function formulation.
- [2026-01-26]: Basis function selection: Fourier Features for positional encoding.
- [2026-01-26]: Normalization scheme: Monte Carlo integral normalization (1/n) for Galerkin attention.
- [2026-01-26]: LBB Stability: dim(Key) >= dim(Query) to satisfy inf-sup condition.

## Architecture Decisions
- [2026-01-26]: Strategy Body uses GalerkinLinearAttention for O(N) global influence modeling.
- [2026-01-26]: Tactical Head uses SoftmaxAttention to preserve injectivity for local reading.
- [2026-01-26]: FNet mixing uses real-valued FFT (torch.fft.rfft2) for efficiency.
- [2026-01-26]: All tensor operations use einops for dimension clarity.

## Key Mathematical Operators

### GalerkinAttention
Implements Petrov-Galerkin projection with O(N) complexity:
- Projects values onto Key basis: K^T V (Monte Carlo integral)
- Reconstructs in Query basis: Q * Context
- Normalization: 1/n (not 1/sqrt(d))

### FNetBlock
FFT-based mixing for high-speed rollouts:
- FFT2D -> Spectral Mixing -> iFFT2D
- Enables batch MCTS leaf evaluation

### StabilityGuard
Monitors LBB condition during training:
- Computes singular values of Key-to-Value projection
- Ensures sigma_min > beta > 0

## Known Issues
- [None yet]

## Verification Commands
```bash
ruff check src/
mypy src/ --strict
pytest tests/math_kernel/ -v
pytest tests/integration/ -v
python -m src.tools.verify_invariance --train-size 9 --infer-size 19
```

## Directory Structure
```
src/
  modeling/     - Neural architectures and layers
  math_kernel/  - Basis functions, integral approximations
  mcts/         - Monte Carlo Tree Search logic
  tools/        - Verification and utility scripts
tests/
  math_kernel/  - Property-based tests for mathematical operators
  integration/  - End-to-end integration tests
config/         - Hydra/Pydantic configuration schemas
```
