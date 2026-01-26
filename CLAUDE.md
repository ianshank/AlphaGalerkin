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

## Training Infrastructure
- [2026-01-26]: Added complete training pipeline with self-play, replay buffer, and trainer.
- [2026-01-26]: Loss = policy_CE + value_MSE + lbb_regularization for Galerkin stability.
- [2026-01-26]: Replay buffer supports uniform and prioritized experience replay.
- [2026-01-26]: Variable board size batching via padding and masking.
- [2026-01-26]: Checkpoint manager with best model tracking and rotation.

## Known Issues
- [None yet]

## Verification Commands
```bash
# Linting and type checking
ruff check src/
mypy src/ --strict

# Unit tests
pytest tests/math_kernel/ -v
pytest tests/training/ -v

# Integration tests
pytest tests/integration/ -v

# Full test suite
pytest tests/ -v

# Verify resolution independence
python -m src.tools.verify_invariance --train-size 9 --infer-size 19
```

## Training Commands
```bash
# Default training (full config)
python -m scripts.train

# Fast test training (small model, few steps)
python -m scripts.train --config-name=train_fast

# Override parameters
python -m scripts.train training.batch_size=64 training.total_steps=10000

# Resume from checkpoint
python -m scripts.train +resume=checkpoints/alphagalerkin/checkpoint_00010000.pt

# Train on GPU with custom experiment name
python -m scripts.train device=cuda experiment_name=my_experiment
```

## Directory Structure
```
src/
  modeling/     - Neural architectures and layers
  math_kernel/  - Basis functions, integral approximations
  mcts/         - Monte Carlo Tree Search logic
  tools/        - Verification and utility scripts
  training/     - Training infrastructure (NEW)
    loss.py           - AlphaGalerkinLoss (policy + value + LBB)
    replay_buffer.py  - Uniform and prioritized replay buffers
    self_play.py      - MCTS-based self-play game generation
    trainer.py        - Main Trainer class
    checkpoint.py     - Checkpoint save/load management
    evaluation.py     - Win rate and policy agreement metrics
  data/         - Data loading and preprocessing (NEW)
    dataset.py        - PyTorch Dataset classes
    collate.py        - Variable board size collation
tests/
  math_kernel/  - Property-based tests for mathematical operators
  training/     - Tests for training infrastructure (NEW)
  integration/  - End-to-end integration tests
config/         - Hydra/Pydantic configuration schemas
  train.yaml          - Default training config (NEW)
  train_fast.yaml     - Fast test config (NEW)
scripts/        - CLI entry points (NEW)
  train.py            - Training CLI with Hydra
```
