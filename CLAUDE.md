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

## Physics PoC (Supervised Learning Validation)
- [2026-01-26]: Added Poisson equation solver for synthetic data generation.
- [2026-01-26]: PhysicsOperator neural network for influence field prediction.
- [2026-01-26]: Zero-shot transfer validation: Train on 9x9 → Evaluate on 19x19.
- [2026-01-26]: Success criterion: MSE < 0.05 on 19x19 without retraining.

## PoC Scenario Framework
- [2026-01-26]: Added configuration-driven PoC scenario framework (src/poc/).
- [2026-01-26]: Three built-in scenarios: transfer, complexity, stability.
- [2026-01-26]: Pydantic-validated configs with no hardcoded values.
- [2026-01-26]: Structured logging via structlog throughout.
- [2026-01-26]: C4 architecture documentation in docs/architecture/c4_model.md.
- [2026-01-26]: Added comprehensive C4 architecture in Mermaid format (docs/architecture/c4_mermaid.md).

## Validation Framework
- [2026-01-26]: Added comprehensive validation framework (src/validation/).
- [2026-01-26]: Four validation scenarios: tolerance_fix, gpu_training, transfer_validation, merge_readiness.
- [2026-01-26]: Tolerance helpers with dtype-aware precision handling.
- [2026-01-26]: GPU training validation with AMP support.
- [2026-01-26]: Zero-shot transfer verification (9x9 → 19x19).
- [2026-01-26]: PR merge readiness checker with configurable requirements.

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

## Physics PoC Commands
```bash
# Train physics operator on Poisson data (supervised learning)
python -m src.experiments.train_physics

# Custom training configuration
python -m src.experiments.train_physics --train-size 9 --eval-size 19 --n-epochs 100

# Verify zero-shot transfer (train 9x9 → eval 9,13,19)
python -m src.experiments.verify_transfer

# Verify with existing model
python -m src.experiments.verify_transfer --model-path outputs/physics_poc/best_model.pt

# Run FNet vs Softmax speed benchmark
python -m src.experiments.benchmark_fnet

# Benchmark with custom sizes
python -m src.experiments.benchmark_fnet --sizes 81,169,361,625 --batch-size 64

# Run Fredholm integral property tests
pytest tests/math_kernel/test_fredholm.py -v
```

## PoC Scenario Framework Commands
```bash
# List available scenarios
python -m src.poc.cli list

# Show scenario details
python -m src.poc.cli info transfer

# Run all scenarios
python -m src.poc.cli run

# Run specific scenario
python -m src.poc.cli run --scenario transfer

# Run from config file (full suite)
python -m src.poc.cli run --config config/scenarios/poc_full.yaml

# Run quick validation suite
python -m src.poc.cli run --config config/scenarios/poc_quick.yaml

# Run with parallel workers
python -m src.poc.cli run --parallel 4

# Compare two runs
python -m src.poc.cli compare run_a run_b

# PoC framework unit tests
pytest tests/poc/ -v
```

## Validation Framework Commands
```bash
# Run all validations (default config)
python -m src.validation.cli run

# Run specific validation
python -m src.validation.cli run --only gpu_training
python -m src.validation.cli run --only transfer
python -m src.validation.cli run --only tolerance
python -m src.validation.cli run --only merge_readiness

# Run with custom config
python -m src.validation.cli run --config config/validation/default.yaml

# Quick validation (reduced scope)
python -m src.validation.cli run --config config/validation/quick.yaml

# Run in parallel
python -m src.validation.cli run --parallel

# Stop on first failure
python -m src.validation.cli run --stop-on-failure

# Check PR merge readiness
python -m src.validation.cli merge-check --pr 7
python -m src.validation.cli merge-check --pr 7 --allow-failures 5

# Analyze tolerance issues in tests
python -m src.validation.cli tolerance-check
python -m src.validation.cli tolerance-check --test-dir tests/math_kernel/

# Show/generate configuration
python -m src.validation.cli config --show
python -m src.validation.cli config --generate config/validation/custom.yaml

# Validation framework tests
pytest tests/validation/ -v
```

## Directory Structure
```
src/
  modeling/     - Neural architectures and layers
  math_kernel/  - Basis functions, integral approximations
  mcts/         - Monte Carlo Tree Search logic
  tools/        - Verification and utility scripts
  training/     - Training infrastructure
    loss.py           - AlphaGalerkinLoss (policy + value + LBB)
    replay_buffer.py  - Uniform and prioritized replay buffers
    self_play.py      - MCTS-based self-play game generation
    trainer.py        - Main Trainer class
    checkpoint.py     - Checkpoint save/load management
    evaluation.py     - Win rate and policy agreement metrics
  data/         - Data loading and preprocessing
    dataset.py        - PyTorch Dataset classes
    collate.py        - Variable board size collation
  physics/      - Synthetic physics data generation
    poisson.py        - Poisson equation solver (DST-based)
  experiments/  - Physics PoC experiments
    physics_model.py  - PhysicsOperator neural network
    train_physics.py  - Supervised learning on Poisson data
    verify_transfer.py - Zero-shot transfer verification
    benchmark_fnet.py - FNet O(N log N) speed benchmark
  poc/          - PoC scenario framework
    config.py         - Pydantic configuration schemas
    registry.py       - Scenario registration and discovery
    runner.py         - Scenario execution engine
    results.py        - Result collection and persistence
    logging.py        - Structured logging utilities
    cli.py            - CLI entry point
    scenarios/        - Built-in scenario implementations
      transfer.py     - Zero-shot transfer scenario
      complexity.py   - O(N) complexity benchmark
      stability.py    - LBB stability monitoring
  validation/   - Validation framework for next steps
    config.py         - Pydantic configuration schemas
    tolerance.py      - Tolerance/precision helpers
    logging.py        - Structured logging utilities
    runner.py         - Validation orchestration
    cli.py            - CLI entry point
    scenarios/        - Validation scenario implementations
      base.py         - Base validator class
      gpu_training.py - GPU training validation
      transfer.py     - Zero-shot transfer verification
      tolerance_fixer.py - Tolerance issue analyzer
      merge_readiness.py - PR merge readiness checker
tests/
  math_kernel/  - Property-based tests for mathematical operators
    test_fredholm.py  - Fredholm integral equation tests
  training/     - Tests for training infrastructure
  integration/  - End-to-end integration tests
  poc/          - PoC framework tests
    test_config.py    - Configuration validation tests
    test_registry.py  - Scenario registration tests
    test_runner.py    - Runner execution tests
    test_results.py   - Result collection tests
  validation/   - Validation framework tests
    test_config.py    - Configuration validation tests
    test_tolerance.py - Tolerance helper tests
    test_scenarios.py - Scenario tests
    test_runner.py    - Runner execution tests
config/         - Hydra/Pydantic configuration schemas
  train.yaml          - Default training config
  train_fast.yaml     - Fast test config
  scenarios/          - PoC scenario configurations
    poc_full.yaml     - Full PoC suite
    poc_quick.yaml    - Quick validation suite
    transfer_ablation.yaml - Transfer ablation study
  validation/         - Validation framework configurations
    default.yaml      - Default validation config
    quick.yaml        - Quick validation config
docs/           - Documentation
  architecture/       - C4 architecture diagrams
    c4_model.md       - C4 model documentation
  PROMPT_TEMPLATE.md  - Agentic coding prompt template
scripts/        - CLI entry points
  train.py            - Training CLI with Hydra
```
