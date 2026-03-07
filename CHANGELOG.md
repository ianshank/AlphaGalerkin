# Changelog

All notable changes to AlphaGalerkin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Chess Self-Play Training Pipeline** (AlphaZero methodology)
  - `ActionPolicyHead` for dense 4672-action policy output (`src/modeling/model.py`)
  - `StatefulGameWrapper` bridging stateless `GameInterface` to MCTS (`src/games/wrapper.py`)
  - Chess training CLI (`scripts/train_chess.py`) with Hydra config (`config/train_chess.yaml`)
  - `game_type` and `action_space_size` fields in `OperatorConfig` (`config/schemas.py`)
  - PRD and ADR documentation (`docs/prd/prd-chess-self-play.md`, `docs/architecture/ADR-chess-self-play.md`)

- **Chess Training Tests**
  - `tests/games/test_wrapper.py` — StatefulGameWrapper unit tests (10 tests)
  - `tests/modeling/test_chess_model.py` — ActionPolicyHead and chess model tests (12 tests)
  - `tests/training/test_chess_self_play.py` — Chess self-play integration tests (7 tests)

### Changed

- **Game-agnostic self-play**: `SelfPlayWorker` now accepts optional `GameInterface` parameter
- **Game-agnostic trainer**: `Trainer.__init__()` accepts `game` parameter, forwarded to worker
- **Game-agnostic collator**: `VariableSizeCollator` and `SameSizeCollator` derive action mask size from `target_policy` tensor instead of hardcoded `board_size²+1`
- `AlphaGalerkinModel` and `AlphaGalerkinFast` auto-select policy head by `action_space_size`

### Fixed

- **Underpromotion encode/decode mismatch** (`src/games/chess.py`): `_decode_move` used `[-1, 0, 1]` but `_encode_move` used `straight=0, left=1, right=2` — straight promotion from column 0 decoded as `to_col=-1`. Fixed to `[0, -1, 1]`.
- **Collator action mask size** (`src/data/collate.py`): Both collators hardcoded `n_actions = board_size²+1` causing tensor size mismatch with chess's 4672-action policy. Fixed to detect per-experience policy encoding.

## [0.2.0] - 2026-01-26

### Milestones Achieved

- **Zero-Shot Transfer Validated**: Physics PoC demonstrated resolution-independence
  - Trained on 9x9 grids, achieved MSE 0.000209 on 19x19 grids (240x better than 0.05 threshold)
  - Validates core Galerkin approach for continuous operator learning

- **Training Pipeline Operational**: End-to-end training with self-play working on GPU
  - MCTS-based self-play generates training experiences
  - LBB stability monitoring integrated into training loop

### Added

- **W&B Integration for Physics PoC**
  - `--wandb` flag for `train_physics.py` to enable Weights & Biases logging
  - Logs training loss, evaluation MSE, transfer MSE, and learning rate
  - Final summary includes success status and best transfer MSE

- **GameInterface Protocol Implementation**
  - Added `apply_action()` method to `SimpleGoGame` class
  - Enables MCTS integration with Go game state

- **Security Tests** (`tests/security/`)
  - Input sanitization tests for GTP interface
  - DoS protection via input length limits

- **E2E Tests** (`tests/e2e/`)
  - CLI journey tests for help and train commands

### Changed

- Replaced Unicode checkmarks with ASCII `[PASS]`/`[FAIL]` for Windows compatibility
- Updated `.gitignore` with additional patterns:
  - `nul` (Windows device file)
  - `*.log`, `*.dist-info/`
  - `hydra_outputs/`

### Fixed

- Fixed `AttributeError: 'SimpleGoGame' object has no attribute 'apply_action'`
- Fixed unused loop variable warning in `BoardSizeBatchSampler`
- Fixed line length issue in W&B initialization

## [0.1.0] - 2026-01-26

### Added

- **Core Architecture**
  - `AlphaGalerkinModel`: Resolution-independent Go AI using continuous operators
  - `GalerkinLinearAttention`: O(N) complexity global influence modeling
  - `SoftmaxAttention`: Local tactical reading with injectivity preservation
  - `FNetBlock`: FFT-based mixing for fast MCTS rollouts

- **Mathematical Kernel**
  - Fredholm integral equation with Green's function formulation
  - Fourier features for positional encoding
  - Monte Carlo integral normalization (1/n) for Galerkin attention
  - LBB stability monitoring (dim(Key) >= dim(Query))

- **Training Infrastructure**
  - Self-play with MCTS for experience generation
  - Uniform and prioritized replay buffers
  - `AlphaGalerkinLoss`: policy_CE + value_MSE + LBB_regularization
  - Checkpoint management with best model tracking
  - Hydra configuration system

- **Physics PoC**
  - Poisson equation solver for synthetic data generation
  - `PhysicsOperator` neural network for influence field prediction
  - Zero-shot transfer verification scripts

- **PoC Scenario Framework**
  - Configuration-driven scenario execution
  - Built-in scenarios: transfer, complexity, stability
  - Pydantic-validated configs
  - Structured logging via structlog

### Documentation

- C4 architecture diagrams
- CLAUDE.md with project context and verification commands
