# Changelog

All notable changes to AlphaGalerkin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-27

### Summary

This release focuses on CI/CD stabilisation, dead-code removal, eliminating
all hard-coded magic numbers, and lifting branch coverage to 85 %.  It also
closes the gap between the distributed-training configuration and the tests
that exercise it, and introduces the `PDEGameAdapter` that bridges PDE games
to the generic MCTS engine.

### Added

- **PDEGameAdapter** (`src/pde/mcts_adapter.py`) — bridges `PDEGame` to the
  `MCTS.GameInterface` protocol; maps error reduction to `{-1, 0, +1}` reward;
  exposes `reset()`, `current_error`, and `error_reduction` helpers
- **`_get_float_attr()` helper** — safe float extraction from Pydantic configs
  or mock objects (prevents silent `float(MagicMock()) == 1.0` bugs)
- **Trainer branch tests** (`tests/training/test_trainer_branches.py`) — 46
  new targeted tests for previously uncovered branches in `trainer.py`
- **Configurable LBB parameters** (`config/schemas.py`):
  `lbb_eps`, `lbb_target`, `lbb_log_penalty_weight`
- **Configurable Elo thresholds** (`config/schemas.py`):
  `elo_win_threshold` (0.55), `elo_loss_threshold` (0.45)
- **Configurable PER beta increment** (`config/schemas.py`):
  `per_beta_increment` (0.001)
- **Configurable PDE thresholds** (`src/pde/config.py`):
  `good_reduction_threshold` (0.1), `poor_reduction_threshold` (0.5),
  `explore_error_threshold` (0.1)
- **`DistributedInfraConfig` extensions** (`src/distributed/config.py`):
  `gradient_compression`, `learning_rate_scaling`, `launcher` field,
  `should_sync_at_step()`, `should_save_checkpoint()`,
  `requires_barrier_before_checkpoint()`, `scale_learning_rate()`,
  `get_node_rank()`
- **`SelfPlayDistributedConfig` extensions**: `num_workers`, `batch_size`,
  `total_games` property, `get_games_for_worker()`
- **`create_distributed_config()` improvements**: accepts `enabled` override
  kwarg; derives `world_size` from launcher when provided

### Changed

- **`AlphaGalerkinLoss`** (`src/training/loss.py`) — `log_penalty_weight`
  parameter wired from config; `lbb_eps`/`lbb_target` sourced from config
- **`Trainer`** (`src/training/trainer.py`) — Elo win/loss thresholds sourced
  from config (`elo_win_threshold`, `elo_loss_threshold`)
- **`create_replay_buffer()`** (`src/training/replay_buffer.py`) — threads
  `beta_increment` through from config instead of hard-coding
- **`BasisSelectionGame`** (`src/pde/games/basis_selection.py`) — uses
  `config.explore_error_threshold` instead of the literal `0.1`
- **`.gitignore`** — extended with ML artefact patterns: `*.safetensors`,
  `*.onnx`, `*.agk`, `vertex_outputs/`, `benchmark_results/`,
  distributed-training artefacts, and temporary test output directories

### Fixed

- **`PoissonOperator.residual()`** (and three sibling operators) in
  `src/pde/operators.py` — unconditionally called `compute_derivatives()`
  even when the flag was `False`, raising `RuntimeError` on tensors without
  `grad_fn`
- **`PhysicsLoss._compute_laplacian()`** (`src/experiments/physics_model.py`)
  — for linear functions the first-order gradient is constant (no `grad_fn`);
  the second `autograd.grad` call is now skipped correctly
- **Chess insufficient-material check** (`src/games/chess.py:906`) — replaced
  bare `pass` stub with `return True`
- **Quantizer forward pass** (`src/video_compression/models/quantizer.py:43`)
  — replaced empty `pass` stub with `return x`
- **`tests/tools/test_cli.py`** — `patch("sys.exit")` missing `as mock_exit`
  causing `NameError`
- **`tests/mcts/test_node.py`**, **`test_search.py`** — tree pruning tests
  captured child reference before `prune_except`/`advance`, preventing dangling-reference failures
- **`tests/games/test_chess.py`** — `test_illegal_move_notation` used notation
  that is legal (`e1e8`); changed to genuinely invalid strings (`z9z9`, `ab`)
- **`tests/data/test_dataset.py`** — `DataLoader` collation for `Experience`
  dataclass now passes `collate_fn=list`
- **`tests/data/test_collate.py`** — added `@pytest.mark.skipif(not cuda)`
  decorators; removed duplicate `import torch`
- **`tests/distributed/test_launcher.py`** — removed `spec=subprocess.Popen`
  from mock that failed when `Popen` was itself patched
- **`tests/distributed/test_multiprocess.py`** — updated assertions to match
  `from_environment()` returning `tuple[int, int, int]` instead of a config object
- **`tests/training/test_self_play.py`** — added CUDA skip decorators; fixed
  multiprocessing fallback test to trigger `RuntimeError` via `share_memory`
- **`tests/training/test_extended_config.py`** — wrapped omegaconf import in
  `try/except` to gracefully skip when antlr4 version is incompatible
- **`tests/vertex/test_auth.py`** — fixed gcloud path mocking at construction
  time; skips when `google.auth` is unavailable
- **`tests/integration/test_video_workflow.py`** — skips gracefully when `cv2`
  is not installed
- **`tests/poc/test_scenarios_transfer.py`** — registry isolated per-test via
  `autouse` fixture to prevent ordering-dependent failures

### Coverage

- **85 % branch coverage** (up from ~75 %) — measured with `--cov-branch`
- **5 322 tests passing**, 154 skipped (environment-dependent: CUDA,
  `google.auth`, `cv2`, `omegaconf`/antlr4)
- **0 CI failures**

---

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
  - `tests/games/test_chess_exhaustive.py` — Exhaustive encode/decode roundtrip + edge cases (20 tests)
  - `tests/training/test_trainer_chess.py` — Checkpoint save/load/resume, engine eval, config tests (11 tests)
  - `tests/security/test_chess_security.py` — Invalid actions, OOB states, corrupted data (15 tests)
  - `tests/e2e/test_chess_training_e2e.py` — E2E training smoke tests (3 tests)

- **Stockfish Benchmark Evaluation**
  - Engine eval config fields in `TrainingConfig` (path, depth, games, movetime)
  - `Trainer._run_engine_evaluation()` with W&B Elo metric logging
  - Engine eval section in `config/train_chess.yaml`

- **CI/CD Chess Pipeline**
  - Stage 8: Chess Pipeline Tests in `.github/workflows/ci.yml`
  - Coverage gate `--cov-fail-under=80` for `chess.py` (97%) and `wrapper.py` (100%)
  - CI Success gate requires chess tests

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
