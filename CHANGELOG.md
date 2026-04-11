# Changelog

All notable changes to AlphaGalerkin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — E2E Dashboard (`dashboard/`)

- **`dashboard/app.py`** — Gradio Blocks application factory (`build_app()`) and CLI entry point (`main()`). Launches a tabbed UI exposing all AlphaGalerkin capabilities at `http://localhost:7860`. Accepts `--host`, `--port`, `--share`, `--debug` flags.

- **`dashboard/config.py`** — Full Pydantic v2 config hierarchy eliminating every hardcoded value:
  `AppConfig`, `GameConfig`, `PDEConfig`, `ComplexityRunConfig`, `StabilityRunConfig`,
  `TransferMilestone`, `PoCConfig`, `TrainingConfig`, `DashboardConfig`.
  `DEFAULT_CONFIG` singleton for zero-configuration startup.

- **`dashboard/utils.py`** — Shared utility module:
  - `fig_to_pil()` — always closes matplotlib figure (even on exception); `.copy()` detaches from buffer
  - `device_str()` — CUDA/CPU detection with graceful fallback
  - `format_exc()` — consistent exception formatting
  - `configure_structlog()` — idempotent structured logging setup

- **`dashboard/tabs/game_tab.py`** — Go AI tab. Thread-safe lazy model loading via `threading.Lock` (double-checked locking). Human vs AI and AI vs AI modes with 9×9/13×13/19×19 board support (zero-shot transfer). Config-injected via `GameConfig`.

- **`dashboard/tabs/pde_tab.py`** — Interactive Poisson equation solver. Five charge patterns (Point Charge, Dipole, Quadrupole, Ring, Random), multi-resolution comparison with zoom-upsampling MSE. Config-injected via `PDEConfig`.

- **`dashboard/tabs/poc_tab.py`** — PoC scenario runner. O(N) complexity benchmark, LBB stability monitoring, zero-shot transfer milestone display. Module-level optional imports for test patchability. Config-injected via `PoCConfig`.

- **`dashboard/tabs/training_tab.py`** — Architecture summary, simulated training curves (policy/value/LBB losses), and loss breakdown diagram. Config-injected via `TrainingConfig`.

### Added — Dashboard Test Suite (`tests/dashboard/`)

- **203 tests**, **89% line coverage** (gate: 85%), all passing with zero ruff violations.
- `conftest.py` — shared fixtures, `matplotlib.use("Agg")`, config fixture hierarchy, mock scenario results, charge-grid fixtures.
- `test_app.py` (24 tests) — CSS builder, arg parser, `build_app()`, `main()`.
- `test_config.py` (31 tests) — all Pydantic models, validation errors, JSON round-trip.
- `test_utils.py` (24 tests) — `fig_to_pil` (close on error, detached buffer), `device_str`, `format_exc`, `configure_structlog`.
- `test_pde_tab.py` (37 tests) — all charge patterns, Poisson solve integration, `solve_and_visualize`, `compare_resolutions` with shape-matching mock.
- `test_poc_tab.py` (32 tests) — `_parse_int_list`, `run_complexity`, `run_stability` (mocked), `show_transfer_milestone` (live).
- `test_training_tab.py` (28 tests) — model summary (fallback on import error), training curves, loss breakdown.
- `test_game_tab.py` (27 tests) — `autouse` fixture resetting module globals, fallback board, `_ensure_loaded` idempotency, human/AI move handlers.

- **Intercept Module** (`src/intercept/`)
  - `InterceptGame` implementing `GameInterface` protocol for MCTS-guided missile defense
  - 6-DOF rigid body dynamics (`dynamics.py`, `interceptor_dynamics.py`)
  - Proportional Navigation guidance (`guidance.py`)
  - `ExtendedKalmanFilter` for target tracking (`tracking.py`)
  - `RadarSensor`, `SensorFusion` for multi-sensor tracking (`sensors.py`)
  - `HungarianAssigner` for weapon-target assignment (`assignment.py`)
  - `ISAAtmosphere`, `WindModel` for atmospheric modeling (`atmosphere.py`)
  - `AeroModel`, `TabularAeroModel` for aerodynamic coefficients (`aero.py`)
  - `FrameTransform`, `QuaternionOps` for reference frame conversions (`frames.py`)
  - Pydantic-validated `InterceptorConfig`, `EngagementConfig`, `ThreatConfig`

- **Backend Abstraction** (`src/backend/`)
  - `BackendInterface` protocol for unified PyTorch/JAX operations
  - `TorchBackend`, `JaxBackend` implementations
  - `Array`, `Precision`, `DeviceType` type abstractions (`types.py`)
  - Random number generator abstraction (`rng.py`)
  - Backend-aware logging and debug utilities

- **Prototyping Module** (`src/prototyping/`)
  - `ModelBuilder`, `PrototypeModel` for rapid architecture iteration
  - `QuickTrainer`, `TrainResult` for fast experiment loops
  - `QuickEvaluator`, `EvalResult` for quick model evaluation
  - `DataGenerator`, `SyntheticData` for synthetic data creation
  - `Visualizer` with multiple plot types
  - `ExperimentTemplate`, `TemplateRegistry` for experiment patterns

- **Analysis Module** (`src/analysis/`)
  - `PositionEvaluator`, `EvaluationResult` for position evaluation
  - `GameReviewer`, `MoveAnalysis` for game review and move quality assessment
  - `PatternMatcher`, `PatternLibrary` for board pattern detection
  - `GameStatistics`, `StatisticsCollector` for game statistics aggregation
  - `AnalysisConfig`, `AnalysisMode` Pydantic configuration

- **Tournament Module** (`src/tournament/`)
  - `TournamentManager`, `TournamentState` supporting Round-Robin, Swiss, Elimination formats
  - `TournamentScheduler` for match scheduling
  - `EloRating`, `RatingSystem` for player rating computation
  - `Player`, `PlayerRegistry` for participant management
  - `Match`, `MatchResult`, `MatchStatus` for match tracking

### Changed

- **`pyproject.toml`** — Added `[[tool.mypy.overrides]]` for `dashboard.*` modules (relaxed strict checks for Gradio code). Added `[tool.coverage.report]` with `fail_under = 85` and `show_missing = true`. Added `dashboard` pytest marker.
- **Gradio 6 compatibility** — CSS argument moved from `Blocks()` constructor to `launch()`.

> **Branch and PR cleanup** — removed 28 stale remote branches and 6 open stale PRs.

## [0.3.0] - 2026-04-01

### Summary

Key highlights of this release:

- **Chess Self-Play Training Pipeline** — AlphaZero methodology, 4672-action dense policy, 119-channel state encoding
- **SBIR Readiness Infrastructure** — Navy N252-088, DOE ASCR, NSF SBIR, AFWERX proposal configs and benchmark suite
- **Advanced PDE Operators** — NavierStokes (Taylor-Green), L-shaped Poisson (singularity), enhanced Burgers (Cole-Hopf)
- **Domain Geometry & Time-Stepping module** — Rectangular, L-shaped, Cylinder domains; ForwardEuler, RK4, CrankNicolson
- **Multi-Agent Swarm Planning** — PettingZoo `ParallelEnv` adapter, potential field obstacle avoidance
- **Unified Loss Package & BaseTrainer consolidation** — `LossRegistry`, `get_loss()` factory, shared AMP/grad/LR in `BaseTrainer`
- **CI/CD hardening** — 85% coverage gates, nightly schedule, Stage 8 chess pipeline
- **218+ new tests** across PDE, research, training, and games modules

---

### Added

- **SBIR Readiness Infrastructure** (Navy N252-088, DOE ASCR, NSF, AFWERX)
  - `config/proposals/navy_n252_088.yaml`, `nsf_sbir.yaml` — SBIR-specific benchmark configs
  - `config/benchmarks/sbir_suite.yaml` — 3-problem benchmark suite (L-shaped Poisson, Burgers shock, NS Taylor-Green)
  - `src/research/baselines.py` — Classical PDE solver baselines: UniformFDMSolver, DorflerAMRSolver, SimplePINNSolver
  - `src/research/pde_benchmarks.py` — PDEBenchmarkRunner with JSON/Markdown report generation and convergence rate computation
  - `docs/proposals/templates/sbir_phase1.md` — Reusable SBIR Phase I proposal template
  - `docs/proposals/IP_STRATEGY.md` — 3 provisional patent claims, trade secret boundaries, publication plan

- **Advanced PDE Operators**
  - `NavierStokesOperator` — Taylor-Green vortex benchmark with analytical solution, configurable Re
  - `BurgersOperator` enhanced — Cole-Hopf exact solution, configurable shock params, convergence rate method
  - `LShapedPoissonOperator` — r^(2/3)*sin(2theta/3) singularity for AMR benchmarking

- **Domain Geometry Abstractions** (`src/pde/geometry.py`)
  - `RectangularDomain`, `LShapedDomain`, `CylinderFlowDomain` (DFG benchmark)
  - Rejection sampling for non-convex domains, proportional boundary sampling
  - `GeometryConfig` Pydantic schema and `create_geometry()` factory

- **Time-Stepping Module** (`src/pde/time_stepping.py`)
  - `ForwardEuler`, `RK4`, `CrankNicolson` (fixed-point iteration) with factory pattern
  - `TimeSteppingConfig` Pydantic schema, `integrate()` with snapshot saving

- **S500 Swarm Planning Game** (`src/pde/games/swarm_planning.py`)
  - `SwarmPlanningGame` with round-robin multi-agent control (7 actions per agent)
  - Potential field obstacle avoidance (Laplace equation connection), coverage rewards
  - `SwarmPlanningConfig` — fully Pydantic-validated with no hardcoded values

- **PettingZoo Adapter** (`src/games/pettingzoo_adapter.py`)
  - `PettingZooAdapter` wrapping `GameInterface` as PettingZoo `ParallelEnv`
  - Optional dependency with graceful degradation (`HAS_PETTINGZOO` flag)

- **Unified Loss Package** (`src/training/losses/`)
  - `LossRegistry` with decorator-based registration (`"alphagalerkin"`, `"l2_relative"`, `"h1"`, `"mse"`)
  - `get_loss()` factory function for config-driven loss instantiation
  - Backwards-compatible thin wrappers in `src/training/loss.py` and `src/training/physics_loss.py`

- **BaseTrainer Consolidation** (`src/training/base_trainer.py`)
  - Abstract `BaseTrainer[ConfigT]` with shared AMP, gradient clipping, LR scheduling, checkpoint save/load
  - `BaseTrainerConfig` Pydantic schema covering all shared hyperparameters
  - `StepResult` dataclass for structured step output

- **Checkpoint Migration System** (`src/training/checkpoint_migration.py`)
  - Version-aware migration with `@register_migration` decorator
  - Migration path: `0.0.0 -> 1.0.0 -> 1.1.0` (LBB config fields added)

- **Property-Based and Numerical Stability Tests**
  - `tests/training/test_loss_properties.py` — hypothesis tests: non-negativity, CE = log(n), gradient flow
  - `tests/training/test_numerical_stability.py` — extreme values, near-zero denominators, NaN propagation
  - `tests/pde/test_operator_properties.py` — PDE operator invariants, linearity, collocation in domain
  - `tests/modeling/test_attention_properties.py` — Galerkin attention shape, LBB positivity, resolution independence

- **Comprehensive Coverage Tests** (218 new tests)
  - `tests/pde/test_geometry.py` — 65 tests for domain geometries
  - `tests/pde/test_time_stepping.py` — 37 tests for time-stepping methods
  - `tests/research/test_baselines.py` — 39 tests for classical solver baselines
  - `tests/research/test_pde_benchmarks.py` — 38 tests for benchmark runner
  - `tests/training/test_base_trainer.py` — 39 tests for BaseTrainer
  - `tests/pde/test_swarm_planning.py` — 50 tests for swarm planning game
  - `tests/games/test_pettingzoo_adapter.py` — 11 tests for PettingZoo adapter

### Changed

- **CI/CD Hardening** (`.github/workflows/ci.yml`)
  - MyPy strict enforcement (`continue-on-error: false`)
  - Coverage gates raised: 75% -> 85% overall, 80% -> 85% per-module (pde, modeling, training)
  - Added `research` module coverage gate at 85%
  - Added nightly schedule (`cron: '0 4 * * *'`) and performance benchmark job on main merges

- **Config-Driven LBB Loss** (`config/schemas.py`)
  - Surfaced `lbb_loss_weight`, `lbb_target`, `lbb_eps`, `log_barrier_weight` as Pydantic fields
  - Added mathematical documentation (Babuska-Brezzi motivation) in field descriptions

- **Race Condition Fix** (`src/modeling/model.py`)
  - Removed `_training_resolution` mutation from `forward()` (DDP-unsafe)
  - Added explicit `set_training_resolution()` public method

### Fixed

- `advection_coeff` dimension mismatch in `PDEBenchmarkRunner._create_operator()` — was hardcoded `[0.0, 0.0]` for any dim

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
