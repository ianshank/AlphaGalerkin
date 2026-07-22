# AlphaGalerkin Components Reference

> **Last Updated:** 2026-04-10 | **Modules:** 25 | **Tests:** 5,100+

This document provides a reference for all modules in the AlphaGalerkin system, grouped by domain.

---

## 1. Core: Math Kernel & Modeling

### 1.1 Math Kernel (`src/math_kernel/`)

Continuous domain mathematics enabling resolution independence.

- **`basis.py`** — `FourierBasis`, `ChebyshevBasis`, `create_grid_coordinates`. Maps continuous coordinates to high-dimensional features via Random Fourier Features or Chebyshev polynomials. Supports both PyTorch and JAX backends.
- **`integral.py`** — `GalerkinProjection`, `MonteCarloIntegral`, `PetrovGalerkinProjection`. Core Galerkin integral operators.
- **`spectral.py`** — `SpectralFilter`, `ResolutionAdapter`. Anti-aliasing and resolution transfer logic.

### 1.2 Modeling (`src/modeling/`)

Neural network architecture — the core innovation.

- **`attention.py`** — `GalerkinAttention` (O(N) Petrov-Galerkin projection), `SoftmaxAttention` (O(N^2) precision), `HybridAttention` (learnable gate between both).
- **`model.py`** — `AlphaGalerkinModel`. Main architecture: Continuous Embedding -> Strategy Body (Galerkin) -> FNet Mixing -> Tactical Head (Softmax) -> Policy/Value Heads.
- **`fnet.py`** — `FNetBlock`. FFT-based O(N log N) spatial mixing for fast MCTS rollouts.
- **`fno_layer.py`** — Fourier Neural Operator layers.
- **`embeddings.py`** — `ContinuousEmbedding`, `FourierFeatures`.
- **`multiscale_fourier.py`** — `MultiScaleFourierFeatures`, `AdaptiveFourierFeatures`, `ProgressiveFourierFeatures` for spectral bias mitigation.
- **`stability.py`** — `StabilityGuard`. Monitors LBB inf-sup condition via singular values.

---

## 2. Search & Games

### 2.1 MCTS (`src/mcts/`)

Monte Carlo Tree Search with neural evaluation.

- **`search.py`** — Core `MCTS` class with tree search and neural leaf evaluation.
- **`node.py`** — `MCTSNode` tree structure.
- **`evaluator.py`** — `ModelEvaluator`, `FNetEvaluator` for rollout evaluation.
- **`gumbel.py`** — Gumbel AlphaZero implementation with sequential halving and improved policy targets.

### 2.2 Games (`src/games/`)

Multi-game support with abstract interface.

- **`interface.py`** — `GameInterface` abstract base defining action space, state encoding, legal moves, symmetries.
- **`registry.py`** — `GameRegistry` with `@register_game` decorator for game discovery.
- **`go.py`** — Full Go rules (Chinese scoring, superko, 8-fold symmetry).
- **`chess.py`** — Full Chess rules (castling, en passant, promotion, 119-plane AlphaZero encoding, 4672-action space).
- **`wrapper.py`** — `StatefulGameWrapper` bridging stateless `GameInterface` to MCTS protocol.
- **`pettingzoo_adapter.py`** — `PettingZooAdapter` wrapping `GameInterface` as PettingZoo `ParallelEnv`.

### 2.3 Engines (`src/engines/`)

External engine integration for benchmarking.

- **`uci.py`** — `UCIEngine` subprocess-based UCI protocol for Stockfish/other engines.
- **`adapter.py`** — `EngineEvaluator` bridging UCI engines to MCTS Evaluator protocol.
- **`match.py`** — `EngineMatch` orchestration with color alternation and PGN output.
- **`elo.py`** — `EloCalculator` with confidence intervals.

### 2.4 Tournament (`src/tournament/`)

Tournament management and rating.

- **`manager.py`** — `TournamentManager` supporting Round-Robin, Swiss, and Elimination formats.
- **`rating.py`** — `EloRating`, `RatingSystem`.
- **`match.py`** — `Match`, `MatchResult`, `MatchStatus`.
- **`player.py`** — `Player`, `PlayerRegistry`.

---

## 3. PDE & Physics

### 3.1 PDE Game Framework (`src/pde/`)

PDE solving as sequential decision-making for MCTS-guided Galerkin approximation.

- **`game.py`** — `PDEGame`, `PDEState`, `PDEResult` abstractions.
- **`operators.py`** — `PoissonOperator`, `BurgersOperator`, `NavierStokesOperator`, `AdvectionDiffusionOperator`, `LShapedPoissonOperator`.
- **`geometry.py`** — `RectangularDomain`, `LShapedDomain`, `CylinderFlowDomain` with rejection sampling.
- **`time_stepping.py`** — `ForwardEuler`, `RK4`, `CrankNicolson` with factory pattern.
- **`game_interface.py`** — `PDEGameInterface` bridging PDE games to `GameRegistry`.
- **`register_games.py`** — Auto-registration of `pde_basis` and `pde_mesh` games.
- **`games/basis_selection.py`** — Galerkin basis selection game.
- **`games/mesh_refinement.py`** — Adaptive h/p-refinement game.

### 3.2 Physics (`src/physics/`)

Synthetic physics data generation.

- **`poisson.py`** — `PoissonSolver`, `PoissonDataset` (DST-based).
- **`solver.py`** — Generic solver interface.
- **`darcy.py`**, **`heat.py`**, **`elasticity.py`** — Additional PDE solvers.

### 3.3 Agents (`src/agents/`)

Multi-physics PDE solving with specialized sub-agents.

- **`orchestrator.py`** — `AgentOrchestrator` entry point.
- **`solver.py`** — `SolverAgent` wrapping PDEGame + MCTS.
- **`decomposition.py`** — `DecompositionAgent` splitting coupled systems.
- **`coupling.py`** — `CouplingAgent` enforcing interface conditions.
- **`meta.py`** — `MetaAgent` coordinating the full pipeline.
- **`message.py`** — `AgentMessage`, `MessageBus` for inter-agent communication.

---

## 4. Training

### 4.1 Training Infrastructure (`src/training/`)

Self-play, losses, replay buffer, checkpointing, curriculum.

- **`trainer.py`** — Main `Trainer` class (game-agnostic).
- **`base_trainer.py`** — `BaseTrainer[ConfigT]` with shared AMP, gradient clipping, LR scheduling.
- **`loss.py`** — `AlphaGalerkinLoss` (policy_CE + value_MSE + LBB_regularization).
- **`physics_loss.py`** — `CombinedAlphaGalerkinPhysicsLoss`, `ResidualLoss`, `BoundaryLoss`, `ConservationLoss`.
- **`loss_balancing.py`** — `ReLoBRaLo`, `GradNorm`, `SoftAdapt`, `UncertaintyWeighting`.
- **`losses/`** — `LossRegistry` with decorator-based registration and `get_loss()` factory.
- **`replay_buffer.py`** — Uniform and `PrioritizedReplayBuffer`.
- **`self_play.py`** — MCTS-based self-play game generation (parallel workers).
- **`checkpoint.py`** — `CheckpointManager` with version-aware migration.
- **`evaluation.py`** — Win rate and policy agreement metrics.

### 4.2 Curriculum (`src/curriculum/`)

Progressive training with stage transitions.

- **`manager.py`** — `CurriculumManager`.
- **`scheduler.py`** — `CurriculumScheduler`.
- **`stage.py`** — `CurriculumStage`, `StageStatus`.

### 4.3 Distributed Training (`src/distributed/`)

Multi-node DDP training with NCCL gradient sync.

- **`trainer.py`** — `DistributedTrainer` with DDP.
- **`gradient_sync.py`** — `GradientSynchronizer` (NCCL-based).
- **`launcher.py`** — `DistributedLauncher` (torchrun, SLURM).
- **`worker.py`** — `SelfPlayCoordinator` for distributed self-play.
- **`model_zoo.py`** — `ModelZoo` for checkpoint management.

---

## 5. Deployment & Cloud

### 5.1 ONNX Deployment (`src/deployment/`)

Model export and edge deployment.

- **`export_onnx.py`** — `ONNXExporter` with dynamic shape support.
- **`quantize.py`** — `ModelQuantizer` (dynamic/static INT8).
- **`runtime.py`** — ONNX Runtime inference wrapper with multi-provider support.
- **`validate.py`** — `ModelValidator` against PyTorch outputs.

---

## 6. Research & Validation

### 6.1 PoC Scenario Framework (`src/poc/`)

Configuration-driven validation and benchmarking.

- **`runner.py`** — `ScenarioRunner` execution engine.
- **`registry.py`** — `ScenarioRegistry` with `@scenario` decorator.
- **`scenarios/`** — `TransferScenario`, `ComplexityScenario`, `StabilityScenario`.
- **`tuning/`** — `HyperparameterTuner` with TPE, grid, and random samplers.
- **`statistics/`** — `StatisticalAnalyzer` with t-test, Mann-Whitney, bootstrap, effect sizes.
- **`visualization/`** — `PlotRegistry` (5 plot types), `HTMLReportGenerator`.

### 6.2 Research (`src/research/`)

Experiment tracking, baselines, benchmarking.

- **`experiment.py`** — `ExperimentTracker`.
- **`benchmark.py`** — `BenchmarkSuite`, `BenchmarkResult`.
- **`baselines.py`** — Classical solver baselines (FDM, Dorfler AMR, PINN).
- **`pde_benchmarks.py`** — `PDEBenchmarkRunner` with JSON/Markdown reports.
- **`reporter.py`** — `Reporter` (HTML/JSON/Markdown formats).

### 6.3 Demos (`src/demos/`)

Interactive demonstrations (Hugging Face Spaces).

- **`physics_demo.py`** — Zero-shot transfer visualization.
- **`benchmark_demo.py`** — Performance benchmarking.
- **`sbir_demo.py`** — SBIR benchmark demo with HTML report generation.
- **`architecture_demo.py`** — Attention visualization.

### 6.4 Experiments (`src/experiments/`)

Physics PoC validation scripts.

- **`train_physics.py`** — Supervised learning on Poisson data.
- **`verify_transfer.py`** — Zero-shot transfer verification (9x9 -> 19x19).
- **`benchmark_fnet.py`** — FNet vs Softmax speed comparison.
- **`physics_model.py`** — `PhysicsOperator` neural network.

---

## 7. Infrastructure

### 7.1 Templates (`src/templates/`)

Reusable module development infrastructure.

- **`config.py`** — `BaseModuleConfig`, `create_config_class()` factory.
- **`registry.py`** — `BaseRegistry`, `create_registry()` factory.
- **`logging.py`** — `BaseModuleLogger`, `create_logger_class()`.
- **`base.py`** — `BaseExecutable`, `ExecutionResult`.

### 7.2 Backend (`src/backend/`)

PyTorch/JAX unified interface.

- **`interface.py`** — `BackendInterface` protocol.
- **`torch_backend.py`**, **`jax_backend.py`** — Backend implementations.
- **`types.py`** — `Array`, `Precision`, `DeviceType` enums.
- **`rng.py`** — Random number generator abstraction.

### 7.3 Data (`src/data/`)

Data loading and preprocessing.

- **`dataset.py`** — Base Dataset classes.
- **`physics_dataset.py`** — `PhysicsDataset`.
- **`collate.py`** — Variable board size collation (game-agnostic).

### 7.4 Tools (`src/tools/`)

Utility and verification scripts.

- **`verify_invariance.py`** — Resolution invariance verification.
- **`gtp.py`** — Go Text Protocol implementation.
- **`cli.py`** — CLI entry point.

### 7.5 Analysis (`src/analysis/`)

Game analysis and review.

- **`evaluator.py`** — `PositionEvaluator`, `EvaluationResult`.
- **`reviewer.py`** — `GameReviewer`, `MoveAnalysis`.
- **`patterns.py`** — `PatternMatcher`, `PatternLibrary`.
- **`statistics.py`** — `GameStatistics`, `StatisticsCollector`.

### 7.6 Prototyping (`src/prototyping/`)

Rapid experimentation tools.

- **`builder.py`** — `ModelBuilder`, `PrototypeModel`.
- **`trainer.py`** — `QuickTrainer`, `TrainResult`.
- **`evaluator.py`** — `QuickEvaluator`, `EvalResult`.
- **`data.py`** — `DataGenerator`, `SyntheticData`.

---

## 9. Configuration (`config/`)

All configuration uses Pydantic schemas with Hydra CLI overrides.

- **`config/schemas.py`** — `OperatorConfig`, `TrainingConfig`, `MCTSConfig`.
- **`config/train.yaml`** — Default training config.
- **`config/train_chess.yaml`** — Chess-specific parameters.
- **`config/train_pde.yaml`** — PDE basis selection training.
- **`config/scenarios/`** — PoC scenario configurations.
- **`config/benchmarks/`** — SBIR benchmark suite.
- **`config/proposals/`** — SBIR proposal configurations.
