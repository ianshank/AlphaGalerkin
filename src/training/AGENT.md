# AGENT.md - Training Infrastructure Module (`src/training/`)

## Persona

**Name**: Training Engineer
**Expertise**: Deep learning training pipelines, loss function design, experience replay, self-play systems, distributed training, checkpoint management, curriculum learning
**Mindset**: You orchestrate the full training loop — from self-play game generation through loss computation to checkpoint persistence. Reliability, reproducibility, and stability monitoring are paramount.

## Module Overview

This module implements the complete AlphaGalerkin training pipeline: composite loss functions (policy + value + LBB + physics), adaptive loss balancing (5 strategies), prioritized experience replay, MCTS-driven self-play, curriculum learning over board sizes, checkpoint management with atomic saves, Elo tracking, Langfuse experiment tracking, and the main `Trainer` orchestration loop.

## Design Patterns

### 1. Strategy Pattern (Loss Balancing)
Five interchangeable loss balancing strategies behind a common `LossBalancer` interface:
- `StaticWeighting`: Fixed weights (baseline)
- `ReLoBRaLo`: Relative Loss Balancing with Random Lookback
- `GradNorm`: Gradient normalization across tasks
- `UncertaintyWeighting`: Homoscedastic uncertainty with learnable log-variance
- `SoftAdapt`: Rate-based adaptation prioritizing slower losses

### 2. Factory Pattern
- `create_loss_balancer(config, loss_names)` — instantiates the correct strategy
- `create_replay_buffer(capacity, prioritized)` — uniform vs prioritized buffer
- `create_tracker(langfuse_config, training_config)` — Langfuse experiment tracker
- `create_model_from_checkpoint(path)` — model reconstruction

### 3. Composite Pattern (Loss Composition)
Loss functions compose hierarchically:
```
CombinedAlphaGalerkinPhysicsLoss
  ├── AlphaGalerkinLoss (policy_CE + value_MSE + lbb_reg)
  └── PhysicsInformedLoss
        ├── ResidualLoss (PDE residual minimization)
        ├── BoundaryLoss (Dirichlet/Neumann/Robin)
        ├── InitialConditionLoss (time-dependent PDEs)
        └── ConservationLoss (integral conservation)
```

### 4. Dataclass State Containers
- `Experience`: Training sample (board state, policy target, value target)
- `GameRecord`: Complete self-play game with state/policy/action sequence
- `TrainingMetrics`: Per-step metrics for logging
- `CheckpointState`: Complete training state for persistence
- `LossOutput`, `PhysicsLossOutput`, `LossTerms`: Loss computation results
- `EloRating`: Checkpoint strength tracking

### 5. Thread-Safe Data Structures
- `UniformReplayBuffer` / `PrioritizedReplayBuffer`: `RLock`-protected circular buffers
- `SumTree`: Binary tree for O(log N) priority-based sampling
- `LangfuseTracker`: Thread-safe experiment tracking

### 6. Atomic Checkpoint Management
`CheckpointManager` writes to temp files first, then renames — preventing corruption from interrupted saves. Supports best model tracking, rotation, and version compatibility.

### 7. Training Loop (Trainer Lifecycle)
```
__init__() → train() { _fill_buffer → [_training_step → _run_evaluation → save_checkpoint] × N }
```
Setup is performed in `__init__()` and at the beginning of `train()`. There is no explicit `setup()`/`cleanup()` API.

## Skills Required

- **Multi-objective optimization**: Balancing policy, value, LBB, and physics losses simultaneously
- **Experience replay**: Priority sampling, importance sampling weights, beta annealing
- **Self-play systems**: MCTS game generation, perspective adjustment, variable board sizes
- **Curriculum learning**: Progressive board size introduction with weighted sampling
- **Distributed training**: DDP wrapping, gradient accumulation, rank-aware operations
- **Mixed precision**: AMP with GradScaler for memory efficiency
- **Checkpoint engineering**: Atomic saves, best model tracking, version compatibility
- **Experiment tracking**: Langfuse tracing, metric collection, checkpoint provenance

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **Loss Designer** | `loss.py`, `loss_balancing.py`, `physics_loss.py` | Adding new loss terms, tuning balancing |
| **Replay Buffer Engineer** | `replay_buffer.py` | Modifying sampling strategies, priority updates |
| **Self-Play Specialist** | `self_play.py` | Game generation, experience extraction |
| **Checkpoint Manager** | `checkpoint.py` | Save/load logic, migration, compatibility |
| **Curriculum Designer** | `curriculum.py` | Board size scheduling, transition logic |
| **Stability Monitor** | `stability.py` | Early stopping, plateau detection, gradient health |
| **Evaluation Specialist** | `evaluation.py`, `eval_utils/elo_tracker.py` | Win rate, policy agreement, Elo ratings |
| **Trainer Orchestrator** | `trainer.py`, `operator_trainer.py` | Main loop, distributed setup, Langfuse integration |

## Tools & Commands

```bash
# Run training tests
pytest tests/training/ -v

# Specific test areas
pytest tests/training/test_loss.py -v
pytest tests/training/test_loss_balancing.py -v
pytest tests/training/test_replay_buffer.py -v
pytest tests/training/test_checkpoint.py -v
pytest tests/training/test_curriculum.py -v
pytest tests/training/test_trainer.py -v
pytest tests/training/test_eval_utils.py -v

# Integration test
pytest tests/integration/ -v

# Start training
python -m scripts.train
python -m scripts.train --config-name=train_fast
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `loss.py` | Core training loss | `AlphaGalerkinLoss`, `LossOutput`, `EntropyRegularizer` |
| `loss_balancing.py` | 5 adaptive strategies | `LossBalancer` (ABC), `BalancingStrategy`, `LossBalancingConfig`, `LossTerms`, `ReLoBRaLo`, `GradNorm`, `UncertaintyWeighting`, `SoftAdapt`, `StaticWeighting` |
| `losses.py` | Foundational loss functions | `L2RelativeLoss`, `H1Loss`, `MSELoss`, `get_loss()` |
| `physics_loss.py` | Physics-informed loss | `PhysicsLossConfig`, `PhysicsLossOutput`, `ResidualLoss`, `BoundaryLoss`, `InitialConditionLoss`, `ConservationLoss`, `PhysicsInformedLoss`, `CombinedAlphaGalerkinPhysicsLoss` |
| `replay_buffer.py` | Experience storage | `UniformReplayBuffer`, `PrioritizedReplayBuffer`, `SumTree`, `Experience` |
| `checkpoint.py` | State persistence | `CheckpointManager`, `CheckpointState` |
| `self_play.py` | Game generation | `SelfPlayWorker`, `ParallelSelfPlayWorker`, `GameRecord` |
| `evaluation.py` | Model assessment | `Evaluator`, `EvaluationResult` |
| `stability.py` | Training monitoring | `EarlyStopping`, `PlateauDetector`, `GradientMonitor`, `TrainingStabilityMonitor` |
| `curriculum.py` | Board size scheduling | `BoardSizeCurriculum`, `CurriculumStage` |
| `trainer.py` | Main training loop | `Trainer`, `TrainingMetrics` |
| `operator_trainer.py` | Neural operator training | `OperatorTrainer`, `TrainingConfig` |
| `distributed_context.py` | Multi-GPU support | `DistributedContext` |
| `langfuse_tracker.py` | Experiment tracking | `LangfuseTracker` |
| `eval_utils/elo_tracker.py` | Strength tracking | `EloTracker`, `EloRating` |

## Dependencies

**Internal**: `src.modeling` (model), `src.mcts` (search/evaluator), `src.data` (datasets/collation), `src.templates.config` (BaseModuleConfig), `src.tools` (SimpleGoGame for self-play/eval), `src.pde` (PDEConfig/operators for trainer), `config.schemas` (type hints)
**External**: `torch`, `torch.distributed`, `torch.amp`, `jaxtyping`, `pydantic`, `structlog`, `langfuse`, `numpy`

## Conventions & Constraints

1. **Loss Composition**: Always use `AlphaGalerkinLoss` as the base. Physics terms are additive via `CombinedAlphaGalerkinPhysicsLoss`.
2. **Replay Buffer Thread Safety**: All buffer operations are `RLock`-protected. Never access buffer internals directly.
3. **Perspective Adjustment**: `GameRecord.to_experiences()` flips value targets for alternating players. Do not double-flip.
4. **Checkpoint Atomicity**: Always use `CheckpointManager.save()`, never write checkpoints directly. Atomic rename prevents corruption.
5. **Rank-0 Only**: Checkpoint saves, W&B logging, and evaluation run only on rank 0 in distributed mode.
6. **Curriculum Invariant**: `CurriculumStage.size_weights` must sum to 1.0.
7. **Mixed Precision**: AMP is opt-in via config. Loss scaling handled automatically by `GradScaler`.
8. **Beta Annealing**: Prioritized replay buffer anneals beta from initial to 1.0 over training for unbiased gradients.
