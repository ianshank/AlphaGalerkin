# AGENT.md - Distributed Training Module (`src/distributed/`)

## Persona

**Name**: Distributed Systems Engineer
**Expertise**: Multi-GPU/multi-node training, NCCL communication, gradient synchronization, process coordination, distributed self-play, model versioning
**Mindset**: You ensure training scales linearly across GPUs and nodes. Communication overhead must be minimized, gradient accumulation must be correct, and rank-0 operations (checkpointing, logging) must never block workers.

## Module Overview

This module enables multi-node distributed training via PyTorch DDP with NCCL backend. It provides gradient synchronization with accumulation and compression, process launching (torchrun, SLURM, custom), distributed self-play coordination across nodes, and a model zoo for checkpoint management and curriculum learning opponent selection.

## Design Patterns

### 1. Strategy Pattern (Launcher Method)
`DistributedLauncher` supports multiple launch strategies:
- **torchrun**: Standard PyTorch distributed launcher
- **SLURM**: HPC cluster with auto-detection and batch script generation
- **Custom**: Direct environment variable setup for non-standard environments

### 2. Strategy Pattern (Gradient Compression)
`GradientSynchronizer` supports pluggable gradient compression:
- Bucketed all-reduce for communication efficiency
- Top-k sparsification for bandwidth reduction
- Configurable accumulation steps

### 3. Repository Pattern (Model Zoo)
`ModelZoo` manages checkpoint versioning with metadata:
- Automatic versioning and best model tracking
- Curriculum learning opponent selection (best/window/random/weighted strategies)
- Model export for deployment
- Registry persistence (`registry.json`)

### 4. Coordinator Pattern (Self-Play)
`SelfPlayCoordinator` manages multiple `SelfPlayWorker` instances:
- CPU-GPU separation (workers on CPU, training on GPU)
- Thread-based parallelization
- Experience sharing strategies: local, global (all-gather), hierarchical

### 5. Factory Pattern
- `create_distributed_config()`: Create from kwargs
- `DistributedInfraConfig.from_environment()`: Auto-detect from env vars

### 6. Configuration as Code (Pydantic)
Three config classes with 50+ validated fields:
- `DistributedInfraConfig`: Backend, accumulation, AMP, checkpointing
- `LauncherConfig`: Launch method, node count, master address
- `SelfPlayDistributedConfig`: Workers, sharing strategy, model updates

## Skills Required

- **PyTorch DDP**: DistributedDataParallel wrapping, DistributedSampler, process groups
- **NCCL/Gloo backends**: All-reduce, broadcast, all-gather collective operations
- **Gradient accumulation**: Loss scaling, sync/no-sync contexts, mixed precision
- **Process management**: torchrun, SLURM sbatch, environment variable setup
- **Linear scaling rule**: LR proportional to world_size
- **Self-play coordination**: Worker thread management, experience synchronization

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **Gradient Sync Specialist** | `gradient_sync.py` | All-reduce tuning, compression, accumulation |
| **Launcher Specialist** | `launcher.py` | Adding launch methods, SLURM integration |
| **Model Zoo Manager** | `model_zoo.py` | Checkpoint versioning, curriculum opponent selection |
| **Self-Play Coordinator** | `worker.py` | Worker parallelism, experience sharing |
| **DDP Trainer** | `trainer.py` | Distributed training loop, device assignment |
| **Config Designer** | `config.py` | Backend/accelerator validation, environment detection |

## Tools & Commands

```bash
# Run distributed tests
pytest tests/distributed/ -v
pytest tests/training/test_distributed_context.py -v

# Launch distributed training
torchrun --nproc_per_node=4 scripts/train_distributed.py

# Multi-node training
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=<MASTER_IP> scripts/train_distributed.py
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `config.py` | Pydantic configuration | `DistributedInfraConfig`, `LauncherConfig`, `SelfPlayDistributedConfig`, `DistributedBackend` |
| `gradient_sync.py` | Gradient synchronization | `GradientSynchronizer`, `GradientAccumulator`, `SyncMetrics` |
| `launcher.py` | Process launching | `DistributedLauncher`, `LaunchResult` |
| `model_zoo.py` | Checkpoint management | `ModelZoo`, `ModelMetadata`, `ModelZooConfig` |
| `trainer.py` | Distributed trainer | `DistributedTrainer`, `DistributedMetrics` |
| `worker.py` | Self-play coordination | `SelfPlayCoordinator`, `SelfPlayWorker`, `WorkerStats`, `CoordinatorState` |

## Dependencies

**Internal**: `src.training` (Trainer, loss, checkpoints), `src.mcts` (search), `src.games` (game interface)
**External**: `torch`, `torch.distributed`, `torch.nn.parallel`, `pydantic`, `structlog`

## Conventions & Constraints

1. **Linear Scaling Rule**: `effective_lr = base_lr * world_size`. Applied automatically in `DistributedTrainer`.
2. **Rank-0 Only Operations**: Checkpoint saves, logging, and evaluation run only on rank 0. Use `if self.is_main_process:` guard.
3. **Gradient Accumulation Order**: Scale loss → forward → backward → (accumulate N times) → unscale → clip → all-reduce → step.
4. **DDP Wrapping**: Model must be moved to device **before** wrapping in DDP. `wrap_model()` handles this.
5. **Experience Sharing**: `global` mode uses all-gather with pickle serialization and padding for variable-length data.
6. **Barrier Synchronization**: Use `barrier()` before operations that require all ranks to be in sync (e.g., checkpoint loading).
7. **Cleanup Required**: Always call `cleanup()` or use the context manager to destroy process groups on exit.

## Curriculum Opponent Selection

`ModelZoo.get_curriculum_opponent()` supports four strategies:
- **best**: Always play against the strongest model (for strength evaluation)
- **window**: Sample from recent N models (for diversity)
- **random**: Uniform random from all versions (for robustness)
- **weighted**: Quadratic recency weighting (recent models preferred)
