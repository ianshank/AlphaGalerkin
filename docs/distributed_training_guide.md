# Distributed Training Guide

> Closes Milestone 5 / Epic 5.2 of [docs/NEXT_STEPS_PLAN.md](NEXT_STEPS_PLAN.md). Companion to the spawn-based integration tests in [tests/distributed/test_spawn_integration.py](../tests/distributed/test_spawn_integration.py).

AlphaGalerkin trains on a single GPU, multi-GPU on one machine, multi-node clusters via SLURM/torchrun, and managed cloud platforms (Vertex AI). This guide covers all four launch modes plus the new dual-GPU local rig (RTX 5060 Ti + 5060) and the canonical config knobs.

## TL;DR

| Setup | Command |
|------|---------|
| Single GPU | `python -m scripts.train` |
| Local dual-GPU (5060 Ti + 5060) | `torchrun --nproc_per_node=2 scripts/train_distributed.py` |
| Single node, 4 GPUs | `torchrun --nproc_per_node=4 scripts/train_distributed.py` |
| Two nodes, 4 GPUs each | `torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$RANK --master_addr=$MASTER scripts/train_distributed.py` |
| SLURM | `sbatch scripts/slurm_train.sh` (uses srun + torchrun) |
| Vertex AI | `python -m scripts.train_vertex --project ... --machine-type a2-highgpu-8g --replica-count 4` |

## Local dual-GPU: RTX 5060 Ti + RTX 5060

The two cards differ in VRAM (Ti has more) and SM count. Use `per_rank_batch_size` to balance memory pressure rather than crash the smaller card.

### 1. Pin the cards to a known order

```bash
# Bash / WSL
export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

```powershell
# PowerShell
$env:CUDA_VISIBLE_DEVICES = "0,1"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

Confirm rank 0 is the RTX 5060 Ti before launching — `nvidia-smi` lists devices in the same order CUDA sees them. If the order is reversed, swap with `CUDA_VISIBLE_DEVICES=1,0`.

### 2. Configure asymmetric per-rank batch sizes

In your config (or via Hydra override):

```yaml
# config/train_distributed.yaml
distributed:
  enabled: true
  world_size: 2
  backend: nccl                      # GPU rig => nccl, never gloo
  per_rank_batch_size: [128, 64]     # rank 0 = 5060 Ti (more VRAM), rank 1 = 5060
```

Or as a CLI override:

```bash
torchrun --nproc_per_node=2 scripts/train_distributed.py \
    distributed.per_rank_batch_size=[128,64]
```

The trainer reads the per-rank value via `DistributedInfraConfig.get_rank_batch_size(rank, default=...)`. When `per_rank_batch_size` is `None` (the default) the legacy uniform behavior is preserved — every rank uses the caller-supplied default.

### 3. Launch

```bash
torchrun --nproc_per_node=2 --master_port=29500 scripts/train_distributed.py
```

Watch both cards:

```bash
watch -n 1 nvidia-smi
```

Expected: rank 0 utilization tracks rank 1; the smaller card peaks at a slightly lower memory footprint thanks to the per-rank batch override.

## Single-node multi-GPU (≥4 cards)

```bash
torchrun --nproc_per_node=4 scripts/train_distributed.py
```

`world_size` is inferred from `--nproc_per_node`. With identical GPUs leave `per_rank_batch_size` at its default (`None`); set it only for asymmetric rigs.

## Multi-node via torchrun

```bash
# On node 0 (rendezvous master)
torchrun \
    --nnodes=2 --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 \
    scripts/train_distributed.py

# On node 1
torchrun \
    --nnodes=2 --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=29500 \
    scripts/train_distributed.py
```

`MASTER_ADDR` must be reachable from every node — typically a private IP on the same subnet. NCCL prefers RDMA-capable interconnect (InfiniBand) but falls back to TCP. If you see ~50 MB/s collectives over TCP, set `NCCL_SOCKET_IFNAME=eth0` (or your fast NIC) to force the right interface.

## SLURM

`scripts/slurm_train.sh` (template):

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/train_distributed.py
```

`LauncherConfig.rdzv_backend` defaults to `"static"`; switch to `"c10d"` for elastic SLURM jobs that survive node preemption.

## Vertex AI

The full Vertex AI flow is documented in [VERTEX_TRAINING_PLAN.md](VERTEX_TRAINING_PLAN.md). Quick summary:

```bash
# Build the container
./scripts/build_vertex_container.sh my-project us-central1

# Launch a 4-replica multi-node job
python -m scripts.train_vertex \
    --project my-project \
    --bucket gs://my-training-bucket \
    --machine-type a2-highgpu-8g \
    --accelerator-type NVIDIA_TESLA_A100 \
    --accelerator-count 8 \
    --replica-count 4

# Spot/preemptible variant (≈70% cheaper, signal-handled)
python -m scripts.train_vertex --spot ...
```

## Running the integration tests

The Track-D test suite spawns real worker processes. Two modes:

**CPU-only (CI safe, gloo backend):** runs anywhere except Windows by default.

```bash
pytest tests/distributed/test_spawn_integration.py -m distributed -v
# or, on Windows, opt in:
RUN_DISTRIBUTED_TESTS=1 pytest tests/distributed/test_spawn_integration.py -m distributed -v
```

**Local dual-GPU (NCCL):** the `TestSpawnNccl` class is gated on:

* `torch.cuda.is_available()` — true on the user's rig
* `torch.cuda.device_count() >= 2` — true on the dual-Blackwell rig
* `torch.distributed.is_nccl_available()` — false on Windows wheels (PyTorch ships without NCCL on Windows). On Linux + CUDA wheels this is true.

So this test runs end-to-end on Linux + dual-GPU, and skips cleanly on Windows / single-GPU / CPU machines.

## Troubleshooting

### `RuntimeError: CUDA error: invalid device ordinal`

`CUDA_VISIBLE_DEVICES` does not match the physical card count. Echo it before launch:

```bash
echo $CUDA_VISIBLE_DEVICES        # e.g. "0,1"
nvidia-smi --query-gpu=index --format=csv
```

### `Address already in use` on `MASTER_PORT`

Another torchrun process is still holding the port. Kill it:

```bash
ss -ltnp | grep :29500
kill <pid>
```

Or pick a different port: `torchrun --master_port=29501 ...`.

### `NCCL WARN Cuda failure 'no kernel image is available for execution'`

The PyTorch build does not include kernels for your GPU's compute capability. Reinstall a build that targets your arch (Blackwell needs PyTorch ≥ 2.9 with CUDA 12.6+).

### NCCL collective hang / very slow all-reduce

Almost always one of:

* Wrong network interface — set `NCCL_SOCKET_IFNAME=<nic>`.
* Mixed CUDA/NCCL versions across nodes.
* Driver/PyTorch mismatch — check with `python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.nccl.version())"`.

Set `NCCL_DEBUG=INFO` to see which collective is stuck.

### Windows + spawn flakiness

The `TestSpawnGloo` class skips on Windows by default. Override with `RUN_DISTRIBUTED_TESTS=1`. If the spawn worker hangs, prefer WSL2 — Windows-native multiprocessing has issues with file-descriptor inheritance that gloo relies on.

## CI integration

The `distributed-cpu` job runs only on Linux (`ubuntu-latest`) and only the gloo subset:

```yaml
# .github/workflows/ci.yml (Track D delta)
distributed-cpu:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.11" }
    - run: pip install -e .[dev]
    - run: pytest tests/distributed/ -m distributed -v
```

NCCL tests are not gated in CI — they run on the user's local rig only.

## Reference

* Config: [src/distributed/config.py](../src/distributed/config.py) — `DistributedInfraConfig`, `DistributedBackend`, `LauncherConfig`, `per_rank_batch_size` field, `get_rank_batch_size()` helper.
* Trainer: [src/distributed/trainer.py](../src/distributed/trainer.py) — `DistributedTrainer` (DDP wrapper).
* Gradient sync: [src/distributed/gradient_sync.py](../src/distributed/gradient_sync.py).
* Launcher utilities: [src/distributed/launcher.py](../src/distributed/launcher.py).
* CLI entry point: [scripts/train_distributed.py](../scripts/train_distributed.py).
* Vertex AI: [VERTEX_TRAINING_PLAN.md](VERTEX_TRAINING_PLAN.md).
* Tests: [tests/distributed/](../tests/distributed/).
