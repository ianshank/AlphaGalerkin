# AGENT.md - Google Vertex AI Training Module (`src/vertex/`)

## Persona

**Name**: Cloud ML Engineer
**Expertise**: Google Cloud Platform, Vertex AI Custom Jobs, GCS storage, spot instance management, distributed cloud training, cost optimization
**Mindset**: You manage the full lifecycle of cloud training jobs — from authentication and cost estimation through job submission to preemption-safe checkpointing. Every dollar matters: spot instances save 70%, but you must handle preemption gracefully.

## Module Overview

This module provides complete Vertex AI integration for cloud-based training: GCP authentication (ADC, service account, gcloud CLI), Pydantic configuration for machine types and accelerators, GCS checkpoint management with local caching and atomic operations, multi-node distributed training with automatic environment detection, spot instance preemption handling with emergency checkpoints, cost tracking and estimation, and a Vertex-aware trainer wrapper.

## Design Patterns

### 1. Strategy Pattern (Authentication)
`GCPAuthenticator` supports three auth methods:
- **ADC** (Application Default Credentials): Via google-auth library
- **SERVICE_ACCOUNT**: JSON key file validation
- **GCLOUD**: CLI-based with `gcloud auth login`

Cross-platform: Windows PowerShell uses `cmd /c` wrapper to bypass PSSecurityException.

### 2. Repository Pattern (GCS Checkpoints)
`GCSCheckpointManager` provides checkpoint CRUD with:
- Atomic writes (temp file → rename)
- Local caching for fast resume
- Retry with exponential backoff
- Best model tracking (`best.pt` symlink)
- Lazy GCS client initialization

### 3. Observer Pattern (Preemption)
`PreemptionHandler` registers signal handlers (SIGTERM/SIGINT) and triggers emergency checkpoint callbacks. `PreemptionMonitor` runs a background thread for continuous monitoring.

### 4. Observer Pattern (Cost Tracking)
`CostTracker` monitors job duration and applies pricing tables to estimate costs in real-time. Supports spot discounts (70% off) and multi-replica multiplication.

### 5. Facade Pattern (VertexTrainer)
`VertexTrainer` wraps all Vertex AI components into a unified training interface:
```
VertexTrainer
  ├── DistributedContext (multi-node setup)
  ├── GCSCheckpointManager (persistent storage)
  ├── PreemptionHandler (spot safety)
  └── CostTracker (budget monitoring)
```

### 6. Factory Pattern (Environment Detection)
`VertexDistributedSetup.setup_distributed_training()` auto-detects the runtime environment:
1. PyTorch env vars (RANK, WORLD_SIZE) — standard torchrun
2. CLUSTER_SPEC JSON — Vertex AI native
3. TF_CONFIG JSON — Cloud ML Engine format
4. Default: single-node single-GPU

### 7. Configuration as Code (Pydantic)
Comprehensive enums and validated configs: 30+ machine types, 12 GPU/TPU types, 30+ GCP regions, disk types, network configs, and resource configs.

## Skills Required

- **GCP APIs**: Vertex AI SDK, Cloud Storage, IAM authentication
- **Docker**: Container building for training jobs (`Dockerfile.vertex`)
- **NCCL on GCP**: Disabled InfiniBand, eth0 socket interface, P2P/SHM settings
- **Spot instances**: Preemption signals, grace periods, emergency checkpoints
- **Cost optimization**: Machine type selection, spot vs on-demand tradeoffs
- **Multi-node setup**: Environment variable parsing (CLUSTER_SPEC, TF_CONFIG)

## Sub-Agents

| Sub-Agent | Scope | When to Invoke |
|-----------|-------|----------------|
| **Auth Specialist** | `auth.py` | Credential validation, cross-platform CLI |
| **Config Specialist** | `config.py` | Machine types, accelerators, resource validation |
| **Storage Specialist** | `storage.py` | GCS upload/download, caching, retry logic |
| **Launch Specialist** | `launcher.py` | Job submission, status polling, cancellation |
| **Multi-Node Specialist** | `multi_node.py` | Environment detection, NCCL configuration |
| **Preemption Specialist** | `preemption.py` | Signal handling, emergency checkpoints |
| **Cost Specialist** | `cost.py` | Pricing tables, estimation, budget tracking |
| **Trainer Specialist** | `trainer.py` | Training loop with Vertex integration |

## Tools & Commands

```bash
# Run Vertex AI tests (124 tests)
pytest tests/vertex/ -v

# Build and push training container
./scripts/build_vertex_container.sh my-project us-central1

# Launch training job
python -m scripts.train_vertex \
    --project my-project \
    --region us-central1 \
    --bucket gs://my-training-bucket \
    --machine-type a2-highgpu-1g \
    --accelerator-type NVIDIA_TESLA_A100

# Launch spot training (70% cost savings)
python -m scripts.train_vertex --project my-project --bucket gs://my-bucket --spot

# Job management
python -m scripts.vertex_jobs list --project my-project
python -m scripts.vertex_jobs show JOB_ID --project my-project
python -m scripts.vertex_jobs cancel JOB_ID --project my-project
python -m scripts.vertex_jobs logs JOB_ID --project my-project
```

## Key Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `config.py` | Resource configuration | `VertexTrainingConfig`, `VertexResourceConfig`, `VertexStorageConfig`, `VertexMachineType`, `AcceleratorType`, `VertexRegion` |
| `auth.py` | GCP authentication | `GCPAuthenticator`, `AuthMethod`, `PlatformInfo`, `ValidationResult` |
| `storage.py` | GCS checkpoint management | `GCSCheckpointManager`, `GCSCheckpointMetadata`, `GCSDataSource` |
| `launcher.py` | Job submission | `VertexLauncher`, `VertexLaunchResult`, `JobStatus`, `JobState` |
| `multi_node.py` | Distributed setup | `VertexDistributedSetup`, `DistributedContext` |
| `preemption.py` | Spot instance safety | `PreemptionHandler`, `PreemptionMonitor`, `PreemptionEvent` |
| `cost.py` | Cost tracking | `CostTracker`, `CostEstimate`, `CostBreakdown` |
| `trainer.py` | Vertex-aware trainer | `VertexTrainer`, `VertexTrainingResult` |

## Dependencies

**Internal**: `src.training` (base trainer), `src.distributed` (DDP setup)
**External**: `google-cloud-aiplatform`, `google-cloud-storage`, `google-auth`, `torch`, `pydantic`, `structlog`

## Conventions & Constraints

1. **Pre-flight Auth Validation**: When `validate_auth_before_launch=True`, credentials are validated before submitting jobs to catch auth errors early.
2. **Atomic GCS Writes**: Always write to temp file, then rename. Never write directly to the final checkpoint path.
3. **Local Caching**: `GCSCheckpointManager` caches checkpoints locally. Resume checks local cache before downloading from GCS.
4. **Spot Checkpointing**: When `enable_spot=True`, checkpoint interval reduces to 10 minutes (configurable via `aggressive_checkpointing_on_spot`).
5. **NCCL for GCP**: Disable InfiniBand (`NCCL_IB_DISABLE=1`), use eth0 (`NCCL_SOCKET_IFNAME=eth0`).
6. **Grace Period**: 30 seconds after SIGTERM to save emergency checkpoint before VM is reclaimed.
7. **Cost Tables**: Pricing data is approximate 2025 USD. Update periodically as GCP pricing changes.
8. **Container Images**: Training containers use `docker/Dockerfile.vertex`. Build with `scripts/build_vertex_container.sh`.

## Job Lifecycle

```
1. Configure: VertexTrainingConfig (project, region, machine, GPUs, bucket)
2. Validate Auth: GCPAuthenticator.validate_credentials()
3. Launch: VertexLauncher.launch(config)
     → Vertex AI CustomJob submitted
     → Returns job_name, console_url
4. Monitor: VertexLauncher.get_job_status(job_name)
5. Inside Job:
     a. VertexDistributedSetup.setup_distributed_training()
     b. VertexTrainer.setup() → GCS, preemption, cost tracking
     c. VertexTrainer.train() → training loop
        - On preemption signal: emergency checkpoint to GCS
        - On interval: regular checkpoint to GCS
     d. Final checkpoint + cost summary
6. Complete: VertexLauncher.wait_for_completion(job_name)
```
