# Google Vertex AI Training Integration - Implementation Plan

## Executive Summary

This plan outlines the integration of Google Vertex AI training support into AlphaGalerkin. The implementation leverages the existing distributed training infrastructure (`src/distributed/`) and follows established patterns from the codebase (Pydantic configs, structured logging, factory functions).

---

## Phase 1: Configuration Infrastructure

### 1.1 Create Vertex AI Configuration Schema

**File: `src/vertex/config.py`**

Create Pydantic configuration models following the pattern in `src/distributed/config.py` and `src/deployment/config.py`.

```python
from enum import Enum
from pydantic import BaseModel, Field

class VertexMachineType(str, Enum):
    """Vertex AI machine types for training."""
    N1_STANDARD_8 = "n1-standard-8"
    N1_HIGHMEM_8 = "n1-highmem-8"
    A2_HIGHGPU_1G = "a2-highgpu-1g"   # 1x A100
    A2_HIGHGPU_2G = "a2-highgpu-2g"   # 2x A100
    A2_HIGHGPU_4G = "a2-highgpu-4g"   # 4x A100
    A2_HIGHGPU_8G = "a2-highgpu-8g"   # 8x A100
    A3_HIGHGPU_8G = "a3-highgpu-8g"   # 8x H100
    N1_STANDARD_96_T4_X4 = "n1-standard-96"  # 4x T4

class AcceleratorType(str, Enum):
    """GPU accelerator types."""
    NVIDIA_TESLA_T4 = "NVIDIA_TESLA_T4"
    NVIDIA_TESLA_A100 = "NVIDIA_TESLA_A100"
    NVIDIA_H100_80GB = "NVIDIA_H100_80GB"

class VertexStorageConfig(BaseModel):
    """GCS storage configuration."""
    bucket_name: str
    checkpoint_prefix: str = "checkpoints/"
    data_prefix: str = "data/"
    staging_prefix: str = "staging/"
    artifact_prefix: str = "artifacts/"

class VertexResourceConfig(BaseModel):
    """Compute resource configuration."""
    machine_type: VertexMachineType
    accelerator_type: AcceleratorType | None = None
    accelerator_count: int = Field(default=1, ge=0)
    replica_count: int = Field(default=1, ge=1)
    boot_disk_size_gb: int = Field(default=200, ge=100)

class VertexTrainingConfig(BaseModel):
    """Complete Vertex AI training configuration."""
    project_id: str
    region: str = "us-central1"
    staging_bucket: str
    service_account: str | None = None
    network: str | None = None  # VPC network for private IPs
    resources: VertexResourceConfig
    storage: VertexStorageConfig
    tensorboard_name: str | None = None
    enable_web_access: bool = False
    timeout_hours: int = Field(default=24, ge=1, le=168)
    enable_spot: bool = True
    restart_on_preemption: bool = True
```

### 1.2 Extend Main Configuration Schema

**File: `config/schemas.py`** (modification)

Add a `VertexConfig` class that integrates with the existing `AlphaGalerkinConfig`:

```python
class VertexConfig(BaseModel):
    """Vertex AI training configuration."""
    enabled: bool = Field(default=False, description="Enable Vertex AI training")
    project_id: str | None = Field(default=None, description="GCP project ID")
    region: str = Field(default="us-central1", description="GCP region")
    staging_bucket: str | None = Field(default=None, description="GCS staging bucket")
    # ... additional fields
```

### 1.3 Add Hydra Configuration

**File: `config/vertex.yaml`**

```yaml
# Vertex AI training configuration
vertex:
  enabled: false
  project_id: null  # Must be set for Vertex training
  region: us-central1
  staging_bucket: null  # gs://bucket-name

  resources:
    machine_type: a2-highgpu-1g
    accelerator_type: NVIDIA_TESLA_A100
    accelerator_count: 1
    replica_count: 1
    boot_disk_size_gb: 200

  storage:
    checkpoint_prefix: checkpoints/
    data_prefix: data/
    artifact_prefix: artifacts/

  # Cost optimization
  enable_spot: true
  restart_on_preemption: true
  max_run_duration_hours: 24
```

---

## Phase 2: Cloud Storage Integration

### 2.1 GCS Checkpoint Manager

**File: `src/vertex/storage.py`**

Create a GCS-aware checkpoint manager that extends the existing `CheckpointManager` from `src/training/checkpoint.py`.

Key features:
- Transparent GCS read/write using `google-cloud-storage`
- Streaming upload/download for large checkpoints
- Local caching for checkpoint restoration
- Automatic retry with exponential backoff
- Support for checkpoint sharding across GCS objects

```python
from pathlib import Path
from google.cloud import storage
import torch.nn as nn

class GCSCheckpointManager:
    """GCS-backed checkpoint manager with local caching."""

    def __init__(
        self,
        bucket_name: str,
        checkpoint_prefix: str,
        local_cache_dir: Path,
        max_checkpoints: int = 5,
    ):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = checkpoint_prefix
        self.local_cache = local_cache_dir
        self.max_checkpoints = max_checkpoints
        self.local_cache.mkdir(parents=True, exist_ok=True)

    def save(self, step: int, model: nn.Module, optimizer, metrics: dict) -> str:
        """Save checkpoint to GCS with atomic upload."""
        # 1. Save to local temp file
        # 2. Upload to GCS with atomic rename
        # 3. Rotate old checkpoints

    def load(self, gcs_path: str) -> dict:
        """Load checkpoint from GCS with caching."""
        # 1. Check local cache
        # 2. Download from GCS if not cached
        # 3. Return loaded checkpoint

    def sync_to_gcs(self, local_path: Path) -> str:
        """Upload local checkpoint to GCS."""

    def sync_from_gcs(self, gcs_path: str) -> Path:
        """Download GCS checkpoint to local cache."""

    def list_checkpoints(self) -> list[str]:
        """List available checkpoints in GCS."""

    def get_latest_checkpoint(self) -> str | None:
        """Get the most recent checkpoint path."""
```

### 2.2 GCS Data Loader Integration

**File: `src/vertex/data.py`**

Support loading training data from GCS:

```python
from typing import Iterator
from google.cloud import storage

class GCSDataSource:
    """Stream training data from GCS."""

    def __init__(self, bucket: str, prefix: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)
        self.prefix = prefix

    def list_shards(self) -> list[str]:
        """List available data shards."""
        blobs = self.bucket.list_blobs(prefix=self.prefix)
        return [blob.name for blob in blobs if blob.name.endswith('.pt')]

    def stream_shard(self, shard_path: str) -> Iterator:
        """Stream experiences from a GCS shard."""
        # Download shard to temp file and yield experiences

    def upload_experiences(self, experiences: list, shard_name: str) -> str:
        """Upload self-play experiences to GCS."""
```

---

## Phase 3: Vertex AI Launcher

### 3.1 Vertex AI Launcher Module

**File: `src/vertex/launcher.py`**

Create a launcher that follows the pattern in `src/distributed/launcher.py`:

```python
from dataclasses import dataclass
from google.cloud import aiplatform
from google.cloud.aiplatform import CustomContainerTrainingJob

@dataclass
class VertexLaunchResult:
    """Result of launching a Vertex AI training job."""
    job_name: str
    job_id: str
    console_url: str
    state: str

class VertexLauncher:
    """Launcher for Vertex AI training jobs."""

    def __init__(self, config: VertexTrainingConfig):
        self.config = config
        self._initialize_aiplatform()

    def _initialize_aiplatform(self) -> None:
        """Initialize Vertex AI SDK."""
        aiplatform.init(
            project=self.config.project_id,
            location=self.config.region,
            staging_bucket=self.config.staging_bucket,
        )

    def create_training_job(
        self,
        display_name: str,
        container_uri: str,
        args: list[str],
    ) -> CustomContainerTrainingJob:
        """Create Vertex AI custom training job."""
        return CustomContainerTrainingJob(
            display_name=display_name,
            container_uri=container_uri,
            command=["python", "-m", "src.vertex.entrypoint"],
            args=args,
        )

    def launch(
        self,
        display_name: str,
        container_uri: str,
        args: list[str] | None = None,
    ) -> VertexLaunchResult:
        """Launch training job on Vertex AI."""
        job = self.create_training_job(display_name, container_uri, args or [])

        # Configure machine specs
        machine_spec = {
            "machine_type": self.config.resources.machine_type.value,
        }
        if self.config.resources.accelerator_type:
            machine_spec["accelerator_type"] = self.config.resources.accelerator_type.value
            machine_spec["accelerator_count"] = self.config.resources.accelerator_count

        # Run the job
        job.run(
            replica_count=self.config.resources.replica_count,
            machine_type=machine_spec["machine_type"],
            accelerator_type=machine_spec.get("accelerator_type"),
            accelerator_count=machine_spec.get("accelerator_count", 0),
            boot_disk_size_gb=self.config.resources.boot_disk_size_gb,
            service_account=self.config.service_account,
            network=self.config.network,
            tensorboard=self.config.tensorboard_name,
            enable_web_access=self.config.enable_web_access,
            timeout=self.config.timeout_hours * 3600,
            sync=False,  # Return immediately
        )

        return VertexLaunchResult(
            job_name=job.display_name,
            job_id=job.resource_name,
            console_url=f"https://console.cloud.google.com/vertex-ai/training/{job.resource_name}",
            state=job.state.name,
        )

    def get_job_status(self, job_name: str) -> str:
        """Query job status."""
        job = aiplatform.CustomJob.get(job_name)
        return job.state.name

    def cancel_job(self, job_name: str) -> None:
        """Cancel a running job."""
        job = aiplatform.CustomJob.get(job_name)
        job.cancel()
```

### 3.2 Multi-Node Training Support

**File: `src/vertex/multi_node.py`**

Handle Vertex AI's multi-node training setup:

```python
import json
import os
from dataclasses import dataclass

@dataclass
class DistributedContext:
    """Distributed training context from Vertex AI."""
    world_size: int
    rank: int
    local_rank: int
    master_addr: str
    master_port: int

class VertexDistributedSetup:
    """Configure distributed training on Vertex AI."""

    @staticmethod
    def setup_from_environment() -> DistributedContext:
        """Configure DDP from Vertex AI environment variables.

        Vertex AI sets:
        - CLUSTER_SPEC: JSON with worker/master addresses
        - TF_CONFIG: Similar format for TensorFlow
        - RANK, WORLD_SIZE, LOCAL_RANK (for PyTorch)
        """
        # Check for PyTorch-style env vars first
        if "RANK" in os.environ:
            return DistributedContext(
                world_size=int(os.environ["WORLD_SIZE"]),
                rank=int(os.environ["RANK"]),
                local_rank=int(os.environ.get("LOCAL_RANK", 0)),
                master_addr=os.environ.get("MASTER_ADDR", "localhost"),
                master_port=int(os.environ.get("MASTER_PORT", 29500)),
            )

        # Parse CLUSTER_SPEC for Vertex AI
        cluster_spec = os.environ.get("CLUSTER_SPEC", "{}")
        spec = json.loads(cluster_spec)

        workers = spec.get("cluster", {}).get("worker", [])
        task_index = spec.get("task", {}).get("index", 0)

        master_addr = workers[0].split(":")[0] if workers else "localhost"
        master_port = int(workers[0].split(":")[1]) if workers else 29500

        return DistributedContext(
            world_size=len(workers) or 1,
            rank=task_index,
            local_rank=0,
            master_addr=master_addr,
            master_port=master_port,
        )

    @staticmethod
    def get_master_addr() -> str:
        """Extract master address from CLUSTER_SPEC."""
        ctx = VertexDistributedSetup.setup_from_environment()
        return ctx.master_addr

    @staticmethod
    def configure_nccl_for_vertex() -> None:
        """Set NCCL environment variables for Vertex AI networking."""
        # Optimize NCCL for GCP networking
        os.environ.setdefault("NCCL_IB_DISABLE", "1")  # Disable InfiniBand
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
        os.environ.setdefault("NCCL_DEBUG", "INFO")
```

---

## Phase 4: Container Infrastructure

### 4.1 Dockerfile for Vertex AI

**File: `docker/Dockerfile.vertex`**

```dockerfile
# Base image with PyTorch and CUDA
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install -e ".[vertex]"

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Set entrypoint for Vertex AI
ENTRYPOINT ["python", "-m", "src.vertex.entrypoint"]
```

### 4.2 Training Entrypoint

**File: `src/vertex/entrypoint.py`**

```python
#!/usr/bin/env python3
"""Vertex AI training entrypoint."""

import argparse
import os
import sys
from pathlib import Path

import structlog
import torch
import torch.distributed as dist

from src.vertex.config import VertexTrainingConfig
from src.vertex.multi_node import VertexDistributedSetup
from src.vertex.storage import GCSCheckpointManager
from src.vertex.preemption import PreemptionHandler

logger = structlog.get_logger()

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vertex AI Training Entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    return parser.parse_args()

def setup_distributed() -> None:
    """Initialize distributed training from Vertex environment."""
    VertexDistributedSetup.configure_nccl_for_vertex()
    ctx = VertexDistributedSetup.setup_from_environment()

    os.environ["MASTER_ADDR"] = ctx.master_addr
    os.environ["MASTER_PORT"] = str(ctx.master_port)
    os.environ["RANK"] = str(ctx.rank)
    os.environ["WORLD_SIZE"] = str(ctx.world_size)
    os.environ["LOCAL_RANK"] = str(ctx.local_rank)

    if ctx.world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{ctx.master_addr}:{ctx.master_port}",
            world_size=ctx.world_size,
            rank=ctx.rank,
        )
        logger.info(
            "distributed_initialized",
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

def main() -> int:
    """Main entrypoint for Vertex AI training container."""
    args = parse_args()

    logger.info("vertex_training_started", config=args.config)

    # 1. Setup distributed training
    setup_distributed()

    # 2. Load configuration
    # TODO: Load from args.config

    # 3. Initialize GCS checkpoint manager
    # TODO: Create checkpoint manager

    # 4. Setup preemption handler
    # TODO: Register signal handlers

    # 5. Create model and trainer
    # TODO: Initialize training components

    # 6. Run training loop
    # TODO: Execute training

    # 7. Upload final artifacts
    # TODO: Sync to GCS

    logger.info("vertex_training_completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### 4.3 Container Build Script

**File: `scripts/build_vertex_container.sh`**

```bash
#!/bin/bash
# Build and push container to Google Artifact Registry
set -euo pipefail

PROJECT_ID=${1:-$(gcloud config get project)}
REGION=${2:-us-central1}
IMAGE_TAG=${3:-latest}
REPO_NAME=${4:-alphagalerkin}

# Ensure Artifact Registry repository exists
gcloud artifacts repositories describe "${REPO_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" 2>/dev/null || \
gcloud artifacts repositories create "${REPO_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --repository-format=docker \
    --description="AlphaGalerkin training containers"

# Build and push using Cloud Build
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/trainer:${IMAGE_TAG}"

echo "Building and pushing ${IMAGE_URI}..."

gcloud builds submit \
    --project="${PROJECT_ID}" \
    --tag "${IMAGE_URI}" \
    --file docker/Dockerfile.vertex \
    .

echo "Container pushed: ${IMAGE_URI}"
```

---

## Phase 5: Cost Optimization

### 5.1 Spot/Preemptible Instance Support

**File: `src/vertex/preemption.py`**

```python
"""Handle Vertex AI spot instance preemption."""

import signal
import threading
from pathlib import Path
from typing import Callable

import structlog
import torch.nn as nn

logger = structlog.get_logger()

class PreemptionHandler:
    """Handle Vertex AI spot instance preemption."""

    def __init__(
        self,
        checkpoint_callback: Callable[[], None],
        checkpoint_interval: int = 100,
    ):
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_interval = checkpoint_interval
        self._preempted = threading.Event()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Register SIGTERM handler for preemption notice."""
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        logger.info("preemption_handler_registered")

    def _handle_sigterm(self, signum: int, frame) -> None:
        """Handle SIGTERM signal (preemption notice)."""
        logger.warning("preemption_notice_received")
        self._preempted.set()
        self.on_preemption()

    def on_preemption(self) -> None:
        """Save emergency checkpoint on preemption."""
        logger.info("saving_emergency_checkpoint")
        try:
            self.checkpoint_callback()
            logger.info("emergency_checkpoint_saved")
        except Exception as e:
            logger.error("emergency_checkpoint_failed", error=str(e))

    @property
    def is_preempted(self) -> bool:
        """Check if preemption signal was received."""
        return self._preempted.is_set()

    def should_save_checkpoint(self, step: int) -> bool:
        """Aggressive checkpointing for spot instances."""
        # More frequent checkpoints on spot instances
        return step % self.checkpoint_interval == 0
```

### 5.2 Cost Monitoring

**File: `src/vertex/cost.py`**

```python
"""Track and estimate Vertex AI training costs."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.vertex.config import VertexMachineType, AcceleratorType

@dataclass
class CostEstimate:
    """Cost estimate for a training job."""
    machine_cost_per_hour: float
    accelerator_cost_per_hour: float
    total_cost_per_hour: float
    estimated_total_cost: float
    duration_hours: float
    is_spot: bool

class CostTracker:
    """Track and estimate Vertex AI training costs."""

    # Approximate hourly rates (USD) - update based on current pricing
    MACHINE_RATES = {
        VertexMachineType.N1_STANDARD_8: 0.38,
        VertexMachineType.N1_HIGHMEM_8: 0.47,
        VertexMachineType.A2_HIGHGPU_1G: 3.67,
        VertexMachineType.A2_HIGHGPU_2G: 7.35,
        VertexMachineType.A2_HIGHGPU_4G: 14.69,
        VertexMachineType.A2_HIGHGPU_8G: 29.39,
        VertexMachineType.A3_HIGHGPU_8G: 45.00,  # H100 pricing
    }

    ACCELERATOR_RATES = {
        AcceleratorType.NVIDIA_TESLA_T4: 0.35,
        AcceleratorType.NVIDIA_TESLA_A100: 2.93,
        AcceleratorType.NVIDIA_H100_80GB: 5.00,
    }

    SPOT_DISCOUNT = 0.7  # 70% discount for spot instances

    def __init__(self):
        self.start_time: datetime | None = None
        self.machine_type: VertexMachineType | None = None
        self.accelerator_type: AcceleratorType | None = None
        self.accelerator_count: int = 0
        self.is_spot: bool = False

    def start_tracking(
        self,
        machine_type: VertexMachineType,
        accelerator_type: AcceleratorType | None = None,
        accelerator_count: int = 0,
        is_spot: bool = False,
    ) -> None:
        """Start tracking costs for a training job."""
        self.start_time = datetime.now()
        self.machine_type = machine_type
        self.accelerator_type = accelerator_type
        self.accelerator_count = accelerator_count
        self.is_spot = is_spot

    def estimate_job_cost(
        self,
        machine_type: VertexMachineType,
        duration_hours: float,
        accelerator_type: AcceleratorType | None = None,
        accelerator_count: int = 0,
        is_spot: bool = False,
    ) -> CostEstimate:
        """Estimate total job cost."""
        machine_cost = self.MACHINE_RATES.get(machine_type, 0.0)

        accel_cost = 0.0
        if accelerator_type and accelerator_count > 0:
            accel_cost = self.ACCELERATOR_RATES.get(accelerator_type, 0.0) * accelerator_count

        total_hourly = machine_cost + accel_cost

        if is_spot:
            total_hourly *= (1 - self.SPOT_DISCOUNT)

        return CostEstimate(
            machine_cost_per_hour=machine_cost,
            accelerator_cost_per_hour=accel_cost,
            total_cost_per_hour=total_hourly,
            estimated_total_cost=total_hourly * duration_hours,
            duration_hours=duration_hours,
            is_spot=is_spot,
        )

    def get_current_cost(self) -> CostEstimate | None:
        """Get current accumulated cost."""
        if self.start_time is None or self.machine_type is None:
            return None

        duration = datetime.now() - self.start_time
        hours = duration.total_seconds() / 3600

        return self.estimate_job_cost(
            machine_type=self.machine_type,
            duration_hours=hours,
            accelerator_type=self.accelerator_type,
            accelerator_count=self.accelerator_count,
            is_spot=self.is_spot,
        )
```

---

## Phase 6: Integration with Existing Infrastructure

### 6.1 Trainer Integration

**File: `src/vertex/trainer.py`**

Create a Vertex-aware trainer wrapper that integrates with the existing `Trainer` from `src/training/trainer.py`:

```python
"""Vertex AI-aware training wrapper."""

from pathlib import Path
from typing import Any

import structlog
import torch

from src.modeling.alpha_galerkin import AlphaGalerkinModel
from src.training.trainer import Trainer, create_trainer
from src.vertex.config import VertexTrainingConfig
from src.vertex.storage import GCSCheckpointManager
from src.vertex.preemption import PreemptionHandler
from src.vertex.cost import CostTracker

logger = structlog.get_logger()

class VertexTrainer:
    """Vertex AI-aware training wrapper."""

    def __init__(
        self,
        model: AlphaGalerkinModel,
        config: dict[str, Any],
        vertex_config: VertexTrainingConfig,
    ):
        self.model = model
        self.config = config
        self.vertex_config = vertex_config

        # Create GCS checkpoint manager
        self.checkpoint_manager = GCSCheckpointManager(
            bucket_name=vertex_config.storage.bucket_name,
            checkpoint_prefix=vertex_config.storage.checkpoint_prefix,
            local_cache_dir=Path("/tmp/checkpoints"),
        )

        # Create cost tracker
        self.cost_tracker = CostTracker()
        self.cost_tracker.start_tracking(
            machine_type=vertex_config.resources.machine_type,
            accelerator_type=vertex_config.resources.accelerator_type,
            accelerator_count=vertex_config.resources.accelerator_count,
            is_spot=vertex_config.enable_spot,
        )

        # Create preemption handler
        self.preemption_handler = PreemptionHandler(
            checkpoint_callback=self._emergency_checkpoint,
            checkpoint_interval=100 if vertex_config.enable_spot else 500,
        )

        # Create base trainer
        self.trainer: Trainer | None = None

    def _emergency_checkpoint(self) -> None:
        """Save emergency checkpoint on preemption."""
        if self.trainer is not None:
            state = self.trainer.get_state()
            self.checkpoint_manager.save(
                step=state["step"],
                model=self.model,
                optimizer=self.trainer.optimizer,
                metrics={"emergency": True},
            )

    def train(self, resume_from: str | None = None) -> dict[str, Any]:
        """Run training with Vertex AI integration."""
        # Load checkpoint if resuming
        if resume_from:
            checkpoint = self.checkpoint_manager.load(resume_from)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("checkpoint_loaded", path=resume_from)

        # Initialize trainer
        self.trainer = create_trainer(
            model=self.model,
            config=self.config,
            checkpoint_dir=Path("/tmp/checkpoints"),
        )

        # Training loop with preemption checking
        step = 0
        while not self.preemption_handler.is_preempted:
            metrics = self.trainer.train_step()
            step += 1

            # Checkpoint based on preemption handler recommendation
            if self.preemption_handler.should_save_checkpoint(step):
                gcs_path = self.checkpoint_manager.save(
                    step=step,
                    model=self.model,
                    optimizer=self.trainer.optimizer,
                    metrics=metrics,
                )
                logger.info("checkpoint_saved", step=step, path=gcs_path)

            # Log cost periodically
            if step % 1000 == 0:
                cost = self.cost_tracker.get_current_cost()
                if cost:
                    logger.info(
                        "cost_update",
                        step=step,
                        cost_usd=f"{cost.estimated_total_cost:.2f}",
                        hours=f"{cost.duration_hours:.2f}",
                    )

        # Final checkpoint
        final_path = self.checkpoint_manager.save(
            step=step,
            model=self.model,
            optimizer=self.trainer.optimizer,
            metrics={"final": True},
        )

        final_cost = self.cost_tracker.get_current_cost()
        return {
            "final_checkpoint": final_path,
            "total_steps": step,
            "cost_estimate": final_cost,
        }
```

### 6.2 W&B Integration for Vertex AI

**File: `src/vertex/logging.py`**

```python
"""Logging utilities for Vertex AI training."""

import wandb
import structlog

logger = structlog.get_logger()

class VertexWandbLogger:
    """W&B logger with Vertex AI-specific metrics."""

    def __init__(
        self,
        project: str,
        name: str,
        config: dict,
        vertex_config: dict,
    ):
        self.run = wandb.init(
            project=project,
            name=name,
            config={**config, "vertex": vertex_config},
            tags=["vertex-ai"],
        )

    def log_vertex_metrics(
        self,
        step: int,
        job_name: str,
        estimated_cost: float,
        gpu_utilization: float | None = None,
    ) -> None:
        """Log Vertex AI-specific metrics."""
        metrics = {
            "vertex/job_name": job_name,
            "vertex/estimated_cost_usd": estimated_cost,
        }
        if gpu_utilization is not None:
            metrics["vertex/gpu_utilization"] = gpu_utilization

        wandb.log(metrics, step=step)

    def log_preemption(self, step: int) -> None:
        """Log preemption event."""
        wandb.log({"vertex/preempted": 1}, step=step)
        logger.warning("preemption_logged_to_wandb", step=step)

    def finish(self) -> None:
        """Finish W&B run."""
        self.run.finish()
```

---

## Phase 7: CLI and Scripts

### 7.1 Vertex AI Training CLI

**File: `scripts/train_vertex.py`**

```python
#!/usr/bin/env python3
"""Launch AlphaGalerkin training on Vertex AI."""

import argparse
import sys

import structlog
import yaml

from src.vertex.config import VertexTrainingConfig
from src.vertex.launcher import VertexLauncher

logger = structlog.get_logger()

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch training on Vertex AI")
    parser.add_argument(
        "--config",
        type=str,
        default="config/vertex.yaml",
        help="Path to Vertex AI config file",
    )
    parser.add_argument(
        "--container-uri",
        type=str,
        required=True,
        help="Container image URI in Artifact Registry",
    )
    parser.add_argument(
        "--display-name",
        type=str,
        default="alphagalerkin-training",
        help="Display name for the training job",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="GCS path to checkpoint to resume from",
    )
    return parser.parse_args()

def main() -> int:
    """Launch training job on Vertex AI."""
    args = parse_args()

    # Load configuration
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    vertex_config = VertexTrainingConfig(**config_dict["vertex"])

    # Create launcher
    launcher = VertexLauncher(vertex_config)

    # Build training arguments
    training_args = ["--config", "/app/config/train.yaml"]
    if args.resume:
        training_args.extend(["--resume", args.resume])

    # Launch job
    logger.info("launching_vertex_job", display_name=args.display_name)
    result = launcher.launch(
        display_name=args.display_name,
        container_uri=args.container_uri,
        args=training_args,
    )

    logger.info(
        "job_launched",
        job_name=result.job_name,
        job_id=result.job_id,
        console_url=result.console_url,
    )

    print(f"\nJob launched successfully!")
    print(f"  Job Name: {result.job_name}")
    print(f"  Console:  {result.console_url}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### 7.2 Job Management CLI

**File: `scripts/vertex_jobs.py`**

```python
#!/usr/bin/env python3
"""Manage Vertex AI training jobs."""

import argparse
import sys

import structlog
from google.cloud import aiplatform

logger = structlog.get_logger()

def list_jobs(args: argparse.Namespace) -> int:
    """List running and recent jobs."""
    aiplatform.init(project=args.project, location=args.region)

    jobs = aiplatform.CustomJob.list(
        filter=f'display_name="{args.filter}"' if args.filter else None,
        order_by="create_time desc",
    )

    print(f"\n{'Job Name':<40} {'State':<15} {'Created'}")
    print("-" * 80)
    for job in jobs[:args.limit]:
        print(f"{job.display_name:<40} {job.state.name:<15} {job.create_time}")

    return 0

def cancel_job(args: argparse.Namespace) -> int:
    """Cancel a running job."""
    aiplatform.init(project=args.project, location=args.region)

    job = aiplatform.CustomJob.get(args.job_name)
    job.cancel()

    logger.info("job_cancelled", job_name=args.job_name)
    print(f"Job '{args.job_name}' cancellation requested")

    return 0

def stream_logs(args: argparse.Namespace) -> int:
    """Stream logs from a job."""
    # Note: Use 'gcloud ai custom-jobs stream-logs' for real-time streaming
    print(f"Run: gcloud ai custom-jobs stream-logs {args.job_name} --region={args.region}")
    return 0

def main() -> int:
    """Entry point for job management CLI."""
    parser = argparse.ArgumentParser(description="Manage Vertex AI training jobs")
    parser.add_argument("--project", type=str, help="GCP project ID")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP region")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    list_parser = subparsers.add_parser("list", help="List training jobs")
    list_parser.add_argument("--filter", type=str, help="Filter by display name")
    list_parser.add_argument("--limit", type=int, default=10, help="Max jobs to show")
    list_parser.set_defaults(func=list_jobs)

    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a running job")
    cancel_parser.add_argument("job_name", type=str, help="Job resource name")
    cancel_parser.set_defaults(func=cancel_job)

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Stream logs from a job")
    logs_parser.add_argument("job_name", type=str, help="Job resource name")
    logs_parser.set_defaults(func=stream_logs)

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
```

---

## Phase 8: Testing Strategy

### 8.1 Unit Tests

**Directory: `tests/vertex/`**

```
tests/vertex/
├── __init__.py
├── conftest.py          # Test fixtures
├── test_config.py       # Configuration validation
├── test_storage.py      # GCS checkpoint manager (mocked)
├── test_launcher.py     # Launcher logic (mocked SDK)
├── test_preemption.py   # Preemption handler
├── test_multi_node.py   # Distributed setup
└── test_cost.py         # Cost estimation
```

**File: `tests/vertex/conftest.py`**

```python
"""Test fixtures for Vertex AI tests."""

import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_gcs_client():
    """Mock GCS client for unit tests."""
    with patch("google.cloud.storage.Client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client

@pytest.fixture
def mock_aiplatform():
    """Mock Vertex AI SDK for unit tests."""
    with patch("google.cloud.aiplatform") as mock:
        yield mock

@pytest.fixture
def vertex_config():
    """Sample Vertex AI configuration."""
    from src.vertex.config import (
        VertexTrainingConfig,
        VertexResourceConfig,
        VertexStorageConfig,
        VertexMachineType,
        AcceleratorType,
    )
    return VertexTrainingConfig(
        project_id="test-project",
        region="us-central1",
        staging_bucket="gs://test-bucket",
        resources=VertexResourceConfig(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
            accelerator_count=1,
        ),
        storage=VertexStorageConfig(
            bucket_name="test-bucket",
        ),
    )
```

**File: `tests/vertex/test_config.py`**

```python
"""Tests for Vertex AI configuration."""

import pytest
from pydantic import ValidationError

from src.vertex.config import (
    VertexTrainingConfig,
    VertexResourceConfig,
    VertexStorageConfig,
    VertexMachineType,
    AcceleratorType,
)

class TestVertexTrainingConfig:
    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = VertexTrainingConfig(
            project_id="test",
            staging_bucket="gs://test",
            resources=VertexResourceConfig(
                machine_type=VertexMachineType.A2_HIGHGPU_1G,
            ),
            storage=VertexStorageConfig(bucket_name="test"),
        )
        assert config.region == "us-central1"
        assert config.timeout_hours == 24
        assert config.enable_spot is True

    def test_machine_type_validation(self) -> None:
        """Test machine type enum validation."""
        with pytest.raises(ValidationError):
            VertexResourceConfig(machine_type="invalid-machine")

    def test_accelerator_count_validation(self) -> None:
        """Test accelerator count constraints."""
        with pytest.raises(ValidationError):
            VertexResourceConfig(
                machine_type=VertexMachineType.A2_HIGHGPU_1G,
                accelerator_count=-1,
            )
```

### 8.2 Integration Tests

**File: `tests/vertex/test_integration.py`**

```python
"""Integration tests for Vertex AI (requires GCP credentials)."""

import pytest
import os

# Skip if no GCP credentials
pytestmark = pytest.mark.skipif(
    "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ,
    reason="GCP credentials not configured"
)

class TestVertexIntegration:
    @pytest.mark.vertex_required
    def test_gcs_checkpoint_roundtrip(self, vertex_config) -> None:
        """Test saving/loading checkpoint to/from GCS."""
        # Requires actual GCS bucket
        pass

    @pytest.mark.vertex_required
    def test_aiplatform_initialization(self, vertex_config) -> None:
        """Test Vertex AI SDK initialization."""
        from google.cloud import aiplatform
        aiplatform.init(
            project=vertex_config.project_id,
            location=vertex_config.region,
        )
```

---

## Phase 9: Documentation

### 9.1 Update CLAUDE.md

Add new section to CLAUDE.md:

```markdown
## Vertex AI Training (src/vertex/)
- [2026-XX-XX]: Added Google Vertex AI training support.
- [2026-XX-XX]: GCS checkpoint manager with automatic syncing.
- [2026-XX-XX]: Spot instance support with preemption handling.
- [2026-XX-XX]: Multi-node distributed training on Vertex AI.
- [2026-XX-XX]: Cost tracking and estimation utilities.

## Vertex AI Commands
```bash
# Build and push training container
./scripts/build_vertex_container.sh PROJECT_ID REGION TAG

# Launch training job
python scripts/train_vertex.py \
    --config config/vertex.yaml \
    --container-uri REGION-docker.pkg.dev/PROJECT/alphagalerkin/trainer:TAG

# List jobs
python scripts/vertex_jobs.py --project PROJECT list

# Cancel job
python scripts/vertex_jobs.py --project PROJECT cancel JOB_NAME

# Stream logs
gcloud ai custom-jobs stream-logs JOB_NAME --region=REGION
```
```

---

## Implementation Order Summary

| Phase | Priority | Estimated Effort | Dependencies |
|-------|----------|------------------|--------------|
| Phase 1: Configuration | High | 2 days | None |
| Phase 2: GCS Storage | High | 3 days | Phase 1 |
| Phase 3: Launcher | High | 3 days | Phase 1, 2 |
| Phase 4: Container | High | 2 days | Phase 1, 2, 3 |
| Phase 5: Cost Optimization | Medium | 2 days | Phase 3, 4 |
| Phase 6: Integration | High | 3 days | Phase 1-4 |
| Phase 7: CLI | Medium | 1 day | Phase 3 |
| Phase 8: Testing | High | 3 days | All phases |
| Phase 9: Documentation | Medium | 2 days | All phases |

---

## Dependencies to Add

**Update `pyproject.toml`:**

```toml
[project.optional-dependencies]
vertex = [
    "google-cloud-aiplatform>=1.38.0",
    "google-cloud-storage>=2.10.0",
    "google-auth>=2.23.0",
]
```

---

## Key Design Decisions

1. **Extend, Don't Replace**: Build on existing `DistributedTrainer`, `CheckpointManager`, and logging infrastructure rather than replacing them.

2. **Configuration-Driven**: All Vertex AI parameters are exposed through Pydantic configs and Hydra YAML files, following the no-hardcoded-values principle.

3. **Local-First Development**: The same training code runs locally (with mocked GCS) and on Vertex AI, enabling easy debugging.

4. **Cost-Aware by Default**: Spot instance support and aggressive checkpointing are enabled by default to minimize costs.

5. **Observability**: Full integration with W&B and Vertex AI TensorBoard for experiment tracking.

---

## New Files Summary

```
src/vertex/
├── __init__.py
├── config.py          # Pydantic configuration schemas
├── storage.py         # GCS checkpoint manager
├── data.py            # GCS data loading
├── launcher.py        # Vertex AI job launcher
├── multi_node.py      # Distributed training setup
├── entrypoint.py      # Container entrypoint
├── preemption.py      # Spot instance preemption handling
├── cost.py            # Cost tracking and estimation
├── trainer.py         # Vertex-aware trainer wrapper
└── logging.py         # W&B/TensorBoard integration

docker/
└── Dockerfile.vertex  # Training container

scripts/
├── build_vertex_container.sh
├── train_vertex.py
└── vertex_jobs.py

config/
└── vertex.yaml        # Default Vertex AI configuration

tests/vertex/
├── __init__.py
├── conftest.py
├── test_config.py
├── test_storage.py
├── test_launcher.py
├── test_preemption.py
├── test_multi_node.py
├── test_cost.py
└── test_integration.py

docs/
└── vertex_training.md # User guide
```
