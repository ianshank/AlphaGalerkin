"""Google Vertex AI training integration for AlphaGalerkin.

This module provides infrastructure for running distributed training
on Google Cloud Vertex AI, including:

- Configuration schemas for Vertex AI resources
- GCS-backed checkpoint management with local caching
- Multi-node distributed training setup
- Spot instance preemption handling
- Cost tracking and estimation
- Integration with existing training infrastructure

Example:
    from src.vertex import (
        VertexTrainingConfig,
        VertexLauncher,
        GCSCheckpointManager,
    )

    # Configure Vertex AI training
    config = VertexTrainingConfig(
        project_id="my-project",
        staging_bucket="gs://my-bucket",
        resources=VertexResourceConfig(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
        ),
        storage=VertexStorageConfig(bucket_name="my-bucket"),
    )

    # Launch training job
    launcher = VertexLauncher(config)
    result = launcher.launch(
        display_name="alphagalerkin-training",
        container_uri="gcr.io/my-project/trainer:latest",
    )
"""

from src.vertex.config import (
    AcceleratorType,
    DiskType,
    VertexMachineType,
    VertexNetworkConfig,
    VertexRegion,
    VertexResourceConfig,
    VertexStorageConfig,
    VertexTrainingConfig,
    create_vertex_config,
)
from src.vertex.cost import (
    CostBreakdown,
    CostEstimate,
    CostTracker,
    estimate_job_cost,
    get_hourly_rate,
)
from src.vertex.launcher import (
    JobState,
    JobStatus,
    VertexLaunchResult,
    VertexLauncher,
    create_launcher,
)
from src.vertex.multi_node import (
    DistributedContext,
    VertexDistributedSetup,
    setup_distributed_training,
)
from src.vertex.preemption import (
    PreemptionEvent,
    PreemptionHandler,
    PreemptionMonitor,
    create_preemption_handler,
)
from src.vertex.storage import (
    GCSCheckpointManager,
    GCSCheckpointMetadata,
    GCSDataSource,
)
from src.vertex.trainer import (
    VertexTrainer,
    VertexTrainingResult,
    create_vertex_trainer,
)

__all__ = [
    # Enums
    "AcceleratorType",
    "DiskType",
    "JobState",
    "VertexMachineType",
    "VertexRegion",
    # Config classes
    "VertexNetworkConfig",
    "VertexResourceConfig",
    "VertexStorageConfig",
    "VertexTrainingConfig",
    # Launcher
    "JobStatus",
    "VertexLaunchResult",
    "VertexLauncher",
    # Storage
    "GCSCheckpointManager",
    "GCSCheckpointMetadata",
    "GCSDataSource",
    # Multi-node
    "DistributedContext",
    "VertexDistributedSetup",
    # Preemption
    "PreemptionEvent",
    "PreemptionHandler",
    "PreemptionMonitor",
    # Cost
    "CostBreakdown",
    "CostEstimate",
    "CostTracker",
    # Trainer
    "VertexTrainer",
    "VertexTrainingResult",
    # Factory functions
    "create_launcher",
    "create_preemption_handler",
    "create_vertex_config",
    "create_vertex_trainer",
    "estimate_job_cost",
    "get_hourly_rate",
    "setup_distributed_training",
]
