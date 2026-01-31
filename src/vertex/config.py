"""Configuration schemas for Google Vertex AI training.

This module defines Pydantic models for Vertex AI training configuration,
ensuring type safety, validation, and serialization.

Design Principles:
    - No hardcoded values: All constants are configurable with sensible defaults
    - Backwards compatible: New fields have defaults, removed fields are deprecated
    - Validated: Pydantic enforces types and constraints at runtime
    - Serializable: All configs can be saved/loaded from YAML/JSON

Example:
    from src.vertex.config import (
        VertexTrainingConfig,
        VertexResourceConfig,
        VertexStorageConfig,
        VertexMachineType,
        AcceleratorType,
    )

    config = VertexTrainingConfig(
        project_id="my-gcp-project",
        staging_bucket="gs://my-bucket",
        resources=VertexResourceConfig(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
            accelerator_count=1,
        ),
        storage=VertexStorageConfig(bucket_name="my-bucket"),
    )

"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Import AuthConfig for type annotation (deferred to avoid circular imports)
# We use a lazy import pattern in get_auth_config() instead


class VertexMachineType(str, Enum):
    """Vertex AI machine types for training.

    See: https://cloud.google.com/vertex-ai/docs/training/configure-compute
    """

    # Standard machines
    N1_STANDARD_4 = "n1-standard-4"
    N1_STANDARD_8 = "n1-standard-8"
    N1_STANDARD_16 = "n1-standard-16"
    N1_STANDARD_32 = "n1-standard-32"
    N1_STANDARD_64 = "n1-standard-64"
    N1_STANDARD_96 = "n1-standard-96"

    # High-memory machines
    N1_HIGHMEM_2 = "n1-highmem-2"
    N1_HIGHMEM_4 = "n1-highmem-4"
    N1_HIGHMEM_8 = "n1-highmem-8"
    N1_HIGHMEM_16 = "n1-highmem-16"
    N1_HIGHMEM_32 = "n1-highmem-32"
    N1_HIGHMEM_64 = "n1-highmem-64"
    N1_HIGHMEM_96 = "n1-highmem-96"

    # A2 machines (A100 GPUs)
    A2_HIGHGPU_1G = "a2-highgpu-1g"   # 1x A100 40GB
    A2_HIGHGPU_2G = "a2-highgpu-2g"   # 2x A100 40GB
    A2_HIGHGPU_4G = "a2-highgpu-4g"   # 4x A100 40GB
    A2_HIGHGPU_8G = "a2-highgpu-8g"   # 8x A100 40GB
    A2_MEGAGPU_16G = "a2-megagpu-16g"  # 16x A100 40GB
    A2_ULTRAGPU_1G = "a2-ultragpu-1g"  # 1x A100 80GB
    A2_ULTRAGPU_2G = "a2-ultragpu-2g"  # 2x A100 80GB
    A2_ULTRAGPU_4G = "a2-ultragpu-4g"  # 4x A100 80GB
    A2_ULTRAGPU_8G = "a2-ultragpu-8g"  # 8x A100 80GB

    # A3 machines (H100 GPUs)
    A3_HIGHGPU_8G = "a3-highgpu-8g"   # 8x H100 80GB

    # G2 machines (L4 GPUs)
    G2_STANDARD_4 = "g2-standard-4"   # 1x L4
    G2_STANDARD_8 = "g2-standard-8"   # 1x L4
    G2_STANDARD_12 = "g2-standard-12"  # 1x L4
    G2_STANDARD_16 = "g2-standard-16"  # 1x L4
    G2_STANDARD_24 = "g2-standard-24"  # 2x L4
    G2_STANDARD_32 = "g2-standard-32"  # 1x L4
    G2_STANDARD_48 = "g2-standard-48"  # 4x L4
    G2_STANDARD_96 = "g2-standard-96"  # 8x L4


class AcceleratorType(str, Enum):
    """GPU accelerator types for Vertex AI.

    See: https://cloud.google.com/vertex-ai/docs/training/configure-compute#accelerator-types
    """

    NVIDIA_TESLA_K80 = "NVIDIA_TESLA_K80"
    NVIDIA_TESLA_P100 = "NVIDIA_TESLA_P100"
    NVIDIA_TESLA_V100 = "NVIDIA_TESLA_V100"
    NVIDIA_TESLA_P4 = "NVIDIA_TESLA_P4"
    NVIDIA_TESLA_T4 = "NVIDIA_TESLA_T4"
    NVIDIA_TESLA_A100 = "NVIDIA_TESLA_A100"
    NVIDIA_A100_80GB = "NVIDIA_A100_80GB"
    NVIDIA_H100_80GB = "NVIDIA_H100_80GB"
    NVIDIA_L4 = "NVIDIA_L4"
    TPU_V2 = "TPU_V2"
    TPU_V3 = "TPU_V3"
    TPU_V4_POD = "TPU_V4_POD"


class VertexRegion(str, Enum):
    """Google Cloud regions that support Vertex AI training.

    See: https://cloud.google.com/vertex-ai/docs/general/locations
    """

    # Americas
    US_CENTRAL1 = "us-central1"
    US_EAST1 = "us-east1"
    US_EAST4 = "us-east4"
    US_SOUTH1 = "us-south1"
    US_WEST1 = "us-west1"
    US_WEST2 = "us-west2"
    US_WEST3 = "us-west3"
    US_WEST4 = "us-west4"
    NORTHAMERICA_NORTHEAST1 = "northamerica-northeast1"
    NORTHAMERICA_NORTHEAST2 = "northamerica-northeast2"
    SOUTHAMERICA_EAST1 = "southamerica-east1"

    # Europe
    EUROPE_WEST1 = "europe-west1"
    EUROPE_WEST2 = "europe-west2"
    EUROPE_WEST3 = "europe-west3"
    EUROPE_WEST4 = "europe-west4"
    EUROPE_WEST6 = "europe-west6"
    EUROPE_WEST8 = "europe-west8"
    EUROPE_WEST9 = "europe-west9"
    EUROPE_NORTH1 = "europe-north1"
    EUROPE_CENTRAL2 = "europe-central2"

    # Asia Pacific
    ASIA_EAST1 = "asia-east1"
    ASIA_EAST2 = "asia-east2"
    ASIA_NORTHEAST1 = "asia-northeast1"
    ASIA_NORTHEAST2 = "asia-northeast2"
    ASIA_NORTHEAST3 = "asia-northeast3"
    ASIA_SOUTH1 = "asia-south1"
    ASIA_SOUTHEAST1 = "asia-southeast1"
    ASIA_SOUTHEAST2 = "asia-southeast2"
    AUSTRALIA_SOUTHEAST1 = "australia-southeast1"
    AUSTRALIA_SOUTHEAST2 = "australia-southeast2"

    # Middle East
    ME_WEST1 = "me-west1"
    ME_CENTRAL1 = "me-central1"


class DiskType(str, Enum):
    """Boot disk types for Vertex AI training VMs."""

    PD_STANDARD = "pd-standard"  # Standard persistent disk
    PD_SSD = "pd-ssd"  # SSD persistent disk
    PD_BALANCED = "pd-balanced"  # Balanced persistent disk


class VertexStorageConfig(BaseModel):
    """GCS storage configuration for training artifacts.

    Attributes:
        bucket_name: GCS bucket name (without gs:// prefix).
        checkpoint_prefix: Prefix path for checkpoint files.
        data_prefix: Prefix path for training data.
        staging_prefix: Prefix path for staging files.
        artifact_prefix: Prefix path for output artifacts.
        max_checkpoints: Maximum number of checkpoints to retain.
        enable_versioning: Enable GCS object versioning for safety.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    bucket_name: str = Field(
        ...,
        min_length=3,
        max_length=63,
        description="GCS bucket name (without gs:// prefix)",
    )
    checkpoint_prefix: str = Field(
        default="checkpoints/",
        description="Prefix path for checkpoint files",
    )
    data_prefix: str = Field(
        default="data/",
        description="Prefix path for training data",
    )
    staging_prefix: str = Field(
        default="staging/",
        description="Prefix path for staging files",
    )
    artifact_prefix: str = Field(
        default="artifacts/",
        description="Prefix path for output artifacts",
    )
    max_checkpoints: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of checkpoints to retain",
    )
    enable_versioning: bool = Field(
        default=True,
        description="Enable GCS object versioning for safety",
    )
    local_cache_dir: str = Field(
        default="/tmp/alphagalerkin_cache",
        description="Local cache directory for checkpoints",
    )

    @field_validator("bucket_name")
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validate GCS bucket name format."""
        # Remove gs:// prefix if present
        if v.startswith("gs://"):
            v = v[5:]
        # Remove trailing slash if present
        v = v.rstrip("/")

        # GCS bucket naming rules
        if not v[0].isalnum() or not v[-1].isalnum():
            raise ValueError("Bucket name must start and end with alphanumeric")
        if ".." in v:
            raise ValueError("Bucket name cannot contain '..'")
        return v

    @field_validator("checkpoint_prefix", "data_prefix", "staging_prefix", "artifact_prefix")
    @classmethod
    def validate_prefix(cls, v: str) -> str:
        """Ensure prefix ends with /."""
        if v and not v.endswith("/"):
            v = v + "/"
        return v

    def get_gcs_uri(self, prefix: str = "") -> str:
        """Get full GCS URI for a path.

        Args:
            prefix: Path prefix to append.

        Returns:
            Full GCS URI (gs://bucket/path).

        """
        return f"gs://{self.bucket_name}/{prefix}".rstrip("/")

    def get_checkpoint_uri(self, name: str = "") -> str:
        """Get GCS URI for checkpoint."""
        return self.get_gcs_uri(f"{self.checkpoint_prefix}{name}")

    def get_data_uri(self, name: str = "") -> str:
        """Get GCS URI for data."""
        return self.get_gcs_uri(f"{self.data_prefix}{name}")


class VertexResourceConfig(BaseModel):
    """Compute resource configuration for Vertex AI training.

    Attributes:
        machine_type: VM machine type.
        accelerator_type: GPU/TPU accelerator type.
        accelerator_count: Number of accelerators per replica.
        replica_count: Number of training replicas.
        boot_disk_type: Boot disk type.
        boot_disk_size_gb: Boot disk size in GB.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    machine_type: VertexMachineType = Field(
        default=VertexMachineType.N1_STANDARD_8,
        description="VM machine type",
    )
    accelerator_type: AcceleratorType | None = Field(
        default=None,
        description="GPU/TPU accelerator type (None for CPU only)",
    )
    accelerator_count: int = Field(
        default=0,
        ge=0,
        le=16,
        description="Number of accelerators per replica",
    )
    replica_count: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of training replicas (workers)",
    )
    boot_disk_type: DiskType = Field(
        default=DiskType.PD_SSD,
        description="Boot disk type",
    )
    boot_disk_size_gb: int = Field(
        default=200,
        ge=100,
        le=4096,
        description="Boot disk size in GB",
    )

    @model_validator(mode="after")
    def validate_accelerator_config(self) -> VertexResourceConfig:
        """Validate accelerator configuration consistency."""
        if self.accelerator_type is not None and self.accelerator_count == 0:
            # Auto-set to 1 if type specified but count is 0
            self.accelerator_count = 1
        if self.accelerator_count > 0 and self.accelerator_type is None:
            raise ValueError(
                "accelerator_type must be specified when accelerator_count > 0"
            )
        return self

    @model_validator(mode="after")
    def validate_machine_accelerator_compatibility(self) -> VertexResourceConfig:
        """Validate machine type and accelerator compatibility."""
        machine = self.machine_type.value

        # A2 machines have integrated A100 GPUs
        if (
            machine.startswith("a2-")
            and self.accelerator_type is not None
            and self.accelerator_type
            not in (AcceleratorType.NVIDIA_TESLA_A100, AcceleratorType.NVIDIA_A100_80GB)
        ):
            raise ValueError(
                f"A2 machines only support A100 GPUs, got {self.accelerator_type}"
            )

        # A3 machines have integrated H100 GPUs
        if (
            machine.startswith("a3-")
            and self.accelerator_type is not None
            and self.accelerator_type != AcceleratorType.NVIDIA_H100_80GB
        ):
            raise ValueError(
                f"A3 machines only support H100 GPUs, got {self.accelerator_type}"
            )

        # G2 machines have integrated L4 GPUs
        if (
            machine.startswith("g2-")
            and self.accelerator_type is not None
            and self.accelerator_type != AcceleratorType.NVIDIA_L4
        ):
            raise ValueError(
                f"G2 machines only support L4 GPUs, got {self.accelerator_type}"
            )

        return self

    def get_world_size(self) -> int:
        """Calculate total world size for distributed training.

        Returns:
            Total number of processes (replicas * accelerators per replica).

        """
        gpus_per_replica = max(1, self.accelerator_count)
        return self.replica_count * gpus_per_replica


class VertexNetworkConfig(BaseModel):
    """Network configuration for Vertex AI training.

    Attributes:
        network: VPC network resource name.
        subnetwork: Subnetwork resource name.
        enable_private_service_connect: Use Private Service Connect.
        reserved_ip_ranges: Reserved IP ranges for PSC.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    network: str | None = Field(
        default=None,
        description="VPC network (projects/{project}/global/networks/{network})",
    )
    subnetwork: str | None = Field(
        default=None,
        description="Subnetwork resource name",
    )
    enable_private_service_connect: bool = Field(
        default=False,
        description="Use Private Service Connect for VPC peering",
    )
    reserved_ip_ranges: list[str] = Field(
        default_factory=list,
        description="Reserved IP ranges for Private Service Connect",
    )

    @model_validator(mode="after")
    def validate_psc_config(self) -> VertexNetworkConfig:
        """Validate PSC configuration."""
        if self.enable_private_service_connect and not self.network:
            raise ValueError(
                "network must be specified when enable_private_service_connect is True"
            )
        return self


class VertexTrainingConfig(BaseModel):
    """Complete Vertex AI training configuration.

    This is the main configuration class that combines all Vertex AI
    training settings including compute resources, storage, networking,
    and training behavior.

    Attributes:
        project_id: GCP project ID.
        region: GCP region for training.
        staging_bucket: GCS bucket for staging (gs://bucket format).
        service_account: Service account email for training job.
        resources: Compute resource configuration.
        storage: GCS storage configuration.
        network: Network configuration.
        tensorboard_name: Vertex AI TensorBoard instance name.
        enable_web_access: Enable web access to training container.
        timeout_hours: Maximum training duration in hours.
        enable_spot: Use spot/preemptible VMs for cost savings.
        restart_on_preemption: Restart training on spot preemption.
        max_restarts: Maximum restart attempts on preemption.
        labels: Custom labels for the training job.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Core settings
    project_id: str = Field(
        ...,
        min_length=6,
        max_length=30,
        description="GCP project ID",
    )
    region: VertexRegion = Field(
        default=VertexRegion.US_CENTRAL1,
        description="GCP region for training",
    )
    staging_bucket: str = Field(
        ...,
        description="GCS bucket for staging (gs://bucket format)",
    )

    # Service account
    service_account: str | None = Field(
        default=None,
        description="Service account email (None uses default compute SA)",
    )

    # Sub-configurations
    resources: VertexResourceConfig = Field(
        default_factory=VertexResourceConfig,
        description="Compute resource configuration",
    )
    storage: VertexStorageConfig = Field(
        ...,
        description="GCS storage configuration",
    )
    network: VertexNetworkConfig = Field(
        default_factory=VertexNetworkConfig,
        description="Network configuration",
    )

    # Authentication settings (primitive fields to avoid circular imports)
    auth_method: str = Field(
        default="adc",
        description="Auth method: 'adc', 'service_account', or 'gcloud'",
    )
    service_account_key_path: str | None = Field(
        default=None,
        description="Path to service account JSON key file",
    )
    validate_auth_before_launch: bool = Field(
        default=True,
        description="Validate credentials before job submission",
    )

    # Observability
    tensorboard_name: str | None = Field(
        default=None,
        description="Vertex AI TensorBoard instance name",
    )
    enable_web_access: bool = Field(
        default=False,
        description="Enable web access to training container",
    )

    # Training duration
    timeout_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # 7 days max
        description="Maximum training duration in hours",
    )

    # Spot instances / cost optimization
    enable_spot: bool = Field(
        default=True,
        description="Use spot/preemptible VMs for cost savings",
    )
    restart_on_preemption: bool = Field(
        default=True,
        description="Restart training on spot preemption",
    )
    max_restarts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum restart attempts on preemption",
    )

    # Checkpointing behavior
    checkpoint_interval_minutes: int = Field(
        default=30,
        ge=5,
        le=360,
        description="Checkpoint interval in minutes",
    )
    aggressive_checkpointing_on_spot: bool = Field(
        default=True,
        description="More frequent checkpoints when using spot instances",
    )

    # Custom labels
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Custom labels for the training job",
    )

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, v: str) -> str:
        """Validate GCP project ID format."""
        if not v[0].isalpha():
            raise ValueError("Project ID must start with a letter")
        if not all(c.isalnum() or c == "-" for c in v):
            raise ValueError("Project ID can only contain letters, numbers, and hyphens")
        return v.lower()

    @field_validator("staging_bucket")
    @classmethod
    def validate_staging_bucket(cls, v: str) -> str:
        """Validate and normalize staging bucket format."""
        if not v.startswith("gs://"):
            v = f"gs://{v}"
        return v.rstrip("/")

    @field_validator("service_account")
    @classmethod
    def validate_service_account(cls, v: str | None) -> str | None:
        """Validate service account email format."""
        if v is not None and "@" not in v:
            raise ValueError("Service account must be a valid email address")
        return v

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate label format."""
        for key, value in v.items():
            if len(key) > 63 or len(value) > 63:
                raise ValueError("Label keys and values must be <= 63 characters")
            if not key[0].isalpha():
                raise ValueError(f"Label key '{key}' must start with a letter")
        return v

    @field_validator("auth_method")
    @classmethod
    def validate_auth_method(cls, v: str) -> str:
        """Validate auth method value."""
        valid_methods = {"adc", "service_account", "gcloud"}
        if v not in valid_methods:
            raise ValueError(
                f"auth_method must be one of {valid_methods}, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_spot_config(self) -> VertexTrainingConfig:
        """Validate spot instance configuration."""
        if not self.enable_spot and self.restart_on_preemption:
            import warnings
            warnings.warn(
                "restart_on_preemption is True but enable_spot is False. "
                "This setting will have no effect.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def get_effective_checkpoint_interval(self) -> int:
        """Get checkpoint interval considering spot instance settings.

        Returns:
            Checkpoint interval in minutes.

        """
        if self.enable_spot and self.aggressive_checkpointing_on_spot:
            # Use more frequent checkpoints for spot instances
            return min(self.checkpoint_interval_minutes, 10)
        return self.checkpoint_interval_minutes

    def get_timeout_seconds(self) -> int:
        """Get timeout in seconds.

        Returns:
            Training timeout in seconds.

        """
        return self.timeout_hours * 3600

    def get_auth_config(self) -> Any:
        """Get authentication configuration as AuthConfig instance.

        Returns:
            AuthConfig instance for credential validation.

        """
        # Lazy import to avoid circular dependency
        from src.vertex.auth import AuthConfig as AuthConfigClass
        from src.vertex.auth import AuthMethod

        return AuthConfigClass(
            auth_method=AuthMethod(self.auth_method),
            service_account_key_path=self.service_account_key_path,
            project_id=self.project_id,
            validate_before_launch=self.validate_auth_before_launch,
        )

    def to_environment_vars(self) -> dict[str, str]:
        """Convert to environment variables for training container.

        Returns:
            Dictionary of environment variable name-value pairs.

        """
        env_vars = {
            "VERTEX_PROJECT_ID": self.project_id,
            "VERTEX_REGION": self.region.value,
            "VERTEX_STAGING_BUCKET": self.staging_bucket,
            "VERTEX_STORAGE_BUCKET": self.storage.bucket_name,
            "VERTEX_CHECKPOINT_PREFIX": self.storage.checkpoint_prefix,
            "VERTEX_DATA_PREFIX": self.storage.data_prefix,
            "VERTEX_ENABLE_SPOT": str(self.enable_spot).lower(),
            "VERTEX_CHECKPOINT_INTERVAL": str(self.get_effective_checkpoint_interval()),
            "VERTEX_AUTH_METHOD": self.auth_method,
        }

        # Add service account key path if configured
        if self.service_account_key_path is not None:
            env_vars["GOOGLE_APPLICATION_CREDENTIALS"] = self.service_account_key_path

        return env_vars

    @classmethod
    def from_environment(cls, **overrides: Any) -> VertexTrainingConfig:
        """Create configuration from environment variables.

        Environment variables (VERTEX_* prefix):
            VERTEX_PROJECT_ID: GCP project ID
            VERTEX_REGION: GCP region
            VERTEX_STAGING_BUCKET: Staging bucket
            VERTEX_STORAGE_BUCKET: Storage bucket name

        Args:
            **overrides: Override specific configuration values.

        Returns:
            VertexTrainingConfig instance.

        Raises:
            ValueError: If required environment variables are missing.

        """
        project_id = os.environ.get("VERTEX_PROJECT_ID", overrides.get("project_id"))
        staging_bucket = os.environ.get("VERTEX_STAGING_BUCKET", overrides.get("staging_bucket"))
        storage_bucket = os.environ.get("VERTEX_STORAGE_BUCKET")
        region = os.environ.get("VERTEX_REGION", "us-central1")

        if not project_id:
            raise ValueError("VERTEX_PROJECT_ID environment variable is required")
        if not staging_bucket:
            raise ValueError("VERTEX_STAGING_BUCKET environment variable is required")
        if not storage_bucket:
            storage_bucket = staging_bucket.replace("gs://", "")

        # Build storage config
        storage = VertexStorageConfig(
            bucket_name=storage_bucket,
            checkpoint_prefix=os.environ.get("VERTEX_CHECKPOINT_PREFIX", "checkpoints/"),
            data_prefix=os.environ.get("VERTEX_DATA_PREFIX", "data/"),
        )

        # Merge with overrides
        config_dict = {
            "project_id": project_id,
            "region": region,
            "staging_bucket": staging_bucket,
            "storage": storage,
            **overrides,
        }

        return cls(**config_dict)


def create_vertex_config(
    project_id: str,
    bucket_name: str,
    machine_type: str = "n1-standard-8",
    accelerator_type: str | None = None,
    accelerator_count: int = 0,
    region: str = "us-central1",
    enable_spot: bool = True,
    **kwargs: Any,
) -> VertexTrainingConfig:
    """Factory function to create Vertex AI training configuration.

    Provides a simpler interface for common configuration patterns.

    Args:
        project_id: GCP project ID.
        bucket_name: GCS bucket name for storage.
        machine_type: VM machine type.
        accelerator_type: GPU accelerator type (None for CPU).
        accelerator_count: Number of GPUs.
        region: GCP region.
        enable_spot: Use spot instances.
        **kwargs: Additional configuration options.

    Returns:
        Configured VertexTrainingConfig instance.

    Example:
        config = create_vertex_config(
            project_id="my-project",
            bucket_name="my-bucket",
            machine_type="a2-highgpu-1g",
            accelerator_type="NVIDIA_TESLA_A100",
            accelerator_count=1,
        )

    """
    # Parse machine type
    machine_enum = VertexMachineType(machine_type)

    # Parse accelerator type if specified
    accel_enum = None
    if accelerator_type:
        accel_enum = AcceleratorType(accelerator_type)

    # Parse region
    region_enum = VertexRegion(region)

    # Build resource config
    resources = VertexResourceConfig(
        machine_type=machine_enum,
        accelerator_type=accel_enum,
        accelerator_count=accelerator_count,
    )

    # Build storage config
    storage = VertexStorageConfig(bucket_name=bucket_name)

    return VertexTrainingConfig(
        project_id=project_id,
        region=region_enum,
        staging_bucket=f"gs://{bucket_name}",
        resources=resources,
        storage=storage,
        enable_spot=enable_spot,
        **kwargs,
    )
