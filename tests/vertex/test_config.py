"""Tests for Vertex AI configuration schemas."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

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

class TestVertexStorageConfig:
    """Tests for VertexStorageConfig."""

    def test_valid_bucket_name(self) -> None:
        """Test valid bucket name."""
        config = VertexStorageConfig(bucket_name="my-test-bucket")
        assert config.bucket_name == "my-test-bucket"

    def test_bucket_name_strips_gs_prefix(self) -> None:
        """Test bucket name strips gs:// prefix."""
        config = VertexStorageConfig(bucket_name="gs://my-bucket")
        assert config.bucket_name == "my-bucket"

    def test_bucket_name_strips_trailing_slash(self) -> None:
        """Test bucket name strips trailing slash."""
        config = VertexStorageConfig(bucket_name="my-bucket/")
        assert config.bucket_name == "my-bucket"

    def test_invalid_bucket_name_start(self) -> None:
        """Test invalid bucket name starting with non-alphanumeric."""
        with pytest.raises(ValidationError, match="must start and end with alphanumeric"):
            VertexStorageConfig(bucket_name="-invalid-bucket")

    def test_invalid_bucket_name_end(self) -> None:
        """Test invalid bucket name ending with non-alphanumeric."""
        with pytest.raises(ValidationError, match="must start and end with alphanumeric"):
            VertexStorageConfig(bucket_name="invalid-bucket-")

    def test_invalid_bucket_name_double_dot(self) -> None:
        """Test invalid bucket name with double dots."""
        with pytest.raises(ValidationError, match="cannot contain"):
            VertexStorageConfig(bucket_name="invalid..bucket")

    def test_prefix_adds_trailing_slash(self) -> None:
        """Test prefix auto-adds trailing slash."""
        config = VertexStorageConfig(
            bucket_name="bucket",
            checkpoint_prefix="checkpoints",
        )
        assert config.checkpoint_prefix == "checkpoints/"

    def test_default_prefixes(self) -> None:
        """Test default prefix values."""
        config = VertexStorageConfig(bucket_name="bucket")
        assert config.checkpoint_prefix == "checkpoints/"
        assert config.data_prefix == "data/"
        assert config.staging_prefix == "staging/"
        assert config.artifact_prefix == "artifacts/"

    def test_get_gcs_uri(self) -> None:
        """Test GCS URI generation."""
        config = VertexStorageConfig(bucket_name="my-bucket")
        assert config.get_gcs_uri("path/to/file") == "gs://my-bucket/path/to/file"
        assert config.get_gcs_uri() == "gs://my-bucket"

    def test_get_checkpoint_uri(self) -> None:
        """Test checkpoint URI generation."""
        config = VertexStorageConfig(bucket_name="my-bucket")
        assert config.get_checkpoint_uri("model.pt") == "gs://my-bucket/checkpoints/model.pt"

    def test_max_checkpoints_validation(self) -> None:
        """Test max checkpoints validation."""
        with pytest.raises(ValidationError):
            VertexStorageConfig(bucket_name="bucket", max_checkpoints=0)
        with pytest.raises(ValidationError):
            VertexStorageConfig(bucket_name="bucket", max_checkpoints=101)


class TestVertexResourceConfig:
    """Tests for VertexResourceConfig."""

    def test_default_values(self) -> None:
        """Test default resource configuration."""
        config = VertexResourceConfig()
        assert config.machine_type == VertexMachineType.N1_STANDARD_8
        assert config.accelerator_type is None
        assert config.accelerator_count == 0
        assert config.replica_count == 1
        assert config.boot_disk_type == DiskType.PD_SSD
        assert config.boot_disk_size_gb == 200

    def test_valid_gpu_config(self) -> None:
        """Test valid GPU configuration."""
        config = VertexResourceConfig(
            machine_type=VertexMachineType.N1_STANDARD_8,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_T4,
            accelerator_count=4,
        )
        assert config.accelerator_count == 4

    def test_accelerator_type_with_zero_count_auto_sets(self) -> None:
        """Test accelerator type with zero count auto-sets to 1."""
        config = VertexResourceConfig(
            machine_type=VertexMachineType.N1_STANDARD_8,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_T4,
            accelerator_count=0,
        )
        assert config.accelerator_count == 1

    def test_accelerator_count_without_type_fails(self) -> None:
        """Test accelerator count without type fails."""
        with pytest.raises(ValidationError, match="accelerator_type must be specified"):
            VertexResourceConfig(
                machine_type=VertexMachineType.N1_STANDARD_8,
                accelerator_type=None,
                accelerator_count=4,
            )

    def test_a2_machine_requires_a100(self) -> None:
        """Test A2 machine requires A100 GPU."""
        with pytest.raises(ValidationError, match="A2 machines only support A100"):
            VertexResourceConfig(
                machine_type=VertexMachineType.A2_HIGHGPU_1G,
                accelerator_type=AcceleratorType.NVIDIA_TESLA_T4,
                accelerator_count=1,
            )

    def test_a2_machine_with_a100_valid(self) -> None:
        """Test A2 machine with A100 is valid."""
        config = VertexResourceConfig(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
            accelerator_count=1,
        )
        assert config.machine_type == VertexMachineType.A2_HIGHGPU_1G

    def test_a3_machine_requires_h100(self) -> None:
        """Test A3 machine requires H100 GPU."""
        with pytest.raises(ValidationError, match="A3 machines only support H100"):
            VertexResourceConfig(
                machine_type=VertexMachineType.A3_HIGHGPU_8G,
                accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
                accelerator_count=1,
            )

    def test_g2_machine_requires_l4(self) -> None:
        """Test G2 machine requires L4 GPU."""
        with pytest.raises(ValidationError, match="G2 machines only support L4"):
            VertexResourceConfig(
                machine_type=VertexMachineType.G2_STANDARD_8,
                accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
                accelerator_count=1,
            )

    def test_get_world_size_single_gpu(self) -> None:
        """Test world size calculation for single GPU."""
        config = VertexResourceConfig(
            machine_type=VertexMachineType.A2_HIGHGPU_1G,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
            accelerator_count=1,
            replica_count=1,
        )
        assert config.get_world_size() == 1

    def test_get_world_size_multi_gpu(self) -> None:
        """Test world size calculation for multi-GPU."""
        config = VertexResourceConfig(
            machine_type=VertexMachineType.A2_HIGHGPU_4G,
            accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
            accelerator_count=4,
            replica_count=2,
        )
        assert config.get_world_size() == 8

    def test_get_world_size_cpu_only(self) -> None:
        """Test world size calculation for CPU-only."""
        config = VertexResourceConfig(
            machine_type=VertexMachineType.N1_STANDARD_8,
            replica_count=4,
        )
        assert config.get_world_size() == 4


class TestVertexNetworkConfig:
    """Tests for VertexNetworkConfig."""

    def test_default_values(self) -> None:
        """Test default network configuration."""
        config = VertexNetworkConfig()
        assert config.network is None
        assert config.enable_private_service_connect is False

    def test_psc_requires_network(self) -> None:
        """Test PSC requires network to be specified."""
        with pytest.raises(ValidationError, match="network must be specified"):
            VertexNetworkConfig(
                enable_private_service_connect=True,
                network=None,
            )

    def test_psc_with_network_valid(self) -> None:
        """Test PSC with network is valid."""
        config = VertexNetworkConfig(
            network="projects/my-project/global/networks/my-network",
            enable_private_service_connect=True,
        )
        assert config.enable_private_service_connect is True


class TestVertexTrainingConfig:
    """Tests for VertexTrainingConfig."""

    def test_valid_minimal_config(
        self,
        sample_storage_config: VertexStorageConfig,
    ) -> None:
        """Test valid minimal configuration."""
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://my-bucket",
            storage=sample_storage_config,
        )
        assert config.project_id == "my-project"
        assert config.region == VertexRegion.US_CENTRAL1

    def test_project_id_validation_lowercase(self) -> None:
        """Test project ID is lowercased."""
        config = VertexTrainingConfig(
            project_id="My-Project",
            staging_bucket="gs://bucket",
            storage=VertexStorageConfig(bucket_name="bucket"),
        )
        assert config.project_id == "my-project"

    def test_project_id_must_start_with_letter(self) -> None:
        """Test project ID must start with letter."""
        with pytest.raises(ValidationError, match="must start with a letter"):
            VertexTrainingConfig(
                project_id="123-project",
                staging_bucket="gs://bucket",
                storage=VertexStorageConfig(bucket_name="bucket"),
            )

    def test_staging_bucket_adds_gs_prefix(self) -> None:
        """Test staging bucket adds gs:// prefix."""
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="my-bucket",
            storage=VertexStorageConfig(bucket_name="my-bucket"),
        )
        assert config.staging_bucket == "gs://my-bucket"

    def test_service_account_validation(self) -> None:
        """Test service account email validation."""
        with pytest.raises(ValidationError, match="valid email"):
            VertexTrainingConfig(
                project_id="my-project",
                staging_bucket="gs://bucket",
                storage=VertexStorageConfig(bucket_name="bucket"),
                service_account="invalid-service-account",
            )

    def test_valid_service_account(self) -> None:
        """Test valid service account."""
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://bucket",
            storage=VertexStorageConfig(bucket_name="bucket"),
            service_account="sa@project.iam.gserviceaccount.com",
        )
        assert config.service_account == "sa@project.iam.gserviceaccount.com"

    def test_labels_validation_length(self) -> None:
        """Test labels key/value length validation."""
        with pytest.raises(ValidationError, match="<= 63 characters"):
            VertexTrainingConfig(
                project_id="my-project",
                staging_bucket="gs://bucket",
                storage=VertexStorageConfig(bucket_name="bucket"),
                labels={"a" * 64: "value"},
            )

    def test_labels_key_must_start_with_letter(self) -> None:
        """Test labels key must start with letter."""
        with pytest.raises(ValidationError, match="must start with a letter"):
            VertexTrainingConfig(
                project_id="my-project",
                staging_bucket="gs://bucket",
                storage=VertexStorageConfig(bucket_name="bucket"),
                labels={"123key": "value"},
            )

    def test_timeout_validation(self) -> None:
        """Test timeout hour validation."""
        with pytest.raises(ValidationError):
            VertexTrainingConfig(
                project_id="my-project",
                staging_bucket="gs://bucket",
                storage=VertexStorageConfig(bucket_name="bucket"),
                timeout_hours=200,  # Max is 168
            )

    def test_get_effective_checkpoint_interval_non_spot(
        self,
        sample_storage_config: VertexStorageConfig,
    ) -> None:
        """Test checkpoint interval for non-spot instances."""
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://bucket",
            storage=sample_storage_config,
            enable_spot=False,
            checkpoint_interval_minutes=30,
        )
        assert config.get_effective_checkpoint_interval() == 30

    def test_get_effective_checkpoint_interval_spot(
        self,
        sample_storage_config: VertexStorageConfig,
    ) -> None:
        """Test checkpoint interval for spot instances (more aggressive)."""
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://bucket",
            storage=sample_storage_config,
            enable_spot=True,
            aggressive_checkpointing_on_spot=True,
            checkpoint_interval_minutes=30,
        )
        assert config.get_effective_checkpoint_interval() == 10

    def test_get_timeout_seconds(
        self,
        sample_storage_config: VertexStorageConfig,
    ) -> None:
        """Test timeout seconds conversion."""
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://bucket",
            storage=sample_storage_config,
            timeout_hours=24,
        )
        assert config.get_timeout_seconds() == 86400

    def test_to_environment_vars(
        self,
        sample_storage_config: VertexStorageConfig,
    ) -> None:
        """Test environment variable conversion."""
        config = VertexTrainingConfig(
            project_id="my-project",
            region=VertexRegion.US_WEST1,
            staging_bucket="gs://my-bucket",
            storage=sample_storage_config,
        )
        env = config.to_environment_vars()
        assert env["VERTEX_PROJECT_ID"] == "my-project"
        assert env["VERTEX_REGION"] == "us-west1"
        assert env["VERTEX_STAGING_BUCKET"] == "gs://my-bucket"

    def test_from_environment(
        self,
        env_vars_for_vertex: dict[str, str],
    ) -> None:
        """Test configuration from environment variables."""
        # Set environment variables
        for key, value in env_vars_for_vertex.items():
            os.environ[key] = value

        try:
            config = VertexTrainingConfig.from_environment()
            assert config.project_id == "env-test-project"
            assert config.region == VertexRegion.US_WEST1
            assert config.staging_bucket == "gs://env-test-bucket"
        finally:
            # Clean up environment variables
            for key in env_vars_for_vertex:
                os.environ.pop(key, None)

    def test_from_environment_missing_project_id(self) -> None:
        """Test error when project ID is missing."""
        # Ensure env vars are not set
        os.environ.pop("VERTEX_PROJECT_ID", None)
        os.environ.pop("VERTEX_STAGING_BUCKET", None)

        with pytest.raises(ValueError, match="VERTEX_PROJECT_ID"):
            VertexTrainingConfig.from_environment()


class TestCreateVertexConfig:
    """Tests for create_vertex_config factory function."""

    def test_basic_config(self) -> None:
        """Test basic configuration creation."""
        config = create_vertex_config(
            project_id="my-project",
            bucket_name="my-bucket",
        )
        assert config.project_id == "my-project"
        assert config.storage.bucket_name == "my-bucket"

    def test_gpu_config(self) -> None:
        """Test GPU configuration creation."""
        config = create_vertex_config(
            project_id="my-project",
            bucket_name="my-bucket",
            machine_type="a2-highgpu-1g",
            accelerator_type="NVIDIA_TESLA_A100",
            accelerator_count=1,
        )
        assert config.resources.machine_type == VertexMachineType.A2_HIGHGPU_1G
        assert config.resources.accelerator_type == AcceleratorType.NVIDIA_TESLA_A100
        assert config.resources.accelerator_count == 1

    def test_region_config(self) -> None:
        """Test region configuration."""
        config = create_vertex_config(
            project_id="my-project",
            bucket_name="my-bucket",
            region="europe-west1",
        )
        assert config.region == VertexRegion.EUROPE_WEST1

    def test_spot_config(self) -> None:
        """Test spot instance configuration."""
        config = create_vertex_config(
            project_id="my-project",
            bucket_name="my-bucket",
            enable_spot=False,
        )
        assert config.enable_spot is False


class TestVertexMachineType:
    """Tests for VertexMachineType enum."""

    def test_all_machine_types_have_values(self) -> None:
        """Test all machine types have string values."""
        for machine_type in VertexMachineType:
            assert isinstance(machine_type.value, str)
            assert len(machine_type.value) > 0

    def test_a2_machines_start_with_a2(self) -> None:
        """Test A2 machines start with a2-."""
        a2_types = [m for m in VertexMachineType if m.name.startswith("A2_")]
        for machine_type in a2_types:
            assert machine_type.value.startswith("a2-")


class TestAcceleratorType:
    """Tests for AcceleratorType enum."""

    def test_all_accelerator_types_have_values(self) -> None:
        """Test all accelerator types have string values."""
        for accel_type in AcceleratorType:
            assert isinstance(accel_type.value, str)
            assert len(accel_type.value) > 0

    def test_nvidia_types_start_with_nvidia(self) -> None:
        """Test NVIDIA types start with NVIDIA_."""
        nvidia_types = [a for a in AcceleratorType if a.name.startswith("NVIDIA_")]
        for accel_type in nvidia_types:
            assert accel_type.value.startswith("NVIDIA_")


class TestAuthIntegration:
    """Tests for auth integration in VertexTrainingConfig."""

    @pytest.fixture
    def sample_storage_config(self) -> VertexStorageConfig:
        """Create sample storage config."""
        return VertexStorageConfig(bucket_name="test-bucket")

    def test_default_auth_method(
        self, sample_storage_config: VertexStorageConfig
    ) -> None:
        """Test default auth method is ADC."""
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://bucket",
            storage=sample_storage_config,
        )
        assert config.auth_method == "adc"
        assert config.service_account_key_path is None
        assert config.validate_auth_before_launch is True

    def test_gcloud_auth_method(
        self, sample_storage_config: VertexStorageConfig
    ) -> None:
        """Test gcloud auth method configuration."""
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://bucket",
            storage=sample_storage_config,
            auth_method="gcloud",
        )
        assert config.auth_method == "gcloud"

    def test_service_account_auth_method(
        self, sample_storage_config: VertexStorageConfig, tmp_path: Path
    ) -> None:
        """Test service account auth method configuration."""
        key_file = tmp_path / "key.json"
        key_file.write_text('{"type": "service_account"}')

        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://bucket",
            storage=sample_storage_config,
            auth_method="service_account",
            service_account_key_path=str(key_file),
        )
        assert config.auth_method == "service_account"
        assert config.service_account_key_path == str(key_file)

    def test_invalid_auth_method(
        self, sample_storage_config: VertexStorageConfig
    ) -> None:
        """Test invalid auth method is rejected."""
        with pytest.raises(ValidationError, match="auth_method must be one of"):
            VertexTrainingConfig(
                project_id="my-project",
                staging_bucket="gs://bucket",
                storage=sample_storage_config,
                auth_method="invalid",
            )

    def test_validate_auth_before_launch_false(
        self, sample_storage_config: VertexStorageConfig
    ) -> None:
        """Test disabling auth validation before launch."""
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://bucket",
            storage=sample_storage_config,
            validate_auth_before_launch=False,
        )
        assert config.validate_auth_before_launch is False

    def test_get_auth_config(
        self, sample_storage_config: VertexStorageConfig
    ) -> None:
        """Test get_auth_config creates AuthConfig instance."""
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://bucket",
            storage=sample_storage_config,
            auth_method="gcloud",
        )
        auth_config = config.get_auth_config()

        from src.vertex.auth import AuthConfig, AuthMethod

        assert isinstance(auth_config, AuthConfig)
        assert auth_config.auth_method == AuthMethod.GCLOUD_CLI
        assert auth_config.project_id == "my-project"

    def test_to_environment_vars_includes_auth(
        self, sample_storage_config: VertexStorageConfig
    ) -> None:
        """Test environment variables include auth settings."""
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://bucket",
            storage=sample_storage_config,
            auth_method="gcloud",
        )
        env_vars = config.to_environment_vars()
        assert env_vars["VERTEX_AUTH_METHOD"] == "gcloud"

    def test_to_environment_vars_with_service_account(
        self, sample_storage_config: VertexStorageConfig, tmp_path: Path
    ) -> None:
        """Test environment variables include service account key path."""
        key_path = str(tmp_path / "key.json")
        config = VertexTrainingConfig(
            project_id="my-project",
            staging_bucket="gs://bucket",
            storage=sample_storage_config,
            auth_method="service_account",
            service_account_key_path=key_path,
        )
        env_vars = config.to_environment_vars()
        assert env_vars["GOOGLE_APPLICATION_CREDENTIALS"] == key_path
