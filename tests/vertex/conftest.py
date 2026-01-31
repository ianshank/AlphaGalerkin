"""Test fixtures for Vertex AI tests."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from src.vertex.config import (
    AcceleratorType,
    VertexMachineType,
    VertexRegion,
    VertexResourceConfig,
    VertexStorageConfig,
    VertexTrainingConfig,
)

if TYPE_CHECKING:
    pass


# Mock google.cloud module tree if not installed
_google_mock = MagicMock()
_google_cloud_mock = MagicMock()
_aiplatform_mock = MagicMock()
_storage_mock = MagicMock()

# Set up module hierarchy
_google_cloud_mock.aiplatform = _aiplatform_mock
_google_cloud_mock.storage = _storage_mock
_google_mock.cloud = _google_cloud_mock

# Only add to sys.modules if not already installed
if "google" not in sys.modules:
    sys.modules["google"] = _google_mock
if "google.cloud" not in sys.modules:
    sys.modules["google.cloud"] = _google_cloud_mock
if "google.cloud.aiplatform" not in sys.modules:
    sys.modules["google.cloud.aiplatform"] = _aiplatform_mock
if "google.cloud.storage" not in sys.modules:
    sys.modules["google.cloud.storage"] = _storage_mock


@pytest.fixture
def sample_storage_config() -> VertexStorageConfig:
    """Create a sample storage configuration."""
    return VertexStorageConfig(
        bucket_name="test-bucket",
        checkpoint_prefix="checkpoints/",
        data_prefix="data/",
    )


@pytest.fixture
def sample_resource_config() -> VertexResourceConfig:
    """Create a sample resource configuration."""
    return VertexResourceConfig(
        machine_type=VertexMachineType.A2_HIGHGPU_1G,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
        accelerator_count=1,
    )


@pytest.fixture
def sample_vertex_config(
    sample_storage_config: VertexStorageConfig,
    sample_resource_config: VertexResourceConfig,
) -> VertexTrainingConfig:
    """Create a sample Vertex AI training configuration."""
    return VertexTrainingConfig(
        project_id="test-project",
        region=VertexRegion.US_CENTRAL1,
        staging_bucket="gs://test-bucket",
        resources=sample_resource_config,
        storage=sample_storage_config,
    )


@pytest.fixture
def mock_gcs_client() -> Generator[MagicMock, None, None]:
    """Mock Google Cloud Storage client."""
    with patch("google.cloud.storage.Client") as mock:
        client = MagicMock()
        bucket = MagicMock()
        client.bucket.return_value = bucket
        mock.return_value = client
        yield mock


@pytest.fixture
def mock_aiplatform() -> Generator[MagicMock, None, None]:
    """Mock Google Cloud AI Platform SDK."""
    with patch("google.cloud.aiplatform") as mock:
        yield mock


@pytest.fixture
def mock_blob() -> MagicMock:
    """Create a mock GCS blob."""
    blob = MagicMock()
    blob.name = "test-blob"
    blob.size = 1024
    blob.updated = "2026-01-01T00:00:00Z"
    return blob


@pytest.fixture
def cpu_only_resource_config() -> VertexResourceConfig:
    """Create a CPU-only resource configuration."""
    return VertexResourceConfig(
        machine_type=VertexMachineType.N1_STANDARD_8,
        accelerator_type=None,
        accelerator_count=0,
    )


@pytest.fixture
def multi_gpu_resource_config() -> VertexResourceConfig:
    """Create a multi-GPU resource configuration."""
    return VertexResourceConfig(
        machine_type=VertexMachineType.A2_HIGHGPU_4G,
        accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
        accelerator_count=4,
        replica_count=2,
    )


@pytest.fixture
def env_vars_for_vertex() -> dict[str, str]:
    """Environment variables for Vertex AI configuration."""
    return {
        "VERTEX_PROJECT_ID": "env-test-project",
        "VERTEX_REGION": "us-west1",
        "VERTEX_STAGING_BUCKET": "gs://env-test-bucket",
        "VERTEX_STORAGE_BUCKET": "env-test-bucket",
        "VERTEX_CHECKPOINT_PREFIX": "env-checkpoints/",
        "VERTEX_DATA_PREFIX": "env-data/",
    }
