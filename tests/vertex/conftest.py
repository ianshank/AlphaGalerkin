"""Test fixtures for Vertex AI tests."""

from __future__ import annotations

import sys
from collections.abc import Generator
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

# Mock google.cloud module tree if not installed
_google_mock = MagicMock()
_google_cloud_mock = MagicMock()
_aiplatform_mock = MagicMock()
_storage_mock = MagicMock()

# Set up module hierarchy
_google_cloud_mock.aiplatform = _aiplatform_mock
_google_cloud_mock.storage = _storage_mock
_google_mock.cloud = _google_cloud_mock

# Install the mocks. Previous logic only installed when ``"google" not
# in sys.modules``, but ``google`` is a PEP 420 implicit namespace
# package that can be imported by any upstream test (hypothesis /
# huggingface-hub / wandb / etc. all do). Once the real, empty
# namespace is already in ``sys.modules`` without a ``cloud``
# attribute, ``unittest.mock.patch("google.cloud.storage.Client")``
# fails during target resolution with ``AttributeError: module
# 'google' has no attribute 'cloud'`` — which is what the 3.10
# CI matrix was hitting but 3.11/3.12 happened to miss because a
# different upstream test happened to import differently there.
_existing_google = sys.modules.get("google")
if _existing_google is None:
    sys.modules["google"] = _google_mock
elif not hasattr(_existing_google, "cloud"):
    # Real namespace package is already imported (empty); attach our
    # cloud mock so importlib attribute resolution for google.cloud.*
    # succeeds.
    _existing_google.cloud = _google_cloud_mock  # type: ignore[attr-defined]
sys.modules["google.cloud"] = _google_cloud_mock
sys.modules["google.cloud.aiplatform"] = _aiplatform_mock
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
