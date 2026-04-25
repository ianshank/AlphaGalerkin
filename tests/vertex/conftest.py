"""Test fixtures for Vertex AI tests.

The ``google-cloud-storage`` and ``google-cloud-aiplatform`` packages are in
the optional ``[vertex]`` extra, so the default CI matrix (`[dev]`) runs
without them.  This conftest stubs the ``google.cloud.*`` module tree so
``unittest.mock.patch("google.cloud.storage.Client")`` resolves cleanly.

Historical bug (fixed): an earlier version only installed the stubs when
``"google" not in sys.modules``.  That is insufficient on Python 3.10
because packages like ``wandb`` pull in the ``google`` namespace package
transitively — so ``sys.modules["google"]`` already exists, but the real
module has no ``cloud`` attribute, and ``patch("google.cloud.storage.Client")``
fails with ``AttributeError: module 'google' has no attribute 'cloud'``
during test setup.

The fix: if ``google-cloud-storage`` is not actually importable, install
our mock tree via ``sys.modules`` **and** set the ``cloud`` attribute on
whatever ``google`` object is live so attribute walks used by ``patch``
succeed regardless of how ``google`` was originally loaded.
"""

from __future__ import annotations

import importlib.util
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


def _google_cloud_storage_installed() -> bool:
    """True when the real ``google-cloud-storage`` package can be imported.

    ``importlib.util.find_spec`` raises ``ModuleNotFoundError`` when a
    parent package in the dotted path is itself missing, so we catch it
    and treat it the same as "not installed".
    """
    try:
        return importlib.util.find_spec("google.cloud.storage") is not None
    except (ModuleNotFoundError, ValueError):
        return False


if not _google_cloud_storage_installed():
    _google_mock = MagicMock(name="google")
    _google_cloud_mock = MagicMock(name="google.cloud")
    _aiplatform_mock = MagicMock(name="google.cloud.aiplatform")
    _storage_mock = MagicMock(name="google.cloud.storage")

    _google_cloud_mock.aiplatform = _aiplatform_mock
    _google_cloud_mock.storage = _storage_mock
    _google_mock.cloud = _google_cloud_mock

    # Force-install the stub tree.  We do NOT guard behind
    # ``"google" not in sys.modules`` because other deps (e.g. wandb) may
    # have already imported the real namespace package — in that case we
    # still need ``google.cloud`` to be discoverable.
    sys.modules["google"] = _google_mock
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
