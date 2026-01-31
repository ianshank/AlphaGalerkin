"""Tests for Vertex AI launcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.vertex.config import (
    AcceleratorType,
    VertexMachineType,
    VertexResourceConfig,
    VertexStorageConfig,
    VertexTrainingConfig,
)
from src.vertex.launcher import (
    JobState,
    JobStatus,
    VertexLauncher,
    VertexLaunchResult,
    create_launcher,
)

class TestJobState:
    """Tests for JobState enum."""

    def test_is_terminal(self) -> None:
        """Test terminal state detection."""
        assert JobState.SUCCEEDED.is_terminal is True
        assert JobState.FAILED.is_terminal is True
        assert JobState.CANCELLED.is_terminal is True
        assert JobState.EXPIRED.is_terminal is True
        assert JobState.RUNNING.is_terminal is False
        assert JobState.PENDING.is_terminal is False

    def test_is_running(self) -> None:
        """Test running state detection."""
        assert JobState.RUNNING.is_running is True
        assert JobState.UPDATING.is_running is True
        assert JobState.PENDING.is_running is False
        assert JobState.SUCCEEDED.is_running is False

    def test_is_pending(self) -> None:
        """Test pending state detection."""
        assert JobState.QUEUED.is_pending is True
        assert JobState.PENDING.is_pending is True
        assert JobState.RUNNING.is_pending is False


class TestVertexLaunchResult:
    """Tests for VertexLaunchResult."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = VertexLaunchResult(
            job_name="test-job",
            job_id="projects/proj/locations/us/customJobs/123",
            console_url="https://console.cloud.google.com/...",
            state=JobState.RUNNING,
            create_time="2026-01-01T00:00:00",
            labels={"env": "test"},
        )
        d = result.to_dict()
        assert d["job_name"] == "test-job"
        assert d["state"] == "JOB_STATE_RUNNING"
        assert d["labels"]["env"] == "test"


class TestJobStatus:
    """Tests for JobStatus."""

    def test_creation(self) -> None:
        """Test job status creation."""
        status = JobStatus(
            job_id="job-123",
            state=JobState.RUNNING,
            state_message="Job is running",
            start_time="2026-01-01T00:00:00",
        )
        assert status.job_id == "job-123"
        assert status.state == JobState.RUNNING
        assert status.error_message is None


class TestVertexLauncher:
    """Tests for VertexLauncher."""

    @pytest.fixture
    def sample_config(self) -> VertexTrainingConfig:
        """Create sample configuration."""
        return VertexTrainingConfig(
            project_id="test-project",
            staging_bucket="gs://test-bucket",
            resources=VertexResourceConfig(
                machine_type=VertexMachineType.A2_HIGHGPU_1G,
                accelerator_type=AcceleratorType.NVIDIA_TESLA_A100,
                accelerator_count=1,
            ),
            storage=VertexStorageConfig(bucket_name="test-bucket"),
        )

    @pytest.fixture
    def launcher(self, sample_config: VertexTrainingConfig) -> VertexLauncher:
        """Create launcher instance."""
        return VertexLauncher(sample_config)

    def test_initialization(self, launcher: VertexLauncher) -> None:
        """Test launcher initialization."""
        assert launcher.config.project_id == "test-project"
        assert launcher._initialized is False

    def test_build_machine_spec_with_gpu(
        self,
        launcher: VertexLauncher,
    ) -> None:
        """Test machine spec building with GPU."""
        spec = launcher._build_machine_spec()
        assert spec["machine_type"] == "a2-highgpu-1g"
        assert spec["accelerator_type"] == "NVIDIA_TESLA_A100"
        assert spec["accelerator_count"] == 1

    def test_build_machine_spec_cpu_only(self) -> None:
        """Test machine spec building for CPU only."""
        config = VertexTrainingConfig(
            project_id="test-project",
            staging_bucket="gs://test-bucket",
            resources=VertexResourceConfig(
                machine_type=VertexMachineType.N1_STANDARD_8,
            ),
            storage=VertexStorageConfig(bucket_name="test-bucket"),
        )
        launcher = VertexLauncher(config)
        spec = launcher._build_machine_spec()
        assert spec["machine_type"] == "n1-standard-8"
        assert "accelerator_type" not in spec
        assert "accelerator_count" not in spec

    def test_build_worker_pool_specs(
        self,
        launcher: VertexLauncher,
    ) -> None:
        """Test worker pool spec building."""
        machine_spec = launcher._build_machine_spec()
        specs = launcher._build_worker_pool_specs(
            container_uri="gcr.io/project/image:tag",
            args=["--config", "train.yaml"],
            environment_variables={"KEY": "value"},
            machine_spec=machine_spec,
        )

        assert len(specs) == 1
        spec = specs[0]
        assert spec["machine_spec"] == machine_spec
        assert spec["replica_count"] == 1
        assert spec["container_spec"]["image_uri"] == "gcr.io/project/image:tag"
        assert spec["container_spec"]["args"] == ["--config", "train.yaml"]
        assert {"name": "KEY", "value": "value"} in spec["container_spec"]["env"]

    def test_build_console_url(self, launcher: VertexLauncher) -> None:
        """Test console URL building."""
        resource_name = "projects/test-project/locations/us-central1/customJobs/123456"
        url = launcher._build_console_url(resource_name)
        assert "console.cloud.google.com" in url
        assert "123456" in url
        assert "test-project" in url

    def test_build_console_url_fallback(self, launcher: VertexLauncher) -> None:
        """Test console URL fallback for invalid resource name."""
        url = launcher._build_console_url("invalid-resource-name")
        assert "console.cloud.google.com" in url
        assert "test-project" in url

    @patch("google.cloud.aiplatform")
    def test_launch_calls_aiplatform(
        self,
        mock_aiplatform: MagicMock,
        launcher: VertexLauncher,
    ) -> None:
        """Test launch calls AI Platform SDK correctly."""
        # Mock the job
        mock_job = MagicMock()
        mock_job.display_name = "test-job"
        mock_job.resource_name = "projects/test/locations/us/customJobs/123"
        mock_job.state = MagicMock()
        mock_job.state.name = "JOB_STATE_PENDING"
        mock_job.create_time = None

        mock_aiplatform.CustomJob.return_value = mock_job

        # Mark as initialized to skip aiplatform.init
        launcher._initialized = True

        result = launcher.launch(
            display_name="test-job",
            container_uri="gcr.io/project/image:tag",
            args=["--arg1", "value1"],
        )

        assert result.job_name == "test-job"
        mock_job.run.assert_called_once()

    @patch("google.cloud.aiplatform")
    def test_get_job_status(
        self,
        mock_aiplatform: MagicMock,
        launcher: VertexLauncher,
    ) -> None:
        """Test getting job status."""
        mock_job = MagicMock()
        mock_job.resource_name = "job-123"
        mock_job.state = MagicMock()
        mock_job.state.name = "JOB_STATE_RUNNING"
        mock_job.start_time = None
        mock_job.end_time = None
        mock_job.error = None

        mock_aiplatform.CustomJob.get.return_value = mock_job

        launcher._initialized = True
        status = launcher.get_job_status("job-123")

        assert status.job_id == "job-123"
        assert status.state == JobState.RUNNING

    @patch("google.cloud.aiplatform")
    def test_cancel_job(
        self,
        mock_aiplatform: MagicMock,
        launcher: VertexLauncher,
    ) -> None:
        """Test job cancellation."""
        mock_job = MagicMock()
        mock_aiplatform.CustomJob.get.return_value = mock_job

        launcher._initialized = True
        result = launcher.cancel_job("job-123")

        assert result is True
        mock_job.cancel.assert_called_once()

    @patch("google.cloud.aiplatform")
    def test_cancel_job_failure(
        self,
        mock_aiplatform: MagicMock,
        launcher: VertexLauncher,
    ) -> None:
        """Test job cancellation failure."""
        mock_aiplatform.CustomJob.get.side_effect = Exception("API error")

        launcher._initialized = True
        result = launcher.cancel_job("job-123")

        assert result is False

    @patch("google.cloud.aiplatform")
    def test_list_jobs(
        self,
        mock_aiplatform: MagicMock,
        launcher: VertexLauncher,
    ) -> None:
        """Test listing jobs."""
        mock_jobs = []
        for i in range(3):
            job = MagicMock()
            job.display_name = f"job-{i}"
            job.resource_name = f"projects/test/locations/us/customJobs/{i}"
            job.state = MagicMock()
            job.state.name = "JOB_STATE_SUCCEEDED"
            job.create_time = None
            job.labels = {}
            mock_jobs.append(job)

        mock_aiplatform.CustomJob.list.return_value = mock_jobs

        launcher._initialized = True
        results = launcher.list_jobs(limit=10)

        assert len(results) == 3
        assert results[0].job_name == "job-0"


class TestCreateLauncher:
    """Tests for create_launcher factory function."""

    def test_creates_launcher(self) -> None:
        """Test factory creates launcher."""
        config = VertexTrainingConfig(
            project_id="test-project",
            staging_bucket="gs://bucket",
            storage=VertexStorageConfig(bucket_name="bucket"),
        )
        launcher = create_launcher(config)
        assert isinstance(launcher, VertexLauncher)
        assert launcher.config.project_id == "test-project"
