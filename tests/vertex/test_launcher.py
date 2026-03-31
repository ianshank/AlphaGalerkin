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

# Check if aiplatform SDK is available for mocking
try:
    from google.cloud import aiplatform  # noqa: F401

    HAS_AIPLATFORM = True
except ImportError:
    HAS_AIPLATFORM = False


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
        """Create sample configuration with N1 machine and attached GPU."""
        return VertexTrainingConfig(
            project_id="test-project",
            staging_bucket="gs://test-bucket",
            resources=VertexResourceConfig(
                machine_type=VertexMachineType.N1_STANDARD_8,
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
        """Test machine spec building with GPU (N1 machine with attached accelerator)."""
        spec = launcher._build_machine_spec()
        assert spec["machine_type"] == "n1-standard-8"
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

    def test_launch_calls_aiplatform(
        self,
        launcher: VertexLauncher,
    ) -> None:
        """Test launch calls AI Platform SDK correctly."""
        mock_job = MagicMock()
        mock_job.display_name = "test-job"
        mock_job.resource_name = "projects/test/locations/us-central1/customJobs/123"
        mock_job.state = MagicMock()
        mock_job.state.name = "JOB_STATE_PENDING"
        mock_job.create_time = None

        mock_aiplatform = MagicMock()
        mock_aiplatform.CustomJob.return_value = mock_job

        # Mark as initialized to skip aiplatform.init and auth validated to skip
        # the GCP credential check (which would make real network calls).
        launcher._initialized = True
        launcher._auth_validated = True

        # Patch google.cloud.aiplatform in sys.modules so that
        # `from google.cloud import aiplatform` inside the method gets our mock.
        # Also patch the google.cloud package's attribute lookup.
        import google.cloud as google_cloud_pkg
        with patch.dict("sys.modules", {"google.cloud.aiplatform": mock_aiplatform}):
            with patch.object(google_cloud_pkg, "aiplatform", mock_aiplatform, create=True):
                result = launcher.launch(
                    display_name="test-job",
                    container_uri="gcr.io/project/image:tag",
                    args=["--arg1", "value1"],
                )

        assert result.job_name == "test-job"
        mock_job.submit.assert_called_once()

    def test_get_job_status(
        self,
        launcher: VertexLauncher,
    ) -> None:
        """Test getting job status."""
        mock_job = MagicMock()
        mock_job.resource_name = "projects/test/locations/us-central1/customJobs/456"
        mock_job.state = MagicMock()
        mock_job.state.name = "JOB_STATE_RUNNING"
        mock_job.start_time = None
        mock_job.end_time = None
        mock_job.error = None

        mock_aiplatform = MagicMock()
        mock_aiplatform.CustomJob.get.return_value = mock_job

        launcher._initialized = True

        import google.cloud as google_cloud_pkg
        with patch.dict("sys.modules", {"google.cloud.aiplatform": mock_aiplatform}):
            with patch.object(google_cloud_pkg, "aiplatform", mock_aiplatform, create=True):
                status = launcher.get_job_status(
                    "projects/test/locations/us-central1/customJobs/456"
                )

        assert status.state == JobState.RUNNING
        assert status.error_message is None
        mock_aiplatform.CustomJob.get.assert_called_once_with(
            "projects/test/locations/us-central1/customJobs/456"
        )

    def test_cancel_job(
        self,
        launcher: VertexLauncher,
    ) -> None:
        """Test job cancellation."""
        mock_job = MagicMock()

        mock_aiplatform = MagicMock()
        mock_aiplatform.CustomJob.get.return_value = mock_job

        launcher._initialized = True

        import google.cloud as google_cloud_pkg
        with patch.dict("sys.modules", {"google.cloud.aiplatform": mock_aiplatform}):
            with patch.object(google_cloud_pkg, "aiplatform", mock_aiplatform, create=True):
                success = launcher.cancel_job(
                    "projects/test/locations/us-central1/customJobs/789"
                )

        assert success is True
        mock_job.cancel.assert_called_once()

    def test_cancel_job_failure(
        self,
        launcher: VertexLauncher,
    ) -> None:
        """Test job cancellation failure."""
        mock_aiplatform = MagicMock()
        mock_aiplatform.CustomJob.get.side_effect = Exception("Job not found")

        launcher._initialized = True

        import google.cloud as google_cloud_pkg
        with patch.dict("sys.modules", {"google.cloud.aiplatform": mock_aiplatform}):
            with patch.object(google_cloud_pkg, "aiplatform", mock_aiplatform, create=True):
                success = launcher.cancel_job(
                    "projects/test/locations/us-central1/customJobs/999"
                )

        assert success is False

    def test_list_jobs(
        self,
        launcher: VertexLauncher,
    ) -> None:
        """Test listing jobs."""
        mock_job1 = MagicMock()
        mock_job1.display_name = "job-1"
        mock_job1.resource_name = "projects/test/locations/us-central1/customJobs/1"
        mock_job1.state = MagicMock()
        mock_job1.state.name = "JOB_STATE_SUCCEEDED"
        mock_job1.create_time = None
        mock_job1.labels = {"env": "test"}

        mock_job2 = MagicMock()
        mock_job2.display_name = "job-2"
        mock_job2.resource_name = "projects/test/locations/us-central1/customJobs/2"
        mock_job2.state = MagicMock()
        mock_job2.state.name = "JOB_STATE_RUNNING"
        mock_job2.create_time = None
        mock_job2.labels = {}

        mock_aiplatform = MagicMock()
        mock_aiplatform.CustomJob.list.return_value = [mock_job1, mock_job2]

        launcher._initialized = True

        import google.cloud as google_cloud_pkg
        with patch.dict("sys.modules", {"google.cloud.aiplatform": mock_aiplatform}):
            with patch.object(google_cloud_pkg, "aiplatform", mock_aiplatform, create=True):
                results = launcher.list_jobs()

        assert len(results) == 2
        assert results[0].job_name == "job-1"
        assert results[0].state == JobState.SUCCEEDED
        assert results[1].job_name == "job-2"
        assert results[1].state == JobState.RUNNING
        mock_aiplatform.CustomJob.list.assert_called_once()


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


class TestLauncherAuthIntegration:
    """Tests for launcher authentication integration."""

    @pytest.fixture
    def launcher_with_auth(self) -> VertexLauncher:
        """Create launcher with auth validation enabled."""
        config = VertexTrainingConfig(
            project_id="test-project",
            staging_bucket="gs://bucket",
            storage=VertexStorageConfig(bucket_name="bucket"),
            validate_auth_before_launch=True,
        )
        return VertexLauncher(config)

    @pytest.fixture
    def launcher_without_auth(self) -> VertexLauncher:
        """Create launcher with auth validation disabled."""
        config = VertexTrainingConfig(
            project_id="test-project",
            staging_bucket="gs://bucket",
            storage=VertexStorageConfig(bucket_name="bucket"),
            validate_auth_before_launch=False,
        )
        return VertexLauncher(config)

    def test_auth_validated_flag_initial(self, launcher_with_auth: VertexLauncher) -> None:
        """Test auth_validated flag is initially False."""
        assert launcher_with_auth._auth_validated is False

    def test_validate_auth_method(self, launcher_with_auth: VertexLauncher) -> None:
        """Test validate_auth method returns ValidationResult."""
        from src.vertex.auth import ValidationResult

        with patch("src.vertex.launcher.GCPAuthenticator") as mock_auth_class:
            mock_auth = MagicMock()
            mock_auth.validate_credentials.return_value = ValidationResult(
                is_valid=True,
                account="test@example.com",
                project="test-project",
            )
            mock_auth_class.return_value = mock_auth

            result = launcher_with_auth.validate_auth()

            assert result.is_valid is True
            assert result.account == "test@example.com"
            mock_auth.validate_credentials.assert_called_once()

    def test_ensure_authenticated_success(self, launcher_with_auth: VertexLauncher) -> None:
        """Test _ensure_authenticated sets flag on success."""
        from src.vertex.auth import ValidationResult

        with patch("src.vertex.launcher.GCPAuthenticator") as mock_auth_class:
            mock_auth = MagicMock()
            mock_auth.validate_credentials.return_value = ValidationResult(
                is_valid=True,
                account="test@example.com",
                project="test-project",
            )
            mock_auth_class.return_value = mock_auth

            launcher_with_auth._ensure_authenticated()

            assert launcher_with_auth._auth_validated is True

    def test_ensure_authenticated_failure_raises(self, launcher_with_auth: VertexLauncher) -> None:
        """Test _ensure_authenticated raises on failure when validation enabled."""
        from src.vertex.auth import ValidationResult
        from src.vertex.launcher import AuthenticationError

        with patch("src.vertex.launcher.GCPAuthenticator") as mock_auth_class:
            mock_auth = MagicMock()
            mock_auth.validate_credentials.return_value = ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message="No credentials found",
                error_code="NO_ADC",
            )
            mock_auth_class.return_value = mock_auth

            with pytest.raises(AuthenticationError) as exc_info:
                launcher_with_auth._ensure_authenticated()

            assert "No credentials found" in str(exc_info.value)
            assert launcher_with_auth._auth_validated is False

    def test_ensure_authenticated_cached(self, launcher_with_auth: VertexLauncher) -> None:
        """Test _ensure_authenticated uses cached result."""
        # Pre-set the validated flag
        launcher_with_auth._auth_validated = True

        with patch("src.vertex.launcher.GCPAuthenticator") as mock_auth_class:
            result = launcher_with_auth._ensure_authenticated()

            # Should not call authenticator since already validated
            mock_auth_class.assert_not_called()
            assert result.is_valid is True

    def test_ensure_authenticated_disabled_no_raise(
        self, launcher_without_auth: VertexLauncher
    ) -> None:
        """Test _ensure_authenticated doesn't raise when validation disabled."""
        from src.vertex.auth import ValidationResult

        with patch("src.vertex.launcher.GCPAuthenticator") as mock_auth_class:
            mock_auth = MagicMock()
            mock_auth.validate_credentials.return_value = ValidationResult(
                is_valid=False,
                account=None,
                project=None,
                error_message="No credentials",
                error_code="NO_ADC",
            )
            mock_auth_class.return_value = mock_auth

            # Should not raise even with invalid credentials
            result = launcher_without_auth._ensure_authenticated()
            assert result.is_valid is False
