"""Vertex AI job launcher for AlphaGalerkin training.

This module provides utilities for launching and managing training jobs
on Google Cloud Vertex AI, including job submission, status monitoring,
and cancellation.

Example:
    from src.vertex.launcher import VertexLauncher
    from src.vertex.config import VertexTrainingConfig

    config = VertexTrainingConfig(...)
    launcher = VertexLauncher(config)

    result = launcher.launch(
        display_name="alphagalerkin-run-001",
        container_uri="gcr.io/my-project/trainer:latest",
        args=["--config", "config.yaml"],
    )
    print(f"Job launched: {result.console_url}")

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from src.vertex.config import VertexTrainingConfig

logger = structlog.get_logger(__name__)


class JobState(str, Enum):
    """Vertex AI job states."""

    UNSPECIFIED = "JOB_STATE_UNSPECIFIED"
    QUEUED = "JOB_STATE_QUEUED"
    PENDING = "JOB_STATE_PENDING"
    RUNNING = "JOB_STATE_RUNNING"
    SUCCEEDED = "JOB_STATE_SUCCEEDED"
    FAILED = "JOB_STATE_FAILED"
    CANCELLING = "JOB_STATE_CANCELLING"
    CANCELLED = "JOB_STATE_CANCELLED"
    PAUSED = "JOB_STATE_PAUSED"
    EXPIRED = "JOB_STATE_EXPIRED"
    UPDATING = "JOB_STATE_UPDATING"

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.EXPIRED,
        )

    @property
    def is_running(self) -> bool:
        """Check if job is actively running."""
        return self in (JobState.RUNNING, JobState.UPDATING)

    @property
    def is_pending(self) -> bool:
        """Check if job is waiting to start."""
        return self in (JobState.QUEUED, JobState.PENDING)


@dataclass
class VertexLaunchResult:
    """Result of launching a Vertex AI training job.

    Attributes:
        job_name: Display name of the job.
        job_id: Full resource name of the job.
        console_url: URL to the job in GCP Console.
        state: Current job state.
        create_time: Job creation timestamp.
        labels: Job labels.

    """

    job_name: str
    job_id: str
    console_url: str
    state: JobState
    create_time: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_name": self.job_name,
            "job_id": self.job_id,
            "console_url": self.console_url,
            "state": self.state.value,
            "create_time": self.create_time,
            "labels": self.labels,
        }


@dataclass
class JobStatus:
    """Current status of a Vertex AI job.

    Attributes:
        job_id: Full resource name of the job.
        state: Current job state.
        state_message: Additional state information.
        start_time: When job started running.
        end_time: When job finished.
        error_message: Error message if failed.

    """

    job_id: str
    state: JobState
    state_message: str = ""
    start_time: str | None = None
    end_time: str | None = None
    error_message: str | None = None


class VertexLauncher:
    """Launcher for Vertex AI training jobs.

    This class handles the creation and management of Vertex AI
    custom training jobs, including multi-node distributed training.

    Example:
        config = VertexTrainingConfig(...)
        launcher = VertexLauncher(config)

        # Launch job
        result = launcher.launch(
            display_name="my-training-job",
            container_uri="gcr.io/my-project/trainer:latest",
        )

        # Monitor job
        status = launcher.get_job_status(result.job_id)
        if status.state == JobState.FAILED:
            print(f"Job failed: {status.error_message}")

    """

    def __init__(self, config: VertexTrainingConfig) -> None:
        """Initialize Vertex AI launcher.

        Args:
            config: Vertex AI training configuration.

        """
        self.config = config
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize Vertex AI SDK if not already done."""
        if self._initialized:
            return

        try:
            from google.cloud import aiplatform
            aiplatform.init(
                project=self.config.project_id,
                location=self.config.region.value,
                staging_bucket=self.config.staging_bucket,
            )
            self._initialized = True
            logger.info(
                "vertex_ai_initialized",
                project=self.config.project_id,
                region=self.config.region.value,
            )
        except ImportError as e:
            raise ImportError(
                "google-cloud-aiplatform is required for Vertex AI operations. "
                "Install with: pip install google-cloud-aiplatform"
            ) from e

    def launch(
        self,
        display_name: str,
        container_uri: str,
        args: list[str] | None = None,
        environment_variables: dict[str, str] | None = None,
        sync: bool = False,
    ) -> VertexLaunchResult:
        """Launch a training job on Vertex AI.

        Args:
            display_name: Human-readable name for the job.
            container_uri: Container image URI (gcr.io or Artifact Registry).
            args: Command-line arguments for the container.
            environment_variables: Additional environment variables.
            sync: If True, block until job completes.

        Returns:
            VertexLaunchResult with job details.

        """
        self._ensure_initialized()
        from google.cloud import aiplatform

        args = args or []
        environment_variables = environment_variables or {}

        # Merge with config environment variables
        env_vars = {**self.config.to_environment_vars(), **environment_variables}

        # Build machine spec
        machine_spec = self._build_machine_spec()

        # Build worker pool specs
        worker_pool_specs = self._build_worker_pool_specs(
            container_uri=container_uri,
            args=args,
            environment_variables=env_vars,
            machine_spec=machine_spec,
        )

        # Merge labels
        labels = {
            "alphagalerkin": "true",
            "spot": str(self.config.enable_spot).lower(),
            **self.config.labels,
        }

        logger.info(
            "launching_vertex_job",
            display_name=display_name,
            container_uri=container_uri,
            machine_type=self.config.resources.machine_type.value,
            accelerator_type=self.config.resources.accelerator_type.value if self.config.resources.accelerator_type else None,
            accelerator_count=self.config.resources.accelerator_count,
            replica_count=self.config.resources.replica_count,
            enable_spot=self.config.enable_spot,
        )

        # Create custom job
        job = aiplatform.CustomJob(
            display_name=display_name,
            worker_pool_specs=worker_pool_specs,
            labels=labels,
            staging_bucket=self.config.staging_bucket,
        )

        # Run job
        job.run(
            service_account=self.config.service_account,
            network=self.config.network.network,
            timeout=self.config.get_timeout_seconds(),
            restart_job_on_worker_restart=self.config.restart_on_preemption,
            enable_web_access=self.config.enable_web_access,
            tensorboard=self.config.tensorboard_name,
            sync=sync,
        )

        # Build result
        result = VertexLaunchResult(
            job_name=job.display_name,
            job_id=job.resource_name,
            console_url=self._build_console_url(job.resource_name),
            state=JobState(job.state.name) if job.state else JobState.PENDING,
            create_time=job.create_time.isoformat() if job.create_time else "",
            labels=labels,
        )

        logger.info(
            "vertex_job_launched",
            job_name=result.job_name,
            job_id=result.job_id,
            console_url=result.console_url,
        )

        return result

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get current status of a job.

        Args:
            job_id: Full resource name of the job.

        Returns:
            JobStatus with current state.

        """
        self._ensure_initialized()
        from google.cloud import aiplatform

        job = aiplatform.CustomJob.get(job_id)

        error_message = None
        if job.error and job.error.message:
            error_message = job.error.message

        return JobStatus(
            job_id=job.resource_name,
            state=JobState(job.state.name),
            state_message=str(job.state),
            start_time=job.start_time.isoformat() if job.start_time else None,
            end_time=job.end_time.isoformat() if job.end_time else None,
            error_message=error_message,
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Full resource name of the job.

        Returns:
            True if cancellation was requested successfully.

        """
        self._ensure_initialized()
        from google.cloud import aiplatform

        try:
            job = aiplatform.CustomJob.get(job_id)
            job.cancel()
            logger.info("job_cancellation_requested", job_id=job_id)
            return True
        except Exception as e:
            logger.error("job_cancellation_failed", job_id=job_id, error=str(e))
            return False

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 30.0,
        timeout: float | None = None,
    ) -> JobStatus:
        """Wait for a job to complete.

        Args:
            job_id: Full resource name of the job.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait (None for no limit).

        Returns:
            Final job status.

        Raises:
            TimeoutError: If timeout exceeded.

        """
        start_time = time.time()

        while True:
            status = self.get_job_status(job_id)

            if status.state.is_terminal:
                logger.info(
                    "job_completed",
                    job_id=job_id,
                    state=status.state.value,
                    duration=time.time() - start_time,
                )
                return status

            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout} seconds"
                )

            logger.debug(
                "waiting_for_job",
                job_id=job_id,
                state=status.state.value,
                elapsed=time.time() - start_time,
            )
            time.sleep(poll_interval)

    def list_jobs(
        self,
        filter_str: str | None = None,
        limit: int = 20,
    ) -> list[VertexLaunchResult]:
        """List training jobs.

        Args:
            filter_str: Filter expression (e.g., 'display_name="my-job"').
            limit: Maximum number of jobs to return.

        Returns:
            List of job results.

        """
        self._ensure_initialized()
        from google.cloud import aiplatform

        jobs = aiplatform.CustomJob.list(
            filter=filter_str,
            order_by="create_time desc",
        )

        results = []
        for job in jobs[:limit]:
            result = VertexLaunchResult(
                job_name=job.display_name,
                job_id=job.resource_name,
                console_url=self._build_console_url(job.resource_name),
                state=JobState(job.state.name) if job.state else JobState.UNSPECIFIED,
                create_time=job.create_time.isoformat() if job.create_time else "",
                labels=dict(job.labels) if job.labels else {},
            )
            results.append(result)

        return results

    def _build_machine_spec(self) -> dict[str, Any]:
        """Build machine specification for the job."""
        spec: dict[str, Any] = {
            "machine_type": self.config.resources.machine_type.value,
        }

        if self.config.resources.accelerator_type is not None:
            spec["accelerator_type"] = self.config.resources.accelerator_type.value
            spec["accelerator_count"] = self.config.resources.accelerator_count

        return spec

    def _build_worker_pool_specs(
        self,
        container_uri: str,
        args: list[str],
        environment_variables: dict[str, str],
        machine_spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build worker pool specifications."""
        # Convert env vars to list format
        env_list = [
            {"name": k, "value": v}
            for k, v in environment_variables.items()
        ]

        container_spec = {
            "image_uri": container_uri,
            "args": args,
            "env": env_list,
        }

        disk_spec = {
            "boot_disk_type": self.config.resources.boot_disk_type.value,
            "boot_disk_size_gb": self.config.resources.boot_disk_size_gb,
        }

        worker_pool_spec = {
            "machine_spec": machine_spec,
            "replica_count": self.config.resources.replica_count,
            "container_spec": container_spec,
            "disk_spec": disk_spec,
        }

        return [worker_pool_spec]

    def _build_console_url(self, resource_name: str) -> str:
        """Build GCP Console URL for a job."""
        # resource_name format: projects/{project}/locations/{location}/customJobs/{job_id}
        parts = resource_name.split("/")
        if len(parts) >= 6:
            job_id = parts[-1]
            location = parts[3]
            return (
                f"https://console.cloud.google.com/vertex-ai/training/"
                f"{job_id}/cpu?project={self.config.project_id}&region={location}"
            )
        return f"https://console.cloud.google.com/vertex-ai/training?project={self.config.project_id}"


def create_launcher(config: VertexTrainingConfig) -> VertexLauncher:
    """Factory function to create a Vertex AI launcher.

    Args:
        config: Vertex AI training configuration.

    Returns:
        Configured VertexLauncher instance.

    """
    return VertexLauncher(config)
