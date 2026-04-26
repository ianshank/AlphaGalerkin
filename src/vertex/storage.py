"""GCS storage integration for Vertex AI training.

This module provides GCS-backed checkpoint management with local caching,
enabling efficient save/load of training state during Vertex AI jobs.

Features:
    - Transparent GCS read/write with automatic retry
    - Local caching for checkpoint restoration
    - Streaming upload/download for large checkpoints
    - Atomic operations to prevent corruption
    - Checkpoint rotation to manage storage

Example:
    from src.vertex.config import VertexStorageConfig
    from src.vertex.storage import GCSCheckpointManager

    config = VertexStorageConfig(bucket_name="my-training-bucket")
    manager = GCSCheckpointManager(
        bucket_name=config.bucket_name,
        checkpoint_prefix="experiments/run-001/",
        local_cache_dir=Path(config.local_cache_dir),
    )

    # Save checkpoint
    gcs_path = manager.save(step=1000, model=model, optimizer=optimizer)

    # Load checkpoint
    state = manager.load(gcs_path)
    model.load_state_dict(state["model_state_dict"])

"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.constants import CHECKPOINT_BEST
from src.vertex.config import VertexStorageConfig

if TYPE_CHECKING:
    from google.cloud.storage import Bucket, Client
    from torch import nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

logger = structlog.get_logger(__name__)

# Checkpoint format version
GCS_CHECKPOINT_VERSION = "1.0.0"

# Default retry settings
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_RETRY_MULTIPLIER = 2.0
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks


def _get_torch() -> Any:
    """Lazily import torch to avoid import errors when not installed."""
    import torch

    return torch


@dataclass
class GCSCheckpointMetadata:
    """Metadata for a GCS checkpoint.

    Attributes:
        step: Training step number.
        gcs_path: Full GCS path (gs://bucket/path).
        local_path: Local cached path if available.
        timestamp: Creation timestamp.
        size_bytes: Checkpoint size in bytes.
        md5_hash: MD5 hash for integrity verification.
        metrics: Training metrics at checkpoint time.

    """

    step: int
    gcs_path: str
    local_path: Path | None = None
    timestamp: str = ""
    size_bytes: int = 0
    md5_hash: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "gcs_path": self.gcs_path,
            "local_path": str(self.local_path) if self.local_path else None,
            "timestamp": self.timestamp,
            "size_bytes": self.size_bytes,
            "md5_hash": self.md5_hash,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GCSCheckpointMetadata:
        """Create from dictionary."""
        local_path = data.get("local_path")
        return cls(
            step=data["step"],
            gcs_path=data["gcs_path"],
            local_path=Path(local_path) if local_path else None,
            timestamp=data.get("timestamp", ""),
            size_bytes=data.get("size_bytes", 0),
            md5_hash=data.get("md5_hash", ""),
            metrics=data.get("metrics", {}),
        )


def _with_retry(
    func: Callable[[], Any],
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_RETRY_DELAY,
    multiplier: float = DEFAULT_RETRY_MULTIPLIER,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """Execute function with exponential backoff retry.

    Args:
        func: Function to execute.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        multiplier: Multiplier for exponential backoff.
        exceptions: Tuple of exception types to catch and retry.

    Returns:
        Result of function execution.

    Raises:
        Exception: If all retry attempts fail.

    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    "operation_failed_retrying",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(e),
                )
                time.sleep(delay)
                delay *= multiplier
            else:
                logger.error(
                    "operation_failed_all_retries_exhausted",
                    attempts=max_retries + 1,
                    error=str(e),
                )

    raise last_exception


class GCSCheckpointManager:
    """GCS-backed checkpoint manager with local caching.

    This manager provides efficient checkpoint save/load operations
    for Vertex AI training, with features including:
    - Transparent GCS read/write with retry
    - Local caching for fast checkpoint restoration
    - Atomic uploads to prevent corruption
    - Checkpoint rotation to manage storage costs

    Example:
        manager = GCSCheckpointManager(
            bucket_name="my-bucket",
            checkpoint_prefix="training/run-001/",
        )

        # Save checkpoint
        path = manager.save(step=1000, model=model, optimizer=optimizer)

        # List checkpoints
        checkpoints = manager.list_checkpoints()

        # Load latest
        state = manager.load_latest()

    """

    def __init__(
        self,
        bucket_name: str,
        checkpoint_prefix: str = "checkpoints/",
        local_cache_dir: Path | None = None,
        max_checkpoints: int = 5,
        max_retries: int = DEFAULT_MAX_RETRIES,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Initialize GCS checkpoint manager.

        Args:
            bucket_name: GCS bucket name.
            checkpoint_prefix: Prefix path for checkpoints.
            local_cache_dir: Local cache directory (auto-created if None).
            max_checkpoints: Maximum checkpoints to retain.
            max_retries: Maximum retry attempts for GCS operations.
            chunk_size: Chunk size for streaming uploads.

        """
        self.bucket_name = bucket_name
        self.checkpoint_prefix = checkpoint_prefix.rstrip("/") + "/"
        self.max_checkpoints = max_checkpoints
        self.max_retries = max_retries
        self.chunk_size = chunk_size

        # Initialize local cache
        if local_cache_dir is None:
            local_cache_dir = Path(tempfile.mkdtemp(prefix="vertex_checkpoint_"))
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded GCS client
        self._client: Client | None = None
        self._bucket: Bucket | None = None

        # Track best checkpoint
        self._best_value: float | None = None
        self._best_mode: str = "min"
        self._best_metric: str = "loss"

        logger.info(
            "gcs_checkpoint_manager_initialized",
            bucket=bucket_name,
            prefix=checkpoint_prefix,
            local_cache=str(local_cache_dir),
            max_checkpoints=max_checkpoints,
        )

    @classmethod
    def from_config(cls, config: VertexStorageConfig) -> GCSCheckpointManager:
        """Create manager from VertexStorageConfig.

        Args:
            config: Storage configuration.

        Returns:
            Configured GCSCheckpointManager instance.

        """
        return cls(
            bucket_name=config.bucket_name,
            checkpoint_prefix=config.checkpoint_prefix,
            local_cache_dir=Path(config.local_cache_dir),
            max_checkpoints=config.max_checkpoints,
        )

    @property
    def client(self) -> Client:
        """Get or create GCS client (lazy initialization)."""
        if self._client is None:
            try:
                from google.cloud import storage

                self._client = storage.Client()
            except ImportError as e:
                raise ImportError(
                    "google-cloud-storage is required for GCS operations. "
                    "Install with: pip install google-cloud-storage"
                ) from e
        return self._client

    @property
    def bucket(self) -> Bucket:
        """Get or create bucket reference."""
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        config: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Save checkpoint to GCS.

        Saves checkpoint atomically by first writing to a temp file,
        then uploading to GCS with retry logic.

        Args:
            step: Training step number.
            model: Model to save.
            optimizer: Optimizer state to save.
            scheduler: LR scheduler state to save.
            config: Configuration dictionary.
            metrics: Training metrics.
            extra: Additional data to include.

        Returns:
            GCS path of saved checkpoint (gs://bucket/path).

        """
        metrics = metrics or {}
        extra = extra or {}

        # Build checkpoint state
        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": config,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "version": GCS_CHECKPOINT_VERSION,
            **extra,
        }

        # Save to local temp file first
        checkpoint_name = f"checkpoint_{step:08d}.pt"
        local_path = self.local_cache_dir / checkpoint_name
        temp_path = local_path.with_suffix(".pt.tmp")

        logger.debug("saving_checkpoint_locally", path=str(temp_path), step=step)
        _get_torch().save(state, temp_path)

        # Calculate MD5 for integrity
        md5_hash = self._calculate_md5(temp_path)

        # Rename temp to final local path
        temp_path.replace(local_path)

        # Upload to GCS
        gcs_path = f"{self.checkpoint_prefix}{checkpoint_name}"
        self._upload_file(local_path, gcs_path)

        full_gcs_path = f"gs://{self.bucket_name}/{gcs_path}"

        logger.info(
            "checkpoint_saved_to_gcs",
            step=step,
            gcs_path=full_gcs_path,
            size_bytes=local_path.stat().st_size,
            md5=md5_hash,
            metrics=metrics,
        )

        # Update best checkpoint if applicable
        if self._best_metric in metrics:
            self._update_best(step, local_path, metrics[self._best_metric])

        # Rotate old checkpoints
        self._rotate_checkpoints()

        return full_gcs_path

    def load(
        self,
        gcs_path: str | None = None,
        step: int | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Load checkpoint from GCS.

        Args:
            gcs_path: Full GCS path (gs://bucket/path) or relative path.
            step: Specific step to load (alternative to gcs_path).
            use_cache: Use local cache if available.

        Returns:
            Checkpoint state dictionary.

        Raises:
            FileNotFoundError: If checkpoint not found.
            ValueError: If neither gcs_path nor step provided.

        """
        if gcs_path is None and step is None:
            raise ValueError("Either gcs_path or step must be provided")

        # Resolve path
        if gcs_path is not None:
            # Strip gs://bucket/ prefix if present
            if gcs_path.startswith("gs://"):
                parts = gcs_path.replace("gs://", "").split("/", 1)
                if len(parts) == 2:
                    gcs_path = parts[1]
        else:
            gcs_path = f"{self.checkpoint_prefix}checkpoint_{step:08d}.pt"

        # Check local cache first
        checkpoint_name = Path(gcs_path).name
        local_path = self.local_cache_dir / checkpoint_name

        if use_cache and local_path.exists():
            logger.debug("loading_from_cache", path=str(local_path))
            return _get_torch().load(local_path, map_location="cpu", weights_only=False)

        # Download from GCS
        self._download_file(gcs_path, local_path)

        # Load and return
        state = _get_torch().load(local_path, map_location="cpu", weights_only=False)

        logger.info(
            "checkpoint_loaded_from_gcs",
            gcs_path=gcs_path,
            step=state.get("step"),
            version=state.get("version"),
        )

        return state

    def load_latest(self) -> dict[str, Any]:
        """Load the most recent checkpoint.

        Returns:
            Checkpoint state dictionary.

        Raises:
            FileNotFoundError: If no checkpoints found.

        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found at gs://{self.bucket_name}/{self.checkpoint_prefix}"
            )

        latest = checkpoints[-1]
        return self.load(gcs_path=latest.gcs_path)

    def load_best(self) -> dict[str, Any]:
        """Load the best checkpoint.

        Returns:
            Checkpoint state dictionary.

        Raises:
            FileNotFoundError: If no best checkpoint found.

        """
        best_path = f"{self.checkpoint_prefix}best.pt"
        return self.load(gcs_path=best_path)

    def list_checkpoints(self) -> list[GCSCheckpointMetadata]:
        """List all checkpoints in GCS.

        Returns:
            List of checkpoint metadata, sorted by step.

        """
        prefix = self.checkpoint_prefix
        blobs = self.bucket.list_blobs(prefix=prefix)

        checkpoints = []
        for blob in blobs:
            name = blob.name
            # Skip non-checkpoint files
            if not name.endswith(".pt") or "checkpoint_" not in name:
                continue
            if name.endswith(CHECKPOINT_BEST):
                continue

            # Extract step from name
            try:
                step_str = name.split("checkpoint_")[1].replace(".pt", "")
                step = int(step_str)
            except (IndexError, ValueError):
                continue

            metadata = GCSCheckpointMetadata(
                step=step,
                gcs_path=f"gs://{self.bucket_name}/{name}",
                timestamp=blob.updated.isoformat() if blob.updated else "",
                size_bytes=blob.size or 0,
                md5_hash=blob.md5_hash or "",
            )
            checkpoints.append(metadata)

        # Sort by step
        checkpoints.sort(key=lambda x: x.step)
        return checkpoints

    def get_latest_step(self) -> int | None:
        """Get the step number of the latest checkpoint.

        Returns:
            Latest step number, or None if no checkpoints.

        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1].step if checkpoints else None

    def exists(self, step: int) -> bool:
        """Check if checkpoint exists for a given step.

        Args:
            step: Training step number.

        Returns:
            True if checkpoint exists.

        """
        gcs_path = f"{self.checkpoint_prefix}checkpoint_{step:08d}.pt"
        blob = self.bucket.blob(gcs_path)
        return blob.exists()

    def delete(self, step: int) -> bool:
        """Delete checkpoint for a given step.

        Args:
            step: Training step number.

        Returns:
            True if deleted, False if not found.

        """
        gcs_path = f"{self.checkpoint_prefix}checkpoint_{step:08d}.pt"
        blob = self.bucket.blob(gcs_path)

        if blob.exists():
            blob.delete()
            logger.info("checkpoint_deleted", step=step, gcs_path=gcs_path)

            # Also delete local cache
            local_path = self.local_cache_dir / f"checkpoint_{step:08d}.pt"
            if local_path.exists():
                local_path.unlink()

            return True
        return False

    def sync_to_gcs(self, local_path: Path, gcs_path: str | None = None) -> str:
        """Upload a local file to GCS.

        Args:
            local_path: Local file path.
            gcs_path: GCS destination path (uses filename if None).

        Returns:
            Full GCS path (gs://bucket/path).

        """
        if gcs_path is None:
            gcs_path = f"{self.checkpoint_prefix}{local_path.name}"

        self._upload_file(local_path, gcs_path)
        return f"gs://{self.bucket_name}/{gcs_path}"

    def sync_from_gcs(self, gcs_path: str, local_path: Path | None = None) -> Path:
        """Download a file from GCS.

        Args:
            gcs_path: GCS source path.
            local_path: Local destination (auto-generated if None).

        Returns:
            Local file path.

        """
        if local_path is None:
            filename = Path(gcs_path).name
            local_path = self.local_cache_dir / filename

        self._download_file(gcs_path, local_path)
        return local_path

    def clear_cache(self) -> int:
        """Clear local cache directory.

        Returns:
            Number of files removed.

        """
        count = 0
        for file in self.local_cache_dir.glob("*.pt"):
            file.unlink()
            count += 1
        for file in self.local_cache_dir.glob("*.pt.tmp"):
            file.unlink()
            count += 1

        logger.info("cache_cleared", files_removed=count)
        return count

    def set_best_tracking(
        self,
        metric: str = "loss",
        mode: str = "min",
    ) -> None:
        """Configure best checkpoint tracking.

        Args:
            metric: Metric name to track.
            mode: "min" or "max" for best value determination.

        """
        self._best_metric = metric
        self._best_mode = mode
        self._best_value = None

    def _upload_file(self, local_path: Path, gcs_path: str) -> None:
        """Upload file to GCS with retry."""
        blob = self.bucket.blob(gcs_path)

        def upload() -> None:
            blob.upload_from_filename(
                str(local_path),
                timeout=300,
            )

        _with_retry(
            upload,
            max_retries=self.max_retries,
            exceptions=(Exception,),
        )

        logger.debug(
            "file_uploaded_to_gcs",
            local=str(local_path),
            gcs=gcs_path,
            size=local_path.stat().st_size,
        )

    def _download_file(self, gcs_path: str, local_path: Path) -> None:
        """Download file from GCS with retry."""
        # Strip gs:// prefix if present
        if gcs_path.startswith(f"gs://{self.bucket_name}/"):
            gcs_path = gcs_path.replace(f"gs://{self.bucket_name}/", "")

        blob = self.bucket.blob(gcs_path)

        if not blob.exists():
            raise FileNotFoundError(f"GCS object not found: gs://{self.bucket_name}/{gcs_path}")

        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download to temp then rename
        temp_path = local_path.with_suffix(".tmp")

        def download() -> None:
            blob.download_to_filename(str(temp_path), timeout=300)

        _with_retry(
            download,
            max_retries=self.max_retries,
            exceptions=(Exception,),
        )

        temp_path.replace(local_path)

        logger.debug(
            "file_downloaded_from_gcs",
            gcs=gcs_path,
            local=str(local_path),
            size=local_path.stat().st_size,
        )

    def _calculate_md5(self, path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _update_best(self, step: int, local_path: Path, metric_value: float) -> None:
        """Update best checkpoint if metric improved."""
        is_better = False

        if (
            self._best_value is None
            or (self._best_mode == "min" and metric_value < self._best_value)
            or (self._best_mode == "max" and metric_value > self._best_value)
        ):
            is_better = True

        if is_better:
            self._best_value = metric_value

            # Copy to best checkpoint locally
            best_local = self.local_cache_dir / CHECKPOINT_BEST
            shutil.copy2(local_path, best_local)

            # Upload to GCS
            best_gcs = f"{self.checkpoint_prefix}best.pt"
            self._upload_file(best_local, best_gcs)

            logger.info(
                "best_checkpoint_updated",
                step=step,
                metric=self._best_metric,
                value=metric_value,
            )

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            to_delete = checkpoints[: -self.max_checkpoints]
            for ckpt in to_delete:
                try:
                    self.delete(ckpt.step)
                except Exception as e:
                    logger.warning(
                        "failed_to_delete_old_checkpoint",
                        step=ckpt.step,
                        error=str(e),
                    )


class GCSDataSource:
    """Stream training data from GCS.

    Provides utilities for loading training data shards from GCS,
    supporting both full downloads and streaming reads.

    Example:
        source = GCSDataSource(
            bucket_name="my-bucket",
            prefix="training-data/",
        )

        for shard in source.list_shards():
            data = source.load_shard(shard)
            for batch in data:
                train(batch)

    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "data/",
        local_cache_dir: Path | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Initialize GCS data source.

        Args:
            bucket_name: GCS bucket name.
            prefix: Prefix path for data files.
            local_cache_dir: Local cache directory.
            max_retries: Maximum retry attempts.

        """
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        self.max_retries = max_retries

        if local_cache_dir is None:
            local_cache_dir = Path(tempfile.mkdtemp(prefix="vertex_data_"))
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        self._client: Client | None = None
        self._bucket: Bucket | None = None

    @property
    def client(self) -> Client:
        """Get or create GCS client."""
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client()
        return self._client

    @property
    def bucket(self) -> Bucket:
        """Get bucket reference."""
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    def list_shards(self, pattern: str = "*.pt") -> list[str]:
        """List available data shards.

        Args:
            pattern: Glob pattern for shard files.

        Returns:
            List of shard paths relative to prefix.

        """
        blobs = self.bucket.list_blobs(prefix=self.prefix)

        shards = []
        for blob in blobs:
            name = blob.name
            if name.endswith(".pt"):
                # Return relative path
                relative = name.replace(self.prefix, "")
                shards.append(relative)

        return sorted(shards)

    def load_shard(self, shard_name: str, use_cache: bool = True) -> Any:
        """Load a data shard.

        Args:
            shard_name: Shard filename or relative path.
            use_cache: Use local cache if available.

        Returns:
            Loaded shard data.

        """
        local_path = self.local_cache_dir / shard_name

        if use_cache and local_path.exists():
            return _get_torch().load(local_path, map_location="cpu", weights_only=False)

        # Download from GCS
        gcs_path = f"{self.prefix}{shard_name}"
        blob = self.bucket.blob(gcs_path)

        if not blob.exists():
            raise FileNotFoundError(f"Shard not found: {gcs_path}")

        # Ensure parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        def download() -> None:
            blob.download_to_filename(str(local_path))

        _with_retry(download, max_retries=self.max_retries)

        return _get_torch().load(local_path, map_location="cpu", weights_only=False)

    def upload_shard(
        self,
        data: Any,
        shard_name: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload data as a shard.

        Args:
            data: Data to save.
            shard_name: Shard filename.
            metadata: Optional blob metadata.

        Returns:
            GCS path of uploaded shard.

        """
        local_path = self.local_cache_dir / shard_name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Save locally first
        _get_torch().save(data, local_path)

        # Upload to GCS
        gcs_path = f"{self.prefix}{shard_name}"
        blob = self.bucket.blob(gcs_path)

        if metadata:
            blob.metadata = metadata

        def upload() -> None:
            blob.upload_from_filename(str(local_path))

        _with_retry(upload, max_retries=self.max_retries)

        return f"gs://{self.bucket_name}/{gcs_path}"

    def stream_shard(self, shard_name: str) -> Iterator[Any]:
        """Stream data from a shard without full download.

        Note: This requires the shard to be saved in a streamable format.
        Standard _get_torch().save files cannot be streamed.

        Args:
            shard_name: Shard filename.

        Yields:
            Items from the shard.

        """
        # For now, just load the full shard
        # A true streaming implementation would require a different format
        data = self.load_shard(shard_name)
        if isinstance(data, list | tuple):
            yield from data
        else:
            yield data
