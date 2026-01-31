"""Tests for GCS storage integration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

# Skip tests if torch is not available
torch = pytest.importorskip("torch")
from torch import nn

from src.vertex.config import VertexStorageConfig
from src.vertex.storage import (
    DEFAULT_MAX_RETRIES,
    GCS_CHECKPOINT_VERSION,
    GCSCheckpointManager,
    GCSCheckpointMetadata,
    GCSDataSource,
    _with_retry,
)

if TYPE_CHECKING:
    pass


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestWithRetry:
    """Tests for retry utility function."""

    def test_success_on_first_try(self) -> None:
        """Test function succeeds on first try."""
        mock_func = MagicMock(return_value="success")
        result = _with_retry(mock_func, max_retries=3)
        assert result == "success"
        assert mock_func.call_count == 1

    def test_success_after_retries(self) -> None:
        """Test function succeeds after retries."""
        mock_func = MagicMock(side_effect=[Exception("fail"), Exception("fail"), "success"])
        result = _with_retry(mock_func, max_retries=3, initial_delay=0.01)
        assert result == "success"
        assert mock_func.call_count == 3

    def test_failure_after_all_retries(self) -> None:
        """Test function fails after exhausting retries."""
        mock_func = MagicMock(side_effect=Exception("always fail"))
        with pytest.raises(Exception, match="always fail"):
            _with_retry(mock_func, max_retries=2, initial_delay=0.01)
        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_specific_exception_types(self) -> None:
        """Test only catching specific exception types."""
        mock_func = MagicMock(side_effect=ValueError("specific error"))
        with pytest.raises(ValueError):
            _with_retry(
                mock_func,
                max_retries=3,
                initial_delay=0.01,
                exceptions=(TypeError,),  # Only catch TypeError
            )
        assert mock_func.call_count == 1  # No retry for ValueError


class TestGCSCheckpointMetadata:
    """Tests for GCSCheckpointMetadata."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metadata = GCSCheckpointMetadata(
            step=1000,
            gcs_path="gs://bucket/checkpoint.pt",
            local_path=Path("/tmp/checkpoint.pt"),
            timestamp="2026-01-01T00:00:00",
            size_bytes=1024,
            md5_hash="abc123",
            metrics={"loss": 0.5},
        )
        d = metadata.to_dict()
        assert d["step"] == 1000
        assert d["gcs_path"] == "gs://bucket/checkpoint.pt"
        assert d["local_path"] == "/tmp/checkpoint.pt"
        assert d["metrics"]["loss"] == 0.5

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        d = {
            "step": 1000,
            "gcs_path": "gs://bucket/checkpoint.pt",
            "local_path": "/tmp/checkpoint.pt",
            "timestamp": "2026-01-01T00:00:00",
            "size_bytes": 1024,
            "md5_hash": "abc123",
            "metrics": {"loss": 0.5},
        }
        metadata = GCSCheckpointMetadata.from_dict(d)
        assert metadata.step == 1000
        assert metadata.gcs_path == "gs://bucket/checkpoint.pt"
        assert metadata.local_path == Path("/tmp/checkpoint.pt")

    def test_from_dict_no_local_path(self) -> None:
        """Test creation from dict without local path."""
        d = {
            "step": 1000,
            "gcs_path": "gs://bucket/checkpoint.pt",
        }
        metadata = GCSCheckpointMetadata.from_dict(d)
        assert metadata.local_path is None


class TestGCSCheckpointManager:
    """Tests for GCSCheckpointManager."""

    @pytest.fixture
    def temp_cache_dir(self) -> Path:
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_storage_module(self) -> MagicMock:
        """Mock google.cloud.storage module."""
        with patch.dict("sys.modules", {"google.cloud.storage": MagicMock()}):
            yield

    @pytest.fixture
    def manager_with_mocked_gcs(
        self,
        temp_cache_dir: Path,
        mock_gcs_client: MagicMock,
    ) -> GCSCheckpointManager:
        """Create manager with mocked GCS client."""
        manager = GCSCheckpointManager(
            bucket_name="test-bucket",
            checkpoint_prefix="checkpoints/",
            local_cache_dir=temp_cache_dir,
            max_checkpoints=3,
        )
        # Inject mocked client
        manager._client = mock_gcs_client.return_value
        manager._bucket = manager._client.bucket("test-bucket")
        return manager

    def test_initialization(self, temp_cache_dir: Path) -> None:
        """Test manager initialization."""
        with patch("src.vertex.storage.GCSCheckpointManager.client", new_callable=lambda: MagicMock()):
            manager = GCSCheckpointManager(
                bucket_name="test-bucket",
                checkpoint_prefix="checkpoints",
                local_cache_dir=temp_cache_dir,
                max_checkpoints=5,
            )
            assert manager.bucket_name == "test-bucket"
            assert manager.checkpoint_prefix == "checkpoints/"
            assert manager.max_checkpoints == 5
            assert manager.local_cache_dir == temp_cache_dir

    def test_prefix_normalization(self, temp_cache_dir: Path) -> None:
        """Test checkpoint prefix is normalized."""
        with patch("src.vertex.storage.GCSCheckpointManager.client", new_callable=lambda: MagicMock()):
            manager = GCSCheckpointManager(
                bucket_name="bucket",
                checkpoint_prefix="path/to/checkpoints",
                local_cache_dir=temp_cache_dir,
            )
            assert manager.checkpoint_prefix == "path/to/checkpoints/"

    def test_from_config(self) -> None:
        """Test creation from VertexStorageConfig."""
        config = VertexStorageConfig(
            bucket_name="config-bucket",
            checkpoint_prefix="config-checkpoints/",
            max_checkpoints=10,
        )
        with patch("src.vertex.storage.GCSCheckpointManager.client", new_callable=lambda: MagicMock()):
            manager = GCSCheckpointManager.from_config(config)
            assert manager.bucket_name == "config-bucket"
            assert manager.checkpoint_prefix == "config-checkpoints/"
            assert manager.max_checkpoints == 10

    def test_save_creates_local_file(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test save creates local checkpoint file."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Mock the upload
        manager_with_mocked_gcs._upload_file = MagicMock()

        gcs_path = manager_with_mocked_gcs.save(
            step=1000,
            model=model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )

        # Check local file was created
        local_path = manager_with_mocked_gcs.local_cache_dir / "checkpoint_00001000.pt"
        assert local_path.exists()

        # Check GCS path
        assert gcs_path == "gs://test-bucket/checkpoints/checkpoint_00001000.pt"

        # Check upload was called
        manager_with_mocked_gcs._upload_file.assert_called_once()

    def test_save_checkpoint_content(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test saved checkpoint contains correct data."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        manager_with_mocked_gcs._upload_file = MagicMock()

        manager_with_mocked_gcs.save(
            step=500,
            model=model,
            optimizer=optimizer,
            config={"learning_rate": 0.001},
            metrics={"loss": 0.3, "accuracy": 0.95},
        )

        # Load local file and check contents
        local_path = manager_with_mocked_gcs.local_cache_dir / "checkpoint_00000500.pt"
        state = torch.load(local_path, weights_only=False)

        assert state["step"] == 500
        assert state["version"] == GCS_CHECKPOINT_VERSION
        assert "model_state_dict" in state
        assert "optimizer_state_dict" in state
        assert state["config"]["learning_rate"] == 0.001
        assert state["metrics"]["loss"] == 0.3
        assert "timestamp" in state

    def test_load_from_cache(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test loading from local cache."""
        # Create a cached checkpoint
        state = {
            "step": 100,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "version": GCS_CHECKPOINT_VERSION,
        }
        local_path = manager_with_mocked_gcs.local_cache_dir / "checkpoint_00000100.pt"
        torch.save(state, local_path)

        # Load should use cache
        loaded = manager_with_mocked_gcs.load(step=100, use_cache=True)
        assert loaded["step"] == 100

    def test_load_requires_path_or_step(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test load raises error without path or step."""
        with pytest.raises(ValueError, match="Either gcs_path or step"):
            manager_with_mocked_gcs.load()

    def test_exists_checks_gcs(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test exists method checks GCS."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        manager_with_mocked_gcs._bucket.blob.return_value = mock_blob

        assert manager_with_mocked_gcs.exists(step=1000) is True
        manager_with_mocked_gcs._bucket.blob.assert_called_with(
            "checkpoints/checkpoint_00001000.pt"
        )

    def test_delete_removes_gcs_and_local(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test delete removes both GCS and local files."""
        # Create local file
        local_path = manager_with_mocked_gcs.local_cache_dir / "checkpoint_00000200.pt"
        local_path.touch()

        # Mock GCS blob
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        manager_with_mocked_gcs._bucket.blob.return_value = mock_blob

        result = manager_with_mocked_gcs.delete(step=200)

        assert result is True
        mock_blob.delete.assert_called_once()
        assert not local_path.exists()

    def test_delete_nonexistent(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test delete returns False for nonexistent checkpoint."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        manager_with_mocked_gcs._bucket.blob.return_value = mock_blob

        result = manager_with_mocked_gcs.delete(step=9999)
        assert result is False

    def test_clear_cache(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test clearing local cache."""
        # Create some cache files
        (manager_with_mocked_gcs.local_cache_dir / "checkpoint_00000001.pt").touch()
        (manager_with_mocked_gcs.local_cache_dir / "checkpoint_00000002.pt").touch()
        (manager_with_mocked_gcs.local_cache_dir / "temp.pt.tmp").touch()

        count = manager_with_mocked_gcs.clear_cache()

        assert count == 3
        assert len(list(manager_with_mocked_gcs.local_cache_dir.glob("*.pt"))) == 0

    def test_set_best_tracking(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test configuring best checkpoint tracking."""
        manager_with_mocked_gcs.set_best_tracking(metric="accuracy", mode="max")

        assert manager_with_mocked_gcs._best_metric == "accuracy"
        assert manager_with_mocked_gcs._best_mode == "max"
        assert manager_with_mocked_gcs._best_value is None

    def test_calculate_md5(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test MD5 calculation."""
        test_file = manager_with_mocked_gcs.local_cache_dir / "test.bin"
        test_file.write_bytes(b"test content")

        md5 = manager_with_mocked_gcs._calculate_md5(test_file)

        assert len(md5) == 32  # MD5 hex string length
        assert all(c in "0123456789abcdef" for c in md5)

    def test_list_checkpoints(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test listing checkpoints."""
        # Mock blobs
        mock_blobs = []
        for step in [100, 200, 300]:
            blob = MagicMock()
            blob.name = f"checkpoints/checkpoint_{step:08d}.pt"
            blob.updated = None
            blob.size = 1024 * step
            blob.md5_hash = f"hash{step}"
            mock_blobs.append(blob)

        manager_with_mocked_gcs._bucket.list_blobs.return_value = mock_blobs

        checkpoints = manager_with_mocked_gcs.list_checkpoints()

        assert len(checkpoints) == 3
        assert checkpoints[0].step == 100
        assert checkpoints[1].step == 200
        assert checkpoints[2].step == 300
        assert checkpoints[0].size_bytes == 102400

    def test_get_latest_step(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test getting latest checkpoint step."""
        mock_blobs = [MagicMock() for _ in range(2)]
        mock_blobs[0].name = "checkpoints/checkpoint_00000100.pt"
        mock_blobs[0].updated = None
        mock_blobs[0].size = 1024
        mock_blobs[0].md5_hash = "hash1"
        mock_blobs[1].name = "checkpoints/checkpoint_00000500.pt"
        mock_blobs[1].updated = None
        mock_blobs[1].size = 1024
        mock_blobs[1].md5_hash = "hash2"

        manager_with_mocked_gcs._bucket.list_blobs.return_value = mock_blobs

        latest = manager_with_mocked_gcs.get_latest_step()
        assert latest == 500

    def test_get_latest_step_empty(
        self,
        manager_with_mocked_gcs: GCSCheckpointManager,
    ) -> None:
        """Test getting latest step with no checkpoints."""
        manager_with_mocked_gcs._bucket.list_blobs.return_value = []
        assert manager_with_mocked_gcs.get_latest_step() is None


class TestGCSDataSource:
    """Tests for GCSDataSource."""

    @pytest.fixture
    def temp_cache_dir(self) -> Path:
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def data_source_with_mocked_gcs(
        self,
        temp_cache_dir: Path,
        mock_gcs_client: MagicMock,
    ) -> GCSDataSource:
        """Create data source with mocked GCS client."""
        source = GCSDataSource(
            bucket_name="data-bucket",
            prefix="training-data/",
            local_cache_dir=temp_cache_dir,
        )
        source._client = mock_gcs_client.return_value
        source._bucket = source._client.bucket("data-bucket")
        return source

    def test_initialization(self, temp_cache_dir: Path) -> None:
        """Test data source initialization."""
        with patch("src.vertex.storage.GCSDataSource.client", new_callable=lambda: MagicMock()):
            source = GCSDataSource(
                bucket_name="bucket",
                prefix="data",
                local_cache_dir=temp_cache_dir,
            )
            assert source.bucket_name == "bucket"
            assert source.prefix == "data/"

    def test_list_shards(
        self,
        data_source_with_mocked_gcs: GCSDataSource,
    ) -> None:
        """Test listing data shards."""
        mock_blobs = [MagicMock() for _ in range(3)]
        mock_blobs[0].name = "training-data/shard_0.pt"
        mock_blobs[1].name = "training-data/shard_1.pt"
        mock_blobs[2].name = "training-data/shard_2.pt"

        data_source_with_mocked_gcs._bucket.list_blobs.return_value = mock_blobs

        shards = data_source_with_mocked_gcs.list_shards()

        assert len(shards) == 3
        assert "shard_0.pt" in shards
        assert "shard_1.pt" in shards
        assert "shard_2.pt" in shards

    def test_load_shard_from_cache(
        self,
        data_source_with_mocked_gcs: GCSDataSource,
    ) -> None:
        """Test loading shard from local cache."""
        # Create cached shard
        shard_data = {"samples": [1, 2, 3]}
        cache_path = data_source_with_mocked_gcs.local_cache_dir / "test_shard.pt"
        torch.save(shard_data, cache_path)

        loaded = data_source_with_mocked_gcs.load_shard("test_shard.pt", use_cache=True)

        assert loaded["samples"] == [1, 2, 3]

    def test_upload_shard(
        self,
        data_source_with_mocked_gcs: GCSDataSource,
    ) -> None:
        """Test uploading a data shard."""
        shard_data = {"samples": [4, 5, 6]}

        mock_blob = MagicMock()
        data_source_with_mocked_gcs._bucket.blob.return_value = mock_blob

        gcs_path = data_source_with_mocked_gcs.upload_shard(
            data=shard_data,
            shard_name="new_shard.pt",
            metadata={"source": "test"},
        )

        assert gcs_path == "gs://data-bucket/training-data/new_shard.pt"
        mock_blob.upload_from_filename.assert_called_once()
        assert mock_blob.metadata == {"source": "test"}

    def test_stream_shard(
        self,
        data_source_with_mocked_gcs: GCSDataSource,
    ) -> None:
        """Test streaming shard data."""
        # Create cached shard with list data
        shard_data = [{"a": 1}, {"a": 2}, {"a": 3}]
        cache_path = data_source_with_mocked_gcs.local_cache_dir / "stream_shard.pt"
        torch.save(shard_data, cache_path)

        items = list(data_source_with_mocked_gcs.stream_shard("stream_shard.pt"))

        assert len(items) == 3
        assert items[0]["a"] == 1
        assert items[2]["a"] == 3
