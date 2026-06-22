"""Filesystem storage tests for VideoCodecZoo."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.video_compression.zoo.config import (
    ModelZooEntryConfig,
    StorageBackend,
)
from src.video_compression.zoo.storage import VideoCodecZoo


class TestParseGcsUri:
    """The shared gs:// URI parser used by the GCS zoo backend."""

    def test_bucket_and_prefix(self) -> None:
        from src.vertex.storage import parse_gcs_uri

        assert parse_gcs_uri("gs://bucket/a/b") == ("bucket", "a/b")
        assert parse_gcs_uri("gs://bucket") == ("bucket", "")
        assert parse_gcs_uri("gs://bucket/a/b/") == ("bucket", "a/b")

    def test_rejects_non_gcs_uri(self) -> None:
        from src.vertex.storage import parse_gcs_uri

        with pytest.raises(ValueError, match="not a gs://"):
            parse_gcs_uri("/local/path")

    def test_rejects_missing_bucket(self) -> None:
        from src.vertex.storage import parse_gcs_uri

        with pytest.raises(ValueError, match="missing bucket"):
            parse_gcs_uri("gs://")


def _entry(entry_id: str = "e1") -> ModelZooEntryConfig:
    return ModelZooEntryConfig(
        entry_id=entry_id,
        lambda_rd=0.01,
        target_bpp=0.5,
        target_psnr_db=33.0,
        train_steps=1000,
    )


def _state_dict() -> dict[str, object]:
    return {
        "model": {"w": torch.tensor([1.0, 2.0, 3.0])},
        "step": 100,
        "lambda_rd": 0.01,
    }


class TestVideoCodecZooFilesystem:
    def test_save_and_round_trip(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        entry = _entry()
        artifacts = zoo.save_entry(
            entry,
            _state_dict(),
            metrics={"psnr_db": 33.5, "bpp": 0.51},
        )
        assert artifacts.entry_id == "e1"
        assert artifacts.checkpoint_path.exists()
        assert (artifacts.entry_dir / "entry.json").exists()
        assert (artifacts.entry_dir / "metrics.json").exists()
        assert artifacts.metrics["psnr_db"] == pytest.approx(33.5)
        assert zoo.has_entry("e1")
        assert zoo.list_entries() == ["e1"]

    def test_load_state_dict(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        zoo.save_entry(_entry(), _state_dict(), metrics={"psnr_db": 33.0})
        bundle = zoo.load_state_dict("e1", map_location="cpu")
        assert "model" in bundle
        assert torch.allclose(
            bundle["model"]["w"],
            torch.tensor([1.0, 2.0, 3.0]),
        )
        assert bundle["step"] == 100

    def test_load_metrics(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        zoo.save_entry(_entry(), _state_dict(), metrics={"a": 1.0, "b": 2.5})
        m = zoo.load_metrics("e1")
        assert m == {"a": 1.0, "b": 2.5}

    def test_missing_entry_raises(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        assert not zoo.has_entry("missing")
        with pytest.raises(FileNotFoundError):
            zoo.load_state_dict("missing")
        with pytest.raises(FileNotFoundError):
            zoo.load_metrics("missing")

    def test_list_entries_multiple(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        for i in range(3):
            zoo.save_entry(
                _entry(f"e{i}"),
                _state_dict(),
                metrics={"psnr_db": 30.0 + i},
            )
        assert zoo.list_entries() == ["e0", "e1", "e2"]

    def test_metric_keys_must_be_str(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        with pytest.raises(TypeError, match="metric key"):
            zoo.save_entry(_entry(), _state_dict(), metrics={1: 0.0})  # type: ignore[dict-item]

    def test_metric_values_coerced_to_float(self, tmp_path: Path) -> None:
        zoo = VideoCodecZoo(tmp_path / "zoo")
        artifacts = zoo.save_entry(
            _entry(),
            _state_dict(),
            metrics={"a": 1, "b": 2.5},  # int -> float
        )
        assert isinstance(artifacts.metrics["a"], float)
        assert artifacts.metrics["a"] == 1.0


class TestVideoCodecZooGCS:
    """Direct-upload GCS backend, exercised against an in-memory fake client."""

    def _zoo(self, uri: str = "gs://bucket/runs/zoo") -> VideoCodecZoo:
        return VideoCodecZoo(uri, backend=StorageBackend.GCS)

    def test_parses_uri_on_construction(self, fake_gcs: dict[str, bytes]) -> None:
        zoo = self._zoo("gs://my-bucket/some/prefix")
        assert zoo._gcs_bucket_name == "my-bucket"
        assert zoo._gcs_prefix == "some/prefix"

    def test_save_and_round_trip(self, fake_gcs: dict[str, bytes]) -> None:
        zoo = self._zoo()
        artifacts = zoo.save_entry(
            _entry(),
            _state_dict(),
            metrics={"psnr_db": 33.5, "bpp": 0.51},
        )
        assert artifacts.entry_id == "e1"
        assert artifacts.checkpoint_path == "gs://bucket/runs/zoo/e1/checkpoint.pt"
        assert artifacts.entry_dir == "gs://bucket/runs/zoo/e1"
        assert artifacts.metrics["psnr_db"] == pytest.approx(33.5)
        # Objects are written under the full prefix/entry/file blob name.
        assert "runs/zoo/e1/checkpoint.pt" in fake_gcs
        assert "runs/zoo/e1/entry.json" in fake_gcs
        assert "runs/zoo/e1/metrics.json" in fake_gcs

        assert zoo.has_entry("e1")
        assert not zoo.has_entry("missing")
        assert zoo.list_entries() == ["e1"]

    def test_load_state_dict(self, fake_gcs: dict[str, bytes]) -> None:
        zoo = self._zoo()
        zoo.save_entry(_entry(), _state_dict(), metrics={"psnr_db": 33.0})
        bundle = zoo.load_state_dict("e1", map_location="cpu", weights_only=False)
        assert torch.allclose(bundle["model"]["w"], torch.tensor([1.0, 2.0, 3.0]))
        assert bundle["step"] == 100

    def test_load_metrics(self, fake_gcs: dict[str, bytes]) -> None:
        zoo = self._zoo()
        zoo.save_entry(_entry(), _state_dict(), metrics={"a": 1.0, "b": 2.5})
        assert zoo.load_metrics("e1") == {"a": 1.0, "b": 2.5}

    def test_missing_entry_raises(self, fake_gcs: dict[str, bytes]) -> None:
        zoo = self._zoo()
        with pytest.raises(FileNotFoundError):
            zoo.load_state_dict("missing")
        with pytest.raises(FileNotFoundError):
            zoo.load_metrics("missing")

    def test_list_entries_multiple(self, fake_gcs: dict[str, bytes]) -> None:
        zoo = self._zoo()
        for i in range(3):
            zoo.save_entry(_entry(f"e{i}"), _state_dict(), metrics={"psnr_db": 30.0 + i})
        assert zoo.list_entries() == ["e0", "e1", "e2"]

    def test_empty_prefix_uri(self, fake_gcs: dict[str, bytes]) -> None:
        zoo = self._zoo("gs://bucket")
        assert zoo._gcs_prefix == ""
        artifacts = zoo.save_entry(_entry(), _state_dict(), metrics={"a": 1.0})
        assert artifacts.checkpoint_path == "gs://bucket/e1/checkpoint.pt"
        assert "e1/checkpoint.pt" in fake_gcs
        assert zoo.list_entries() == ["e1"]

    def test_metric_keys_must_be_str(self, fake_gcs: dict[str, bytes]) -> None:
        zoo = self._zoo()
        with pytest.raises(TypeError, match="metric key"):
            zoo.save_entry(_entry(), _state_dict(), metrics={1: 0.0})  # type: ignore[dict-item]
