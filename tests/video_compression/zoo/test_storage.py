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


class TestUnsupportedBackend:
    def test_gcs_save_not_implemented(self, tmp_path: Path) -> None:
        # GCS backend instantiation requires the [vertex] extra; if
        # available, save_entry is still gated behind NotImplementedError.
        try:
            zoo = VideoCodecZoo("gs://bucket/prefix", backend=StorageBackend.GCS)
        except ImportError:
            pytest.skip("[vertex] extra not installed")
        with pytest.raises(NotImplementedError):
            zoo.save_entry(_entry(), _state_dict(), metrics={"a": 0.0})

    def test_gcs_load_state_dict_not_implemented(self) -> None:
        try:
            zoo = VideoCodecZoo("gs://bucket/prefix", backend=StorageBackend.GCS)
        except ImportError:
            pytest.skip("[vertex] extra not installed")
        with pytest.raises(NotImplementedError, match="load_state_dict"):
            zoo.load_state_dict("any_id")

    def test_gcs_load_metrics_not_implemented(self) -> None:
        try:
            zoo = VideoCodecZoo("gs://bucket/prefix", backend=StorageBackend.GCS)
        except ImportError:
            pytest.skip("[vertex] extra not installed")
        with pytest.raises(NotImplementedError, match="load_metrics"):
            zoo.load_metrics("any_id")
