"""Tests for model zoo checkpoint management.

Tests cover:
- ModelZooConfig: Configuration validation
- ModelMetadata: Metadata serialization
- ModelZoo: Initialization, model registration, best tracking, cleanup
- create_model_zoo: Factory function
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from src.distributed.model_zoo import (
    ModelMetadata,
    ModelZoo,
    ModelZooConfig,
    create_model_zoo,
)


# --- Config Tests ---


class TestModelZooConfig:
    """Tests for ModelZooConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ModelZooConfig()

        assert config.zoo_dir == "models/zoo"
        assert config.max_models == 20
        assert config.keep_best_n == 5
        assert config.curriculum_enabled is True
        assert config.curriculum_strategy == "window"
        assert config.curriculum_window_size == 10
        assert config.eval_opponent_strategy == "recent"
        assert config.eval_against_best is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ModelZooConfig(
            zoo_dir="/tmp/zoo",
            max_models=10,
            keep_best_n=3,
            curriculum_strategy="best",
        )

        assert config.zoo_dir == "/tmp/zoo"
        assert config.max_models == 10
        assert config.keep_best_n == 3
        assert config.curriculum_strategy == "best"

    def test_max_models_validation(self) -> None:
        """Test that max_models must be >= 1."""
        with pytest.raises(ValidationError):
            ModelZooConfig(max_models=0)

    def test_keep_best_n_validation(self) -> None:
        """Test that keep_best_n must be >= 1."""
        with pytest.raises(ValidationError):
            ModelZooConfig(keep_best_n=0)

    def test_curriculum_window_size_validation(self) -> None:
        """Test that curriculum_window_size must be >= 1."""
        with pytest.raises(ValidationError):
            ModelZooConfig(curriculum_window_size=0)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ModelZooConfig(unknown_field="value")


# --- Metadata Tests ---


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_to_dict(self) -> None:
        """Test metadata serialization to dict."""
        metadata = ModelMetadata(
            version=1,
            path=Path("/tmp/model_v1.pt"),
            step=1000,
            timestamp="2026-01-01T00:00:00",
            metrics={"total_loss": 0.5, "policy_loss": 0.3},
            config_hash="abc123",
            is_best=True,
            win_rate_vs_previous=0.55,
            elo_rating=1500.0,
        )

        d = metadata.to_dict()

        assert d["version"] == 1
        assert d["path"] == "/tmp/model_v1.pt"
        assert d["step"] == 1000
        assert d["metrics"]["total_loss"] == 0.5
        assert d["is_best"] is True
        assert d["win_rate_vs_previous"] == 0.55
        assert d["elo_rating"] == 1500.0

    def test_from_dict(self) -> None:
        """Test metadata deserialization from dict."""
        data = {
            "version": 2,
            "path": "/tmp/model_v2.pt",
            "step": 2000,
            "timestamp": "2026-01-02T00:00:00",
            "metrics": {"total_loss": 0.4},
            "config_hash": "def456",
            "is_best": False,
            "win_rate_vs_previous": None,
            "elo_rating": 1600.0,
        }

        metadata = ModelMetadata.from_dict(data)

        assert metadata.version == 2
        assert metadata.path == Path("/tmp/model_v2.pt")
        assert metadata.step == 2000
        assert metadata.metrics == {"total_loss": 0.4}
        assert metadata.is_best is False
        assert metadata.elo_rating == 1600.0

    def test_from_dict_missing_optional_fields(self) -> None:
        """Test deserialization with missing optional fields."""
        data = {
            "version": 0,
            "path": "/tmp/model_v0.pt",
            "step": 0,
            "timestamp": "2026-01-01T00:00:00",
        }

        metadata = ModelMetadata.from_dict(data)

        assert metadata.metrics == {}
        assert metadata.config_hash == ""
        assert metadata.is_best is False
        assert metadata.win_rate_vs_previous is None
        assert metadata.elo_rating is None

    def test_roundtrip(self) -> None:
        """Test to_dict -> from_dict roundtrip."""
        original = ModelMetadata(
            version=5,
            path=Path("/tmp/model_v5.pt"),
            step=5000,
            timestamp="2026-01-05T00:00:00",
            metrics={"total_loss": 0.1},
            config_hash="xyz",
            is_best=True,
        )

        reconstructed = ModelMetadata.from_dict(original.to_dict())

        assert reconstructed.version == original.version
        assert reconstructed.step == original.step
        assert reconstructed.metrics == original.metrics
        assert reconstructed.is_best == original.is_best


# --- ModelZoo Tests ---


class TestModelZoo:
    """Tests for ModelZoo."""

    @pytest.fixture
    def zoo(self, tmp_path: Path) -> ModelZoo:
        """Create a ModelZoo with tmp directory."""
        config = ModelZooConfig(zoo_dir=str(tmp_path / "zoo"), max_models=5, keep_best_n=2)
        return ModelZoo(config)

    @pytest.fixture
    def simple_state_dict(self) -> dict:
        """Create a simple state dict for testing."""
        return {"weight": torch.randn(4, 4), "bias": torch.randn(4)}

    def test_initialization_creates_directory(self, tmp_path: Path) -> None:
        """Test that zoo directory is created on init."""
        zoo_dir = tmp_path / "new_zoo"
        config = ModelZooConfig(zoo_dir=str(zoo_dir))
        ModelZoo(config)

        assert zoo_dir.exists()

    def test_empty_zoo(self, zoo: ModelZoo) -> None:
        """Test empty zoo state."""
        assert len(zoo.models) == 0
        assert zoo.get_latest_version() is None
        assert zoo.get_best_version() is None
        assert zoo.list_models() == []

    def test_add_model(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test adding a model to the zoo."""
        metadata = zoo.add_model(
            model=simple_state_dict,
            step=100,
            metrics={"total_loss": 1.0},
        )

        assert metadata.version == 0
        assert metadata.step == 100
        assert metadata.is_best is True
        assert metadata.path.exists()
        assert len(zoo.models) == 1

    def test_add_multiple_models(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test adding multiple models."""
        for i in range(3):
            zoo.add_model(
                model=simple_state_dict,
                step=i * 100,
                metrics={"total_loss": 1.0 - i * 0.1},
            )

        assert len(zoo.models) == 3
        assert zoo.get_latest_version() == 2

    def test_best_model_tracking_lower_is_better(
        self, zoo: ModelZoo, simple_state_dict: dict
    ) -> None:
        """Test best model tracking when lower metric is better."""
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})
        zoo.add_model(model=simple_state_dict, step=200, metrics={"total_loss": 0.5})
        zoo.add_model(model=simple_state_dict, step=300, metrics={"total_loss": 0.8})

        assert zoo.get_best_version() == 1  # Version 1 has loss 0.5

    def test_best_model_tracking_higher_is_better(
        self, zoo: ModelZoo, simple_state_dict: dict
    ) -> None:
        """Test best model tracking when higher metric is better."""
        zoo.add_model(
            model=simple_state_dict,
            step=100,
            metrics={"accuracy": 0.7},
            primary_metric="accuracy",
            lower_is_better=False,
        )
        zoo.add_model(
            model=simple_state_dict,
            step=200,
            metrics={"accuracy": 0.9},
            primary_metric="accuracy",
            lower_is_better=False,
        )
        zoo.add_model(
            model=simple_state_dict,
            step=300,
            metrics={"accuracy": 0.8},
            primary_metric="accuracy",
            lower_is_better=False,
        )

        assert zoo.get_best_version() == 1  # Version 1 has accuracy 0.9

    def test_only_latest_is_best(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test that only the current best model has is_best=True."""
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})
        zoo.add_model(model=simple_state_dict, step=200, metrics={"total_loss": 0.5})

        # Version 0 should no longer be best
        assert zoo.models[0].is_best is False
        assert zoo.models[1].is_best is True

    def test_get_model_by_version(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test retrieving a model by version number."""
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})

        result = zoo.get_model(version=0)

        assert result is not None
        state_dict, metadata = result
        assert metadata.version == 0
        assert "weight" in state_dict

    def test_get_best_model(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test retrieving the best model."""
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})
        zoo.add_model(model=simple_state_dict, step=200, metrics={"total_loss": 0.3})

        result = zoo.get_model(load_best=True)

        assert result is not None
        _, metadata = result
        assert metadata.version == 1

    def test_get_model_nonexistent_version(self, zoo: ModelZoo) -> None:
        """Test retrieving a nonexistent version returns None."""
        result = zoo.get_model(version=999)
        assert result is None

    def test_get_model_none_version(self, zoo: ModelZoo) -> None:
        """Test get_model with version=None returns None."""
        result = zoo.get_model(version=None)
        assert result is None

    def test_max_models_cleanup(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test that old models are cleaned up when exceeding max_models."""
        # zoo.config.max_models = 5
        for i in range(7):
            zoo.add_model(
                model=simple_state_dict,
                step=i * 100,
                metrics={"total_loss": 1.0 - i * 0.1},
            )

        assert len(zoo.models) <= 5

    def test_cleanup_preserves_best(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test that cleanup preserves the best models."""
        # Add models with decreasing loss, then increasing
        losses = [1.0, 0.5, 0.3, 0.8, 0.9, 1.2, 1.5]
        for i, loss in enumerate(losses):
            zoo.add_model(
                model=simple_state_dict,
                step=i * 100,
                metrics={"total_loss": loss},
            )

        # Best model (version 2, loss=0.3) should be preserved
        assert 2 in zoo.models

    def test_list_models_sorted(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test that list_models returns sorted by version."""
        for i in range(3):
            zoo.add_model(
                model=simple_state_dict,
                step=i * 100,
                metrics={"total_loss": float(i)},
            )

        models = zoo.list_models()

        assert len(models) == 3
        assert models[0].version < models[1].version < models[2].version

    def test_update_metrics(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test updating metrics for a model."""
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})

        zoo.update_metrics(0, {"win_rate": 0.6, "elo": 1500})

        assert zoo.models[0].metrics["win_rate"] == 0.6
        assert zoo.models[0].metrics["elo"] == 1500
        # Original metric preserved
        assert zoo.models[0].metrics["total_loss"] == 1.0

    def test_update_metrics_nonexistent(self, zoo: ModelZoo) -> None:
        """Test updating metrics for nonexistent version does nothing."""
        zoo.update_metrics(999, {"win_rate": 0.6})
        # Should not raise

    def test_update_elo(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test updating Elo rating."""
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})

        zoo.update_elo(0, 1500.0)

        assert zoo.models[0].elo_rating == 1500.0

    def test_update_elo_nonexistent(self, zoo: ModelZoo) -> None:
        """Test updating Elo for nonexistent version does nothing."""
        zoo.update_elo(999, 1500.0)
        # Should not raise

    def test_registry_persistence(self, tmp_path: Path, simple_state_dict: dict) -> None:
        """Test that registry survives zoo recreation."""
        zoo_dir = str(tmp_path / "persist_zoo")

        # Create and populate
        config = ModelZooConfig(zoo_dir=zoo_dir, max_models=10)
        zoo1 = ModelZoo(config)
        zoo1.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 0.5})
        zoo1.add_model(model=simple_state_dict, step=200, metrics={"total_loss": 0.3})

        # Recreate from same directory
        zoo2 = ModelZoo(ModelZooConfig(zoo_dir=zoo_dir, max_models=10))

        assert len(zoo2.models) == 2
        assert zoo2.get_best_version() == 1
        assert zoo2.get_latest_version() == 1

    def test_registry_json_exists(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test that registry.json is written."""
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})

        registry_path = zoo.zoo_dir / "registry.json"
        assert registry_path.exists()

        with open(registry_path) as f:
            data = json.load(f)

        assert "models" in data
        assert len(data["models"]) == 1
        assert data["best_version"] == 0

    def test_export_model(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test exporting a model."""
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})

        export_path = zoo.export_model(version=0)

        assert export_path is not None
        assert export_path.exists()
        checkpoint = torch.load(export_path, map_location="cpu", weights_only=True)
        assert "state_dict" in checkpoint
        assert "metadata" in checkpoint

    def test_export_best_model(self, zoo: ModelZoo, simple_state_dict: dict) -> None:
        """Test exporting the best model (version=None)."""
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})
        zoo.add_model(model=simple_state_dict, step=200, metrics={"total_loss": 0.5})

        export_path = zoo.export_model()  # version=None => exports best

        assert export_path is not None
        checkpoint = torch.load(export_path, map_location="cpu", weights_only=True)
        assert checkpoint["metadata"]["version"] == 1

    def test_export_to_custom_dir(
        self, zoo: ModelZoo, simple_state_dict: dict, tmp_path: Path
    ) -> None:
        """Test exporting to a custom directory."""
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})
        export_dir = tmp_path / "custom_exports"

        export_path = zoo.export_model(version=0, export_dir=export_dir)

        assert export_path is not None
        assert export_path.parent == export_dir

    def test_export_nonexistent_returns_none(self, zoo: ModelZoo) -> None:
        """Test exporting a nonexistent model returns None."""
        result = zoo.export_model(version=999)
        assert result is None

    def test_curriculum_opponent_empty_zoo(self, zoo: ModelZoo) -> None:
        """Test curriculum opponent from empty zoo returns None."""
        result = zoo.get_curriculum_opponent()
        assert result is None

    def test_curriculum_opponent_best_strategy(
        self, tmp_path: Path, simple_state_dict: dict
    ) -> None:
        """Test curriculum opponent with 'best' strategy."""
        config = ModelZooConfig(
            zoo_dir=str(tmp_path / "zoo"),
            curriculum_strategy="best",
        )
        zoo = ModelZoo(config)
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})
        zoo.add_model(model=simple_state_dict, step=200, metrics={"total_loss": 0.5})

        result = zoo.get_curriculum_opponent()

        assert result is not None
        _, metadata = result
        assert metadata.version == 1  # Best model

    def test_curriculum_opponent_window_strategy(
        self, tmp_path: Path, simple_state_dict: dict
    ) -> None:
        """Test curriculum opponent with 'window' strategy returns a valid model."""
        config = ModelZooConfig(
            zoo_dir=str(tmp_path / "zoo"),
            curriculum_strategy="window",
            curriculum_window_size=2,
        )
        zoo = ModelZoo(config)
        for i in range(5):
            zoo.add_model(
                model=simple_state_dict,
                step=i * 100,
                metrics={"total_loss": float(i)},
            )

        result = zoo.get_curriculum_opponent()

        assert result is not None
        _, metadata = result
        # Should be from last 2 versions
        assert metadata.version in [3, 4]

    def test_curriculum_opponent_random_strategy(
        self, tmp_path: Path, simple_state_dict: dict
    ) -> None:
        """Test curriculum opponent with 'random' strategy returns a valid model."""
        config = ModelZooConfig(
            zoo_dir=str(tmp_path / "zoo"),
            curriculum_strategy="random",
        )
        zoo = ModelZoo(config)
        for i in range(3):
            zoo.add_model(
                model=simple_state_dict,
                step=i * 100,
                metrics={"total_loss": float(i)},
            )

        result = zoo.get_curriculum_opponent()

        assert result is not None
        _, metadata = result
        assert metadata.version in [0, 1, 2]

    def test_curriculum_opponent_weighted_strategy(
        self, tmp_path: Path, simple_state_dict: dict
    ) -> None:
        """Test curriculum opponent with 'weighted' strategy returns a valid model."""
        config = ModelZooConfig(
            zoo_dir=str(tmp_path / "zoo"),
            curriculum_strategy="weighted",
        )
        zoo = ModelZoo(config)
        for i in range(3):
            zoo.add_model(
                model=simple_state_dict,
                step=i * 100,
                metrics={"total_loss": float(i)},
            )

        result = zoo.get_curriculum_opponent()

        assert result is not None

    def test_curriculum_opponent_unknown_strategy(
        self, tmp_path: Path, simple_state_dict: dict
    ) -> None:
        """Test curriculum opponent with unknown strategy falls back to most recent."""
        config = ModelZooConfig(
            zoo_dir=str(tmp_path / "zoo"),
            curriculum_strategy="nonexistent",
        )
        zoo = ModelZoo(config)
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})
        zoo.add_model(model=simple_state_dict, step=200, metrics={"total_loss": 0.5})

        result = zoo.get_curriculum_opponent()

        assert result is not None
        _, metadata = result
        assert metadata.version == 1  # Most recent

    def test_evaluation_opponents_empty(self, zoo: ModelZoo) -> None:
        """Test evaluation opponents from empty zoo."""
        opponents = zoo.get_evaluation_opponents()
        assert opponents == []

    def test_evaluation_opponents_includes_best(
        self, zoo: ModelZoo, simple_state_dict: dict
    ) -> None:
        """Test evaluation opponents includes the best model."""
        zoo.add_model(model=simple_state_dict, step=100, metrics={"total_loss": 1.0})
        zoo.add_model(model=simple_state_dict, step=200, metrics={"total_loss": 0.3})
        zoo.add_model(model=simple_state_dict, step=300, metrics={"total_loss": 0.8})

        opponents = zoo.get_evaluation_opponents(n=3)

        versions = [m.version for _, m in opponents]
        assert 1 in versions  # Best model (loss=0.3)

    def test_evaluation_opponents_limited_by_n(
        self, zoo: ModelZoo, simple_state_dict: dict
    ) -> None:
        """Test that evaluation opponents are limited by n."""
        for i in range(5):
            zoo.add_model(
                model=simple_state_dict,
                step=i * 100,
                metrics={"total_loss": float(i)},
            )

        opponents = zoo.get_evaluation_opponents(n=2)

        assert len(opponents) <= 2


# --- Factory Tests ---


class TestCreateModelZoo:
    """Tests for create_model_zoo factory function."""

    def test_factory_creates_zoo(self, tmp_path: Path) -> None:
        """Test factory function creates ModelZoo."""
        zoo = create_model_zoo(zoo_dir=tmp_path / "factory_zoo")

        assert isinstance(zoo, ModelZoo)
        assert (tmp_path / "factory_zoo").exists()

    def test_factory_with_kwargs(self, tmp_path: Path) -> None:
        """Test factory function with additional kwargs."""
        zoo = create_model_zoo(
            zoo_dir=tmp_path / "factory_zoo",
            max_models=10,
            keep_best_n=3,
        )

        assert zoo.config.max_models == 10
        assert zoo.config.keep_best_n == 3
