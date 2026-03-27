"""Tests for model zoo checkpoint management and curriculum learning."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
import torch

from src.distributed.model_zoo import (
    ModelMetadata,
    ModelZoo,
    ModelZooConfig,
    create_model_zoo,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42


@pytest.fixture
def seed() -> int:
    """Provide a deterministic seed."""
    return DEFAULT_SEED


@pytest.fixture(autouse=True)
def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture
def zoo_dir(tmp_path: Path) -> Path:
    """Create a temporary zoo directory."""
    return tmp_path / "zoo"


@pytest.fixture
def zoo_config(zoo_dir: Path) -> ModelZooConfig:
    """Create a default model zoo configuration."""
    return ModelZooConfig(
        zoo_dir=str(zoo_dir),
        max_models=5,
        keep_best_n=2,
        curriculum_enabled=True,
        curriculum_strategy="window",
        curriculum_window_size=3,
    )


@pytest.fixture
def zoo(zoo_config: ModelZooConfig) -> ModelZoo:
    """Create an empty model zoo."""
    return ModelZoo(zoo_config)


def _make_state_dict(width: int = 8, seed: int = 0) -> dict[str, Any]:
    """Create a minimal fake state dict."""
    torch.manual_seed(seed)
    return {"weight": torch.randn(width, width), "bias": torch.randn(width)}


def _add_n_models(
    zoo: ModelZoo,
    n: int,
    base_loss: float = 1.0,
    loss_decay: float = 0.1,
) -> list[ModelMetadata]:
    """Add *n* models with decreasing loss to the zoo."""
    results = []
    for i in range(n):
        loss = base_loss - i * loss_decay
        meta = zoo.add_model(
            model=_make_state_dict(seed=i),
            step=(i + 1) * 100,
            metrics={"total_loss": loss},
            config_hash=f"hash_{i}",
        )
        results.append(meta)
    return results


# ---------------------------------------------------------------------------
# ModelZooConfig
# ---------------------------------------------------------------------------


class TestModelZooConfig:
    """Tests for ModelZooConfig validation."""

    def test_default_values(self) -> None:
        """Defaults are sensible."""
        config = ModelZooConfig()
        assert config.max_models == 20
        assert config.keep_best_n == 5
        assert config.curriculum_strategy == "window"

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelZooConfig(unknown_field="bad")  # type: ignore[call-arg]

    @pytest.mark.parametrize("max_models", [1, 10, 100])
    def test_max_models_valid(self, max_models: int) -> None:
        """Various valid max_models values are accepted."""
        config = ModelZooConfig(max_models=max_models)
        assert config.max_models == max_models

    def test_max_models_invalid(self) -> None:
        """max_models < 1 is rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelZooConfig(max_models=0)


# ---------------------------------------------------------------------------
# ModelMetadata serialization
# ---------------------------------------------------------------------------


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """to_dict / from_dict preserves all fields."""
        path = tmp_path / "model.pt"
        path.touch()

        original = ModelMetadata(
            version=3,
            path=path,
            step=1000,
            timestamp="2026-01-01T00:00:00",
            metrics={"total_loss": 0.5, "accuracy": 0.9},
            config_hash="abc123",
            is_best=True,
            win_rate_vs_previous=0.55,
            elo_rating=1500.0,
        )

        restored = ModelMetadata.from_dict(original.to_dict())

        assert restored.version == original.version
        assert restored.step == original.step
        assert restored.metrics == original.metrics
        assert restored.config_hash == original.config_hash
        assert restored.is_best == original.is_best
        assert restored.win_rate_vs_previous == original.win_rate_vs_previous
        assert restored.elo_rating == original.elo_rating

    def test_from_dict_missing_optional_fields(self) -> None:
        """from_dict handles missing optional fields gracefully."""
        data: dict[str, Any] = {
            "version": 0,
            "path": "/tmp/model.pt",
            "step": 0,
            "timestamp": "",
        }
        meta = ModelMetadata.from_dict(data)
        assert meta.metrics == {}
        assert meta.config_hash == ""
        assert meta.is_best is False
        assert meta.win_rate_vs_previous is None
        assert meta.elo_rating is None


# ---------------------------------------------------------------------------
# Add / Get / Versioning
# ---------------------------------------------------------------------------


class TestModelZooAddGet:
    """Tests for adding and retrieving models."""

    def test_add_model_assigns_incrementing_version(self, zoo: ModelZoo) -> None:
        """Each added model gets a unique, incrementing version."""
        metas = _add_n_models(zoo, n=3)
        assert [m.version for m in metas] == [0, 1, 2]

    def test_add_model_saves_file(self, zoo: ModelZoo) -> None:
        """A .pt checkpoint is created on disk."""
        meta = zoo.add_model(
            model=_make_state_dict(),
            step=100,
            metrics={"total_loss": 0.5},
        )
        assert meta.path.exists()

    def test_get_model_by_version(self, zoo: ModelZoo) -> None:
        """get_model with explicit version returns the correct model."""
        zoo.add_model(model=_make_state_dict(seed=0), step=100, metrics={"total_loss": 1.0})
        zoo.add_model(model=_make_state_dict(seed=1), step=200, metrics={"total_loss": 0.5})

        result = zoo.get_model(version=1)
        assert result is not None
        state_dict, meta = result
        assert meta.version == 1
        assert meta.step == 200
        assert "weight" in state_dict

    def test_get_model_nonexistent_returns_none(self, zoo: ModelZoo) -> None:
        """Requesting a non-existent version returns None."""
        assert zoo.get_model(version=999) is None

    def test_get_model_none_version_returns_none(self, zoo: ModelZoo) -> None:
        """get_model(version=None) with load_best=False returns None."""
        assert zoo.get_model(version=None, load_best=False) is None


# ---------------------------------------------------------------------------
# Best version tracking
# ---------------------------------------------------------------------------


class TestGetBestVersion:
    """Tests for best version tracking."""

    def test_first_model_is_best(self, zoo: ModelZoo) -> None:
        """The first added model is always best."""
        zoo.add_model(model=_make_state_dict(), step=100, metrics={"total_loss": 1.0})
        assert zoo.get_best_version() == 0

    def test_lower_loss_becomes_best(self, zoo: ModelZoo) -> None:
        """A model with lower total_loss becomes the new best."""
        zoo.add_model(model=_make_state_dict(seed=0), step=100, metrics={"total_loss": 1.0})
        zoo.add_model(model=_make_state_dict(seed=1), step=200, metrics={"total_loss": 0.2})
        assert zoo.get_best_version() == 1

    def test_higher_loss_does_not_replace_best(self, zoo: ModelZoo) -> None:
        """A model with higher loss does not replace the best."""
        zoo.add_model(model=_make_state_dict(seed=0), step=100, metrics={"total_loss": 0.1})
        zoo.add_model(model=_make_state_dict(seed=1), step=200, metrics={"total_loss": 0.5})
        assert zoo.get_best_version() == 0

    def test_load_best_model(self, zoo: ModelZoo) -> None:
        """get_model(load_best=True) returns the best checkpoint."""
        zoo.add_model(model=_make_state_dict(seed=0), step=100, metrics={"total_loss": 1.0})
        zoo.add_model(model=_make_state_dict(seed=1), step=200, metrics={"total_loss": 0.1})

        result = zoo.get_model(load_best=True)
        assert result is not None
        _, meta = result
        assert meta.version == 1

    @pytest.mark.parametrize("lower_is_better", [True, False])
    def test_lower_is_better_flag(
        self, zoo: ModelZoo, lower_is_better: bool
    ) -> None:
        """add_model respects the lower_is_better flag."""
        zoo.add_model(
            model=_make_state_dict(seed=0),
            step=100,
            metrics={"score": 0.5},
            primary_metric="score",
            lower_is_better=lower_is_better,
        )
        zoo.add_model(
            model=_make_state_dict(seed=1),
            step=200,
            metrics={"score": 0.9},
            primary_metric="score",
            lower_is_better=lower_is_better,
        )
        if lower_is_better:
            assert zoo.get_best_version() == 0  # 0.5 < 0.9
        else:
            assert zoo.get_best_version() == 1  # 0.9 > 0.5

    def test_empty_zoo_returns_none(self, zoo: ModelZoo) -> None:
        """An empty zoo has no best version."""
        assert zoo.get_best_version() is None


# ---------------------------------------------------------------------------
# Curriculum opponent strategies
# ---------------------------------------------------------------------------


class TestGetCurriculumOpponent:
    """Tests for curriculum opponent selection."""

    def test_empty_zoo_returns_none(self, zoo: ModelZoo) -> None:
        """No opponent when zoo is empty."""
        assert zoo.get_curriculum_opponent() is None

    def test_best_strategy(self, zoo_dir: Path) -> None:
        """'best' strategy returns the best model."""
        config = ModelZooConfig(
            zoo_dir=str(zoo_dir),
            curriculum_strategy="best",
            max_models=10,
        )
        zoo = ModelZoo(config)
        _add_n_models(zoo, n=5)

        result = zoo.get_curriculum_opponent()
        assert result is not None
        _, meta = result
        assert meta.version == zoo.get_best_version()

    def test_window_strategy_selects_from_recent(self, zoo_dir: Path, seed: int) -> None:
        """'window' strategy samples from the most recent N models."""
        window_size = 3
        config = ModelZooConfig(
            zoo_dir=str(zoo_dir),
            curriculum_strategy="window",
            curriculum_window_size=window_size,
            max_models=20,
        )
        zoo = ModelZoo(config)
        metas = _add_n_models(zoo, n=10)
        recent_versions = {m.version for m in metas[-window_size:]}

        random.seed(seed)
        result = zoo.get_curriculum_opponent()
        assert result is not None
        _, meta = result
        assert meta.version in recent_versions

    def test_random_strategy(self, zoo_dir: Path, seed: int) -> None:
        """'random' strategy can return any model."""
        config = ModelZooConfig(
            zoo_dir=str(zoo_dir),
            curriculum_strategy="random",
            max_models=20,
        )
        zoo = ModelZoo(config)
        _add_n_models(zoo, n=5)

        random.seed(seed)
        result = zoo.get_curriculum_opponent()
        assert result is not None
        _, meta = result
        assert meta.version in zoo.models

    def test_weighted_strategy(self, zoo_dir: Path, seed: int) -> None:
        """'weighted' strategy returns a valid model."""
        config = ModelZooConfig(
            zoo_dir=str(zoo_dir),
            curriculum_strategy="weighted",
            max_models=20,
        )
        zoo = ModelZoo(config)
        _add_n_models(zoo, n=5)

        random.seed(seed)
        result = zoo.get_curriculum_opponent()
        assert result is not None
        _, meta = result
        assert meta.version in zoo.models


# ---------------------------------------------------------------------------
# Evaluation opponents
# ---------------------------------------------------------------------------


class TestGetEvaluationOpponents:
    """Tests for evaluation opponent selection."""

    def test_empty_zoo(self, zoo: ModelZoo) -> None:
        """Empty zoo returns no opponents."""
        assert zoo.get_evaluation_opponents(n=3) == []

    def test_returns_at_most_n(self, zoo: ModelZoo) -> None:
        """Never returns more than n opponents."""
        _add_n_models(zoo, n=5)
        opponents = zoo.get_evaluation_opponents(n=2)
        assert len(opponents) <= 2

    def test_includes_best_when_configured(self, zoo_dir: Path) -> None:
        """When eval_against_best is True, best model is included."""
        config = ModelZooConfig(
            zoo_dir=str(zoo_dir),
            eval_against_best=True,
            max_models=10,
        )
        zoo = ModelZoo(config)
        _add_n_models(zoo, n=5)

        opponents = zoo.get_evaluation_opponents(n=3)
        versions = {meta.version for _, meta in opponents}
        best = zoo.get_best_version()
        assert best is not None
        assert best in versions

    def test_recent_strategy(self, zoo_dir: Path) -> None:
        """'recent' strategy includes the most recent models."""
        config = ModelZooConfig(
            zoo_dir=str(zoo_dir),
            eval_opponent_strategy="recent",
            eval_against_best=False,
            max_models=10,
        )
        zoo = ModelZoo(config)
        _add_n_models(zoo, n=5)

        opponents = zoo.get_evaluation_opponents(n=3)
        versions = [meta.version for _, meta in opponents]
        # Should include the most recent versions
        all_versions = sorted(zoo.models.keys(), reverse=True)
        for v in versions:
            assert v in all_versions


# ---------------------------------------------------------------------------
# Registry save/load persistence
# ---------------------------------------------------------------------------


class TestRegistryPersistence:
    """Tests for registry save/load round-trip."""

    def test_save_and_reload(self, zoo_config: ModelZooConfig) -> None:
        """Models survive a save-reload cycle."""
        zoo1 = ModelZoo(zoo_config)
        _add_n_models(zoo1, n=3)
        best_v1 = zoo1.get_best_version()

        # Create a new zoo pointing at the same directory
        zoo2 = ModelZoo(zoo_config)

        assert len(zoo2.models) == len(zoo1.models)
        assert zoo2.get_best_version() == best_v1
        for version in zoo1.models:
            assert version in zoo2.models

    def test_registry_json_structure(self, zoo: ModelZoo, zoo_dir: Path) -> None:
        """The registry.json has the expected top-level keys."""
        _add_n_models(zoo, n=2)

        registry_path = zoo_dir / "registry.json"
        assert registry_path.exists()

        data = json.loads(registry_path.read_text())
        assert "models" in data
        assert "next_version" in data
        assert "best_version" in data
        assert "last_updated" in data


# ---------------------------------------------------------------------------
# Cleanup of old versions
# ---------------------------------------------------------------------------


class TestCleanup:
    """Tests for old model cleanup."""

    def test_cleanup_respects_max_models(self, zoo: ModelZoo) -> None:
        """Zoo never exceeds max_models."""
        _add_n_models(zoo, n=10)
        assert len(zoo.models) <= zoo.config.max_models

    def test_cleanup_preserves_best(self, zoo: ModelZoo) -> None:
        """Best models survive cleanup."""
        _add_n_models(zoo, n=10)
        best = zoo.get_best_version()
        assert best is not None
        assert best in zoo.models

    def test_cleanup_removes_files(self, zoo: ModelZoo, zoo_dir: Path) -> None:
        """Removed models have their checkpoint files deleted."""
        _add_n_models(zoo, n=10)
        remaining_paths = {m.path for m in zoo.models.values()}
        all_model_files = set(zoo_dir.glob("model_v*.pt"))
        assert all_model_files == remaining_paths


# ---------------------------------------------------------------------------
# Update metrics / Elo
# ---------------------------------------------------------------------------


class TestUpdateMetrics:
    """Tests for metric and Elo updates."""

    def test_update_metrics(self, zoo: ModelZoo) -> None:
        """update_metrics merges new values into existing metrics."""
        zoo.add_model(model=_make_state_dict(), step=100, metrics={"total_loss": 0.5})
        zoo.update_metrics(version=0, metrics={"accuracy": 0.9})

        assert zoo.models[0].metrics["accuracy"] == 0.9
        assert zoo.models[0].metrics["total_loss"] == 0.5

    def test_update_elo(self, zoo: ModelZoo) -> None:
        """update_elo stores the Elo rating."""
        zoo.add_model(model=_make_state_dict(), step=100, metrics={"total_loss": 0.5})
        zoo.update_elo(version=0, elo=1600.0)
        assert zoo.models[0].elo_rating == 1600.0

    def test_update_nonexistent_version_is_noop(self, zoo: ModelZoo) -> None:
        """Updating a version that does not exist does nothing."""
        zoo.update_metrics(version=999, metrics={"x": 1.0})
        zoo.update_elo(version=999, elo=1000.0)
        # Should not raise


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateModelZoo:
    """Tests for the create_model_zoo factory."""

    def test_creates_zoo(self, tmp_path: Path) -> None:
        """Factory returns a working ModelZoo."""
        zoo = create_model_zoo(zoo_dir=tmp_path / "factory_zoo", max_models=3)
        assert isinstance(zoo, ModelZoo)
        assert zoo.config.max_models == 3

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Factory creates the zoo directory on disk."""
        zoo_dir = tmp_path / "new_zoo"
        create_model_zoo(zoo_dir=zoo_dir)
        assert zoo_dir.exists()
