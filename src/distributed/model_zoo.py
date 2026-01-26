"""Model zoo for checkpoint management and curriculum learning.

This module provides utilities for managing multiple model checkpoints
and implementing curriculum learning strategies.

Features:
    - Model versioning and tracking
    - Checkpoint rotation and cleanup
    - Curriculum learning support
    - Model selection for evaluation opponents
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import torch
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.modeling.model import AlphaGalerkinModel

logger = structlog.get_logger(__name__)


class ModelZooConfig(BaseModel):
    """Configuration for model zoo.

    Attributes:
        zoo_dir: Directory for storing models.
        max_models: Maximum number of models to keep.
        keep_best_n: Keep top N best-performing models.
        curriculum_enabled: Enable curriculum learning.
        curriculum_strategy: Strategy for model selection.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Storage
    zoo_dir: str = Field(
        default="models/zoo",
        description="Directory for model storage",
    )
    max_models: int = Field(
        default=20,
        ge=1,
        description="Maximum models to keep in zoo",
    )
    keep_best_n: int = Field(
        default=5,
        ge=1,
        description="Always keep top N best models",
    )

    # Curriculum learning
    curriculum_enabled: bool = Field(
        default=True,
        description="Enable curriculum learning",
    )
    curriculum_strategy: str = Field(
        default="window",
        description="Strategy: 'window', 'best', 'random', 'weighted'",
    )
    curriculum_window_size: int = Field(
        default=10,
        ge=1,
        description="Window size for 'window' strategy",
    )

    # Evaluation
    eval_opponent_strategy: str = Field(
        default="recent",
        description="Strategy for selecting evaluation opponents",
    )
    eval_against_best: bool = Field(
        default=True,
        description="Always evaluate against best model",
    )


@dataclass
class ModelMetadata:
    """Metadata for a model in the zoo."""

    version: int
    path: Path
    step: int
    timestamp: str
    metrics: dict[str, float]
    config_hash: str
    is_best: bool = False
    win_rate_vs_previous: float | None = None
    elo_rating: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "path": str(self.path),
            "step": self.step,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "config_hash": self.config_hash,
            "is_best": self.is_best,
            "win_rate_vs_previous": self.win_rate_vs_previous,
            "elo_rating": self.elo_rating,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create from dictionary."""
        return cls(
            version=data["version"],
            path=Path(data["path"]),
            step=data["step"],
            timestamp=data["timestamp"],
            metrics=data.get("metrics", {}),
            config_hash=data.get("config_hash", ""),
            is_best=data.get("is_best", False),
            win_rate_vs_previous=data.get("win_rate_vs_previous"),
            elo_rating=data.get("elo_rating"),
        )


class ModelZoo:
    """Manages a collection of model checkpoints.

    Provides model versioning, curriculum learning support,
    and opponent selection for evaluation.

    Attributes:
        config: Model zoo configuration.
        models: Dictionary of model metadata by version.

    """

    def __init__(self, config: ModelZooConfig) -> None:
        """Initialize model zoo.

        Args:
            config: Model zoo configuration.

        """
        self.config = config
        self.zoo_dir = Path(config.zoo_dir)
        self.zoo_dir.mkdir(parents=True, exist_ok=True)

        self.models: dict[int, ModelMetadata] = {}
        self._next_version = 0
        self._best_version: int | None = None
        self._best_metric: float | None = None

        self._logger = structlog.get_logger(__name__).bind(
            zoo_dir=str(self.zoo_dir),
        )

        # Load existing models
        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from disk."""
        registry_path = self.zoo_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path) as f:
                data = json.load(f)

            for model_data in data.get("models", []):
                metadata = ModelMetadata.from_dict(model_data)
                if metadata.path.exists():
                    self.models[metadata.version] = metadata

            self._next_version = data.get("next_version", 0)
            self._best_version = data.get("best_version")
            self._best_metric = data.get("best_metric")

            self._logger.info(
                "registry_loaded",
                n_models=len(self.models),
                best_version=self._best_version,
            )

    def _save_registry(self) -> None:
        """Save model registry to disk."""
        registry_path = self.zoo_dir / "registry.json"

        data = {
            "models": [m.to_dict() for m in self.models.values()],
            "next_version": self._next_version,
            "best_version": self._best_version,
            "best_metric": self._best_metric,
            "last_updated": datetime.now().isoformat(),
        }

        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_model(
        self,
        model: AlphaGalerkinModel | dict[str, Any],
        step: int,
        metrics: dict[str, float],
        config_hash: str = "",
        primary_metric: str = "total_loss",
        lower_is_better: bool = True,
    ) -> ModelMetadata:
        """Add a model to the zoo.

        Args:
            model: Model instance or state dict.
            step: Training step.
            metrics: Model metrics.
            config_hash: Configuration hash for reproducibility.
            primary_metric: Metric for determining best model.
            lower_is_better: Whether lower metric is better.

        Returns:
            Metadata for the added model.

        """
        version = self._next_version
        self._next_version += 1

        # Save model
        model_path = self.zoo_dir / f"model_v{version:06d}.pt"

        if isinstance(model, dict):
            state_dict = model
        else:
            state_dict = model.state_dict()

        torch.save(
            {
                "state_dict": state_dict,
                "step": step,
                "version": version,
                "metrics": metrics,
            },
            model_path,
        )

        # Check if best
        metric_value = metrics.get(primary_metric, float("inf") if lower_is_better else float("-inf"))
        is_best = False

        if self._best_metric is None:
            is_best = True
        elif lower_is_better and metric_value < self._best_metric:
            is_best = True
        elif not lower_is_better and metric_value > self._best_metric:
            is_best = True

        if is_best:
            self._best_version = version
            self._best_metric = metric_value

            # Update previous best
            for m in self.models.values():
                m.is_best = False

        # Create metadata
        metadata = ModelMetadata(
            version=version,
            path=model_path,
            step=step,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            config_hash=config_hash,
            is_best=is_best,
        )

        self.models[version] = metadata

        # Cleanup old models
        self._cleanup()

        # Save registry
        self._save_registry()

        self._logger.info(
            "model_added",
            version=version,
            step=step,
            is_best=is_best,
            metric=metric_value,
        )

        return metadata

    def _cleanup(self) -> None:
        """Remove old models beyond the limit."""
        if len(self.models) <= self.config.max_models:
            return

        # Sort by version (oldest first)
        sorted_versions = sorted(self.models.keys())

        # Keep best models
        best_versions = set()
        if self.config.keep_best_n > 0:
            # Sort by primary metric (assuming loss, lower is better)
            by_metric = sorted(
                self.models.values(),
                key=lambda m: m.metrics.get("total_loss", float("inf")),
            )
            best_versions = {m.version for m in by_metric[: self.config.keep_best_n]}

        # Remove oldest models, keeping best
        to_remove = len(self.models) - self.config.max_models
        removed = 0

        for version in sorted_versions:
            if removed >= to_remove:
                break

            if version in best_versions:
                continue

            metadata = self.models[version]
            if metadata.path.exists():
                metadata.path.unlink()

            del self.models[version]
            removed += 1

            self._logger.debug("model_removed", version=version)

    def get_model(
        self,
        version: int | None = None,
        load_best: bool = False,
    ) -> tuple[dict[str, Any], ModelMetadata] | None:
        """Get a model from the zoo.

        Args:
            version: Specific version to load.
            load_best: Load the best model.

        Returns:
            Tuple of (state_dict, metadata) or None if not found.

        """
        if load_best:
            version = self._best_version

        if version is None:
            return None

        metadata = self.models.get(version)
        if metadata is None or not metadata.path.exists():
            return None

        checkpoint = torch.load(metadata.path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        return state_dict, metadata

    def get_curriculum_opponent(self) -> tuple[dict[str, Any], ModelMetadata] | None:
        """Get an opponent model for curriculum learning.

        Returns:
            Tuple of (state_dict, metadata) or None if no models available.

        """
        if not self.models:
            return None

        strategy = self.config.curriculum_strategy
        versions = sorted(self.models.keys())

        if strategy == "best":
            return self.get_model(load_best=True)

        elif strategy == "window":
            # Select from recent window
            import random

            window = versions[-self.config.curriculum_window_size :]
            version = random.choice(window)
            return self.get_model(version=version)

        elif strategy == "random":
            import random

            version = random.choice(versions)
            return self.get_model(version=version)

        elif strategy == "weighted":
            # Weight by recency (more recent = higher weight)
            import random

            weights = [(i + 1) ** 2 for i in range(len(versions))]
            total = sum(weights)
            weights = [w / total for w in weights]
            version = random.choices(versions, weights=weights, k=1)[0]
            return self.get_model(version=version)

        else:
            # Default to most recent
            return self.get_model(version=versions[-1])

    def get_evaluation_opponents(self, n: int = 3) -> list[tuple[dict[str, Any], ModelMetadata]]:
        """Get opponents for model evaluation.

        Args:
            n: Number of opponents to return.

        Returns:
            List of (state_dict, metadata) tuples.

        """
        opponents: list[tuple[dict[str, Any], ModelMetadata]] = []

        if not self.models:
            return opponents

        versions = sorted(self.models.keys())

        # Always include best if configured
        if self.config.eval_against_best and self._best_version is not None:
            result = self.get_model(load_best=True)
            if result:
                opponents.append(result)

        # Add recent models
        strategy = self.config.eval_opponent_strategy

        if strategy == "recent":
            for version in reversed(versions):
                if len(opponents) >= n:
                    break
                if version == self._best_version:
                    continue
                result = self.get_model(version=version)
                if result:
                    opponents.append(result)

        elif strategy == "spread":
            # Spread across training history
            import numpy as np

            indices = np.linspace(0, len(versions) - 1, n, dtype=int)
            for idx in indices:
                version = versions[idx]
                if version == self._best_version:
                    continue
                result = self.get_model(version=version)
                if result:
                    opponents.append(result)

        return opponents[:n]

    def update_metrics(
        self,
        version: int,
        metrics: dict[str, float],
    ) -> None:
        """Update metrics for a model.

        Args:
            version: Model version.
            metrics: New metrics to add.

        """
        if version not in self.models:
            return

        self.models[version].metrics.update(metrics)
        self._save_registry()

    def update_elo(self, version: int, elo: float) -> None:
        """Update Elo rating for a model.

        Args:
            version: Model version.
            elo: New Elo rating.

        """
        if version not in self.models:
            return

        self.models[version].elo_rating = elo
        self._save_registry()

    def get_latest_version(self) -> int | None:
        """Get the latest model version.

        Returns:
            Latest version number or None if empty.

        """
        if not self.models:
            return None
        return max(self.models.keys())

    def get_best_version(self) -> int | None:
        """Get the best model version.

        Returns:
            Best version number or None if not set.

        """
        return self._best_version

    def list_models(self) -> list[ModelMetadata]:
        """List all models in the zoo.

        Returns:
            List of model metadata, sorted by version.

        """
        return sorted(self.models.values(), key=lambda m: m.version)

    def export_model(
        self,
        version: int | None = None,
        export_dir: Path | str | None = None,
    ) -> Path | None:
        """Export a model for deployment.

        Args:
            version: Version to export (None for best).
            export_dir: Export directory.

        Returns:
            Path to exported model or None if not found.

        """
        result = self.get_model(version=version, load_best=version is None)
        if result is None:
            return None

        state_dict, metadata = result

        if export_dir is None:
            export_dir = self.zoo_dir / "exports"

        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        export_path = export_dir / f"model_v{metadata.version}_export.pt"

        torch.save(
            {
                "state_dict": state_dict,
                "metadata": metadata.to_dict(),
            },
            export_path,
        )

        self._logger.info("model_exported", path=str(export_path))

        return export_path


def create_model_zoo(
    zoo_dir: str | Path = "models/zoo",
    **kwargs: Any,
) -> ModelZoo:
    """Factory function to create model zoo.

    Args:
        zoo_dir: Directory for model storage.
        **kwargs: Additional configuration options.

    Returns:
        Configured ModelZoo instance.

    """
    config = ModelZooConfig(zoo_dir=str(zoo_dir), **kwargs)
    return ModelZoo(config)
