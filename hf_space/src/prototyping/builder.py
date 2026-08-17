"""Model builder for rapid prototyping.

Provides quick model creation with sensible defaults
and flexible customization.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import structlog

from src.prototyping.config import PresetType, PrototypeConfig, create_prototype_config

logger = structlog.get_logger(__name__)


class ModelProtocol(Protocol):
    """Protocol for model interfaces."""

    def forward(self, x: Any) -> Any:
        """Forward pass."""
        ...

    def parameters(self) -> Any:
        """Get model parameters."""
        ...


@dataclass
class PrototypeModel:
    """Wrapper for prototype models.

    Attributes:
        model_id: Unique identifier.
        config: Prototype configuration.
        model: Underlying model instance.
        created_at: Creation timestamp.
        metadata: Additional metadata.

    """

    model_id: str
    config: PrototypeConfig
    model: Any
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Get model name."""
        return self.config.name

    @property
    def n_parameters(self) -> int:
        """Count trainable parameters."""
        try:
            if hasattr(self.model, "parameters"):
                return sum(
                    p.numel() if hasattr(p, "numel") else 0
                    for p in self.model.parameters()
                    if hasattr(p, "requires_grad") and p.requires_grad
                )
        except Exception as e:
            logger.warning(
                "failed_to_count_parameters",
                model_id=self.model_id,
                error=str(e),
            )
        return self.metadata.get("n_parameters", 0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "config_hash": self.config.compute_hash(),
            "created_at": self.created_at,
            "n_parameters": self.n_parameters,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate model summary."""
        lines = [
            f"Model: {self.name}",
            f"ID: {self.model_id}",
            f"Parameters: {self.n_parameters:,}",
            f"Preset: {self.config.preset.value}",
            f"d_model: {self.config.d_model}",
            f"n_layers: {self.config.n_layers}",
            f"Created: {self.created_at}",
        ]
        return "\n".join(lines)


class ModelBuilder:
    """Builder for creating prototype models.

    Supports various model architectures with preset
    configurations for rapid experimentation.

    Attributes:
        config: Prototype configuration.
        model_factory: Factory function for creating models.

    """

    def __init__(
        self,
        config: PrototypeConfig | None = None,
        model_factory: Callable[[PrototypeConfig], Any] | None = None,
    ) -> None:
        """Initialize model builder.

        Args:
            config: Prototype configuration.
            model_factory: Optional custom model factory.

        """
        self.config = config or create_prototype_config()
        self._model_factory = model_factory or self._default_factory
        self._models: list[PrototypeModel] = []
        self._custom_components: dict[str, Callable[..., Any]] = {}
        self._logger = logger.bind(builder="ModelBuilder")

    @property
    def models(self) -> list[PrototypeModel]:
        """Get all built models."""
        return self._models

    def register_component(
        self,
        name: str,
        factory: Callable[..., Any],
    ) -> None:
        """Register a custom component factory.

        Args:
            name: Component name.
            factory: Factory function.

        """
        self._custom_components[name] = factory
        self._logger.info("registered_component", name=name)

    def build(
        self,
        config: PrototypeConfig | None = None,
        name: str | None = None,
        **overrides: Any,
    ) -> PrototypeModel:
        """Build a prototype model.

        Args:
            config: Optional configuration override.
            name: Optional model name.
            **overrides: Configuration overrides.

        Returns:
            Built PrototypeModel.

        """
        # Merge configurations
        if config is None:
            config = self.config

        if overrides:
            config_dict = config.model_dump()
            config_dict.update(overrides)
            if name:
                config_dict["name"] = name
            config = PrototypeConfig(**config_dict)
        elif name:
            config = config.model_copy(update={"name": name})

        self._logger.info(
            "building_model",
            name=config.name,
            preset=config.preset.value,
            d_model=config.d_model,
        )

        # Create model
        model = self._model_factory(config)

        # Wrap in PrototypeModel
        prototype = PrototypeModel(
            model_id=str(uuid.uuid4())[:8],
            config=config,
            model=model,
            metadata={
                "preset": config.preset.value,
                "board_sizes": config.board_sizes,
            },
        )

        self._models.append(prototype)
        self._logger.info(
            "model_built",
            model_id=prototype.model_id,
            n_parameters=prototype.n_parameters,
        )

        return prototype

    def build_from_preset(
        self,
        preset: str | PresetType,
        name: str | None = None,
        **overrides: Any,
    ) -> PrototypeModel:
        """Build a model from a preset.

        Args:
            preset: Preset type.
            name: Optional model name.
            **overrides: Configuration overrides.

        Returns:
            Built PrototypeModel.

        """
        config = create_prototype_config(
            name=name or f"prototype_{preset if isinstance(preset, str) else preset.value}",
            preset=preset,
            **overrides,
        )
        return self.build(config=config)

    def build_sweep(
        self,
        param_name: str,
        values: list[Any],
        base_config: PrototypeConfig | None = None,
    ) -> list[PrototypeModel]:
        """Build multiple models sweeping a parameter.

        Args:
            param_name: Parameter name to sweep.
            values: Values to try.
            base_config: Base configuration.

        Returns:
            List of built models.

        """
        models = []
        base = base_config or self.config

        for value in values:
            name = f"{base.name}_{param_name}_{value}"
            model = self.build(
                config=base,
                name=name,
                **{param_name: value},
            )
            models.append(model)

        self._logger.info(
            "sweep_complete",
            param=param_name,
            n_models=len(models),
        )

        return models

    def _default_factory(self, config: PrototypeConfig) -> Any:
        """Default model factory.

        Creates a simple placeholder model for testing.
        Real implementations should override this.
        """

        # Create a simple mock model for prototyping
        class SimpleMockModel:
            def __init__(self, config: PrototypeConfig) -> None:
                self._config = config
                self._params: list[Any] = []
                # Simulate parameters
                n_params = config.d_model * config.n_layers * 4
                self._n_params = n_params

            def forward(self, x: Any) -> Any:
                return x

            def parameters(self) -> list[Any]:
                return self._params

            def __call__(self, x: Any) -> Any:
                return self.forward(x)

        return SimpleMockModel(config)

    def clear(self) -> None:
        """Clear all built models."""
        self._models.clear()
        self._logger.info("models_cleared")


def create_model_builder(
    preset: str | PresetType = PresetType.SMALL,
    model_factory: Callable[[PrototypeConfig], Any] | None = None,
    **kwargs: Any,
) -> ModelBuilder:
    """Create a model builder.

    Args:
        preset: Preset configuration type.
        model_factory: Optional custom model factory.
        **kwargs: Additional configuration.

    Returns:
        Configured ModelBuilder.

    """
    config = create_prototype_config(preset=preset, **kwargs)
    return ModelBuilder(config=config, model_factory=model_factory)
