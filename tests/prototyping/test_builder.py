"""Tests for model builder."""

from __future__ import annotations

from typing import Any


from src.prototyping.config import PrototypeConfig, PresetType
from src.prototyping.builder import (
    ModelBuilder,
    PrototypeModel,
    create_model_builder,
)


class TestPrototypeModel:
    """Tests for PrototypeModel dataclass."""

    def test_initialization(self, prototype_model: PrototypeModel) -> None:
        """Test model initialization."""
        assert prototype_model.model_id is not None
        assert prototype_model.config is not None
        assert prototype_model.model is not None

    def test_name_property(self, prototype_model: PrototypeModel) -> None:
        """Test name property."""
        assert prototype_model.name == "test_model"

    def test_n_parameters(self, prototype_model: PrototypeModel) -> None:
        """Test parameter counting."""
        # Mock model may return 0
        assert prototype_model.n_parameters >= 0

    def test_to_dict(self, prototype_model: PrototypeModel) -> None:
        """Test serialization to dict."""
        data = prototype_model.to_dict()

        assert "model_id" in data
        assert "name" in data
        assert "config_hash" in data
        assert "n_parameters" in data

    def test_summary(self, prototype_model: PrototypeModel) -> None:
        """Test summary generation."""
        summary = prototype_model.summary()

        assert "test_model" in summary
        assert "Parameters" in summary
        assert "d_model" in summary


class TestModelBuilder:
    """Tests for ModelBuilder."""

    def test_initialization(self, model_builder: ModelBuilder) -> None:
        """Test builder initialization."""
        assert model_builder.config is not None
        assert len(model_builder.models) == 0

    def test_build_default(self, model_builder: ModelBuilder) -> None:
        """Test building with default config."""
        model = model_builder.build()

        assert model is not None
        assert model.model is not None
        assert len(model_builder.models) == 1

    def test_build_with_name(self, model_builder: ModelBuilder) -> None:
        """Test building with custom name."""
        model = model_builder.build(name="custom_model")

        assert model.name == "custom_model"

    def test_build_with_overrides(self, model_builder: ModelBuilder) -> None:
        """Test building with config overrides."""
        model = model_builder.build(d_model=128, n_layers=4)

        assert model.config.d_model == 128
        assert model.config.n_layers == 4

    def test_build_from_preset(self, model_builder: ModelBuilder) -> None:
        """Test building from preset."""
        model = model_builder.build_from_preset(PresetType.LARGE)

        assert model.config.preset == PresetType.LARGE
        assert model.config.d_model == 256

    def test_build_sweep(self, model_builder: ModelBuilder) -> None:
        """Test building parameter sweep."""
        models = model_builder.build_sweep(
            param_name="d_model",
            values=[32, 64, 128],
        )

        assert len(models) == 3
        assert models[0].config.d_model == 32
        assert models[1].config.d_model == 64
        assert models[2].config.d_model == 128

    def test_register_component(self, model_builder: ModelBuilder) -> None:
        """Test registering custom component."""
        def custom_factory(config: PrototypeConfig) -> dict[str, Any]:
            return {"type": "custom", "config": config}

        model_builder.register_component("custom", custom_factory)

        assert "custom" in model_builder._custom_components

    def test_clear(self, model_builder: ModelBuilder) -> None:
        """Test clearing models."""
        model_builder.build()
        model_builder.build()
        assert len(model_builder.models) == 2

        model_builder.clear()
        assert len(model_builder.models) == 0

    def test_custom_model_factory(self) -> None:
        """Test with custom model factory."""
        class CustomModel:
            def __init__(self, config: PrototypeConfig) -> None:
                self.d_model = config.d_model

        def factory(config: PrototypeConfig) -> CustomModel:
            return CustomModel(config)

        builder = ModelBuilder(model_factory=factory)
        model = builder.build()

        assert isinstance(model.model, CustomModel)
        assert model.model.d_model == 64  # Default small preset


class TestCreateModelBuilder:
    """Tests for create_model_builder factory."""

    def test_create_default(self) -> None:
        """Test creating default builder."""
        builder = create_model_builder()
        assert builder.config.preset == PresetType.SMALL

    def test_create_with_preset(self) -> None:
        """Test creating with preset."""
        builder = create_model_builder(preset="large")
        assert builder.config.preset == PresetType.LARGE

    def test_create_with_factory(self) -> None:
        """Test creating with custom factory."""
        def factory(config: PrototypeConfig) -> str:
            return "custom"

        builder = create_model_builder(model_factory=factory)
        model = builder.build()

        assert model.model == "custom"
