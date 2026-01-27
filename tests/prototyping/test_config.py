"""Tests for prototyping configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.prototyping.config import (
    PrototypeConfig,
    QuickTrainConfig,
    QuickEvalConfig,
    PresetType,
    PRESETS,
    create_prototype_config,
    create_quick_train_config,
)


class TestPresetType:
    """Tests for PresetType enum."""

    def test_all_presets_exist(self) -> None:
        """Test all presets exist."""
        assert PresetType.MINIMAL.value == "minimal"
        assert PresetType.SMALL.value == "small"
        assert PresetType.MEDIUM.value == "medium"
        assert PresetType.LARGE.value == "large"
        assert PresetType.DEBUG.value == "debug"
        assert PresetType.TRANSFER.value == "transfer"
        assert PresetType.BENCHMARK.value == "benchmark"

    def test_presets_have_configs(self) -> None:
        """Test all presets have configurations."""
        for preset in PresetType:
            assert preset in PRESETS


class TestPrototypeConfig:
    """Tests for PrototypeConfig."""

    def test_default_values(
        self, default_prototype_config: PrototypeConfig
    ) -> None:
        """Test default configuration values."""
        assert default_prototype_config.name == "test_prototype"
        assert default_prototype_config.preset == PresetType.SMALL
        assert default_prototype_config.board_sizes == [9]

    def test_required_name(self) -> None:
        """Test name is required."""
        # Empty name should fail
        with pytest.raises(ValidationError):
            PrototypeConfig(name="")

    def test_board_sizes_validation(self) -> None:
        """Test board sizes must be >= 5."""
        with pytest.raises(ValidationError):
            PrototypeConfig(board_sizes=[3])

    def test_board_sizes_sorted(self) -> None:
        """Test board sizes are sorted."""
        config = PrototypeConfig(board_sizes=[19, 9, 13])
        assert config.board_sizes == [9, 13, 19]

    def test_d_model_validation(self) -> None:
        """Test d_model bounds."""
        # Valid
        PrototypeConfig(d_model=16)
        PrototypeConfig(d_model=4096)

        # Invalid
        with pytest.raises(ValidationError):
            PrototypeConfig(d_model=8)  # Below minimum
        with pytest.raises(ValidationError):
            PrototypeConfig(d_model=5000)  # Above maximum

    def test_dropout_validation(self) -> None:
        """Test dropout bounds."""
        # Valid
        PrototypeConfig(dropout=0.0)
        PrototypeConfig(dropout=0.5)

        # Invalid
        with pytest.raises(ValidationError):
            PrototypeConfig(dropout=1.0)  # At boundary
        with pytest.raises(ValidationError):
            PrototypeConfig(dropout=-0.1)  # Below minimum

    def test_compute_hash(self, default_prototype_config: PrototypeConfig) -> None:
        """Test hash computation."""
        hash1 = default_prototype_config.compute_hash()
        hash2 = default_prototype_config.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_different_configs_different_hash(self) -> None:
        """Test different configs have different hashes."""
        config1 = PrototypeConfig(name="test1")
        config2 = PrototypeConfig(name="test2")
        assert config1.compute_hash() != config2.compute_hash()


class TestQuickTrainConfig:
    """Tests for QuickTrainConfig."""

    def test_default_values(
        self, default_train_config: QuickTrainConfig
    ) -> None:
        """Test default training config values."""
        assert default_train_config.n_epochs == 2
        assert default_train_config.batch_size == 16
        assert default_train_config.learning_rate == 1e-3

    def test_n_epochs_validation(self) -> None:
        """Test n_epochs bounds."""
        # Valid
        QuickTrainConfig(n_epochs=1)
        QuickTrainConfig(n_epochs=10000)

        # Invalid
        with pytest.raises(ValidationError):
            QuickTrainConfig(n_epochs=0)

    def test_learning_rate_validation(self) -> None:
        """Test learning rate bounds."""
        # Valid
        QuickTrainConfig(learning_rate=1e-5)
        QuickTrainConfig(learning_rate=0.5)

        # Invalid
        with pytest.raises(ValidationError):
            QuickTrainConfig(learning_rate=0)
        with pytest.raises(ValidationError):
            QuickTrainConfig(learning_rate=1.0)


class TestQuickEvalConfig:
    """Tests for QuickEvalConfig."""

    def test_default_values(
        self, default_eval_config: QuickEvalConfig
    ) -> None:
        """Test default eval config values."""
        assert default_eval_config.n_samples == 100
        assert default_eval_config.batch_size == 32
        assert "mse" in default_eval_config.metrics

    def test_n_samples_validation(self) -> None:
        """Test n_samples minimum."""
        QuickEvalConfig(n_samples=10)
        with pytest.raises(ValidationError):
            QuickEvalConfig(n_samples=5)

    def test_metrics_required(self) -> None:
        """Test at least one metric required."""
        with pytest.raises(ValidationError):
            QuickEvalConfig(metrics=[])


class TestCreatePrototypeConfig:
    """Tests for create_prototype_config factory."""

    def test_create_default(self) -> None:
        """Test creating default config."""
        config = create_prototype_config()
        assert config.name == "prototype"
        assert config.preset == PresetType.SMALL

    def test_create_with_preset(self) -> None:
        """Test creating with preset."""
        config = create_prototype_config(preset="large")
        assert config.preset == PresetType.LARGE
        assert config.d_model == 256
        assert config.n_layers == 8

    def test_create_with_overrides(self) -> None:
        """Test creating with overrides."""
        config = create_prototype_config(
            name="custom",
            preset="small",
            d_model=128,
        )
        assert config.name == "custom"
        assert config.d_model == 128


class TestCreateQuickTrainConfig:
    """Tests for create_quick_train_config factory."""

    def test_create_default(self) -> None:
        """Test creating default train config."""
        config = create_quick_train_config()
        assert config.n_epochs == 10
        assert config.batch_size == 32

    def test_create_with_preset(self) -> None:
        """Test creating with preset."""
        config = create_quick_train_config(preset="debug")
        assert config.n_epochs == 2
        assert config.batch_size == 8
        assert config.log_interval == 1

    def test_create_with_overrides(self) -> None:
        """Test creating with overrides."""
        config = create_quick_train_config(
            preset="small",
            n_epochs=50,
        )
        assert config.n_epochs == 50
