"""Tests for notebook configuration utilities."""

from __future__ import annotations

import pytest

from notebooks.utils.config import (
    BenchmarkConfig,
    DemoConfig,
    GoBoardConfig,
    ModelConfig,
    PhysicsConfig,
    VisualizationConfig,
    create_demo_config,
    get_board_labels,
    get_default_board_sizes,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ModelConfig()
        assert config.d_model == 64
        assert config.n_heads == 4
        assert config.n_galerkin_layers == 2
        assert config.n_softmax_layers == 1
        assert config.n_fourier_features == 32
        assert config.input_channels == 17

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ModelConfig(d_model=128, n_heads=8)
        assert config.d_model == 128
        assert config.n_heads == 8

    def test_validation_d_model_divisible_by_heads(self) -> None:
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(ValueError, match="must be divisible by"):
            ModelConfig(d_model=65, n_heads=4)

    def test_validation_positive_d_model(self) -> None:
        """Test that d_model must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            ModelConfig(d_model=0, n_heads=1)

    def test_validation_positive_n_heads(self) -> None:
        """Test that n_heads must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            ModelConfig(d_model=64, n_heads=0)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BenchmarkConfig()
        assert config.n_warmup_runs == 5
        assert config.n_timed_runs == 50
        assert config.batch_size == 4
        assert config.n_evals == 100


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = VisualizationConfig()
        assert config.figure_width == 14.0
        assert config.figure_height == 6.0
        assert config.colormap_diverging == "RdBu"
        assert config.colormap_sequential == "viridis"


class TestPhysicsConfig:
    """Tests for PhysicsConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PhysicsConfig()
        assert config.n_charges == 3
        assert config.boundary_value == 0.0
        assert config.charge_std == 1.0


class TestGoBoardConfig:
    """Tests for GoBoardConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = GoBoardConfig()
        assert len(config.black_stone_positions) == 5
        assert len(config.white_stone_positions) == 4
        assert config.stone_radius == 0.4
        assert config.board_color == 0.82

    def test_custom_positions(self) -> None:
        """Test custom stone positions."""
        config = GoBoardConfig(
            black_stone_positions=[(0, 0), (1, 1)],
            white_stone_positions=[(2, 2)],
        )
        assert len(config.black_stone_positions) == 2
        assert len(config.white_stone_positions) == 1


class TestDemoConfig:
    """Tests for DemoConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DemoConfig()
        assert list(config.board_sizes) == [5, 9, 13, 19]
        # seq_lengths is auto-computed from board_sizes
        assert list(config.seq_lengths) == [25, 81, 169, 361]  # 5², 9², 13², 19²
        assert list(config.physics_board_sizes) == [9, 13, 19, 25]
        assert config.random_seed == 42

    def test_seq_lengths_auto_computed(self) -> None:
        """Test that seq_lengths is auto-computed from board_sizes."""
        config = DemoConfig(board_sizes=[9, 19])
        assert list(config.seq_lengths) == [81, 361]  # 9², 19²

    def test_seq_lengths_length_validation(self) -> None:
        """Test that custom seq_lengths must match board_sizes length."""
        with pytest.raises(ValueError, match="must match"):
            DemoConfig(board_sizes=[9, 19], seq_lengths=[81, 169, 361])

    def test_sub_configs_initialized(self) -> None:
        """Test that sub-configurations are initialized."""
        config = DemoConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.benchmark, BenchmarkConfig)
        assert isinstance(config.visualization, VisualizationConfig)
        assert isinstance(config.physics, PhysicsConfig)
        assert isinstance(config.go_board, GoBoardConfig)

    def test_custom_board_sizes(self) -> None:
        """Test custom board sizes."""
        config = DemoConfig(board_sizes=[9, 19])
        assert list(config.board_sizes) == [9, 19]

    def test_custom_random_seed(self) -> None:
        """Test custom random seed."""
        config = DemoConfig(random_seed=123)
        assert config.random_seed == 123


class TestCreateDemoConfig:
    """Tests for create_demo_config function."""

    def test_create_default(self) -> None:
        """Test creating default config."""
        config = create_demo_config()
        assert isinstance(config, DemoConfig)

    def test_create_with_overrides(self) -> None:
        """Test creating config with overrides."""
        config = create_demo_config(random_seed=999, board_sizes=[7, 13])
        assert config.random_seed == 999
        assert list(config.board_sizes) == [7, 13]


class TestGetDefaultBoardSizes:
    """Tests for get_default_board_sizes function."""

    def test_returns_expected_sizes(self) -> None:
        """Test that function returns expected board sizes."""
        sizes = get_default_board_sizes()
        assert sizes == [5, 9, 13, 19]

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        sizes = get_default_board_sizes()
        assert isinstance(sizes, list)


class TestGetBoardLabels:
    """Tests for get_board_labels function."""

    def test_single_size(self) -> None:
        """Test label for single size."""
        labels = get_board_labels([9])
        assert labels == ["9×9"]

    def test_multiple_sizes(self) -> None:
        """Test labels for multiple sizes."""
        labels = get_board_labels([5, 9, 13, 19])
        assert labels == ["5×5", "9×9", "13×13", "19×19"]

    def test_empty_list(self) -> None:
        """Test empty input."""
        labels = get_board_labels([])
        assert labels == []
