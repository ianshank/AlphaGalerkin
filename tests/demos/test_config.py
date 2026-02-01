"""Tests for demo configuration schemas.

Tests cover:
- Configuration validation
- Constraint enforcement
- Cross-field validation
- Environment variable loading
- Default values
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.demos.config import (
    ArchitectureDemoConfig,
    BenchmarkDemoConfig,
    ColorScheme,
    DemoConfig,
    GameDemoConfig,
    PhysicsDemoConfig,
    VisualizationConfig,
)


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = VisualizationConfig()
        assert config.figure_width == 8.0
        assert config.figure_height == 6.0
        assert config.dpi == 100
        assert config.color_scheme == ColorScheme.VIRIDIS
        assert config.board_wood_color == "#e3c586"

    def test_valid_custom_values(self) -> None:
        """Test valid custom values are accepted."""
        config = VisualizationConfig(
            figure_width=10.0,
            figure_height=8.0,
            dpi=150,
            color_scheme=ColorScheme.PLASMA,
        )
        assert config.figure_width == 10.0
        assert config.figure_height == 8.0
        assert config.dpi == 150
        assert config.color_scheme == ColorScheme.PLASMA

    def test_figure_width_bounds(self) -> None:
        """Test figure_width constraint enforcement."""
        with pytest.raises(ValidationError):
            VisualizationConfig(figure_width=1.0)  # Too small
        with pytest.raises(ValidationError):
            VisualizationConfig(figure_width=25.0)  # Too large

    def test_dpi_bounds(self) -> None:
        """Test dpi constraint enforcement."""
        with pytest.raises(ValidationError):
            VisualizationConfig(dpi=10)  # Too small
        with pytest.raises(ValidationError):
            VisualizationConfig(dpi=500)  # Too large

    def test_valid_hex_color(self) -> None:
        """Test valid hex color formats."""
        config = VisualizationConfig(background_color="#ff0000")
        assert config.background_color == "#ff0000"

        config = VisualizationConfig(background_color="#fff")
        assert config.background_color == "#fff"

    def test_invalid_hex_color(self) -> None:
        """Test invalid hex color formats are rejected."""
        with pytest.raises(ValidationError):
            VisualizationConfig(background_color="#ff00")  # Invalid length

    def test_animation_interval_bounds(self) -> None:
        """Test animation_interval_ms constraints."""
        config = VisualizationConfig(animation_interval_ms=500)
        assert config.animation_interval_ms == 500

        with pytest.raises(ValidationError):
            VisualizationConfig(animation_interval_ms=5)  # Too small


class TestPhysicsDemoConfig:
    """Tests for PhysicsDemoConfig."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = PhysicsDemoConfig()
        assert config.train_grid_size == 9
        assert config.eval_grid_sizes == [9, 13, 19]
        assert config.n_charges == 5
        assert config.mse_threshold == 0.05

    def test_eval_sizes_sorted_and_unique(self) -> None:
        """Test eval_grid_sizes are sorted and deduplicated."""
        config = PhysicsDemoConfig(eval_grid_sizes=[19, 9, 13, 9])
        assert config.eval_grid_sizes == [9, 13, 19]

    def test_eval_sizes_validation(self) -> None:
        """Test eval_grid_sizes validation."""
        with pytest.raises(ValidationError):
            PhysicsDemoConfig(eval_grid_sizes=[])  # Empty

        with pytest.raises(ValidationError):
            PhysicsDemoConfig(eval_grid_sizes=[3])  # Too small

        with pytest.raises(ValidationError):
            PhysicsDemoConfig(eval_grid_sizes=[100])  # Too large

    def test_max_grid_size_constraint(self) -> None:
        """Test eval sizes must not exceed max_grid_size."""
        with pytest.raises(ValidationError):
            PhysicsDemoConfig(
                max_grid_size=15,
                eval_grid_sizes=[9, 13, 19],  # 19 > 15
            )

    def test_charge_std_bounds(self) -> None:
        """Test charge_std constraint enforcement."""
        config = PhysicsDemoConfig(charge_std=5.0)
        assert config.charge_std == 5.0

        with pytest.raises(ValidationError):
            PhysicsDemoConfig(charge_std=0.0)  # Must be > 0

        with pytest.raises(ValidationError):
            PhysicsDemoConfig(charge_std=20.0)  # Too large


class TestBenchmarkDemoConfig:
    """Tests for BenchmarkDemoConfig."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = BenchmarkDemoConfig()
        assert config.benchmark_sizes == [81, 169, 361, 625, 900]
        assert config.batch_size == 32
        assert config.d_model == 256
        assert config.n_heads == 8

    def test_benchmark_sizes_sorted(self) -> None:
        """Test benchmark_sizes are sorted."""
        config = BenchmarkDemoConfig(benchmark_sizes=[900, 81, 361])
        assert config.benchmark_sizes == [81, 361, 900]

    def test_device_options(self) -> None:
        """Test device option validation."""
        for device in ["auto", "cpu", "cuda"]:
            config = BenchmarkDemoConfig(device=device)
            assert config.device == device

        with pytest.raises(ValidationError):
            BenchmarkDemoConfig(device="tpu")  # Invalid

    def test_n_benchmark_runs_bounds(self) -> None:
        """Test n_benchmark_runs constraints."""
        config = BenchmarkDemoConfig(n_benchmark_runs=50)
        assert config.n_benchmark_runs == 50

        with pytest.raises(ValidationError):
            BenchmarkDemoConfig(n_benchmark_runs=2)  # Too small

        with pytest.raises(ValidationError):
            BenchmarkDemoConfig(n_benchmark_runs=200)  # Too large


class TestGameDemoConfig:
    """Tests for GameDemoConfig."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = GameDemoConfig()
        assert config.board_size == 9
        assert config.komi == 6.5
        assert config.n_simulations == 60
        assert config.c_puct == 1.5

    def test_board_size_bounds(self) -> None:
        """Test board_size constraint enforcement."""
        config = GameDemoConfig(board_size=19)
        assert config.board_size == 19

        with pytest.raises(ValidationError):
            GameDemoConfig(board_size=3)  # Too small

        with pytest.raises(ValidationError):
            GameDemoConfig(board_size=25)  # Too large

    def test_n_simulations_bounds(self) -> None:
        """Test n_simulations constraint enforcement."""
        config = GameDemoConfig(n_simulations=100)
        assert config.n_simulations == 100

        with pytest.raises(ValidationError):
            GameDemoConfig(n_simulations=5)  # Too small

        with pytest.raises(ValidationError):
            GameDemoConfig(n_simulations=2000)  # Too large

    def test_temperature_bounds(self) -> None:
        """Test temperature constraint enforcement."""
        config = GameDemoConfig(temperature=0.5)
        assert config.temperature == 0.5

        config = GameDemoConfig(temperature=0.0)  # Zero is valid
        assert config.temperature == 0.0

        with pytest.raises(ValidationError):
            GameDemoConfig(temperature=-0.1)  # Negative invalid

    def test_analysis_features(self) -> None:
        """Test analysis feature toggles."""
        config = GameDemoConfig(
            show_policy_heatmap=False,
            show_value_estimate=False,
            show_move_suggestions=5,
        )
        assert config.show_policy_heatmap is False
        assert config.show_value_estimate is False
        assert config.show_move_suggestions == 5


class TestArchitectureDemoConfig:
    """Tests for ArchitectureDemoConfig."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = ArchitectureDemoConfig()
        assert config.sample_board_size == 9
        assert config.n_attention_heads == 8
        assert config.n_fourier_samples == 1000

    def test_fourier_frequency_range(self) -> None:
        """Test fourier_frequency_range validation."""
        config = ArchitectureDemoConfig(fourier_frequency_range=(0.5, 5.0))
        assert config.fourier_frequency_range == (0.5, 5.0)


class TestDemoConfig:
    """Tests for root DemoConfig."""

    def test_default_values(self) -> None:
        """Test default values and sub-configs are created."""
        config = DemoConfig()
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.device == "auto"
        assert config.seed == 42
        assert isinstance(config.game, GameDemoConfig)
        assert isinstance(config.physics, PhysicsDemoConfig)
        assert isinstance(config.benchmark, BenchmarkDemoConfig)
        assert isinstance(config.architecture, ArchitectureDemoConfig)

    def test_custom_sub_configs(self) -> None:
        """Test custom sub-configurations."""
        config = DemoConfig(
            game=GameDemoConfig(board_size=13),
            physics=PhysicsDemoConfig(train_grid_size=7),
            debug=True,
        )
        assert config.game.board_size == 13
        assert config.physics.train_grid_size == 7
        assert config.debug is True

    def test_from_env_with_defaults(self) -> None:
        """Test from_env with no environment variables set."""
        # Should work without errors and use defaults
        config = DemoConfig.from_env()
        assert config.debug is False  # Default

    def test_log_level_options(self) -> None:
        """Test log_level validation."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config = DemoConfig(log_level=level)
            assert config.log_level == level

        with pytest.raises(ValidationError):
            DemoConfig(log_level="TRACE")  # Invalid


class TestColorScheme:
    """Tests for ColorScheme enum."""

    def test_all_schemes_available(self) -> None:
        """Test all color schemes are defined."""
        schemes = [
            ColorScheme.VIRIDIS,
            ColorScheme.PLASMA,
            ColorScheme.INFERNO,
            ColorScheme.MAGMA,
            ColorScheme.COOLWARM,
            ColorScheme.SEISMIC,
        ]
        assert len(schemes) == 6

    def test_scheme_values(self) -> None:
        """Test scheme values match matplotlib names."""
        assert ColorScheme.VIRIDIS.value == "viridis"
        assert ColorScheme.PLASMA.value == "plasma"


class TestConfigImmutability:
    """Tests for configuration immutability."""

    def test_forbid_extra_fields(self) -> None:
        """Test extra fields are rejected."""
        with pytest.raises(ValidationError):
            VisualizationConfig(unknown_field=42)

        with pytest.raises(ValidationError):
            PhysicsDemoConfig(extra_param="value")

    def test_validate_on_assignment(self) -> None:
        """Test validation occurs on assignment."""
        config = VisualizationConfig()

        # This should work
        config.figure_width = 10.0
        assert config.figure_width == 10.0

        # This should fail validation
        with pytest.raises(ValidationError):
            config.figure_width = 1.0  # Below minimum


class TestNestedValidation:
    """Tests for nested configuration validation."""

    def test_nested_visualization_config(self) -> None:
        """Test nested VisualizationConfig validation."""
        config = PhysicsDemoConfig(
            visualization=VisualizationConfig(dpi=200)
        )
        assert config.visualization.dpi == 200

    def test_nested_invalid_config(self) -> None:
        """Test invalid nested config is rejected."""
        with pytest.raises(ValidationError):
            PhysicsDemoConfig(
                visualization=VisualizationConfig(dpi=10)  # Invalid
            )
