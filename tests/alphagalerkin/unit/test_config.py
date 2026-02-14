"""Tests for configuration system."""
from __future__ import annotations

import pytest

from src.alphagalerkin.core.config import (
    AlphaGalerkinConfig,
    EnvironmentConfig,
    MCTSConfig,
    ReplayConfig,
)
from src.alphagalerkin.core.exceptions import (
    ConfigError,
)


class TestAlphaGalerkinConfig:
    """Tests for the root AlphaGalerkinConfig."""

    def test_default_config_is_valid(self) -> None:
        """Default construction should produce valid config."""
        config = AlphaGalerkinConfig()
        assert config.mcts.num_simulations > 0
        assert config.mcts.c_puct > 0

    def test_device_accepts_cpu(self) -> None:
        config = AlphaGalerkinConfig(device="cpu")
        assert config.device == "cpu"

    def test_device_accepts_cuda(self) -> None:
        config = AlphaGalerkinConfig(device="cuda:0")
        assert config.device == "cuda:0"

    def test_config_serialization_roundtrip(
        self, default_config: AlphaGalerkinConfig,
    ) -> None:
        """JSON serialization then deserialization preserves values."""
        json_str = default_config.model_dump_json()
        restored = AlphaGalerkinConfig.model_validate_json(json_str)
        assert (
            restored.mcts.num_simulations
            == default_config.mcts.num_simulations
        )
        assert restored.device == default_config.device

    def test_from_yaml_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(ConfigError):
            AlphaGalerkinConfig.from_yaml(
                tmp_path / "nonexistent.yaml"
            )

    def test_from_yaml_loads_values(self, tmp_path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "mcts:\n  num_simulations: 100\n"
        )
        config = AlphaGalerkinConfig.from_yaml(config_file)
        assert config.mcts.num_simulations == 100

    def test_env_var_override(
        self, tmp_path, monkeypatch,
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "mcts:\n  num_simulations: 100\n"
        )
        monkeypatch.setenv(
            "AG_MCTS__NUM_SIMULATIONS", "1600"
        )
        config = AlphaGalerkinConfig.from_yaml(config_file)
        assert config.mcts.num_simulations == 1600

    def test_seed_propagation(self) -> None:
        """Root seed should propagate to training sub-config."""
        config = AlphaGalerkinConfig(seed=123)
        assert config.training.seed == 123

    def test_experiment_name_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            AlphaGalerkinConfig(experiment_name="")


class TestMCTSConfig:
    """Tests for MCTS configuration validation."""

    def test_valid_defaults(self) -> None:
        config = MCTSConfig()
        assert config.num_simulations >= 1
        assert config.c_puct > 0
        assert 0.0 <= config.dirichlet_epsilon <= 1.0

    @pytest.mark.parametrize(
        "field,value",
        [
            ("num_simulations", 0),
            ("c_puct", -1.0),
            ("max_tree_depth", 0),
            ("dirichlet_epsilon", 1.5),
            ("dirichlet_epsilon", -0.1),
            ("dirichlet_alpha", 0.0),
            ("dirichlet_alpha", -1.0),
        ],
    )
    def test_rejects_invalid_values(
        self, field: str, value: float,
    ) -> None:
        with pytest.raises(ValueError):
            MCTSConfig(**{field: value})

    def test_action_topk_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            MCTSConfig(action_topk=0)


class TestEnvironmentConfig:
    """Tests for environment configuration validation."""

    def test_valid_defaults(self) -> None:
        config = EnvironmentConfig()
        assert config.max_dof > 0
        assert config.max_steps >= 1
        assert config.error_tolerance > 0

    def test_initial_poly_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError):
            EnvironmentConfig(
                initial_polynomial_order=10,
                max_polynomial_order=5,
            )

    def test_min_element_size_positive(self) -> None:
        with pytest.raises(ValueError):
            EnvironmentConfig(min_element_size=-1.0)


class TestReplayConfig:
    """Tests for replay buffer configuration validation."""

    def test_valid_defaults(self) -> None:
        config = ReplayConfig()
        assert config.capacity >= 1000
        assert config.min_size_to_train >= 1

    def test_beta_start_exceeds_end_raises(self) -> None:
        with pytest.raises(ValueError):
            ReplayConfig(
                priority_beta_start=0.9,
                priority_beta_end=0.4,
            )

    def test_priority_alpha_bounds(self) -> None:
        ReplayConfig(priority_alpha=0.0)
        ReplayConfig(priority_alpha=1.0)
        with pytest.raises(ValueError):
            ReplayConfig(priority_alpha=1.5)
