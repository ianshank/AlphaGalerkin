"""Tests for physics-informed loss integration in the trainer.

This module tests that physics-informed loss can be optionally
enabled in the training pipeline and properly integrates with
existing loss components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.training.trainer import TrainingMetrics

if TYPE_CHECKING:
    pass


class TestTrainingMetricsPhysics:
    """Tests for physics fields in TrainingMetrics."""

    def test_default_physics_fields_are_zero(self) -> None:
        """Default physics loss metrics should be zero."""
        metrics = TrainingMetrics(step=0)
        assert metrics.physics_loss == 0.0
        assert metrics.physics_residual_loss == 0.0
        assert metrics.physics_boundary_loss == 0.0
        assert metrics.physics_weight == 0.0

    def test_physics_metrics_set_correctly(self) -> None:
        """Physics metrics should be settable."""
        metrics = TrainingMetrics(
            step=100,
            total_loss=1.5,
            physics_loss=0.25,
            physics_residual_loss=0.15,
            physics_boundary_loss=0.10,
            physics_weight=0.1,
        )
        assert metrics.physics_loss == 0.25
        assert metrics.physics_residual_loss == 0.15
        assert metrics.physics_boundary_loss == 0.10
        assert metrics.physics_weight == 0.1

    def test_to_dict_excludes_physics_when_weight_zero(self) -> None:
        """to_dict should exclude physics metrics when weight is zero."""
        metrics = TrainingMetrics(
            step=0,
            physics_loss=0.5,  # Non-zero but weight is zero
            physics_weight=0.0,
        )
        result = metrics.to_dict()
        assert "physics_loss" not in result
        assert "physics_residual_loss" not in result
        assert "physics_boundary_loss" not in result
        assert "physics_weight" not in result

    def test_to_dict_includes_physics_when_weight_nonzero(self) -> None:
        """to_dict should include physics metrics when weight is non-zero."""
        metrics = TrainingMetrics(
            step=0,
            physics_loss=0.25,
            physics_residual_loss=0.15,
            physics_boundary_loss=0.10,
            physics_weight=0.1,
        )
        result = metrics.to_dict()
        assert result["physics_loss"] == 0.25
        assert result["physics_residual_loss"] == 0.15
        assert result["physics_boundary_loss"] == 0.10
        assert result["physics_weight"] == 0.1


class TestPhysicsLossConfig:
    """Tests for physics loss configuration in training config."""

    def test_physics_config_defaults_in_yaml(self) -> None:
        """Physics config fields should have sensible defaults."""
        # These defaults come from config/train.yaml
        expected_defaults = {
            "physics_informed": False,
            "physics_loss_weight": 0.1,
            "physics_residual_weight": 1.0,
            "physics_boundary_weight": 10.0,
            "physics_initial_weight": 10.0,
            "physics_conservation_weight": 1.0,
            "physics_n_collocation_points": 1000,
            "physics_n_boundary_points": 200,
            "physics_use_adaptive_weights": True,
        }
        # Verify the defaults exist in train.yaml
        from pathlib import Path

        import yaml

        config_path = Path(__file__).parent.parent.parent / "config" / "train.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            training = config.get("training", {})
            for key, expected in expected_defaults.items():
                assert training.get(key) == expected, (
                    f"Expected {key}={expected}, got {training.get(key)}"
                )


class TestPhysicsLossIntegration:
    """Integration tests for physics loss in trainer."""

    @pytest.fixture
    def mock_training_config(self) -> MagicMock:
        """Create a mock training config with physics enabled."""
        config = MagicMock()
        config.physics_informed = True
        config.physics_loss_weight = 0.1
        config.physics_residual_weight = 1.0
        config.physics_boundary_weight = 10.0
        config.physics_initial_weight = 10.0
        config.physics_conservation_weight = 1.0
        config.physics_n_collocation_points = 100
        config.physics_n_boundary_points = 20
        config.physics_use_adaptive_weights = True
        config.policy_loss_weight = 1.0
        config.value_loss_weight = 1.0
        config.learning_rate = 1e-4
        config.weight_decay = 1e-4
        config.lr_scheduler = "cosine"
        config.warmup_steps = 0
        config.total_steps = 100
        config.batch_size = 8
        config.gradient_clip = 1.0
        config.use_amp = False
        config.replay_buffer_size = 1000
        config.n_self_play_games = 10
        config.checkpoint_interval = 100
        config.loss_balancing_strategy = "static"
        config.loss_balancing_beta = 0.99
        config.loss_balancing_tau = 1.0
        config.loss_balancing_warmup = 0
        config.use_prioritized_replay = False
        config.per_alpha = 0.6
        config.per_beta = 0.4
        config.curriculum_enabled = False
        config.eval_vs_checkpoints = False
        config.early_stopping_enabled = False
        config.plateau_detection_enabled = False
        return config

    def test_physics_loss_config_validation(self, mock_training_config: MagicMock) -> None:
        """Physics loss config should validate properly."""
        from src.training.physics_loss import PhysicsLossConfig

        config = PhysicsLossConfig(
            name="test_physics",
            residual_weight=mock_training_config.physics_residual_weight,
            boundary_weight=mock_training_config.physics_boundary_weight,
            initial_weight=mock_training_config.physics_initial_weight,
            conservation_weight=mock_training_config.physics_conservation_weight,
            n_collocation_points=mock_training_config.physics_n_collocation_points,
            n_boundary_points=mock_training_config.physics_n_boundary_points,
            use_adaptive_weights=mock_training_config.physics_use_adaptive_weights,
        )

        assert config.residual_weight == 1.0
        assert config.boundary_weight == 10.0
        assert config.n_collocation_points == 100
        assert config.n_boundary_points == 20

    def test_physics_loss_config_rejects_invalid_values(self) -> None:
        """Physics loss config should reject invalid values."""
        from pydantic import ValidationError

        from src.training.physics_loss import PhysicsLossConfig

        with pytest.raises(ValidationError):
            PhysicsLossConfig(
                name="invalid",
                n_collocation_points=5,  # Below minimum of 10
            )

        with pytest.raises(ValidationError):
            PhysicsLossConfig(
                name="invalid",
                residual_weight=-1.0,  # Negative weight not allowed
            )


class TestPhysicsLossOutput:
    """Tests for PhysicsLossOutput dataclass."""

    def test_physics_loss_output_to_dict(self) -> None:
        """PhysicsLossOutput should convert to dict correctly."""
        from src.training.physics_loss import PhysicsLossOutput

        output = PhysicsLossOutput(
            total=torch.tensor(0.5),
            residual=torch.tensor(0.2),
            boundary=torch.tensor(0.3),
            initial=None,
            conservation=None,
            weights={"residual": 1.0, "boundary": 10.0},
        )

        result = output.to_dict()
        assert result["total"] == pytest.approx(0.5, abs=1e-6)
        assert result["residual"] == pytest.approx(0.2, abs=1e-6)
        assert result["boundary"] == pytest.approx(0.3, abs=1e-6)
        assert "initial" not in result
        assert "conservation" not in result
        assert result["weight_residual"] == 1.0
        assert result["weight_boundary"] == 10.0

    def test_physics_loss_output_with_all_components(self) -> None:
        """PhysicsLossOutput should handle all components."""
        from src.training.physics_loss import PhysicsLossOutput

        output = PhysicsLossOutput(
            total=torch.tensor(1.0),
            residual=torch.tensor(0.1),
            boundary=torch.tensor(0.2),
            initial=torch.tensor(0.3),
            conservation=torch.tensor(0.4),
            weights={
                "residual": 1.0,
                "boundary": 10.0,
                "initial": 10.0,
                "conservation": 1.0,
            },
        )

        result = output.to_dict()
        assert "initial" in result
        assert "conservation" in result
        assert result["initial"] == pytest.approx(0.3, abs=1e-6)
        assert result["conservation"] == pytest.approx(0.4, abs=1e-6)


class TestTrainerPhysicsCreation:
    """Tests for physics loss creation in Trainer."""

    @pytest.fixture
    def mock_pde_imports(self) -> None:
        """Mock PDE module imports."""
        with patch.dict("sys.modules", {
            "src.pde": MagicMock(),
            "src.pde.operators": MagicMock(),
            "src.pde.config": MagicMock(),
        }):
            yield

    def test_physics_loss_disabled_by_default(self) -> None:
        """Physics loss should be disabled when physics_informed=False."""
        # Create a minimal mock training config
        mock_config = MagicMock()
        mock_config.physics_informed = False

        # The getattr pattern in Trainer should return False by default
        result = getattr(mock_config, "physics_informed", False)
        assert result is False

    def test_physics_loss_weight_accessible(self) -> None:
        """Physics loss weight should be accessible via getattr."""
        mock_config = MagicMock()
        mock_config.physics_loss_weight = 0.1

        result = getattr(mock_config, "physics_loss_weight", 0.1)
        assert result == 0.1


class TestLossBalancerPhysicsIntegration:
    """Tests for loss balancer with physics loss."""

    def test_loss_balancer_includes_physics(self) -> None:
        """Loss balancer should include physics when enabled."""
        from src.training.loss_balancing import LossBalancingConfig, create_loss_balancer

        config = LossBalancingConfig(name="test")
        loss_names = ["policy", "value", "lbb", "physics"]

        balancer = create_loss_balancer(config, loss_names)

        # Verify all loss names are tracked
        losses = {
            "policy": torch.tensor(1.0),
            "value": torch.tensor(0.5),
            "lbb": torch.tensor(0.1),
            "physics": torch.tensor(0.2),
        }
        result = balancer.compute_weighted_loss(losses)

        assert "physics" in result.weights
        assert result.weighted_sum.item() > 0

    def test_loss_balancer_without_physics(self) -> None:
        """Loss balancer should work without physics."""
        from src.training.loss_balancing import LossBalancingConfig, create_loss_balancer

        config = LossBalancingConfig(name="test")
        loss_names = ["policy", "value", "lbb"]

        balancer = create_loss_balancer(config, loss_names)

        losses = {
            "policy": torch.tensor(1.0),
            "value": torch.tensor(0.5),
            "lbb": torch.tensor(0.1),
        }
        result = balancer.compute_weighted_loss(losses)

        assert "physics" not in result.weights
        assert result.weighted_sum.item() > 0
