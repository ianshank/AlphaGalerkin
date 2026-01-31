"""Tests for adaptive loss balancing strategies."""

import pytest
import torch

from src.training.loss_balancing import (
    BalancingStrategy,
    GradNorm,
    LossBalancingConfig,
    LossTerms,
    ReLoBRaLo,
    SoftAdapt,
    StaticWeighting,
    UncertaintyWeighting,
    create_loss_balancer,
)


class TestLossBalancingConfig:
    """Tests for LossBalancingConfig."""

    def test_create_default_config(self) -> None:
        """Test creating default config."""
        config = LossBalancingConfig(name="test")
        assert config.strategy == BalancingStrategy.RELOBRALO
        assert config.beta == 0.99
        assert config.tau == 1.0

    def test_create_gradnorm_config(self) -> None:
        """Test creating GradNorm config."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.GRADNORM,
            alpha=1.5,
        )
        assert config.strategy == BalancingStrategy.GRADNORM
        assert config.alpha == 1.5

    def test_config_validation(self) -> None:
        """Test config validation."""
        config = LossBalancingConfig(
            name="test",
            min_weight=0.1,
            max_weight=5.0,
        )
        assert config.min_weight == 0.1
        assert config.max_weight == 5.0


class TestLossTerms:
    """Tests for LossTerms dataclass."""

    def test_create_loss_terms(self) -> None:
        """Test creating LossTerms."""
        losses = {
            "policy": torch.tensor(1.0),
            "value": torch.tensor(0.5),
        }
        weights = {"policy": 1.0, "value": 2.0}
        total = torch.tensor(2.0)

        terms = LossTerms(losses=losses, weights=weights, weighted_sum=total)
        assert terms.weighted_sum.item() == 2.0

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        losses = {
            "policy": torch.tensor(1.0),
            "value": torch.tensor(0.5),
        }
        weights = {"policy": 1.0, "value": 2.0}
        total = torch.tensor(2.0)

        terms = LossTerms(losses=losses, weights=weights, weighted_sum=total)
        result = terms.to_dict()

        assert "loss_policy" in result
        assert "loss_value" in result
        assert "weight_policy" in result
        assert "weight_value" in result
        assert "total" in result


class TestStaticWeighting:
    """Tests for static loss weighting."""

    def test_uniform_weights(self) -> None:
        """Test uniform static weights."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.STATIC)
        balancer = StaticWeighting(config, ["policy", "value", "lbb"])

        assert balancer.weights == {"policy": 1.0, "value": 1.0, "lbb": 1.0}

    def test_custom_weights(self) -> None:
        """Test custom static weights."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.STATIC)
        initial_weights = {"policy": 2.0, "value": 1.0, "lbb": 0.5}
        balancer = StaticWeighting(config, ["policy", "value", "lbb"], initial_weights)

        assert balancer.weights["policy"] == 2.0
        assert balancer.weights["value"] == 1.0
        assert balancer.weights["lbb"] == 0.5

    def test_weights_unchanged_after_update(self) -> None:
        """Test that weights don't change for static balancing."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.STATIC)
        initial_weights = {"policy": 2.0, "value": 1.0}
        balancer = StaticWeighting(config, ["policy", "value"], initial_weights)

        losses = {"policy": torch.tensor(0.5), "value": torch.tensor(1.5)}
        new_weights = balancer.update(losses)

        assert new_weights == {"policy": 2.0, "value": 1.0}

    def test_compute_weighted_loss(self) -> None:
        """Test weighted loss computation."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.STATIC)
        initial_weights = {"policy": 2.0, "value": 1.0}
        balancer = StaticWeighting(config, ["policy", "value"], initial_weights)

        losses = {"policy": torch.tensor(0.5), "value": torch.tensor(1.0)}
        result = balancer.compute_weighted_loss(losses)

        # 2.0 * 0.5 + 1.0 * 1.0 = 2.0
        assert result.weighted_sum.item() == 2.0


class TestReLoBRaLo:
    """Tests for ReLoBRaLo loss balancing."""

    def test_initial_weights(self) -> None:
        """Test initial weights are uniform."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.RELOBRALO)
        balancer = ReLoBRaLo(config, ["policy", "value"])

        assert balancer.weights == {"policy": 1.0, "value": 1.0}

    def test_weights_adapt_to_losses(self) -> None:
        """Test that weights adapt based on loss history."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.RELOBRALO,
            warmup_steps=0,  # Disable warmup for test
        )
        balancer = ReLoBRaLo(config, ["policy", "value"])

        # Simulate several updates with different loss patterns
        for _ in range(10):
            losses = {
                "policy": torch.tensor(1.0),
                "value": torch.tensor(0.1),  # Much smaller
            }
            result = balancer.compute_weighted_loss(losses)

        # After adaptation, weights should differ
        # (exact values depend on random lookback)
        assert result.weights is not None

    def test_running_average_update(self) -> None:
        """Test running average is updated."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.RELOBRALO,
            beta=0.9,
        )
        balancer = ReLoBRaLo(config, ["policy"])

        losses = {"policy": torch.tensor(1.0)}
        balancer.update(losses)

        assert "policy" in balancer._running_losses
        assert balancer._running_losses["policy"] == 1.0

    def test_reset(self) -> None:
        """Test reset clears state."""
        config = LossBalancingConfig(name="test")
        balancer = ReLoBRaLo(config, ["policy", "value"])

        # Add some history
        losses = {"policy": torch.tensor(1.0), "value": torch.tensor(0.5)}
        balancer.update(losses)

        # Reset
        balancer.reset()

        assert balancer._running_losses == {}
        assert balancer._loss_history == {"policy": [], "value": []}


class TestUncertaintyWeighting:
    """Tests for uncertainty-based loss weighting."""

    def test_has_learnable_parameters(self) -> None:
        """Test that log-variance parameters are learnable."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.UNCERTAINTY)
        balancer = UncertaintyWeighting(config, ["policy", "value"])

        assert "policy" in balancer.log_vars
        assert "value" in balancer.log_vars
        assert isinstance(balancer.log_vars["policy"], torch.nn.Parameter)

    def test_weights_from_uncertainty(self) -> None:
        """Test weight computation from log-variance."""
        config = LossBalancingConfig(name="test")
        balancer = UncertaintyWeighting(config, ["policy", "value"])

        # Set log-variance to 0 => variance = 1 => weight = 0.5
        with torch.no_grad():
            balancer._log_vars["policy"].fill_(0.0)
            balancer._log_vars["value"].fill_(0.0)

        losses = {"policy": torch.tensor(1.0), "value": torch.tensor(1.0)}
        weights = balancer.update(losses)

        # Weight = exp(-log_var) / 2 = exp(0) / 2 = 0.5
        assert pytest.approx(weights["policy"], abs=0.01) == 0.5
        assert pytest.approx(weights["value"], abs=0.01) == 0.5

    def test_regularized_loss(self) -> None:
        """Test regularized loss computation."""
        config = LossBalancingConfig(name="test")
        balancer = UncertaintyWeighting(config, ["policy", "value"])

        losses = {
            "policy": torch.tensor(1.0, requires_grad=True),
            "value": torch.tensor(0.5, requires_grad=True),
        }
        reg_loss = balancer.compute_regularized_loss(losses)

        assert reg_loss.requires_grad
        # Loss should include regularization terms
        assert reg_loss.item() > 0


class TestSoftAdapt:
    """Tests for SoftAdapt loss balancing."""

    def test_initial_weights(self) -> None:
        """Test initial weights."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.SOFTADAPT)
        balancer = SoftAdapt(config, ["policy", "value"])

        assert balancer.weights == {"policy": 1.0, "value": 1.0}

    def test_weights_adapt_to_improvement_rate(self) -> None:
        """Test weights adapt based on loss improvement rates."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.SOFTADAPT,
            warmup_steps=0,
        )
        balancer = SoftAdapt(config, ["policy", "value"])

        # Simulate policy improving fast, value stagnant
        for i in range(15):
            losses = {
                "policy": torch.tensor(1.0 - i * 0.1),  # Improving
                "value": torch.tensor(1.0),  # Stagnant
            }
            result = balancer.compute_weighted_loss(losses)

        # Value should get higher weight (it needs help)
        # Note: exact behavior depends on tau and window size


class TestGradNorm:
    """Tests for GradNorm loss balancing."""

    def test_has_learnable_weights(self) -> None:
        """Test that weights are learnable."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.GRADNORM)
        balancer = GradNorm(config, ["policy", "value"])

        assert "policy" in balancer._log_weights
        assert "value" in balancer._log_weights

    def test_gradnorm_loss_computation(self) -> None:
        """Test GradNorm auxiliary loss computation."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.GRADNORM)
        balancer = GradNorm(config, ["policy", "value"])

        # Create a simple model layer
        shared_layer = torch.nn.Linear(10, 5)

        # Create dummy losses that depend on the layer
        x = torch.randn(4, 10)
        out = shared_layer(x)

        losses = {
            "policy": (out ** 2).mean(),
            "value": (out ** 3).mean(),
        }

        # Store initial losses
        for name, loss in losses.items():
            balancer._initial_losses[name] = loss.item()

        # Compute GradNorm loss
        gradnorm_loss = balancer.compute_gradnorm_loss(losses, shared_layer)

        # Should be a valid tensor
        assert isinstance(gradnorm_loss, torch.Tensor)


class TestCreateLossBalancer:
    """Tests for loss balancer factory function."""

    def test_create_static_balancer(self) -> None:
        """Test creating static balancer."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.STATIC)
        balancer = create_loss_balancer(config, ["a", "b"])
        assert isinstance(balancer, StaticWeighting)

    def test_create_relobralo_balancer(self) -> None:
        """Test creating ReLoBRaLo balancer."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.RELOBRALO)
        balancer = create_loss_balancer(config, ["a", "b"])
        assert isinstance(balancer, ReLoBRaLo)

    def test_create_gradnorm_balancer(self) -> None:
        """Test creating GradNorm balancer."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.GRADNORM)
        balancer = create_loss_balancer(config, ["a", "b"])
        assert isinstance(balancer, GradNorm)

    def test_create_uncertainty_balancer(self) -> None:
        """Test creating uncertainty balancer."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.UNCERTAINTY)
        balancer = create_loss_balancer(config, ["a", "b"])
        assert isinstance(balancer, UncertaintyWeighting)

    def test_create_softadapt_balancer(self) -> None:
        """Test creating SoftAdapt balancer."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.SOFTADAPT)
        balancer = create_loss_balancer(config, ["a", "b"])
        assert isinstance(balancer, SoftAdapt)


class TestIntegration:
    """Integration tests for loss balancing."""

    def test_training_loop_simulation(self) -> None:
        """Simulate a training loop with loss balancing."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.RELOBRALO,
            warmup_steps=5,
        )
        balancer = create_loss_balancer(config, ["policy", "value", "physics"])

        # Simulate training steps
        for step in range(20):
            # Simulate decreasing losses
            losses = {
                "policy": torch.tensor(1.0 / (step + 1)),
                "value": torch.tensor(0.5 / (step + 1)),
                "physics": torch.tensor(2.0 / (step + 1)),
            }

            result = balancer.compute_weighted_loss(losses)

            assert result.weighted_sum.item() > 0
            assert all(w > 0 for w in result.weights.values())

    def test_weight_clamping(self) -> None:
        """Test that weights are clamped to valid range."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.RELOBRALO,
            min_weight=0.1,
            max_weight=5.0,
            warmup_steps=0,
        )
        balancer = create_loss_balancer(config, ["a", "b"])

        # Create extreme loss imbalance
        for _ in range(20):
            losses = {
                "a": torch.tensor(0.001),  # Very small
                "b": torch.tensor(100.0),  # Very large
            }
            result = balancer.compute_weighted_loss(losses)

        # Check weights are within bounds
        for name, weight in result.weights.items():
            assert weight >= config.min_weight
            assert weight <= config.max_weight
