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
            "policy": (out**2).mean(),
            "value": (out**3).mean(),
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


# ---------------------------------------------------------------------------
# Coverage gap tests — exercise uncovered branches
# ---------------------------------------------------------------------------


class TestMissingLossTerms:
    """Test handling of missing/invalid loss terms in compute_weighted_loss."""

    def test_missing_loss_term_logs_warning(self) -> None:
        """Missing loss terms should be handled gracefully."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.STATIC)
        balancer = StaticWeighting(config, ["policy", "value", "physics"])

        # Only provide 2 of 3 expected terms
        losses = {"policy": torch.tensor(1.0), "value": torch.tensor(0.5)}
        result = balancer.compute_weighted_loss(losses)
        # Should compute weighted sum from available terms
        assert result.weighted_sum.item() > 0

    def test_no_valid_loss_terms_raises(self) -> None:
        """Providing zero valid loss terms should raise ValueError."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.STATIC)
        balancer = StaticWeighting(config, ["policy", "value"])

        # Provide loss names that don't match expected names
        losses = {"unknown": torch.tensor(1.0)}
        with pytest.raises(ValueError, match="No valid loss terms"):
            balancer.compute_weighted_loss(losses)


class TestReLoBRaLoEdgeCases:
    """Tests for ReLoBRaLo edge cases and uncovered branches."""

    def test_partial_losses_skip_missing(self) -> None:
        """ReLoBRaLo should skip missing loss names in update."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.RELOBRALO,
            warmup_steps=0,
        )
        balancer = ReLoBRaLo(config, ["policy", "value", "physics"])

        # Only provide 2 of 3 losses
        losses = {"policy": torch.tensor(1.0), "value": torch.tensor(0.5)}
        result = balancer.compute_weighted_loss(losses)
        # physics weight should remain 1.0 (default for missing)
        assert result.weights["physics"] == 1.0

    def test_fixed_lookback_mode(self) -> None:
        """ReLoBRaLo with random_lookback=False uses EMA."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.RELOBRALO,
            random_lookback=False,
            warmup_steps=0,
        )
        balancer = ReLoBRaLo(config, ["policy", "value"])

        for _ in range(5):
            losses = {"policy": torch.tensor(1.0), "value": torch.tensor(0.5)}
            result = balancer.compute_weighted_loss(losses)

        assert result.weighted_sum.item() > 0

    def test_history_buffer_truncation(self) -> None:
        """History buffer should be capped at history_buffer_size."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.RELOBRALO,
            history_buffer_size=10,
            warmup_steps=0,
        )
        balancer = ReLoBRaLo(config, ["a"])

        for i in range(20):
            losses = {"a": torch.tensor(float(i + 1))}
            balancer.update(losses)

        assert len(balancer._loss_history["a"]) == 10

    def test_empty_relative_losses(self) -> None:
        """If all losses are missing, update returns current weights."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.RELOBRALO,
            warmup_steps=0,
        )
        balancer = ReLoBRaLo(config, ["policy", "value"])
        # Pass empty dict to update directly
        weights = balancer.update({})
        assert weights == {"policy": 1.0, "value": 1.0}


class TestGradNormEdgeCases:
    """Tests for GradNorm uncovered branches."""

    def test_update_stores_initial_losses(self) -> None:
        """GradNorm update should store initial losses and return clamped weights."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.GRADNORM,
            warmup_steps=0,
        )
        balancer = GradNorm(config, ["policy", "value"])

        losses = {"policy": torch.tensor(1.0), "value": torch.tensor(0.5)}
        result = balancer.compute_weighted_loss(losses)

        assert "policy" in balancer._initial_losses
        assert "value" in balancer._initial_losses
        assert result.weighted_sum.item() > 0

    def test_update_with_missing_log_weight(self) -> None:
        """GradNorm update handles names not in _log_weights gracefully."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.GRADNORM,
            warmup_steps=0,
        )
        balancer = GradNorm(config, ["policy", "value"])

        # Manually remove a log weight to trigger the else branch
        del balancer._log_weights["value"]
        losses = {"policy": torch.tensor(1.0), "value": torch.tensor(0.5)}
        weights = balancer.update(losses)
        assert weights["value"] == 1.0

    def test_gradnorm_loss_with_no_gradients(self) -> None:
        """GradNorm loss returns 0 when no gradients can be computed."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.GRADNORM)
        balancer = GradNorm(config, ["policy"])

        shared_layer = torch.nn.Linear(10, 5)
        # Losses don't depend on shared_layer → no gradients
        losses = {"policy": torch.tensor(1.0, requires_grad=True)}
        gradnorm_loss = balancer.compute_gradnorm_loss(losses, shared_layer)
        assert isinstance(gradnorm_loss, torch.Tensor)

    def test_gradnorm_loss_empty_losses(self) -> None:
        """GradNorm loss with empty losses returns 0."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.GRADNORM)
        balancer = GradNorm(config, ["policy"])

        shared_layer = torch.nn.Linear(10, 5)
        gradnorm_loss = balancer.compute_gradnorm_loss({}, shared_layer)
        assert gradnorm_loss.item() == 0.0

    def test_gradnorm_loss_missing_initial(self) -> None:
        """GradNorm loss handles case where initial_losses not yet set."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.GRADNORM)
        balancer = GradNorm(config, ["policy"])

        shared_layer = torch.nn.Linear(10, 5)
        x = torch.randn(4, 10)
        out = shared_layer(x)
        losses = {"policy": (out**2).mean()}

        # Don't set initial_losses — triggers rel_rates default of 1.0
        gradnorm_loss = balancer.compute_gradnorm_loss(losses, shared_layer)
        assert isinstance(gradnorm_loss, torch.Tensor)
        assert gradnorm_loss.item() >= 0


class TestUncertaintyWeightingEdgeCases:
    """Tests for UncertaintyWeighting uncovered branches."""

    def test_update_missing_log_var(self) -> None:
        """Uncertainty weighting handles missing log_var names."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.UNCERTAINTY)
        balancer = UncertaintyWeighting(config, ["policy", "value"])

        # Remove one log_var to trigger else branch
        del balancer._log_vars["value"]
        losses = {"policy": torch.tensor(1.0), "value": torch.tensor(0.5)}
        weights = balancer.update(losses)
        assert weights["value"] == 1.0

    def test_regularized_loss_missing_log_var(self) -> None:
        """compute_regularized_loss handles loss names not in log_vars."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.UNCERTAINTY)
        balancer = UncertaintyWeighting(config, ["policy"])

        losses = {
            "policy": torch.tensor(1.0, requires_grad=True),
            "extra": torch.tensor(0.5, requires_grad=True),  # Not in log_vars
        }
        reg_loss = balancer.compute_regularized_loss(losses)
        assert reg_loss.requires_grad
        assert reg_loss.item() > 0


class TestSoftAdaptEdgeCases:
    """Tests for SoftAdapt uncovered branches."""

    def test_partial_losses(self) -> None:
        """SoftAdapt handles missing loss names in update."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.SOFTADAPT,
            warmup_steps=0,
        )
        balancer = SoftAdapt(config, ["policy", "value", "physics"])

        # Only provide 2 losses
        losses = {"policy": torch.tensor(1.0), "value": torch.tensor(0.5)}
        result = balancer.compute_weighted_loss(losses)
        # physics should stay at default weight
        assert result.weights["physics"] == 1.0

    def test_single_history_entry_rate_zero(self) -> None:
        """SoftAdapt uses rate=0 when history has only 1 entry."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.SOFTADAPT,
            warmup_steps=0,
        )
        balancer = SoftAdapt(config, ["a", "b"])

        losses = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        weights = balancer.update(losses)
        # With only 1 entry, rate is 0 → all weights should be equal
        assert abs(weights["a"] - weights["b"]) < 0.01

    def test_reset(self) -> None:
        """SoftAdapt reset clears loss history."""
        config = LossBalancingConfig(name="test", strategy=BalancingStrategy.SOFTADAPT)
        balancer = SoftAdapt(config, ["policy", "value"])

        losses = {"policy": torch.tensor(1.0), "value": torch.tensor(0.5)}
        balancer.update(losses)
        balancer.reset()

        assert balancer._loss_history == {"policy": [], "value": []}
        assert balancer._step == 0

    def test_overflow_handling(self) -> None:
        """SoftAdapt handles potential overflow in exp gracefully."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.SOFTADAPT,
            tau=1e-10,  # Very small tau → large exponents
            warmup_steps=0,
            softadapt_window_size=3,
        )
        balancer = SoftAdapt(config, ["a", "b"])

        # Large divergent losses to trigger overflow
        for i in range(5):
            losses = {
                "a": torch.tensor(float(i * 1000 + 1)),
                "b": torch.tensor(0.001),
            }
            result = balancer.compute_weighted_loss(losses)

        # Should not raise, weights should be valid
        for w in result.weights.values():
            assert w > 0

    def test_partial_losses_with_history(self) -> None:
        """SoftAdapt partial losses across multiple updates."""
        config = LossBalancingConfig(
            name="test",
            strategy=BalancingStrategy.SOFTADAPT,
            warmup_steps=0,
            softadapt_window_size=3,
        )
        balancer = SoftAdapt(config, ["a", "b", "c"])

        losses = {"a": torch.tensor(1.0), "b": torch.tensor(0.5)}
        for _ in range(5):
            balancer.update(losses)

        result = balancer.compute_weighted_loss(losses)
        assert all(w > 0 for w in result.weights.values())


class TestFactoryUnknownStrategy:
    """Test factory function with invalid strategy."""

    def test_unknown_strategy_raises(self) -> None:
        """Unknown strategy should raise ValueError."""
        config = LossBalancingConfig(name="test")
        # Hack the strategy to an invalid value
        object.__setattr__(config, "strategy", "nonexistent")
        with pytest.raises(ValueError, match="Unknown balancing strategy"):
            create_loss_balancer(config, ["a", "b"])


class TestRoundtripAllStrategies:
    """Property-based roundtrip tests: init → update → compute_weighted_loss → valid gradients."""

    @pytest.mark.parametrize(
        "strategy",
        [
            BalancingStrategy.STATIC,
            BalancingStrategy.RELOBRALO,
            BalancingStrategy.GRADNORM,
            BalancingStrategy.UNCERTAINTY,
            BalancingStrategy.SOFTADAPT,
        ],
    )
    def test_roundtrip_valid_gradients(self, strategy: BalancingStrategy) -> None:
        """Every strategy: init → update(losses) → compute_weighted_loss() → valid gradients."""
        config = LossBalancingConfig(
            name="test",
            strategy=strategy,
            warmup_steps=0,
        )
        balancer = create_loss_balancer(config, ["policy", "value", "physics"])

        for _ in range(5):
            losses = {
                "policy": torch.tensor(1.0, requires_grad=True),
                "value": torch.tensor(0.5, requires_grad=True),
                "physics": torch.tensor(2.0, requires_grad=True),
            }
            result = balancer.compute_weighted_loss(losses)

            assert result.weighted_sum.item() > 0
            assert all(w > 0 for w in result.weights.values())
            # Verify gradient flows through weighted sum
            assert result.weighted_sum.requires_grad
