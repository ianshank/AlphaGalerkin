"""Tests for reward computation."""
from __future__ import annotations

from src.alphagalerkin.env.rewards import RewardComposer


class TestRewardComposer:
    """Tests for the multi-component reward composer."""

    def test_default_weights(self) -> None:
        composer = RewardComposer()
        reward = composer.compute(
            accuracy=1.0, efficiency=1.0, stability=1.0,
        )
        assert reward > 0

    def test_custom_weights(self) -> None:
        composer = RewardComposer(
            {"accuracy": 2.0, "efficiency": 0.0, "stability": 0.0}
        )
        reward = composer.compute(
            accuracy=1.0, efficiency=1.0, stability=1.0,
        )
        assert abs(reward - 2.0) < 1e-10

    def test_zero_components_give_zero_reward(self) -> None:
        composer = RewardComposer()
        reward = composer.compute(
            accuracy=0.0, efficiency=0.0, stability=0.0,
        )
        assert abs(reward) < 1e-10

    def test_extra_components_supported(self) -> None:
        composer = RewardComposer({"custom": 3.0})
        reward = composer.compute(custom=2.0)
        assert abs(reward - 6.0) < 1e-10

    def test_unlisted_component_weight_is_zero(self) -> None:
        """A component not in the weight map gets weight 0."""
        composer = RewardComposer({"accuracy": 1.0})
        reward = composer.compute(
            accuracy=0.0, custom_metric=5.0,
        )
        # custom_metric is not in weights -> weight 0
        assert abs(reward) < 1e-10


class TestAccuracyReward:
    """Tests for residual-based accuracy reward."""

    def test_accuracy_reward_improves_with_lower_residual(
        self,
    ) -> None:
        composer = RewardComposer()
        r1 = composer.accuracy_reward(
            0.5, prev_residual=1.0,
        )
        r2 = composer.accuracy_reward(
            0.1, prev_residual=1.0,
        )
        assert r2 > r1

    def test_accuracy_reward_clamped_to_zero(self) -> None:
        """If residual increases, reward should be 0."""
        composer = RewardComposer()
        reward = composer.accuracy_reward(
            2.0, prev_residual=1.0,
        )
        assert reward >= 0.0

    def test_accuracy_reward_without_prev(self) -> None:
        composer = RewardComposer()
        reward = composer.accuracy_reward(0.3)
        assert abs(reward - 0.7) < 1e-10

    def test_accuracy_reward_zero_residual(self) -> None:
        composer = RewardComposer()
        reward = composer.accuracy_reward(
            0.0, prev_residual=1.0,
        )
        assert abs(reward - 1.0) < 1e-10


class TestEfficiencyReward:
    """Tests for DOF-efficiency reward."""

    def test_efficiency_reward_decreases_with_more_dof(
        self,
    ) -> None:
        composer = RewardComposer()
        r1 = composer.efficiency_reward(100, 1000)
        r2 = composer.efficiency_reward(500, 1000)
        assert r1 > r2

    def test_efficiency_reward_at_zero_dof(self) -> None:
        composer = RewardComposer()
        reward = composer.efficiency_reward(0, 1000)
        assert abs(reward - 1.0) < 1e-10

    def test_efficiency_reward_at_max_dof(self) -> None:
        composer = RewardComposer()
        reward = composer.efficiency_reward(1000, 1000)
        assert abs(reward) < 1e-10


class TestStabilityReward:
    """Tests for conditioning-based stability reward."""

    def test_well_conditioned_gets_high_reward(self) -> None:
        composer = RewardComposer()
        reward = composer.stability_reward(1.0)
        assert reward > 0.8

    def test_poorly_conditioned_gets_low_reward(self) -> None:
        composer = RewardComposer()
        reward = composer.stability_reward(1e6)
        assert reward <= 0.01

    def test_negative_condition_gives_zero(self) -> None:
        composer = RewardComposer()
        reward = composer.stability_reward(-1.0)
        assert reward == 0.0
