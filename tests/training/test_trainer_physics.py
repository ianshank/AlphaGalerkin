"""Tests for CombinedAlphaGalerkinPhysicsLoss integration in Trainer.

Verifies that:
- physics_loss_type and physics_weight fields exist in TrainingConfig
- Trainer conditionally sets combined_physics_loss_fn from config
- physics_loss_type="none" (default) leaves combined_physics_loss_fn as None
- physics_loss_type="combined" instantiates CombinedAlphaGalerkinPhysicsLoss
- Config validation rejects negative physics_weight values and unknown variants
- The combined loss forward pass produces finite outputs
- 10 training steps with physics_loss_type="combined" complete without error
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import pytest
import torch
from pydantic import ValidationError

from config.schemas import (
    AlphaGalerkinConfig,
    DomainConfig,
    MCTSConfig,
    OperatorConfig,
    TrainingConfig,
)
from src.modeling.model import AlphaGalerkinModel
from src.training.losses.physics import CombinedAlphaGalerkinPhysicsLoss
from src.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def op_config() -> OperatorConfig:
    """Minimal operator config for fast testing."""
    return OperatorConfig(
        d_model=32,
        d_key=16,
        d_value=16,
        d_ffn=64,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=16,
        use_fnet_mixing=False,
    )


@pytest.fixture
def small_model(op_config: OperatorConfig) -> AlphaGalerkinModel:
    """Minimal model for testing."""
    return AlphaGalerkinModel(op_config)


def _make_config(
    op_config: OperatorConfig,
    physics_loss_type: Literal["none", "combined"] = "none",
    physics_weight: float = 0.01,
) -> AlphaGalerkinConfig:
    """Helper to build a minimal AlphaGalerkinConfig."""
    return AlphaGalerkinConfig(
        domain=DomainConfig(),
        operator=op_config,
        mcts=MCTSConfig(
            n_simulations=5,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        ),
        training=TrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=4,
            gradient_clip=1.0,
            lr_scheduler="constant",
            warmup_steps=0,
            total_steps=20,
            n_self_play_games=2,
            replay_buffer_size=50,
            checkpoint_interval=100,
            use_amp=False,
            # Physics combined loss
            physics_loss_type=physics_loss_type,
            physics_weight=physics_weight,
        ),
        experiment_name="test_physics",
        seed=42,
    )


# ---------------------------------------------------------------------------
# TrainingConfig field validation tests
# ---------------------------------------------------------------------------


class TestTrainingConfigPhysicsFields:
    """Verify new fields on TrainingConfig."""

    def test_default_physics_loss_type_is_none(self) -> None:
        """physics_loss_type should default to 'none'."""
        cfg = TrainingConfig()
        assert cfg.physics_loss_type == "none"

    def test_default_physics_weight(self) -> None:
        """physics_weight should default to 0.01."""
        cfg = TrainingConfig()
        assert cfg.physics_weight == 0.01

    def test_physics_loss_type_combined(self) -> None:
        """physics_loss_type='combined' is a valid value."""
        cfg = TrainingConfig(physics_loss_type="combined")
        assert cfg.physics_loss_type == "combined"

    def test_invalid_physics_loss_type_raises(self) -> None:
        """Invalid physics_loss_type should raise ValidationError."""
        with pytest.raises(ValidationError):
            TrainingConfig(physics_loss_type="unknown_variant")  # type: ignore[arg-type]

    def test_negative_physics_weight_rejected(self) -> None:
        """Negative physics_weight should fail validation (ge=0 constraint)."""
        with pytest.raises(ValidationError):
            TrainingConfig(physics_weight=-0.1)

    def test_zero_physics_weight_allowed(self) -> None:
        """Zero physics_weight is allowed (ge=0)."""
        cfg = TrainingConfig(physics_weight=0.0)
        assert cfg.physics_weight == 0.0

    def test_physics_weight_set_correctly(self) -> None:
        """Custom physics_weight is stored correctly."""
        cfg = TrainingConfig(physics_weight=0.5)
        assert cfg.physics_weight == 0.5


# ---------------------------------------------------------------------------
# Trainer initialization tests
# ---------------------------------------------------------------------------


class TestTrainerCombinedPhysicsLoss:
    """Verify Trainer sets combined_physics_loss_fn from config."""

    def test_default_config_combined_loss_is_none(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """With physics_loss_type='none', combined_physics_loss_fn should be None."""
        config = _make_config(op_config, physics_loss_type="none")
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=small_model,
                config=config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
            )
            assert trainer.combined_physics_loss_fn is None

    def test_combined_type_creates_combined_loss(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """With physics_loss_type='combined', combined_physics_loss_fn should be set."""
        config = _make_config(op_config, physics_loss_type="combined", physics_weight=0.05)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=small_model,
                config=config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
            )
            assert trainer.combined_physics_loss_fn is not None
            assert isinstance(trainer.combined_physics_loss_fn, CombinedAlphaGalerkinPhysicsLoss)

    def test_physics_weight_passed_to_combined_loss(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """The physics_weight from config should be reflected in combined_physics_loss_fn."""
        expected_weight = 0.123
        config = _make_config(
            op_config, physics_loss_type="combined", physics_weight=expected_weight
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=small_model,
                config=config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
            )
            assert trainer.combined_physics_loss_fn is not None
            assert trainer.combined_physics_loss_fn.physics_weight == pytest.approx(expected_weight)

    def test_existing_loss_fn_preserved(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """Enabling combined physics loss must not break the existing loss_fn."""
        from src.training.loss import AlphaGalerkinLoss

        config = _make_config(op_config, physics_loss_type="combined")
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=small_model,
                config=config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
            )
            # loss_fn must still be the original AlphaGalerkinLoss
            assert isinstance(trainer.loss_fn, AlphaGalerkinLoss)


# ---------------------------------------------------------------------------
# Combined loss forward pass tests
# ---------------------------------------------------------------------------


class TestCombinedPhysicsLossForward:
    """Test the combined loss forward pass produces valid outputs."""

    def test_forward_produces_finite_total(self) -> None:
        """Forward pass should return a finite total loss."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.01)
        board_size = 9
        n_actions = board_size * board_size + 1
        batch = 2

        policy_logits = torch.randn(batch, n_actions)
        value = torch.tanh(torch.randn(batch, 1))
        target_policy = torch.softmax(torch.randn(batch, n_actions), dim=-1)
        target_value = torch.zeros(batch, 1)

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert "total" in result
        total = result["total"]
        assert torch.isfinite(total), f"Expected finite total loss, got {total}"

    def test_forward_returns_all_components(self) -> None:
        """Forward should return policy, value, lbb, physics keys."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.01)
        board_size = 9
        n_actions = board_size * board_size + 1

        policy_logits = torch.randn(1, n_actions)
        value = torch.tanh(torch.randn(1, 1))
        target_policy = torch.softmax(torch.randn(1, n_actions), dim=-1)
        target_value = torch.zeros(1, 1)

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        for key in ("total", "policy", "value", "lbb", "physics"):
            assert key in result, f"Missing key '{key}' in result"


# ---------------------------------------------------------------------------
# End-to-end training tests
# ---------------------------------------------------------------------------


class TestTrainerPhysicsTraining:
    """End-to-end tests: training steps with physics_loss_type='combined'."""

    def test_10_steps_with_combined_physics_loss(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """10 training steps with physics_loss_type='combined' should complete without error."""
        config = _make_config(op_config, physics_loss_type="combined", physics_weight=0.01)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=small_model,
                config=config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
            )
            # Run 10 training steps
            trainer.train(n_steps=10, log_interval=5, checkpoint_interval=100)
            assert trainer.global_step == 10

    def test_10_steps_with_none_physics_loss(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """10 training steps with physics_loss_type='none' should also complete (baseline)."""
        config = _make_config(op_config, physics_loss_type="none")
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=small_model,
                config=config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
            )
            trainer.train(n_steps=10, log_interval=5, checkpoint_interval=100)
            assert trainer.global_step == 10

    def test_combined_produces_finite_loss(
        self, op_config: OperatorConfig
    ) -> None:
        """Both 'combined' and 'none' physics_loss_type should produce finite losses."""
        results: dict[str, list[float]] = {}

        for loss_type in ("none", "combined"):
            model = AlphaGalerkinModel(op_config)
            config = _make_config(op_config, physics_loss_type=loss_type, physics_weight=0.01)
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = Trainer(
                    model=model,
                    config=config,
                    device="cpu",
                    checkpoint_dir=Path(tmpdir),
                )
                trainer.train(n_steps=10, log_interval=1, checkpoint_interval=100)
                history = trainer.get_metrics_history()
                total_losses = [m["total_loss"] for m in history]
                results[loss_type] = total_losses

        # Both variants must produce finite losses
        for loss_type, losses in results.items():
            assert all(
                torch.isfinite(torch.tensor(v)) for v in losses
            ), f"Non-finite loss in physics_loss_type='{loss_type}': {losses}"
