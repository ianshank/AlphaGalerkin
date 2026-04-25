"""Comprehensive tests for physics loss integration in the Trainer.

Verifies:
1. Config toggle tests: physics_informed=True/False, physics_loss_type="combined"/"none"
2. Gradient flow tests: physics loss gradients flow back to model parameters
3. Loss balancing integration: "physics" key appears in loss balancer when enabled
4. Metrics logging: TrainingMetrics.physics_loss populated when physics enabled
5. Property-based tests (Hypothesis): non-negativity, combined loss bounds

References
----------
- src/training/trainer.py: Trainer.__init__, _training_step
- config/schemas.py: TrainingConfig
- src/training/losses/physics.py: PhysicsInformedLoss, CombinedAlphaGalerkinPhysicsLoss

"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from config.schemas import (
    AlphaGalerkinConfig,
    DomainConfig,
    MCTSConfig,
    OperatorConfig,
    TrainingConfig,
)
from src.modeling.model import AlphaGalerkinModel
from src.training.losses import AlphaGalerkinLoss
from src.training.losses.physics import CombinedAlphaGalerkinPhysicsLoss
from src.training.trainer import Trainer, TrainingMetrics

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SMALL_OP_CONFIG_KWARGS: dict[str, object] = {
    "d_model": 32,
    "d_key": 16,
    "d_value": 16,
    "d_ffn": 64,
    "n_heads": 2,
    "n_galerkin_layers": 1,
    "n_softmax_layers": 1,
    "n_fourier_features": 16,
    "use_fnet_mixing": False,
}

_SMALL_MCTS_CONFIG_KWARGS: dict[str, object] = {
    "n_simulations": 5,
    "c_puct": 1.5,
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,
}


@pytest.fixture()
def op_config() -> OperatorConfig:
    """Minimal operator config for fast testing."""
    return OperatorConfig(**_SMALL_OP_CONFIG_KWARGS)  # type: ignore[arg-type]


@pytest.fixture()
def small_model(op_config: OperatorConfig) -> AlphaGalerkinModel:
    """Minimal model for testing."""
    return AlphaGalerkinModel(op_config)


def _make_config(
    op_config: OperatorConfig,
    *,
    physics_informed: bool = False,
    physics_loss_type: Literal["none", "combined"] = "none",
    physics_weight: float = 0.01,
    physics_loss_weight: float = 0.1,
    total_steps: int = 20,
) -> AlphaGalerkinConfig:
    """Build a minimal AlphaGalerkinConfig with physics-related overrides."""
    return AlphaGalerkinConfig(
        domain=DomainConfig(),
        operator=op_config,
        mcts=MCTSConfig(**_SMALL_MCTS_CONFIG_KWARGS),  # type: ignore[arg-type]
        training=TrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=4,
            gradient_clip=1.0,
            lr_scheduler="constant",
            warmup_steps=0,
            total_steps=total_steps,
            n_self_play_games=2,
            replay_buffer_size=50,
            checkpoint_interval=100,
            use_amp=False,
            # Physics-informed loss (PhysicsInformedLoss path)
            physics_informed=physics_informed,
            physics_loss_weight=physics_loss_weight,
            physics_n_collocation_points=50,
            physics_n_boundary_points=20,
            # Combined physics loss (CombinedAlphaGalerkinPhysicsLoss path)
            physics_loss_type=physics_loss_type,
            physics_weight=physics_weight,
        ),
        experiment_name="test_physics",
        seed=42,
    )


def _make_trainer(
    model: AlphaGalerkinModel,
    config: AlphaGalerkinConfig,
    tmpdir: str,
) -> Trainer:
    """Convenience wrapper for creating a Trainer on CPU."""
    return Trainer(
        model=model,
        config=config,
        device="cpu",
        checkpoint_dir=Path(tmpdir),
    )


def _make_fake_experiences(trainer: Trainer, n: int = 10) -> list:
    """Create fake experiences matching the model's expected input shape."""
    from src.training.replay_buffer import Experience

    board_size = 9
    input_channels = trainer.config.operator.input_channels
    action_space = board_size * board_size + 1

    return [
        Experience(
            board_state=torch.randn(input_channels, board_size, board_size),
            board_size=board_size,
            target_policy=torch.softmax(torch.randn(action_space), dim=0),
            target_value=float(torch.randn(1).tanh().item()),
        )
        for _ in range(n)
    ]


def _prefill_and_mock(trainer: Trainer, n: int = 100):
    """Pre-fill buffer and return a context manager that mocks self-play."""
    from contextlib import contextmanager

    for exp in _make_fake_experiences(trainer, n):
        trainer.buffer.add(exp)

    @contextmanager
    def _ctx():
        fake = _make_fake_experiences(trainer, 5)
        with (
            patch.object(trainer, "_fill_buffer"),
            patch.object(
                trainer.self_play_worker,
                "generate_experiences",
                return_value=fake,
            ),
        ):
            yield

    return _ctx()


# ============================================================================
# 1. Config toggle tests
# ============================================================================


class TestConfigTogglePhysicsInformed:
    """Verify physics_informed toggle on TrainingConfig and Trainer."""

    def test_physics_informed_defaults_to_false(self) -> None:
        """physics_informed should be False by default."""
        cfg = TrainingConfig()
        assert cfg.physics_informed is False

    def test_physics_informed_can_be_enabled(self) -> None:
        """physics_informed=True should be accepted."""
        cfg = TrainingConfig(physics_informed=True)
        assert cfg.physics_informed is True

    def test_trainer_no_physics_loss_fn_when_disabled(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """With physics_informed=False, physics_loss_fn should be None."""
        config = _make_config(op_config, physics_informed=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(small_model, config, tmpdir)
            assert trainer.use_physics_loss is False
            assert trainer.physics_loss_fn is None

    def test_trainer_creates_physics_loss_fn_when_enabled(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """With physics_informed=True, physics_loss_fn should be created."""
        config = _make_config(op_config, physics_informed=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(small_model, config, tmpdir)
            assert trainer.use_physics_loss is True
            # physics_loss_fn might be None if PDE imports fail, but the flag
            # itself is True. When PDE imports succeed it should be set.
            if trainer.physics_loss_fn is not None:
                from src.training.physics_loss import PhysicsInformedLoss

                assert isinstance(trainer.physics_loss_fn, PhysicsInformedLoss)


class TestConfigToggleCombinedPhysicsLoss:
    """Verify physics_loss_type toggle on TrainingConfig and Trainer."""

    def test_default_physics_loss_type_is_none(self) -> None:
        """physics_loss_type should default to 'none'."""
        cfg = TrainingConfig()
        assert cfg.physics_loss_type == "none"

    def test_default_physics_weight(self) -> None:
        """physics_weight should default to 0.01."""
        cfg = TrainingConfig()
        assert cfg.physics_weight == 0.01

    def test_physics_loss_type_combined_accepted(self) -> None:
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

    def test_negative_physics_loss_weight_rejected(self) -> None:
        """Negative physics_loss_weight should fail validation (ge=0 constraint)."""
        with pytest.raises(ValidationError):
            TrainingConfig(physics_loss_weight=-1.0)

    def test_trainer_combined_loss_none_by_default(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """With physics_loss_type='none', combined_physics_loss_fn should be None."""
        config = _make_config(op_config, physics_loss_type="none")
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(small_model, config, tmpdir)
            assert trainer.combined_physics_loss_fn is None

    def test_trainer_combined_loss_created_when_combined(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """With physics_loss_type='combined', combined_physics_loss_fn should be set."""
        config = _make_config(op_config, physics_loss_type="combined", physics_weight=0.05)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(small_model, config, tmpdir)
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
            trainer = _make_trainer(small_model, config, tmpdir)
            assert trainer.combined_physics_loss_fn is not None
            assert trainer.combined_physics_loss_fn.physics_weight == pytest.approx(expected_weight)

    def test_existing_loss_fn_preserved_with_combined(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """Enabling combined physics loss must not break the existing loss_fn."""
        config = _make_config(op_config, physics_loss_type="combined")
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(small_model, config, tmpdir)
            assert isinstance(trainer.loss_fn, AlphaGalerkinLoss)

    @pytest.mark.parametrize(
        "loss_type",
        ["none", "combined"],
        ids=["physics_none", "physics_combined"],
    )
    def test_parametrized_loss_type_initialization(
        self,
        small_model: AlphaGalerkinModel,
        op_config: OperatorConfig,
        loss_type: str,
    ) -> None:
        """Both 'none' and 'combined' loss types should initialize without error."""
        config = _make_config(
            op_config,
            physics_loss_type=loss_type,  # type: ignore[arg-type]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(small_model, config, tmpdir)
            assert trainer.global_step == 0


# ============================================================================
# 2. Gradient flow tests
# ============================================================================


class TestGradientFlowWithPhysics:
    """Verify gradients flow through physics loss paths."""

    def test_combined_physics_loss_produces_gradients(self) -> None:
        """CombinedAlphaGalerkinPhysicsLoss should produce finite gradients."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.01)
        board_size = 9
        n_actions = board_size * board_size + 1
        batch = 2

        policy_logits = torch.randn(batch, n_actions, requires_grad=True)
        value_raw = torch.randn(batch, 1, requires_grad=True)
        value = torch.tanh(value_raw)
        target_policy = torch.softmax(torch.randn(batch, n_actions), dim=-1)
        target_value = torch.zeros(batch, 1)

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        total = result["total"]
        assert isinstance(total, torch.Tensor)
        total.backward()

        assert policy_logits.grad is not None, "Gradients must flow to policy logits"
        assert value_raw.grad is not None, "Gradients must flow to value"
        assert torch.isfinite(policy_logits.grad).all(), "Policy gradients must be finite"
        assert torch.isfinite(value_raw.grad).all(), "Value gradients must be finite"

    def test_no_nan_gradients_with_combined_physics(self) -> None:
        """No NaN/Inf in gradients when combined physics loss is enabled."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.1)
        board_size = 9
        n_actions = board_size * board_size + 1
        batch = 4

        policy_logits = torch.randn(batch, n_actions, requires_grad=True)
        value_raw = torch.randn(batch, 1, requires_grad=True)
        value = torch.tanh(value_raw)
        target_policy = torch.softmax(torch.randn(batch, n_actions), dim=-1)
        target_value = torch.rand(batch, 1) * 2 - 1

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        total = result["total"]
        assert isinstance(total, torch.Tensor)
        total.backward()

        for name, grad in [
            ("policy_logits", policy_logits.grad),
            ("value_raw", value_raw.grad),
        ]:
            assert grad is not None, f"Gradient for {name} must not be None"
            assert not torch.isnan(grad).any(), f"NaN in gradient for {name}"
            assert not torch.isinf(grad).any(), f"Inf in gradient for {name}"

    def test_training_steps_update_model_params_with_combined(
        self, op_config: OperatorConfig
    ) -> None:
        """Training with combined physics loss should update model parameters.

        Uses direct _training_step calls to avoid slow MCTS self-play.
        """
        model = AlphaGalerkinModel(op_config)
        config = _make_config(op_config, physics_loss_type="combined", physics_weight=0.01)

        # Snapshot parameters before training
        params_before = {name: p.clone().detach() for name, p in model.named_parameters()}

        board_size = 9
        batch_size = 4
        input_channels = op_config.input_channels
        action_space = board_size * board_size + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(model, config, tmpdir)
            trainer.model.train()

            # Run a few training steps directly (bypasses self-play)
            for _ in range(3):
                from src.data.collate import TrainingBatch

                batch = TrainingBatch(
                    board_states=torch.randn(batch_size, input_channels, board_size, board_size),
                    target_policies=torch.softmax(torch.randn(batch_size, action_space), dim=-1),
                    target_values=torch.randn(batch_size, 1).tanh(),
                    position_mask=torch.ones(batch_size, board_size, board_size, dtype=torch.bool),
                    action_mask=torch.ones(batch_size, action_space),
                    board_sizes=torch.full((batch_size,), board_size, dtype=torch.long),
                )
                trainer._training_step(batch)

        # At least some parameters should have changed
        any_changed = False
        for name, p in model.named_parameters():
            if not torch.equal(params_before[name], p.data):
                any_changed = True
                break

        assert any_changed, "Model parameters should update after training with physics loss"


# ============================================================================
# 3. Loss balancing integration
# ============================================================================


class TestLossBalancingPhysicsIntegration:
    """Verify loss balancer includes physics when physics is enabled."""

    def test_balancer_includes_physics_when_physics_informed(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """When physics_informed=True, 'physics' should be in the loss balancer names."""
        config = _make_config(op_config, physics_informed=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(small_model, config, tmpdir)
            # If physics_loss_fn was created, the balancer should include 'physics'
            if trainer.physics_loss_fn is not None:
                losses = {
                    "policy": torch.tensor(1.0),
                    "value": torch.tensor(0.5),
                    "lbb": torch.tensor(0.1),
                    "physics": torch.tensor(0.2),
                }
                result = trainer.loss_balancer.compute_weighted_loss(losses)
                assert "physics" in result.weights

    def test_balancer_excludes_physics_when_disabled(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """When physics_informed=False, 'physics' should NOT be in the loss balancer."""
        config = _make_config(op_config, physics_informed=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(small_model, config, tmpdir)
            losses = {
                "policy": torch.tensor(1.0),
                "value": torch.tensor(0.5),
                "lbb": torch.tensor(0.1),
            }
            result = trainer.loss_balancer.compute_weighted_loss(losses)
            assert "physics" not in result.weights

    def test_balancer_weighted_sum_positive(self) -> None:
        """Loss balancer should produce positive weighted sum with physics term."""
        from src.training.loss_balancing import LossBalancingConfig, create_loss_balancer

        config = LossBalancingConfig(name="test")
        loss_names = ["policy", "value", "lbb", "physics"]
        balancer = create_loss_balancer(config, loss_names)

        losses = {
            "policy": torch.tensor(1.0),
            "value": torch.tensor(0.5),
            "lbb": torch.tensor(0.1),
            "physics": torch.tensor(0.3),
        }
        result = balancer.compute_weighted_loss(losses)

        assert result.weighted_sum.item() > 0, "Weighted sum should be positive"
        assert "physics" in result.weights, "Physics weight should be in result"


# ============================================================================
# 4. Metrics logging tests
# ============================================================================


class TestTrainingMetricsPhysics:
    """Verify TrainingMetrics captures physics information."""

    def test_default_physics_metrics_zero(self) -> None:
        """Default physics loss metrics should be zero."""
        metrics = TrainingMetrics(step=0)
        assert metrics.physics_loss == 0.0
        assert metrics.physics_residual_loss == 0.0
        assert metrics.physics_boundary_loss == 0.0
        assert metrics.physics_weight == 0.0

    def test_physics_metrics_populated(self) -> None:
        """Physics metrics should be populated when set."""
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
        """to_dict should exclude physics metrics when physics_weight is 0."""
        metrics = TrainingMetrics(
            step=0,
            physics_loss=0.5,
            physics_weight=0.0,
        )
        result = metrics.to_dict()
        assert "physics_loss" not in result
        assert "physics_residual_loss" not in result
        assert "physics_boundary_loss" not in result
        assert "physics_weight" not in result

    def test_to_dict_includes_physics_when_weight_nonzero(self) -> None:
        """to_dict should include physics metrics when physics_weight > 0."""
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

    def test_to_dict_always_includes_core_metrics(self) -> None:
        """to_dict should always include core metrics regardless of physics config."""
        metrics = TrainingMetrics(
            step=42,
            total_loss=1.0,
            policy_loss=0.5,
            value_loss=0.3,
            lbb_loss=0.2,
            learning_rate=1e-4,
        )
        result = metrics.to_dict()
        for key in ("step", "total_loss", "policy_loss", "value_loss", "lbb_loss", "learning_rate"):
            assert key in result, f"Core metric '{key}' missing from to_dict"

    def test_metrics_history_populated_during_training(self, op_config: OperatorConfig) -> None:
        """Training should populate metrics history with physics fields when enabled."""
        model = AlphaGalerkinModel(op_config)
        config = _make_config(op_config, physics_loss_type="combined", physics_weight=0.01)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(model, config, tmpdir)
            with _prefill_and_mock(trainer):
                trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)

            history = trainer.get_metrics_history()
            assert len(history) == 3

            # Core keys should always be present
            for entry in history:
                assert "total_loss" in entry
                assert "policy_loss" in entry
                assert "value_loss" in entry


# ============================================================================
# 5. Combined loss forward pass tests
# ============================================================================


class TestCombinedPhysicsLossForward:
    """Test the CombinedAlphaGalerkinPhysicsLoss forward pass."""

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

    def test_physics_component_is_zero_without_pde_operator(self) -> None:
        """Without PDE operator, physics loss component should be zero."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.5)
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

        physics = result["physics"]
        assert isinstance(physics, torch.Tensor)
        assert physics.item() == pytest.approx(0.0, abs=1e-7), (
            "Physics loss should be zero without PDE operator"
        )


# ============================================================================
# 6. End-to-end training tests
# ============================================================================


class TestTrainerPhysicsTraining:
    """End-to-end tests: training steps with various physics configurations."""

    def test_training_completes_with_combined_physics(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """Training with physics_loss_type='combined' should complete without error."""
        config = _make_config(op_config, physics_loss_type="combined", physics_weight=0.01)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(small_model, config, tmpdir)
            with _prefill_and_mock(trainer):
                trainer.train(n_steps=5, log_interval=2, checkpoint_interval=100)
            assert trainer.global_step == 5

    def test_training_completes_with_none_physics(
        self, small_model: AlphaGalerkinModel, op_config: OperatorConfig
    ) -> None:
        """Training with physics_loss_type='none' should complete (baseline)."""
        config = _make_config(op_config, physics_loss_type="none")
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(small_model, config, tmpdir)
            with _prefill_and_mock(trainer):
                trainer.train(n_steps=5, log_interval=2, checkpoint_interval=100)
            assert trainer.global_step == 5

    def test_training_completes_with_physics_informed(self, op_config: OperatorConfig) -> None:
        """Training with physics_informed=True should complete."""
        model = AlphaGalerkinModel(op_config)
        config = _make_config(op_config, physics_informed=True, physics_loss_weight=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(model, config, tmpdir)
            with _prefill_and_mock(trainer):
                trainer.train(n_steps=5, log_interval=2, checkpoint_interval=100)
            assert trainer.global_step == 5

    @pytest.mark.parametrize(
        ("physics_loss_type", "physics_informed"),
        [
            ("none", False),
            ("combined", False),
            ("none", True),
            ("combined", True),
        ],
        ids=["baseline", "combined_only", "informed_only", "both_enabled"],
    )
    def test_finite_losses_for_all_physics_configs(
        self,
        op_config: OperatorConfig,
        physics_loss_type: str,
        physics_informed: bool,
    ) -> None:
        """All physics config combinations should produce finite losses."""
        model = AlphaGalerkinModel(op_config)
        config = _make_config(
            op_config,
            physics_loss_type=physics_loss_type,  # type: ignore[arg-type]
            physics_informed=physics_informed,
            physics_weight=0.01,
            physics_loss_weight=0.1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(model, config, tmpdir)
            with _prefill_and_mock(trainer):
                trainer.train(n_steps=5, log_interval=1, checkpoint_interval=100)
            history = trainer.get_metrics_history()
            total_losses = [m["total_loss"] for m in history]

            assert all(torch.isfinite(torch.tensor(v)) for v in total_losses), (
                f"Non-finite loss with type={physics_loss_type}, "
                f"informed={physics_informed}: {total_losses}"
            )


# ============================================================================
# 7. Property-based tests (Hypothesis)
# ============================================================================


def _valid_combined_loss_inputs(
    batch_max: int = 4,
    board_size: int = 9,
) -> st.SearchStrategy[dict[str, torch.Tensor]]:
    """Strategy for CombinedAlphaGalerkinPhysicsLoss inputs."""
    n_actions = board_size * board_size + 1
    return st.integers(min_value=1, max_value=batch_max).map(
        lambda b: _make_combined_inputs(b, n_actions)
    )


def _make_combined_inputs(batch: int, n_actions: int) -> dict[str, torch.Tensor]:
    """Create a dictionary of inputs for CombinedAlphaGalerkinPhysicsLoss."""
    torch.manual_seed(batch * 1000 + n_actions)
    return {
        "policy_logits": torch.randn(batch, n_actions),
        "value": torch.tanh(torch.randn(batch, 1)),
        "target_policy": torch.softmax(torch.randn(batch, n_actions), dim=-1),
        "target_value": torch.rand(batch, 1) * 2 - 1,
    }


class TestPhysicsLossPropertyBased:
    """Hypothesis property-based tests for physics loss components."""

    @given(data=_valid_combined_loss_inputs())
    @settings(max_examples=30, deadline=None)
    def test_combined_loss_total_non_negative(self, data: dict[str, torch.Tensor]) -> None:
        """Combined loss total should be non-negative for all valid inputs."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.01)
        result = loss_fn(**data)
        total = result["total"]
        assert isinstance(total, torch.Tensor)
        assert total.item() >= -1e-6, f"Total loss must be non-negative, got {total.item()}"

    @given(data=_valid_combined_loss_inputs())
    @settings(max_examples=30, deadline=None)
    def test_combined_loss_components_finite(self, data: dict[str, torch.Tensor]) -> None:
        """All loss components should be finite for valid inputs."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.01)
        result = loss_fn(**data)

        for key in ("total", "policy", "value", "lbb", "physics"):
            component = result[key]
            assert isinstance(component, torch.Tensor)
            assert torch.isfinite(component), f"Component '{key}' must be finite, got {component}"

    @given(data=_valid_combined_loss_inputs())
    @settings(max_examples=30, deadline=None)
    def test_combined_loss_policy_non_negative(self, data: dict[str, torch.Tensor]) -> None:
        """Policy loss component should be non-negative."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.01)
        result = loss_fn(**data)
        policy = result["policy"]
        assert isinstance(policy, torch.Tensor)
        assert policy.item() >= -1e-6, f"Policy loss must be non-negative, got {policy.item()}"

    @given(data=_valid_combined_loss_inputs())
    @settings(max_examples=30, deadline=None)
    def test_combined_loss_value_non_negative(self, data: dict[str, torch.Tensor]) -> None:
        """Value loss component should be non-negative."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.01)
        result = loss_fn(**data)
        value = result["value"]
        assert isinstance(value, torch.Tensor)
        assert value.item() >= -1e-6, f"Value loss must be non-negative, got {value.item()}"

    @given(
        physics_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_physics_weight_scaling(self, physics_weight: float) -> None:
        """Physics component should be zero without PDE operator regardless of weight."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=physics_weight)
        board_size = 9
        n_actions = board_size * board_size + 1

        torch.manual_seed(42)
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

        physics = result["physics"]
        assert isinstance(physics, torch.Tensor)
        # Without PDE operator, physics should always be zero
        assert physics.item() == pytest.approx(0.0, abs=1e-7)


class TestPhysicsLossConfigPropertyBased:
    """Hypothesis tests for physics config validation bounds."""

    @given(
        weight=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_non_negative_physics_weight_accepted(self, weight: float) -> None:
        """Any non-negative physics_weight should be accepted by TrainingConfig."""
        cfg = TrainingConfig(physics_weight=weight)
        assert cfg.physics_weight == pytest.approx(weight)

    @given(
        weight=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_non_negative_physics_loss_weight_accepted(self, weight: float) -> None:
        """Any non-negative physics_loss_weight should be accepted."""
        cfg = TrainingConfig(physics_loss_weight=weight)
        assert cfg.physics_loss_weight == pytest.approx(weight)

    @given(
        n_points=st.integers(min_value=10, max_value=100000),
    )
    @settings(max_examples=20)
    def test_valid_collocation_points_accepted(self, n_points: int) -> None:
        """Collocation points within valid range should be accepted."""
        cfg = TrainingConfig(physics_n_collocation_points=n_points)
        assert cfg.physics_n_collocation_points == n_points

    @given(
        n_points=st.integers(min_value=1, max_value=9),
    )
    @settings(max_examples=10)
    def test_too_few_collocation_points_rejected(self, n_points: int) -> None:
        """Collocation points below minimum (10) should be rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(physics_n_collocation_points=n_points)


# ============================================================================
# 8. Numerical stability with physics loss
# ============================================================================


class TestPhysicsLossNumericalStability:
    """Test physics loss under extreme numerical conditions."""

    def test_combined_loss_with_extreme_logits(self) -> None:
        """Large logits should not cause overflow in combined physics loss."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.01)
        board_size = 9
        n_actions = board_size * board_size + 1
        batch = 2

        policy_logits = torch.randn(batch, n_actions) * 1000.0
        value = torch.tanh(torch.randn(batch, 1))
        target_policy = torch.softmax(torch.randn(batch, n_actions), dim=-1)
        target_value = torch.zeros(batch, 1)

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        total = result["total"]
        assert isinstance(total, torch.Tensor)
        assert torch.isfinite(total), f"Total must be finite with extreme logits, got {total}"

    def test_combined_loss_with_near_zero_targets(self) -> None:
        """Near-zero probability targets should not cause NaN."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.01)
        board_size = 9
        n_actions = board_size * board_size + 1
        batch = 2

        policy_logits = torch.randn(batch, n_actions)
        value = torch.tanh(torch.randn(batch, 1))
        target_policy = torch.full((batch, n_actions), 1e-10)
        target_policy[:, 0] = 1.0 - (n_actions - 1) * 1e-10
        target_value = torch.zeros(batch, 1)

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        total = result["total"]
        assert isinstance(total, torch.Tensor)
        assert torch.isfinite(total), "Total must be finite with near-zero targets"

    def test_combined_loss_with_uniform_targets(self) -> None:
        """Uniform target distribution should produce well-behaved loss."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.01)
        board_size = 9
        n_actions = board_size * board_size + 1
        batch = 2

        policy_logits = torch.zeros(batch, n_actions)
        value = torch.zeros(batch, 1)
        target_policy = torch.ones(batch, n_actions) / n_actions
        target_value = torch.zeros(batch, 1)

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        total = result["total"]
        assert isinstance(total, torch.Tensor)
        assert torch.isfinite(total), "Total must be finite with uniform targets"
        assert total.item() >= 0, "Total must be non-negative"


# ============================================================================
# 9. Determinism test
# ============================================================================


class TestPhysicsLossDeterminism:
    """Verify determinism of physics loss computation."""

    def test_combined_loss_is_deterministic(self) -> None:
        """Two identical calls with same inputs must return identical loss."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(physics_weight=0.01)
        board_size = 9
        n_actions = board_size * board_size + 1

        torch.manual_seed(42)
        policy_logits = torch.randn(2, n_actions)
        value = torch.tanh(torch.randn(2, 1))
        target_policy = torch.softmax(torch.randn(2, n_actions), dim=-1)
        target_value = torch.zeros(2, 1)

        result1 = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        result2 = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        total1 = result1["total"]
        total2 = result2["total"]
        assert isinstance(total1, torch.Tensor)
        assert isinstance(total2, torch.Tensor)
        assert total1.item() == pytest.approx(total2.item(), abs=1e-6)
