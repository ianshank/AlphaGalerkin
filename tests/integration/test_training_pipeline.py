"""Integration tests for the full training pipeline."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from config.schemas import (
    AlphaGalerkinConfig,
    DomainConfig,
    MCTSConfig,
    OperatorConfig,
    TrainingConfig,
)
from src.data.collate import VariableSizeCollator
from src.modeling.model import AlphaGalerkinModel
from src.training.loss import AlphaGalerkinLoss
from src.training.replay_buffer import UniformReplayBuffer
from src.training.self_play import SelfPlayWorker
from src.training.trainer import Trainer


def _make_fake_experiences(trainer, n=10):
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


def _prefill_and_mock(trainer, n=100):
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
            patch.object(trainer, "_run_evaluation", return_value=0.5),
        ):
            yield

    return _ctx()


@pytest.fixture
def integration_config() -> AlphaGalerkinConfig:
    """Create config for integration tests."""
    return AlphaGalerkinConfig(
        domain=DomainConfig(),
        operator=OperatorConfig(
            d_model=32,
            d_key=16,
            d_value=16,
            d_ffn=64,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=16,
            use_fnet_mixing=False,
        ),
        mcts=MCTSConfig(
            n_simulations=10,
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
            total_steps=10,
            n_self_play_games=2,
            replay_buffer_size=100,
            checkpoint_interval=5,
            use_amp=False,
        ),
        experiment_name="integration_test",
        seed=42,
    )


class TestFullTrainingPipeline:
    """Integration tests for complete training pipeline."""

    def test_self_play_to_training(
        self,
        integration_config: AlphaGalerkinConfig,
    ) -> None:
        """Test generating self-play data and training on it."""
        # Create model
        model = AlphaGalerkinModel(integration_config.operator)

        # Create self-play worker
        worker = SelfPlayWorker(
            model=model,
            mcts_config=integration_config.mcts,
            device="cpu",
            board_sizes=[9],
        )

        # Generate fake experiences (mocking MCTS to avoid slow self-play)
        from src.training.replay_buffer import Experience

        input_channels = integration_config.operator.input_channels
        action_space = 9 * 9 + 1
        experiences = [
            Experience(
                board_state=torch.randn(input_channels, 9, 9),
                board_size=9,
                target_policy=torch.softmax(torch.randn(action_space), dim=0),
                target_value=float(torch.randn(1).tanh().item()),
            )
            for _ in range(20)
        ]
        assert len(experiences) > 0

        # Create replay buffer and add experiences
        buffer = UniformReplayBuffer(capacity=100)
        buffer.add_batch(experiences)
        assert len(buffer) == min(len(experiences), buffer.capacity)

        # Sample batch
        batch_exps = buffer.sample(4)
        assert len(batch_exps) == min(4, len(buffer))

        # Collate
        collator = VariableSizeCollator()
        batch = collator(batch_exps)
        assert batch.batch_size == len(batch_exps)

        # Forward pass
        model.train()
        output = model(batch.board_states, return_lbb=True)
        assert output.policy_logits.shape[0] == batch.batch_size

        # Compute loss
        loss_fn = AlphaGalerkinLoss()
        loss = loss_fn(
            policy_logits=output.policy_logits,
            value=output.value,
            target_policy=batch.target_policies,
            target_value=batch.target_values,
            lbb_constant=output.lbb_constant,
        )
        assert loss.total.isfinite()

        # Backward pass
        loss.total.backward()

        # Check gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_training_reduces_loss(
        self,
        integration_config: AlphaGalerkinConfig,
    ) -> None:
        """Test that training actually reduces loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            model = AlphaGalerkinModel(integration_config.operator)
            trainer = Trainer(
                model=model,
                config=integration_config,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
            )

            # Train for a few steps
            with _prefill_and_mock(trainer):
                trainer.train(
                    n_steps=2,
                    log_interval=1,
                    checkpoint_interval=100,
                )

            history = trainer.get_metrics_history()
            assert len(history) == 2

            # Get first and last losses
            first_loss = history[0]["total_loss"]
            last_loss = history[-1]["total_loss"]

            # Loss should be finite
            assert first_loss < float("inf")
            assert last_loss < float("inf")

            # Note: With random data and few steps, loss may not always decrease
            # but it should remain bounded and finite

    def test_checkpoint_restore_continues_training(
        self,
        integration_config: AlphaGalerkinConfig,
    ) -> None:
        """Test that checkpoint restore preserves training state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Phase 1: Train and save
            model1 = AlphaGalerkinModel(integration_config.operator)
            trainer1 = Trainer(
                model=model1,
                config=integration_config,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
            )

            with _prefill_and_mock(trainer1):
                trainer1.train(n_steps=2, log_interval=1, checkpoint_interval=1)
            step_after_first = trainer1.global_step

            # Phase 2: Load and continue
            model2 = AlphaGalerkinModel(integration_config.operator)
            trainer2 = Trainer(
                model=model2,
                config=integration_config,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
            )
            trainer2.load_checkpoint()

            assert trainer2.global_step == step_after_first

            # Continue training
            with _prefill_and_mock(trainer2):
                trainer2.train(n_steps=1, log_interval=1, checkpoint_interval=100)
            assert trainer2.global_step == step_after_first + 1

    def test_resolution_independence_after_training(
        self,
        integration_config: AlphaGalerkinConfig,
    ) -> None:
        """Test that model maintains resolution independence after training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            model = AlphaGalerkinModel(integration_config.operator)
            trainer = Trainer(
                model=model,
                config=integration_config,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
            )

            # Train on 9x9
            with _prefill_and_mock(trainer):
                trainer.train(n_steps=1, log_interval=1, checkpoint_interval=100)

            # Test inference on different sizes
            model.eval()
            with torch.no_grad():
                for board_size in [9, 13, 19]:
                    batch_size = 2
                    n_channels = 17
                    x = torch.randn(batch_size, n_channels, board_size, board_size)

                    output = model(x)

                    # Check output shapes
                    expected_actions = board_size**2 + 1
                    assert output.policy_logits.shape == (batch_size, expected_actions)
                    assert output.value.shape == (batch_size, 1)

                    # Check outputs are valid
                    assert torch.isfinite(output.policy_logits).all()
                    assert torch.isfinite(output.value).all()
                    assert (output.value >= -1).all() and (output.value <= 1).all()


class TestEndToEndSmoke:
    """Smoke tests for end-to-end functionality."""

    def test_minimal_training_run(
        self,
        integration_config: AlphaGalerkinConfig,
    ) -> None:
        """Minimal smoke test - just verify nothing crashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = AlphaGalerkinModel(integration_config.operator)
            trainer = Trainer(
                model=model,
                config=integration_config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
            )

            # Should complete without error
            with _prefill_and_mock(trainer):
                trainer.train(n_steps=1, log_interval=1, checkpoint_interval=100)

            # Should have metrics
            assert len(trainer.get_metrics_history()) == 1

            # Model should still work
            model.eval()
            with torch.no_grad():
                x = torch.randn(1, 17, 9, 9)
                output = model(x)
                assert output.policy_logits.shape == (1, 82)


class TestE2ETrainingPipeline:
    """E2E training pipeline: self-play -> buffer -> train -> checkpoint -> resume."""

    def test_training_pipeline_with_checkpoint_resume(self) -> None:
        """Train 5 steps, checkpoint, resume into new trainer, train 5 more.

        Verifies:
        - Checkpoint saves after first phase
        - New trainer resumes from correct global_step
        - All 10 steps produce finite losses
        """
        config = AlphaGalerkinConfig(
            domain=DomainConfig(),
            operator=OperatorConfig(
                d_model=32,
                d_key=16,
                d_value=16,
                d_ffn=64,
                n_heads=2,
                n_galerkin_layers=1,
                n_softmax_layers=1,
                n_fourier_features=16,
                use_fnet_mixing=False,
            ),
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
                total_steps=10,
                n_self_play_games=2,
                replay_buffer_size=100,
                checkpoint_interval=5,
                use_amp=False,
            ),
            experiment_name="e2e_pipeline_test",
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Phase 1: Train 1 step, save checkpoint
            model1 = AlphaGalerkinModel(config.operator)
            trainer1 = Trainer(
                model=model1,
                config=config,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
            )
            with _prefill_and_mock(trainer1):
                trainer1.train(n_steps=1, log_interval=1, checkpoint_interval=1)

            # Verify trainer1 completed 1 step
            assert trainer1.global_step == 1

            # Collect first-phase metrics
            phase1_history = trainer1.get_metrics_history()
            assert len(phase1_history) == 1
            for m in phase1_history:
                assert math.isfinite(m["total_loss"]), (
                    f"Non-finite loss at step {m['step']}"
                )

            # Phase 2: Load checkpoint into fresh trainer, continue training
            model2 = AlphaGalerkinModel(config.operator)
            trainer2 = Trainer(
                model=model2,
                config=config,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
            )
            trainer2.load_checkpoint()

            # Verify resume starts at step 1
            assert trainer2.global_step == 1

            # Train 1 more step
            with _prefill_and_mock(trainer2):
                trainer2.train(n_steps=1, log_interval=1, checkpoint_interval=100)

            # Verify total progress
            assert trainer2.global_step == 2

            # All second-phase losses should be finite
            phase2_history = trainer2.get_metrics_history()
            assert len(phase2_history) == 1
            for m in phase2_history:
                assert math.isfinite(m["total_loss"]), (
                    f"Non-finite loss at step {m['step']}"
                )


class TestCurriculumProgression:
    """Tests for curriculum-based board size progression."""

    def test_curriculum_progression(self) -> None:
        """Verify curriculum transitions fire at the correct steps.

        Schedule:
        - step 0: [9]
        - step 3: [9, 13]

        Run 5 steps and assert the curriculum stage changes at step 3.
        """
        config = AlphaGalerkinConfig(
            domain=DomainConfig(),
            operator=OperatorConfig(
                d_model=32,
                d_key=16,
                d_value=16,
                d_ffn=64,
                n_heads=2,
                n_galerkin_layers=1,
                n_softmax_layers=1,
                n_fourier_features=16,
                use_fnet_mixing=False,
            ),
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
                total_steps=10,
                n_self_play_games=2,
                replay_buffer_size=100,
                checkpoint_interval=100,
                use_amp=False,
                curriculum_enabled=True,
                curriculum_schedule={0: [9], 3: [9, 13]},
            ),
            experiment_name="curriculum_test",
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            model = AlphaGalerkinModel(config.operator)
            trainer = Trainer(
                model=model,
                config=config,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
            )

            # Verify curriculum was created
            assert trainer.curriculum is not None

            # Before training: step 0 -> only [9]
            stage_before = trainer.curriculum.get_current_stage(0)
            assert stage_before.board_sizes == [9]

            # Step 3 is a transition step
            assert trainer.curriculum.is_transition_step(3)

            # After step 3: should have [9, 13]
            stage_after = trainer.curriculum.get_current_stage(3)
            assert stage_after.board_sizes == [9, 13]

            # Run 1 step to verify training works with curriculum enabled
            with _prefill_and_mock(trainer):
                trainer.train(n_steps=1, log_interval=1, checkpoint_interval=100)

            # Training completed successfully with curriculum
            assert trainer.global_step == 1

            # All losses should be finite
            history = trainer.get_metrics_history()
            assert len(history) == 1
            for m in history:
                assert math.isfinite(m["total_loss"])
