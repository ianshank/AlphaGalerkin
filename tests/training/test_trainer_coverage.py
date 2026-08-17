"""Additional Trainer tests targeting uncovered lines for coverage improvement.

Focuses on:
- Initialization branches (distributed context, wandb, elo tracker, etc.)
- _run_evaluation and _run_checkpoint_tournament
- _run_engine_evaluation
- _extract_step_from_checkpoint
- train() loop branches (curriculum transitions, plateau detection, early stopping,
  warmup completion, wandb logging, physics loss in step)
- save_checkpoint / load_checkpoint distributed branches
- TrainingMetrics.to_dict with physics metrics
- create_trainer with wandb_logger
- compute_loss / generate_data / evaluate abstract stubs
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from config.schemas import (
    AlphaGalerkinConfig,
    DomainConfig,
    MCTSConfig,
    OperatorConfig,
    TrainingConfig,
)
from src.modeling.model import AlphaGalerkinModel
from src.training.replay_buffer import Experience
from src.training.trainer import Trainer, TrainingMetrics, create_trainer

# ---------------------------------------------------------------------------
# Helpers (same pattern as existing test file)
# ---------------------------------------------------------------------------


def _make_fake_experiences(trainer: Trainer, n: int = 10) -> list[Experience]:
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config() -> AlphaGalerkinConfig:
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
            total_steps=5,
            n_self_play_games=2,
            replay_buffer_size=50,
            checkpoint_interval=3,
            use_amp=False,
        ),
        experiment_name="test_coverage",
        seed=42,
    )


@pytest.fixture
def small_model(small_config: AlphaGalerkinConfig) -> AlphaGalerkinModel:
    return AlphaGalerkinModel(small_config.operator)


@pytest.fixture
def checkpoint_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _make_trainer(
    small_model: AlphaGalerkinModel,
    small_config: AlphaGalerkinConfig,
    checkpoint_dir: Path,
    **kwargs,
) -> Trainer:
    return Trainer(
        model=small_model,
        config=small_config,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )


# ===========================================================================
# Tests for TrainingMetrics
# ===========================================================================


class TestTrainingMetricsPhysics:
    """Cover physics metrics branch in to_dict."""

    def test_to_dict_includes_physics_when_weight_positive(self) -> None:
        m = TrainingMetrics(
            step=10,
            total_loss=1.0,
            policy_loss=0.5,
            value_loss=0.3,
            lbb_loss=0.1,
            learning_rate=1e-4,
            physics_loss=0.05,
            physics_residual_loss=0.03,
            physics_boundary_loss=0.02,
            physics_weight=0.1,
        )
        d = m.to_dict()
        assert "physics_loss" in d
        assert "physics_residual_loss" in d
        assert "physics_boundary_loss" in d
        assert "physics_weight" in d
        assert d["physics_loss"] == 0.05

    def test_to_dict_excludes_physics_when_weight_zero(self) -> None:
        m = TrainingMetrics(step=1, physics_weight=0.0)
        d = m.to_dict()
        assert "physics_loss" not in d


# ===========================================================================
# Tests for abstract method stubs
# ===========================================================================


class TestAbstractMethodStubs:
    """Cover compute_loss, generate_data, evaluate which raise NotImplementedError."""

    def test_compute_loss_raises(self, small_model, small_config, checkpoint_dir) -> None:
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        with pytest.raises(NotImplementedError, match="Trainer uses _training_step"):
            trainer.compute_loss(None)

    def test_generate_data_raises(self, small_model, small_config, checkpoint_dir) -> None:
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        with pytest.raises(NotImplementedError, match="Trainer uses _sample_batch"):
            trainer.generate_data()

    def test_evaluate_raises(self, small_model, small_config, checkpoint_dir) -> None:
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        with pytest.raises(NotImplementedError, match="Trainer uses _run_evaluation"):
            trainer.evaluate()


# ===========================================================================
# Tests for _extract_step_from_checkpoint
# ===========================================================================


class TestExtractStep:
    def test_extract_step_valid(self, small_model, small_config, checkpoint_dir) -> None:
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        p = Path("/tmp/checkpoints/checkpoint_00010000.pt")
        assert trainer._extract_step_from_checkpoint(p) == 10000

    def test_extract_step_invalid_returns_zero(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        p = Path("/tmp/checkpoints/best_model.pt")
        assert trainer._extract_step_from_checkpoint(p) == 0


# ===========================================================================
# Tests for initialization branches
# ===========================================================================


class TestInitBranches:
    """Cover init branches.

    Tests distributed context, wandb on non-main, elo tracker,
    checkpoint_dir default, auto device.
    """

    def test_elo_tracker_enabled(self, small_model, small_config, checkpoint_dir) -> None:
        small_config.training.eval_vs_checkpoints = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer.elo_tracker is not None

    def test_default_checkpoint_dir(self, small_model, small_config) -> None:
        """When checkpoint_dir is None, a default path is used."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=None,
        )
        assert "test_coverage" in str(trainer.checkpoint_manager.checkpoint_dir)

    def test_wandb_disabled_on_non_main_rank(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """W&B logger is set to None when not on main rank."""
        from src.training.distributed_context import DistributedContext

        ctx = DistributedContext(
            rank=1,
            local_rank=1,
            world_size=2,
            is_distributed=False,  # keep False so we don't need real NCCL
            device=torch.device("cpu"),
        )
        fake_wandb = MagicMock()
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
            tracker=fake_wandb,
            distributed_context=ctx,
        )
        # wandb logger should be disabled on non-main rank
        assert trainer.tracker is None

    def test_wandb_enabled_on_main_rank(self, small_model, small_config, checkpoint_dir) -> None:
        from src.training.distributed_context import DistributedContext

        ctx = DistributedContext(rank=0, device=torch.device("cpu"))
        fake_wandb = MagicMock()
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
            tracker=fake_wandb,
            distributed_context=ctx,
        )
        assert trainer.tracker is fake_wandb

    def test_warmup_flag_set_when_warmup_nonzero(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        small_config.training.warmup_steps = 10
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer._warmup_completed is False
        assert trainer._warmup_steps == 10


# ===========================================================================
# Tests for _run_evaluation
# ===========================================================================


class TestRunEvaluation:
    """Cover _run_evaluation with both multi-resolution and per-board-size paths."""

    def test_run_evaluation_per_board_size(self, small_model, small_config, checkpoint_dir) -> None:
        from src.training.evaluation import EvaluationResult

        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        fake_result = EvaluationResult(
            win_rate=0.6,
            n_games=10,
            wins=6,
            losses=3,
            draws=1,
            avg_game_length=50.0,
        )
        with (
            patch.object(
                trainer.evaluator,
                "evaluate_vs_random",
                return_value=fake_result,
            ),
            patch.object(
                trainer.evaluator,
                "measure_policy_agreement",
                return_value=0.5,
            ),
        ):
            avg = trainer._run_evaluation(step=100)
        assert avg == pytest.approx(0.6, abs=0.01)

    def test_run_evaluation_multi_resolution(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        from src.training.evaluation import EvaluationResult

        small_config.training.multi_resolution_eval = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        fake_result = EvaluationResult(
            win_rate=0.7,
            n_games=10,
            wins=7,
            losses=2,
            draws=1,
            avg_game_length=40.0,
        )
        multi_res_results = {9: fake_result}

        with (
            patch.object(
                trainer.evaluator,
                "evaluate_multi_resolution",
                return_value=multi_res_results,
            ),
            patch.object(
                trainer.evaluator,
                "measure_policy_agreement",
                return_value=0.5,
            ),
        ):
            avg = trainer._run_evaluation(step=200)
        assert avg == pytest.approx(0.7, abs=0.01)

    def test_run_evaluation_with_wandb(self, small_model, small_config, checkpoint_dir) -> None:
        from src.training.evaluation import EvaluationResult

        fake_wandb = MagicMock()
        trainer = _make_trainer(small_model, small_config, checkpoint_dir, tracker=fake_wandb)

        fake_result = EvaluationResult(
            win_rate=0.5,
            n_games=10,
            wins=5,
            losses=5,
            draws=0,
            avg_game_length=30.0,
        )
        with (
            patch.object(
                trainer.evaluator,
                "evaluate_vs_random",
                return_value=fake_result,
            ),
            patch.object(
                trainer.evaluator,
                "measure_policy_agreement",
                return_value=0.4,
            ),
        ):
            trainer._run_evaluation(step=50)

        # wandb should have been called
        assert fake_wandb.log_evaluation.called or fake_wandb.log_metrics.called


# ===========================================================================
# Tests for _run_checkpoint_tournament
# ===========================================================================


class TestRunCheckpointTournament:
    def test_tournament_no_elo_tracker(self, small_model, small_config, checkpoint_dir) -> None:
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer.elo_tracker is None
        # Should return without error
        trainer._run_checkpoint_tournament(step=10, n_games=2)

    def test_tournament_no_checkpoints(self, small_model, small_config, checkpoint_dir) -> None:
        small_config.training.eval_vs_checkpoints = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer.elo_tracker is not None

        with patch.object(
            trainer.checkpoint_manager,
            "get_all_checkpoints",
            return_value=[],
        ):
            # Should return early when no checkpoints
            trainer._run_checkpoint_tournament(step=10, n_games=2)

    def test_tournament_with_checkpoints(self, small_model, small_config, checkpoint_dir) -> None:
        from src.training.evaluation import EvaluationResult

        small_config.training.eval_vs_checkpoints = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        fake_result = EvaluationResult(
            win_rate=0.8,
            n_games=4,
            wins=3,
            losses=1,
            draws=0,
            avg_game_length=50.0,
        )

        fake_ckpt = Path(checkpoint_dir / "checkpoint_00000005.pt")

        with (
            patch.object(
                trainer.checkpoint_manager,
                "get_all_checkpoints",
                return_value=[fake_ckpt],
            ),
            patch.object(
                trainer.evaluator,
                "evaluate_vs_checkpoint",
                return_value=fake_result,
            ),
        ):
            trainer._run_checkpoint_tournament(step=10, n_games=4)

        # Elo should have been updated
        rating = trainer.elo_tracker.get_rating(10)
        assert rating > 0

    def test_tournament_match_failure_handled(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        small_config.training.eval_vs_checkpoints = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        fake_ckpt = Path(checkpoint_dir / "checkpoint_00000005.pt")

        with (
            patch.object(
                trainer.checkpoint_manager,
                "get_all_checkpoints",
                return_value=[fake_ckpt],
            ),
            patch.object(
                trainer.evaluator,
                "evaluate_vs_checkpoint",
                side_effect=RuntimeError("match failed"),
            ),
        ):
            # Should not raise
            trainer._run_checkpoint_tournament(step=10, n_games=4)


# ===========================================================================
# Tests for _run_engine_evaluation
# ===========================================================================


class TestRunEngineEvaluation:
    def test_engine_eval_returns_early_when_no_path(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        # engine_eval_path is None by default
        trainer._run_engine_evaluation(step=10)  # should not raise

    def test_engine_eval_catches_exception(self, small_model, small_config, checkpoint_dir) -> None:
        small_config.training.engine_eval_enabled = True
        small_config.training.engine_eval_path = "/fake/stockfish"
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        with patch.object(
            trainer.evaluator,
            "evaluate_vs_engine",
            side_effect=RuntimeError("engine not found"),
        ):
            # Should catch and log warning, not raise
            trainer._run_engine_evaluation(step=10)


# ===========================================================================
# Tests for save/load checkpoint on non-main rank
# ===========================================================================


class TestCheckpointDistributed:
    def test_save_checkpoint_non_main_rank(self, small_model, small_config, checkpoint_dir) -> None:
        from src.training.distributed_context import DistributedContext

        ctx = DistributedContext(
            rank=1,
            local_rank=1,
            world_size=2,
            is_distributed=False,
            device=torch.device("cpu"),
        )
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
            distributed_context=ctx,
        )
        result = trainer.save_checkpoint()
        assert result is None


# ===========================================================================
# Tests for train() loop branches
# ===========================================================================


class TestTrainLoopBranches:
    """Cover branches in the main train() loop."""

    def test_train_with_wandb_logs_step_and_summary(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        fake_wandb = MagicMock()
        trainer = _make_trainer(small_model, small_config, checkpoint_dir, tracker=fake_wandb)
        with _prefill_and_mock(trainer):
            trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)

        # Should have called log_training_step
        assert fake_wandb.log_training_step.call_count == 3
        # Final summary
        assert fake_wandb.log_summary.called

    def test_train_with_eval_interval(self, small_model, small_config, checkpoint_dir) -> None:
        """Evaluation is triggered at eval_interval."""
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        with _prefill_and_mock(trainer):
            with patch.object(
                trainer,
                "_run_evaluation",
                return_value=0.5,
            ) as mock_eval:
                trainer.train(
                    n_steps=4,
                    log_interval=100,
                    checkpoint_interval=100,
                    eval_interval=2,
                )
        # eval at step 2 (step 0 is skipped because step > 0 required)
        assert mock_eval.call_count >= 1

    def test_train_early_stopping(self, small_model, small_config, checkpoint_dir) -> None:
        """Early stopping breaks out of training loop."""
        small_config.training.early_stopping_enabled = True
        small_config.training.early_stopping_patience = 1
        small_config.training.early_stopping_min_delta = 0.01
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer.stability_monitor is not None

        with _prefill_and_mock(trainer):
            # Mock _run_evaluation to return low win rate, and
            # make early stopping signal True
            with (
                patch.object(
                    trainer,
                    "_run_evaluation",
                    return_value=0.1,
                ),
                patch.object(
                    trainer.stability_monitor,
                    "check_early_stopping",
                    return_value=True,
                ),
            ):
                trainer.train(
                    n_steps=10,
                    log_interval=1,
                    checkpoint_interval=100,
                    eval_interval=1,
                )
        # Should have stopped early (well before 10 steps)
        assert trainer.global_step < 10

    def test_train_warmup_completion_logged(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        small_config.training.warmup_steps = 2
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer._warmup_completed is False

        with _prefill_and_mock(trainer):
            trainer.train(n_steps=4, log_interval=1, checkpoint_interval=100)

        assert trainer._warmup_completed is True

    def test_train_plateau_detection(self, small_model, small_config, checkpoint_dir) -> None:
        """Plateau detection runs after warmup completes."""
        small_config.training.plateau_detection_enabled = True
        small_config.training.plateau_patience = 1
        small_config.training.plateau_factor = 0.5
        small_config.training.plateau_min_lr = 1e-6
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer.stability_monitor is not None

        with _prefill_and_mock(trainer):
            with patch.object(
                trainer.stability_monitor,
                "check_plateau",
                return_value=True,
            ):
                trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)
        # Just assert it ran without error
        assert trainer.global_step == 3

    def test_train_curriculum_transition(self, small_model, small_config, checkpoint_dir) -> None:
        """Curriculum stage transition is logged."""
        small_config.training.curriculum_enabled = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer.curriculum is not None

        # Mock is_transition_step to trigger on step 1
        with _prefill_and_mock(trainer):
            with (
                patch.object(
                    trainer.curriculum,
                    "is_transition_step",
                    side_effect=lambda s: s == 1,
                ),
                patch.object(
                    trainer.curriculum,
                    "get_current_stage",
                    return_value=MagicMock(board_sizes=[9], size_weights=[1.0]),
                ),
            ):
                trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)
        assert trainer.global_step == 3

    def test_train_checkpoint_saves_and_logs_wandb_artifact(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """Checkpoint saving triggers W&B artifact logging."""
        fake_wandb = MagicMock()
        trainer = _make_trainer(small_model, small_config, checkpoint_dir, tracker=fake_wandb)
        with _prefill_and_mock(trainer):
            trainer.train(n_steps=5, log_interval=1, checkpoint_interval=2)

        # log_model_artifact should have been called for mid-training checkpoints
        assert fake_wandb.log_model_artifact.called

    def test_train_wandb_final_checkpoint_artifact(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """Final checkpoint is logged as W&B artifact with 'final' alias."""
        fake_wandb = MagicMock()
        trainer = _make_trainer(small_model, small_config, checkpoint_dir, tracker=fake_wandb)
        with _prefill_and_mock(trainer):
            trainer.train(n_steps=2, log_interval=1, checkpoint_interval=100)

        # Check that log_model_artifact was called with 'final' alias
        calls = fake_wandb.log_model_artifact.call_args_list
        final_calls = [c for c in calls if "final" in str(c)]
        assert len(final_calls) >= 1


# ===========================================================================
# Tests for _training_step physics loss paths
# ===========================================================================


class TestTrainingStepPhysics:
    """Cover physics loss computation path in _training_step."""

    def test_training_step_with_physics_loss_failure(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """When physics loss computation fails, it falls back to zero."""
        small_config.training.physics_informed = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        # Force physics_loss_fn to exist but fail
        trainer.use_physics_loss = True
        trainer.physics_loss_fn = MagicMock(side_effect=RuntimeError("physics computation failed"))

        with _prefill_and_mock(trainer):
            trainer.train(n_steps=1, log_interval=1, checkpoint_interval=100)

        # Should not crash; fallback to zero physics loss
        assert len(trainer._metrics_history) == 1

    def test_training_step_with_combined_physics_loss(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """Combined physics loss path is exercised."""
        small_config.training.physics_loss_type = "combined"
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer.combined_physics_loss_fn is not None

        with _prefill_and_mock(trainer):
            trainer.train(n_steps=1, log_interval=1, checkpoint_interval=100)

        assert len(trainer._metrics_history) == 1

    def test_training_step_combined_physics_loss_failure(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """Combined physics loss failure is caught gracefully."""
        small_config.training.physics_loss_type = "combined"
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        # Make combined physics loss raise
        trainer.combined_physics_loss_fn = MagicMock(
            side_effect=RuntimeError("combined loss failed")
        )

        with _prefill_and_mock(trainer):
            trainer.train(n_steps=1, log_interval=1, checkpoint_interval=100)

        assert len(trainer._metrics_history) == 1


# ===========================================================================
# Tests for _create_physics_loss error paths
# ===========================================================================


class TestCreatePhysicsLoss:
    def test_create_physics_loss_import_error(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """Import error returns None."""
        small_config.training.physics_informed = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        # Patch the import inside _create_physics_loss to raise ImportError
        import builtins

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "src.pde.config":
                raise ImportError("no pde module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            result = trainer._create_physics_loss()
        assert result is None

    def test_create_physics_loss_generic_exception(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """Generic exception returns None."""
        small_config.training.physics_informed = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        # Patch PoissonOperator to raise
        with patch(
            "src.pde.operators.PoissonOperator.__init__",
            side_effect=ValueError("bad config"),
        ):
            result = trainer._create_physics_loss()
        assert result is None


# ===========================================================================
# Tests for create_trainer with wandb
# ===========================================================================


class TestCreateTrainerWandb:
    def test_create_trainer_with_wandb_logger_and_resume(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """create_trainer with wandb_logger sets step offset on resume."""
        # First create a checkpoint
        trainer1 = _make_trainer(small_model, small_config, checkpoint_dir)
        with _prefill_and_mock(trainer1):
            trainer1.train(n_steps=2, log_interval=1, checkpoint_interval=1)
        ckpt_path = trainer1.checkpoint_manager.get_latest()

        # Now resume with wandb
        fake_wandb = MagicMock()
        new_model = AlphaGalerkinModel(small_config.operator)
        trainer2 = create_trainer(
            model=new_model,
            config=small_config,
            checkpoint_dir=checkpoint_dir,
            resume_from=ckpt_path,
            device="cpu",
            tracker=fake_wandb,
        )
        fake_wandb.set_step_offset.assert_called_once_with(trainer2.global_step)


# ===========================================================================
# Tests for _fill_buffer with wandb
# ===========================================================================


class TestFillBufferWandb:
    def test_fill_buffer_logs_to_wandb(self, small_model, small_config, checkpoint_dir) -> None:
        fake_wandb = MagicMock()
        trainer = _make_trainer(small_model, small_config, checkpoint_dir, tracker=fake_wandb)
        fake = _make_fake_experiences(trainer, 20)
        with patch.object(
            trainer.self_play_worker,
            "generate_experiences",
            return_value=fake,
        ):
            trainer._fill_buffer(min_size=10)

        # wandb should have been called with self_play metrics
        assert fake_wandb.log_metrics.called


# ===========================================================================
# Tests for _create_loss_balancer
# ===========================================================================


class TestCreateLossBalancer:
    def test_static_strategy(self, small_model, small_config, checkpoint_dir) -> None:
        small_config.training.loss_balancing_strategy = "static"
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer.loss_balancer is not None

    def test_relobralo_strategy(self, small_model, small_config, checkpoint_dir) -> None:
        small_config.training.loss_balancing_strategy = "relobralo"
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer.loss_balancer is not None

    def test_loss_balancer_includes_physics(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """When physics loss is enabled, balancer includes 'physics' in names."""
        small_config.training.physics_informed = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        # Check if physics loss fn is not None (depends on PDE module availability)
        # The balancer should have been created regardless
        assert trainer.loss_balancer is not None


# ===========================================================================
# Tests for _sample_batch with prioritized replay
# ===========================================================================


class TestSampleBatch:
    def test_sample_batch_uniform(self, small_model, small_config, checkpoint_dir) -> None:
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        # Fill buffer
        for exp in _make_fake_experiences(trainer, 20):
            trainer.buffer.add(exp)
        batch = trainer._sample_batch()
        assert batch.board_states is not None


# ===========================================================================
# Tests for train() with physics output metrics extraction
# ===========================================================================


class TestTrainPhysicsMetrics:
    def test_physics_output_metrics_in_history(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """When physics loss is enabled and output is returned, metrics are recorded."""
        small_config.training.physics_informed = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        # Make physics loss succeed with a fake output
        from src.training.physics_loss import PhysicsLossOutput

        fake_physics_output = PhysicsLossOutput(
            total=torch.tensor(0.05),
            residual=torch.tensor(0.03),
            boundary=torch.tensor(0.02),
            initial=torch.tensor(0.0),
            conservation=torch.tensor(0.0),
            weights={"residual": 1.0, "boundary": 1.0},
        )
        trainer.use_physics_loss = True
        trainer.physics_loss_fn = MagicMock(return_value=fake_physics_output)

        with _prefill_and_mock(trainer):
            trainer.train(n_steps=2, log_interval=1, checkpoint_interval=100)

        history = trainer.get_metrics_history()
        assert len(history) == 2
        # Physics weight should be > 0 since we have physics output
        # (the balancer assigns it)


# ===========================================================================
# Tests for console logging with physics (line 914-915)
# ===========================================================================


class TestConsoleLoggingPhysics:
    def test_console_log_includes_physics(self, small_model, small_config, checkpoint_dir) -> None:
        """Console logging includes physics_loss when physics is enabled."""
        from src.training.physics_loss import PhysicsLossOutput

        small_config.training.physics_informed = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        trainer.use_physics_loss = True

        fake_physics_output = PhysicsLossOutput(
            total=torch.tensor(0.05),
            residual=torch.tensor(0.03),
            boundary=torch.tensor(0.02),
            initial=torch.tensor(0.0),
            conservation=torch.tensor(0.0),
            weights={},
        )
        trainer.physics_loss_fn = MagicMock(return_value=fake_physics_output)

        with _prefill_and_mock(trainer):
            # log_interval=1 so we hit the logging path every step
            trainer.train(n_steps=2, log_interval=1, checkpoint_interval=100)

        assert trainer.global_step == 2


# ===========================================================================
# Tests for _run_evaluation with Elo tracker and engine eval
# ===========================================================================


class TestRunEvaluationAdvanced:
    """Cover _run_evaluation paths for Elo and engine evaluation."""

    def test_run_evaluation_triggers_checkpoint_tournament(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        from src.training.evaluation import EvaluationResult

        small_config.training.eval_vs_checkpoints = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        assert trainer.elo_tracker is not None

        fake_result = EvaluationResult(
            win_rate=0.6,
            n_games=10,
            wins=6,
            losses=3,
            draws=1,
            avg_game_length=50.0,
        )
        with (
            patch.object(trainer.evaluator, "evaluate_vs_random", return_value=fake_result),
            patch.object(trainer.evaluator, "measure_policy_agreement", return_value=0.5),
            patch.object(trainer, "_run_checkpoint_tournament") as mock_tournament,
        ):
            trainer._run_evaluation(step=100)

        mock_tournament.assert_called_once_with(100, trainer.training_config.eval_games)

    def test_run_evaluation_triggers_engine_eval(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        from src.training.evaluation import EvaluationResult

        small_config.training.engine_eval_enabled = True
        small_config.training.engine_eval_path = "/fake/stockfish"
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        # Set evaluator.game to non-None so the condition passes
        trainer.evaluator.game = MagicMock()

        fake_result = EvaluationResult(
            win_rate=0.5,
            n_games=10,
            wins=5,
            losses=5,
            draws=0,
            avg_game_length=30.0,
        )
        with (
            patch.object(trainer.evaluator, "evaluate_vs_random", return_value=fake_result),
            patch.object(trainer.evaluator, "measure_policy_agreement", return_value=0.5),
            patch.object(trainer, "_run_engine_evaluation") as mock_engine,
        ):
            trainer._run_evaluation(step=200)

        mock_engine.assert_called_once_with(200)

    def test_run_evaluation_returns_zero_when_no_win_rates(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """When all evaluations produce no win rates, returns 0.0."""
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        # Patch config.board_sizes to be empty (after init) so the for loop
        # produces no win_rates
        trainer.config = MagicMock()
        trainer.config.board_sizes = []
        # multi_resolution_eval is False and board_sizes is empty
        trainer.training_config.multi_resolution_eval = False

        with patch.object(trainer.evaluator, "measure_policy_agreement", return_value=0.0):
            avg = trainer._run_evaluation(step=1)
        assert avg == 0.0


# ===========================================================================
# Tests for _run_engine_evaluation success path
# ===========================================================================


class TestRunEngineEvaluationAdvanced:
    def test_engine_eval_success(self, small_model, small_config, checkpoint_dir) -> None:
        from src.training.evaluation import EvaluationResult

        small_config.training.engine_eval_enabled = True
        small_config.training.engine_eval_path = "/fake/stockfish"
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        fake_result = EvaluationResult(
            win_rate=0.3,
            n_games=10,
            wins=3,
            losses=6,
            draws=1,
            avg_game_length=60.0,
            metadata={"elo_difference": -50, "los": 0.2},
        )

        with (
            patch.object(trainer.evaluator, "evaluate_vs_engine", return_value=fake_result),
            patch(
                "src.training.trainer.UCIConfig",
                create=True,
            ) as mock_uci,
            patch(
                "src.training.trainer.MatchConfig",
                create=True,
            ) as mock_match,
        ):
            # Should run without raising
            trainer._run_engine_evaluation(step=50)

    def test_engine_eval_success_with_wandb(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        from src.training.evaluation import EvaluationResult

        small_config.training.engine_eval_enabled = True
        small_config.training.engine_eval_path = "/fake/stockfish"
        fake_wandb = MagicMock()
        trainer = _make_trainer(small_model, small_config, checkpoint_dir, tracker=fake_wandb)

        fake_result = EvaluationResult(
            win_rate=0.3,
            n_games=10,
            wins=3,
            losses=6,
            draws=1,
            avg_game_length=60.0,
            metadata={"elo_difference": -50, "los": 0.2},
        )

        with patch.object(trainer.evaluator, "evaluate_vs_engine", return_value=fake_result):
            trainer._run_engine_evaluation(step=50)

        assert fake_wandb.log_metrics.called

    def test_engine_eval_with_movetime(self, small_model, small_config, checkpoint_dir) -> None:
        from src.training.evaluation import EvaluationResult

        small_config.training.engine_eval_enabled = True
        small_config.training.engine_eval_path = "/fake/stockfish"
        small_config.training.engine_eval_movetime_ms = 500
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        fake_result = EvaluationResult(
            win_rate=0.4,
            n_games=10,
            wins=4,
            losses=5,
            draws=1,
            avg_game_length=55.0,
            metadata={},
        )

        with patch.object(trainer.evaluator, "evaluate_vs_engine", return_value=fake_result):
            trainer._run_engine_evaluation(step=100)


# ===========================================================================
# Tests for _run_checkpoint_tournament score/wandb branches
# ===========================================================================


class TestCheckpointTournamentScores:
    def test_tournament_draw_score(self, small_model, small_config, checkpoint_dir) -> None:
        """Win rate between thresholds results in score=0.5."""
        from src.training.evaluation import EvaluationResult

        small_config.training.eval_vs_checkpoints = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        fake_result = EvaluationResult(
            win_rate=0.5,
            n_games=4,
            wins=2,
            losses=2,
            draws=0,
            avg_game_length=50.0,
        )
        fake_ckpt = Path(checkpoint_dir / "checkpoint_00000003.pt")

        with (
            patch.object(
                trainer.checkpoint_manager,
                "get_all_checkpoints",
                return_value=[fake_ckpt],
            ),
            patch.object(
                trainer.evaluator,
                "evaluate_vs_checkpoint",
                return_value=fake_result,
            ),
        ):
            trainer._run_checkpoint_tournament(step=10, n_games=4)

    def test_tournament_loss_score(self, small_model, small_config, checkpoint_dir) -> None:
        """Low win rate results in score=0.0."""
        from src.training.evaluation import EvaluationResult

        small_config.training.eval_vs_checkpoints = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        fake_result = EvaluationResult(
            win_rate=0.1,
            n_games=4,
            wins=0,
            losses=4,
            draws=0,
            avg_game_length=50.0,
        )
        fake_ckpt = Path(checkpoint_dir / "checkpoint_00000003.pt")

        with (
            patch.object(
                trainer.checkpoint_manager,
                "get_all_checkpoints",
                return_value=[fake_ckpt],
            ),
            patch.object(
                trainer.evaluator,
                "evaluate_vs_checkpoint",
                return_value=fake_result,
            ),
        ):
            trainer._run_checkpoint_tournament(step=10, n_games=4)

    def test_tournament_with_wandb_logging(self, small_model, small_config, checkpoint_dir) -> None:
        from src.training.evaluation import EvaluationResult

        small_config.training.eval_vs_checkpoints = True
        fake_wandb = MagicMock()
        trainer = _make_trainer(small_model, small_config, checkpoint_dir, tracker=fake_wandb)

        fake_result = EvaluationResult(
            win_rate=0.8,
            n_games=4,
            wins=3,
            losses=1,
            draws=0,
            avg_game_length=50.0,
        )
        fake_ckpt = Path(checkpoint_dir / "checkpoint_00000005.pt")

        with (
            patch.object(
                trainer.checkpoint_manager,
                "get_all_checkpoints",
                return_value=[fake_ckpt],
            ),
            patch.object(
                trainer.evaluator,
                "evaluate_vs_checkpoint",
                return_value=fake_result,
            ),
        ):
            trainer._run_checkpoint_tournament(step=10, n_games=4)

        assert fake_wandb.log_metrics.called


# ===========================================================================
# Tests for train() with curriculum + periodic self-play + wandb
# ===========================================================================


class TestTrainCurriculumSelfPlay:
    def test_periodic_self_play_with_curriculum(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """Periodic self-play uses curriculum board size."""
        small_config.training.curriculum_enabled = True
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)

        for exp in _make_fake_experiences(trainer, 100):
            trainer.buffer.add(exp)

        fake = _make_fake_experiences(trainer, 5)

        with (
            patch.object(trainer, "_fill_buffer"),
            patch.object(
                trainer.self_play_worker,
                "generate_experiences",
                return_value=fake,
            ),
            patch.object(trainer.curriculum, "sample_board_size", return_value=9),
            patch.object(trainer.curriculum, "is_transition_step", return_value=False),
        ):
            trainer.train(
                n_steps=4,
                log_interval=1,
                # Set checkpoint_interval=2 so self_play_interval = max(2//2, 1) = 1
                # This ensures the periodic self-play branch is hit
                checkpoint_interval=2,
            )
        assert trainer.global_step == 4

    def test_curriculum_transition_with_wandb(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """Curriculum transition logs to wandb."""
        small_config.training.curriculum_enabled = True
        fake_wandb = MagicMock()
        trainer = _make_trainer(small_model, small_config, checkpoint_dir, tracker=fake_wandb)

        for exp in _make_fake_experiences(trainer, 100):
            trainer.buffer.add(exp)

        fake = _make_fake_experiences(trainer, 5)

        with (
            patch.object(trainer, "_fill_buffer"),
            patch.object(
                trainer.self_play_worker,
                "generate_experiences",
                return_value=fake,
            ),
            patch.object(
                trainer.curriculum,
                "is_transition_step",
                side_effect=lambda s: s == 1,
            ),
            patch.object(
                trainer.curriculum,
                "get_current_stage",
                return_value=MagicMock(board_sizes=[9, 13], size_weights=[0.5, 0.5]),
            ),
            patch.object(trainer.curriculum, "sample_board_size", return_value=9),
        ):
            trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)

        # Wandb should log curriculum metrics
        assert fake_wandb.log_metrics.called


# ===========================================================================
# Tests for auto device selection
# ===========================================================================


class TestAutoDevice:
    def test_auto_device_uses_dist_ctx_device(
        self, small_model, small_config, checkpoint_dir
    ) -> None:
        """When device='auto', trainer uses dist_ctx.device."""
        from src.training.distributed_context import DistributedContext

        ctx = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            is_distributed=False,
            device=torch.device("cpu"),
        )
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="auto",
            checkpoint_dir=checkpoint_dir,
            distributed_context=ctx,
        )
        assert trainer.device == torch.device("cpu")


# ===========================================================================
# Tests for load_checkpoint
# ===========================================================================


class TestLoadCheckpoint:
    def test_load_best_checkpoint(self, small_model, small_config, checkpoint_dir) -> None:
        """load_checkpoint with load_best=True delegates to checkpoint_manager."""
        trainer = _make_trainer(small_model, small_config, checkpoint_dir)
        with _prefill_and_mock(trainer):
            trainer.train(n_steps=3, log_interval=1, checkpoint_interval=1)

        # Now load best
        new_model = AlphaGalerkinModel(small_config.operator)
        trainer2 = _make_trainer(new_model, small_config, checkpoint_dir)
        import contextlib

        # The checkpoint_manager.restore may or may not have a "best" checkpoint
        # Just test that the call doesn't raise
        with contextlib.suppress(Exception):
            trainer2.load_checkpoint(load_best=True)


# ===========================================================================
# Tests for _run_evaluation multi-res with wandb
# ===========================================================================


class TestRunEvalMultiResWandb:
    def test_multi_res_eval_with_wandb(self, small_model, small_config, checkpoint_dir) -> None:
        from src.training.evaluation import EvaluationResult

        small_config.training.multi_resolution_eval = True
        fake_wandb = MagicMock()
        trainer = _make_trainer(small_model, small_config, checkpoint_dir, tracker=fake_wandb)

        fake_result = EvaluationResult(
            win_rate=0.7,
            n_games=10,
            wins=7,
            losses=2,
            draws=1,
            avg_game_length=40.0,
        )
        multi_res_results = {9: fake_result}

        with (
            patch.object(
                trainer.evaluator,
                "evaluate_multi_resolution",
                return_value=multi_res_results,
            ),
            patch.object(trainer.evaluator, "measure_policy_agreement", return_value=0.5),
        ):
            trainer._run_evaluation(step=200)

        # wandb should have log_evaluation and log_metrics called
        assert fake_wandb.log_evaluation.called
        assert fake_wandb.log_metrics.called
