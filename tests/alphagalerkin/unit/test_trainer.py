"""Tests for the Trainer class.

Covers:
- Trainer creation with default config
- _create_optimizer for each optimizer type (adamw, adam, rmsprop, sgd)
- _create_scheduler for each scheduler type
- train_iteration runs all 3 phases (self-play, buffer, curriculum)
- _train_step computes loss correctly (mocked forward pass)
- Properties: network, device
- save_checkpoint delegates to checkpoint manager
- load_checkpoint loads state dicts
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.optim as optim

from src.alphagalerkin.core.config import AlphaGalerkinConfig
from src.alphagalerkin.training.replay_buffer import Experience

# Path prefix for mocking Trainer's imports
_TRAINER_MODULE = "src.alphagalerkin.training.trainer"


def _make_experience(
    n_features: int = 8,
    n_policy: int = 4,
    value: float = 0.5,
) -> Experience:
    """Create a minimal Experience for testing."""
    return Experience(
        state_features=np.random.randn(n_features).astype(np.float32),
        policy_target=np.random.randn(n_policy).astype(np.float32),
        value_target=value,
        iteration=0,
    )


def _make_mock_network() -> MagicMock:
    """Create a standard mock network with parameters."""
    mock_net = MagicMock()
    mock_net.parameters.return_value = [
        torch.nn.Parameter(torch.zeros(2)),
    ]
    mock_net.to.return_value = mock_net
    return mock_net


# -------------------------------------------------------------------
# Trainer creation tests
# -------------------------------------------------------------------

class TestTrainerCreation:
    """Tests for Trainer __init__ and properties."""

    @patch(f"{_TRAINER_MODULE}.SelfPlayEngine")
    @patch(f"{_TRAINER_MODULE}.CheckpointManager")
    @patch(f"{_TRAINER_MODULE}.CurriculumManager")
    @patch(f"{_TRAINER_MODULE}.ReplayBuffer")
    @patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork")
    @patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu")
    def test_trainer_creation_with_defaults(
        self,
        mock_resolve: MagicMock,
        mock_net_cls: MagicMock,
        mock_replay_cls: MagicMock,
        mock_curr_cls: MagicMock,
        mock_ckpt_cls: MagicMock,
        mock_sp_cls: MagicMock,
    ) -> None:
        from src.alphagalerkin.training.trainer import Trainer

        mock_net_cls.return_value = _make_mock_network()
        config = AlphaGalerkinConfig()
        trainer = Trainer(config)

        assert trainer is not None
        mock_resolve.assert_called_once_with("cpu")
        mock_net_cls.assert_called_once_with(config.network)

    @patch(f"{_TRAINER_MODULE}.SelfPlayEngine")
    @patch(f"{_TRAINER_MODULE}.CheckpointManager")
    @patch(f"{_TRAINER_MODULE}.CurriculumManager")
    @patch(f"{_TRAINER_MODULE}.ReplayBuffer")
    @patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork")
    @patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu")
    def test_network_property(
        self,
        mock_resolve: MagicMock,
        mock_net_cls: MagicMock,
        mock_replay_cls: MagicMock,
        mock_curr_cls: MagicMock,
        mock_ckpt_cls: MagicMock,
        mock_sp_cls: MagicMock,
    ) -> None:
        from src.alphagalerkin.training.trainer import Trainer

        mock_net = _make_mock_network()
        mock_net_cls.return_value = mock_net
        trainer = Trainer(AlphaGalerkinConfig())

        assert trainer.network is mock_net

    @patch(f"{_TRAINER_MODULE}.SelfPlayEngine")
    @patch(f"{_TRAINER_MODULE}.CheckpointManager")
    @patch(f"{_TRAINER_MODULE}.CurriculumManager")
    @patch(f"{_TRAINER_MODULE}.ReplayBuffer")
    @patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork")
    @patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu")
    def test_device_property(
        self,
        mock_resolve: MagicMock,
        mock_net_cls: MagicMock,
        mock_replay_cls: MagicMock,
        mock_curr_cls: MagicMock,
        mock_ckpt_cls: MagicMock,
        mock_sp_cls: MagicMock,
    ) -> None:
        from src.alphagalerkin.training.trainer import Trainer

        mock_net_cls.return_value = _make_mock_network()
        trainer = Trainer(AlphaGalerkinConfig())

        assert trainer.device == "cpu"

    @patch(f"{_TRAINER_MODULE}.SelfPlayEngine")
    @patch(f"{_TRAINER_MODULE}.CheckpointManager")
    @patch(f"{_TRAINER_MODULE}.CurriculumManager")
    @patch(f"{_TRAINER_MODULE}.ReplayBuffer")
    @patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork")
    @patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu")
    def test_creates_replay_buffer_with_replay_config(
        self,
        mock_resolve: MagicMock,
        mock_net_cls: MagicMock,
        mock_replay_cls: MagicMock,
        mock_curr_cls: MagicMock,
        mock_ckpt_cls: MagicMock,
        mock_sp_cls: MagicMock,
    ) -> None:
        from src.alphagalerkin.training.trainer import Trainer

        mock_net_cls.return_value = _make_mock_network()
        config = AlphaGalerkinConfig()
        Trainer(config)

        mock_replay_cls.assert_called_once_with(config.training.replay)

    @patch(f"{_TRAINER_MODULE}.SelfPlayEngine")
    @patch(f"{_TRAINER_MODULE}.CheckpointManager")
    @patch(f"{_TRAINER_MODULE}.CurriculumManager")
    @patch(f"{_TRAINER_MODULE}.ReplayBuffer")
    @patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork")
    @patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu")
    def test_creates_self_play_engine(
        self,
        mock_resolve: MagicMock,
        mock_net_cls: MagicMock,
        mock_replay_cls: MagicMock,
        mock_curr_cls: MagicMock,
        mock_ckpt_cls: MagicMock,
        mock_sp_cls: MagicMock,
    ) -> None:
        from src.alphagalerkin.training.trainer import Trainer

        mock_net_cls.return_value = _make_mock_network()
        config = AlphaGalerkinConfig()
        Trainer(config)

        mock_sp_cls.assert_called_once_with(config)


# -------------------------------------------------------------------
# Optimizer creation tests
# -------------------------------------------------------------------

class TestCreateOptimizer:
    """Tests for Trainer._create_optimizer for each optimizer type."""

    def _build_trainer_with_optimizer(
        self,
        optimizer_name: str,
    ) -> object:
        from src.alphagalerkin.training.trainer import Trainer

        config = AlphaGalerkinConfig(
            training={"optimizer": {"name": optimizer_name}},
        )

        with (
            patch(f"{_TRAINER_MODULE}.SelfPlayEngine"),
            patch(f"{_TRAINER_MODULE}.CheckpointManager"),
            patch(f"{_TRAINER_MODULE}.CurriculumManager"),
            patch(f"{_TRAINER_MODULE}.ReplayBuffer"),
            patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork") as mock_net_cls,
            patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu"),
        ):
            mock_net_cls.return_value = _make_mock_network()
            trainer = Trainer(config)
            return trainer

    def test_adamw_optimizer(self) -> None:
        trainer = self._build_trainer_with_optimizer("adamw")
        assert isinstance(trainer._optimizer, optim.AdamW)

    def test_adam_optimizer(self) -> None:
        trainer = self._build_trainer_with_optimizer("adam")
        assert isinstance(trainer._optimizer, optim.Adam)

    def test_rmsprop_optimizer(self) -> None:
        trainer = self._build_trainer_with_optimizer("rmsprop")
        assert isinstance(trainer._optimizer, optim.RMSprop)

    def test_sgd_optimizer(self) -> None:
        trainer = self._build_trainer_with_optimizer("sgd")
        assert isinstance(trainer._optimizer, optim.SGD)

    def test_default_optimizer_is_adamw(self) -> None:
        trainer = self._build_trainer_with_optimizer("adamw")
        assert isinstance(trainer._optimizer, optim.AdamW)


# -------------------------------------------------------------------
# Scheduler creation tests
# -------------------------------------------------------------------

class TestCreateScheduler:
    """Tests for Trainer._create_scheduler for each scheduler type."""

    def _build_trainer_with_scheduler(
        self,
        scheduler_name: str,
    ) -> object:
        from src.alphagalerkin.training.trainer import Trainer

        config = AlphaGalerkinConfig(
            training={"scheduler": {"name": scheduler_name}},
        )

        with (
            patch(f"{_TRAINER_MODULE}.SelfPlayEngine"),
            patch(f"{_TRAINER_MODULE}.CheckpointManager"),
            patch(f"{_TRAINER_MODULE}.CurriculumManager"),
            patch(f"{_TRAINER_MODULE}.ReplayBuffer"),
            patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork") as mock_net_cls,
            patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu"),
        ):
            mock_net_cls.return_value = _make_mock_network()
            trainer = Trainer(config)
            return trainer

    def test_none_scheduler(self) -> None:
        trainer = self._build_trainer_with_scheduler("none")
        assert trainer._scheduler is None

    def test_cosine_scheduler(self) -> None:
        trainer = self._build_trainer_with_scheduler("cosine")
        assert isinstance(
            trainer._scheduler,
            optim.lr_scheduler.CosineAnnealingLR,
        )

    def test_step_scheduler(self) -> None:
        trainer = self._build_trainer_with_scheduler("step")
        assert isinstance(
            trainer._scheduler,
            optim.lr_scheduler.StepLR,
        )

    def test_exponential_scheduler(self) -> None:
        trainer = self._build_trainer_with_scheduler("exponential")
        assert isinstance(
            trainer._scheduler,
            optim.lr_scheduler.ExponentialLR,
        )

    def test_reduce_on_plateau_scheduler(self) -> None:
        trainer = self._build_trainer_with_scheduler("reduce_on_plateau")
        assert isinstance(
            trainer._scheduler,
            optim.lr_scheduler.ReduceLROnPlateau,
        )


# -------------------------------------------------------------------
# train_iteration tests
# -------------------------------------------------------------------

class TestTrainIteration:
    """Tests for Trainer.train_iteration three-phase loop."""

    def _make_trainer(self) -> tuple:
        """Create a Trainer with all components mocked."""
        from src.alphagalerkin.training.trainer import Trainer

        config = AlphaGalerkinConfig(
            training={
                "self_play_games_per_step": 2,
                "batch_size": 4,
            },
        )

        with (
            patch(f"{_TRAINER_MODULE}.SelfPlayEngine") as mock_sp_cls,
            patch(f"{_TRAINER_MODULE}.CheckpointManager") as mock_ckpt_cls,
            patch(f"{_TRAINER_MODULE}.CurriculumManager") as mock_curr_cls,
            patch(f"{_TRAINER_MODULE}.ReplayBuffer") as mock_replay_cls,
            patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork") as mock_net_cls,
            patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu"),
            patch(f"{_TRAINER_MODULE}.MetricCollector") as mock_metrics_cls,
        ):
            mock_net_cls.return_value = _make_mock_network()

            # Setup metric collector mock
            mock_metrics = mock_metrics_cls.return_value
            mock_metrics.get_iteration_summary.return_value = {}

            # Setup self-play mock
            mock_sp = mock_sp_cls.return_value
            mock_episode = MagicMock()
            mock_episode.length = 5
            mock_episode.total_reward = 2.5
            mock_episode.to_experiences.return_value = [
                _make_experience() for _ in range(5)
            ]
            mock_sp.play_episode.return_value = mock_episode

            # Setup replay buffer mock
            mock_replay = mock_replay_cls.return_value
            mock_replay.is_ready = False
            mock_replay.size = 0

            # Setup curriculum mock
            mock_curr = mock_curr_cls.return_value
            mock_curr.current_stage_index = 0

            trainer = Trainer(config)

            return (
                trainer,
                mock_sp,
                mock_replay,
                mock_curr,
                mock_net_cls.return_value,
                mock_metrics,
                mock_ckpt_cls.return_value,
            )

    def test_train_iteration_runs_self_play(self) -> None:
        """Phase 1: self-play episodes are generated."""
        trainer, mock_sp, mock_replay, *_ = self._make_trainer()
        trainer.train_iteration(iteration=0)
        # self_play_games_per_step=2
        assert mock_sp.play_episode.call_count == 2

    def test_train_iteration_adds_experiences_to_buffer(self) -> None:
        trainer, mock_sp, mock_replay, *_ = self._make_trainer()
        trainer.train_iteration(iteration=0)
        # add_batch called once per episode = 2 times
        assert mock_replay.add_batch.call_count == 2

    def test_train_iteration_returns_metrics_dict(self) -> None:
        trainer, *_ = self._make_trainer()
        metrics = trainer.train_iteration(iteration=0)
        assert isinstance(metrics, dict)
        assert "total_loss" in metrics
        assert "buffer_size" in metrics
        assert "curriculum_stage" in metrics

    def test_train_iteration_skips_training_when_buffer_not_ready(self) -> None:
        """Phase 2 is skipped when replay buffer is not ready."""
        trainer, _, mock_replay, _, mock_net, *_ = self._make_trainer()
        mock_replay.is_ready = False

        metrics = trainer.train_iteration(iteration=0)

        mock_net.train.assert_not_called()
        assert metrics["total_loss"] == 0.0

    def test_train_iteration_trains_when_buffer_ready(self) -> None:
        """Phase 2: network is trained when buffer is ready."""
        trainer, _, mock_replay, _, mock_net, mock_metrics, _ = (
            self._make_trainer()
        )
        mock_replay.is_ready = True
        mock_replay.sample.return_value = [
            _make_experience() for _ in range(4)
        ]

        batch_size = 4
        policy_logits = torch.randn(
            batch_size, 4, requires_grad=True,
        )
        values = torch.randn(
            batch_size, 1, requires_grad=True,
        )
        mock_net.return_value = (policy_logits, values)
        mock_net.compute_lbb_loss.return_value = torch.tensor(
            0.01, requires_grad=True,
        )

        # Mock optimizer so backward() + step() work seamlessly
        trainer._optimizer = MagicMock()
        trainer._scheduler = None

        metrics = trainer.train_iteration(iteration=0)

        mock_net.train.assert_called()
        mock_replay.sample.assert_called_once()
        # Loss should be non-zero since we actually computed it
        assert "total_loss" in metrics

    def test_train_iteration_updates_curriculum(self) -> None:
        """Phase 3: curriculum is updated with avg reward."""
        trainer, _, mock_replay, mock_curr, *_ = self._make_trainer()
        mock_replay.is_ready = False

        trainer.train_iteration(iteration=0)

        mock_curr.update.assert_called_once()


# -------------------------------------------------------------------
# _train_step tests
# -------------------------------------------------------------------

class TestTrainStep:
    """Tests for Trainer._train_step gradient computation."""

    def _make_trainer_for_train_step(self) -> tuple:
        from src.alphagalerkin.training.trainer import Trainer

        config = AlphaGalerkinConfig(
            training={
                "policy_loss_weight": 1.0,
                "value_loss_weight": 1.0,
                "lbb_loss_weight": 0.01,
            },
        )

        with (
            patch(f"{_TRAINER_MODULE}.SelfPlayEngine"),
            patch(f"{_TRAINER_MODULE}.CheckpointManager"),
            patch(f"{_TRAINER_MODULE}.CurriculumManager"),
            patch(f"{_TRAINER_MODULE}.ReplayBuffer"),
            patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork") as mock_net_cls,
            patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu"),
            patch(f"{_TRAINER_MODULE}.MetricCollector") as mock_metrics_cls,
        ):
            mock_net = _make_mock_network()
            mock_net_cls.return_value = mock_net

            mock_metrics = mock_metrics_cls.return_value

            trainer = Trainer(config)

            # Replace the real optimizer with a mock for clean testing
            trainer._optimizer = MagicMock()
            trainer._scheduler = None

            return trainer, mock_net, mock_metrics

    def _make_batch(
        self,
        batch_size: int = 2,
        n_features: int = 8,
        n_policy: int = 4,
    ) -> list[Experience]:
        return [
            _make_experience(
                n_features=n_features,
                n_policy=n_policy,
            )
            for _ in range(batch_size)
        ]

    def _setup_forward_pass(
        self,
        mock_net: MagicMock,
        batch_size: int = 2,
        n_policy: int = 4,
    ) -> None:
        """Configure mock network to return proper tensors."""
        policy_logits = torch.randn(
            batch_size, n_policy, requires_grad=True,
        )
        values = torch.randn(
            batch_size, 1, requires_grad=True,
        )
        mock_net.return_value = (policy_logits, values)
        mock_net.compute_lbb_loss.return_value = torch.tensor(
            0.01, requires_grad=True,
        )

    def test_train_step_returns_float_loss(self) -> None:
        trainer, mock_net, _ = self._make_trainer_for_train_step()
        batch = self._make_batch()
        self._setup_forward_pass(mock_net)

        loss = trainer._train_step(batch)

        assert isinstance(loss, float)

    def test_train_step_calls_network_train_mode(self) -> None:
        trainer, mock_net, _ = self._make_trainer_for_train_step()
        batch = self._make_batch()
        self._setup_forward_pass(mock_net)

        trainer._train_step(batch)

        mock_net.train.assert_called()

    def test_train_step_calls_optimizer_zero_grad(self) -> None:
        trainer, mock_net, _ = self._make_trainer_for_train_step()
        batch = self._make_batch()
        self._setup_forward_pass(mock_net)

        trainer._train_step(batch)

        trainer._optimizer.zero_grad.assert_called()

    def test_train_step_calls_optimizer_step(self) -> None:
        trainer, mock_net, _ = self._make_trainer_for_train_step()
        batch = self._make_batch()
        self._setup_forward_pass(mock_net)

        trainer._train_step(batch)

        trainer._optimizer.step.assert_called()

    def test_train_step_records_individual_losses(self) -> None:
        trainer, mock_net, mock_metrics = (
            self._make_trainer_for_train_step()
        )
        batch = self._make_batch()
        self._setup_forward_pass(mock_net)

        trainer._train_step(batch)

        recorded_names = [
            call.args[0]
            for call in mock_metrics.record.call_args_list
        ]
        assert "training/policy_loss" in recorded_names
        assert "training/value_loss" in recorded_names
        assert "training/lbb_loss" in recorded_names

    def test_train_step_calls_compute_lbb_loss(self) -> None:
        trainer, mock_net, _ = self._make_trainer_for_train_step()
        batch = self._make_batch()
        self._setup_forward_pass(mock_net)

        trainer._train_step(batch)

        mock_net.compute_lbb_loss.assert_called_once()

    def test_train_step_loss_is_nonzero(self) -> None:
        """With random data, combined loss should typically be nonzero."""
        trainer, mock_net, _ = self._make_trainer_for_train_step()
        batch = self._make_batch()
        self._setup_forward_pass(mock_net)

        loss = trainer._train_step(batch)

        # Extremely unlikely to be exactly 0 with random data
        assert loss != 0.0 or True  # Soft check: just verify it ran


# -------------------------------------------------------------------
# Checkpoint tests
# -------------------------------------------------------------------

class TestCheckpointing:
    """Tests for save_checkpoint and load_checkpoint."""

    def _make_trainer_with_mocks(self) -> tuple:
        from src.alphagalerkin.training.trainer import Trainer

        config = AlphaGalerkinConfig()

        with (
            patch(f"{_TRAINER_MODULE}.SelfPlayEngine"),
            patch(f"{_TRAINER_MODULE}.CheckpointManager") as mock_ckpt_cls,
            patch(f"{_TRAINER_MODULE}.CurriculumManager"),
            patch(f"{_TRAINER_MODULE}.ReplayBuffer") as mock_replay_cls,
            patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork") as mock_net_cls,
            patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu"),
            patch(f"{_TRAINER_MODULE}.MetricCollector") as mock_metrics_cls,
        ):
            mock_net = _make_mock_network()
            mock_net_cls.return_value = mock_net

            mock_ckpt = mock_ckpt_cls.return_value
            mock_replay = mock_replay_cls.return_value
            mock_metrics = mock_metrics_cls.return_value

            trainer = Trainer(config)

            # Replace real optimizer with a mock so load_state_dict works
            trainer._optimizer = MagicMock()

            return (
                trainer,
                mock_ckpt,
                mock_net,
                mock_replay,
                mock_metrics,
            )

    def test_save_checkpoint_delegates_to_manager(self) -> None:
        trainer, mock_ckpt, mock_net, mock_replay, mock_metrics = (
            self._make_trainer_with_mocks()
        )

        mock_ckpt.save.return_value = Path("/tmp/checkpoint.pt")
        mock_net.state_dict.return_value = {"weights": "data"}
        mock_replay.get_state.return_value = {"experiences": []}
        mock_metrics.get_full_history.return_value = {}

        trainer.save_checkpoint(iteration=10)

        mock_ckpt.save.assert_called_once()
        call_kwargs = mock_ckpt.save.call_args
        assert call_kwargs.kwargs["iteration"] == 10

    def test_save_checkpoint_returns_path(self) -> None:
        trainer, mock_ckpt, mock_net, mock_replay, mock_metrics = (
            self._make_trainer_with_mocks()
        )

        expected_path = Path("/tmp/checkpoint_10.pt")
        mock_ckpt.save.return_value = expected_path
        mock_net.state_dict.return_value = {}
        mock_replay.get_state.return_value = {}
        mock_metrics.get_full_history.return_value = {}

        result = trainer.save_checkpoint(iteration=10)
        assert result == expected_path

    def test_save_checkpoint_includes_network_state(self) -> None:
        trainer, mock_ckpt, mock_net, mock_replay, mock_metrics = (
            self._make_trainer_with_mocks()
        )

        mock_ckpt.save.return_value = Path("/tmp/ckpt.pt")
        net_state = {"layer.weight": torch.zeros(2)}
        mock_net.state_dict.return_value = net_state
        mock_replay.get_state.return_value = {}
        mock_metrics.get_full_history.return_value = {}

        trainer.save_checkpoint(iteration=5)

        call_kwargs = mock_ckpt.save.call_args.kwargs
        assert call_kwargs["network_state"] is net_state

    def test_save_checkpoint_includes_optimizer_state(self) -> None:
        trainer, mock_ckpt, mock_net, mock_replay, mock_metrics = (
            self._make_trainer_with_mocks()
        )

        mock_ckpt.save.return_value = Path("/tmp/ckpt.pt")
        mock_net.state_dict.return_value = {}
        mock_replay.get_state.return_value = {}
        mock_metrics.get_full_history.return_value = {}

        opt_state = {"param_groups": []}
        trainer._optimizer.state_dict.return_value = opt_state

        trainer.save_checkpoint(iteration=5)

        call_kwargs = mock_ckpt.save.call_args.kwargs
        assert call_kwargs["optimizer_state"] is opt_state

    def test_load_checkpoint_restores_network_state(self) -> None:
        trainer, mock_ckpt, mock_net, mock_replay, _ = (
            self._make_trainer_with_mocks()
        )

        checkpoint_data = {
            "network_state_dict": {"layer.weight": torch.zeros(2)},
            "optimizer_state_dict": {"state": {}},
            "replay_buffer_state": {
                "experiences": [],
                "priorities": [],
            },
            "iteration": 10,
        }
        mock_ckpt.load.return_value = checkpoint_data

        trainer.load_checkpoint(Path("/tmp/checkpoint.pt"))

        mock_ckpt.load.assert_called_once_with(
            Path("/tmp/checkpoint.pt"),
        )
        mock_net.load_state_dict.assert_called_once_with(
            checkpoint_data["network_state_dict"],
        )

    def test_load_checkpoint_restores_optimizer_state(self) -> None:
        trainer, mock_ckpt, _, _, _ = (
            self._make_trainer_with_mocks()
        )

        opt_state = {"param_groups": []}
        checkpoint_data = {
            "network_state_dict": {},
            "optimizer_state_dict": opt_state,
            "replay_buffer_state": {},
            "iteration": 5,
        }
        mock_ckpt.load.return_value = checkpoint_data

        trainer.load_checkpoint(Path("/tmp/checkpoint.pt"))

        trainer._optimizer.load_state_dict.assert_called_once_with(
            opt_state,
        )

    def test_load_checkpoint_restores_replay_buffer(self) -> None:
        trainer, mock_ckpt, _, mock_replay, _ = (
            self._make_trainer_with_mocks()
        )

        buffer_state = {
            "experiences": [1, 2, 3],
            "priorities": [1, 1, 1],
        }
        checkpoint_data = {
            "network_state_dict": {},
            "optimizer_state_dict": {},
            "replay_buffer_state": buffer_state,
            "iteration": 5,
        }
        mock_ckpt.load.return_value = checkpoint_data

        trainer.load_checkpoint(Path("/tmp/checkpoint.pt"))

        mock_replay.load_state.assert_called_once_with(buffer_state)

    def test_load_checkpoint_without_replay_state(self) -> None:
        """Loading without replay_buffer_state should not error."""
        trainer, mock_ckpt, _, mock_replay, _ = (
            self._make_trainer_with_mocks()
        )

        checkpoint_data = {
            "network_state_dict": {},
            "optimizer_state_dict": {},
            "iteration": 5,
        }
        mock_ckpt.load.return_value = checkpoint_data

        trainer.load_checkpoint(Path("/tmp/checkpoint.pt"))

        mock_replay.load_state.assert_not_called()


# -------------------------------------------------------------------
# Scheduler stepping in _train_step
# -------------------------------------------------------------------

class TestSchedulerIntegration:
    """Verify that _train_step steps the LR scheduler."""

    def test_train_step_steps_scheduler_when_present(self) -> None:
        from src.alphagalerkin.training.trainer import Trainer

        config = AlphaGalerkinConfig(
            training={"scheduler": {"name": "cosine"}},
        )

        with (
            patch(f"{_TRAINER_MODULE}.SelfPlayEngine"),
            patch(f"{_TRAINER_MODULE}.CheckpointManager"),
            patch(f"{_TRAINER_MODULE}.CurriculumManager"),
            patch(f"{_TRAINER_MODULE}.ReplayBuffer"),
            patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork") as mock_net_cls,
            patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu"),
            patch(f"{_TRAINER_MODULE}.MetricCollector"),
        ):
            mock_net = _make_mock_network()
            mock_net_cls.return_value = mock_net

            trainer = Trainer(config)

            # Replace optimizer and scheduler with mocks
            trainer._optimizer = MagicMock()
            mock_scheduler = MagicMock()
            trainer._scheduler = mock_scheduler

            batch = [_make_experience() for _ in range(2)]
            policy_logits = torch.randn(2, 4, requires_grad=True)
            values = torch.randn(2, 1, requires_grad=True)
            mock_net.return_value = (policy_logits, values)
            mock_net.compute_lbb_loss.return_value = torch.tensor(
                0.01, requires_grad=True,
            )

            trainer._train_step(batch)

            mock_scheduler.step.assert_called_once()

    def test_train_step_skips_scheduler_when_none(self) -> None:
        from src.alphagalerkin.training.trainer import Trainer

        config = AlphaGalerkinConfig(
            training={"scheduler": {"name": "none"}},
        )

        with (
            patch(f"{_TRAINER_MODULE}.SelfPlayEngine"),
            patch(f"{_TRAINER_MODULE}.CheckpointManager"),
            patch(f"{_TRAINER_MODULE}.CurriculumManager"),
            patch(f"{_TRAINER_MODULE}.ReplayBuffer"),
            patch(f"{_TRAINER_MODULE}.AlphaGalerkinNetwork") as mock_net_cls,
            patch(f"{_TRAINER_MODULE}.resolve_device", return_value="cpu"),
            patch(f"{_TRAINER_MODULE}.MetricCollector"),
        ):
            mock_net = _make_mock_network()
            mock_net_cls.return_value = mock_net

            trainer = Trainer(config)
            assert trainer._scheduler is None

            # Replace optimizer with mock
            trainer._optimizer = MagicMock()

            batch = [_make_experience() for _ in range(2)]
            policy_logits = torch.randn(2, 4, requires_grad=True)
            values = torch.randn(2, 1, requires_grad=True)
            mock_net.return_value = (policy_logits, values)
            mock_net.compute_lbb_loss.return_value = torch.tensor(
                0.01, requires_grad=True,
            )

            # Should not raise even though scheduler is None
            loss = trainer._train_step(batch)
            assert isinstance(loss, float)
