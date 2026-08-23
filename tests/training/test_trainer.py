"""Tests for the main Trainer class."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError as PydanticValidationError

from config.schemas import (
    AlphaGalerkinConfig,
    DomainConfig,
    MCTSConfig,
    OperatorConfig,
    TrainingConfig,
)
from src.modeling.model import AlphaGalerkinModel
from src.training.trainer import BufferFillError, Trainer, TrainingMetrics, create_trainer

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    """Pre-fill buffer and return a context manager that mocks self-play.

    Usage::

        trainer = Trainer(...)
        with _prefill_and_mock(trainer):
            trainer.train(n_steps=3)
    """
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


@pytest.fixture
def small_config() -> AlphaGalerkinConfig:
    """Create small config for fast testing."""
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
        experiment_name="test",
        seed=42,
    )


@pytest.fixture
def small_model(small_config: AlphaGalerkinConfig) -> AlphaGalerkinModel:
    """Create small model."""
    return AlphaGalerkinModel(small_config.operator)


@pytest.fixture
def checkpoint_dir() -> Path:
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test trainer initialization."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        assert trainer.model is small_model
        assert trainer.global_step == 0
        assert trainer.device == torch.device("cpu")

    def test_training_step_increments(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test that training steps increment correctly."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        initial_step = trainer.global_step
        with _prefill_and_mock(trainer):
            trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)

        assert trainer.global_step == initial_step + 3

    def test_metrics_logged(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test that metrics are logged during training."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        with _prefill_and_mock(trainer):
            trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)

        history = trainer.get_metrics_history()
        assert len(history) == 3

        # Check metrics structure
        first_metrics = history[0]
        assert "total_loss" in first_metrics
        assert "policy_loss" in first_metrics
        assert "value_loss" in first_metrics
        assert "learning_rate" in first_metrics

    def test_checkpoint_saved(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test that checkpoints are saved at intervals."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        with _prefill_and_mock(trainer):
            trainer.train(n_steps=5, log_interval=1, checkpoint_interval=2)

        # Should have checkpoints
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) >= 1

    def test_resume_from_checkpoint(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test resuming training from checkpoint."""
        # Initial training
        trainer1 = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        with _prefill_and_mock(trainer1):
            trainer1.train(n_steps=3, log_interval=1, checkpoint_interval=1)
        saved_step = trainer1.global_step

        # Create new trainer and resume
        new_model = AlphaGalerkinModel(small_config.operator)
        trainer2 = Trainer(
            model=new_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        trainer2.load_checkpoint()

        assert trainer2.global_step == saved_step

    def test_lr_schedule_applied(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test that learning rate schedule is applied."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        initial_lr = trainer.get_current_lr()
        with _prefill_and_mock(trainer):
            trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)
        final_lr = trainer.get_current_lr()

        # With constant scheduler, LR should be same
        # With cosine, it would decrease
        assert final_lr > 0


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Test metrics serialization."""
        metrics = TrainingMetrics(
            step=100,
            total_loss=0.5,
            policy_loss=0.3,
            value_loss=0.2,
            lbb_loss=0.0,
            learning_rate=1e-4,
        )

        d = metrics.to_dict()

        assert d["step"] == 100
        assert d["total_loss"] == 0.5
        assert d["learning_rate"] == 1e-4


class TestCreateTrainer:
    """Tests for create_trainer factory function."""

    def test_create_trainer_basic(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test creating trainer with factory function."""
        trainer = create_trainer(
            model=small_model,
            config=small_config,
            checkpoint_dir=checkpoint_dir,
            device="cpu",
        )

        assert isinstance(trainer, Trainer)
        assert trainer.global_step == 0

    def test_create_trainer_with_resume(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Test creating trainer with checkpoint resumption."""
        # First, create and save a checkpoint
        trainer1 = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        with _prefill_and_mock(trainer1):
            trainer1.train(n_steps=2, log_interval=1, checkpoint_interval=1)
        ckpt_path = trainer1.checkpoint_manager.get_latest()

        # Create new trainer resuming from checkpoint
        new_model = AlphaGalerkinModel(small_config.operator)
        trainer2 = create_trainer(
            model=new_model,
            config=small_config,
            checkpoint_dir=checkpoint_dir,
            resume_from=ckpt_path,
            device="cpu",
        )

        assert trainer2.global_step == trainer1.global_step


class TestTrainerFillBuffer:
    """Tests for the Trainer._fill_buffer method."""

    def test_fill_buffer_populates_replay(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """_fill_buffer should populate the replay buffer to min_size."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        fake = _make_fake_experiences(trainer, 20)
        with patch.object(
            trainer.self_play_worker,
            "generate_experiences",
            return_value=fake,
        ):
            trainer._fill_buffer(min_size=10)
        assert len(trainer.buffer) >= 10

    def test_fill_buffer_increments_total_games(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """_fill_buffer increments total_games_generated."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        fake = _make_fake_experiences(trainer, 20)
        with patch.object(
            trainer.self_play_worker,
            "generate_experiences",
            return_value=fake,
        ):
            trainer._fill_buffer(min_size=10)
        assert trainer.total_games_generated > 0

    def test_fill_buffer_raises_when_self_play_yields_nothing(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """A self-play worker that never yields experiences must not hang.

        Regression test for the unbounded ``_fill_buffer`` loop: previously,
        if ``generate_experiences()`` ever netted zero new usable
        experiences per call (e.g. a game-length or config bug), the loop
        had no iteration cap and no wall-clock bound, so it would re-invoke
        full self-play MCTS generation forever. It must now raise
        ``BufferFillError`` after exactly
        ``max_buffer_fill_iterations`` self-play calls instead of hanging.
        """
        small_config.training.max_buffer_fill_iterations = 3
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        with patch.object(
            trainer.self_play_worker,
            "generate_experiences",
            return_value=[],
        ) as mock_generate:
            with pytest.raises(
                BufferFillError,
                match="did not reach the minimum buffer size",
            ):
                trainer._fill_buffer(min_size=10)

        # Bounded: exactly max_buffer_fill_iterations calls were made --
        # proof the loop terminated instead of hanging indefinitely.
        assert mock_generate.call_count == 3
        assert len(trainer.buffer) == 0


class TestTrainerStabilityMonitor:
    """Tests for _create_stability_monitor."""

    def test_no_monitor_when_disabled(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """No stability monitor when early stopping and plateau are disabled."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        monitor = trainer._create_stability_monitor()
        # Default config has both disabled
        assert monitor is None

    def test_monitor_with_early_stopping(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Stability monitor created when early stopping is enabled."""
        small_config.training.early_stopping_enabled = True
        small_config.training.early_stopping_patience = 5
        small_config.training.early_stopping_min_delta = 0.01
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        monitor = trainer._create_stability_monitor()
        assert monitor is not None
        assert monitor.early_stopping is not None

    def test_monitor_with_plateau_detection(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Stability monitor created when plateau detection is enabled."""
        small_config.training.plateau_detection_enabled = True
        small_config.training.plateau_patience = 10
        small_config.training.plateau_factor = 0.5
        small_config.training.plateau_min_lr = 1e-6
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        monitor = trainer._create_stability_monitor()
        assert monitor is not None
        assert monitor.plateau_detector is not None


class TestTrainerCurriculum:
    """Tests for trainer curriculum creation."""

    def test_curriculum_created_when_enabled(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Curriculum is created when use_curriculum is enabled."""
        small_config.training.curriculum_enabled = True
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        assert trainer.curriculum is not None

    def test_no_curriculum_by_default(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """No curriculum by default."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        assert trainer.curriculum is None


class TestTrainerPhysicsLoss:
    """Tests for physics loss creation."""

    def test_combined_physics_loss_created_when_type_combined(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Combined physics loss is created when physics_loss_type='combined'."""
        small_config.training.physics_loss_type = "combined"
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        assert trainer.combined_physics_loss_fn is not None

    def test_no_combined_physics_loss_by_default(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """No combined physics loss when type is 'none'."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        assert trainer.combined_physics_loss_fn is None

    def test_no_physics_informed_loss_by_default(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """No physics-informed loss by default."""
        trainer = Trainer(
            model=small_model,
            config=small_config,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        assert trainer.physics_loss_fn is None


# ---------------------------------------------------------------------------
# LR-scheduler configuration wiring
# ---------------------------------------------------------------------------


def _lr_trajectory(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    n_steps: int,
) -> list[float]:
    """Sample the LR that ``optimizer`` sees at each of ``n_steps`` steps."""
    lrs: list[float] = []
    for _ in range(n_steps):
        lrs.append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()
    return lrs


def _reference_trajectory(
    training_cfg: TrainingConfig,
    min_lr_ratio: float,
    warmup_start_factor: float,
    n_steps: int,
) -> list[float]:
    """Build a scheduler directly from literals, bypassing the config plumbing."""
    from src.training.base_trainer import BaseTrainer

    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=training_cfg.learning_rate)
    sched = BaseTrainer._create_scheduler(
        optimizer=opt,
        scheduler_type=training_cfg.lr_scheduler,
        warmup_steps=training_cfg.warmup_steps,
        total_steps=training_cfg.total_steps,
        min_lr_ratio=min_lr_ratio,
        warmup_start_factor=warmup_start_factor,
    )
    return _lr_trajectory(opt, sched, n_steps)


def _cosine_config(base: AlphaGalerkinConfig, **training_overrides: object) -> AlphaGalerkinConfig:
    """Clone ``base`` with a cosine+warmup schedule and optional training overrides."""
    training = base.training.model_copy(
        update={
            "lr_scheduler": "cosine",
            "warmup_steps": 3,
            "total_steps": 12,
            **training_overrides,
        }
    )
    return base.model_copy(update={"training": training})


class TestSchedulerConfigWiring:
    """Guards that Trainer's LR schedule comes from config, not from literals.

    ``Trainer._create_scheduler`` used to pass ``min_lr_ratio=0.1`` /
    ``warmup_start_factor=0.1`` as inline literals. The refactor routes them
    through ``TrainingConfig``; these tests assert both halves of that claim:
    the plumbing carries the config values, *and* the default config still
    reproduces the exact pre-refactor LR trajectory (zero numeric change).
    """

    HISTORICAL_MIN_LR_RATIO = 0.1
    HISTORICAL_WARMUP_START_FACTOR = 0.1

    def test_training_config_defaults_preserve_historical_literals(self) -> None:
        """The new fields default to the values Trainer used to hardcode."""
        cfg = TrainingConfig()
        assert cfg.min_lr_ratio == self.HISTORICAL_MIN_LR_RATIO
        assert cfg.warmup_start_factor == self.HISTORICAL_WARMUP_START_FACTOR

    def test_create_scheduler_receives_config_values(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Trainer forwards the *config* ratios into the base helper."""
        from src.training.base_trainer import BaseTrainer

        cfg = _cosine_config(small_config, min_lr_ratio=0.37, warmup_start_factor=0.73)
        trainer = Trainer(
            model=small_model,
            config=cfg,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )

        captured: dict[str, object] = {}
        real = BaseTrainer._create_scheduler

        def _spy(**kwargs: object):
            captured.update(kwargs)
            return real(**kwargs)  # type: ignore[arg-type]

        with patch.object(BaseTrainer, "_create_scheduler", staticmethod(_spy)):
            trainer._create_scheduler()

        assert captured["min_lr_ratio"] == 0.37
        assert captured["warmup_start_factor"] == 0.73
        assert captured["min_lr_ratio"] == cfg.training.min_lr_ratio
        assert captured["warmup_start_factor"] == cfg.training.warmup_start_factor

    def test_default_config_reproduces_pre_refactor_lr_trajectory(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Zero numeric change: default config == the old inline 0.1 / 0.1."""
        cfg = _cosine_config(small_config)
        trainer = Trainer(
            model=small_model,
            config=cfg,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        n_steps = cfg.training.total_steps

        # trainer.scheduler is the production object built in __init__.
        actual = _lr_trajectory(trainer.optimizer, trainer.scheduler, n_steps)
        expected = _reference_trajectory(
            cfg.training,
            min_lr_ratio=self.HISTORICAL_MIN_LR_RATIO,
            warmup_start_factor=self.HISTORICAL_WARMUP_START_FACTOR,
            n_steps=n_steps,
        )

        assert actual == pytest.approx(expected)

    def test_non_default_ratios_change_the_lr_trajectory(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """Mutation guard: a re-hardcoded 0.1 / 0.1 would make this fail."""
        cfg = _cosine_config(small_config, min_lr_ratio=0.5, warmup_start_factor=0.9)
        trainer = Trainer(
            model=small_model,
            config=cfg,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        n_steps = cfg.training.total_steps

        actual = _lr_trajectory(trainer.optimizer, trainer.scheduler, n_steps)
        historical = _reference_trajectory(
            cfg.training,
            min_lr_ratio=self.HISTORICAL_MIN_LR_RATIO,
            warmup_start_factor=self.HISTORICAL_WARMUP_START_FACTOR,
            n_steps=n_steps,
        )
        configured = _reference_trajectory(
            cfg.training,
            min_lr_ratio=0.5,
            warmup_start_factor=0.9,
            n_steps=n_steps,
        )

        assert actual == pytest.approx(configured)
        assert actual != pytest.approx(historical)
        # Warmup start factor is visible on the very first step.
        assert actual[0] == pytest.approx(cfg.training.learning_rate * 0.9)

    def test_warmup_start_factor_sets_the_step_zero_lr(
        self,
        small_model: AlphaGalerkinModel,
        small_config: AlphaGalerkinConfig,
        checkpoint_dir: Path,
    ) -> None:
        """The default 0.1 factor puts step-0 LR at 10% of peak."""
        cfg = _cosine_config(small_config)
        trainer = Trainer(
            model=small_model,
            config=cfg,
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        assert trainer.scheduler is not None
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
            cfg.training.learning_rate * self.HISTORICAL_WARMUP_START_FACTOR
        )


class TestTrainingConfigSchedulerFieldBounds:
    """Boundary validation for the two new config.schemas.TrainingConfig fields."""

    @pytest.mark.parametrize("value", [0.0, 0.1, 1.0])
    def test_min_lr_ratio_accepts_closed_unit_interval(self, value: float) -> None:
        """ge=0, le=1 -- both endpoints valid."""
        assert TrainingConfig(min_lr_ratio=value).min_lr_ratio == value

    @pytest.mark.parametrize("value", [-1e-9, -0.5, 1.0000001, 2.0])
    def test_min_lr_ratio_rejects_out_of_range(self, value: float) -> None:
        """Values outside [0, 1] are rejected, not clamped."""
        with pytest.raises(PydanticValidationError):
            TrainingConfig(min_lr_ratio=value)

    @pytest.mark.parametrize("value", [1e-9, 0.1, 1.0])
    def test_warmup_start_factor_accepts_half_open_interval(self, value: float) -> None:
        """gt=0, le=1 -- upper endpoint valid."""
        assert TrainingConfig(warmup_start_factor=value).warmup_start_factor == value

    @pytest.mark.parametrize("value", [0.0, -1e-9, -0.5, 1.0000001, 2.0])
    def test_warmup_start_factor_rejects_zero_and_out_of_range(self, value: float) -> None:
        """gt=0 rejects exactly zero: a zero start factor freezes warmup at LR 0."""
        with pytest.raises(PydanticValidationError):
            TrainingConfig(warmup_start_factor=value)

    @pytest.mark.parametrize("field", ["min_lr_ratio", "warmup_start_factor"])
    def test_none_is_rejected(self, field: str) -> None:
        """Neither field is Optional; None must not fall back to the default."""
        with pytest.raises(PydanticValidationError):
            TrainingConfig(**{field: None})

    def test_yaml_default_config_round_trips(self) -> None:
        """config/train.yaml's explicit values load into the new fields."""
        yaml = pytest.importorskip("yaml")
        raw = yaml.safe_load((REPO_ROOT / "config" / "train.yaml").read_text())
        cfg = TrainingConfig(**raw["training"])
        assert cfg.min_lr_ratio == 0.1
        assert cfg.warmup_start_factor == 0.1
