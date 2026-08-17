#!/usr/bin/env python3
"""Chess self-play training CLI for AlphaGalerkin.

Trains a chess model via pure self-play (AlphaZero methodology).
Stockfish is used only for benchmark evaluation, not for training.

Usage:
    python -m scripts.train_chess
    python -m scripts.train_chess training.total_steps=1000
    python -m scripts.train_chess --engine-path /path/to/stockfish

"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra  # noqa: E402
import structlog  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from config.schemas import AlphaGalerkinConfig  # noqa: E402
from src.games.chess import ChessGame  # noqa: E402
from src.modeling.model import AlphaGalerkinModel  # noqa: E402
from src.training.langfuse_tracker import create_tracker  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402

logger = structlog.get_logger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    import logging

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level),
    )


@hydra.main(version_base=None, config_path="../config", config_name="train_chess")
def main(cfg: DictConfig) -> None:
    """Main chess training entry point.

    Args:
        cfg: Hydra configuration.

    """
    # Convert to dict for Pydantic
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Setup logging
    log_level = cfg_dict.get("log_level", "INFO")
    setup_logging(log_level)

    logger.info("chess_training_starting", config=cfg_dict)

    # Set random seed
    seed = cfg_dict.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create config
    try:
        config = AlphaGalerkinConfig(**cfg_dict)
    except Exception as e:
        logger.error("config_validation_failed", error=str(e))
        raise

    # Create chess game instance
    chess_game = ChessGame()
    logger.info(
        "chess_game_initialized",
        action_space_size=chess_game.action_space_size,
        state_channels=chess_game.state_channels,
    )

    # Validate config matches chess
    assert config.operator.input_channels == chess_game.state_channels, (
        f"Config input_channels ({config.operator.input_channels}) != "
        f"chess state_channels ({chess_game.state_channels})"
    )
    assert config.operator.action_space_size == chess_game.action_space_size, (
        f"Config action_space_size ({config.operator.action_space_size}) != "
        f"chess action_space_size ({chess_game.action_space_size})"
    )

    # Create model
    logger.info(
        "creating_chess_model",
        d_model=config.operator.d_model,
        input_channels=config.operator.input_channels,
        action_space_size=config.operator.action_space_size,
    )
    model = AlphaGalerkinModel(config.operator)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model_created", n_parameters=f"{n_params:,}")

    # Device selection
    device = cfg_dict.get("device", "auto")

    # Checkpoint directory
    checkpoint_dir = Path(cfg_dict.get("checkpoint_dir", "checkpoints")) / config.experiment_name

    # Resume path
    resume_from = cfg_dict.get("resume", None)

    # Initialize Langfuse experiment tracker
    tracker = None
    langfuse_config = cfg_dict.get("langfuse", {})
    if langfuse_config.get("enabled", True):
        if langfuse_config.get("run_name") is None:
            langfuse_config["run_name"] = config.experiment_name

        existing_tags = langfuse_config.get("tags") or []
        tags = list(existing_tags) if existing_tags else []
        tags.append("chess")
        tags.append(f"actions:{config.operator.action_space_size}")
        langfuse_config["tags"] = tags

        tracker = create_tracker(
            langfuse_config=langfuse_config,
            training_config=cfg_dict,
        )
        logger.info("tracker_created", enabled=tracker.is_enabled)

    # Create trainer with chess game injected
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        tracker=tracker,
        game=chess_game,
    )

    # Resume from checkpoint if specified
    if resume_from is not None:
        from src.training.checkpoint import load_checkpoint

        logger.info("resuming_from_checkpoint", path=resume_from)
        load_checkpoint(
            path=resume_from,
            model=model,
            optimizer=trainer.optimizer,
        )

    # Run training
    try:
        trainer.train(
            n_steps=config.training.total_steps,
            log_interval=cfg_dict.get("log_interval", 25),
            checkpoint_interval=config.training.checkpoint_interval,
            eval_interval=getattr(config.training, "eval_interval", None),
        )
    except KeyboardInterrupt:
        logger.info("training_interrupted")
        trainer.save_checkpoint()
    except Exception as e:
        logger.error("training_failed", error=str(e))
        trainer.save_checkpoint()
        raise
    finally:
        if tracker is not None:
            tracker.finish()

    logger.info("chess_training_complete")


if __name__ == "__main__":
    main()
