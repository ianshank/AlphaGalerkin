#!/usr/bin/env python3
"""Training CLI entry point for AlphaGalerkin.

Usage:
    python -m scripts.train                        # Default training
    python -m scripts.train --config-name=fast     # Fast test config
    python -m scripts.train training.batch_size=64 # Override batch size
    python -m scripts.train +resume=/path/to/ckpt  # Resume from checkpoint

Examples:
    # Quick test run
    python -m scripts.train --config-name=fast training.total_steps=100

    # Full training on GPU
    python -m scripts.train device=cuda training.total_steps=100000

    # Resume training
    python -m scripts.train +resume=checkpoints/alphagalerkin/checkpoint_00010000.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
import structlog
import torch
from omegaconf import DictConfig, OmegaConf

from config.schemas import AlphaGalerkinConfig
from src.modeling.model import AlphaGalerkinModel
from src.training.trainer import create_trainer
from src.training.wandb_logger import create_wandb_logger

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


def create_config_from_dict(cfg_dict: dict) -> AlphaGalerkinConfig:
    """Create Pydantic config from Hydra dict config.

    Args:
        cfg_dict: Dictionary configuration.

    Returns:
        Validated AlphaGalerkinConfig.
    """
    # Handle nested configs
    return AlphaGalerkinConfig(**cfg_dict)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Hydra configuration.
    """
    # Convert to dict for Pydantic
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Setup logging
    log_level = cfg_dict.get("log_level", "INFO")
    setup_logging(log_level)

    logger.info("training_starting", config=cfg_dict)

    # Set random seed
    seed = cfg_dict.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create config
    try:
        config = create_config_from_dict(cfg_dict)
    except Exception as e:
        logger.error("config_validation_failed", error=str(e))
        raise

    # Create model
    logger.info("creating_model", d_model=config.operator.d_model)
    model = AlphaGalerkinModel(config.operator)

    # Log model info
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model_created", n_parameters=f"{n_params:,}")

    # Device selection
    device = cfg_dict.get("device", "auto")

    # Checkpoint directory
    checkpoint_dir = Path(cfg_dict.get("checkpoint_dir", "checkpoints")) / config.experiment_name

    # Resume path
    resume_from = cfg_dict.get("resume", None)

    # Initialize W&B logger
    wandb_logger = None
    wandb_config = cfg_dict.get("wandb", {})
    if wandb_config.get("enabled", True):
        # Set run name to experiment name if not specified
        if wandb_config.get("name") is None:
            wandb_config["name"] = config.experiment_name

        # Add useful tags
        tags = list(wandb_config.get("tags", []))
        if device != "auto":
            tags.append(f"device:{device}")
        tags.append(f"lr:{config.training.learning_rate}")
        tags.append(f"batch:{config.training.batch_size}")
        wandb_config["tags"] = tags

        wandb_logger = create_wandb_logger(
            wandb_config=wandb_config,
            training_config=cfg_dict,
        )
        logger.info("wandb_logger_created", enabled=wandb_logger.is_enabled)

    # Create trainer
    trainer = create_trainer(
        model=model,
        config=config,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        device=device,
        wandb_logger=wandb_logger,
    )

    # Run training
    try:
        trainer.train(
            n_steps=config.training.total_steps,
            log_interval=cfg_dict.get("log_interval", 100),
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
        # Ensure W&B run is properly closed
        if wandb_logger is not None:
            wandb_logger.finish()

    logger.info("training_complete")


if __name__ == "__main__":
    main()
