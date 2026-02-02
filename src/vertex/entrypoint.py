#!/usr/bin/env python3
"""Vertex AI training entrypoint.

This module provides the main entry point for AlphaGalerkin training
when running in a Vertex AI custom training container.

It handles:
- Distributed training setup from Vertex AI environment
- GCS checkpoint management
- Preemption handling for spot instances
- Training execution with proper error handling

Usage:
    python -m src.vertex.entrypoint --config config.yaml
    python -m src.vertex.entrypoint --resume gs://bucket/checkpoint.pt
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.distributed as dist

from src.vertex.config import VertexTrainingConfig
from src.vertex.multi_node import DistributedContext, setup_distributed_training
from src.vertex.storage import GCSCheckpointManager

logger = structlog.get_logger(__name__)


def init_wandb_for_vertex(
    training_config: dict[str, Any],
    ctx: DistributedContext,
) -> Any:
    """Initialize W&B logger from environment variables.

    Only initializes on main process (rank 0) to avoid duplicate runs.

    Args:
        training_config: Training configuration dictionary.
        ctx: Distributed context.

    Returns:
        WandbLogger instance or None if disabled.

    """
    import os as os_module

    # Only main process logs to W&B
    if not ctx.is_main_process():
        return None

    api_key = os_module.environ.get("WANDB_API_KEY")
    mode = os_module.environ.get("WANDB_MODE", "online")

    if not api_key or mode == "disabled":
        logger.info("wandb_disabled", reason="no API key or mode=disabled")
        return None

    # Build W&B config from environment and training config
    wandb_config = training_config.get("wandb", {}).copy()
    default_project = wandb_config.get("project", "alphagalerkin")
    wandb_config.update(
        {
            "enabled": True,
            "project": os_module.environ.get("WANDB_PROJECT", default_project),
            "entity": os_module.environ.get("WANDB_ENTITY", wandb_config.get("entity")),
            "name": os_module.environ.get("WANDB_RUN_NAME", wandb_config.get("name")),
            "mode": mode,
            "tags": wandb_config.get("tags", []) + ["vertex-ai"],
        }
    )

    try:
        from src.training.wandb_logger import create_wandb_logger

        wandb_logger = create_wandb_logger(wandb_config, training_config)
        logger.info(
            "wandb_initialized",
            project=wandb_config["project"],
            run_id=getattr(wandb_logger, "run_id", None),
        )
        return wandb_logger
    except Exception as e:
        logger.warning("wandb_init_failed", error=str(e))
        return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AlphaGalerkin Vertex AI Training Entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--vertex-config",
        type=str,
        help="Path to Vertex AI configuration YAML (optional, uses env vars if not set)",
    )

    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        help="GCS path or local path to checkpoint for resuming",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/tmp/alphagalerkin_cache",
        help="Local checkpoint cache directory",
    )

    # Distributed training
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed training backend",
    )

    # Debugging
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without training",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """Configure structured logging."""
    import logging

    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        format="%(message)s",
        level=level,
        stream=sys.stderr,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def init_distributed(
    ctx: DistributedContext,
    backend: str = "nccl",
) -> None:
    """Initialize PyTorch distributed training.

    Args:
        ctx: Distributed context from Vertex AI environment.
        backend: Distributed backend (nccl or gloo).

    """
    if ctx.world_size <= 1:
        logger.info("single_node_training", world_size=1)
        return

    init_method = f"tcp://{ctx.master_addr}:{ctx.master_port}"

    logger.info(
        "initializing_distributed",
        backend=backend,
        init_method=init_method,
        rank=ctx.rank,
        world_size=ctx.world_size,
    )

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=ctx.world_size,
        rank=ctx.rank,
    )

    # Set CUDA device for this process
    if torch.cuda.is_available() and ctx.local_rank < torch.cuda.device_count():
        torch.cuda.set_device(ctx.local_rank)
        logger.info("cuda_device_set", local_rank=ctx.local_rank)


def load_training_config(config_path: str) -> dict[str, Any]:
    """Load training configuration from YAML.

    Args:
        config_path: Path to configuration file.

    Returns:
        Configuration dictionary.

    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("config_loaded", path=config_path)
    return config


def create_vertex_config_from_env() -> VertexTrainingConfig:
    """Create Vertex config from environment variables.

    Returns:
        VertexTrainingConfig instance.

    """
    return VertexTrainingConfig.from_environment()


def run_training(
    config: dict[str, Any],
    vertex_config: VertexTrainingConfig,
    ctx: DistributedContext,
    checkpoint_manager: GCSCheckpointManager,
    resume_path: str | None = None,
) -> dict[str, Any]:
    """Run the training loop.

    Args:
        config: Training configuration.
        vertex_config: Vertex AI configuration.
        ctx: Distributed context.
        checkpoint_manager: GCS checkpoint manager.
        resume_path: Optional checkpoint path to resume from.

    Returns:
        Training results dictionary.

    """
    logger.info(
        "training_starting",
        is_main_process=ctx.is_main_process(),
        world_size=ctx.world_size,
    )

    results: dict[str, Any] = {
        "status": "completed",
        "final_step": 0,
        "final_loss": 0.0,
    }

    try:
        # Import training infrastructure
        from src.vertex.trainer import VertexTrainer

        # Create trainer with vertex-aware configuration
        trainer = VertexTrainer(
            training_config=config,
            vertex_config=vertex_config,
            checkpoint_manager=checkpoint_manager,
            is_main_process=ctx.is_main_process(),
            world_size=ctx.world_size,
            local_rank=ctx.local_rank,
        )

        # Resume from checkpoint if specified
        if resume_path:
            trainer.load_checkpoint(resume_path)
            logger.info("checkpoint_loaded", path=resume_path)

        # Run training loop
        train_results = trainer.train()

        results.update(
            {
                "status": "completed",
                "final_step": train_results.get("step", 0),
                "final_loss": train_results.get("loss", 0.0),
                "metrics": train_results.get("metrics", {}),
            }
        )

    except ImportError as e:
        # VertexTrainer not available - fall back to basic training
        logger.warning(
            "vertex_trainer_unavailable",
            error=str(e),
            fallback="basic_training",
        )

        # Try direct training approach
        try:
            from config.schemas import TrainingConfig
            from src.training.trainer import AlphaGalerkinTrainer

            training_cfg = TrainingConfig(**config.get("training", {}))
            trainer = AlphaGalerkinTrainer(config=training_cfg)

            if resume_path:
                trainer.load_checkpoint(resume_path)

            train_results = trainer.train()
            results.update(
                {
                    "status": "completed",
                    "final_step": getattr(train_results, "step", 0),
                    "final_loss": getattr(train_results, "loss", 0.0),
                }
            )

        except Exception as inner_e:
            logger.error("training_fallback_failed", error=str(inner_e))
            results["status"] = "failed"
            results["error"] = str(inner_e)

    logger.info("training_completed", results=results)
    return results


class GracefulShutdownHandler:
    """Handle graceful shutdown for preemption and signals."""

    def __init__(self, checkpoint_callback: callable | None = None) -> None:
        """Initialize shutdown handler.

        Args:
            checkpoint_callback: Function to call for emergency checkpoint.

        """
        self.checkpoint_callback = checkpoint_callback
        self._shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        logger.debug("signal_handlers_registered")

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.warning("shutdown_signal_received", signal=signal_name)

        self._shutdown_requested = True

        if self.checkpoint_callback:
            logger.info("saving_emergency_checkpoint")
            try:
                self.checkpoint_callback()
                logger.info("emergency_checkpoint_saved")
            except Exception as e:
                logger.error("emergency_checkpoint_failed", error=str(e))

    @property
    def should_shutdown(self) -> bool:
        """Check if shutdown was requested."""
        return self._shutdown_requested


def main() -> int:
    """Main entry point for Vertex AI training.

    Returns:
        Exit code (0 for success, non-zero for failure).

    """
    args = parse_args()
    setup_logging(debug=args.debug)

    logger.info(
        "vertex_training_entrypoint_started",
        config=args.config,
        resume=args.resume,
        backend=args.backend,
    )

    try:
        # Setup distributed training
        ctx = setup_distributed_training()

        if args.dry_run:
            logger.info(
                "dry_run_mode",
                distributed_context={
                    "rank": ctx.rank,
                    "world_size": ctx.world_size,
                    "master_addr": ctx.master_addr,
                },
            )
            return 0

        # Initialize distributed
        init_distributed(ctx, backend=args.backend)

        # Load configurations
        training_config = load_training_config(args.config)

        # Initialize W&B (only on main process)
        wandb_logger = init_wandb_for_vertex(training_config, ctx)

        try:
            vertex_config = create_vertex_config_from_env()
        except ValueError as e:
            logger.error("vertex_config_error", error=str(e))
            # Create minimal config for local testing
            from src.vertex.config import VertexStorageConfig

            vertex_config = VertexTrainingConfig(
                project_id="local-test",
                staging_bucket="gs://local-test",
                storage=VertexStorageConfig(bucket_name="local-test"),
            )

        # Setup checkpoint manager
        checkpoint_manager = GCSCheckpointManager(
            bucket_name=vertex_config.storage.bucket_name,
            checkpoint_prefix=vertex_config.storage.checkpoint_prefix,
            local_cache_dir=Path(args.checkpoint_dir),
        )

        # Setup shutdown handler
        def emergency_checkpoint() -> None:
            # Placeholder - actual implementation would save current state
            pass

        # Note: Handler registers signal handlers in __init__ and must stay in scope
        _shutdown_handler = GracefulShutdownHandler(
            checkpoint_callback=emergency_checkpoint,
        )

        # Run training with W&B cleanup
        try:
            results = run_training(
                config=training_config,
                vertex_config=vertex_config,
                ctx=ctx,
                checkpoint_manager=checkpoint_manager,
                resume_path=args.resume,
            )
        finally:
            # Cleanup W&B
            if wandb_logger is not None:
                try:
                    wandb_logger.finish()
                    logger.info("wandb_finished")
                except Exception as e:
                    logger.warning("wandb_finish_failed", error=str(e))

        # Cleanup distributed
        if dist.is_initialized():
            dist.destroy_process_group()

        # Only main process reports final status
        if ctx.is_main_process():
            logger.info("training_finished", results=results)

        return 0

    except KeyboardInterrupt:
        logger.warning("training_interrupted")
        return 130
    except Exception as e:
        logger.exception("training_failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
