"""CLI entry point for AlphaGalerkin."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
import structlog

from src.alphagalerkin.core.config import AlphaGalerkinConfig
from src.alphagalerkin.utils.logging import configure_logging

logger = structlog.get_logger("cli")


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
@click.option(
    "--log-format",
    default="console",
    help="Log format: json or console",
)
@click.pass_context
def main(
    ctx: click.Context,
    log_level: str,
    log_format: str,
) -> None:
    """AlphaGalerkin: AlphaZero-style MCTS for PDE discretization."""
    configure_logging(level=log_level, format=log_format)
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    help="Config YAML path",
)
@click.option(
    "--pde-type",
    default="elliptic",
    help="PDE type (elliptic, parabolic, hyperbolic, mixed)",
)
@click.option(
    "--override",
    multiple=True,
    help="Config overrides: key=value",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate config without training",
)
def train(
    config_path: str | None,
    pde_type: str,
    override: tuple[str, ...],
    dry_run: bool,
) -> None:
    """Train AlphaGalerkin on a PDE problem."""
    overrides: dict[str, Any] = {}
    for o in override:
        key, _, value = o.partition("=")
        parts = key.split(".")
        target = overrides
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = _parse_value(value)

    overrides.setdefault("physics", {})["pde_type"] = pde_type

    if config_path:
        config = AlphaGalerkinConfig.from_yaml(
            Path(config_path),
            overrides=overrides,
        )
    else:
        config = AlphaGalerkinConfig.from_dict(overrides)

    logger.info(
        "cli.train.config_loaded",
        pde_type=config.physics.pde_type,
        device=config.device,
    )

    if dry_run:
        click.echo(
            f"Config validated successfully. "
            f"PDE type: {config.physics.pde_type}"
        )
        click.echo(
            f"MCTS simulations: {config.mcts.num_simulations}"
        )
        click.echo(
            f"Training steps: {config.training.total_steps}"
        )
        return

    # Import and run trainer
    from src.alphagalerkin.training.trainer import Trainer
    from src.alphagalerkin.utils.seeding import seed_everything

    seed_everything(config.training.seed)
    trainer = Trainer(config)

    for i in range(config.training.total_steps):
        _metrics = trainer.train_iteration(i)

        if (
            (i + 1)
            % config.checkpoint.save_interval_steps
            == 0
        ):
            trainer.save_checkpoint(i)

    click.echo("Training complete.")


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    required=True,
)
def validate_config(config_path: str) -> None:
    """Validate a configuration file."""
    try:
        config = AlphaGalerkinConfig.from_yaml(
            Path(config_path),
        )
        click.echo(f"Configuration valid: {config_path}")
        click.echo(
            f"  PDE type: {config.physics.pde_type}"
        )
        click.echo(f"  Device: {config.device}")
        click.echo(
            f"  MCTS sims: {config.mcts.num_simulations}"
        )
    except Exception as e:
        click.echo(f"Configuration invalid: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    required=True,
    help="Config YAML path",
)
@click.option(
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model checkpoint",
)
@click.option(
    "--num-episodes",
    default=10,
    type=int,
    help="Number of evaluation episodes",
)
def evaluate(
    config_path: str,
    checkpoint_path: str,
    num_episodes: int,
) -> None:
    """Evaluate a trained policy on test problems."""
    config = AlphaGalerkinConfig.from_yaml(
        Path(config_path),
    )

    from src.alphagalerkin.evaluation.evaluator import (
        PolicyEvaluator,
    )

    evaluator = PolicyEvaluator(config)
    metrics = evaluator.evaluate_from_checkpoint(
        checkpoint_path=Path(checkpoint_path),
        num_episodes=num_episodes,
    )

    click.echo("Evaluation results:")
    for key, value in sorted(metrics.items()):
        click.echo(f"  {key}: {value:.6f}")


def _parse_value(value: str) -> Any:
    """Parse a string value to appropriate Python type."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


if __name__ == "__main__":
    main()
