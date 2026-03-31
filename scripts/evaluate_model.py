#!/usr/bin/env python3
"""Automated evaluation script for AlphaGalerkin Go models.

This script evaluates trained models against a random player and measures
policy agreement with MCTS search. It properly loads model configuration
from checkpoints for architecture compatibility.

Usage:
    python -m scripts.evaluate_model --model checkpoints/best.pt --board-size 9

Default values:
    --n-games: 20 (games to play vs random)
    --board-size: 9 (board size for evaluation)
    --n-sims: 100 (MCTS simulations per move)
    --n-positions: 50 (positions for policy agreement)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import structlog
import torch

from config.schemas import MCTSConfig
from src.training.checkpoint import create_model_from_checkpoint
from src.training.evaluation import Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# Default evaluation parameters
DEFAULT_N_GAMES = 20
DEFAULT_BOARD_SIZE = 9
DEFAULT_N_SIMS = 100
DEFAULT_N_POSITIONS = 50


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate AlphaGalerkin Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=DEFAULT_N_GAMES,
        help="Number of games to play vs Random",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=DEFAULT_BOARD_SIZE,
        help="Board size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=DEFAULT_N_SIMS,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--n-positions",
        type=int,
        default=DEFAULT_N_POSITIONS,
        help="Positions for policy agreement measurement",
    )
    return parser.parse_args()


def main() -> int:
    """Run model evaluation.

    Returns:
        Exit code (0 for success, 1 for failure).

    """
    args = parse_args()

    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        logger.error("model_not_found", path=str(checkpoint_path))
        return 1

    # Load model with proper config from checkpoint
    logger.info(
        "loading_model",
        path=str(checkpoint_path),
        device=args.device,
    )

    try:
        model, config_dict = create_model_from_checkpoint(
            path=checkpoint_path,
            device=args.device,
        )
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        return 1

    # Log config source
    if config_dict:
        logger.info("model_config_loaded_from_checkpoint")
    else:
        logger.warning("using_default_model_config")

    # Configure MCTS for evaluation
    mcts_config = MCTSConfig(n_simulations=args.n_sims)

    # Run evaluation
    logger.info(
        "starting_evaluation",
        board_size=args.board_size,
        n_games=args.n_games,
        n_sims=args.n_sims,
    )

    evaluator = Evaluator(
        model=model,
        mcts_config=mcts_config,
        device=args.device,
        board_sizes=[args.board_size],
    )

    # 1. Evaluate vs Random
    results = evaluator.evaluate_vs_random(
        n_games=args.n_games,
        board_size=args.board_size,
    )

    print("\n" + "=" * 40)
    print(" EVALUATION RESULTS (vs Random)")
    print("=" * 40)
    print(f" Model:      {checkpoint_path.name}")
    print(f" Board Size: {args.board_size}x{args.board_size}")
    print(f" Games:      {results.n_games}")
    print(f" Win Rate:   {results.win_rate:.2%}")
    print(f" Record:     {results.wins}W - {results.losses}L - {results.draws}D")
    print(f" Avg Length: {results.avg_game_length:.1f} moves")
    print("=" * 40 + "\n")

    # 2. Measure Policy Agreement
    logger.info("measuring_policy_agreement", n_positions=args.n_positions)
    agreement = evaluator.measure_policy_agreement(
        n_positions=args.n_positions,
        board_size=args.board_size,
    )
    print(f" Policy Agreement with MCTS: {agreement:.2%}\n")

    logger.info(
        "evaluation_complete",
        win_rate=results.win_rate,
        policy_agreement=agreement,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
