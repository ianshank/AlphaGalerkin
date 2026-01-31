#!/usr/bin/env python3
"""Automated evaluation script for AlphaGalerkin Go models."""

import argparse
import logging
from pathlib import Path

import torch
import structlog
from config.schemas import OperatorConfig, MCTSConfig
from src.modeling.model import AlphaGalerkinModel
from src.training.evaluation import Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaGalerkin Model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--n-games", type=int, default=20, help="Number of games to play vs Random")
    parser.add_argument("--board-size", type=int, default=9, help="Board size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--n-sims", type=int, default=100, help="MCTS simulations per move")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        logger.error("model_not_found", path=str(checkpoint_path))
        return

    logger.info("loading_model", path=str(checkpoint_path), device=args.device)
    
    # Initialize model
    config = OperatorConfig()
    model = AlphaGalerkinModel(config)
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(args.device)
    model.eval()

    # Configure MCTS for evaluation
    mcts_config = MCTSConfig(n_simulations=args.n_sims)
    
    # Run evaluation
    logger.info("starting_evaluation", board_size=args.board_size, n_games=args.n_games)
    
    evaluator = Evaluator(
        model=model,
        mcts_config=mcts_config,
        device=args.device,
        board_sizes=[args.board_size]
    )

    # 1. Evaluate vs Random
    results = evaluator.evaluate_vs_random(n_games=args.n_games, board_size=args.board_size)
    
    print("\n" + "="*40)
    print(f" EVALUATION RESULTS (vs Random)")
    print("="*40)
    print(f" Model:      {checkpoint_path.name}")
    print(f" Board Size: {args.board_size}x{args.board_size}")
    print(f" Games:      {results.n_games}")
    print(f" Win Rate:   {results.win_rate:.2%}")
    print(f" Record:     {results.wins}W - {results.losses}L - {results.draws}D")
    print(f" Avg Length: {results.avg_game_length:.1f} moves")
    print("="*40 + "\n")

    # 2. Measure Policy Agreement
    logger.info("measuring_policy_agreement", n_positions=50)
    agreement = evaluator.measure_policy_agreement(n_positions=50, board_size=args.board_size)
    print(f" Policy Agreement with MCTS: {agreement:.2%}\n")

if __name__ == "__main__":
    main()
