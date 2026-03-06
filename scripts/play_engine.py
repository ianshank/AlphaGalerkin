r"""CLI for playing AlphaGalerkin against a UCI chess engine.

Usage:
    # Play against Stockfish with depth limit
    python -m scripts.play_engine \
        --engine-path /usr/bin/stockfish \
        --n-games 10 \
        --depth 15 \
        --model checkpoints/chess_model.pt

    # Play with movetime control and PGN output
    python -m scripts.play_engine \
        --engine-path /usr/bin/stockfish \
        --movetime-ms 1000 \
        --n-games 100 \
        --pgn-output results.pgn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog
import torch

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Play AlphaGalerkin chess model against a UCI engine",
    )

    # Engine settings
    parser.add_argument(
        "--engine-path",
        type=Path,
        required=True,
        help="Path to UCI engine binary (e.g., /usr/bin/stockfish)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Engine search depth limit",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Engine search nodes limit",
    )
    parser.add_argument(
        "--movetime-ms",
        type=int,
        default=None,
        help="Engine time per move in milliseconds",
    )
    parser.add_argument(
        "--hash-mb",
        type=int,
        default=64,
        help="Engine hash table size in MB (default: 64)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Engine search threads (default: 1)",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to AlphaGalerkin model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device (default: auto)",
    )

    # Match settings
    parser.add_argument(
        "--n-games",
        type=int,
        default=10,
        help="Number of games to play (default: 10)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=500,
        help="Maximum moves per game (default: 500)",
    )
    parser.add_argument(
        "--opening-fen",
        type=str,
        default=None,
        help="Starting position FEN (default: standard opening)",
    )
    parser.add_argument(
        "--pgn-output",
        type=Path,
        default=None,
        help="Path to write PGN output",
    )
    parser.add_argument(
        "--no-alternate-colors",
        action="store_true",
        help="Always play model as white",
    )

    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    """Run the engine match."""
    args = parse_args()

    # Validate search limits
    if args.depth is None and args.nodes is None and args.movetime_ms is None:
        logger.error("At least one search limit required: --depth, --nodes, or --movetime-ms")
        sys.exit(1)

    # Lazy imports to avoid import overhead when showing help
    from src.engines.config import MatchConfig, UCIConfig
    from src.engines.match import EngineMatch
    from src.games.chess import ChessGame
    from src.modeling.model import AlphaGalerkinModel

    # Configure engine
    engine_config = UCIConfig(
        name="opponent",
        engine_path=args.engine_path,
        depth_limit=args.depth,
        nodes_limit=args.nodes,
        movetime_ms=args.movetime_ms,
        hash_mb=args.hash_mb,
        threads=args.threads,
    )

    # Configure match
    match_config = MatchConfig(
        name="cli_match",
        n_games=args.n_games,
        max_moves=args.max_moves,
        alternate_colors=not args.no_alternate_colors,
        opening_fen=args.opening_fen,
        pgn_output_path=args.pgn_output,
    )

    # Load model
    device = resolve_device(args.device)
    logger.info("loading_model", path=str(args.model), device=str(device))

    checkpoint = torch.load(args.model, map_location=device)
    model_config = checkpoint.get("config", {})
    if isinstance(model_config, dict):
        from config.schemas import AlphaGalerkinConfig

        config = AlphaGalerkinConfig(**model_config)
        model = AlphaGalerkinModel(config.operator)
    else:
        raise ValueError("Cannot determine model config from checkpoint")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Create game and match
    game = ChessGame()

    match = EngineMatch(
        model=model,
        engine_config=engine_config,
        match_config=match_config,
        game=game,
        device=device,
    )

    # Play match
    logger.info(
        "match_starting",
        engine=str(args.engine_path),
        n_games=args.n_games,
    )

    result = match.play_match()

    # Print results
    print(f"\n{'=' * 50}")
    print(f"Match Results: {result.wins}W / {result.losses}L / {result.draws}D")
    print(f"Win Rate: {result.win_rate:.1%}")

    if result.elo_estimate:
        elo = result.elo_estimate
        print(f"Elo Difference: {elo.elo_difference:+.0f}")
        print(f"95% CI: [{elo.confidence_interval[0]:+.0f}, {elo.confidence_interval[1]:+.0f}]")
        print(f"Likelihood of Superiority: {elo.likelihood_of_superiority:.1%}")

    if args.pgn_output:
        print(f"PGN written to: {args.pgn_output}")

    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
