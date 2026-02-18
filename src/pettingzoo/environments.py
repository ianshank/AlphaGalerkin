"""PettingZoo environment factory functions for AlphaGalerkin games.

Each factory creates a configured AlphaGalerkinAECEnv backed by
the appropriate GameInterface implementation. Board sizes,
reward structures, and wrapper behavior are all configurable.

Usage:
    from src.pettingzoo.environments import go_env, othello_env, hex_env

    # Standard Go on 19×19
    env = go_env()

    # Small Go for fast experiments
    env = go_env(board_size=9)

    # Othello for cross-resolution transfer
    env = othello_env(board_size=6)  # train
    env = othello_env(board_size=10) # evaluate

    # Hex for resolution independence
    env = hex_env(board_size=7)   # train
    env = hex_env(board_size=19)  # evaluate
"""

from __future__ import annotations

from src.games.go import GoGame
from src.games.hex import HexGame
from src.games.othello import OthelloGame
from src.pettingzoo.config import PettingZooConfig
from src.pettingzoo.wrapper import AlphaGalerkinAECEnv


def go_env(
    board_size: int | None = None,
    komi: float = 7.5,
    render_mode: str | None = None,
    **config_kwargs: object,
) -> AlphaGalerkinAECEnv:
    """Create a PettingZoo AEC environment for Go.

    Args:
        board_size: Board size (default: 19). Supports 5–25.
        komi: Komi compensation for white (default: 7.5).
        render_mode: "ansi" for text rendering, None for headless.
        **config_kwargs: Additional PettingZooConfig parameters.

    Returns:
        Configured AlphaGalerkinAECEnv for Go.

    """
    game = GoGame(komi=komi)
    config = PettingZooConfig(
        board_size=board_size,
        render_mode=render_mode,
        agent_prefix="player",
        **config_kwargs,  # type: ignore[arg-type]
    )
    return AlphaGalerkinAECEnv(game=game, config=config)


def othello_env(
    board_size: int | None = None,
    render_mode: str | None = None,
    **config_kwargs: object,
) -> AlphaGalerkinAECEnv:
    """Create a PettingZoo AEC environment for Othello.

    Args:
        board_size: Board size (default: 8). Must be even, supports 4–16.
        render_mode: "ansi" for text rendering, None for headless.
        **config_kwargs: Additional PettingZooConfig parameters.

    Returns:
        Configured AlphaGalerkinAECEnv for Othello.

    """
    game = OthelloGame()
    config = PettingZooConfig(
        board_size=board_size,
        render_mode=render_mode,
        agent_prefix="player",
        **config_kwargs,  # type: ignore[arg-type]
    )
    return AlphaGalerkinAECEnv(game=game, config=config)


def hex_env(
    board_size: int | None = None,
    render_mode: str | None = None,
    **config_kwargs: object,
) -> AlphaGalerkinAECEnv:
    """Create a PettingZoo AEC environment for Hex.

    Args:
        board_size: Board size (default: 11). Supports 3–19.
        render_mode: "ansi" for text rendering, None for headless.
        **config_kwargs: Additional PettingZooConfig parameters.

    Returns:
        Configured AlphaGalerkinAECEnv for Hex.

    """
    game = HexGame()
    config = PettingZooConfig(
        board_size=board_size,
        render_mode=render_mode,
        agent_prefix="player",
        **config_kwargs,  # type: ignore[arg-type]
    )
    return AlphaGalerkinAECEnv(game=game, config=config)
