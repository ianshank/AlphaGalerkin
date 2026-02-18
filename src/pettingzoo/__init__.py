"""PettingZoo integration for AlphaGalerkin games.

This module provides PettingZoo AEC environment wrappers for all
games registered in the AlphaGalerkin GameRegistry, enabling
standard multi-agent RL benchmarking and interoperability.

Key Components:
    - AlphaGalerkinAECEnv: Generic wrapper adapting GameInterface → PettingZoo AEC
    - PettingZooConfig: Pydantic configuration for wrapper behavior
    - Environment factory functions for Go, Othello, Hex, etc.

Usage:
    from src.pettingzoo import go_env, othello_env, hex_env

    # Create Go environment with variable board size
    env = go_env(board_size=9, render_mode=None)

    # Standard PettingZoo AEC loop
    env.reset()
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        action = env.action_space(agent).sample(obs["action_mask"])
        env.step(action)
"""

from src.pettingzoo.config import PettingZooConfig
from src.pettingzoo.environments import go_env, hex_env, othello_env
from src.pettingzoo.wrapper import AlphaGalerkinAECEnv

__all__ = [
    "AlphaGalerkinAECEnv",
    "PettingZooConfig",
    "go_env",
    "hex_env",
    "othello_env",
]
