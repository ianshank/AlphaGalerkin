"""Pydantic configuration for PettingZoo integration.

Provides validated, no-hardcoded-value configuration for the AEC wrapper,
controlling observation format, reward structure, and agent naming.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PettingZooConfig(BaseModel):
    """Configuration for PettingZoo AEC environment wrapper.

    All values are configurable; nothing is hardcoded in the wrapper itself.

    Attributes:
        board_size: Board size for the game (uses game default if None).
        reward_win: Reward given to the winning agent.
        reward_lose: Reward given to the losing agent.
        reward_draw: Reward given to both agents on a draw.
        reward_illegal: Reward for taking an illegal move (terminates game).
        max_cycles: Maximum number of steps before truncation (None = no limit).
        agent_prefix: Prefix for agent names (e.g., "player" → "player_0").
        terminate_on_illegal: Whether illegal moves terminate the game.
        render_mode: Render mode (None or "ansi" for text rendering).

    """

    board_size: int | None = Field(
        default=None,
        ge=1,
        description="Board size for the game (uses game default if None)",
    )
    reward_win: float = Field(default=1.0, description="Reward for winning")
    reward_lose: float = Field(default=-1.0, description="Reward for losing")
    reward_draw: float = Field(default=0.0, description="Reward for draw")
    reward_illegal: float = Field(
        default=-1.0,
        description="Reward for illegal move (terminates game)",
    )
    max_cycles: int | None = Field(
        default=None,
        ge=1,
        description="Maximum steps before truncation (None = no limit)",
    )
    agent_prefix: str = Field(
        default="player",
        min_length=1,
        description="Prefix for agent names",
    )
    terminate_on_illegal: bool = Field(
        default=True,
        description="Whether illegal moves terminate the game",
    )
    render_mode: str | None = Field(
        default=None,
        description="Render mode (None or 'ansi')",
    )

    model_config = {"frozen": False}

    def agent_name(self, index: int) -> str:
        """Generate agent name from index.

        Args:
            index: Zero-based agent index.

        Returns:
            Agent name string (e.g., "player_0").

        """
        return f"{self.agent_prefix}_{index}"
