"""Declarative Self-Play Skill for generating training experiences via MCTS."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.templates.logging import create_logger_class

if TYPE_CHECKING:
    from src.core.protocols import EvaluatorProtocol, GameProtocol

SelfPlaySkillLogger = create_logger_class("SelfPlaySkill")


@dataclass
class SelfPlayConfig:
    """Configuration for self-play experience generation."""

    n_games: int = 1
    max_moves_per_game: int = 100
    temperature: float = 1.0
    temperature_drop_step: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfPlayResult:
    """Aggregated outcome of self-play generation."""

    total_games: int
    total_positions: int
    generation_duration_s: float
    positions_per_second: float
    outcomes: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SelfPlaySkill:
    """Reusable skill for autonomous self-play data generation."""

    def __init__(self, config: SelfPlayConfig) -> None:
        self.config = config
        self.logger = SelfPlaySkillLogger(component="SelfPlay")

    def run_rollout(
        self,
        game: GameProtocol,
        evaluator: EvaluatorProtocol,
        step_callback: Any = None,
    ) -> tuple[list[Any], float]:
        """Execute a single self-play game trajectory until terminal or max moves."""
        game_state = game.clone()
        positions: list[Any] = []
        step_count = 0

        while not game_state.is_terminal() and step_count < self.config.max_moves_per_game:
            legal_actions = game_state.get_legal_actions()
            if not legal_actions:
                break

            current_state = game_state.get_state()
            eval_result = evaluator.evaluate(current_state, legal_actions)

            # Store experience tuple
            positions.append((current_state, eval_result.policy, eval_result.value))

            # Pick top legal action
            action = legal_actions[0]
            if hasattr(eval_result, "policy") and eval_result.policy:
                # If policy is dictionary of action -> prob
                if isinstance(eval_result.policy, dict):
                    action = max(eval_result.policy.items(), key=lambda kv: kv[1])[0]

            game_state.apply_action(action)
            step_count += 1
            if step_callback:
                step_callback(step_count, game_state)

        outcome = float(game_state.get_winner())
        return positions, outcome

    def generate(
        self,
        game_factory: Any,
        evaluator_factory: Any,
    ) -> SelfPlayResult:
        """Run batch of self-play games."""
        self.logger.info("self_play_started", n_games=self.config.n_games)
        start_time = time.perf_counter()

        all_positions: list[Any] = []
        outcomes: list[float] = []

        for game_idx in range(self.config.n_games):
            game = game_factory()
            evaluator = evaluator_factory()
            positions, outcome = self.run_rollout(game, evaluator)
            all_positions.extend(positions)
            outcomes.append(outcome)
            self.logger.debug(
                "game_finished",
                game_idx=game_idx,
                steps=len(positions),
                outcome=outcome,
            )

        duration = time.perf_counter() - start_time
        pps = (len(all_positions) / duration) if duration > 0 else 0.0

        result = SelfPlayResult(
            total_games=self.config.n_games,
            total_positions=len(all_positions),
            generation_duration_s=duration,
            positions_per_second=pps,
            outcomes=outcomes,
            metadata=self.config.metadata,
        )

        self.logger.info(
            "self_play_finished",
            total_positions=len(all_positions),
            duration_s=duration,
            positions_per_sec=pps,
        )
        return result
