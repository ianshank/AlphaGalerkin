"""Match orchestration framework for engine vs AlphaGalerkin games.

Manages the lifecycle of multi-game matches, including color alternation,
game recording, PGN output, and result aggregation with Elo estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import torch

from src.engines.config import EloConfig, MatchConfig, UCIConfig
from src.engines.elo import EloCalculator, EloEstimate
from src.engines.protocol import EngineCrashError, EngineTimeoutError
from src.engines.uci import UCIEngine
from src.games.fen import STARTING_FEN, fen_to_state, state_to_fen

if TYPE_CHECKING:
    from src.games.interface import GameInterface
    from src.modeling.model import AlphaGalerkinModel

logger = structlog.get_logger(__name__)


@dataclass
class GameRecord:
    """Record of a single game in a match.

    Attributes:
        moves: List of UCI move strings played.
        result: Game result ("1-0", "0-1", "1/2-1/2").
        result_reason: Reason for game end (checkmate, stalemate, etc.).
        model_color: Color the model played ("white" or "black").
        opening_fen: Starting position FEN.
        move_count: Total number of moves.
        metadata: Additional game metadata.

    """

    moves: list[str] = field(default_factory=list)
    result: str = ""
    result_reason: str = ""
    model_color: str = ""
    opening_fen: str = STARTING_FEN
    move_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:
    """Aggregated results from a match.

    Attributes:
        wins: Games won by the model.
        losses: Games lost by the model.
        draws: Drawn games.
        games: List of individual game records.
        elo_estimate: Elo difference estimation (if calculated).
        pgn: PGN string of all games (if generated).

    """

    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: list[GameRecord] = field(default_factory=list)
    elo_estimate: EloEstimate | None = None
    pgn: str | None = None

    @property
    def total_games(self) -> int:
        """Total number of completed games."""
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        """Win rate (W + 0.5*D) / N."""
        n = self.total_games
        if n == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / n

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "total_games": self.total_games,
            "win_rate": self.win_rate,
        }
        if self.elo_estimate:
            result["elo_difference"] = self.elo_estimate.elo_difference
            result["elo_ci"] = self.elo_estimate.confidence_interval
            result["los"] = self.elo_estimate.likelihood_of_superiority
        return result


class EngineMatch:
    """Orchestrates matches between an AlphaGalerkin model and a UCI engine.

    Manages the game loop, MCTS search for the model player,
    engine communication for the opponent, and result collection.

    Args:
        model: AlphaGalerkin neural network model.
        engine_config: UCI engine configuration.
        match_config: Match settings (games, time control, etc.).
        game: Game interface instance (e.g., ChessGame).
        mcts_config: MCTS search configuration dict.
        device: Torch device for model inference.

    """

    def __init__(
        self,
        model: AlphaGalerkinModel,
        engine_config: UCIConfig,
        match_config: MatchConfig,
        game: GameInterface,
        mcts_config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._model = model
        self._engine_config = engine_config
        self._match_config = match_config
        self._game = game
        self._mcts_config = mcts_config or {}
        self._device = device or torch.device("cpu")

    def play_match(self) -> MatchResult:
        """Play a full match of n_games.

        Returns:
            MatchResult with wins/losses/draws and optional Elo.

        """
        result = MatchResult()
        pgn_parts: list[str] = []

        for game_idx in range(self._match_config.n_games):
            # Determine colors
            if self._match_config.alternate_colors:
                model_is_white = game_idx % 2 == 0
            else:
                model_is_white = True

            opening_fen = self._match_config.opening_fen or STARTING_FEN

            logger.info(
                "game_start",
                game_num=game_idx + 1,
                total=self._match_config.n_games,
                model_color="white" if model_is_white else "black",
            )

            try:
                outcome, record = self._play_single_game(
                    model_is_white=model_is_white,
                    opening_fen=opening_fen,
                )
            except (EngineCrashError, EngineTimeoutError) as e:
                logger.error("game_error", game_num=game_idx + 1, error=str(e))
                # Engine crash counts as a win for the model
                record = GameRecord(
                    result="engine_error",
                    result_reason=str(e),
                    model_color="white" if model_is_white else "black",
                    opening_fen=opening_fen,
                )
                outcome = 1.0

            result.games.append(record)

            if outcome > 0:
                result.wins += 1
            elif outcome < 0:
                result.losses += 1
            else:
                result.draws += 1

            # Generate PGN for this game
            if self._match_config.pgn_output_path is not None:
                pgn_parts.append(self._game_to_pgn(record, game_idx))

            logger.info(
                "game_end",
                game_num=game_idx + 1,
                result=record.result,
                reason=record.result_reason,
                moves=record.move_count,
                score=f"+{result.wins}-{result.losses}={result.draws}",
            )

        # Calculate Elo estimate
        if result.total_games > 0:
            elo_config = EloConfig(name="match_elo")
            calculator = EloCalculator(elo_config)
            result.elo_estimate = calculator.estimate_elo_difference(
                wins=result.wins,
                losses=result.losses,
                draws=result.draws,
            )

        # Write PGN
        if pgn_parts and self._match_config.pgn_output_path is not None:
            result.pgn = "\n\n".join(pgn_parts)
            self._write_pgn(result.pgn, self._match_config.pgn_output_path)

        logger.info(
            "match_complete",
            wins=result.wins,
            losses=result.losses,
            draws=result.draws,
            win_rate=f"{result.win_rate:.2%}",
            elo_diff=(
                f"{result.elo_estimate.elo_difference:.0f}" if result.elo_estimate else "N/A"
            ),
        )

        return result

    def _play_single_game(
        self,
        model_is_white: bool,
        opening_fen: str,
    ) -> tuple[float, GameRecord]:
        """Play a single game between model and engine.

        Args:
            model_is_white: Whether the model plays white.
            opening_fen: Starting position FEN.

        Returns:
            Tuple of (outcome_from_model_perspective, game_record).
            Outcome: 1.0=model wins, -1.0=model loses, 0.0=draw.

        """
        from src.mcts.evaluator import FNetEvaluator
        from src.mcts.search import MCTS

        # Initialize game state
        state = fen_to_state(opening_fen)
        record = GameRecord(
            model_color="white" if model_is_white else "black",
            opening_fen=opening_fen,
        )

        # Create MCTS for model
        neural_evaluator = FNetEvaluator(
            self._model,
            device=self._device,
            use_fast_path=True,
        )
        mcts = MCTS(evaluator=neural_evaluator, **self._mcts_config)

        # Create engine
        with UCIEngine(self._engine_config) as engine:
            engine.new_game()
            move_history: list[str] = []

            for _move_num in range(self._match_config.max_moves):
                if self._game.is_terminal(state):
                    break

                is_model_turn = (state.current_player == 1 and model_is_white) or (
                    state.current_player == -1 and not model_is_white
                )

                legal_actions = self._game.get_legal_actions(state)
                if not legal_actions:
                    break

                if is_model_turn:
                    # Model plays via MCTS
                    action = mcts.get_action(
                        self._game,
                        state,
                        temperature=0.0,
                        add_noise=False,
                    )
                else:
                    # Engine plays — send full move history for proper
                    # repetition detection and hash table usage
                    fen = state_to_fen(state)
                    engine.set_position(fen, moves=None)
                    best_move_uci, _info = engine.go()
                    action = self._game.string_to_action(best_move_uci, state)

                    if action is None or action not in legal_actions:
                        logger.warning(
                            "engine_illegal_move",
                            move=best_move_uci,
                            legal_count=len(legal_actions),
                        )
                        # Pick first legal move as fallback
                        action = legal_actions[0]

                # Record move
                move_uci = self._game.action_to_string(action, state)
                move_history.append(move_uci)
                record.moves.append(move_uci)

                # Apply move
                state = self._game.apply_action(state, action)
                record.move_count += 1

                # Advance MCTS tree on every move (model + engine)
                # to properly prune the search tree and reuse subtrees
                mcts.advance(action)

        # Determine result
        game_result = self._game.get_result(state)

        if game_result.winner == 0:
            record.result = "1/2-1/2"
            record.result_reason = game_result.reason
            outcome = 0.0
        elif (game_result.winner == 1 and model_is_white) or (
            game_result.winner == -1 and not model_is_white
        ):
            record.result = "1-0" if model_is_white else "0-1"
            record.result_reason = game_result.reason
            outcome = 1.0
        else:
            record.result = "0-1" if model_is_white else "1-0"
            record.result_reason = game_result.reason
            outcome = -1.0

        return outcome, record

    def _game_to_pgn(self, record: GameRecord, game_idx: int) -> str:
        """Convert a game record to PGN format.

        Args:
            record: Game record with moves and result.
            game_idx: Game index in the match.

        Returns:
            PGN string for the game.

        """
        date_str = datetime.now(tz=timezone.utc).strftime("%Y.%m.%d")
        white_name = "AlphaGalerkin" if record.model_color == "white" else "Engine"
        black_name = "AlphaGalerkin" if record.model_color == "black" else "Engine"

        headers = [
            '[Event "AlphaGalerkin Match"]',
            '[Site "Local"]',
            f'[Date "{date_str}"]',
            f'[Round "{game_idx + 1}"]',
            f'[White "{white_name}"]',
            f'[Black "{black_name}"]',
            f'[Result "{record.result}"]',
        ]

        if record.opening_fen != STARTING_FEN:
            headers.append(f'[FEN "{record.opening_fen}"]')
            headers.append('[SetUp "1"]')

        # Format moves (simple UCI notation for now)
        move_text_parts: list[str] = []
        for i, move in enumerate(record.moves):
            if i % 2 == 0:
                move_text_parts.append(f"{i // 2 + 1}.")
            move_text_parts.append(move)

        move_text = " ".join(move_text_parts)
        if record.result:
            move_text += f" {record.result}"

        return "\n".join(headers) + "\n\n" + move_text

    def _write_pgn(self, pgn: str, path: Path) -> None:
        """Write PGN string to file.

        Args:
            pgn: PGN content.
            path: Output file path.

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pgn, encoding="utf-8")
        logger.info("pgn_written", path=str(path))
