"""Bridge between game-review primitives and the real Go engine / model.

The :mod:`src.analysis` review path historically reconstructed board state with a
naive replay that never removed captured stones, and it never wired a real model
into :class:`~src.analysis.evaluator.PositionEvaluator` (so every evaluation fell
through to the uniform ``win_rate=0.5`` dummy). This module closes both gaps:

* :func:`reconstruct_board` replays a move list through :class:`~src.games.go.GoGame`
  so captures and ko are handled correctly, then projects the result back into the
  reviewer's ``list[list[int]]`` board convention (``1`` = black, ``2`` = white,
  ``0`` = empty, indexed ``board[y][x]``).
* :func:`make_model_evaluator` adapts any MCTS-style evaluator (anything exposing
  ``evaluate(state, legal_actions) -> result`` with ``.value`` / ``.policy``, e.g.
  :class:`~src.mcts.evaluator.FNetEvaluator`) into the ``board_state -> (value,
  policy)`` callable that :class:`PositionEvaluator` expects.
* :func:`build_checkpoint_model_evaluator` wires a trained checkpoint end-to-end.

Coordinate convention
----------------------
The reviewer addresses moves as ``(x, y)`` and the board as ``board[y][x]``. The Go
engine uses a flat ``action = row * board_size + col`` index over an ``np.int8``
array indexed ``board[row, col]``. We map ``row == y`` and ``col == x``, so
``action = y * board_size + x``; the round trip is verified in the tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from src.games.go import BLACK, WHITE, GoGame
from src.games.state import GameState

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Reviewer 2D-list board marks (distinct from the engine's BLACK=1 / WHITE=-1).
BLACK_MARK = 1
WHITE_MARK = 2
EMPTY_MARK = 0

_LOGGER = structlog.get_logger(__name__)


class BoardState(list):  # type: ignore[type-arg]
    """A 2D mark board that also carries the engine's player-to-move.

    Subclasses :class:`list` so it stays a drop-in ``list[list[int]]`` (indexing,
    ``len``, iteration, and the evaluator's list-based cache key all keep working),
    while exposing :attr:`current_player` so the true side-to-move (taken straight
    from the reconstructed engine state) reaches the model evaluator. This avoids
    the stone-parity heuristic, which is unreliable once captures change stone
    counts.
    """

    def __init__(self, rows: Sequence[Sequence[int]], current_player: int = BLACK) -> None:
        super().__init__([list(row) for row in rows])
        self.current_player = current_player


def move_to_action(x: int, y: int, board_size: int) -> int:
    """Convert a reviewer ``(x, y)`` move to a flat Go engine action index.

    Args:
        x: Column coordinate (0-indexed).
        y: Row coordinate (0-indexed).
        board_size: Board dimension.

    Returns:
        Flat action index ``y * board_size + x``.

    """
    return y * board_size + x


def action_to_move(action: int, board_size: int) -> tuple[int, int]:
    """Inverse of :func:`move_to_action`.

    Args:
        action: Flat action index.
        board_size: Board dimension.

    Returns:
        ``(x, y)`` move coordinates.

    """
    return action % board_size, action // board_size


def _color_to_player(color: str) -> int:
    """Map a ``"B"``/``"W"`` move color to the engine's player value."""
    return BLACK if color.upper().startswith("B") else WHITE


def reconstruct_board(
    moves: Sequence[tuple[str, int, int]],
    board_size: int,
    *,
    game: GoGame | None = None,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> BoardState:
    """Replay *moves* through the Go engine, returning a capture-correct board.

    Unlike a naive replay, this removes captured stones and respects the engine's
    legality checks. Each move's stated color is honoured (the side-to-move is set
    explicitly before applying), so non-strictly-alternating move lists still place
    stones of the correct colour. Illegal moves (occupied point, suicide, superko)
    are skipped with a debug log rather than aborting the reconstruction.

    Args:
        moves: Sequence of ``(color, x, y)`` tuples. Out-of-bounds coordinates
            (e.g. a ``(-1, -1)`` pass sentinel) are ignored.
        board_size: Board dimension.
        game: Optional :class:`GoGame` instance (a fresh one is created otherwise).
        logger: Optional logger for skipped-move diagnostics.

    Returns:
        A :class:`BoardState` (``board[y][x]`` marks) whose ``current_player`` is the
        engine's side-to-move after the replay.

    """
    game = game or GoGame()
    log = logger or _LOGGER
    state = game.initial_state(board_size)

    for color, x, y in moves:
        if not (0 <= x < board_size and 0 <= y < board_size):
            # Pass / resign / out-of-bounds sentinel: no stone to place.
            continue

        player = _color_to_player(color)
        # Force the side-to-move so the placed stone and its captures resolve for
        # the stated colour regardless of strict alternation.
        state = GameState(
            board=state.board,
            current_player=player,
            move_number=state.move_number,
            move_history=state.move_history,
            metadata=state.metadata,
        )
        action = move_to_action(x, y, board_size)
        try:
            state = game.apply_action(state, action)
        except ValueError as exc:
            log.debug("review_skip_illegal_move", x=x, y=y, color=color, error=str(exc))
            continue

    return BoardState(
        _board_to_marks(state.board, board_size),
        current_player=state.current_player,
    )


def _board_to_marks(board: NDArray[Any], board_size: int) -> list[list[int]]:
    """Project an engine ``np.int8`` board into the reviewer mark convention."""
    marks = [[EMPTY_MARK] * board_size for _ in range(board_size)]
    for row in range(board_size):
        for col in range(board_size):
            value = int(board[row, col])
            if value == BLACK:
                marks[row][col] = BLACK_MARK
            elif value == WHITE:
                marks[row][col] = WHITE_MARK
    return marks


def _marks_to_board(board_state: Sequence[Sequence[int]], board_size: int) -> NDArray[Any]:
    """Inverse of :func:`_board_to_marks`: marks -> engine ``np.int8`` board."""
    board = np.zeros((board_size, board_size), dtype=np.int8)
    for row in range(board_size):
        for col in range(board_size):
            mark = int(board_state[row][col])
            if mark == BLACK_MARK:
                board[row, col] = BLACK
            elif mark == WHITE_MARK:
                board[row, col] = WHITE
    return board


def _infer_side_to_move(board: NDArray[Any]) -> int:
    """Infer side-to-move from stone parity (black opens; alternation thereafter)."""
    n_black = int(np.count_nonzero(board == BLACK))
    n_white = int(np.count_nonzero(board == WHITE))
    return BLACK if n_black <= n_white else WHITE


def make_model_evaluator(
    mcts_evaluator: Any,
    *,
    game: GoGame | None = None,
) -> Callable[[Any], tuple[float, NDArray[Any]]]:
    """Adapt an MCTS-style evaluator into a ``PositionEvaluator`` model callable.

    The returned callable accepts a reviewer board state (``list[list[int]]`` marks),
    encodes it through :meth:`GoGame.to_tensor`, queries *mcts_evaluator* with the
    legal-action set, and returns ``(value, policy)`` — exactly the contract
    :meth:`PositionEvaluator._process_model_output` consumes (``value`` in
    ``[-1, 1]`` becomes ``win_rate``; ``policy`` is a flat per-intersection array
    whose trailing pass slot the reviewer ignores).

    Args:
        mcts_evaluator: Object exposing ``evaluate(state, legal_actions)`` returning
            a result with ``.value`` (float) and ``.policy`` (array over actions).
        game: Optional :class:`GoGame` instance.

    Returns:
        Callable mapping a board state to ``(value, policy)``.

    """

    def _evaluate(board_state: Any) -> tuple[float, NDArray[Any]]:
        # Fresh engine per call: GoGame carries mutable board-size state, so a
        # shared instance would not be thread-safe across concurrent reviews.
        engine = game or GoGame()
        board_size = len(board_state)
        board = _marks_to_board(board_state, board_size)
        # Prefer the true side-to-move carried by BoardState; fall back to the
        # stone-parity heuristic only when the metadata is absent (e.g. a plain
        # list was passed). Parity is unreliable once captures change counts.
        current_player = getattr(board_state, "current_player", None)
        if current_player is None:
            current_player = _infer_side_to_move(board)
        state = GameState(board=board, current_player=current_player)
        tensor = engine.to_tensor(state).cpu().numpy().astype(np.float32)
        legal_actions = engine.get_legal_actions(state)
        result = mcts_evaluator.evaluate(tensor, legal_actions)
        return float(result.value), np.asarray(result.policy)

    return _evaluate


def build_checkpoint_model_evaluator(
    checkpoint_path: str,
    *,
    device: str = "cpu",
    temperature: float = 1.0,
) -> Callable[[Any], tuple[float, NDArray[Any]]]:
    """Build a model evaluator from a trained checkpoint.

    Loads an :class:`~src.modeling.model.AlphaGalerkinModel` via
    :func:`~src.training.checkpoint.create_model_from_checkpoint`, wraps it in an
    :class:`~src.mcts.evaluator.FNetEvaluator`, and adapts it with
    :func:`make_model_evaluator`.

    Args:
        checkpoint_path: Path to the trained checkpoint.
        device: Device preference (resolved via :func:`src.poc.device.resolve_device`).
        temperature: Policy softmax temperature for the underlying evaluator.

    Returns:
        A ``board_state -> (value, policy)`` callable.

    """
    from src.mcts.evaluator import FNetEvaluator
    from src.poc.device import resolve_device
    from src.training.checkpoint import create_model_from_checkpoint

    resolved = resolve_device(device, context="analysis")
    model, _ = create_model_from_checkpoint(checkpoint_path, device=str(resolved))
    mcts_evaluator = FNetEvaluator(model, device=resolved, temperature=temperature)
    return make_model_evaluator(mcts_evaluator)
