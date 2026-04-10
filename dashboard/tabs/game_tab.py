"""Go Game tab for the AlphaGalerkin dashboard.

Provides Human vs AI and AI vs AI modes with variable board size support,
demonstrating zero-shot resolution transfer (model trained on 9×9 plays
13×13 and 19×19 without retraining).

The game infrastructure (model, evaluator, renderer) is loaded lazily on
first use and cached in module-level singletons.  A threading.Lock guards
against concurrent initialisation.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
import numpy as np
import structlog
from numpy.typing import NDArray

from dashboard.config import DEFAULT_CONFIG, GameConfig
from dashboard.utils import device_str, format_exc

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy-initialised module-level singletons
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_loaded = False
_init_error: str | None = None

# The following are set by _ensure_loaded(); None until then.
_model = None
_evaluator = None
_game_manager = None
_renderer = None
_endgame = None
_space_config = None


def _ensure_loaded() -> bool:
    """Load the model and game infrastructure once, thread-safely.

    Returns:
        ``True`` if all components loaded successfully, ``False`` otherwise.

    """
    global _loaded, _init_error
    global _model, _evaluator, _game_manager, _renderer, _endgame, _space_config

    if _loaded:
        return _model is not None

    with _lock:
        if _loaded:  # double-checked locking
            return _model is not None

        try:
            import torch
            from config.board import get_default_space_config  # type: ignore[import]
            from src.endgame import EndgameDetector  # type: ignore[import]
            from src.game_manager import GameManager  # type: ignore[import]
            from src.rendering.board_renderer import BoardRenderer  # type: ignore[import]

            from src.mcts.evaluator import FNetEvaluator  # type: ignore[import]

            device = device_str()
            _space_config = get_default_space_config()
            _renderer = BoardRenderer(_space_config.render)
            _endgame = EndgameDetector(_space_config.endgame)

            hf_space = Path(__file__).parent.parent.parent / "hf_space"
            ckpt_path = hf_space / "checkpoint.pt"

            if ckpt_path.exists() and ckpt_path.stat().st_size > 1000:
                from config.schemas import AlphaGalerkinConfig  # type: ignore[import]
                from src.modeling.model import AlphaGalerkinModel  # type: ignore[import]

                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                raw_cfg = ckpt.get("config", {})
                if isinstance(raw_cfg, dict):
                    config = AlphaGalerkinConfig(**raw_cfg)
                    _model = AlphaGalerkinModel(config.operator)
                    key = "model_state_dict" if "model_state_dict" in ckpt else None
                    _model.load_state_dict(ckpt[key] if key else ckpt)
                    _model.to(device).eval()
                    _evaluator = FNetEvaluator(_model, device=device, use_fast_path=True)
                    logger.info("game_model_loaded", device=device, ckpt=str(ckpt_path))
                else:
                    logger.warning("checkpoint_config_not_dict", ckpt=str(ckpt_path))
            else:
                logger.warning("checkpoint_not_found_or_empty", ckpt=str(ckpt_path))

            _game_manager = GameManager(config=_space_config, evaluator=_evaluator)
            # Mark initialized; _model is not None only when checkpoint loaded.
            # Return _model is not None so that the first call is consistent
            # with the short-circuit path: `if _loaded: return _model is not None`.
            _loaded = True
            return _model is not None

        except Exception as exc:
            _init_error = format_exc(exc, prefix="Game init failed")
            logger.exception("game_tab_init_failed")
            return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fallback_board(n: int, cfg: GameConfig | None = None) -> NDArray[np.uint8]:
    """Return a plain grey placeholder board image.

    Args:
        n: Board size (unused here; kept for signature consistency).
        cfg: GameConfig; uses ``DEFAULT_CONFIG.game`` when *None*.

    Returns:
        Uint8 RGB array of shape (px, px, 3).

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.game
    px = cfg.fallback_board_size_px
    return np.full((px, px, 3), 220, dtype=np.uint8)


def _board_size_choices(cfg: GameConfig | None = None) -> list[tuple[str, int]]:
    """Return dropdown choices for board size selection.

    Args:
        cfg: GameConfig; uses ``DEFAULT_CONFIG.game`` when *None*.

    Returns:
        List of ``(label, value)`` tuples compatible with ``gr.Dropdown``.

    """
    _ensure_loaded()
    if _game_manager is not None:
        return _game_manager.get_board_size_choices()  # type: ignore[union-attr]

    if cfg is None:
        cfg = DEFAULT_CONFIG.game
    return [(f"{s}×{s}", s) for s in cfg.board_sizes]


def _build_session(game, board_size: int, history: list):  # type: ignore[no-untyped-def]
    """Construct a GameSession from current game state.

    Args:
        game: Active ``SimpleGoGame`` instance.
        board_size: Board dimension.
        history: List of moves played so far.

    Returns:
        A ``GameSession`` object.

    """
    from src.game_manager import GameSession  # type: ignore[import]

    return GameSession(
        game=game,
        board_size=board_size,
        komi=_space_config.get_komi(board_size),  # type: ignore[union-attr]
        move_history=history,
        training_board_size=_space_config.training_board_size,  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------
# Human vs AI handlers
# ---------------------------------------------------------------------------


def human_reset(
    board_size: int,
    cfg: GameConfig | None = None,
) -> tuple[list, str, NDArray[np.uint8], str]:
    """Reset the Human vs AI game to a fresh board.

    Args:
        board_size: Desired board size.
        cfg: Optional GameConfig override.

    Returns:
        Tuple of (move_history, status, board_image, score_text).

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.game

    _ensure_loaded()
    logger.info("human_reset", board_size=board_size)

    if _game_manager is None:
        return [], _init_error or "Model unavailable.", _fallback_board(board_size, cfg), ""

    session = _game_manager.create_game(board_size)  # type: ignore[union-attr]
    zs_label = "Zero-shot transfer" if session.is_zero_shot else "Training size"
    status = f"New game. You are Black. Komi {session.komi} ({zs_label})."
    return (
        [],
        status,
        _renderer.render(session.game),  # type: ignore[union-attr]
        _game_manager.get_score_display(session),  # type: ignore[union-attr]
    )


def human_move(
    history: list,
    board_size: int,
    move_text: str,
    cfg: GameConfig | None = None,
) -> tuple[list, str, NDArray[np.uint8], str]:
    """Apply a human move and get the AI response.

    Args:
        history: Current move history list.
        board_size: Active board size.
        move_text: Raw user input (``"row,col"`` or ``"PASS"``).
        cfg: Optional GameConfig override.

    Returns:
        Tuple of (updated_history, status, board_image, score_text).

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.game

    _ensure_loaded()
    logger.info("human_move_input", board_size=board_size, move_text=move_text)

    if _game_manager is None:
        return history, _init_error or "Model not loaded.", _fallback_board(board_size, cfg), ""

    game = _game_manager.replay_history(history, board_size)  # type: ignore[union-attr]

    try:
        move = _game_manager.parse_move(move_text, board_size)  # type: ignore[union-attr]
    except ValueError as exc:
        session = _build_session(game, board_size, history)
        board_img = _renderer.render(game)  # type: ignore[union-attr]
        return (
            history,
            format_exc(exc, prefix="Invalid move"),
            board_img,
            _game_manager.get_score_display(session),  # type: ignore[union-attr]
        )

    if move == "PASS":
        game.play_pass()
        history = [*history, "PASS"]
    else:
        r, c = move
        if not game.play(r, c):
            session = _build_session(game, board_size, history)
            board_img = _renderer.render(game)  # type: ignore[union-attr]
            return (
                history,
                f"Illegal move at ({r}, {c}).",
                board_img,
                _game_manager.get_score_display(session),  # type: ignore[union-attr]
            )
        history = [*history, (r, c)]

    session = _build_session(game, board_size, history)

    if game.is_terminal():
        final = _game_manager.calculate_final_score(session)  # type: ignore[union-attr]
        board_img = _renderer.render(game)  # type: ignore[union-attr]
        score = _game_manager.get_score_display(session)  # type: ignore[union-attr]
        return history, f"Game over. {final}", board_img, score

    if _evaluator is None:
        board_img = _renderer.render(game)  # type: ignore[union-attr]
        score = _game_manager.get_score_display(session)  # type: ignore[union-attr]
        return history, "Move applied. No AI (model not loaded).", board_img, score

    from src.mcts.search import MCTS  # type: ignore[import]

    mcts = MCTS(evaluator=_evaluator, **_game_manager.mcts_kwargs)  # type: ignore[union-attr]
    action = mcts.get_action(game, temperature=cfg.ai_temperature_vs_human, add_noise=False)

    pass_action = board_size * board_size
    last_idx = None
    if action == pass_action:
        game.play_pass()
        history = [*history, "PASS"]
        ai_str = "AI passes"
    else:
        ar, ac = action // board_size, action % board_size
        game.play(ar, ac)
        history = [*history, (ar, ac)]
        ai_str = f"AI plays {_game_manager.format_move(ar, ac, board_size)}"  # type: ignore[union-attr]
        last_idx = action

    session.move_history = history
    board_img = _renderer.render(game, last_idx)  # type: ignore[union-attr]

    score = _game_manager.get_score_display(session)  # type: ignore[union-attr]
    if game.is_terminal():
        final = _game_manager.calculate_final_score(session)  # type: ignore[union-attr]
        return history, f"{ai_str}. Game over. {final}", board_img, score

    logger.info("human_move_complete", ai_move=ai_str, total_moves=len(history))
    return history, ai_str, board_img, score


# ---------------------------------------------------------------------------
# AI vs AI handlers
# ---------------------------------------------------------------------------


def ai_reset(
    board_size: int,
    cfg: GameConfig | None = None,
) -> tuple[list, str, NDArray[np.uint8], str]:
    """Reset the AI vs AI game.

    Args:
        board_size: Desired board size.
        cfg: Optional GameConfig override.

    Returns:
        Tuple of (move_history, status, board_image, score_text).

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.game

    _ensure_loaded()
    logger.info("ai_reset", board_size=board_size)

    if _game_manager is None:
        return [], _init_error or "Model unavailable.", _fallback_board(board_size, cfg), ""

    session = _game_manager.create_game(board_size, is_human_vs_ai=False)  # type: ignore[union-attr]
    zs_label = "Zero-shot transfer" if session.is_zero_shot else "Training size"
    return (
        [],
        f"Ready ({zs_label}). Click 'Next Move'.",
        _renderer.render(session.game),  # type: ignore[union-attr]
        _game_manager.get_score_display(session),  # type: ignore[union-attr]
    )


def ai_step(
    history: list,
    board_size: int,
    cfg: GameConfig | None = None,
) -> tuple[list, str, NDArray[np.uint8], str]:
    """Execute one AI move in AI vs AI mode.

    Args:
        history: Current move history list.
        board_size: Active board size.
        cfg: Optional GameConfig override.

    Returns:
        Tuple of (updated_history, status, board_image, score_text).

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.game

    _ensure_loaded()

    if _game_manager is None:
        return history, _init_error or "Model not loaded.", _fallback_board(board_size, cfg), ""

    game = _game_manager.replay_history(history, board_size)  # type: ignore[union-attr]
    session = _build_session(game, board_size, history)

    if game.is_terminal():
        final = _game_manager.calculate_final_score(session)  # type: ignore[union-attr]
        board_img = _renderer.render(game)  # type: ignore[union-attr]
        return history, f"Game over. {final}", board_img, _game_manager.get_score_display(session)  # type: ignore[union-attr]

    if _evaluator is None:
        board_img = _renderer.render(game)  # type: ignore[union-attr]
        return history, "AI model not loaded.", board_img, _game_manager.get_score_display(session)  # type: ignore[union-attr]

    from src.mcts.search import MCTS  # type: ignore[import]
    from src.tools.gtp import SimpleGoGame  # type: ignore[import]

    mcts = MCTS(evaluator=_evaluator, **_game_manager.mcts_kwargs)  # type: ignore[union-attr]
    action = mcts.get_action(game, temperature=cfg.ai_temperature_self_play, add_noise=False)

    player = "Black" if game.current_player == SimpleGoGame.BLACK else "White"
    pass_action = board_size * board_size
    last_idx = None

    if action == pass_action:
        game.play_pass()
        history = [*history, "PASS"]
        move_str = f"{player} passes"
    else:
        r, c = action // board_size, action % board_size
        game.play(r, c)
        history = [*history, (r, c)]
        move_str = f"{player} plays {_game_manager.format_move(r, c, board_size)}"  # type: ignore[union-attr]
        last_idx = action

    session.move_history = history
    board_img = _renderer.render(game, last_idx)  # type: ignore[union-attr]

    if game.is_terminal():
        final = _game_manager.calculate_final_score(session)  # type: ignore[union-attr]
        return (
            history,
            f"Move {len(history)}: {move_str}. Game over. {final}",
            board_img,
            _game_manager.get_score_display(session),  # type: ignore[union-attr]
        )

    logger.info("ai_step_complete", move=move_str, total_moves=len(history))
    return (
        history,
        f"Move {len(history)}: {move_str}",
        board_img,
        _game_manager.get_score_display(session),  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------
# Gradio tab builder
# ---------------------------------------------------------------------------


def create_game_tab(cfg: GameConfig | None = None) -> None:
    """Create the Go Game tab inside an existing ``gr.Blocks`` context.

    Args:
        cfg: Optional GameConfig override; uses ``DEFAULT_CONFIG.game`` when *None*.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.game

    choices = _board_size_choices(cfg)
    valid_sizes = dict(choices).values()
    default_size = (
        cfg.default_board_size
        if cfg.default_board_size in valid_sizes
        else (choices[0][1] if choices else 9)
    )
    img_height = cfg.board_image_height_px

    with gr.Tab("Go AI"):
        gr.Markdown(
            "## AlphaGalerkin Go AI\n"
            "Play Go against AlphaGalerkin or watch the AI self-play.\n"
            "**Zero-shot transfer**: model trained on "
            f"{cfg.board_sizes[0]}×{cfg.board_sizes[0]} "
            "plays larger boards with no retraining."
        )

        with gr.Tabs():
            # ── Human vs AI ─────────────────────────────────────────────────
            with gr.Tab("Play vs AI"):
                gr.Markdown("You are **Black** (first). Enter moves as `row,col` or `PASS`.")

                with gr.Row():
                    with gr.Column(scale=2):
                        hva_size = gr.Dropdown(
                            choices=choices,
                            value=default_size,
                            label="Board Size",
                            info=f"{cfg.board_sizes[0]}×{cfg.board_sizes[0]}: training"
                            + (
                                f" | {cfg.board_sizes[1]}×{cfg.board_sizes[1]}+"
                                if len(cfg.board_sizes) > 1
                                else ""
                            )
                            + ": zero-shot",
                        )
                        hva_board = gr.Image(
                            label="Board",
                            value=_fallback_board(default_size, cfg),
                            height=img_height,
                        )
                        hva_score = gr.Textbox(label="Game Info", interactive=False)

                    with gr.Column(scale=1):
                        hva_status = gr.Textbox(
                            label="Status",
                            value="Initialising…",
                            interactive=False,
                            lines=2,
                        )
                        hva_input = gr.Textbox(
                            label="Your Move",
                            placeholder="e.g. 4,4 or PASS",
                        )
                        hva_submit = gr.Button("Submit Move", variant="primary")
                        hva_reset = gr.Button("New Game")
                        gr.Markdown(
                            "**Coordinates:** row 0 = top, col 0 = left.  "
                            f"Centre of {cfg.board_sizes[0]}×{cfg.board_sizes[0]} = "
                            f"`{cfg.board_sizes[0] // 2},{cfg.board_sizes[0] // 2}`."
                        )

                hva_history = gr.State([])
                _hva_out = [hva_history, hva_status, hva_board, hva_score]

                hva_size.change(human_reset, [hva_size], _hva_out)
                hva_submit.click(human_move, [hva_history, hva_size, hva_input], _hva_out)
                hva_input.submit(human_move, [hva_history, hva_size, hva_input], _hva_out)
                hva_reset.click(human_reset, [hva_size], _hva_out)

            # ── AI vs AI ─────────────────────────────────────────────────────
            with gr.Tab("Watch AI vs AI"):
                gr.Markdown("Watch AlphaGalerkin play both sides. Click **Next Move** per turn.")

                with gr.Row():
                    with gr.Column(scale=2):
                        _ava_info = (
                            f"{cfg.board_sizes[0]}×{cfg.board_sizes[0]}: training"
                            " | larger: zero-shot"
                        )
                        ava_size = gr.Dropdown(
                            choices=choices,
                            value=default_size,
                            label="Board Size",
                            info=_ava_info,
                        )
                        ava_board = gr.Image(
                            label="Board",
                            value=_fallback_board(default_size, cfg),
                            height=img_height,
                        )
                        ava_score = gr.Textbox(label="Game Info", interactive=False)

                    with gr.Column(scale=1):
                        ava_status = gr.Textbox(
                            label="Status",
                            value="Initialising…",
                            interactive=False,
                            lines=2,
                        )
                        ava_next = gr.Button("Next Move", variant="primary")
                        ava_reset = gr.Button("New Game")

                ava_history = gr.State([])
                _ava_out = [ava_history, ava_status, ava_board, ava_score]

                ava_size.change(ai_reset, [ava_size], _ava_out)
                ava_next.click(ai_step, [ava_history, ava_size], _ava_out)
                ava_reset.click(ai_reset, [ava_size], _ava_out)


__all__ = [
    "_board_size_choices",
    "_ensure_loaded",
    "_fallback_board",
    "ai_reset",
    "ai_step",
    "create_game_tab",
    "human_move",
    "human_reset",
]
