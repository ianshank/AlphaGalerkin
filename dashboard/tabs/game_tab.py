"""Go game tab for AlphaGalerkin dashboard.

Provides Human vs AI and AI vs AI modes with variable board size support,
demonstrating zero-shot resolution transfer (model trained on 9×9 plays 13×13 / 19×19).
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy-initialised globals (populated once in _ensure_loaded)
# ---------------------------------------------------------------------------
_MODEL = None
_EVALUATOR = None
_GAME_MANAGER = None
_RENDERER = None
_ENDGAME = None
_SPACE_CONFIG = None
_LOADED = False


def _ensure_loaded() -> bool:
    """Load model and game infrastructure on first call.  Returns True on success."""
    global _MODEL, _EVALUATOR, _GAME_MANAGER, _RENDERER, _ENDGAME, _SPACE_CONFIG, _LOADED

    if _LOADED:
        return _MODEL is not None

    _LOADED = True
    try:
        import torch
        from config.board import get_default_space_config
        from config.schemas import AlphaGalerkinConfig
        from src.endgame import EndgameDetector
        from src.game_manager import GameManager
        from src.mcts.evaluator import FNetEvaluator
        from src.rendering.board_renderer import BoardRenderer

        HF_SPACE = Path(__file__).parent.parent.parent / "hf_space"
        CKPT = HF_SPACE / "checkpoint.pt"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        _SPACE_CONFIG = get_default_space_config()
        _RENDERER = BoardRenderer(_SPACE_CONFIG.render)
        _ENDGAME = EndgameDetector(_SPACE_CONFIG.endgame)

        if CKPT.exists() and CKPT.stat().st_size > 1000:
            from src.modeling.model import AlphaGalerkinModel

            ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
            cfg = ckpt.get("config", {})
            if isinstance(cfg, dict):
                config = AlphaGalerkinConfig(**cfg)
                _MODEL = AlphaGalerkinModel(config.operator)
                key = "model_state_dict" if "model_state_dict" in ckpt else None
                _MODEL.load_state_dict(ckpt[key] if key else ckpt)
                _MODEL.to(DEVICE).eval()
                _EVALUATOR = FNetEvaluator(_MODEL, device=DEVICE, use_fast_path=True)
                logger.info("game_model_loaded", device=DEVICE)

        _GAME_MANAGER = GameManager(config=_SPACE_CONFIG, evaluator=_EVALUATOR)
        return True

    except Exception as exc:
        logger.warning("game_tab_init_failed", error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Helper: render blank board when model unavailable
# ---------------------------------------------------------------------------


def _fallback_board(board_size: int) -> np.ndarray:
    """Render an empty board without requiring the evaluator."""
    try:
        from src.tools.gtp import SimpleGoGame

        if _RENDERER:
            return _RENDERER.render(SimpleGoGame(board_size))
    except Exception:
        pass
    # Minimal fallback: a white square
    return np.ones((400, 400, 3), dtype=np.uint8) * 240


def _board_size_choices() -> list[tuple[str, int]]:
    _ensure_loaded()
    if _GAME_MANAGER:
        return _GAME_MANAGER.get_board_size_choices()
    return [("9×9 (training)", 9), ("13×13 (zero-shot)", 13), ("19×19 (zero-shot)", 19)]


# ---------------------------------------------------------------------------
# Human vs AI
# ---------------------------------------------------------------------------


def human_reset(board_size: int) -> tuple[list, str, np.ndarray, str]:
    _ensure_loaded()
    if not _GAME_MANAGER:
        return [], "Model unavailable — no MCTS moves.", _fallback_board(board_size), ""
    session = _GAME_MANAGER.create_game(board_size)
    komi = session.komi
    zs = "Zero-shot transfer" if session.is_zero_shot else "Training size"
    return (
        [],
        f"New game. You are Black. Komi {komi} ({zs}).",
        _RENDERER.render(session.game),
        _GAME_MANAGER.get_score_display(session),
    )


def human_move(
    history: list, board_size: int, move_text: str
) -> tuple[list, str, np.ndarray, str]:
    _ensure_loaded()
    if not _GAME_MANAGER:
        return history, "Model not loaded.", _fallback_board(board_size), ""

    from src.game_manager import GameSession
    from src.mcts.search import MCTS
    from src.tools.gtp import SimpleGoGame

    game = _GAME_MANAGER.replay_history(history, board_size)

    try:
        move = _GAME_MANAGER.parse_move(move_text, board_size)
    except ValueError as exc:
        session = GameSession(
            game=game, board_size=board_size,
            komi=_SPACE_CONFIG.get_komi(board_size),
            move_history=history,
            training_board_size=_SPACE_CONFIG.training_board_size,
        )
        return history, f"Invalid move: {exc}", _RENDERER.render(game), _GAME_MANAGER.get_score_display(session)

    if move == "PASS":
        game.play_pass()
        history.append("PASS")
    else:
        r, c = move
        if not game.play(r, c):
            session = GameSession(
                game=game, board_size=board_size,
                komi=_SPACE_CONFIG.get_komi(board_size),
                move_history=history,
                training_board_size=_SPACE_CONFIG.training_board_size,
            )
            return history, f"Illegal move at {r},{c}.", _RENDERER.render(game), _GAME_MANAGER.get_score_display(session)

    session = GameSession(
        game=game, board_size=board_size,
        komi=_SPACE_CONFIG.get_komi(board_size),
        move_history=history,
        training_board_size=_SPACE_CONFIG.training_board_size,
    )

    if game.is_terminal():
        final = _GAME_MANAGER.calculate_final_score(session)
        return history, f"Game over. {final}", _RENDERER.render(game), _GAME_MANAGER.get_score_display(session)

    if not _EVALUATOR:
        return history, "Human move applied. No AI (model not loaded).", _RENDERER.render(game), _GAME_MANAGER.get_score_display(session)

    mcts = MCTS(evaluator=_EVALUATOR, **_GAME_MANAGER.mcts_kwargs)
    action = mcts.get_action(game, temperature=0.0, add_noise=False)

    pass_action = board_size * board_size
    last_idx = None
    if action == pass_action:
        game.play_pass()
        history.append("PASS")
        ai_str = "AI passes"
    else:
        ar, ac = action // board_size, action % board_size
        game.play(ar, ac)
        history.append((ar, ac))
        ai_str = f"AI plays {_GAME_MANAGER.format_move(ar, ac, board_size)}"
        last_idx = action

    session.move_history = history
    if game.is_terminal():
        final = _GAME_MANAGER.calculate_final_score(session)
        return history, f"{ai_str}. Game over. {final}", _RENDERER.render(game, last_idx), _GAME_MANAGER.get_score_display(session)

    return history, ai_str, _RENDERER.render(game, last_idx), _GAME_MANAGER.get_score_display(session)


# ---------------------------------------------------------------------------
# AI vs AI
# ---------------------------------------------------------------------------


def ai_reset(board_size: int) -> tuple[list, str, np.ndarray, str]:
    _ensure_loaded()
    if not _GAME_MANAGER:
        return [], "Model unavailable.", _fallback_board(board_size), ""
    session = _GAME_MANAGER.create_game(board_size, is_human_vs_ai=False)
    zs = "Zero-shot transfer" if session.is_zero_shot else "Training size"
    return (
        [],
        f"Ready ({zs}). Click 'Next Move'.",
        _RENDERER.render(session.game),
        _GAME_MANAGER.get_score_display(session),
    )


def ai_step(history: list, board_size: int) -> tuple[list, str, np.ndarray, str]:
    _ensure_loaded()
    if not _GAME_MANAGER:
        return history, "Model not loaded.", _fallback_board(board_size), ""

    from src.game_manager import GameSession
    from src.mcts.search import MCTS
    from src.tools.gtp import SimpleGoGame

    game = _GAME_MANAGER.replay_history(history, board_size)

    session = GameSession(
        game=game, board_size=board_size,
        komi=_SPACE_CONFIG.get_komi(board_size),
        move_history=history,
        training_board_size=_SPACE_CONFIG.training_board_size,
    )

    if game.is_terminal():
        final = _GAME_MANAGER.calculate_final_score(session)
        return history, f"Game over. {final}", _RENDERER.render(game), _GAME_MANAGER.get_score_display(session)

    if not _EVALUATOR:
        return history, "AI model not loaded.", _RENDERER.render(game), _GAME_MANAGER.get_score_display(session)

    mcts = MCTS(evaluator=_EVALUATOR, **_GAME_MANAGER.mcts_kwargs)
    action = mcts.get_action(game, temperature=0.1, add_noise=False)

    player = "Black" if game.current_player == SimpleGoGame.BLACK else "White"
    pass_action = board_size * board_size
    last_idx = None

    if action == pass_action:
        game.play_pass()
        history.append("PASS")
        move_str = f"{player} passes"
    else:
        r, c = action // board_size, action % board_size
        game.play(r, c)
        history.append((r, c))
        move_str = f"{player} plays {_GAME_MANAGER.format_move(r, c, board_size)}"
        last_idx = action

    session.move_history = history
    if game.is_terminal():
        final = _GAME_MANAGER.calculate_final_score(session)
        return history, f"Move {len(history)}: {move_str}. Game over. {final}", _RENDERER.render(game, last_idx), _GAME_MANAGER.get_score_display(session)

    return history, f"Move {len(history)}: {move_str}", _RENDERER.render(game, last_idx), _GAME_MANAGER.get_score_display(session)


# ---------------------------------------------------------------------------
# Gradio tab builder
# ---------------------------------------------------------------------------


def create_game_tab() -> None:
    """Create the Go Game tab inside an existing gr.Blocks context."""
    choices = _board_size_choices()
    default_size = choices[0][1] if choices else 9

    with gr.Tab("Go AI"):
        gr.Markdown(
            """
## AlphaGalerkin Go AI
Play Go against AlphaGalerkin or watch the AI self-play.
**Zero-shot transfer**: model trained on 9×9 plays 13×13 and 19×19 with no retraining.
"""
        )

        with gr.Tabs():
            # ── Human vs AI ─────────────────────────────────────────────────
            with gr.Tab("Play vs AI"):
                gr.Markdown("You are **Black** (first). Enter moves as `row,col` or `PASS`.")

                with gr.Row():
                    with gr.Column(scale=2):
                        hva_size = gr.Dropdown(
                            choices=choices, value=default_size,
                            label="Board Size",
                            info="9×9: training | 13×13, 19×19: zero-shot",
                        )
                        hva_board = gr.Image(
                            label="Board",
                            value=_fallback_board(default_size),
                            height=460,
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
                            "**Coordinates:** row 0 top, col 0 left.  "
                            "Centre of 9×9 = `4,4`."
                        )

                hva_history = gr.State([])

                hva_size.change(human_reset, [hva_size], [hva_history, hva_status, hva_board, hva_score])
                hva_submit.click(human_move, [hva_history, hva_size, hva_input], [hva_history, hva_status, hva_board, hva_score])
                hva_input.submit(human_move, [hva_history, hva_size, hva_input], [hva_history, hva_status, hva_board, hva_score])
                hva_reset.click(human_reset, [hva_size], [hva_history, hva_status, hva_board, hva_score])

            # ── AI vs AI ─────────────────────────────────────────────────────
            with gr.Tab("Watch AI vs AI"):
                gr.Markdown("Watch AlphaGalerkin play both sides. Click **Next Move** each turn.")

                with gr.Row():
                    with gr.Column(scale=2):
                        ava_size = gr.Dropdown(
                            choices=choices, value=default_size,
                            label="Board Size",
                            info="9×9: training | 13×13, 19×19: zero-shot",
                        )
                        ava_board = gr.Image(
                            label="Board",
                            value=_fallback_board(default_size),
                            height=460,
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

                ava_size.change(ai_reset, [ava_size], [ava_history, ava_status, ava_board, ava_score])
                ava_next.click(ai_step, [ava_history, ava_size], [ava_history, ava_status, ava_board, ava_score])
                ava_reset.click(ai_reset, [ava_size], [ava_history, ava_status, ava_board, ava_score])
