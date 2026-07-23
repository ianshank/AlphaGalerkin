"""Hugging Face Space for AlphaGalerkin.

Demonstrates zero-shot resolution transfer across multiple board sizes.
Model trained on 9x9 generalizes to 13x13 and 19x19 without retraining.

Hosted at: hf.co/spaces/ianshank/alphagalerkin-demo
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import structlog
import torch

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

from config.board import get_default_space_config
from src.endgame import EndgameDetector
from src.game_manager import GameManager, GameSession
from src.rendering.board_renderer import BoardRenderer

from config.schemas import AlphaGalerkinConfig
from src.demos.architecture_demo import create_architecture_demo_tab
from src.demos.benchmark_demo import create_benchmark_demo_tab

# Demo modules from PR #20
from src.demos.physics_demo import create_physics_demo_tab
from src.mcts.evaluator import FNetEvaluator
from src.mcts.search import MCTS
from src.modeling.model import AlphaGalerkinModel
from src.tools.gtp import SimpleGoGame

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger(__name__)

# Configuration
SPACE_CONFIG = get_default_space_config()
MODEL_PATH = Path("checkpoint.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HuggingFace Space ID for runtime checkpoint download
HF_SPACE_ID = "ianshank/alphagalerkin-demo"

# Initialize renderer with coordinate labels enabled
RENDERER = BoardRenderer(SPACE_CONFIG.render)

# Initialize endgame detector for proper game termination
ENDGAME_DETECTOR = EndgameDetector(SPACE_CONFIG.endgame)


def _ensure_checkpoint(path: Path) -> Path:
    """Ensure checkpoint file exists, downloading from Hub if needed.

    HuggingFace Spaces may skip LFS smudge during builds. This function
    downloads the checkpoint at runtime if it's missing or is just a pointer.

    Args:
        path: Expected local path to checkpoint.

    Returns:
        Path to the actual checkpoint file.

    """
    # Check if file exists and is a real checkpoint (not LFS pointer)
    if path.exists():
        size = path.stat().st_size
        # LFS pointer files are typically < 200 bytes
        if size > 1000:
            logger.info("checkpoint_found_local", path=str(path), size=size)
            return path
        else:
            logger.warning(
                "checkpoint_appears_to_be_lfs_pointer",
                path=str(path),
                size=size,
            )

    # Download from HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download

        logger.info("downloading_checkpoint_from_hub", repo_id=HF_SPACE_ID)
        downloaded_path = hf_hub_download(
            repo_id=HF_SPACE_ID,
            filename="checkpoint.pt",
            repo_type="space",
        )
        logger.info("checkpoint_downloaded", path=downloaded_path)
        return Path(downloaded_path)
    except Exception as e:
        logger.warning("checkpoint_download_failed", error=str(e))
        return path


def load_model(path: Path) -> AlphaGalerkinModel | None:
    """Load AlphaGalerkin model from checkpoint.

    Args:
        path: Path to checkpoint file.

    Returns:
        Loaded model or None if loading fails.

    """
    # Ensure checkpoint exists (download from Hub if needed)
    path = _ensure_checkpoint(path)

    if not path.exists():
        logger.warning("checkpoint_not_found", path=str(path))
        return None

    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = checkpoint.get("config", {})

        if isinstance(cfg, dict):
            config = AlphaGalerkinConfig(**cfg)
            model = AlphaGalerkinModel(config.operator)
        else:
            raise ValueError("Could not load config from checkpoint")

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(DEVICE)
        model.eval()

        logger.info("model_loaded", device=DEVICE, path=str(path))
        return model

    except Exception as e:
        logger.exception("model_load_failed", error=str(e))
        return None


# Global instances
MODEL = load_model(MODEL_PATH)
EVALUATOR = FNetEvaluator(MODEL, device=DEVICE, use_fast_path=True) if MODEL else None
GAME_MANAGER = GameManager(
    config=SPACE_CONFIG,
    evaluator=EVALUATOR,
)


# ============ HELPER FUNCTIONS ============


def get_board_size_choices() -> list[tuple[str, int]]:
    """Get board size choices for dropdown.

    Returns:
        List of (label, value) tuples for Gradio dropdown.

    """
    return GAME_MANAGER.get_board_size_choices()


def create_game_state(board_size: int) -> tuple[list, np.ndarray, str]:
    """Create initial game state for a board size.

    Args:
        board_size: Board size to create.

    Returns:
        Tuple of (empty history, board image, score display).

    """
    session = GAME_MANAGER.create_game(board_size)
    board_image = RENDERER.render(session.game)
    score_display = GAME_MANAGER.get_score_display(session)
    return [], board_image, score_display


# ============ HUMAN VS AI MODE ============


def update_game(
    history: list,
    board_size: int,
    input_text: str,
) -> tuple[list, str, np.ndarray, str]:
    """Process human move and get AI response.

    Args:
        history: Current move history.
        board_size: Current board size.
        input_text: User's move input.

    Returns:
        Tuple of (updated history, status, board image, score display).

    """
    if not MODEL:
        game = SimpleGoGame(board_size)
        return (
            history,
            "Error: Model failed to load.",
            RENDERER.render(game),
            "",
        )

    game = GAME_MANAGER.replay_history(history, board_size)

    # Parse human move
    try:
        move = GAME_MANAGER.parse_move(input_text, board_size)
    except ValueError as e:
        session = GameSession(
            game=game,
            board_size=board_size,
            komi=SPACE_CONFIG.get_komi(board_size),
            move_history=history,
            training_board_size=SPACE_CONFIG.training_board_size,
        )
        return (
            history,
            f"Warning: {str(e)}",
            RENDERER.render(game),
            GAME_MANAGER.get_score_display(session),
        )

    # Apply human move
    if move == "PASS":
        game.play_pass()
        history.append("PASS")
    else:
        r, c = move
        if game.play(r, c):
            history.append((r, c))
        else:
            session = GameSession(
                game=game,
                board_size=board_size,
                komi=SPACE_CONFIG.get_komi(board_size),
                move_history=history,
                training_board_size=SPACE_CONFIG.training_board_size,
            )
            return (
                history,
                f"Error: Illegal move at {r},{c}",
                RENDERER.render(game),
                GAME_MANAGER.get_score_display(session),
            )

    session = GameSession(
        game=game,
        board_size=board_size,
        komi=SPACE_CONFIG.get_komi(board_size),
        move_history=history,
        training_board_size=SPACE_CONFIG.training_board_size,
    )

    # Check game over after human move
    if game.is_terminal():
        final = GAME_MANAGER.calculate_final_score(session)
        return (
            history,
            f"Game Over. {final}",
            RENDERER.render(game),
            GAME_MANAGER.get_score_display(session),
        )

    # AI Move
    mcts = MCTS(evaluator=EVALUATOR, **GAME_MANAGER.mcts_kwargs)
    action = mcts.get_action(game, temperature=0.0, add_noise=False)

    # Check if we should override MCTS action to pass (endgame detection)
    human_just_passed = move == "PASS"
    if human_just_passed and ENDGAME_DETECTOR.should_override_to_pass(
        game, action, human_just_passed
    ):
        original_action = action
        action = ENDGAME_DETECTOR.get_pass_action(board_size)
        logger.info(
            "endgame_override_applied",
            original_action=original_action,
            new_action=action,
        )

    last_move_idx = None
    pass_action = board_size * board_size
    if action == pass_action:
        game.play_pass()
        history.append("PASS")
        ai_move_str = "Pass"
    else:
        ai_r = action // board_size
        ai_c = action % board_size
        game.play(ai_r, ai_c)
        history.append((ai_r, ai_c))
        ai_move_str = GAME_MANAGER.format_move(ai_r, ai_c, board_size)
        last_move_idx = action

    session.move_history = history

    # Check game over after AI move
    if game.is_terminal():
        final = GAME_MANAGER.calculate_final_score(session)
        return (
            history,
            f"AI played: {ai_move_str}. Game Over. {final}",
            RENDERER.render(game, last_move_idx),
            GAME_MANAGER.get_score_display(session),
        )

    return (
        history,
        f"AI played: {ai_move_str}",
        RENDERER.render(game, last_move_idx),
        GAME_MANAGER.get_score_display(session),
    )


def reset_game(board_size: int) -> tuple[list, str, np.ndarray, str]:
    """Reset the game state.

    Args:
        board_size: Board size for new game.

    Returns:
        Tuple of (empty history, status, board image, score display).

    """
    session = GAME_MANAGER.create_game(board_size)
    komi_info = f"Komi: {session.komi}"
    transfer_info = "Zero-shot transfer" if session.is_zero_shot else "Training size"

    logger.info(
        "game_reset",
        board_size=board_size,
        komi=session.komi,
        is_zero_shot=session.is_zero_shot,
    )

    return (
        [],
        f"Game Reset. You are Black (first). {komi_info} ({transfer_info})",
        RENDERER.render(session.game),
        GAME_MANAGER.get_score_display(session),
    )


def on_board_size_change(board_size: int) -> tuple[list, str, np.ndarray, str]:
    """Handle board size change from dropdown.

    Args:
        board_size: New board size selected.

    Returns:
        Reset game state for new board size.

    """
    logger.info("board_size_changed", new_size=board_size)
    return reset_game(board_size)


# ============ AI VS AI MODE ============


def ai_vs_ai_step(
    history: list,
    board_size: int,
) -> tuple[list, str, np.ndarray, str]:
    """Execute one AI move in AI vs AI mode.

    Args:
        history: Current move history.
        board_size: Current board size.

    Returns:
        Tuple of (updated history, status, board image, score display).

    """
    if not MODEL:
        game = SimpleGoGame(board_size)
        return (
            history,
            "Error: Model failed to load.",
            RENDERER.render(game),
            "",
        )

    game = GAME_MANAGER.replay_history(history, board_size)

    # Check if game is already over
    if game.is_terminal():
        session = GameSession(
            game=game,
            board_size=board_size,
            komi=SPACE_CONFIG.get_komi(board_size),
            move_history=history,
            training_board_size=SPACE_CONFIG.training_board_size,
        )
        final = GAME_MANAGER.calculate_final_score(session)
        return (
            history,
            f"Game Over. {final}",
            RENDERER.render(game),
            GAME_MANAGER.get_score_display(session),
        )

    # Get AI move
    mcts = MCTS(evaluator=EVALUATOR, **GAME_MANAGER.mcts_kwargs)
    action = mcts.get_action(game, temperature=0.1, add_noise=False)

    last_move_idx = None
    current_player = "Black" if game.current_player == SimpleGoGame.BLACK else "White"

    if action == board_size * board_size:  # Pass
        game.play_pass()
        history.append("PASS")
        move_str = f"{current_player} passes"
    else:
        r = action // board_size
        c = action % board_size
        game.play(r, c)
        history.append((r, c))
        formatted_move = GAME_MANAGER.format_move(r, c, board_size)
        move_str = f"{current_player} plays {formatted_move}"
        last_move_idx = action

    session = GameSession(
        game=game,
        board_size=board_size,
        komi=SPACE_CONFIG.get_komi(board_size),
        move_history=history,
        training_board_size=SPACE_CONFIG.training_board_size,
    )

    # Check game over
    if game.is_terminal():
        final = GAME_MANAGER.calculate_final_score(session)
        return (
            history,
            f"Move {len(history)}: {move_str}. Game Over. {final}",
            RENDERER.render(game, last_move_idx),
            GAME_MANAGER.get_score_display(session),
        )

    return (
        history,
        f"Move {len(history)}: {move_str}",
        RENDERER.render(game, last_move_idx),
        GAME_MANAGER.get_score_display(session),
    )


def ai_vs_ai_reset(board_size: int) -> tuple[list, str, np.ndarray, str, bool]:
    """Reset AI vs AI game.

    Args:
        board_size: Board size for new game.

    Returns:
        Tuple of (empty history, status, board image, score display, autoplay=False).

    """
    session = GAME_MANAGER.create_game(board_size, is_human_vs_ai=False)
    transfer_info = "Zero-shot transfer" if session.is_zero_shot else "Training size"

    return (
        [],
        f"Ready ({transfer_info}). Click 'Next Move' or toggle 'Auto-Play'",
        RENDERER.render(session.game),
        GAME_MANAGER.get_score_display(session),
        False,
    )


def ai_vs_ai_auto_step(
    history: list,
    board_size: int,
    is_playing: bool,
) -> tuple[list, str, np.ndarray, str, bool]:
    """Auto-play step for AI vs AI mode.

    Args:
        history: Current move history.
        board_size: Current board size.
        is_playing: Whether auto-play is active.

    Returns:
        Tuple of (history, status, board image, score display, continue_playing).

    """
    if not is_playing:
        game = GAME_MANAGER.replay_history(history, board_size)
        session = GameSession(
            game=game,
            board_size=board_size,
            komi=SPACE_CONFIG.get_komi(board_size),
            move_history=history,
            training_board_size=SPACE_CONFIG.training_board_size,
        )
        return (
            history,
            "Auto-play paused",
            RENDERER.render(game),
            GAME_MANAGER.get_score_display(session),
            False,
        )

    if not MODEL:
        game = SimpleGoGame(board_size)
        return (
            history,
            "Error: Model failed to load.",
            RENDERER.render(game),
            "",
            False,
        )

    game = GAME_MANAGER.replay_history(history, board_size)

    # Check if game is already over
    if game.is_terminal():
        session = GameSession(
            game=game,
            board_size=board_size,
            komi=SPACE_CONFIG.get_komi(board_size),
            move_history=history,
            training_board_size=SPACE_CONFIG.training_board_size,
        )
        final = GAME_MANAGER.calculate_final_score(session)
        return (
            history,
            f"Game Over. {final}",
            RENDERER.render(game),
            GAME_MANAGER.get_score_display(session),
            False,
        )

    # Get AI move
    mcts = MCTS(evaluator=EVALUATOR, **GAME_MANAGER.mcts_kwargs)
    action = mcts.get_action(game, temperature=0.1, add_noise=False)

    last_move_idx = None
    current_player = "Black" if game.current_player == SimpleGoGame.BLACK else "White"

    if action == board_size * board_size:  # Pass
        game.play_pass()
        history.append("PASS")
        move_str = f"{current_player} passes"
    else:
        r = action // board_size
        c = action % board_size
        game.play(r, c)
        history.append((r, c))
        formatted_move = GAME_MANAGER.format_move(r, c, board_size)
        move_str = f"{current_player} plays {formatted_move}"
        last_move_idx = action

    session = GameSession(
        game=game,
        board_size=board_size,
        komi=SPACE_CONFIG.get_komi(board_size),
        move_history=history,
        training_board_size=SPACE_CONFIG.training_board_size,
    )

    # Check game over
    if game.is_terminal():
        final = GAME_MANAGER.calculate_final_score(session)
        return (
            history,
            f"Move {len(history)}: {move_str}. Game Over. {final}",
            RENDERER.render(game, last_move_idx),
            GAME_MANAGER.get_score_display(session),
            False,  # Stop auto-play when game ends
        )

    # Continue playing
    return (
        history,
        f"Playing: Move {len(history)}: {move_str}",
        RENDERER.render(game, last_move_idx),
        GAME_MANAGER.get_score_display(session),
        True,  # Continue auto-play
    )


def on_ai_board_size_change(
    board_size: int,
) -> tuple[list, str, np.ndarray, str, bool]:
    """Handle board size change in AI vs AI mode.

    Args:
        board_size: New board size selected.

    Returns:
        Reset game state for new board size.

    """
    logger.info("ai_vs_ai_board_size_changed", new_size=board_size)
    return ai_vs_ai_reset(board_size)


# ============ GRADIO UI ============

with gr.Blocks(title="AlphaGalerkin Go Demo") as demo:
    gr.Markdown("# AlphaGalerkin Go Demo")
    gr.Markdown(
        "Play Go against AlphaGalerkin or watch AI vs AI. "
        "**Zero-shot transfer**: Model trained on 9x9, generalizes to 13x13 and 19x19."
    )

    with gr.Tabs():
        # ===== TAB 1: Human vs AI =====
        with gr.TabItem("Play vs AI"):
            gr.Markdown("### You are Black. Enter moves as `row,col` (0-indexed) or `PASS`")

            with gr.Row():
                with gr.Column(scale=2):
                    # Board size selector
                    board_size_selector = gr.Dropdown(
                        choices=get_board_size_choices(),
                        value=SPACE_CONFIG.default_board_size,
                        label="Board Size",
                        info="9×9: Training | 13×13, 19×19: Zero-shot transfer",
                    )

                    board_img = gr.Image(
                        label="Board",
                        value=RENDERER.render(SimpleGoGame(SPACE_CONFIG.default_board_size)),
                        interactive=False,
                        height=480,
                    )
                    score_display = gr.Textbox(
                        label="Game Info",
                        value=GAME_MANAGER.get_score_display(
                            GAME_MANAGER.create_game(SPACE_CONFIG.default_board_size)
                        ),
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    status = gr.Textbox(
                        label="Status",
                        value="Ready to play. You are Black (first).",
                        interactive=False,
                        lines=2,
                    )
                    move_input = gr.Textbox(
                        label="Your Move",
                        placeholder="row,col (e.g., 4,4) or PASS",
                    )
                    submit_btn = gr.Button("Submit Move", variant="primary")
                    reset_btn = gr.Button("Reset Game")

                    gr.Markdown("---")
                    gr.Markdown("### Coordinate Guide")
                    gr.Markdown(
                        "- **Row**: 0 at top, increases downward\n"
                        "- **Col**: 0 at left, increases rightward\n"
                        "- **Example**: Center of 9×9 = `4,4`\n"
                        "- **Perimeter labels**: Letters (A-T) + Numbers (1-19)"
                    )

            game_history = gr.State([])

            # Event handlers
            board_size_selector.change(
                on_board_size_change,
                inputs=[board_size_selector],
                outputs=[game_history, status, board_img, score_display],
            )

            submit_btn.click(
                update_game,
                inputs=[game_history, board_size_selector, move_input],
                outputs=[game_history, status, board_img, score_display],
            )

            move_input.submit(
                update_game,
                inputs=[game_history, board_size_selector, move_input],
                outputs=[game_history, status, board_img, score_display],
            )

            reset_btn.click(
                reset_game,
                inputs=[board_size_selector],
                outputs=[game_history, status, board_img, score_display],
            )

        # ===== TAB 2: AI vs AI =====
        with gr.TabItem("Watch AI vs AI"):
            gr.Markdown("### Watch the AI play against itself")

            with gr.Row():
                with gr.Column(scale=2):
                    # Board size selector for AI vs AI
                    ai_board_size_selector = gr.Dropdown(
                        choices=get_board_size_choices(),
                        value=SPACE_CONFIG.default_board_size,
                        label="Board Size",
                        info="9×9: Training | 13×13, 19×19: Zero-shot transfer",
                    )

                    ai_board_img = gr.Image(
                        label="Board",
                        value=RENDERER.render(SimpleGoGame(SPACE_CONFIG.default_board_size)),
                        interactive=False,
                        height=480,
                    )
                    ai_score_display = gr.Textbox(
                        label="Game Info",
                        value=GAME_MANAGER.get_score_display(
                            GAME_MANAGER.create_game(SPACE_CONFIG.default_board_size)
                        ),
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    ai_status = gr.Textbox(
                        label="Status",
                        value="Ready. Click 'Next Move' or toggle 'Auto-Play'",
                        interactive=False,
                        lines=2,
                    )
                    step_btn = gr.Button("Next Move", variant="primary")

                    gr.Markdown("---")
                    gr.Markdown("### Auto-Play")
                    auto_play_checkbox = gr.Checkbox(
                        label="Auto-Play (toggle on/off)",
                        value=False,
                    )
                    speed_slider = gr.Slider(
                        minimum=0.5,
                        maximum=3.0,
                        value=1.0,
                        step=0.5,
                        label="Speed (seconds between moves)",
                    )
                    ai_reset_btn = gr.Button("Reset Game")

                    gr.Markdown("---")
                    gr.Markdown("*Toggle Auto-Play to watch continuously.*")

            ai_game_history = gr.State([])

            # Event handlers for AI vs AI
            ai_board_size_selector.change(
                on_ai_board_size_change,
                inputs=[ai_board_size_selector],
                outputs=[
                    ai_game_history,
                    ai_status,
                    ai_board_img,
                    ai_score_display,
                    auto_play_checkbox,
                ],
            )

            step_btn.click(
                ai_vs_ai_step,
                inputs=[ai_game_history, ai_board_size_selector],
                outputs=[ai_game_history, ai_status, ai_board_img, ai_score_display],
            )

            auto_play_checkbox.change(
                ai_vs_ai_auto_step,
                inputs=[ai_game_history, ai_board_size_selector, auto_play_checkbox],
                outputs=[
                    ai_game_history,
                    ai_status,
                    ai_board_img,
                    ai_score_display,
                    auto_play_checkbox,
                ],
                every=1.0,
            )

            ai_reset_btn.click(
                ai_vs_ai_reset,
                inputs=[ai_board_size_selector],
                outputs=[
                    ai_game_history,
                    ai_status,
                    ai_board_img,
                    ai_score_display,
                    auto_play_checkbox,
                ],
            )

        # ===== TAB 3: Physics Demo (PR #20) =====
        # Note: Physics demo does not use the Go model - it generates ground truth
        # using the Poisson solver. A separate physics model would be needed for
        # neural operator predictions.
        create_physics_demo_tab(model=None, device=DEVICE)

        # ===== TAB 4: Benchmark Demo (PR #20) =====
        create_benchmark_demo_tab()

        # ===== TAB 5: Architecture Demo (PR #20) =====
        create_architecture_demo_tab(model=MODEL, device=DEVICE)

        # ===== TAB 6: About =====
        with gr.TabItem("About"):
            gr.Markdown(
                """
## About AlphaGalerkin

AlphaGalerkin is a resolution-independent neural network for Go
that demonstrates zero-shot transfer across board sizes using
Continuous Operator Learning with Galerkin Transformers and FNet.

**Developer:** Ian Cruickshank

### Key Innovation

The model achieves zero-shot resolution transfer by learning the underlying
dynamics of Go rather than memorizing discrete board positions. A network
trained on 9x9 boards generalizes directly to 13x13 and 19x19 without retraining.

| Board Size | Type | Komi |
|------------|------|------|
| 9x9 | Training size | 5.5 |
| 13x13 | Zero-shot transfer | 6.5 |
| 19x19 | Zero-shot transfer | 7.5 |

### Technical Architecture

- **Galerkin Attention**: O(N) complexity via Petrov-Galerkin projection
- **FNet Mixing**: FFT-based token mixing for efficient MCTS rollouts
- **Fourier Positional Encoding**: Resolution-independent coordinate representation
- **Monte Carlo Tree Search**: Policy-guided search for move selection

### How to Play

1. Select a board size from the dropdown
2. Enter moves as `row,col` (0-indexed from top-left)
3. Use `PASS` to pass your turn
4. The AI responds automatically

**Coordinate System:** Row 0 is at the top (increases downward),
Column 0 is at the left (increases rightward). The board perimeter
displays letters (A-T) and numbers (1-19) for reference.

### Scoring

Uses simplified Chinese rules: Score = Stones + Captures + Komi (White).
Game ends after two consecutive passes.

---
Built with Gradio, PyTorch, and the AlphaGalerkin framework.
            """
            )

if __name__ == "__main__":
    demo.launch()
