"""Hugging Face Space for AlphaGalerkin.

Hosted at: hf.co/spaces/ianshank/alphagalerkin-demo
"""

import logging
import sys
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Ensure local imports work
sys.path.append(str(Path(__file__).parent))

from config.schemas import AlphaGalerkinConfig
from src.mcts.evaluator import FNetEvaluator
from src.mcts.search import MCTS
from src.modeling.model import AlphaGalerkinModel
from src.tools.gtp import SimpleGoGame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = Path("checkpoint.pt")
device = "cpu"  # Force CPU for HF Spaces (unless GPU is available)
if torch.cuda.is_available():
    device = "cuda"

# Constants
BOARD_SIZE = 9
KOMI = 6.5  # Standard komi for 9x9


def load_model(path: Path) -> AlphaGalerkinModel:
    """Load AlphaGalerkin model from checkpoint."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    # Extract config
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

    model.to(device)
    model.eval()
    return model


# Global model instance
try:
    MODEL = load_model(model_path)
    logger.info("Model loaded successfully.")

    # Initialize Evaluator
    EVALUATOR = FNetEvaluator(MODEL, device=device, use_fast_path=True)

    # MCTS Config - modest for web demo speed
    MCTS_KWARGS = {
        "n_simulations": 60,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.03,
        "dirichlet_epsilon": 0.0,
    }
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    MODEL = None
    EVALUATOR = None


# ============ HELPER FUNCTIONS ============


def replay_history(history: list, size: int = BOARD_SIZE) -> SimpleGoGame:
    """Reconstruct game state from move history."""
    game = SimpleGoGame(size)
    for move in history:
        if move == "PASS":
            game.play_pass()
        else:
            r, c = move
            game.play(r, c)
    return game


def get_score_display(game: SimpleGoGame) -> str:
    """Return formatted score string for live display."""
    # Get captures from the game's captures dict
    black_captures = game.captures.get(SimpleGoGame.BLACK, 0)
    white_captures = game.captures.get(SimpleGoGame.WHITE, 0)

    move_count = len(game.move_history)
    current = "Black" if game.current_player == SimpleGoGame.BLACK else "White"

    return (
        f"⚫ Black captures: {black_captures} | ⚪ White captures: {white_captures} "
        f"| Move: {move_count} | {current} to play"
    )


def calculate_final_score(game: SimpleGoGame, komi: float = KOMI) -> str:
    """Calculate and format end-game score."""
    # Count stones on board + captures
    black_stones = (game.board == SimpleGoGame.BLACK).sum()
    white_stones = (game.board == SimpleGoGame.WHITE).sum()

    black_captures = game.captures.get(SimpleGoGame.BLACK, 0)
    white_captures = game.captures.get(SimpleGoGame.WHITE, 0)

    # Simplified scoring: stones + captures
    black_score = float(black_stones + black_captures)
    white_score = float(white_stones + white_captures + komi)

    if black_score > white_score:
        margin = black_score - white_score
        return f"🏆 Black wins by {margin:.1f} points! (B: {black_score:.1f}, W: {white_score:.1f})"
    elif white_score > black_score:
        margin = white_score - black_score
        return f"🏆 White wins by {margin:.1f} points! (B: {black_score:.1f}, W: {white_score:.1f})"
    else:
        return f"🤝 Draw! (B: {black_score:.1f}, W: {white_score:.1f})"


def plot_board(game: SimpleGoGame, last_move: int = None) -> np.ndarray:
    """Render the board using Matplotlib."""
    size = game.board_size

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_facecolor("#e3c586")  # Wood-like color

    # Draw grid
    for i in range(size):
        ax.plot([i, i], [0, size - 1], color="black", linewidth=1, zorder=1)
        ax.plot([0, size - 1], [i, i], color="black", linewidth=1, zorder=1)

    # Draw stars (hoshi)
    if size == 19:
        stars = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
    elif size == 13:
        stars = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
    elif size == 9:
        stars = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
    else:
        stars = []

    for x, y in stars:
        ax.scatter(x, y, s=20, color="black", zorder=2)

    # Draw stones
    for r in range(size):
        for c in range(size):
            p = game.board[r, c]
            if p == SimpleGoGame.BLACK:
                circle = plt.Circle((c, r), 0.45, color="black", zorder=3)
                ax.add_patch(circle)
            elif p == SimpleGoGame.WHITE:
                circle = plt.Circle((c, r), 0.45, color="white", ec="black", zorder=3)
                ax.add_patch(circle)

            # Mark last move
            if last_move is not None:
                lr = last_move // size
                lc = last_move % size
                if lr == r and lc == c:
                    ax.scatter(c, r, s=20, color="red", marker="x", zorder=4)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    # Convert to image
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return image


# ============ HUMAN VS AI MODE ============


def update_game(
    history: list, input_text: str
) -> tuple[list, str, np.ndarray, str]:
    """Process human move and get AI response."""
    if not MODEL:
        return history, "Model failed to load.", None, ""

    game = replay_history(history)
    input_text = input_text.strip().upper()

    # Parse human move
    if input_text == "PASS":
        game.play_pass()
        history.append("PASS")
    else:
        try:
            parts = input_text.split(",")
            if len(parts) == 2:
                r, c = int(parts[0]), int(parts[1])
                if game.play(r, c):
                    history.append((r, c))
                else:
                    return (
                        history,
                        f"❌ Illegal move: {r},{c}",
                        plot_board(game),
                        get_score_display(game),
                    )
            else:
                return (
                    history,
                    "⚠️ Invalid format. Use 'row,col' e.g. '3,3' or 'PASS'",
                    plot_board(game),
                    get_score_display(game),
                )
        except ValueError:
            return (
                history,
                "⚠️ Invalid format. Use numbers 'row,col'",
                plot_board(game),
                get_score_display(game),
            )

    # Check game over after human move
    if game.is_terminal():
        final = calculate_final_score(game)
        return history, f"Game Over! {final}", plot_board(game), get_score_display(game)

    # AI Move
    mcts = MCTS(evaluator=EVALUATOR, **MCTS_KWARGS)
    action = mcts.get_action(game, temperature=0.0, add_noise=False)

    last_move_idx = None
    if action == BOARD_SIZE * BOARD_SIZE:  # Pass
        game.play_pass()
        history.append("PASS")
        ai_move_str = "Pass"
    else:
        ai_r = action // BOARD_SIZE
        ai_c = action % BOARD_SIZE
        game.play(ai_r, ai_c)
        history.append((ai_r, ai_c))
        ai_move_str = f"{ai_r},{ai_c}"
        last_move_idx = action

    # Check game over after AI move
    if game.is_terminal():
        final = calculate_final_score(game)
        return (
            history,
            f"AI played: {ai_move_str}. Game Over! {final}",
            plot_board(game, last_move_idx),
            get_score_display(game),
        )

    return (
        history,
        f"🤖 AI played: {ai_move_str}",
        plot_board(game, last_move_idx),
        get_score_display(game),
    )


def reset_game() -> tuple[list, str, np.ndarray, str]:
    """Reset the game state."""
    game = SimpleGoGame(BOARD_SIZE)
    return [], "♟️ Game Reset. You are Black (first).", plot_board(game), get_score_display(game)


# ============ AI VS AI MODE ============


def ai_vs_ai_step(history: list) -> tuple[list, str, np.ndarray, str]:
    """Execute one AI move."""
    if not MODEL:
        game = SimpleGoGame(BOARD_SIZE)
        return history, "❌ Model failed to load.", plot_board(game), get_score_display(game)

    game = replay_history(history)

    # Check if game is already over
    if game.is_terminal():
        final = calculate_final_score(game)
        return history, f"🏁 Game Over! {final}", plot_board(game), get_score_display(game)

    # Get AI move
    mcts = MCTS(evaluator=EVALUATOR, **MCTS_KWARGS)
    action = mcts.get_action(game, temperature=0.1, add_noise=False)

    last_move_idx = None
    current_player = "Black" if game.current_player == SimpleGoGame.BLACK else "White"

    if action == BOARD_SIZE * BOARD_SIZE:  # Pass
        game.play_pass()
        history.append("PASS")
        move_str = f"{current_player} passes"
    else:
        r = action // BOARD_SIZE
        c = action % BOARD_SIZE
        game.play(r, c)
        history.append((r, c))
        move_str = f"{current_player} plays {r},{c}"
        last_move_idx = action

    # Check game over
    if game.is_terminal():
        final = calculate_final_score(game)
        return (
            history,
            f"Move {len(history)}: {move_str}. 🏁 {final}",
            plot_board(game, last_move_idx),
            get_score_display(game),
        )

    return (
        history,
        f"Move {len(history)}: {move_str}",
        plot_board(game, last_move_idx),
        get_score_display(game),
    )


def ai_vs_ai_reset() -> tuple[list, str, np.ndarray, str, bool]:
    """Reset AI vs AI game."""
    game = SimpleGoGame(BOARD_SIZE)
    return (
        [],
        "🔄 Ready. Click 'Next Move' or toggle 'Auto-Play'",
        plot_board(game),
        get_score_display(game),
        False,
    )


def ai_vs_ai_auto_step(
    history: list, is_playing: bool
) -> tuple[list, str, np.ndarray, str, bool]:
    """Auto-play step - returns updated state and whether to continue."""
    if not is_playing:
        game = replay_history(history)
        return history, "⏸️ Auto-play paused", plot_board(game), get_score_display(game), False

    if not MODEL:
        game = SimpleGoGame(BOARD_SIZE)
        return history, "❌ Model failed to load.", plot_board(game), get_score_display(game), False

    game = replay_history(history)

    # Check if game is already over
    if game.is_terminal():
        final = calculate_final_score(game)
        return history, f"🏁 Game Over! {final}", plot_board(game), get_score_display(game), False

    # Get AI move
    mcts = MCTS(evaluator=EVALUATOR, **MCTS_KWARGS)
    action = mcts.get_action(game, temperature=0.1, add_noise=False)

    last_move_idx = None
    current_player = "Black" if game.current_player == SimpleGoGame.BLACK else "White"

    if action == BOARD_SIZE * BOARD_SIZE:  # Pass
        game.play_pass()
        history.append("PASS")
        move_str = f"{current_player} passes"
    else:
        r = action // BOARD_SIZE
        c = action % BOARD_SIZE
        game.play(r, c)
        history.append((r, c))
        move_str = f"{current_player} plays {r},{c}"
        last_move_idx = action

    # Check game over
    if game.is_terminal():
        final = calculate_final_score(game)
        return (
            history,
            f"Move {len(history)}: {move_str}. 🏁 {final}",
            plot_board(game, last_move_idx),
            get_score_display(game),
            False,
        )

    # Continue playing
    return (
        history,
        f"▶️ Move {len(history)}: {move_str}",
        plot_board(game, last_move_idx),
        get_score_display(game),
        True,
    )


# ============ GRADIO UI ============

with gr.Blocks(title="AlphaGalerkin Go Demo") as demo:
    gr.Markdown("# ⚫ AlphaGalerkin Go Demo ⚪")
    gr.Markdown("Watch the AI play Go (9x9) or challenge it yourself!")

    with gr.Tabs():
        # ===== TAB 1: Human vs AI =====
        with gr.TabItem("🎮 Play vs AI"):
            gr.Markdown("### You are Black. Enter moves as `row,col` (e.g., `3,3`) or `PASS`")

            with gr.Row():
                with gr.Column(scale=2):
                    board_img = gr.Image(
                        label="Board",
                        value=plot_board(SimpleGoGame(BOARD_SIZE)),
                        interactive=False,
                        height=400,
                    )
                    score_display = gr.Textbox(
                        label="Score",
                        value=get_score_display(SimpleGoGame(BOARD_SIZE)),
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    status = gr.Textbox(
                        label="Status",
                        value="♟️ Ready to play. You are Black (first).",
                        interactive=False,
                        lines=2,
                    )
                    move_input = gr.Textbox(
                        label="Your Move", placeholder="row,col (e.g. 3,3) or PASS"
                    )
                    submit_btn = gr.Button("▶️ Submit Move", variant="primary")
                    reset_btn = gr.Button("🔄 Reset Game")

            game_history = gr.State([])

            submit_btn.click(
                update_game,
                inputs=[game_history, move_input],
                outputs=[game_history, status, board_img, score_display],
            )

            move_input.submit(
                update_game,
                inputs=[game_history, move_input],
                outputs=[game_history, status, board_img, score_display],
            )

            reset_btn.click(
                reset_game, inputs=[], outputs=[game_history, status, board_img, score_display]
            )

        # ===== TAB 2: AI vs AI =====
        with gr.TabItem("🤖 Watch AI vs AI"):
            gr.Markdown("### Watch the AI play against itself!")

            with gr.Row():
                with gr.Column(scale=2):
                    ai_board_img = gr.Image(
                        label="Board",
                        value=plot_board(SimpleGoGame(BOARD_SIZE)),
                        interactive=False,
                        height=400,
                    )
                    ai_score_display = gr.Textbox(
                        label="Score",
                        value=get_score_display(SimpleGoGame(BOARD_SIZE)),
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    ai_status = gr.Textbox(
                        label="Status",
                        value="🔄 Ready. Click 'Next Move' or toggle 'Auto-Play'",
                        interactive=False,
                        lines=2,
                    )
                    step_btn = gr.Button("⏭️ Next Move", variant="primary")

                    gr.Markdown("---")
                    gr.Markdown("### Auto-Play")
                    auto_play_checkbox = gr.Checkbox(
                        label="▶️ Auto-Play (toggle on/off)", value=False
                    )
                    speed_slider = gr.Slider(
                        minimum=0.5,
                        maximum=3.0,
                        value=1.0,
                        step=0.5,
                        label="Speed (seconds between moves)",
                    )
                    ai_reset_btn = gr.Button("🔄 Reset Game")

                    gr.Markdown("---")
                    gr.Markdown("*Toggle Auto-Play to watch continuously.*")

            ai_game_history = gr.State([])

            # Timer for auto-play (only available in newer Gradio, using checkbox change instead)
            step_btn.click(
                ai_vs_ai_step,
                inputs=[ai_game_history],
                outputs=[ai_game_history, ai_status, ai_board_img, ai_score_display],
            )

            # Auto-play: when checkbox is toggled on, trigger a step
            # The step function returns whether to continue, which updates the checkbox
            auto_play_checkbox.change(
                ai_vs_ai_auto_step,
                inputs=[ai_game_history, auto_play_checkbox],
                outputs=[
                    ai_game_history,
                    ai_status,
                    ai_board_img,
                    ai_score_display,
                    auto_play_checkbox,
                ],
                every=1.0,  # Check every second when active
            )

            ai_reset_btn.click(
                ai_vs_ai_reset,
                inputs=[],
                outputs=[
                    ai_game_history,
                    ai_status,
                    ai_board_img,
                    ai_score_display,
                    auto_play_checkbox,
                ],
            )

        # ===== TAB 3: About =====
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
## About AlphaGalerkin

**AlphaGalerkin** is a resolution-independent neural network for Go,
built using Continuous Operator Learning (Galerkin Transformers & FNet).

### Key Features
- 🧠 **Zero-shot resolution transfer**: Trained on 9×9, generalizes to 19×19
- 🔄 **MCTS-based search**: Monte Carlo Tree Search for move selection
- ⚡ **FNet acceleration**: Fast Fourier Transform mixing layers

### How to Play
1. **Play vs AI**: Enter moves as `row,col` (0-indexed from top-left)
2. **Watch AI vs AI**: Click "Next Move" to step through an AI game

### Scoring
- Chinese rules with **6.5 komi** (compensation for White)
- Score = Territory + Captures

---
*Built with ❤️ using Gradio, PyTorch, and the AlphaGalerkin framework.*
            """)

if __name__ == "__main__":
    demo.launch()
