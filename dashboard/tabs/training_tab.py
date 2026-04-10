"""Training Dashboard tab for AlphaGalerkin dashboard.

Shows model architecture details, loss function breakdown, and example
training curves, giving a full picture of the training pipeline without
running an actual training job.
"""

from __future__ import annotations

import io

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage


def _fig_to_pil(fig: plt.Figure) -> PILImage.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


# ---------------------------------------------------------------------------
# Model architecture summary
# ---------------------------------------------------------------------------


def get_model_summary(d_model: int, n_galerkin: int, n_softmax: int, n_fourier: int) -> str:
    """Compute parameter counts for a given architecture config."""
    try:
        import sys

        sys.path.insert(
            0, str(__import__("pathlib").Path(__file__).parent.parent.parent / "hf_space")
        )
        from config.schemas import OperatorConfig
        from src.modeling.model import AlphaGalerkinModel

        cfg = OperatorConfig(
            d_model=int(d_model),
            n_galerkin_layers=int(n_galerkin),
            n_softmax_layers=int(n_softmax),
            n_fourier_features=int(n_fourier),
        )
        model = AlphaGalerkinModel(cfg)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        lines = [
            f"d_model={d_model}  d_key={cfg.d_key}  d_ffn={cfg.d_ffn}",
            f"Galerkin layers: {n_galerkin}  |  Softmax layers: {n_softmax}",
            f"Fourier features: {n_fourier}  |  Heads: {cfg.n_heads}",
            f"FNet mixing: {'enabled' if cfg.use_fnet_mixing else 'disabled'}",
            "─" * 42,
            f"Total parameters:     {total:>12,}",
            f"Trainable parameters: {trainable:>12,}",
            f"Approx. VRAM (fp32):  {total * 4 / 1e6:>10.1f} MB",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return (
            f"Could not load model (checkpoint may be missing): {exc}\n\n"
            f"Architecture config:\n"
            f"  d_model={d_model}, Galerkin layers={n_galerkin},\n"
            f"  Softmax layers={n_softmax}, Fourier features={n_fourier}"
        )


# ---------------------------------------------------------------------------
# Training curve plots
# ---------------------------------------------------------------------------


def plot_training_curves(
    total_steps: int,
    lr: float,
    policy_weight: float,
    value_weight: float,
    lbb_weight: float,
) -> tuple[PILImage.Image, str]:
    """Generate representative training curves for the AlphaGalerkin loss."""
    rng = np.random.default_rng(42)
    steps = np.linspace(0, int(total_steps), 200)

    def smooth_loss(scale: float, decay: float, noise: float) -> np.ndarray:
        base = scale * np.exp(-steps / decay) + 0.02
        return base + noise * rng.standard_normal(len(steps)) * np.exp(-steps / (decay * 2))

    policy_loss = smooth_loss(2.5, total_steps * 0.3, 0.05)
    value_loss = smooth_loss(0.8, total_steps * 0.25, 0.02)
    lbb_loss = smooth_loss(0.3, total_steps * 0.4, 0.01)

    total_loss = (
        policy_weight * policy_loss + value_weight * value_loss + lbb_weight * lbb_loss
    )

    # LBB constant (should stay positive)
    lbb_const = 0.05 + 0.08 * (1 - np.exp(-steps / (total_steps * 0.1))) + 0.005 * rng.standard_normal(len(steps))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("AlphaGalerkin Training Dashboard", fontsize=14)

    # ── Total loss ──
    axes[0, 0].plot(steps, np.clip(total_loss, 0, None), color="purple", lw=2)
    axes[0, 0].set_title("Total Loss  (policy + value + LBB)")
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # ── Component losses ──
    axes[0, 1].plot(steps, np.clip(policy_loss, 0, None), "b-", label="Policy CE", lw=1.5)
    axes[0, 1].plot(steps, np.clip(value_loss, 0, None), "r-", label="Value MSE", lw=1.5)
    axes[0, 1].plot(steps, np.clip(lbb_loss, 0, None), "g-", label="LBB reg", lw=1.5)
    axes[0, 1].set_title("Loss Components")
    axes[0, 1].set_xlabel("Training step")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # ── LBB stability constant ──
    axes[1, 0].plot(steps, lbb_const, color="teal", lw=1.5)
    axes[1, 0].axhline(y=1e-6, color="red", ls="--", label="β* threshold")
    axes[1, 0].fill_between(steps, 0, lbb_const, alpha=0.15, color="teal")
    axes[1, 0].set_title("LBB Constant β  (inf-sup condition)")
    axes[1, 0].set_xlabel("Training step")
    axes[1, 0].set_ylabel("β")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # ── LR schedule ──
    warmup = int(total_steps * 0.05)
    lr_arr = np.where(
        steps < warmup,
        lr * steps / max(warmup, 1),
        lr * (0.5 * (1 + np.cos(np.pi * (steps - warmup) / max(total_steps - warmup, 1)))),
    )
    axes[1, 1].plot(steps, lr_arr, color="darkorange", lw=2)
    axes[1, 1].set_title("Learning Rate Schedule  (cosine with warmup)")
    axes[1, 1].set_xlabel("Training step")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    img = _fig_to_pil(fig)

    final_total = float(np.clip(total_loss, 0, None)[-1])
    summary = (
        f"Total steps: {int(total_steps):,}\n"
        f"Peak LR: {lr:.2e}  |  Warmup: {warmup:,} steps\n"
        f"Loss weights — Policy: {policy_weight}  Value: {value_weight}  LBB: {lbb_weight}\n"
        f"Final total loss (simulated): {final_total:.4f}\n"
        "Note: curves are representative; run scripts/train.py for real training."
    )
    return img, summary


# ---------------------------------------------------------------------------
# Loss component breakdown
# ---------------------------------------------------------------------------


def show_loss_breakdown() -> PILImage.Image:
    """Render a diagram of the AlphaGalerkin loss function."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    # Title
    ax.text(
        0.5, 0.95, "AlphaGalerkin Loss = L_policy + L_value + L_lbb",
        transform=ax.transAxes, ha="center", va="top", fontsize=14, fontweight="bold",
    )

    # Three boxes
    boxes = [
        (0.12, "L_policy\n(Policy Cross-Entropy)",
         "Softmax attention output\nvs MCTS visit distribution\n\nCross-entropy loss\nencourages sharp,\ncorrect move predictions",
         "#3498db"),
        (0.45, "L_value\n(Value MSE)",
         "Predicted game value\nvs self-play outcome\n\nMSE loss\naligns value head\nwith win/loss signal",
         "#e74c3c"),
        (0.78, "L_lbb\n(LBB Regularisation)",
         "σ_min(K→V projection)\nPenalises near-zero\nsingular values\n\nEnforces inf-sup\ncondition β > 0",
         "#2ecc71"),
    ]

    for cx, title, body, color in boxes:
        rect = plt.Rectangle(
            (cx - 0.17, 0.08), 0.34, 0.78,
            transform=ax.transAxes, facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(cx, 0.82, title, transform=ax.transAxes,
                ha="center", va="top", fontsize=11, fontweight="bold", color=color)
        ax.text(cx, 0.60, body, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="#333333",
                multialignment="center")

    # Arrows joining the boxes
    for x in [0.30, 0.63]:
        ax.annotate(
            "", xy=(x + 0.02, 0.47), xytext=(x - 0.02, 0.47),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.5),
        )
        ax.text(x, 0.47, "+", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color="#666666")

    plt.tight_layout()
    return _fig_to_pil(fig)


# ---------------------------------------------------------------------------
# Gradio tab builder
# ---------------------------------------------------------------------------


def create_training_tab() -> None:
    """Create the Training Dashboard tab inside an existing gr.Blocks context."""
    with gr.Tab("Training Dashboard"):
        gr.Markdown(
            """
## Training Pipeline Overview
Configure the model, inspect the loss function, and generate representative
training curves.  To run real training use `python -m scripts.train`.
"""
        )

        with gr.Tabs():
            # ── Architecture ────────────────────────────────────────────────
            with gr.Tab("Model Architecture"):
                gr.Markdown(
                    "AlphaGalerkin combines **Galerkin attention** (strategy body, O(N)) "
                    "with **Softmax attention** (tactical head, exact) and "
                    "**FNet mixing** (O(N log N) FFT-based token mixing)."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        m_dmodel = gr.Slider(64, 512, value=256, step=64, label="d_model")
                        m_galerkin = gr.Slider(1, 12, value=6, step=1, label="Galerkin layers")
                        m_softmax = gr.Slider(1, 6, value=2, step=1, label="Softmax layers")
                        m_fourier = gr.Slider(32, 256, value=128, step=32, label="Fourier features")
                        m_btn = gr.Button("Compute Architecture Summary", variant="primary")
                    with gr.Column(scale=2):
                        m_text = gr.Textbox(
                            label="Architecture Summary",
                            lines=10,
                            interactive=False,
                        )

                m_btn.click(
                    get_model_summary,
                    inputs=[m_dmodel, m_galerkin, m_softmax, m_fourier],
                    outputs=[m_text],
                )

            # ── Loss function ───────────────────────────────────────────────
            with gr.Tab("Loss Function"):
                gr.Markdown(
                    "The total loss combines three terms:\n"
                    "- **Policy CE**: guides MCTS move selection\n"
                    "- **Value MSE**: calibrates game outcome prediction\n"
                    "- **LBB regularisation**: maintains inf-sup stability"
                )
                loss_btn = gr.Button("Show Loss Breakdown Diagram", variant="primary")
                loss_img = gr.Image(label="Loss Function Breakdown")
                loss_btn.click(show_loss_breakdown, inputs=[], outputs=[loss_img])

            # ── Training curves ─────────────────────────────────────────────
            with gr.Tab("Training Curves"):
                gr.Markdown(
                    "Generates representative training curves for the selected "
                    "configuration.  All curves are **simulated** to illustrate "
                    "typical behaviour."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        t_steps = gr.Slider(
                            1000, 50000, value=10000, step=1000, label="Total steps"
                        )
                        t_lr = gr.Number(value=3e-4, label="Peak learning rate")
                        t_pw = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Policy weight")
                        t_vw = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Value weight")
                        t_lw = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="LBB weight")
                        t_btn = gr.Button("Generate Training Curves", variant="primary")
                    with gr.Column(scale=2):
                        t_plot = gr.Image(label="Training Curves")
                        t_text = gr.Textbox(label="Summary", lines=5, interactive=False)

                t_btn.click(
                    plot_training_curves,
                    inputs=[t_steps, t_lr, t_pw, t_vw, t_lw],
                    outputs=[t_plot, t_text],
                )
