"""Training Dashboard tab for the AlphaGalerkin dashboard.

Shows model architecture details, the AlphaGalerkin loss function breakdown,
and configurable representative training curves — without running a real
training job.  To launch actual training use ``python -m scripts.train``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import structlog

from dashboard.config import DEFAULT_CONFIG, TrainingConfig
from dashboard.utils import fig_to_pil, format_exc

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = structlog.get_logger(__name__)

# Bytes per parameter for 32-bit floating-point (fp32) VRAM estimation.
_BYTES_PER_FLOAT32: int = 4


# ---------------------------------------------------------------------------
# Architecture summary
# ---------------------------------------------------------------------------


def get_model_summary(
    d_model: int,
    n_galerkin: int,
    n_softmax: int,
    n_fourier: int,
) -> str:
    """Return a human-readable architecture summary with parameter counts.

    Attempts to instantiate ``AlphaGalerkinModel`` via the hf_space path.
    Falls back to a textual description when the model or checkpoint cannot
    be loaded (e.g. missing dependency).

    Args:
        d_model: Model embedding dimension.
        n_galerkin: Number of Galerkin attention layers.
        n_softmax: Number of Softmax attention layers.
        n_fourier: Number of Fourier positional-encoding features.

    Returns:
        Multi-line string with architecture details and parameter counts.

    """
    logger.info(
        "model_summary_requested",
        d_model=d_model,
        n_galerkin=n_galerkin,
        n_softmax=n_softmax,
        n_fourier=n_fourier,
    )
    try:
        # Ensure hf_space is in sys.path so config.schemas is importable.
        import sys

        hf_space = Path(__file__).parent.parent.parent / "hf_space"
        if str(hf_space) not in sys.path:
            sys.path.insert(0, str(hf_space))

        from config.schemas import OperatorConfig  # type: ignore[import]
        from src.modeling.model import AlphaGalerkinModel  # type: ignore[import]

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
            f"d_model={d_model}  d_key={cfg.d_key}  d_value={cfg.d_value}  d_ffn={cfg.d_ffn}",
            f"Galerkin layers: {n_galerkin}  |  Softmax layers: {n_softmax}",
            f"Fourier features: {n_fourier}  |  Heads: {cfg.n_heads}",
            f"FNet mixing: {'enabled' if cfg.use_fnet_mixing else 'disabled'}",
            f"LBB β threshold: {cfg.lbb_beta_threshold}",
            "─" * 44,
            f"Total parameters:     {total:>12,}",
            f"Trainable parameters: {trainable:>12,}",
            f"Approx. VRAM (fp32):  {total * _BYTES_PER_FLOAT32 / 1e6:>10.1f} MB",
        ]
        logger.info("model_summary_complete", total_params=total)
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("model_summary_failed", error=str(exc))
        return (
            format_exc(exc, prefix="Model load failed")
            + "\n\nArchitecture (from config):\n"
            + f"  d_model={d_model}, Galerkin layers={n_galerkin},\n"
            + f"  Softmax layers={n_softmax}, Fourier features={n_fourier}"
        )


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(
    total_steps: int,
    lr: float,
    policy_weight: float,
    value_weight: float,
    lbb_weight: float,
    cfg: TrainingConfig | None = None,
) -> tuple[PILImage.Image, str]:
    """Generate representative training curves for the AlphaGalerkin loss.

    The curves are **simulated** using exponential decay with configurable
    parameters from ``TrainingConfig``; they illustrate typical training
    dynamics but do not reflect any real run.

    Args:
        total_steps: Total number of training steps on the x-axis.
        lr: Peak learning rate (used for the LR schedule display).
        policy_weight: Scalar multiplier on the policy CE loss component.
        value_weight: Scalar multiplier on the value MSE loss component.
        lbb_weight: Scalar multiplier on the LBB regularisation term.
        cfg: Optional TrainingConfig override; uses ``DEFAULT_CONFIG.training`` when *None*.

    Returns:
        Tuple of (PIL Image of the 2×2 plot grid, summary string).

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.training
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi

    logger.info(
        "training_curves_requested",
        total_steps=total_steps,
        lr=lr,
        policy_weight=policy_weight,
        value_weight=value_weight,
        lbb_weight=lbb_weight,
    )

    rng = np.random.default_rng(cfg.random_seed)
    steps = np.linspace(0, float(total_steps), cfg.curve_n_points)

    def _decay_curve(scale: float, decay_frac: float, noise_scale: float) -> np.ndarray:
        decay = total_steps * decay_frac
        base = scale * np.exp(-steps / decay) + 0.02
        noise = noise_scale * rng.standard_normal(len(steps)) * np.exp(-steps / (decay * 2))
        return base + noise

    policy_loss = _decay_curve(cfg.policy_loss_scale, cfg.policy_decay_fraction, 0.05)
    value_loss = _decay_curve(cfg.value_loss_scale, cfg.value_decay_fraction, 0.02)
    lbb_loss = _decay_curve(cfg.lbb_loss_scale, cfg.lbb_decay_fraction, 0.01)
    total_loss = (
        float(policy_weight) * policy_loss
        + float(value_weight) * value_loss
        + float(lbb_weight) * lbb_loss
    )

    lbb_const = (
        cfg.lbb_const_asymptote
        + cfg.lbb_const_amplitude * (1 - np.exp(-steps / max(total_steps * 0.1, 1.0)))
        + cfg.lbb_const_noise_scale * rng.standard_normal(len(steps))
    )

    warmup = max(1, int(total_steps * cfg.warmup_fraction))
    decay_steps = max(1, total_steps - warmup)
    lr_arr = np.where(
        steps < warmup,
        float(lr) * steps / warmup,
        float(lr) * 0.5 * (1 + np.cos(np.pi * (steps - warmup) / decay_steps)),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("AlphaGalerkin Training Dashboard (simulated)", fontsize=14)

    axes[0, 0].plot(steps, np.clip(total_loss, 0, None), color="purple", lw=2)
    axes[0, 0].set_title("Total Loss  (policy + value + LBB)")
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, np.clip(policy_loss, 0, None), "b-", label="Policy CE", lw=1.5)
    axes[0, 1].plot(steps, np.clip(value_loss, 0, None), "r-", label="Value MSE", lw=1.5)
    axes[0, 1].plot(steps, np.clip(lbb_loss, 0, None), "g-", label="LBB reg", lw=1.5)
    axes[0, 1].set_title("Loss Components")
    axes[0, 1].set_xlabel("Training step")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, lbb_const, color="teal", lw=1.5)
    axes[1, 0].axhline(y=cfg.lbb_min_threshold, color="red", ls="--", label="β* threshold")
    axes[1, 0].fill_between(steps, 0, lbb_const, alpha=0.15, color="teal")
    axes[1, 0].set_title("LBB Constant β  (inf-sup condition)")
    axes[1, 0].set_xlabel("Training step")
    axes[1, 0].set_ylabel("β")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, lr_arr, color="darkorange", lw=2)
    axes[1, 1].set_title("Learning Rate Schedule  (cosine + warmup)")
    axes[1, 1].set_xlabel("Training step")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    img = fig_to_pil(fig, dpi=plot_dpi)

    final_total = float(np.clip(total_loss, 0, None)[-1])
    summary = (
        f"Total steps: {int(total_steps):,}\n"
        f"Peak LR: {lr:.2e}  |  Warmup: {warmup:,} steps\n"
        f"Loss weights — Policy: {policy_weight}  Value: {value_weight}  LBB: {lbb_weight}\n"
        f"Final total loss (simulated): {final_total:.4f}\n"
        "Note: curves are representative; run scripts/train.py for real training."
    )
    logger.info("training_curves_complete", final_loss=final_total)
    return img, summary


# ---------------------------------------------------------------------------
# Loss breakdown diagram
# ---------------------------------------------------------------------------


def show_loss_breakdown(
    cfg: TrainingConfig | None = None,
) -> PILImage.Image:
    """Render a static diagram explaining the AlphaGalerkin loss function.

    Args:
        cfg: Optional TrainingConfig override; uses ``DEFAULT_CONFIG.training`` when *None*.

    Returns:
        PIL Image of the breakdown diagram.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.training
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi

    logger.info("loss_breakdown_diagram_requested")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    ax.text(
        0.5,
        0.95,
        "AlphaGalerkin Loss  =  L_policy  +  L_value  +  L_lbb",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    boxes = [
        (
            0.12,
            "L_policy\n(Policy Cross-Entropy)",
            "Softmax attention output\nvs MCTS visit distribution\n\n"
            "Cross-entropy loss\nencourages sharp,\ncorrect move predictions",
            "#3498db",
        ),
        (
            0.45,
            "L_value\n(Value MSE)",
            "Predicted game value\nvs self-play outcome\n\n"
            "MSE loss\naligns value head\nwith win/loss signal",
            "#e74c3c",
        ),
        (
            0.78,
            "L_lbb\n(LBB Regularisation)",
            f"σ_min(K→V projection)\nPenalises near-zero\nsingular values\n\n"
            f"Enforces inf-sup\ncondition β > {cfg.lbb_min_threshold}",
            "#2ecc71",
        ),
    ]

    for cx, title, body, color in boxes:
        rect = plt.Rectangle(
            (cx - 0.17, 0.08),
            0.34,
            0.78,
            transform=ax.transAxes,
            facecolor=color,
            alpha=0.12,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            cx,
            0.82,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=color,
        )
        ax.text(
            cx,
            0.60,
            body,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
            multialignment="center",
        )

    for x in [0.30, 0.63]:
        ax.annotate(
            "",
            xy=(x + 0.02, 0.47),
            xytext=(x - 0.02, 0.47),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.5},
        )
        ax.text(
            x,
            0.47,
            "+",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="#666666",
        )

    plt.tight_layout()
    img = fig_to_pil(fig, dpi=plot_dpi)
    logger.info("loss_breakdown_diagram_complete")
    return img


# ---------------------------------------------------------------------------
# Gradio tab builder
# ---------------------------------------------------------------------------


def create_training_tab(cfg: TrainingConfig | None = None) -> None:
    """Create the Training Dashboard tab inside an existing ``gr.Blocks`` context.

    Args:
        cfg: Optional TrainingConfig override; uses ``DEFAULT_CONFIG.training`` when *None*.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.training

    with gr.Tab("Training Dashboard"):
        gr.Markdown(
            "## Training Pipeline Overview\n"
            "Configure the model, inspect the loss function, and generate representative "
            "training curves.  To run real training: ``python -m scripts.train``."
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
                        m_dmodel = gr.Slider(
                            cfg.d_model_min,
                            cfg.d_model_max,
                            value=cfg.d_model_default,
                            step=cfg.d_model_step,
                            label="d_model",
                        )
                        m_galerkin = gr.Slider(
                            cfg.galerkin_layers_min,
                            cfg.galerkin_layers_max,
                            value=cfg.galerkin_layers_default,
                            step=1,
                            label="Galerkin layers",
                        )
                        m_softmax = gr.Slider(
                            cfg.softmax_layers_min,
                            cfg.softmax_layers_max,
                            value=cfg.softmax_layers_default,
                            step=1,
                            label="Softmax layers",
                        )
                        m_fourier = gr.Slider(
                            cfg.fourier_min,
                            cfg.fourier_max,
                            value=cfg.fourier_default,
                            step=cfg.fourier_step,
                            label="Fourier features",
                        )
                        m_btn = gr.Button("Compute Architecture Summary", variant="primary")
                    with gr.Column(scale=2):
                        m_text = gr.Textbox(
                            label="Architecture Summary", lines=10, interactive=False
                        )

                m_btn.click(
                    get_model_summary,
                    inputs=[m_dmodel, m_galerkin, m_softmax, m_fourier],
                    outputs=[m_text],
                )

            # ── Loss function ───────────────────────────────────────────────
            with gr.Tab("Loss Function"):
                gr.Markdown(
                    "The total loss combines:\n"
                    "- **Policy CE** — guides MCTS move selection\n"
                    "- **Value MSE** — calibrates game outcome prediction\n"
                    "- **LBB regularisation** — maintains inf-sup stability"
                )
                loss_btn = gr.Button("Show Loss Breakdown Diagram", variant="primary")
                loss_img = gr.Image(label="Loss Function Breakdown")
                loss_btn.click(show_loss_breakdown, inputs=[], outputs=[loss_img])

            # ── Training curves ─────────────────────────────────────────────
            with gr.Tab("Training Curves"):
                gr.Markdown(
                    "Generates **representative** training curves for the selected "
                    "configuration.  All curves are simulated."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        t_steps = gr.Slider(
                            cfg.steps_min,
                            cfg.steps_max,
                            value=cfg.steps_default,
                            step=cfg.steps_step,
                            label="Total steps",
                        )
                        t_lr = gr.Number(value=cfg.default_lr, label="Peak learning rate")
                        t_pw = gr.Slider(
                            0.1,
                            2.0,
                            value=cfg.default_policy_weight,
                            step=0.1,
                            label="Policy weight",
                        )
                        t_vw = gr.Slider(
                            0.1,
                            2.0,
                            value=cfg.default_value_weight,
                            step=0.1,
                            label="Value weight",
                        )
                        t_lw = gr.Slider(
                            0.0,
                            1.0,
                            value=cfg.default_lbb_weight,
                            step=0.05,
                            label="LBB weight",
                        )
                        t_btn = gr.Button("Generate Training Curves", variant="primary")
                    with gr.Column(scale=2):
                        t_plot = gr.Image(label="Training Curves")
                        t_text = gr.Textbox(label="Summary", lines=5, interactive=False)

                t_btn.click(
                    plot_training_curves,
                    inputs=[t_steps, t_lr, t_pw, t_vw, t_lw],
                    outputs=[t_plot, t_text],
                )


__all__ = [
    "create_training_tab",
    "get_model_summary",
    "plot_training_curves",
    "show_loss_breakdown",
]
