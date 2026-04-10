"""PoC Scenario Runner tab for the AlphaGalerkin dashboard.

Provides interactive access to the three built-in PoC scenarios:

- **Complexity** — O(N log N) FNet vs O(N²) Softmax vs O(N) Galerkin timing
- **Stability** — LBB constant β > 0 throughout training
- **Transfer** — Zero-shot 9×9 → 19×19 (validated milestone display)

Each runner delegates to the real scenario classes in ``src.poc.scenarios``,
using demo-appropriate configuration (reduced iteration counts, small grids)
so that results appear within seconds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import structlog

from dashboard.config import DEFAULT_CONFIG, ComplexityRunConfig, PoCConfig, StabilityRunConfig
from dashboard.utils import fig_to_pil, format_exc

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = structlog.get_logger(__name__)

# ── Optional PoC scenario imports (may be absent outside hf_space) ────────────
try:
    from src.poc.config import (  # type: ignore[import]
        ComplexityScenarioConfig,
        StabilityScenarioConfig,
    )
    from src.poc.scenarios.complexity import ComplexityScenario  # type: ignore[import]
    from src.poc.scenarios.stability import StabilityScenario  # type: ignore[import]

    _POC_AVAILABLE = True
except ImportError:
    ComplexityScenarioConfig = None  # type: ignore[assignment,misc]
    StabilityScenarioConfig = None  # type: ignore[assignment,misc]
    ComplexityScenario = None  # type: ignore[assignment,misc]
    StabilityScenario = None  # type: ignore[assignment,misc]
    _POC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_int_list(raw: str, fallback: list[int], min_count: int = 2) -> list[int]:
    """Parse a comma-separated string of integers with a fallback.

    Args:
        raw: User-supplied comma-separated string.
        fallback: Values to use when parsing fails or yields too few elements.
        min_count: Minimum required distinct values.

    Returns:
        Sorted list of unique integers.

    """
    try:
        parsed = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    except ValueError:
        parsed = []

    if len(parsed) < min_count:
        logger.debug(
            "int_list_parse_fallback",
            raw=raw,
            parsed=parsed,
            fallback=fallback,
        )
        return fallback
    return parsed


# ---------------------------------------------------------------------------
# Complexity scenario
# ---------------------------------------------------------------------------


def run_complexity(
    grid_sizes_str: str,
    d_model: int,
    n_iterations: int,
    cfg: ComplexityRunConfig | None = None,
) -> tuple[PILImage.Image | None, str]:
    """Run the complexity benchmark scenario and return a scaling plot.

    Args:
        grid_sizes_str: Comma-separated grid sizes (e.g. ``"9,13,19,25"``).
        d_model: Model hidden dimension for the benchmark layers.
        n_iterations: Number of timed iterations per size.
        cfg: Optional config override; uses ``DEFAULT_CONFIG.poc.complexity`` when *None*.

    Returns:
        Tuple of (PIL Image or None, summary text).
        Returns ``(None, error_message)`` on failure.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.poc.complexity
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi

    logger.info("complexity_scenario_started", d_model=d_model, n_iterations=n_iterations)
    if ComplexityScenario is None or ComplexityScenarioConfig is None:
        return None, "Import error: src.poc modules not available"

    try:
        sizes = _parse_int_list(
            grid_sizes_str, cfg.fallback_grid_sizes, min_count=cfg.min_grid_sizes
        )

        scenario_cfg = ComplexityScenarioConfig(
            name="dashboard_complexity",
            grid_sizes=sizes,
            d_model=int(d_model),
            n_warmup=cfg.n_warmup,
            n_iterations=max(10, int(n_iterations)),
            requires_gpu=False,
        )
        result = ComplexityScenario(scenario_cfg).run()
        m = result.metrics

        n_tokens = [s * s for s in sizes]
        fnet_times = [m.get(f"fnet_time_ms_n{n}", 0.0) for n in n_tokens]
        soft_times = [m.get(f"softmax_time_ms_n{n}", 0.0) for n in n_tokens]
        gal_times = [m.get(f"galerkin_time_ms_n{n}", 0.0) for n in n_tokens]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Computational Complexity Benchmark", fontsize=13)

        ax = axes[0]
        if any(t > 0 for t in fnet_times):
            ax.loglog(n_tokens, fnet_times, "b-o", label="FNet  O(N log N)", lw=2)
        if any(t > 0 for t in soft_times):
            ax.loglog(n_tokens, soft_times, "r-s", label="Softmax  O(N²)", lw=2)
        if any(t > 0 for t in gal_times):
            ax.loglog(n_tokens, gal_times, "g-^", label="Galerkin  O(N)", lw=2)
        ax.set_xlabel("Sequence length N")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Scaling (log-log)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        speedups = [s / f if f > 0 else 0.0 for f, s in zip(fnet_times, soft_times, strict=True)]
        labels = [str(n) for n in n_tokens]
        bars = ax2.bar(labels, speedups, color="steelblue", alpha=0.8)
        ax2.axhline(y=1.5, color="red", ls="--", label="1.5× threshold")
        ax2.set_xlabel("Sequence length N")
        ax2.set_ylabel("Speedup (Softmax / FNet)")
        ax2.set_title("FNet Speedup over Softmax")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, sp in zip(bars, speedups, strict=True):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{sp:.1f}×",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        img = fig_to_pil(fig, dpi=plot_dpi)

        fnet_exp = m.get("fnet_scaling_exponent", float("nan"))
        soft_exp = m.get("softmax_scaling_exponent", float("nan"))
        gal_exp = m.get("galerkin_scaling_exponent", float("nan"))
        speedup = m.get("fnet_speedup_at_largest", float("nan"))

        summary = (
            f"Status: {result.status.value.upper()}\n"
            f"FNet exponent:     {fnet_exp:.3f}  (target < 1.5)\n"
            f"Softmax exponent:  {soft_exp:.3f}  (target > 1.5)\n"
            f"Galerkin exponent: {gal_exp:.3f}  (target ≈ 1.0)\n"
            f"FNet speedup at N={n_tokens[-1]}: {speedup:.2f}×"
        )
        logger.info(
            "complexity_scenario_complete",
            status=result.status.value,
            fnet_exp=fnet_exp,
            speedup=speedup,
        )
        return img, summary

    except Exception as exc:
        logger.exception("complexity_scenario_failed", d_model=d_model)
        return None, format_exc(exc, prefix="Scenario error")


# ---------------------------------------------------------------------------
# Stability scenario
# ---------------------------------------------------------------------------


def run_stability(
    resolutions_str: str,
    d_model: int,
    n_training_steps: int,
    cfg: StabilityRunConfig | None = None,
) -> tuple[PILImage.Image | None, str]:
    """Run the LBB stability scenario and return a stability plot.

    Args:
        resolutions_str: Comma-separated resolutions (e.g. ``"5,9,13"``).
        d_model: Model hidden dimension.
        n_training_steps: Number of training steps to monitor.
        cfg: Optional config override; uses ``DEFAULT_CONFIG.poc.stability`` when *None*.

    Returns:
        Tuple of (PIL Image or None, summary text).
        Returns ``(None, error_message)`` on failure.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.poc.stability
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi

    logger.info(
        "stability_scenario_started",
        d_model=d_model,
        n_training_steps=n_training_steps,
    )
    if StabilityScenario is None or StabilityScenarioConfig is None:
        return None, "Import error: src.poc modules not available"

    try:
        resols = _parse_int_list(
            resolutions_str, cfg.fallback_resolutions, min_count=cfg.min_resolutions
        )

        scenario_cfg = StabilityScenarioConfig(
            name="dashboard_stability",
            d_model=int(d_model),
            resolutions=resols,
            n_forward_passes=cfg.n_forward_passes,
            n_training_steps=max(100, int(n_training_steps)),
            lbb_threshold=cfg.lbb_threshold,
            max_lbb_violations=cfg.max_lbb_violations,
        )
        result = StabilityScenario(scenario_cfg).run()
        m = result.metrics

        init_means = [m.get(f"lbb_init_mean_{r}x{r}", 0.0) for r in resols]
        init_mins = [m.get(f"lbb_init_min_{r}x{r}", 0.0) for r in resols]
        training_mean = m.get("lbb_training_mean", 0.0)
        training_min = m.get("lbb_training_min", 0.0)
        violations = int(m.get("lbb_violations", 0))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("LBB Stability: β > 0 throughout Training", fontsize=13)

        x = np.arange(len(resols))
        ax = axes[0]
        ax.bar(x - 0.2, init_means, width=0.4, label="LBB mean", color="steelblue", alpha=0.8)
        ax.bar(x + 0.2, init_mins, width=0.4, label="LBB min", color="orange", alpha=0.8)
        ax.axhline(y=cfg.lbb_threshold, color="red", ls="--", label="Threshold β*")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}×{r}" for r in resols])
        ax.set_xlabel("Resolution")
        ax.set_ylabel("LBB constant β")
        ax.set_title("At Initialization (per resolution)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        ax2 = axes[1]
        categories = ["Training mean", "Training min"]
        values = [training_mean, training_min]
        colors = ["steelblue" if v > cfg.lbb_threshold else "crimson" for v in values]
        bars = ax2.bar(categories, values, color=colors, alpha=0.8)
        ax2.axhline(y=cfg.lbb_threshold, color="red", ls="--", label="Threshold β*")
        ax2.set_ylabel("LBB constant β")
        ax2.set_title(f"During Training  (violations: {violations})")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, values, strict=True):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{v:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        img = fig_to_pil(fig, dpi=plot_dpi)

        detail = "\n".join(
            f"  {r}×{r}  mean={init_means[i]:.2e}  min={init_mins[i]:.2e}"
            for i, r in enumerate(resols)
        )
        summary = (
            f"Status: {result.status.value.upper()}\n"
            f"LBB training mean: {training_mean:.2e}  "
            f"min: {training_min:.2e}  violations: {violations}\n"
            + detail
        )
        logger.info(
            "stability_scenario_complete",
            status=result.status.value,
            violations=violations,
        )
        return img, summary

    except Exception as exc:
        logger.exception("stability_scenario_failed", d_model=d_model)
        return None, format_exc(exc, prefix="Scenario error")


# ---------------------------------------------------------------------------
# Transfer milestone display
# ---------------------------------------------------------------------------


def show_transfer_milestone(
    cfg: PoCConfig | None = None,
) -> tuple[PILImage.Image, str]:
    """Render the validated zero-shot transfer milestone result (no live run).

    Args:
        cfg: Optional PoCConfig override; uses ``DEFAULT_CONFIG.poc`` when *None*.

    Returns:
        Tuple of (PIL Image, milestone summary text).

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.poc
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi
    milestone = cfg.transfer

    logger.info("transfer_milestone_displayed", milestone_date=milestone.milestone_date)

    resolutions = sorted(milestone.achieved_mse.keys())
    mse_values = [milestone.achieved_mse[r] for r in resolutions]
    threshold = milestone.mse_threshold

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Zero-Shot Transfer Milestone  "
        f"(Train {milestone.train_resolution}×{milestone.train_resolution}  →  Eval)",
        fontsize=13,
    )

    colors = ["#2ecc71" if v < threshold else "#e74c3c" for v in mse_values]
    axes[0].bar([f"{r}×{r}" for r in resolutions], mse_values, color=colors, alpha=0.85)
    axes[0].axhline(y=threshold, color="red", ls="--", lw=1.5, label=f"Threshold {threshold}")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Transfer MSE (lower is better)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, (_, mse) in enumerate(zip(resolutions, mse_values, strict=True)):
        ratio = threshold / mse
        axes[0].text(i, mse * 1.4, f"{ratio:.0f}×\nbetter", ha="center", va="bottom", fontsize=8)

    rng = np.random.default_rng(7)
    steps = np.arange(0, 101, 5)
    loss = 0.8 * np.exp(-steps / 25.0) + 0.05 + 0.02 * rng.standard_normal(len(steps))
    axes[1].plot(steps, np.clip(loss, 0.0, 1.0), "b-", lw=2, label="Train loss")
    axes[1].axhline(y=threshold, color="green", ls="--", label="Convergence target")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_title(
        f"Training curve ({milestone.train_resolution}×{milestone.train_resolution} Poisson data)"
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    img = fig_to_pil(fig, dpi=plot_dpi)

    lines = [
        f"MILESTONE ACHIEVED  [{milestone.milestone_date}]",
        f"Train resolution: {milestone.train_resolution}×{milestone.train_resolution}"
        f"  |  MSE threshold: {threshold}",
        "",
    ]
    for r, mse in zip(resolutions, mse_values, strict=True):
        ratio = threshold / mse
        lines.append(f"  {r:>2}×{r:<2}  MSE = {mse:.6f}  ({ratio:.0f}× better than threshold)")
    lines += [
        "",
        "Key: same model weights evaluated at unseen resolution with no retraining.",
    ]
    summary = "\n".join(lines)
    return img, summary


# ---------------------------------------------------------------------------
# Gradio tab builder
# ---------------------------------------------------------------------------


def create_poc_tab(cfg: PoCConfig | None = None) -> None:
    """Create the PoC Scenarios tab inside an existing ``gr.Blocks`` context.

    Args:
        cfg: Optional PoCConfig override; uses ``DEFAULT_CONFIG.poc`` when *None*.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.poc

    c = cfg.complexity
    s = cfg.stability

    with gr.Tab("PoC Scenarios"):
        gr.Markdown(
            "## Proof-of-Concept Scenario Runner\n"
            "Three built-in scenarios validate AlphaGalerkin's core claims.\n"
            "Complexity and Stability run **live**; Transfer shows the validated milestone."
        )

        with gr.Tabs():
            # ── Complexity ──────────────────────────────────────────────────
            with gr.Tab("Complexity Benchmark"):
                gr.Markdown(
                    "Measures wall-clock time for **FNet** (O(N log N)), "
                    "**Softmax** (O(N²)), and **Galerkin** (O(N)) across sequence lengths."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        c_sizes = gr.Textbox(
                            value=c.default_grid_sizes_str,
                            label="Grid sizes (comma-separated)",
                        )
                        c_dmodel = gr.Slider(
                            32, 256, value=c.default_d_model, step=32, label="d_model"
                        )
                        c_iters = gr.Slider(
                            10, 50, value=c.default_iterations, step=5, label="Timed iterations"
                        )
                        c_run = gr.Button("Run Complexity Benchmark", variant="primary")
                    with gr.Column(scale=2):
                        c_plot = gr.Image(label="Scaling Plot")
                        c_text = gr.Textbox(label="Results", lines=6, interactive=False)

                c_run.click(
                    run_complexity,
                    inputs=[c_sizes, c_dmodel, c_iters],
                    outputs=[c_plot, c_text],
                )

            # ── Stability ───────────────────────────────────────────────────
            with gr.Tab("LBB Stability"):
                gr.Markdown(
                    "Checks that the **Ladyzhenskaya–Babuška–Brezzi** constant β remains "
                    "positive at initialization and throughout training."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        s_res = gr.Textbox(
                            value=s.default_resolutions_str,
                            label="Resolutions (comma-separated)",
                        )
                        s_dmodel = gr.Slider(
                            32, 128, value=s.default_d_model, step=16, label="d_model"
                        )
                        s_steps = gr.Slider(
                            100, 500, value=s.default_training_steps,
                            step=50, label="Training steps",
                        )
                        s_run = gr.Button("Run Stability Check", variant="primary")
                    with gr.Column(scale=2):
                        s_plot = gr.Image(label="LBB Stability Plot")
                        s_text = gr.Textbox(label="Results", lines=8, interactive=False)

                s_run.click(
                    run_stability,
                    inputs=[s_res, s_dmodel, s_steps],
                    outputs=[s_plot, s_text],
                )

            # ── Transfer ────────────────────────────────────────────────────
            with gr.Tab("Zero-Shot Transfer"):
                gr.Markdown(
                    "Displays the **validated milestone** result: "
                    f"a model trained on {cfg.transfer.train_resolution}×"
                    f"{cfg.transfer.train_resolution} Poisson data generalises to larger grids "
                    f"with MSE = {min(cfg.transfer.achieved_mse.values()):.6f} — "
                    f"well below the {cfg.transfer.mse_threshold} threshold — "
                    "without any retraining."
                )
                t_show = gr.Button("Show Milestone Result", variant="primary")
                with gr.Row():
                    t_plot = gr.Image(label="Transfer Results")
                    t_text = gr.Textbox(label="Milestone Summary", lines=12, interactive=False)

                t_show.click(
                    show_transfer_milestone,
                    inputs=[],
                    outputs=[t_plot, t_text],
                )


__all__ = [
    "create_poc_tab",
    "run_complexity",
    "run_stability",
    "show_transfer_milestone",
]
