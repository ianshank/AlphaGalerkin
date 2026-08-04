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

from dashboard.config import (
    COMMITTED_TARGET_RESOLUTION,
    DEFAULT_CONFIG,
    ComplexityRunConfig,
    PoCConfig,
    StabilityRunConfig,
)
from dashboard.utils import fig_to_pil, format_exc

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = structlog.get_logger(__name__)

# Minimum number of benchmark iterations to avoid degenerate timing results.
_MIN_ITERATIONS: int = 10

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
            n_iterations=max(_MIN_ITERATIONS, int(n_iterations)),
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
            f"min: {training_min:.2e}  violations: {violations}\n" + detail
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
    """Render the committed zero-shot transfer benchmark (no live run).

    Shows the honest comparison the benchmark actually supports: the operator trained
    only at 9x9 against a discrete CNN retrained at 19x19. The operator loses on
    accuracy; what it buys is zero retraining. Every figure comes from
    ``config/baselines/transfer_ci.json`` via :class:`~dashboard.config.TransferMilestone`.

    A ratio against the legacy 0.05 pass threshold is deliberately *not* rendered --
    ``specs/transfer_baseline_compare.spec.md`` retracts that framing.

    Args:
        cfg: Optional PoCConfig override; uses ``DEFAULT_CONFIG.poc`` when *None*.

    Returns:
        Tuple of (PIL Image, benchmark summary text).

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.poc
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi
    milestone = cfg.transfer
    # Pinned, not derived: the baseline fields below are 19x19-specific, so deriving a
    # target from achieved_mse would let a config override compare mismatched resolutions
    # while labelling them the same. TransferMilestone validates that the key is present.
    target = COMMITTED_TARGET_RESOLUTION

    logger.info(
        "transfer_benchmark_displayed",
        milestone_date=milestone.milestone_date,
        target_resolution=target,
        ratio=milestone.transfer_ratio_19x19,
    )

    resolutions = sorted(milestone.achieved_mse.keys())
    mse_values = [milestone.achieved_mse[r] for r in resolutions]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Zero-Shot Transfer: operator (train {milestone.train_resolution}×"
        f"{milestone.train_resolution}) vs. retrained CNN — committed benchmark",
        fontsize=12,
    )

    # ── Left: the honest three-arm comparison at the target resolution ──────────
    arm_labels = [
        f"Operator\nzero-shot\n(train {milestone.train_resolution}×{milestone.train_resolution})",
        f"CNN\nretrained\n({target}×{target})",
        "CNN\nzero-shot",
    ]
    arm_values = [
        milestone.achieved_mse[target],
        milestone.cnn_retrained_mse_19x19,
        milestone.cnn_zeroshot_mse_19x19,
    ]
    # Red marks the arm that loses; the operator is the one under test.
    arm_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    axes[0].bar(arm_labels, arm_values, color=arm_colors, alpha=0.85)
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")
    axes[0].set_title(f"MSE at {target}×{target} (lower is better)")
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, value in enumerate(arm_values):
        axes[0].text(i, value * 1.15, f"{value:.2e}", ha="center", va="bottom", fontsize=8)
    axes[0].text(
        0.5,
        0.95,
        f"Operator loses to the retrained CNN by {milestone.transfer_ratio_19x19:.1f}×",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "#fdf3e7", "alpha": 0.9},
    )

    # ── Right: the operator's measured degradation across resolutions ───────────
    axes[1].plot(resolutions, mse_values, "o-", color="#e74c3c", lw=2, label="Operator zero-shot")
    axes[1].axhline(
        y=milestone.cnn_retrained_mse_19x19,
        color="#2ecc71",
        ls="--",
        lw=1.5,
        label=f"CNN retrained @ {target}×{target}",
    )
    axes[1].set_xlabel("Evaluation resolution")
    axes[1].set_ylabel("MSE")
    axes[1].set_yscale("log")
    axes[1].set_xticks(resolutions)
    axes[1].set_xticklabels([f"{r}×{r}" for r in resolutions])
    axes[1].set_title("Measured transfer, one model, no retraining")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    img = fig_to_pil(fig, dpi=plot_dpi)

    lines = [
        f"COMMITTED BENCHMARK  [{milestone.milestone_date}]",
        "Source: results/transfer_baseline_compare.csv",
        "        config/baselines/transfer_ci.json",
        "Provenance: representative (median-ranked) seed of a 3-seed run. The",
        f"operator's {target}x{target} MSE is the 3-seed median; each baseline is that",
        "same seed's paired value, so the ratio below is a within-seed ratio.",
        "",
        f"Operator trained at {milestone.train_resolution}×{milestone.train_resolution} only:",
    ]
    for r, mse in zip(resolutions, mse_values, strict=True):
        tag = "  (in-distribution)" if r == milestone.train_resolution else "  (zero-shot)"
        lines.append(f"  {r:>2}×{r:<2}  MSE = {mse:.3e}{tag}")
    lines += [
        "",
        f"Baselines at {target}×{target}:",
        f"  CNN retrained   MSE = {milestone.cnn_retrained_mse_19x19:.3e}",
        f"  CNN zero-shot   MSE = {milestone.cnn_zeroshot_mse_19x19:.3e}",
        "",
        f"RESULT: the operator LOSES by {milestone.transfer_ratio_19x19:.1f}× to a CNN",
        f"retrained at {target}×{target}. The operator transfers without retraining,",
        "but it is not more accurate than a specialist. The value is zero",
        "retraining -- one model, any resolution -- not peak accuracy.",
        "",
        "Benchmark spec: specs/transfer_baseline_compare.spec.md",
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
                            100,
                            500,
                            value=s.default_training_steps,
                            step=50,
                            label="Training steps",
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
            with gr.Tab("Zero-Shot Transfer (honest baseline)"):
                _target = COMMITTED_TARGET_RESOLUTION
                gr.Markdown(
                    "Displays the **committed benchmark**: a model trained on "
                    f"{cfg.transfer.train_resolution}×{cfg.transfer.train_resolution} Poisson "
                    f"data evaluated at {_target}×{_target} without retraining, with "
                    f"MSE = {cfg.transfer.achieved_mse[_target]:.2e}. A discrete CNN retrained "
                    f"at {_target}×{_target} reaches {cfg.transfer.cnn_retrained_mse_19x19:.2e}, "
                    f"so **the operator loses by ≈{cfg.transfer.transfer_ratio_19x19:.0f}×**. "
                    "The value is zero retraining — one model, any resolution — not peak "
                    "accuracy. Source: `results/transfer_baseline_compare.csv`."
                )
                t_show = gr.Button("Show Benchmark Result", variant="primary")
                with gr.Row():
                    t_plot = gr.Image(label="Transfer Benchmark")
                    t_text = gr.Textbox(label="Benchmark Summary", lines=16, interactive=False)

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
