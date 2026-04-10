"""Interactive PDE Solver tab for the AlphaGalerkin dashboard.

Demonstrates Poisson equation solving (∇²φ = ρ) at multiple resolutions,
illustrating the resolution-independence property of the Galerkin operator.
Users can choose a charge pattern, solve, and compare 9×9 / 13×13 / 19×19
side-by-side to see the same physics captured at different scales.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import structlog
from numpy.typing import NDArray

from dashboard.config import DEFAULT_CONFIG, PDEConfig
from dashboard.utils import fig_to_pil, format_exc

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_charge_grid(
    pattern: str,
    n: int,
    cx: float,
    cy: float,
    strength: float,
    cfg: PDEConfig | None = None,
) -> NDArray[np.float32]:
    """Build a (n, n) charge density array for the requested pattern.

    Args:
        pattern: One of the charge pattern names defined in ``PDEConfig``.
        n: Grid dimension (grid is n×n).
        cx: Normalised x position of the primary charge (0–1).
        cy: Normalised y position of the primary charge (0–1).
        strength: Signed amplitude of the primary charge.
        cfg: PDEConfig instance; uses ``DEFAULT_CONFIG.pde`` when *None*.

    Returns:
        Float32 array of shape (n, n) representing charge density ρ.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.pde

    charges: NDArray[np.float32] = np.zeros((n, n), dtype=np.float32)
    n1 = n - 1

    if pattern == "Point Charge":
        charges[round(cx * n1), round(cy * n1)] = float(strength)

    elif pattern == "Dipole":
        mid = n // 2
        charges[mid, max(0, round(n * 0.25))] = float(strength)
        charges[mid, min(n1, round(n * 0.75))] = -float(strength)

    elif pattern == "Quadrupole":
        q = max(1, n // 4)
        charges[q, q] = float(strength)
        charges[q, min(n1, 3 * q)] = -float(strength)
        charges[min(n1, 3 * q), q] = -float(strength)
        charges[min(n1, 3 * q), min(n1, 3 * q)] = float(strength)

    elif pattern == "Ring":
        ci, cj = n // 2, n // 2
        r = max(1, n // 4)
        k = cfg.ring_num_charges
        per_charge = float(strength) / k
        for angle in np.linspace(0, 2 * np.pi, k, endpoint=False):
            xi = int(np.clip(ci + r * np.cos(angle), 0, n1))
            yi = int(np.clip(cj + r * np.sin(angle), 0, n1))
            charges[xi, yi] += per_charge

    elif pattern == "Random":
        rng = np.random.default_rng(42)
        k = max(3, n // 3)
        xs = rng.integers(1, n1, size=k)
        ys = rng.integers(1, n1, size=k)
        vals = rng.uniform(-abs(float(strength)), abs(float(strength)), size=k)
        for i in range(k):
            charges[xs[i], ys[i]] = float(vals[i])

    else:
        logger.warning("unknown_charge_pattern", pattern=pattern)

    logger.debug(
        "charge_grid_created",
        pattern=pattern,
        grid_n=n,
        nonzero=int(np.count_nonzero(charges)),
    )
    return charges


def _poisson_solve(charges: NDArray[np.float32]) -> NDArray[np.float32]:
    """Wrap PoissonSolver.solve for a 2-D charge array.

    Args:
        charges: 2-D float32 array of shape (n, n).

    Returns:
        Potential field of shape (n, n) as float32.

    Raises:
        ImportError: If ``src.physics.poisson`` is not available.
        RuntimeError: If the solver fails.

    """
    from src.physics.poisson import PoissonSolver  # type: ignore[import]

    n = charges.shape[0]
    solver = PoissonSolver(resolution=n)
    result = solver.solve(charges)
    pot2d: NDArray[np.float32] = (
        result if result.ndim == 2 else result.reshape(n, n)
    )
    return pot2d.astype(np.float32)


def _charge_vmax(charges: NDArray[np.float32], cfg: PDEConfig) -> float:
    """Return symmetric colour scale limit for a charge array."""
    return float(max(abs(charges.max()), abs(charges.min()), cfg.epsilon))


# ---------------------------------------------------------------------------
# Public API – functions called by Gradio event handlers
# ---------------------------------------------------------------------------


def solve_and_visualize(
    pattern: str,
    grid_size: int,
    cx: float,
    cy: float,
    strength: float,
    cfg: PDEConfig | None = None,
) -> tuple[PILImage.Image | None, str]:
    """Solve the Poisson equation and produce a side-by-side charge/potential plot.

    Args:
        pattern: Charge pattern name.
        grid_size: Grid dimension N (grid is N×N).
        cx: Normalised x position of the primary charge (0–1).
        cy: Normalised y position of the primary charge (0–1).
        strength: Charge strength.
        cfg: Optional PDEConfig override; uses ``DEFAULT_CONFIG.pde`` when *None*.

    Returns:
        Tuple of (PIL Image or None, metrics string).
        Returns ``(None, error_message)`` on failure.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.pde
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi

    logger.info("pde_solve_requested", pattern=pattern, grid_size=grid_size)
    try:
        n = int(grid_size)
        charges = _make_charge_grid(pattern, n, cx, cy, float(strength), cfg)
        potential = _poisson_solve(charges)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Poisson Equation  ∇²φ = ρ  ·  {n}×{n} grid", fontsize=13)

        vmax = _charge_vmax(charges, cfg)
        im0 = axes[0].imshow(
            charges.T, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax, aspect="equal"
        )
        axes[0].set_title("Charge density  ρ")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].imshow(potential.T, cmap="viridis", origin="lower", aspect="equal")
        axes[1].set_title("Potential  φ  (Galerkin / spectral solve)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        plt.tight_layout()
        img = fig_to_pil(fig, dpi=plot_dpi)

        max_grad = float(np.max(np.abs(np.gradient(potential))))
        metrics = (
            f"Grid: {n}×{n}  |  Tokens (N): {n * n}\n"
            f"Charge range: [{charges.min():.3f}, {charges.max():.3f}]\n"
            f"Potential range: [{potential.min():.5f}, {potential.max():.5f}]\n"
            f"Max |gradient|: {max_grad:.4f}  |  Solver: spectral DST-I"
        )
        logger.info("pde_solve_complete", grid_size=n, max_potential=float(potential.max()))
        return img, metrics

    except Exception as exc:
        msg = format_exc(exc, prefix="PDE solver error")
        logger.exception("pde_solve_failed", pattern=pattern, grid_size=grid_size)
        return None, msg


def compare_resolutions(
    pattern: str,
    strength: float,
    cfg: PDEConfig | None = None,
) -> tuple[PILImage.Image | None, str]:
    """Solve at multiple resolutions and render a comparison grid.

    Args:
        pattern: Charge pattern name.
        strength: Charge strength for all grids.
        cfg: Optional PDEConfig override; uses ``DEFAULT_CONFIG.pde`` when *None*.

    Returns:
        Tuple of (PIL Image or None, summary string).
        Returns ``(None, error_message)`` on failure.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.pde
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi
    sizes = cfg.comparison_sizes

    logger.info("resolution_compare_requested", pattern=pattern, sizes=sizes)
    try:
        from scipy.ndimage import zoom  # type: ignore[import]

        n_cols = len(sizes)
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols + 1, 8))
        fig.suptitle(
            "Resolution Independence — Same Physics at " + " / ".join(f"{s}×{s}" for s in sizes),
            fontsize=13,
        )

        potentials: list[NDArray[np.float32]] = []
        for col, n in enumerate(sizes):
            charges = _make_charge_grid(pattern, n, 0.5, 0.5, float(strength), cfg)
            potential = _poisson_solve(charges)
            potentials.append(potential)

            vmax = _charge_vmax(charges, cfg)
            axes[0, col].imshow(
                charges.T, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax, aspect="equal"
            )
            axes[0, col].set_title(f"ρ  ({n}×{n})")
            axes[0, col].axis("off")

            axes[1, col].imshow(potential.T, cmap="viridis", origin="lower", aspect="equal")
            axes[1, col].set_title(f"φ  ({n}×{n})  N={n * n} tokens")
            axes[1, col].axis("off")

        plt.tight_layout()
        img = fig_to_pil(fig, dpi=plot_dpi)

        # Cross-resolution MSE: upsample each to the largest for comparison
        ref_n = sizes[-1]
        ref_pot = potentials[-1].flatten()
        ref_norm = float(np.linalg.norm(ref_pot)) + cfg.epsilon
        mse_parts: list[str] = []
        for i, n in enumerate(sizes[:-1]):
            scale = ref_n / n
            up = zoom(potentials[i], scale).flatten()[: ref_n * ref_n]
            up_norm = float(np.linalg.norm(up)) + cfg.epsilon
            mse = float(np.mean(((up / up_norm) - (ref_pot / ref_norm)) ** 2))
            mse_parts.append(f"{n}×{n}→{ref_n}×{ref_n} MSE≈{mse:.4f}")

        msg = "Comparison complete.  " + "  |  ".join(mse_parts) if mse_parts else "Complete."
        logger.info("resolution_compare_complete", mse_parts=mse_parts)
        return img, msg

    except Exception as exc:
        msg = format_exc(exc, prefix="Comparison error")
        logger.exception("resolution_compare_failed", pattern=pattern)
        return None, msg


# ---------------------------------------------------------------------------
# Gradio tab builder
# ---------------------------------------------------------------------------


def create_pde_tab(cfg: PDEConfig | None = None) -> None:
    """Create the PDE Solver tab inside an existing ``gr.Blocks`` context.

    Args:
        cfg: Optional PDEConfig override; uses ``DEFAULT_CONFIG.pde`` when *None*.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.pde

    with gr.Tab("PDE Solver"):
        gr.Markdown(
            "## Interactive Poisson Equation Solver\n"
            "**∇²φ = ρ** — charge density drives the potential field.\n\n"
            "The Galerkin neural operator learns the *Green's function* of the Laplacian, "
            "enabling **zero-shot transfer** from any training resolution to any evaluation "
            "resolution.  Use *Resolution Comparison* to see the same physics at "
            + ", ".join(f"{s}×{s}" for s in cfg.comparison_sizes)
            + "."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Charge Configuration")
                pattern = gr.Dropdown(
                    choices=cfg.charge_patterns,
                    value=cfg.default_pattern,
                    label="Charge Pattern",
                )
                grid_size = gr.Dropdown(
                    choices=cfg.grid_sizes,
                    value=cfg.default_grid_size,
                    label="Grid Resolution (N×N)",
                )
                cx = gr.Slider(
                    cfg.position_min, cfg.position_max,
                    value=0.5, step=0.05, label="Charge X Position",
                )
                cy = gr.Slider(
                    cfg.position_min, cfg.position_max,
                    value=0.5, step=0.05, label="Charge Y Position",
                )
                strength = gr.Slider(
                    cfg.strength_min, cfg.strength_max,
                    value=cfg.default_strength, step=0.1, label="Charge Strength",
                )
                solve_btn = gr.Button("Solve Poisson Equation", variant="primary")

            with gr.Column(scale=2):
                solution_img = gr.Image(label="Charge Density & Potential Field")
                metrics_box = gr.Textbox(
                    label="Solution Metrics", lines=4, interactive=False
                )

        gr.Markdown("---")
        gr.Markdown(
            "### Resolution Comparison — Zero-Shot Transfer Demo\n"
            "Solves at "
            + ", ".join(f"{s}×{s}" for s in cfg.comparison_sizes)
            + " with identical physics to demonstrate resolution independence."
        )
        compare_btn = gr.Button(
            "Compare " + " / ".join(f"{s}×{s}" for s in cfg.comparison_sizes)
        )
        compare_img = gr.Image(label="Resolution Comparison")
        compare_status = gr.Textbox(label="", lines=1, interactive=False)

        solve_btn.click(
            solve_and_visualize,
            inputs=[pattern, grid_size, cx, cy, strength],
            outputs=[solution_img, metrics_box],
        )
        compare_btn.click(
            compare_resolutions,
            inputs=[pattern, strength],
            outputs=[compare_img, compare_status],
        )


__all__ = [
    "compare_resolutions",
    "create_pde_tab",
    "solve_and_visualize",
]
