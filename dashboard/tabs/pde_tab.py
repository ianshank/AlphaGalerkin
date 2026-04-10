"""Interactive PDE Solver tab for AlphaGalerkin dashboard.

Demonstrates Poisson equation solving at multiple resolutions,
illustrating the resolution-independence property of the Galerkin operator.
"""

from __future__ import annotations

import io

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage


def _make_charge_grid(
    pattern: str,
    n: int,
    cx: float,
    cy: float,
    strength: float,
) -> NDArray[np.float32]:
    """Build a charge density grid for the requested pattern."""
    charges = np.zeros((n, n), dtype=np.float32)
    n1 = n - 1

    if pattern == "Point Charge":
        charges[round(cx * n1), round(cy * n1)] = strength

    elif pattern == "Dipole":
        ix = n // 2
        charges[ix, round(n * 0.25)] = strength
        charges[ix, round(n * 0.75)] = -strength

    elif pattern == "Quadrupole":
        q = max(1, n // 4)
        charges[q, q] = strength
        charges[q, 3 * q] = -strength
        charges[3 * q, q] = -strength
        charges[3 * q, 3 * q] = strength

    elif pattern == "Ring":
        cx_i, cy_i = n // 2, n // 2
        r = max(1, n // 4)
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            xi = int(np.clip(cx_i + r * np.cos(angle), 0, n1))
            yi = int(np.clip(cy_i + r * np.sin(angle), 0, n1))
            charges[xi, yi] += strength / 8.0

    elif pattern == "Random":
        rng = np.random.default_rng(42)
        k = max(3, n // 3)
        xs = rng.integers(1, n1, size=k)
        ys = rng.integers(1, n1, size=k)
        vals = rng.uniform(-abs(strength), abs(strength), size=k)
        for i in range(k):
            charges[xs[i], ys[i]] = float(vals[i])

    return charges


def _solve(charges: NDArray[np.float32]) -> NDArray[np.float32]:
    from src.physics.poisson import PoissonSolver

    solver = PoissonSolver(resolution=charges.shape[0])
    return solver.solve(charges)


def _fig_to_pil(fig: plt.Figure) -> PILImage.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


def solve_and_visualize(
    pattern: str,
    grid_size: int,
    cx: float,
    cy: float,
    strength: float,
) -> tuple[PILImage.Image | None, str]:
    """Solve Poisson equation and produce side-by-side charge / potential plot."""
    try:
        n = int(grid_size)
        charges = _make_charge_grid(pattern, n, cx, cy, float(strength))
        potential = _solve(charges)

        pot2d = potential if potential.ndim == 2 else potential.reshape(n, n)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Poisson Equation  ∇²φ = ρ  ·  {n}×{n} grid", fontsize=13)

        vmax = max(abs(charges.max()), abs(charges.min()), 1e-9)
        im0 = axes[0].imshow(
            charges.T, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax, aspect="equal"
        )
        axes[0].set_title("Charge density  ρ")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].imshow(pot2d.T, cmap="viridis", origin="lower", aspect="equal")
        axes[1].set_title("Potential  φ  (Galerkin / spectral solve)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        plt.tight_layout()
        img = _fig_to_pil(fig)

        metrics = (
            f"Grid: {n}×{n}  |  Tokens (N): {n * n}\n"
            f"Charge range: [{charges.min():.3f}, {charges.max():.3f}]\n"
            f"Potential range: [{pot2d.min():.5f}, {pot2d.max():.5f}]\n"
            f"Solver: spectral DST-I  |  Complexity: O(N log N)"
        )
        return img, metrics

    except Exception as exc:
        return None, f"Error: {exc}"


def compare_resolutions(pattern: str, strength: float) -> tuple[PILImage.Image | None, str]:
    """Solve at 9×9, 13×13, 19×19 and render a comparison grid."""
    try:
        sizes = [9, 13, 19]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        fig.suptitle(
            "Resolution Independence — Same Physics at 9×9, 13×13, 19×19", fontsize=13
        )

        for col, n in enumerate(sizes):
            charges = _make_charge_grid(pattern, n, 0.5, 0.5, float(strength))
            potential = _solve(charges)
            pot2d = potential if potential.ndim == 2 else potential.reshape(n, n)

            vmax = max(abs(charges.max()), abs(charges.min()), 1e-9)
            axes[0, col].imshow(
                charges.T, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax, aspect="equal"
            )
            axes[0, col].set_title(f"ρ   ({n}×{n})")
            axes[0, col].axis("off")

            axes[1, col].imshow(pot2d.T, cmap="viridis", origin="lower", aspect="equal")
            axes[1, col].set_title(f"φ   ({n}×{n})  N={n * n} tokens")
            axes[1, col].axis("off")

        plt.tight_layout()
        img = _fig_to_pil(fig)
        mses = []
        # Compute cross-resolution MSE as a resolution-independence metric
        ref_charges = _make_charge_grid(pattern, 19, 0.5, 0.5, float(strength))
        ref_pot = _solve(ref_charges)
        ref_flat = ref_pot.flatten() / (np.linalg.norm(ref_pot.flatten()) + 1e-9)
        for n in [9, 13]:
            c = _make_charge_grid(pattern, n, 0.5, 0.5, float(strength))
            p = _solve(c).flatten()
            # Upsample to 19 for rough comparison
            from scipy.ndimage import zoom

            scale = 19 / n
            p_up = zoom(p.reshape(n, n), scale).flatten()[: 19 * 19]
            p_up /= np.linalg.norm(p_up) + 1e-9
            mse = float(np.mean((p_up - ref_flat) ** 2))
            mses.append(f"{n}×{n}→19×19 MSE≈{mse:.4f}")

        msg = "Comparison complete.  " + "  |  ".join(mses)
        return img, msg

    except Exception as exc:
        return None, f"Error: {exc}"


def create_pde_tab() -> None:
    """Create the PDE Solver tab inside an existing gr.Blocks context."""
    with gr.Tab("PDE Solver"):
        gr.Markdown(
            """
## Interactive Poisson Equation Solver
**∇²φ = ρ** — charge density drives the potential field.

The Galerkin neural operator learns the *Green's function* of the Laplacian,
enabling **zero-shot transfer** from any training resolution to any evaluation resolution.
Use the *Resolution Comparison* section to see the same physics at 9×9, 13×13, and 19×19.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Charge Configuration")
                pattern = gr.Dropdown(
                    choices=["Point Charge", "Dipole", "Quadrupole", "Ring", "Random"],
                    value="Point Charge",
                    label="Charge Pattern",
                )
                grid_size = gr.Dropdown(
                    choices=[9, 13, 19, 25, 32],
                    value=9,
                    label="Grid Resolution (N×N)",
                )
                cx = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Charge X Position")
                cy = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Charge Y Position")
                strength = gr.Slider(
                    -2.0, 2.0, value=1.0, step=0.1, label="Charge Strength"
                )
                solve_btn = gr.Button("Solve Poisson Equation", variant="primary")

            with gr.Column(scale=2):
                solution_img = gr.Image(label="Charge Density & Potential Field")
                metrics_box = gr.Textbox(
                    label="Solution Metrics", lines=4, interactive=False
                )

        gr.Markdown("---")
        gr.Markdown(
            "### Resolution Comparison  —  Zero-Shot Transfer Demo\n"
            "Solves at 9×9, 13×13, and 19×19 with identical physics to demonstrate "
            "resolution independence."
        )
        compare_btn = gr.Button("Compare 9×9 / 13×13 / 19×19")
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
