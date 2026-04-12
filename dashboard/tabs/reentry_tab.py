"""Interactive Reentry Thermal Protection System tab for the AlphaGalerkin dashboard.

Demonstrates heat diffusion through a TPS tile during atmospheric reentry,
illustrating how surface heating propagates through the material at multiple
resolutions.  Users can tune thermal diffusivity, surface temperature,
reentry velocity, and simulation time, then compare 9x9 / 13x13 / 19x19
side-by-side to verify resolution independence.
"""

from __future__ import annotations

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import structlog
from numpy.typing import NDArray
from PIL import Image as PILImage

from dashboard.config import DEFAULT_CONFIG, ReentryConfig
from dashboard.utils import fig_to_pil, format_exc

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_initial_temperature(
    n: int,
    interior_temp: float,
) -> NDArray[np.float32]:
    """Build an n x n uniform temperature field at the given interior temperature.

    Args:
        n: Grid dimension (field is n x n).
        interior_temp: Uniform initial temperature in Kelvin.

    Returns:
        Float32 array of shape (n, n) filled with *interior_temp*.

    """
    return np.full((n, n), interior_temp, dtype=np.float32)


def _apply_surface_boundary(
    temp: NDArray[np.float32],
    surface_temp: float,
    velocity: float,
) -> NDArray[np.float32]:
    """Apply a convective heat-flux boundary condition on the top row.

    The top row (``temp[-1, :]``) is set to ``surface_temp * (velocity / 7.5)``
    as a simplified stagnation-point heat-flux proxy that scales linearly with
    reentry velocity relative to a nominal 7.5 km/s reference.

    Args:
        temp: 2-D temperature array of shape (n, n).
        surface_temp: Base surface temperature in Kelvin.
        velocity: Reentry velocity in km/s.

    Returns:
        A copy of *temp* with the top-row boundary applied.

    """
    out = temp.copy()
    out[-1, :] = surface_temp * (velocity / 7.5)
    return out


def _heat_diffusion_step(
    temp: NDArray[np.float32],
    kappa: float,
    dt: float,
    dx: float,
    surface_temp_row: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Advance the temperature field by one explicit finite-difference time step.

    Uses the standard 2-D five-point stencil:

        T_new[i,j] = T[i,j] + kappa * dt / dx**2 *
            (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j])

    Boundary conditions:

    * **Top row** (``i = n-1``): Dirichlet — forced to *surface_temp_row*.
    * **Bottom / left / right edges**: Neumann (zero gradient) — the
      nearest interior value is copied outward.

    Args:
        temp: Current temperature field of shape (n, n).
        kappa: Thermal diffusivity (m^2/s).
        dt: Time step (s).
        dx: Grid spacing (m).
        surface_temp_row: 1-D array of length *n* prescribing the top-row
            Dirichlet values.

    Returns:
        New temperature field of shape (n, n) as float32.

    """
    n = temp.shape[0]
    new = temp.copy()
    coeff = kappa * dt / (dx * dx)

    # Interior update
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            new[i, j] = temp[i, j] + coeff * (
                temp[i + 1, j] + temp[i - 1, j] + temp[i, j + 1] + temp[i, j - 1] - 4.0 * temp[i, j]
            )

    # Neumann (zero gradient) on bottom edge
    new[0, :] = new[1, :]
    # Neumann (zero gradient) on left edge
    new[:, 0] = new[:, 1]
    # Neumann (zero gradient) on right edge
    new[:, -1] = new[:, -2]

    # Dirichlet on top row (surface)
    new[-1, :] = surface_temp_row

    return new.astype(np.float32)


def _simulate_reentry(
    n: int,
    kappa: float,
    surface_temp: float,
    velocity: float,
    total_time: float,
    n_snapshots: int,
    interior_temp: float,
) -> tuple[list[NDArray[np.float32]], NDArray[np.float64]]:
    """Run a full reentry thermal simulation and collect time snapshots.

    The simulation initialises a uniform temperature field, applies a
    velocity-scaled surface boundary, and advances via explicit
    finite-difference with a CFL-limited time step
    (``dt = 0.25 * dx**2 / kappa``).

    Args:
        n: Grid dimension (field is n x n).
        kappa: Thermal diffusivity (m^2/s).
        surface_temp: Base surface temperature (K).
        velocity: Reentry velocity (km/s).
        total_time: Total simulation time (s).
        n_snapshots: Number of evenly spaced snapshots to collect.
        interior_temp: Initial interior temperature (K).

    Returns:
        Tuple of:

        * ``snapshots`` — list of *n_snapshots* temperature arrays.
        * ``times`` — 1-D float64 array of snapshot times.

    """
    dx = 1.0 / (n - 1)
    dt = 0.25 * dx * dx / kappa  # CFL stability limit

    temp = _build_initial_temperature(n, interior_temp)
    temp = _apply_surface_boundary(temp, surface_temp, velocity)

    surface_row = temp[-1, :].copy()

    total_steps = max(1, int(total_time / dt))
    snapshot_interval = max(1, total_steps // n_snapshots)

    snapshots: list[NDArray[np.float32]] = []
    times: list[float] = []

    for step in range(total_steps):
        temp = _heat_diffusion_step(temp, kappa, dt, dx, surface_row)
        if (step + 1) % snapshot_interval == 0 and len(snapshots) < n_snapshots:
            snapshots.append(temp.copy())
            times.append((step + 1) * dt)

    # Always include the final state
    if len(snapshots) < n_snapshots:
        snapshots.append(temp.copy())
        times.append(total_steps * dt)

    logger.debug(
        "reentry_simulation_complete",
        grid_n=n,
        total_steps=total_steps,
        dt=dt,
        snapshots_collected=len(snapshots),
    )
    return snapshots, np.array(times, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API – functions called by Gradio event handlers
# ---------------------------------------------------------------------------


def solve_and_visualize_reentry(
    grid_size: int,
    kappa: float,
    surface_temp: float,
    velocity: float,
    total_time: float,
    cfg: ReentryConfig | None = None,
) -> tuple[PILImage.Image | None, str]:
    """Simulate reentry TPS heating and produce a 2-panel diagnostic plot.

    The left panel shows the final temperature field as a heat-map (``'hot'``
    colour-map), and the right panel shows the centerline temperature profile
    (depth vs. temperature) with a dashed red line at the bondline temperature
    limit.

    Args:
        grid_size: Grid dimension N (field is N x N).
        kappa: Thermal diffusivity (m^2/s).
        surface_temp: Base surface temperature (K).
        velocity: Reentry velocity (km/s).
        total_time: Total simulation time (s).
        cfg: Optional ReentryConfig override; uses
            ``DEFAULT_CONFIG.reentry`` when *None*.

    Returns:
        Tuple of (PIL Image or None, metrics string).
        Returns ``(None, error_message)`` on failure.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.reentry
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi

    logger.info(
        "reentry_solve_requested",
        grid_size=grid_size,
        kappa=kappa,
        velocity=velocity,
    )
    try:
        n = int(grid_size)
        snapshots, times = _simulate_reentry(
            n=n,
            kappa=kappa,
            surface_temp=float(surface_temp),
            velocity=float(velocity),
            total_time=float(total_time),
            n_snapshots=cfg.n_time_snapshots,
            interior_temp=cfg.interior_temp,
        )

        final_temp = snapshots[-1]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        title = (
            f"Reentry TPS \u00b7 {n}\u00d7{n} grid"
            f" \u00b7 \u03ba={kappa:.3f} \u00b7 V={velocity:.1f} km/s"
        )
        fig.suptitle(title, fontsize=13)

        # Left panel — final temperature field
        im0 = axes[0].imshow(
            final_temp,
            cmap="hot",
            origin="lower",
            aspect="equal",
        )
        axes[0].set_title("Final Temperature Field")
        axes[0].set_xlabel("Width")
        axes[0].set_ylabel("Depth")
        plt.colorbar(im0, ax=axes[0], shrink=0.8, label="T (K)")

        # Right panel — centerline temperature profile
        mid_col = n // 2
        centerline = final_temp[:, mid_col]
        depth = np.linspace(0.0, 1.0, n)
        axes[1].plot(centerline, depth, "b-", linewidth=2, label="Centerline T")
        axes[1].axvline(
            x=cfg.bondline_temp_limit,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"Bondline limit ({cfg.bondline_temp_limit:.0f} K)",
        )
        axes[1].set_xlabel("Temperature (K)")
        axes[1].set_ylabel("Depth (normalised)")
        axes[1].set_title("Centerline Temperature Profile")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        img = fig_to_pil(fig, dpi=plot_dpi)

        # Metrics computation
        peak_bondline = float(final_temp[0, :].max())
        surface_gradient = float(np.max(np.abs(np.gradient(final_temp[-1, :]))))

        # Thermal penetration depth: deepest row where T > interior + 10
        penetration_depth = 0.0
        for i in range(n):
            if float(final_temp[i, mid_col]) > cfg.interior_temp + 10.0:
                penetration_depth = float(i) / (n - 1)
                break
        # Invert: depth measured from surface (top = 1.0)
        penetration_depth = 1.0 - penetration_depth if penetration_depth > 0 else 0.0

        metrics = (
            f"Grid: {n}\u00d7{n}  |  Tokens (N): {n * n}\n"
            f"Peak bondline temp: {peak_bondline:.1f} K\n"
            f"Bondline limit: {cfg.bondline_temp_limit:.0f} K\n"
            f"Surface gradient: {surface_gradient:.4f} K/cell\n"
            f"Thermal penetration depth: {penetration_depth:.3f} (normalised)"
        )
        logger.info(
            "reentry_solve_complete",
            grid_size=n,
            peak_bondline=peak_bondline,
        )
        return img, metrics

    except Exception as exc:
        msg = format_exc(exc, prefix="TPS solver error")
        logger.exception("reentry_solve_failed", grid_size=grid_size)
        return None, msg


def compare_resolutions_reentry(
    kappa: float,
    surface_temp: float,
    velocity: float,
    cfg: ReentryConfig | None = None,
) -> tuple[PILImage.Image | None, str]:
    """Solve at multiple resolutions and render a side-by-side comparison.

    Runs the reentry simulation at each resolution in ``cfg.comparison_sizes``
    and displays the final temperature fields in a single row.  Cross-resolution
    MSE is computed by upsampling smaller fields to the largest resolution via
    ``scipy.ndimage.zoom``.

    Args:
        kappa: Thermal diffusivity (m^2/s).
        surface_temp: Base surface temperature (K).
        velocity: Reentry velocity (km/s).
        cfg: Optional ReentryConfig override; uses
            ``DEFAULT_CONFIG.reentry`` when *None*.

    Returns:
        Tuple of (PIL Image or None, MSE summary string).
        Returns ``(None, error_message)`` on failure.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.reentry
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi
    sizes = cfg.comparison_sizes

    logger.info(
        "reentry_resolution_compare_requested",
        sizes=sizes,
        kappa=kappa,
        velocity=velocity,
    )
    try:
        from scipy.ndimage import zoom  # type: ignore[import]

        n_cols = len(sizes)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols + 1, 4))
        fig.suptitle(
            "Resolution Independence \u2014 Reentry TPS at "
            + " / ".join(f"{s}\u00d7{s}" for s in sizes),
            fontsize=13,
        )

        # Handle the single-column edge case
        if n_cols == 1:
            axes = [axes]

        finals: list[NDArray[np.float32]] = []
        for col, n in enumerate(sizes):
            snapshots, _times = _simulate_reentry(
                n=n,
                kappa=kappa,
                surface_temp=float(surface_temp),
                velocity=float(velocity),
                total_time=float(cfg.default_total_time),
                n_snapshots=cfg.n_time_snapshots,
                interior_temp=cfg.interior_temp,
            )
            final = snapshots[-1]
            finals.append(final)

            axes[col].imshow(final, cmap="hot", origin="lower", aspect="equal")
            axes[col].set_title(f"T  ({n}\u00d7{n})  N={n * n}")
            axes[col].axis("off")

        plt.tight_layout()
        img = fig_to_pil(fig, dpi=plot_dpi)

        # Cross-resolution MSE: upsample each to the largest for comparison.
        ref_field = finals[-1]
        ref_shape = ref_field.shape
        ref_flat = ref_field.flatten()
        ref_norm = float(np.linalg.norm(ref_flat)) + cfg.epsilon
        mse_parts: list[str] = []
        for i, n in enumerate(sizes[:-1]):
            src_shape = finals[i].shape
            scale = (ref_shape[0] / src_shape[0], ref_shape[1] / src_shape[1])
            up_2d = zoom(finals[i], scale)
            if up_2d.shape != ref_shape:
                raise ValueError(
                    f"Upsampled field shape mismatch for {n}\u00d7{n}: "
                    f"expected {ref_shape}, got {up_2d.shape}"
                )
            up_flat = up_2d.flatten()
            up_norm = float(np.linalg.norm(up_flat)) + cfg.epsilon
            mse = float(np.mean(((up_flat / up_norm) - (ref_flat / ref_norm)) ** 2))
            mse_parts.append(
                f"{n}\u00d7{n}\u2192{ref_shape[0]}\u00d7{ref_shape[1]} MSE\u2248{mse:.4f}"
            )

        msg = "Comparison complete.  " + "  |  ".join(mse_parts) if mse_parts else "Complete."
        logger.info("reentry_resolution_compare_complete", mse_parts=mse_parts)
        return img, msg

    except Exception as exc:
        msg = format_exc(exc, prefix="Comparison error")
        logger.exception("reentry_resolution_compare_failed")
        return None, msg


# ---------------------------------------------------------------------------
# Gradio tab builder
# ---------------------------------------------------------------------------


def create_reentry_tab(cfg: ReentryConfig | None = None) -> None:
    """Create the Reentry TPS tab inside an existing ``gr.Blocks`` context.

    Args:
        cfg: Optional ReentryConfig override; uses
            ``DEFAULT_CONFIG.reentry`` when *None*.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.reentry

    with gr.Tab("Reentry TPS"):
        gr.Markdown(
            "## Reentry Thermal Protection System Analysis\n"
            "Simulates heat diffusion through a TPS tile during atmospheric reentry.\n\n"
            "Surface heating is modelled as a velocity-scaled Dirichlet boundary, "
            "and the interior evolves via explicit finite-difference diffusion.  "
            "Use *Resolution Comparison* to verify resolution independence at "
            + ", ".join(f"{s}\u00d7{s}" for s in cfg.comparison_sizes)
            + "."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### TPS Configuration")
                grid_size = gr.Dropdown(
                    choices=cfg.grid_sizes,
                    value=cfg.default_grid_size,
                    label="Grid Resolution (N\u00d7N)",
                )
                kappa = gr.Slider(
                    cfg.kappa_min,
                    cfg.kappa_max,
                    value=cfg.default_kappa,
                    step=0.01,
                    label="Thermal Diffusivity \u03ba (m\u00b2/s)",
                )
                surface_temp = gr.Slider(
                    cfg.surface_temp_min,
                    cfg.surface_temp_max,
                    value=cfg.default_surface_temp,
                    step=50.0,
                    label="Surface Temperature (K)",
                )
                velocity = gr.Slider(
                    cfg.velocity_min,
                    cfg.velocity_max,
                    value=cfg.default_velocity,
                    step=0.5,
                    label="Reentry Velocity (km/s)",
                )
                total_time = gr.Slider(
                    cfg.total_time_min,
                    cfg.total_time_max,
                    value=cfg.default_total_time,
                    step=0.1,
                    label="Simulation Time (s)",
                )
                solve_btn = gr.Button("Simulate TPS Heating", variant="primary")

            with gr.Column(scale=2):
                solution_img = gr.Image(label="Temperature Field & Profile")
                metrics_box = gr.Textbox(label="TPS Metrics", lines=5, interactive=False)

        gr.Markdown("---")
        gr.Markdown(
            "### Resolution Comparison \u2014 Zero-Shot Transfer Demo\n"
            "Solves at "
            + ", ".join(f"{s}\u00d7{s}" for s in cfg.comparison_sizes)
            + " with identical physics to demonstrate resolution independence."
        )
        compare_btn = gr.Button(
            "Compare " + " / ".join(f"{s}\u00d7{s}" for s in cfg.comparison_sizes)
        )
        compare_img = gr.Image(label="Resolution Comparison")
        compare_status = gr.Textbox(label="", lines=1, interactive=False)

        solve_btn.click(
            solve_and_visualize_reentry,
            inputs=[grid_size, kappa, surface_temp, velocity, total_time],
            outputs=[solution_img, metrics_box],
        )
        compare_btn.click(
            compare_resolutions_reentry,
            inputs=[kappa, surface_temp, velocity],
            outputs=[compare_img, compare_status],
        )


__all__ = [
    "compare_resolutions_reentry",
    "create_reentry_tab",
    "solve_and_visualize_reentry",
]
