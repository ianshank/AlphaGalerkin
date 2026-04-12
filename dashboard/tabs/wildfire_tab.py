"""Wildfire Spread Simulation tab for the AlphaGalerkin dashboard.

Demonstrates an advection-diffusion fire spread model on a 2-D grid,
illustrating how wind-driven transport and thermal diffusion govern
wildfire propagation.  Users can choose an ignition pattern, set wind
and fuel parameters, and compare solutions at multiple resolutions to
verify resolution-independence of the underlying operator.
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

from dashboard.config import DEFAULT_CONFIG, WildfireConfig
from dashboard.utils import fig_to_pil, format_exc

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_ignition_field(
    pattern: str,
    n: int,
    cfg: WildfireConfig | None = None,
) -> NDArray[np.float32]:
    """Build an (n, n) initial temperature field for the requested ignition pattern.

    Args:
        pattern: One of the ignition pattern names defined in ``WildfireConfig``.
        n: Grid dimension (grid is n x n).
        cfg: WildfireConfig instance; uses ``DEFAULT_CONFIG.wildfire`` when *None*.

    Returns:
        Float32 array of shape (n, n) representing the initial temperature field.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.wildfire

    temp: NDArray[np.float32] = np.zeros((n, n), dtype=np.float32)

    if pattern == "Center":
        sigma = n / 8.0
        cx, cy = n / 2.0, n / 2.0
        yy, xx = np.mgrid[0:n, 0:n]
        temp = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2)).astype(np.float32)

    elif pattern == "Edge":
        temp[0, :] = 1.0

    elif pattern == "Corner":
        sigma = n / 6.0
        yy, xx = np.mgrid[0:n, 0:n]
        temp = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)).astype(np.float32)

    elif pattern == "Line":
        for i in range(n):
            temp[i, i] = 1.0

    elif pattern == "Random":
        rng = np.random.default_rng(42)
        for _ in range(5):
            ri = rng.integers(0, n)
            ci = rng.integers(0, n)
            sigma = n / 10.0
            yy, xx = np.mgrid[0:n, 0:n]
            temp += np.exp(-((xx - ci) ** 2 + (yy - ri) ** 2) / (2.0 * sigma**2)).astype(np.float32)
        temp = np.clip(temp, 0.0, 1.0).astype(np.float32)

    else:
        logger.warning("unknown_ignition_pattern", pattern=pattern)

    logger.debug(
        "ignition_field_created",
        pattern=pattern,
        grid_n=n,
        max_temp=float(temp.max()),
    )
    return temp


def _build_fuel_field(
    n: int,
    fuel_density: float,
    seed: int = 42,
) -> NDArray[np.float32]:
    """Build an (n, n) fuel density field with smooth spatial variation.

    Args:
        n: Grid dimension (grid is n x n).
        fuel_density: Base fuel density value.
        seed: Random seed for reproducible noise.

    Returns:
        Float32 array of shape (n, n) representing fuel density, clipped to
        [0.1, 2 * fuel_density].

    """
    from scipy.ndimage import gaussian_filter  # type: ignore[import]

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n)).astype(np.float32)
    smooth_noise = gaussian_filter(noise, sigma=n / 6.0).astype(np.float32)
    # Normalise smooth noise to [-0.5, 0.5] range then scale
    nmin, nmax = float(smooth_noise.min()), float(smooth_noise.max())
    if nmax - nmin > 0:
        smooth_noise = (smooth_noise - nmin) / (nmax - nmin) - 0.5
    fuel = np.full((n, n), fuel_density, dtype=np.float32) + smooth_noise * fuel_density
    fuel = np.clip(fuel, 0.1, 2.0 * fuel_density).astype(np.float32)

    logger.debug(
        "fuel_field_created",
        grid_n=n,
        fuel_range=(float(fuel.min()), float(fuel.max())),
    )
    return fuel


def _advection_diffusion_step(
    temp: NDArray[np.float32],
    fuel: NDArray[np.float32],
    wind_vx: float,
    wind_vy: float,
    diffusion: float,
    dt: float,
    ignition_threshold: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Advance the fire model by one time step using advection-diffusion-combustion.

    The update applies:
    * **Advection** -- first-order upwind finite differences for wind transport.
    * **Diffusion** -- central difference 5-point Laplacian.
    * **Combustion** -- where temperature exceeds *ignition_threshold* and fuel
      remains above 0.01, heat is added proportional to fuel and fuel is consumed.

    Zero-flux (Neumann) boundary conditions are enforced on all edges after
    the update.

    Args:
        temp: Current temperature field of shape (n, n).
        fuel: Current fuel field of shape (n, n).
        wind_vx: Wind velocity x-component.
        wind_vy: Wind velocity y-component.
        diffusion: Thermal diffusion coefficient.
        dt: Time step.
        ignition_threshold: Temperature above which combustion occurs.

    Returns:
        Tuple of (new_temp, new_fuel) arrays, each of shape (n, n).

    """
    n = temp.shape[0]
    dx = 1.0 / max(n - 1, 1)
    new_temp = temp.copy()
    new_fuel = fuel.copy()

    # ---- Advection (first-order upwind) ----
    advection = np.zeros_like(temp)
    if wind_vx >= 0:
        # backward difference in x
        advection[1:, :] -= wind_vx * (temp[1:, :] - temp[:-1, :]) / dx
        advection[0, :] = advection[1, :] if n > 1 else 0.0
    else:
        # forward difference in x
        advection[:-1, :] -= wind_vx * (temp[1:, :] - temp[:-1, :]) / dx
        advection[-1, :] = advection[-2, :] if n > 1 else 0.0

    if wind_vy >= 0:
        # backward difference in y
        advection[:, 1:] -= wind_vy * (temp[:, 1:] - temp[:, :-1]) / dx
        advection[:, 0] = advection[:, 1] if n > 1 else 0.0
    else:
        # forward difference in y
        advection[:, :-1] -= wind_vy * (temp[:, 1:] - temp[:, :-1]) / dx
        advection[:, -1] = advection[:, -2] if n > 1 else 0.0

    # ---- Diffusion (5-point Laplacian, central differences) ----
    laplacian = np.zeros_like(temp)
    laplacian[1:-1, :] += temp[2:, :] + temp[:-2, :] - 2.0 * temp[1:-1, :]
    laplacian[:, 1:-1] += temp[:, 2:] + temp[:, :-2] - 2.0 * temp[:, 1:-1]
    laplacian /= dx**2

    new_temp += dt * (advection + diffusion * laplacian)

    # ---- Combustion ----
    burn_rate = 0.5
    burning = (temp > ignition_threshold) & (fuel > 0.01)
    heat_release = burn_rate * fuel * burning.astype(np.float32)
    new_temp += dt * heat_release
    new_fuel -= dt * burn_rate * burning.astype(np.float32) * fuel

    new_fuel = np.clip(new_fuel, 0.0, None).astype(np.float32)
    new_temp = np.clip(new_temp, 0.0, None).astype(np.float32)

    # ---- Zero-flux (Neumann) boundary conditions ----
    new_temp[0, :] = new_temp[1, :] if n > 1 else new_temp[0, :]
    new_temp[-1, :] = new_temp[-2, :] if n > 1 else new_temp[-1, :]
    new_temp[:, 0] = new_temp[:, 1] if n > 1 else new_temp[:, 0]
    new_temp[:, -1] = new_temp[:, -2] if n > 1 else new_temp[:, -1]

    return new_temp, new_fuel


def _simulate_wildfire(
    n: int,
    wind_speed: float,
    wind_direction: float,
    diffusion: float,
    fuel_density: float,
    ignition_pattern: str,
    total_time: float,
    n_snapshots: int,
    cfg: WildfireConfig | None = None,
) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]], NDArray[np.float64]]:
    """Run the full wildfire advection-diffusion simulation.

    Args:
        n: Grid dimension (grid is n x n).
        wind_speed: Wind speed magnitude.
        wind_direction: Wind direction in degrees (0 = East, 90 = North).
        diffusion: Thermal diffusion coefficient.
        fuel_density: Base fuel density.
        ignition_pattern: Name of the ignition pattern.
        total_time: Total simulation time.
        n_snapshots: Number of snapshots to record.
        cfg: WildfireConfig instance; uses ``DEFAULT_CONFIG.wildfire`` when *None*.

    Returns:
        Tuple of (temp_snapshots, fuel_snapshots, time_array) where each
        snapshot list has *n_snapshots* entries and *time_array* has shape
        ``(n_snapshots,)``.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.wildfire

    eps = cfg.epsilon
    direction_rad = np.deg2rad(wind_direction)
    vx = wind_speed * np.cos(direction_rad)
    vy = wind_speed * np.sin(direction_rad)

    dx = 1.0 / max(n - 1, 1)

    # CFL-limited time step
    dt_diff = 0.25 * dx**2 / max(diffusion, eps)
    dt_adv = 0.5 * dx / max(abs(vx), abs(vy), eps)
    dt = min(dt_diff, dt_adv)

    temp = _build_ignition_field(ignition_pattern, n, cfg)
    fuel = _build_fuel_field(n, fuel_density)

    n_steps = max(1, int(np.ceil(total_time / dt)))
    snapshot_interval = max(1, n_steps // n_snapshots)

    temp_snapshots: list[NDArray[np.float32]] = [temp.copy()]
    fuel_snapshots: list[NDArray[np.float32]] = [fuel.copy()]
    time_points: list[float] = [0.0]

    for step in range(1, n_steps + 1):
        temp, fuel = _advection_diffusion_step(
            temp, fuel, vx, vy, diffusion, dt, cfg.ignition_threshold
        )
        if step % snapshot_interval == 0 or step == n_steps:
            temp_snapshots.append(temp.copy())
            fuel_snapshots.append(fuel.copy())
            time_points.append(step * dt)

    # Trim to requested number of snapshots (keep first + last)
    while len(temp_snapshots) > n_snapshots:
        mid = len(temp_snapshots) // 2
        temp_snapshots.pop(mid)
        fuel_snapshots.pop(mid)
        time_points.pop(mid)

    time_array = np.array(time_points, dtype=np.float64)

    logger.debug(
        "wildfire_simulation_complete",
        grid_n=n,
        total_steps=n_steps,
        dt=dt,
        n_snapshots=len(temp_snapshots),
    )
    return temp_snapshots, fuel_snapshots, time_array


# ---------------------------------------------------------------------------
# Public API -- functions called by Gradio event handlers
# ---------------------------------------------------------------------------


def solve_and_visualize_wildfire(
    grid_size: int,
    wind_speed: float,
    wind_direction: float,
    diffusion: float,
    fuel_density: float,
    ignition_pattern: str,
    total_time: float,
    cfg: WildfireConfig | None = None,
) -> tuple[PILImage.Image | None, str]:
    """Run the wildfire simulation and produce a 2x2 diagnostic plot.

    The four panels show:

    * **[0,0]** Initial ignition temperature (``hot`` colourmap).
    * **[0,1]** Final temperature with a wind arrow overlay (``hot`` colourmap).
    * **[1,0]** Burned area fraction ``1 - fuel/initial_fuel`` (``YlOrRd`` colourmap).
    * **[1,1]** Burned fraction vs time line plot (burn progression curve).

    Args:
        grid_size: Grid dimension N (grid is N x N).
        wind_speed: Wind speed magnitude.
        wind_direction: Wind direction in degrees.
        diffusion: Thermal diffusion coefficient.
        fuel_density: Base fuel density.
        ignition_pattern: Name of the ignition pattern.
        total_time: Total simulation time.
        cfg: Optional WildfireConfig override; uses ``DEFAULT_CONFIG.wildfire``
            when *None*.

    Returns:
        Tuple of (PIL Image or None, metrics string).
        Returns ``(None, error_message)`` on failure.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.wildfire
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi

    logger.info(
        "wildfire_solve_requested",
        pattern=ignition_pattern,
        grid_size=grid_size,
    )
    try:
        n = int(grid_size)
        temp_snaps, fuel_snaps, time_arr = _simulate_wildfire(
            n,
            wind_speed,
            wind_direction,
            diffusion,
            fuel_density,
            ignition_pattern,
            total_time,
            cfg.n_time_snapshots,
            cfg,
        )

        initial_fuel = fuel_snaps[0]
        initial_temp = temp_snaps[0]
        final_temp = temp_snaps[-1]
        final_fuel = fuel_snaps[-1]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        direction_rad = np.deg2rad(wind_direction)

        fig.suptitle(
            f"Wildfire Spread  {n}x{n} grid  |  "
            f"Wind: {wind_speed:.1f} m/s @ {wind_direction:.0f} deg",
            fontsize=13,
        )

        # [0,0] Initial ignition
        im00 = axes[0, 0].imshow(
            initial_temp.T,
            cmap="hot",
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=max(float(initial_temp.max()), 0.01),
        )
        axes[0, 0].set_title("Initial Ignition")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        plt.colorbar(im00, ax=axes[0, 0], shrink=0.8)

        # [0,1] Final temperature with wind arrow
        im01 = axes[0, 1].imshow(
            final_temp.T,
            cmap="hot",
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=max(float(final_temp.max()), 0.01),
        )
        axes[0, 1].set_title("Final Temperature")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("y")
        plt.colorbar(im01, ax=axes[0, 1], shrink=0.8)

        # Wind direction arrow
        arrow_len = n * 0.15
        cx_arrow, cy_arrow = n * 0.85, n * 0.85
        dx_arrow = arrow_len * np.cos(direction_rad)
        dy_arrow = arrow_len * np.sin(direction_rad)
        axes[0, 1].annotate(
            "Wind",
            xy=(cx_arrow + dx_arrow, cy_arrow + dy_arrow),
            xytext=(cx_arrow, cy_arrow),
            arrowprops={"arrowstyle": "->", "color": "cyan", "lw": 2},
            color="cyan",
            fontsize=9,
            fontweight="bold",
        )

        # [1,0] Burned area
        safe_initial_fuel = np.where(initial_fuel > cfg.epsilon, initial_fuel, cfg.epsilon)
        burned_fraction_map = 1.0 - final_fuel / safe_initial_fuel
        burned_fraction_map = np.clip(burned_fraction_map, 0.0, 1.0)

        im10 = axes[1, 0].imshow(
            burned_fraction_map.T,
            cmap="YlOrRd",
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, 0].set_title("Burned Area")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("y")
        plt.colorbar(im10, ax=axes[1, 0], shrink=0.8)

        # [1,1] Burn progression over time
        burn_fractions: list[float] = []
        for fs in fuel_snaps:
            total_initial = float(initial_fuel.sum()) + cfg.epsilon
            total_remaining = float(fs.sum())
            burn_fractions.append(1.0 - total_remaining / total_initial)

        axes[1, 1].plot(time_arr[: len(burn_fractions)], burn_fractions, "r-o", lw=2)
        axes[1, 1].set_title("Burn Progression")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Burned Fraction")
        axes[1, 1].set_ylim(-0.05, 1.05)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        img = fig_to_pil(fig, dpi=plot_dpi)

        # Metrics summary
        total_burned = burn_fractions[-1] if burn_fractions else 0.0
        max_temp = float(final_temp.max())

        # Time to 50% burn
        time_50_str = "not reached"
        for idx, bf in enumerate(burn_fractions):
            if bf >= 0.5:
                time_50_str = f"{time_arr[idx]:.3f}"
                break

        metrics = (
            f"Grid: {n}x{n}  |  Tokens (N): {n * n}\n"
            f"Total burned fraction: {total_burned:.4f}\n"
            f"Max temperature: {max_temp:.4f}\n"
            f"Time to 50% burn: {time_50_str}\n"
            f"Wind: {wind_speed:.1f} m/s @ {wind_direction:.0f} deg"
        )
        logger.info(
            "wildfire_solve_complete",
            grid_size=n,
            total_burned=total_burned,
            max_temp=max_temp,
        )
        return img, metrics

    except Exception as exc:
        msg = format_exc(exc, prefix="Wildfire simulation error")
        logger.exception(
            "wildfire_solve_failed",
            pattern=ignition_pattern,
            grid_size=grid_size,
        )
        return None, msg


def compare_resolutions_wildfire(
    wind_speed: float,
    wind_direction: float,
    diffusion: float,
    fuel_density: float,
    ignition_pattern: str,
    cfg: WildfireConfig | None = None,
) -> tuple[PILImage.Image | None, str]:
    """Solve at multiple resolutions and render a comparison grid.

    Args:
        wind_speed: Wind speed magnitude.
        wind_direction: Wind direction in degrees.
        diffusion: Thermal diffusion coefficient.
        fuel_density: Base fuel density.
        ignition_pattern: Name of the ignition pattern.
        cfg: Optional WildfireConfig override; uses ``DEFAULT_CONFIG.wildfire``
            when *None*.

    Returns:
        Tuple of (PIL Image or None, MSE summary string).
        Returns ``(None, error_message)`` on failure.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.wildfire
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi
    sizes = cfg.comparison_sizes

    logger.info(
        "wildfire_resolution_compare_requested",
        pattern=ignition_pattern,
        sizes=sizes,
    )
    try:
        from scipy.ndimage import zoom  # type: ignore[import]

        n_cols = len(sizes)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols + 1, 4))
        fig.suptitle(
            "Resolution Independence -- Final Temperature at "
            + " / ".join(f"{s}x{s}" for s in sizes),
            fontsize=13,
        )

        # Handle single-column case
        if n_cols == 1:
            axes = [axes]

        finals: list[NDArray[np.float32]] = []
        for col, n in enumerate(sizes):
            temp_snaps, _fuel_snaps, _time_arr = _simulate_wildfire(
                n,
                wind_speed,
                wind_direction,
                diffusion,
                fuel_density,
                ignition_pattern,
                cfg.default_total_time,
                cfg.n_time_snapshots,
                cfg,
            )
            final = temp_snaps[-1]
            finals.append(final)

            axes[col].imshow(
                final.T,
                cmap="hot",
                origin="lower",
                aspect="equal",
                vmin=0.0,
                vmax=max(float(final.max()), 0.01),
            )
            axes[col].set_title(f"T final ({n}x{n})  N={n * n}")
            axes[col].axis("off")

        plt.tight_layout()
        img = fig_to_pil(fig, dpi=plot_dpi)

        # Cross-resolution MSE: upsample each to the largest for comparison.
        ref_2d = finals[-1]
        ref_shape = ref_2d.shape
        ref_flat = ref_2d.flatten()
        ref_norm = float(np.linalg.norm(ref_flat)) + cfg.epsilon
        mse_parts: list[str] = []
        for i, n in enumerate(sizes[:-1]):
            src_shape = finals[i].shape
            scale = (ref_shape[0] / src_shape[0], ref_shape[1] / src_shape[1])
            up_2d = zoom(finals[i], scale)
            if up_2d.shape != ref_shape:
                raise ValueError(
                    f"Upsampled shape mismatch for {n}x{n}: expected {ref_shape}, got {up_2d.shape}"
                )
            up_flat = up_2d.flatten()
            up_norm = float(np.linalg.norm(up_flat)) + cfg.epsilon
            mse = float(np.mean(((up_flat / up_norm) - (ref_flat / ref_norm)) ** 2))
            mse_parts.append(f"{n}x{n}->{ref_shape[0]}x{ref_shape[1]} MSE={mse:.4f}")

        msg = "Comparison complete.  " + "  |  ".join(mse_parts) if mse_parts else "Complete."
        logger.info("wildfire_resolution_compare_complete", mse_parts=mse_parts)
        return img, msg

    except Exception as exc:
        msg = format_exc(exc, prefix="Comparison error")
        logger.exception("wildfire_resolution_compare_failed", pattern=ignition_pattern)
        return None, msg


# ---------------------------------------------------------------------------
# Gradio tab builder
# ---------------------------------------------------------------------------


def create_wildfire_tab(cfg: WildfireConfig | None = None) -> None:
    """Create the Wildfire Spread Simulation tab inside an existing ``gr.Blocks`` context.

    Args:
        cfg: Optional WildfireConfig override; uses ``DEFAULT_CONFIG.wildfire``
            when *None*.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.wildfire

    with gr.Tab("Wildfire Spread"):
        gr.Markdown(
            "## Wildfire Spread Simulation\n"
            "**Advection-Diffusion Fire Model** -- wind-driven transport and thermal "
            "diffusion govern fire propagation across a fuel-laden terrain.\n\n"
            "The simulation couples advection (wind), diffusion (heat conduction), and "
            "combustion (fuel consumption) on a 2-D grid.  Use *Resolution Comparison* "
            "to verify that the physics is captured consistently at "
            + ", ".join(f"{s}x{s}" for s in cfg.comparison_sizes)
            + " resolutions."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Fire Configuration")
                grid_size = gr.Dropdown(
                    choices=cfg.grid_sizes,
                    value=cfg.default_grid_size,
                    label="Grid Resolution (N x N)",
                )
                ignition_pattern = gr.Dropdown(
                    choices=cfg.ignition_patterns,
                    value=cfg.default_ignition,
                    label="Ignition Pattern",
                )
                wind_speed = gr.Slider(
                    cfg.wind_speed_min,
                    cfg.wind_speed_max,
                    value=cfg.default_wind_speed,
                    step=0.5,
                    label="Wind Speed (m/s)",
                )
                wind_direction = gr.Slider(
                    cfg.wind_direction_min,
                    cfg.wind_direction_max,
                    value=cfg.default_wind_direction,
                    step=5.0,
                    label="Wind Direction (degrees)",
                )
                diffusion_slider = gr.Slider(
                    cfg.diffusion_min,
                    cfg.diffusion_max,
                    value=cfg.default_diffusion,
                    step=0.01,
                    label="Thermal Diffusion",
                )
                fuel_density_slider = gr.Slider(
                    cfg.fuel_density_min,
                    cfg.fuel_density_max,
                    value=cfg.default_fuel_density,
                    step=0.1,
                    label="Fuel Density",
                )
                total_time = gr.Slider(
                    cfg.total_time_min,
                    cfg.total_time_max,
                    value=cfg.default_total_time,
                    step=0.5,
                    label="Total Simulation Time",
                )
                solve_btn = gr.Button("Simulate Fire Spread", variant="primary")

            with gr.Column(scale=2):
                solution_img = gr.Image(label="Wildfire Simulation Results")
                metrics_box = gr.Textbox(label="Simulation Metrics", lines=5, interactive=False)

        gr.Markdown("---")
        gr.Markdown(
            "### Resolution Comparison\n"
            "Simulates at "
            + ", ".join(f"{s}x{s}" for s in cfg.comparison_sizes)
            + " with identical physics to demonstrate resolution independence."
        )
        compare_btn = gr.Button("Compare " + " / ".join(f"{s}x{s}" for s in cfg.comparison_sizes))
        compare_img = gr.Image(label="Resolution Comparison")
        compare_status = gr.Textbox(label="", lines=1, interactive=False)

        solve_btn.click(
            solve_and_visualize_wildfire,
            inputs=[
                grid_size,
                wind_speed,
                wind_direction,
                diffusion_slider,
                fuel_density_slider,
                ignition_pattern,
                total_time,
            ],
            outputs=[solution_img, metrics_box],
        )
        compare_btn.click(
            compare_resolutions_wildfire,
            inputs=[
                wind_speed,
                wind_direction,
                diffusion_slider,
                fuel_density_slider,
                ignition_pattern,
            ],
            outputs=[compare_img, compare_status],
        )


__all__ = [
    "compare_resolutions_wildfire",
    "create_wildfire_tab",
    "solve_and_visualize_wildfire",
]
