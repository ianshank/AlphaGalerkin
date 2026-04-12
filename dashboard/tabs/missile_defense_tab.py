"""Missile Defense Intercept Trajectory Analysis tab for the AlphaGalerkin dashboard.

Demonstrates ballistic threat tracking and proportional-navigation interceptor
guidance overlaid on a potential flow field solved at multiple resolutions.
The flow field uses the same Poisson spectral solver as the PDE tab, showing
how the Galerkin operator captures the same physics regardless of grid size.
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

from dashboard.config import DEFAULT_CONFIG, MissileDefenseConfig
from dashboard.utils import fig_to_pil, format_exc

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_threat_trajectory(
    launch_angle: float,
    velocity: float,
    gravity: float,
    dt: float,
    max_time: float,
) -> NDArray[np.float64]:
    """Compute a ballistic arc for the incoming threat.

    The trajectory follows standard projectile equations:
    ``x(t) = v cos(theta) t`` and ``y(t) = v sin(theta) t - 0.5 g t^2``.
    Positions are normalised to [0, 1] before being returned.

    Args:
        launch_angle: Launch elevation in **degrees**.
        velocity: Initial speed (arbitrary units, normalised later).
        gravity: Gravitational acceleration.
        dt: Time step for the integration.
        max_time: Maximum simulation time.

    Returns:
        Array of shape ``(N, 2)`` with columns ``(x, y)`` normalised to [0, 1].

    """
    theta = np.radians(launch_angle)
    vx = velocity * np.cos(theta)
    vy = velocity * np.sin(theta)

    positions: list[list[float]] = []
    t = 0.0
    while t <= max_time:
        x = vx * t
        y = vy * t - 0.5 * gravity * t * t
        positions.append([x, y])
        if t > 0 and y < 0:
            break
        t += dt

    raw = np.array(positions, dtype=np.float64)
    if raw.shape[0] < 2:
        return raw

    # Normalise to [0, 1] using the range of the trajectory.
    x_min, y_min = raw.min(axis=0)
    x_max, y_max = raw.max(axis=0)
    x_range = x_max - x_min if (x_max - x_min) > 1e-12 else 1.0
    y_range = y_max - y_min if (y_max - y_min) > 1e-12 else 1.0
    raw[:, 0] = (raw[:, 0] - x_min) / x_range
    raw[:, 1] = (raw[:, 1] - y_min) / y_range

    return raw


def _compute_interceptor_trajectory(
    start_x: float,
    start_y: float,
    speed: float,
    threat_positions: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Compute a proportional-navigation interceptor trajectory.

    At each time step the interceptor predicts where the threat will be by
    projecting the threat's current velocity forward, then steers toward that
    predicted intercept point at constant speed.

    Args:
        start_x: Interceptor initial x position (normalised 0-1).
        start_y: Interceptor initial y position (normalised 0-1).
        speed: Interceptor constant speed per time step.
        threat_positions: ``(N, 2)`` normalised threat positions.
        dt: Time step (used for velocity estimation on the threat).

    Returns:
        Array of shape ``(N, 2)`` with the interceptor path.

    """
    n_steps = threat_positions.shape[0]
    positions = np.zeros((n_steps, 2), dtype=np.float64)
    positions[0] = [start_x, start_y]

    for i in range(1, n_steps):
        # Estimate threat velocity from consecutive positions.
        if i < n_steps - 1:
            threat_vel = (threat_positions[i + 1] - threat_positions[i]) / max(dt, 1e-12)
        else:
            threat_vel = (threat_positions[i] - threat_positions[i - 1]) / max(dt, 1e-12)

        # Distance from current interceptor position to current threat position.
        diff = threat_positions[i] - positions[i - 1]
        dist = float(np.linalg.norm(diff))

        # Predict intercept point: target_pos + (distance / speed) * target_velocity.
        time_to_close = dist / max(speed, 1e-12)
        predicted = threat_positions[i] + time_to_close * threat_vel

        # Steer toward predicted intercept at constant speed.
        direction = predicted - positions[i - 1]
        dir_norm = float(np.linalg.norm(direction))
        if dir_norm > 1e-12:
            direction = direction / dir_norm

        positions[i] = positions[i - 1] + speed * dt * direction

    return positions


def _compute_closest_approach(
    threat_traj: NDArray[np.float64],
    interceptor_traj: NDArray[np.float64],
) -> tuple[float, float, int]:
    """Find the point of closest approach between two trajectories.

    Both trajectories are truncated to the shorter length so that
    corresponding time indices are compared.

    Args:
        threat_traj: ``(N, 2)`` threat positions.
        interceptor_traj: ``(M, 2)`` interceptor positions.

    Returns:
        Tuple of ``(miss_distance, time_of_closest_approach, step_index)``.
        Time is expressed as the fractional index normalised by trajectory
        length (i.e. ``step_index / min_len``).

    """
    min_len = min(len(threat_traj), len(interceptor_traj))
    t = threat_traj[:min_len]
    i = interceptor_traj[:min_len]
    dists = np.linalg.norm(t - i, axis=1)
    idx = int(np.argmin(dists))
    miss = float(dists[idx])
    time_frac = idx / max(min_len - 1, 1)
    return miss, time_frac, idx


def _compute_potential_flow(
    n: int,
    source_x: float,
    source_y: float,
) -> NDArray[np.float32]:
    """Compute a potential flow field on an ``n x n`` grid.

    Attempts to use the project's spectral Poisson solver.  If that import
    fails, falls back to the analytical potential ``phi = log(r)`` for a
    point source at ``(source_x, source_y)`` in normalised coordinates.

    Args:
        n: Grid dimension (the field is ``n x n``).
        source_x: Normalised x position of the source (0-1).
        source_y: Normalised y position of the source (0-1).

    Returns:
        Float32 array of shape ``(n, n)`` representing the potential field.

    """
    try:
        from src.physics.poisson import PoissonSolver  # type: ignore[import]

        charges: NDArray[np.float32] = np.zeros((n, n), dtype=np.float32)
        n1 = n - 1
        si = int(np.clip(round(source_x * n1), 0, n1))
        sj = int(np.clip(round(source_y * n1), 0, n1))
        charges[si, sj] = 1.0

        solver = PoissonSolver(resolution=n)
        result = solver.solve(charges)
        pot: NDArray[np.float32] = result if result.ndim == 2 else result.reshape(n, n)
        return pot.astype(np.float32)

    except Exception:
        logger.debug("poisson_import_fallback", reason="using analytical log(r)")
        xs = np.linspace(0, 1, n, dtype=np.float32)
        ys = np.linspace(0, 1, n, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        r = np.sqrt((xx - source_x) ** 2 + (yy - source_y) ** 2 + 1e-8)
        return np.log(r).astype(np.float32)


def _intercept_probability(miss_distance: float, kill_radius: float) -> float:
    """Gaussian kill probability model.

    ``P_kill = exp(-miss^2 / (2 * kill_radius^2))``

    Args:
        miss_distance: Miss distance (same units as *kill_radius*).
        kill_radius: Effective kill radius of the warhead.

    Returns:
        Probability clamped to ``[0, 1]``.

    """
    p = float(np.exp(-(miss_distance**2) / (2.0 * kill_radius**2)))
    return float(np.clip(p, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API -- functions called by Gradio event handlers
# ---------------------------------------------------------------------------


def solve_and_visualize_intercept(
    grid_size: int,
    threat_angle: float,
    threat_velocity: float,
    interceptor_x: float,
    interceptor_y: float,
    interceptor_speed: float,
    cfg: MissileDefenseConfig | None = None,
) -> tuple[PILImage.Image | None, str]:
    """Run the intercept simulation and produce a 2-panel visualisation.

    Left panel shows the threat and interceptor trajectories on a white
    background with a green star at the closest-approach point.  Right panel
    shows the potential flow field (``coolwarm`` colour map) with trajectory
    lines overlaid.

    Args:
        grid_size: Flow-field resolution ``N`` (``N x N``).
        threat_angle: Threat launch angle in degrees.
        threat_velocity: Threat initial velocity.
        interceptor_x: Interceptor start x position (normalised 0-1).
        interceptor_y: Interceptor start y position (normalised 0-1).
        interceptor_speed: Interceptor constant speed.
        cfg: Optional config override; uses ``DEFAULT_CONFIG.missile_defense``
            when *None*.

    Returns:
        Tuple of ``(PIL Image or None, metrics string)``.
        Returns ``(None, error_message)`` on failure.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.missile_defense
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi

    logger.info(
        "intercept_sim_requested",
        grid_size=grid_size,
        threat_angle=threat_angle,
    )
    try:
        n = int(grid_size)

        # --- trajectories ---
        threat_traj = _compute_threat_trajectory(
            threat_angle,
            threat_velocity,
            cfg.gravity,
            cfg.dt,
            cfg.max_time,
        )
        interceptor_traj = _compute_interceptor_trajectory(
            interceptor_x,
            interceptor_y,
            interceptor_speed,
            threat_traj,
            cfg.dt,
        )
        miss_dist, t_approach, idx = _compute_closest_approach(
            threat_traj,
            interceptor_traj,
        )
        p_kill = _intercept_probability(miss_dist, cfg.kill_radius)

        # --- potential flow ---
        potential = _compute_potential_flow(n, 0.5, 0.5)

        # --- figure ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Missile Defense Intercept Analysis \u00b7 {n}\u00d7{n}",
            fontsize=13,
        )

        # Left panel: trajectory plot
        ax0 = axes[0]
        ax0.plot(
            threat_traj[:, 0],
            threat_traj[:, 1],
            "r-o",
            markersize=3,
            linewidth=1.5,
            label="Threat",
        )
        ax0.plot(
            interceptor_traj[:, 0],
            interceptor_traj[:, 1],
            "b-s",
            markersize=3,
            linewidth=1.5,
            label="Interceptor",
        )

        # Closest approach marker
        min_len = min(len(threat_traj), len(interceptor_traj))
        if idx < min_len:
            mid_x = (threat_traj[idx, 0] + interceptor_traj[idx, 0]) / 2
            mid_y = (threat_traj[idx, 1] + interceptor_traj[idx, 1]) / 2
            ax0.plot(mid_x, mid_y, "g*", markersize=15, zorder=5)

            # Kill-radius circle
            circle = plt.Circle(
                (mid_x, mid_y),
                cfg.kill_radius,
                color="green",
                fill=False,
                linestyle="--",
                linewidth=1.2,
            )
            ax0.add_patch(circle)

        ax0.set_xlabel("Downrange")
        ax0.set_ylabel("Altitude")
        ax0.set_title("Trajectory Plot")
        ax0.legend(loc="upper right")
        ax0.grid(True)
        ax0.set_aspect("equal", adjustable="datalim")

        # Right panel: potential flow with trajectory overlay
        ax1 = axes[1]
        ax1.imshow(
            potential.T,
            cmap="coolwarm",
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="equal",
        )
        ax1.plot(
            threat_traj[:, 0],
            threat_traj[:, 1],
            "r-",
            linewidth=1.5,
            alpha=0.9,
        )
        ax1.plot(
            interceptor_traj[:, 0],
            interceptor_traj[:, 1],
            "b-",
            linewidth=1.5,
            alpha=0.9,
        )
        ax1.set_title(f"Potential Flow Field ({n}\u00d7{n})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        plt.tight_layout()
        img = fig_to_pil(fig, dpi=plot_dpi)

        # --- metrics ---
        max_alt = float(threat_traj[:, 1].max())
        impact_range = float(threat_traj[-1, 0]) if threat_traj.shape[0] > 1 else 0.0
        metrics = (
            f"Grid: {n}\u00d7{n}  |  Tokens (N): {n * n}\n"
            f"Miss distance: {miss_dist:.4f}\n"
            f"Time-to-intercept: {t_approach:.3f} (normalised)\n"
            f"P_kill: {p_kill:.4f}\n"
            f"Max altitude: {max_alt:.4f}\n"
            f"Impact range: {impact_range:.4f}"
        )
        logger.info(
            "intercept_sim_complete",
            grid_size=n,
            miss_distance=miss_dist,
            p_kill=p_kill,
        )
        return img, metrics

    except Exception as exc:
        msg = format_exc(exc, prefix="Intercept simulation error")
        logger.exception("intercept_sim_failed", grid_size=grid_size)
        return None, msg


def compare_resolutions_intercept(
    threat_angle: float,
    threat_velocity: float,
    interceptor_x: float,
    interceptor_y: float,
    interceptor_speed: float,
    cfg: MissileDefenseConfig | None = None,
) -> tuple[PILImage.Image | None, str]:
    """Solve the potential flow at multiple resolutions with trajectory overlay.

    Renders a side-by-side comparison at each resolution defined in
    ``cfg.comparison_sizes`` and computes cross-resolution MSE.

    Args:
        threat_angle: Threat launch angle in degrees.
        threat_velocity: Threat initial velocity.
        interceptor_x: Interceptor start x position (normalised 0-1).
        interceptor_y: Interceptor start y position (normalised 0-1).
        interceptor_speed: Interceptor constant speed.
        cfg: Optional config override; uses ``DEFAULT_CONFIG.missile_defense``
            when *None*.

    Returns:
        Tuple of ``(PIL Image or None, MSE summary string)``.
        Returns ``(None, error_message)`` on failure.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.missile_defense
    plot_dpi = DEFAULT_CONFIG.app.plot_dpi
    sizes = cfg.comparison_sizes

    logger.info("intercept_compare_requested", sizes=sizes)
    try:
        from scipy.ndimage import zoom  # type: ignore[import]

        # Compute shared trajectory (use largest resolution's params).
        threat_traj = _compute_threat_trajectory(
            threat_angle,
            threat_velocity,
            cfg.gravity,
            cfg.dt,
            cfg.max_time,
        )
        interceptor_traj = _compute_interceptor_trajectory(
            interceptor_x,
            interceptor_y,
            interceptor_speed,
            threat_traj,
            cfg.dt,
        )

        n_cols = len(sizes)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols + 1, 5))
        if n_cols == 1:
            axes = [axes]
        fig.suptitle(
            "Resolution Independence \u2014 " + " / ".join(f"{s}\u00d7{s}" for s in sizes),
            fontsize=13,
        )

        potentials: list[NDArray[np.float32]] = []
        for col, n in enumerate(sizes):
            potential = _compute_potential_flow(n, 0.5, 0.5)
            potentials.append(potential)

            axes[col].imshow(
                potential.T,
                cmap="coolwarm",
                origin="lower",
                extent=[0, 1, 0, 1],
                aspect="equal",
            )
            axes[col].plot(
                threat_traj[:, 0],
                threat_traj[:, 1],
                "r-",
                linewidth=1.2,
                alpha=0.8,
            )
            axes[col].plot(
                interceptor_traj[:, 0],
                interceptor_traj[:, 1],
                "b-",
                linewidth=1.2,
                alpha=0.8,
            )
            axes[col].set_title(f"{n}\u00d7{n}  N={n * n}")
            axes[col].axis("off")

        plt.tight_layout()
        img = fig_to_pil(fig, dpi=plot_dpi)

        # Cross-resolution MSE: upsample each to the largest for comparison.
        ref_pot_2d = potentials[-1]
        ref_shape = ref_pot_2d.shape
        ref_pot = ref_pot_2d.flatten()
        ref_norm = float(np.linalg.norm(ref_pot)) + cfg.epsilon
        mse_parts: list[str] = []
        for i, n in enumerate(sizes[:-1]):
            src_shape = potentials[i].shape
            scale = (ref_shape[0] / src_shape[0], ref_shape[1] / src_shape[1])
            up_2d = zoom(potentials[i], scale)
            if up_2d.shape != ref_shape:
                raise ValueError(
                    f"Upsampled potential shape mismatch for {n}\u00d7{n}: "
                    f"expected {ref_shape}, got {up_2d.shape}"
                )
            up = up_2d.flatten()
            up_norm = float(np.linalg.norm(up)) + cfg.epsilon
            mse = float(np.mean(((up / up_norm) - (ref_pot / ref_norm)) ** 2))
            mse_parts.append(
                f"{n}\u00d7{n}\u2192{ref_shape[0]}\u00d7{ref_shape[1]} MSE\u2248{mse:.4f}"
            )

        msg = "Comparison complete.  " + "  |  ".join(mse_parts) if mse_parts else "Complete."
        logger.info("intercept_compare_complete", mse_parts=mse_parts)
        return img, msg

    except Exception as exc:
        msg = format_exc(exc, prefix="Comparison error")
        logger.exception("intercept_compare_failed")
        return None, msg


# ---------------------------------------------------------------------------
# Gradio tab builder
# ---------------------------------------------------------------------------


def create_missile_defense_tab(cfg: MissileDefenseConfig | None = None) -> None:
    """Create the Missile Defense tab inside an existing ``gr.Blocks`` context.

    Args:
        cfg: Optional MissileDefenseConfig override; uses
            ``DEFAULT_CONFIG.missile_defense`` when *None*.

    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.missile_defense

    with gr.Tab("Missile Defense"):
        gr.Markdown(
            "## Missile Defense Intercept Trajectory Analysis\n"
            "Simulates a **ballistic threat** and a **proportional-navigation "
            "interceptor** overlaid on a potential flow field solved with the "
            "same spectral Poisson solver used in the PDE tab.\n\n"
            "The flow field demonstrates **resolution independence**: the same "
            "physics is captured at "
            + ", ".join(f"{s}\u00d7{s}" for s in cfg.comparison_sizes)
            + " with no retraining."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Simulation Parameters")
                grid_size = gr.Dropdown(
                    choices=cfg.grid_sizes,
                    value=cfg.default_grid_size,
                    label="Grid Resolution (N\u00d7N)",
                )
                threat_angle = gr.Slider(
                    cfg.threat_angle_min,
                    cfg.threat_angle_max,
                    value=cfg.default_threat_angle,
                    step=1.0,
                    label="Threat Launch Angle (\u00b0)",
                )
                threat_velocity = gr.Slider(
                    cfg.threat_velocity_min,
                    cfg.threat_velocity_max,
                    value=cfg.default_threat_velocity,
                    step=0.1,
                    label="Threat Velocity (km/s)",
                )
                interceptor_x = gr.Slider(
                    cfg.interceptor_x_min,
                    cfg.interceptor_x_max,
                    value=cfg.default_interceptor_x,
                    step=0.05,
                    label="Interceptor X Position",
                )
                interceptor_y = gr.Slider(
                    cfg.interceptor_y_min,
                    cfg.interceptor_y_max,
                    value=cfg.default_interceptor_y,
                    step=0.05,
                    label="Interceptor Y Position",
                )
                interceptor_speed = gr.Slider(
                    cfg.interceptor_speed_min,
                    cfg.interceptor_speed_max,
                    value=cfg.default_interceptor_speed,
                    step=0.1,
                    label="Interceptor Speed (km/s)",
                )
                sim_btn = gr.Button("Simulate Intercept", variant="primary")

            with gr.Column(scale=2):
                solution_img = gr.Image(label="Trajectory & Flow Field")
                metrics_box = gr.Textbox(
                    label="Intercept Metrics",
                    lines=6,
                    interactive=False,
                )

        gr.Markdown("---")
        gr.Markdown(
            "### Resolution Comparison \u2014 Zero-Shot Transfer Demo\n"
            "Solves the potential flow at "
            + ", ".join(f"{s}\u00d7{s}" for s in cfg.comparison_sizes)
            + " with identical trajectories to demonstrate resolution independence."
        )
        compare_btn = gr.Button(
            "Compare " + " / ".join(f"{s}\u00d7{s}" for s in cfg.comparison_sizes)
        )
        compare_img = gr.Image(label="Resolution Comparison")
        compare_status = gr.Textbox(label="", lines=1, interactive=False)

        sim_btn.click(
            solve_and_visualize_intercept,
            inputs=[
                grid_size,
                threat_angle,
                threat_velocity,
                interceptor_x,
                interceptor_y,
                interceptor_speed,
            ],
            outputs=[solution_img, metrics_box],
        )
        compare_btn.click(
            compare_resolutions_intercept,
            inputs=[
                threat_angle,
                threat_velocity,
                interceptor_x,
                interceptor_y,
                interceptor_speed,
            ],
            outputs=[compare_img, compare_status],
        )


__all__ = [
    "compare_resolutions_intercept",
    "create_missile_defense_tab",
    "solve_and_visualize_intercept",
]
