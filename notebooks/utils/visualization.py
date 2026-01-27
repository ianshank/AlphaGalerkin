"""Visualization utilities for AlphaGalerkin demo notebooks.

Provides reusable plotting functions with configurable styling.
"""

from __future__ import annotations

import logging
from typing import Sequence, TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    # Fallback to standard logging if structlog not available
    logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

    from notebooks.utils.config import VisualizationConfig


def _get_plt():
    """Lazy import of matplotlib to avoid import errors in non-notebook contexts."""
    import matplotlib.pyplot as plt

    return plt


def plot_fourier_features(
    fourier_encoder,
    board_sizes: Sequence[int],
    feature_indices: tuple[int, int] = (0, 5),
    figsize: tuple[float, float] = (14, 6),
    cmaps: tuple[str, str] = ("RdBu", "viridis"),
    device: str = "cpu",
) -> "Figure":
    """Plot Fourier features at different resolutions.

    Args:
        fourier_encoder: FourierBasis encoder instance.
        board_sizes: Board sizes to visualize.
        feature_indices: Which features to plot (row 1, row 2).
        figsize: Figure size (width, height).
        cmaps: Colormaps for each row.
        device: Device for tensor operations.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If board_sizes is empty.

    """
    from src.math_kernel.basis import create_grid_coordinates

    # Validate inputs
    if not board_sizes:
        raise ValueError("board_sizes cannot be empty")

    plt = _get_plt()
    n_cols = len(board_sizes)
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)

    # Handle single column case
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for idx, size in enumerate(board_sizes):
        coords = create_grid_coordinates(size, batch_size=1, device=device)
        features = fourier_encoder(coords)

        # Reshape and plot first feature
        feature_grid1 = features[0, :, feature_indices[0]].reshape(size, size)
        if isinstance(feature_grid1, torch.Tensor):
            feature_grid1 = feature_grid1.cpu().numpy()

        ax1 = axes[0, idx]
        ax1.imshow(feature_grid1, cmap=cmaps[0], aspect="equal")
        ax1.set_title(f"{size}×{size} Board", fontsize=11)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot second feature
        feature_grid2 = features[0, :, feature_indices[1]].reshape(size, size)
        if isinstance(feature_grid2, torch.Tensor):
            feature_grid2 = feature_grid2.cpu().numpy()

        ax2 = axes[1, idx]
        ax2.imshow(feature_grid2, cmap=cmaps[1], aspect="equal")
        ax2.set_xticks([])
        ax2.set_yticks([])

    axes[0, 0].set_ylabel(f"Feature {feature_indices[0] + 1}\n(cos wave)", fontsize=10)
    axes[1, 0].set_ylabel(f"Feature {feature_indices[1] + 1}\n(sin wave)", fontsize=10)

    fig.suptitle(
        "Fourier Features at Different Resolutions\n(Same pattern, different samplings)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    logger.debug("plot_fourier_features", n_sizes=len(board_sizes))
    return fig


def plot_attention_comparison(
    galerkin_times: Sequence[float],
    softmax_times: Sequence[float],
    board_labels: Sequence[str],
    figsize: tuple[float, float] = (10, 5),
    colors: tuple[str, str] = ("#2ecc71", "#e74c3c"),
) -> "Figure":
    """Plot attention speed comparison bar chart.

    Args:
        galerkin_times: Galerkin attention times (ms).
        softmax_times: Softmax attention times (ms).
        board_labels: Labels for each bar group.
        figsize: Figure size.
        colors: Colors for (Galerkin, Softmax) bars.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If input sequences are empty or have mismatched lengths.

    """
    # Validate inputs
    if not galerkin_times or not softmax_times:
        raise ValueError("Time sequences cannot be empty")
    if len(galerkin_times) != len(softmax_times):
        raise ValueError(
            f"Sequence length mismatch: galerkin={len(galerkin_times)}, "
            f"softmax={len(softmax_times)}"
        )
    if len(board_labels) != len(galerkin_times):
        raise ValueError(
            f"Label count mismatch: labels={len(board_labels)}, "
            f"data={len(galerkin_times)}"
        )

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(galerkin_times))
    width = 0.35

    ax.bar(
        x_pos - width / 2,
        galerkin_times,
        width,
        label="Galerkin O(N)",
        color=colors[0],
        alpha=0.8,
    )
    ax.bar(
        x_pos + width / 2,
        softmax_times,
        width,
        label="Softmax O(N²)",
        color=colors[1],
        alpha=0.8,
    )

    ax.set_xlabel("Board Size", fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title("Attention Speed: Galerkin vs Softmax\n(Lower is better)", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(board_labels)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Add speedup annotations
    for i, (g, s) in enumerate(zip(galerkin_times, softmax_times, strict=True)):
        speedup = s / g if g > 0 else 0
        ax.annotate(
            f"{speedup:.1f}x faster",
            xy=(i, max(g, s) + 0.5),
            ha="center",
            fontsize=9,
            color="#27ae60",
        )

    plt.tight_layout()
    logger.debug("plot_attention_comparison", n_groups=len(galerkin_times))
    return fig


def plot_poisson_samples(
    samples: Sequence,
    figsize: tuple[float, float] = (14, 6),
    charge_cmap: str = "RdBu",
    potential_cmap: str = "plasma",
) -> "Figure":
    """Plot Poisson equation samples (charges and potentials).

    Args:
        samples: Sequence of PoissonSample objects with grid_size, charges, potential.
        figsize: Figure size.
        charge_cmap: Colormap for charge distribution.
        potential_cmap: Colormap for potential field.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If samples is empty.
        AttributeError: If samples lack required attributes.

    """
    # Validate inputs
    if not samples:
        raise ValueError("samples cannot be empty")

    # Validate sample structure
    required_attrs = ("grid_size", "charges", "potential")
    for attr in required_attrs:
        if not hasattr(samples[0], attr):
            raise AttributeError(f"Sample missing required attribute: {attr}")

    plt = _get_plt()
    n_samples = len(samples)
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)

    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(2, 1)

    for idx, sample in enumerate(samples):
        size = sample.grid_size

        # Plot charges
        charges = sample.charges.reshape(size, size)
        ax1 = axes[0, idx]
        ax1.imshow(charges, cmap=charge_cmap, aspect="equal")
        ax1.set_title(f"{size}×{size} Charges", fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot potential
        potential = sample.potential.reshape(size, size)
        ax2 = axes[1, idx]
        ax2.imshow(potential, cmap=potential_cmap, aspect="equal")
        ax2.set_title(f"{size}×{size} Potential", fontsize=10)
        ax2.set_xticks([])
        ax2.set_yticks([])

    axes[0, 0].set_ylabel("Input\n(Charge ρ)", fontsize=10)
    axes[1, 0].set_ylabel("Output\n(Potential φ)", fontsize=10)

    fig.suptitle(
        "Poisson Equation: Same Physics, Different Resolutions\n"
        "(Train on 9×9 → Evaluate on 19×19)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    logger.debug("plot_poisson_samples", n_samples=n_samples)
    return fig


def plot_go_board(
    board: torch.Tensor,
    ax: "Axes",
    stone_radius: float = 0.4,
    board_color: float = 0.82,
    grid_alpha: float = 0.5,
    grid_linewidth: float = 0.5,
) -> None:
    """Draw a Go board with stones on the given axes.

    Args:
        board: Board tensor of shape (1, channels, height, width).
        ax: Matplotlib axes to draw on.
        stone_radius: Radius of stones.
        board_color: Board background color (0-1 for YlOrBr).
        grid_alpha: Grid line transparency.
        grid_linewidth: Grid line width.

    """
    plt = _get_plt()
    size = board.shape[-1]

    # Draw board background
    ax.imshow(
        np.ones((size, size)) * board_color,
        cmap="YlOrBr",
        vmin=0,
        vmax=1,
        aspect="equal",
    )

    # Draw grid
    for i in range(size):
        ax.axhline(i, color="black", linewidth=grid_linewidth, alpha=grid_alpha)
        ax.axvline(i, color="black", linewidth=grid_linewidth, alpha=grid_alpha)

    # Draw stones
    black_stones = board[0, 0]  # Channel 0: Black
    white_stones = board[0, 1]  # Channel 1: White

    if isinstance(black_stones, torch.Tensor):
        black_stones = black_stones.cpu().numpy()
        white_stones = white_stones.cpu().numpy()

    for r in range(size):
        for c in range(size):
            if black_stones[r, c] > 0:
                circle = plt.Circle((c, r), stone_radius, color="black")
                ax.add_patch(circle)
            elif white_stones[r, c] > 0:
                circle = plt.Circle(
                    (c, r), stone_radius, color="white", edgecolor="black"
                )
                ax.add_patch(circle)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_policy_heatmap(
    policy_logits: torch.Tensor,
    board_size: int,
    ax: "Axes",
    top_k: int = 3,
    cmap: str = "Reds",
    marker_color: str = "blue",
) -> "AxesImage":
    """Plot policy heatmap with top-k moves marked.

    Args:
        policy_logits: Policy logits tensor (batch, n_moves).
        board_size: Board size (height/width).
        ax: Matplotlib axes to draw on.
        top_k: Number of top moves to highlight.
        cmap: Colormap for heatmap.
        marker_color: Color for top-k markers.

    Returns:
        AxesImage from imshow for colorbar attachment.

    Note:
        If policy_logits shape doesn't match expected board_size, a warning
        is logged but plotting continues to allow flexible usage.

    """
    # Validate inputs (warn but don't error to allow flexible usage)
    expected_moves = board_size * board_size + 1  # board positions + pass
    if policy_logits.shape[-1] != expected_moves:
        logger.warning(
            "policy_shape_mismatch",
            expected=expected_moves,
            actual=policy_logits.shape[-1],
        )

    # Convert to probabilities
    policy = torch.softmax(policy_logits, dim=-1)[0]

    # Exclude pass move and reshape
    board_policy = policy[:-1].reshape(board_size, board_size)
    if isinstance(board_policy, torch.Tensor):
        board_policy = board_policy.cpu().numpy()

    # Plot heatmap
    im = ax.imshow(board_policy, cmap=cmap, aspect="equal")

    # Mark top-k moves (clamp to available positions)
    flat_policy = board_policy.flatten()
    actual_top_k = min(top_k, len(flat_policy))
    top_indices = np.argsort(flat_policy)[-actual_top_k:][::-1]

    for rank, idx_flat in enumerate(top_indices):
        r, c = idx_flat // board_size, idx_flat % board_size
        ax.plot(
            c,
            r,
            "o",
            markersize=15,
            markerfacecolor="none",
            markeredgecolor=marker_color,
            markeredgewidth=2,
        )
        ax.text(
            c, r, str(rank + 1), ha="center", va="center", fontsize=9, color=marker_color
        )

    ax.set_xticks([])
    ax.set_yticks([])

    return im


def plot_multi_board_visualization(
    boards: Sequence[torch.Tensor],
    board_sizes: Sequence[int],
    figsize: tuple[float, float] = (14, 4),
    title: str = "Board Visualization",
) -> "Figure":
    """Plot multiple Go boards side by side.

    Args:
        boards: Sequence of board tensors.
        board_sizes: Corresponding board sizes.
        figsize: Figure size.
        title: Figure title.

    Returns:
        Matplotlib Figure object.

    """
    plt = _get_plt()
    n_boards = len(boards)
    fig, axes = plt.subplots(1, n_boards, figsize=figsize)

    if n_boards == 1:
        axes = [axes]

    for idx, (board, size) in enumerate(zip(boards, board_sizes, strict=True)):
        ax = axes[idx]
        plot_go_board(board, ax)
        ax.set_title(f"{size}×{size} Board", fontsize=11)

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    logger.debug("plot_multi_board", n_boards=n_boards)
    return fig
