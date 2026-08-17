"""Regression tests guarding dashboard claims, metrics formatting, and plot rendering."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from PIL import Image

from dashboard.config import DEFAULT_CONFIG
from dashboard.utils import fig_to_pil, format_exc


def test_dashboard_config_defaults_finite() -> None:
    """Regression test: All default dashboard configuration fields are finite numbers."""
    cfg = DEFAULT_CONFIG
    assert math.isfinite(cfg.training.lbb_min_threshold)
    assert math.isfinite(cfg.training.lbb_const_asymptote)
    assert math.isfinite(cfg.training.lbb_const_amplitude)
    assert cfg.training.lbb_const_amplitude >= 0
    assert cfg.training.lbb_min_threshold > 0


def test_fig_to_pil_converts_and_closes_figure() -> None:
    """Regression test: fig_to_pil converts a matplotlib figure to a PIL image and closes it."""
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])

    img = fig_to_pil(fig, dpi=72)
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0
    assert img.size[1] > 0
    # Verify figure was closed
    assert not plt.fignum_exists(fig.number)


def test_format_exc_safe_formatting() -> None:
    """Regression test: format_exc safely formats exceptions to one line without crashing."""
    err = ValueError("Invalid matrix dimension: shape (3, 4) != (4, 4)")
    formatted = format_exc(err, prefix="Solve Error")
    assert formatted == "Solve Error: ValueError: Invalid matrix dimension: shape (3, 4) != (4, 4)"
    assert "\n" not in formatted
