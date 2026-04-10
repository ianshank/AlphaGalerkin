"""Shared utilities for the AlphaGalerkin dashboard.

Provides helpers that are reused across multiple tab modules, eliminating
code duplication and enforcing consistent behaviour for plot rendering,
device detection, and error formatting.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import structlog

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Matplotlib → PIL conversion
# ---------------------------------------------------------------------------


def fig_to_pil(fig: plt.Figure, *, dpi: int = 110) -> PILImage.Image:
    """Convert a matplotlib Figure to a PIL Image and close the figure.

    The figure is always closed after conversion to prevent memory leaks,
    even if the conversion raises an exception.

    Args:
        fig: Matplotlib figure to convert.
        dpi: Dots-per-inch for the rendered PNG.

    Returns:
        A PIL Image in RGB mode.

    """
    from PIL import Image as PILImage  # local import keeps PIL optional at module level

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    try:
        buf.seek(0)
        with PILImage.open(buf) as img:
            # Convert to RGB to match the documented return mode, then copy
            # so the returned image is detached from the temporary buffer.
            return img.convert("RGB").copy()
    finally:
        buf.close()


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def device_str() -> str:
    """Return the best available PyTorch compute device as a string.

    Returns ``"cuda"`` when CUDA is available, ``"cpu"`` otherwise.
    Handles the case where PyTorch is not installed by returning ``"cpu"``.

    Returns:
        Device string suitable for ``torch.device()``.

    """
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug("device_detected", device=dev)
        return dev
    except ImportError:
        logger.debug("torch_not_available_falling_back_to_cpu")
        return "cpu"


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


def format_exc(exc: Exception, *, prefix: str = "Error") -> str:
    """Return a human-readable one-line error string for display in the UI.

    Args:
        exc: The caught exception.
        prefix: Label prepended to the message (e.g. ``"Import error"``).

    Returns:
        Formatted string ``"<prefix>: <ExcType>: <message>"``.

    """
    return f"{prefix}: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Logging configuration helper
# ---------------------------------------------------------------------------


def configure_structlog(level: int = logging.INFO) -> None:
    """Configure structlog for dashboard use.

    Idempotent — safe to call multiple times.  Uses a minimal processor
    chain suitable for interactive use (no colours, ISO timestamps).

    Args:
        level: Python logging level (e.g. ``logging.DEBUG``).

    """
    import structlog as sl

    logging.basicConfig(level=level, format="%(message)s")
    sl.configure(
        processors=[
            sl.stdlib.add_log_level,
            sl.processors.TimeStamper(fmt="%H:%M:%S"),
            sl.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=sl.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=sl.stdlib.LoggerFactory(),
    )


__all__ = [
    "configure_structlog",
    "device_str",
    "fig_to_pil",
    "format_exc",
]
