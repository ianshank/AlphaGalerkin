"""Tests for dashboard/utils.py — shared utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image as PILImage

from dashboard.utils import configure_structlog, device_str, fig_to_pil, format_exc

# ---------------------------------------------------------------------------
# fig_to_pil
# ---------------------------------------------------------------------------


class TestFigToPil:
    def test_returns_pil_image(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        img = fig_to_pil(fig)
        assert isinstance(img, PILImage.Image)

    def test_figure_is_closed(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2])
        _ = fig_to_pil(fig)
        # After conversion the figure should be closed (not in plt.get_fignums()).
        assert fig.number not in plt.get_fignums()

    def test_image_dimensions_positive(self):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1])
        img = fig_to_pil(fig)
        w, h = img.size
        assert w > 0
        assert h > 0

    def test_dpi_affects_size(self):
        """Higher DPI must produce a larger image for the same figsize."""
        fig_lo, ax_lo = plt.subplots(figsize=(4, 4))
        ax_lo.plot([0, 1])
        img_lo = fig_to_pil(fig_lo, dpi=72)

        fig_hi, ax_hi = plt.subplots(figsize=(4, 4))
        ax_hi.plot([0, 1])
        img_hi = fig_to_pil(fig_hi, dpi=200)

        # Higher DPI → more pixels on both axes
        assert img_hi.size[0] > img_lo.size[0]
        assert img_hi.size[1] > img_lo.size[1]

    def test_returns_rgb_mode(self):
        """fig_to_pil must always return an RGB image regardless of backend."""
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        img = fig_to_pil(fig)
        assert img.mode == "RGB"

    def test_figure_closed_on_exception(self):
        """fig_to_pil must close the figure even when savefig raises."""
        fig = plt.figure()
        fig_num = fig.number

        with patch.object(fig, "savefig", side_effect=RuntimeError("save failed")):
            with pytest.raises(RuntimeError, match="save failed"):
                fig_to_pil(fig)

        # The figure must still be closed
        assert fig_num not in plt.get_fignums()

    def test_result_is_detached_from_buffer(self):
        """The returned image must be fully loaded (not lazy-loading from a closed buffer)."""
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((5, 5)))
        img = fig_to_pil(fig)
        # Accessing pixel data would fail if the buffer was closed
        pixel = img.getpixel((0, 0))
        assert pixel is not None


# ---------------------------------------------------------------------------
# device_str
# ---------------------------------------------------------------------------


class TestDeviceStr:
    def test_returns_string(self):
        result = device_str()
        assert isinstance(result, str)

    def test_returns_cpu_or_cuda(self):
        result = device_str()
        assert result in ("cpu", "cuda")

    def test_cpu_when_torch_unavailable(self):
        with patch.dict("sys.modules", {"torch": None}):
            result = device_str()
        assert result == "cpu"

    def test_cuda_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = device_str()
        assert result == "cuda"

    def test_cpu_when_cuda_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = device_str()
        assert result == "cpu"

    def test_import_error_returns_cpu(self):
        # Setting sys.modules["torch"] = None makes `import torch` raise ImportError,
        # exercising the except ImportError branch in device_str().
        with patch.dict("sys.modules", {"torch": None}):
            result = device_str()
        assert result == "cpu"


# ---------------------------------------------------------------------------
# format_exc
# ---------------------------------------------------------------------------


class TestFormatExc:
    def test_basic_format(self):
        exc = ValueError("bad value")
        result = format_exc(exc)
        assert "Error" in result
        assert "ValueError" in result
        assert "bad value" in result

    def test_custom_prefix(self):
        exc = RuntimeError("something broke")
        result = format_exc(exc, prefix="Import error")
        assert result.startswith("Import error:")

    def test_includes_exception_type(self):
        exc = KeyError("missing_key")
        result = format_exc(exc, prefix="Test")
        assert "KeyError" in result

    def test_nested_exception(self):
        try:
            raise TypeError("inner")
        except TypeError as exc:
            result = format_exc(exc, prefix="Outer")
        assert "TypeError" in result
        assert "inner" in result

    @pytest.mark.parametrize(
        "exc_class,msg",
        [
            (ValueError, "test value error"),
            (RuntimeError, "runtime failure"),
            (ImportError, "missing module"),
            (FileNotFoundError, "no such file"),
        ],
    )
    def test_various_exception_types(self, exc_class, msg):
        exc = exc_class(msg)
        result = format_exc(exc, prefix="P")
        assert exc_class.__name__ in result
        assert msg in result


# ---------------------------------------------------------------------------
# configure_structlog
# ---------------------------------------------------------------------------


class TestConfigureStructlog:
    def test_runs_without_error(self):
        import logging

        configure_structlog(level=logging.WARNING)

    def test_idempotent(self):
        import logging

        configure_structlog(level=logging.INFO)
        configure_structlog(level=logging.DEBUG)  # Should not raise
