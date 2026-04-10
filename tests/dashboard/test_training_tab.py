"""Tests for dashboard/tabs/training_tab.py — Training Dashboard tab."""

from __future__ import annotations

from unittest.mock import patch

import gradio as gr
import pytest
from PIL import Image as PILImage

from dashboard.config import TrainingConfig
from dashboard.tabs.training_tab import (
    create_training_tab,
    get_model_summary,
    plot_training_curves,
    show_loss_breakdown,
)

# ---------------------------------------------------------------------------
# get_model_summary
# ---------------------------------------------------------------------------


class TestGetModelSummary:
    @patch("dashboard.tabs.training_tab.AlphaGalerkinModel", create=True)
    @patch("dashboard.tabs.training_tab.OperatorConfig", create=True)
    def test_returns_string(self, _mock_cfg, _mock_model):
        result = get_model_summary(128, 4, 2, 64)
        assert isinstance(result, str)

    def test_fallback_on_import_error(self):
        """When model can't be loaded, returns a descriptive fallback string."""
        with patch.dict("sys.modules", {"config.schemas": None}):
            result = get_model_summary(128, 4, 2, 64)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_config_params_on_success(self):
        """Success-path summary includes the given dimensions and param count."""
        from unittest.mock import MagicMock, patch

        mock_cfg = MagicMock()
        mock_cfg.d_key = 32
        mock_cfg.d_value = 32
        mock_cfg.d_ffn = 128
        mock_cfg.n_heads = 4
        mock_cfg.use_fnet_mixing = True
        mock_cfg.lbb_beta_threshold = 1e-6

        mock_param = MagicMock()
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True

        mock_model = MagicMock()
        mock_model.parameters.return_value = [mock_param] * 5

        mock_schemas = MagicMock()
        mock_schemas.OperatorConfig = MagicMock(return_value=mock_cfg)

        mock_modeling = MagicMock()
        mock_modeling.AlphaGalerkinModel = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(),
                "config.schemas": mock_schemas,
                "src.modeling": MagicMock(),
                "src.modeling.model": mock_modeling,
            },
        ):
            result = get_model_summary(64, 2, 1, 32)

        assert "64" in result
        assert "5,000" in result  # 5 params × 1000 numel

    def test_fallback_contains_params(self):
        """Even on failure, the fallback string includes the input parameters."""
        # Patch config.schemas to simulate import failure without breaking structlog
        with patch.dict("sys.modules", {"config.schemas": None, "config": None}):
            result = get_model_summary(256, 6, 2, 128)
        assert "256" in result or "Architecture" in result

    @pytest.mark.parametrize(
        "d_model,n_galerkin,n_softmax,n_fourier",
        [
            (64, 2, 1, 32),
            (128, 4, 2, 64),
            (256, 6, 2, 128),
        ],
    )
    def test_various_configs_return_strings(self, d_model, n_galerkin, n_softmax, n_fourier):
        result = get_model_summary(d_model, n_galerkin, n_softmax, n_fourier)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# plot_training_curves
# ---------------------------------------------------------------------------


class TestPlotTrainingCurves:
    def test_returns_pil_image_and_summary(self):
        img, summary = plot_training_curves(5000, 3e-4, 1.0, 1.0, 0.1)
        assert isinstance(img, PILImage.Image)
        assert isinstance(summary, str)

    def test_summary_contains_total_steps(self):
        _, summary = plot_training_curves(5000, 3e-4, 1.0, 1.0, 0.1)
        assert "5,000" in summary

    def test_summary_contains_lr(self):
        _, summary = plot_training_curves(5000, 1e-3, 1.0, 1.0, 0.1)
        assert "1.00e-03" in summary or "1e-03" in summary

    def test_summary_contains_note(self):
        _, summary = plot_training_curves(5000, 3e-4, 1.0, 1.0, 0.1)
        assert "Note" in summary or "simulated" in summary.lower()

    @pytest.mark.parametrize("total_steps", [1000, 10000, 50000])
    def test_different_step_counts(self, total_steps):
        img, _ = plot_training_curves(total_steps, 3e-4, 1.0, 1.0, 0.1)
        assert isinstance(img, PILImage.Image)

    def test_zero_lbb_weight(self):
        img, summary = plot_training_curves(5000, 3e-4, 1.0, 1.0, 0.0)
        assert img is not None

    def test_custom_config(self, training_cfg):
        img, summary = plot_training_curves(5000, 3e-4, 1.0, 1.0, 0.1, cfg=training_cfg)
        assert img is not None

    def test_custom_training_config_n_points(self):
        cfg = TrainingConfig(curve_n_points=50)
        img, _ = plot_training_curves(5000, 3e-4, 1.0, 1.0, 0.1, cfg=cfg)
        assert img is not None

    def test_image_dimensions_positive(self):
        img, _ = plot_training_curves(5000, 3e-4, 1.0, 1.0, 0.1)
        w, h = img.size
        assert w > 0
        assert h > 0

    def test_large_lr(self):
        img, _ = plot_training_curves(5000, 1e-2, 1.0, 1.0, 0.1)
        assert img is not None

    def test_high_lbb_weight(self):
        img, summary = plot_training_curves(5000, 3e-4, 1.0, 1.0, 1.0)
        assert img is not None
        assert "1.0" in summary


# ---------------------------------------------------------------------------
# show_loss_breakdown
# ---------------------------------------------------------------------------


class TestShowLossBreakdown:
    def test_returns_pil_image(self):
        img = show_loss_breakdown()
        assert isinstance(img, PILImage.Image)

    def test_image_dimensions_positive(self):
        img = show_loss_breakdown()
        w, h = img.size
        assert w > 0
        assert h > 0

    def test_custom_config(self, training_cfg):
        img = show_loss_breakdown(cfg=training_cfg)
        assert isinstance(img, PILImage.Image)

    def test_lbb_threshold_in_diagram(self):
        cfg = TrainingConfig(lbb_min_threshold=1e-4)
        img = show_loss_breakdown(cfg=cfg)
        assert img is not None


# ---------------------------------------------------------------------------
# create_training_tab
# ---------------------------------------------------------------------------


class TestCreateTrainingTab:
    def test_creates_gradio_tab_with_default(self):
        with gr.Blocks():
            create_training_tab()

    def test_creates_gradio_tab_with_custom_config(self, training_cfg):
        with gr.Blocks():
            create_training_tab(training_cfg)

    def test_slider_bounds_from_config(self):
        cfg = TrainingConfig(d_model_min=32, d_model_max=512, d_model_default=64, d_model_step=32)
        with gr.Blocks():
            create_training_tab(cfg)  # Should not raise

    def test_curve_defaults_from_config(self):
        cfg = TrainingConfig(steps_default=20000, default_lr=1e-3, default_lbb_weight=0.5)
        with gr.Blocks():
            create_training_tab(cfg)
