"""Tests for dashboard/tabs/poc_tab.py — PoC scenario runner tab."""

from __future__ import annotations

from unittest.mock import patch

import gradio as gr
import pytest
from PIL import Image as PILImage

from dashboard.config import ComplexityRunConfig, PoCConfig, StabilityRunConfig
from dashboard.tabs.poc_tab import (
    _parse_int_list,
    create_poc_tab,
    run_complexity,
    run_stability,
    show_transfer_milestone,
)

# ---------------------------------------------------------------------------
# _parse_int_list
# ---------------------------------------------------------------------------


class TestParseIntList:
    def test_valid_input(self):
        result = _parse_int_list("9,13,19,25", fallback=[1, 2, 3], min_count=3)
        assert result == [9, 13, 19, 25]

    def test_deduplication_and_sorting(self):
        result = _parse_int_list("19,9,9,13", fallback=[1, 2, 3], min_count=2)
        assert result == [9, 13, 19]

    def test_too_few_values_uses_fallback(self):
        fallback = [5, 9, 13]
        result = _parse_int_list("9", fallback=fallback, min_count=3)
        assert result == fallback

    def test_invalid_string_uses_fallback(self):
        fallback = [5, 9, 13]
        result = _parse_int_list("a,b,c", fallback=fallback, min_count=2)
        assert result == fallback

    def test_empty_string_uses_fallback(self):
        fallback = [5, 9]
        result = _parse_int_list("", fallback=fallback, min_count=2)
        assert result == fallback

    def test_whitespace_stripped(self):
        result = _parse_int_list(" 9 , 13 , 19 ", fallback=[1, 2, 3], min_count=2)
        assert result == [9, 13, 19]

    def test_exactly_min_count(self):
        result = _parse_int_list("9,13", fallback=[1, 2, 3], min_count=2)
        assert result == [9, 13]


# ---------------------------------------------------------------------------
# run_complexity
# ---------------------------------------------------------------------------


class TestRunComplexity:
    @patch("dashboard.tabs.poc_tab.ComplexityScenario")
    @patch("dashboard.tabs.poc_tab.ComplexityScenarioConfig")
    def test_returns_image_and_summary(
        self, _mock_cfg_cls, mock_scenario_cls, mock_complexity_result
    ):
        mock_scenario_cls.return_value.run.return_value = mock_complexity_result
        img, summary = run_complexity("9,13,19", 64, 10)
        assert isinstance(img, PILImage.Image)
        assert "Status" in summary

    @patch("dashboard.tabs.poc_tab.ComplexityScenario")
    @patch("dashboard.tabs.poc_tab.ComplexityScenarioConfig")
    def test_status_reflected_in_summary(
        self, _mock_cfg_cls, mock_scenario_cls, mock_complexity_result
    ):
        mock_complexity_result.status.value = "passed"
        mock_scenario_cls.return_value.run.return_value = mock_complexity_result
        _, summary = run_complexity("9,13,19", 64, 10)
        assert "PASSED" in summary

    @patch("dashboard.tabs.poc_tab.ComplexityScenario")
    @patch("dashboard.tabs.poc_tab.ComplexityScenarioConfig")
    def test_custom_config_respected(
        self, _mock_cfg_cls, mock_scenario_cls, mock_complexity_result
    ):
        mock_scenario_cls.return_value.run.return_value = mock_complexity_result
        cfg = ComplexityRunConfig(n_warmup=1)
        _, summary = run_complexity("9,13,19", 64, 10, cfg=cfg)
        assert summary  # Any non-empty string

    @patch("dashboard.tabs.poc_tab.ComplexityScenario", side_effect=RuntimeError("crash"))
    @patch("dashboard.tabs.poc_tab.ComplexityScenarioConfig")
    def test_scenario_exception_returns_error(self, _mock_cfg_cls, _mock_scenario_cls):
        img, summary = run_complexity("9,13,19", 64, 10)
        assert img is None
        assert "error" in summary.lower()

    def test_import_error_returns_error_string(self):
        with (
            patch("dashboard.tabs.poc_tab.ComplexityScenario", None),
            patch("dashboard.tabs.poc_tab.ComplexityScenarioConfig", None),
        ):
            img, summary = run_complexity("9,13,19", 64, 10)
        assert img is None
        assert "error" in summary.lower()

    @patch("dashboard.tabs.poc_tab.ComplexityScenario")
    @patch("dashboard.tabs.poc_tab.ComplexityScenarioConfig")
    def test_fallback_grid_sizes_when_too_few(
        self, _mock_cfg_cls, mock_scenario_cls, mock_complexity_result
    ):
        mock_scenario_cls.return_value.run.return_value = mock_complexity_result
        # Only one grid size provided → should fall back
        img, summary = run_complexity("9", 64, 10)
        assert img is not None

    @patch("dashboard.tabs.poc_tab.ComplexityScenario")
    @patch("dashboard.tabs.poc_tab.ComplexityScenarioConfig")
    def test_speedup_shown_in_summary(
        self, _mock_cfg_cls, mock_scenario_cls, mock_complexity_result
    ):
        mock_scenario_cls.return_value.run.return_value = mock_complexity_result
        _, summary = run_complexity("9,13,19", 64, 10)
        assert "speedup" in summary.lower() or "×" in summary


# ---------------------------------------------------------------------------
# run_stability
# ---------------------------------------------------------------------------


class TestRunStability:
    @patch("dashboard.tabs.poc_tab.StabilityScenario")
    @patch("dashboard.tabs.poc_tab.StabilityScenarioConfig")
    def test_returns_image_and_summary(
        self, _mock_cfg_cls, mock_scenario_cls, mock_stability_result
    ):
        mock_scenario_cls.return_value.run.return_value = mock_stability_result
        img, summary = run_stability("5,9,13", 64, 100)
        assert isinstance(img, PILImage.Image)
        assert "Status" in summary

    @patch("dashboard.tabs.poc_tab.StabilityScenario")
    @patch("dashboard.tabs.poc_tab.StabilityScenarioConfig")
    def test_violations_reflected_in_summary(
        self, _mock_cfg_cls, mock_scenario_cls, mock_stability_result
    ):
        mock_stability_result.metrics["lbb_violations"] = 3
        mock_scenario_cls.return_value.run.return_value = mock_stability_result
        _, summary = run_stability("5,9,13", 64, 100)
        assert "3" in summary

    @patch("dashboard.tabs.poc_tab.StabilityScenario", side_effect=ValueError("bad config"))
    @patch("dashboard.tabs.poc_tab.StabilityScenarioConfig")
    def test_exception_returns_error(self, _mock_cfg_cls, _mock_scenario_cls):
        img, summary = run_stability("5,9,13", 64, 100)
        assert img is None
        assert "error" in summary.lower()

    @patch("dashboard.tabs.poc_tab.StabilityScenario")
    @patch("dashboard.tabs.poc_tab.StabilityScenarioConfig")
    def test_custom_config(self, _mock_cfg_cls, mock_scenario_cls, mock_stability_result):
        mock_scenario_cls.return_value.run.return_value = mock_stability_result
        cfg = StabilityRunConfig(n_forward_passes=5)
        img, _ = run_stability("5,9,13", 64, 100, cfg=cfg)
        assert img is not None

    @patch("dashboard.tabs.poc_tab.StabilityScenario")
    @patch("dashboard.tabs.poc_tab.StabilityScenarioConfig")
    def test_fallback_resolutions_when_too_few(
        self, _mock_cfg_cls, mock_scenario_cls, mock_stability_result
    ):
        mock_scenario_cls.return_value.run.return_value = mock_stability_result
        img, _ = run_stability("9", 64, 100)
        assert img is not None


# ---------------------------------------------------------------------------
# show_transfer_milestone
# ---------------------------------------------------------------------------


class TestShowTransferMilestone:
    def test_returns_image_and_summary(self):
        img, summary = show_transfer_milestone()
        assert isinstance(img, PILImage.Image)
        assert isinstance(summary, str)

    def test_summary_contains_milestone_date(self):
        _, summary = show_transfer_milestone()
        assert "2026-01-26" in summary

    def test_summary_contains_mse_values(self):
        _, summary = show_transfer_milestone()
        # Should contain at least one MSE value
        assert "MSE" in summary

    def test_summary_contains_all_resolutions(self):
        from dashboard.config import DEFAULT_CONFIG

        milestone = DEFAULT_CONFIG.poc.transfer
        _, summary = show_transfer_milestone()
        for res in milestone.achieved_mse:
            assert str(res) in summary

    def test_custom_config(self, poc_cfg):
        img, summary = show_transfer_milestone(cfg=poc_cfg)
        assert img is not None
        assert summary

    def test_improvement_ratio_shown(self):
        _, summary = show_transfer_milestone()
        # Should show "X× better" for each resolution
        assert "better" in summary

    def test_image_dimensions_reasonable(self):
        img, _ = show_transfer_milestone()
        w, h = img.size
        assert w > 200
        assert h > 100

    @pytest.mark.parametrize(
        "mse_dict",
        [
            {9: 0.001, 19: 0.002},
            {13: 0.00098, 19: 0.00209},
        ],
    )
    def test_custom_mse_values(self, mse_dict):
        from dashboard.config import PoCConfig, TransferMilestone

        milestone = TransferMilestone(achieved_mse=mse_dict)
        cfg = PoCConfig(transfer=milestone)
        img, summary = show_transfer_milestone(cfg=cfg)
        assert img is not None


# ---------------------------------------------------------------------------
# create_poc_tab
# ---------------------------------------------------------------------------


class TestCreatePocTab:
    def test_creates_gradio_tab(self, poc_cfg):
        with gr.Blocks():
            create_poc_tab(poc_cfg)  # Should not raise

    def test_creates_tab_with_default_config(self):
        with gr.Blocks():
            create_poc_tab()  # Should not raise

    def test_custom_complexity_defaults(self):
        cfg = PoCConfig(complexity=ComplexityRunConfig(default_d_model=128))
        with gr.Blocks():
            create_poc_tab(cfg)

    def test_custom_stability_defaults(self):
        cfg = PoCConfig(stability=StabilityRunConfig(default_training_steps=200))
        with gr.Blocks():
            create_poc_tab(cfg)
