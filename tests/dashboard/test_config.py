"""Tests for dashboard/config.py — Pydantic configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dashboard.config import (
    DEFAULT_CONFIG,
    AppConfig,
    ComplexityRunConfig,
    DashboardConfig,
    GameConfig,
    PDEConfig,
    PoCConfig,
    StabilityRunConfig,
    TrainingConfig,
    TransferMilestone,
)

# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 7860
        assert not cfg.share
        assert not cfg.debug
        assert "px" in cfg.css_tab_font_size
        assert cfg.plot_dpi == 110

    def test_port_lower_bound(self):
        with pytest.raises(ValidationError):
            AppConfig(port=0)

    def test_port_upper_bound(self):
        with pytest.raises(ValidationError):
            AppConfig(port=99999)

    def test_custom_values(self):
        cfg = AppConfig(host="127.0.0.1", port=8080, share=True, debug=True)
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080
        assert cfg.share
        assert cfg.debug

    def test_dpi_range(self):
        cfg = AppConfig(plot_dpi=150)
        assert cfg.plot_dpi == 150
        with pytest.raises(ValidationError):
            AppConfig(plot_dpi=0)


# ---------------------------------------------------------------------------
# GameConfig
# ---------------------------------------------------------------------------


class TestGameConfig:
    def test_defaults(self):
        cfg = GameConfig()
        assert 9 in cfg.board_sizes
        assert cfg.default_board_size == 9
        assert 0.0 <= cfg.ai_temperature_vs_human < 1.0
        assert 0.0 <= cfg.ai_temperature_self_play < 1.0

    def test_custom_board_sizes(self):
        cfg = GameConfig(board_sizes=[5, 9], default_board_size=5)
        assert cfg.board_sizes == [5, 9]

    def test_temperature_non_negative(self):
        with pytest.raises(ValidationError):
            GameConfig(ai_temperature_vs_human=-0.1)

    def test_image_height_positive(self):
        with pytest.raises(ValidationError):
            GameConfig(board_image_height_px=50)


# ---------------------------------------------------------------------------
# PDEConfig
# ---------------------------------------------------------------------------


class TestPDEConfig:
    def test_defaults(self):
        cfg = PDEConfig()
        assert cfg.default_grid_size in cfg.grid_sizes
        assert len(cfg.charge_patterns) >= 4
        assert cfg.default_pattern in cfg.charge_patterns
        assert cfg.strength_min < cfg.default_strength <= cfg.strength_max
        assert 0 < cfg.position_min < cfg.position_max < 1
        assert cfg.epsilon > 0

    def test_comparison_sizes_non_empty(self):
        cfg = PDEConfig()
        assert len(cfg.comparison_sizes) >= 2

    def test_epsilon_must_be_positive(self):
        with pytest.raises(ValidationError):
            PDEConfig(epsilon=0)

    def test_ring_charges_minimum(self):
        with pytest.raises(ValidationError):
            PDEConfig(ring_num_charges=2)


# ---------------------------------------------------------------------------
# ComplexityRunConfig
# ---------------------------------------------------------------------------


class TestComplexityRunConfig:
    def test_defaults(self):
        cfg = ComplexityRunConfig()
        assert len(cfg.fallback_grid_sizes) >= 3
        assert cfg.n_warmup >= 1
        assert cfg.default_iterations >= 10
        assert cfg.min_grid_sizes >= 2

    def test_custom_fallback(self):
        cfg = ComplexityRunConfig(fallback_grid_sizes=[5, 9, 13, 19])
        assert cfg.fallback_grid_sizes == [5, 9, 13, 19]


# ---------------------------------------------------------------------------
# StabilityRunConfig
# ---------------------------------------------------------------------------


class TestStabilityRunConfig:
    def test_defaults(self):
        cfg = StabilityRunConfig()
        assert cfg.lbb_threshold > 0
        assert cfg.max_lbb_violations >= 0
        assert len(cfg.fallback_resolutions) >= 2

    def test_lbb_threshold_positive(self):
        with pytest.raises(ValidationError):
            StabilityRunConfig(lbb_threshold=0)


# ---------------------------------------------------------------------------
# TransferMilestone
# ---------------------------------------------------------------------------


class TestTransferMilestone:
    def test_defaults(self):
        m = TransferMilestone()
        assert m.train_resolution == 9
        assert m.mse_threshold == 0.05
        assert all(v < m.mse_threshold for v in m.achieved_mse.values())

    def test_mse_threshold_positive(self):
        with pytest.raises(ValidationError):
            TransferMilestone(mse_threshold=0)

    def test_achieved_mse_not_empty(self):
        m = TransferMilestone()
        assert len(m.achieved_mse) > 0

    @pytest.mark.parametrize("res", [9, 13, 19])
    def test_each_resolution_present(self, res):
        m = TransferMilestone()
        assert res in m.achieved_mse


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.d_model_min < cfg.d_model_default <= cfg.d_model_max
        assert cfg.steps_min < cfg.steps_default <= cfg.steps_max
        assert cfg.default_lr > 0
        assert 0 < cfg.warmup_fraction < 0.5

    def test_decay_fractions_in_range(self):
        cfg = TrainingConfig()
        for attr in ("policy_decay_fraction", "value_decay_fraction", "lbb_decay_fraction"):
            val = getattr(cfg, attr)
            assert 0 < val < 1, f"{attr} out of range: {val}"

    def test_curve_n_points_minimum(self):
        with pytest.raises(ValidationError):
            TrainingConfig(curve_n_points=5)

    def test_lbb_min_threshold_positive(self):
        with pytest.raises(ValidationError):
            TrainingConfig(lbb_min_threshold=0)


# ---------------------------------------------------------------------------
# DashboardConfig (composite)
# ---------------------------------------------------------------------------


class TestDashboardConfig:
    def test_instantiation(self):
        cfg = DashboardConfig()
        assert isinstance(cfg.app, AppConfig)
        assert isinstance(cfg.game, GameConfig)
        assert isinstance(cfg.pde, PDEConfig)
        assert isinstance(cfg.poc, PoCConfig)
        assert isinstance(cfg.training, TrainingConfig)

    def test_nested_override(self):
        cfg = DashboardConfig(app=AppConfig(port=9000))
        assert cfg.app.port == 9000
        assert cfg.pde.default_grid_size == PDEConfig().default_grid_size

    def test_default_config_singleton(self):
        assert DEFAULT_CONFIG.app.port == 7860
        assert DEFAULT_CONFIG.pde.default_grid_size == 9

    def test_json_round_trip(self):
        cfg = DashboardConfig()
        json_str = cfg.model_dump_json()
        reloaded = DashboardConfig.model_validate_json(json_str)
        assert reloaded.app.port == cfg.app.port
        assert reloaded.pde.default_grid_size == cfg.pde.default_grid_size
