"""Shared fixtures and path setup for dashboard tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest

# ── Ensure hf_space/ is available without letting it shadow the repository src ─
# ROOT must come before HF_SPACE so that `from src.*` imports resolve to the
# main package tree and not to the hf_space copy (which can be out-of-sync).
ROOT = Path(__file__).parent.parent.parent
HF_SPACE = ROOT / "hf_space"
ROOT_STR = str(ROOT)
HF_SPACE_STR = str(HF_SPACE)
# Remove both entries then re-insert in the correct order so this conftest is
# idempotent and doesn't re-order entries on repeated collection.
sys.path[:] = [p for p in sys.path if p not in {ROOT_STR, HF_SPACE_STR}]
sys.path.insert(0, ROOT_STR)
sys.path.insert(1, HF_SPACE_STR)

# Use non-interactive backend for all matplotlib calls in tests
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register dashboard-specific markers."""
    config.addinivalue_line("markers", "dashboard: dashboard UI tests")


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dashboard_cfg():
    """Return the default DashboardConfig."""
    from dashboard.config import DashboardConfig

    return DashboardConfig()


@pytest.fixture
def pde_cfg(dashboard_cfg):
    """Return the default PDEConfig."""
    return dashboard_cfg.pde


@pytest.fixture
def poc_cfg(dashboard_cfg):
    """Return the default PoCConfig."""
    return dashboard_cfg.poc


@pytest.fixture
def training_cfg(dashboard_cfg):
    """Return the default TrainingConfig."""
    return dashboard_cfg.training


@pytest.fixture
def game_cfg(dashboard_cfg):
    """Return the default GameConfig."""
    return dashboard_cfg.game


# ---------------------------------------------------------------------------
# Mock scenario results
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_complexity_result():
    """Return a realistic mock ComplexityScenario result."""
    result = MagicMock()
    result.status.value = "passed"
    result.metrics = {
        "fnet_time_ms_n81": 1.5,
        "fnet_time_ms_n169": 2.1,
        "fnet_time_ms_n361": 3.8,
        "softmax_time_ms_n81": 5.0,
        "softmax_time_ms_n169": 18.0,
        "softmax_time_ms_n361": 72.0,
        "galerkin_time_ms_n81": 1.0,
        "galerkin_time_ms_n169": 1.4,
        "galerkin_time_ms_n361": 1.9,
        "fnet_scaling_exponent": 1.12,
        "softmax_scaling_exponent": 2.18,
        "galerkin_scaling_exponent": 0.65,
        "fnet_speedup_at_largest": 18.9,
    }
    return result


@pytest.fixture
def mock_stability_result():
    """Return a realistic mock StabilityScenario result."""
    result = MagicMock()
    result.status.value = "passed"
    result.metrics = {
        "lbb_init_mean_5x5": 0.12,
        "lbb_init_min_5x5": 0.08,
        "lbb_init_mean_9x9": 0.10,
        "lbb_init_min_9x9": 0.07,
        "lbb_init_mean_13x13": 0.09,
        "lbb_init_min_13x13": 0.06,
        "lbb_training_mean": 0.11,
        "lbb_training_min": 0.05,
        "lbb_violations": 0,
    }
    return result


# ---------------------------------------------------------------------------
# Small charge grids for PDE tests
# ---------------------------------------------------------------------------


@pytest.fixture
def small_charge_grid():
    """Return a 9×9 point-charge grid."""
    grid = np.zeros((9, 9), dtype=np.float32)
    grid[4, 4] = 1.0
    return grid


@pytest.fixture
def mock_poisson_potential(small_charge_grid):
    """Return a mock potential field matching the small_charge_grid shape."""
    n = small_charge_grid.shape[0]
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, n)).astype(np.float32)
