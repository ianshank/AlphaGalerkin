"""Config tests for the ``stochastic_galerkin_compare`` scenario (AC8 + AQA).

Includes the peer-review blocker guard: ``load_config_from_dict`` must return
the typed config (without the lazy dispatch branch it silently degrades to
``BaseScenarioConfig`` and the CLI hard-fails), and the AQA test asserting the
spec's Thresholds table equals ``get_default_thresholds()``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.poc.config import load_config_from_dict
from src.poc.scenarios.stochastic_galerkin_compare_config import (
    DEFAULT_STOCHASTIC_MSE_GATE,
    SCENARIO_NAME,
    StochasticGalerkinCompareConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestConfigValidation:
    def test_defaults_valid(self):
        cfg = StochasticGalerkinCompareConfig()
        assert cfg.name == SCENARIO_NAME
        assert cfg.stochastic_mse_gate == DEFAULT_STOCHASTIC_MSE_GATE

    def test_name_locked(self):
        with pytest.raises(ValidationError, match="dispatch key"):
            StochasticGalerkinCompareConfig(name="something_else")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            StochasticGalerkinCompareConfig(bogus=1)

    def test_dt_exceeding_horizon_rejected(self):
        with pytest.raises(ValidationError, match="strang_dt"):
            StochasticGalerkinCompareConfig(strang_dt=2.0, t_end=1.0)

    def test_p0_range_rejected(self):
        with pytest.raises(ValidationError, match="p0_min"):
            StochasticGalerkinCompareConfig(p0_min=0.4, p0_max=0.2)

    def test_non_2d_rejected(self):
        with pytest.raises(ValidationError, match="2D"):
            StochasticGalerkinCompareConfig(drift_bias=[0.1])
        with pytest.raises(ValidationError, match="drift_matrix"):
            StochasticGalerkinCompareConfig(drift_matrix=[[-1.0]])

    def test_artifact_basename_rules(self):
        with pytest.raises(ValidationError, match="extension"):
            StochasticGalerkinCompareConfig(artifact_basename="foo.csv")

    def test_resolved_seeds(self):
        cfg = StochasticGalerkinCompareConfig(seed=10, n_seeds=3)
        seeds = cfg.resolved_seeds()
        assert len(seeds) == 3
        assert seeds[0] == 10
        assert len(set(seeds)) == 3


class TestDispatchGuard:
    """Peer-review blocker guard: the lazy branch in load_config_from_dict."""

    def test_dispatch_by_name(self):
        config = load_config_from_dict({"name": SCENARIO_NAME})
        assert type(config).__name__ == StochasticGalerkinCompareConfig.__name__

    def test_dispatch_by_scenario_type(self):
        config = load_config_from_dict({}, scenario_type=SCENARIO_NAME)
        assert type(config).__name__ == StochasticGalerkinCompareConfig.__name__

    def test_dispatch_validates_fields(self):
        with pytest.raises(ValidationError):
            load_config_from_dict({"name": SCENARIO_NAME, "strang_dt": 5.0, "t_end": 1.0})


class TestThresholdAqa:
    """AQA: spec Thresholds table == get_default_thresholds() (spec AC8)."""

    def test_single_lower_gate_on_stochastic_mse(self):
        thresholds = StochasticGalerkinCompareConfig().get_default_thresholds()
        assert len(thresholds) == 1
        gate = thresholds[0]
        assert gate.name == "stochastic_density_mse"
        assert gate.operator == "<"
        assert gate.value == DEFAULT_STOCHASTIC_MSE_GATE

    def test_gate_tracks_config_field(self):
        cfg = StochasticGalerkinCompareConfig(stochastic_mse_gate=5e-7)
        assert cfg.get_default_thresholds()[0].value == 5e-7

    def test_no_ratio_gate(self):
        """The honesty rule: the stochastic/deterministic ratio is never gated."""
        names = [t.name for t in StochasticGalerkinCompareConfig().get_default_thresholds()]
        assert all("ratio" not in n for n in names)
        assert all("deterministic" not in n for n in names)


class TestShippedYamls:
    @pytest.mark.parametrize("basename", ["ci", "demo"])
    def test_yaml_loads_and_validates(self, basename):
        path = REPO_ROOT / "config" / "scenarios" / f"stochastic_galerkin_compare_{basename}.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        entries = [entry for entry in raw["scenarios"] if entry.get("name") == SCENARIO_NAME]
        assert len(entries) == 1
        config = load_config_from_dict(entries[0])
        assert type(config).__name__ == StochasticGalerkinCompareConfig.__name__

    def test_ci_yaml_uses_micro_budget(self):
        path = REPO_ROOT / "config" / "scenarios" / "stochastic_galerkin_compare_ci.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        entry = raw["scenarios"][0]
        assert entry["grid_n"] <= 16
        assert entry["n_epochs"] <= 10
        assert entry["n_seeds"] == 1
