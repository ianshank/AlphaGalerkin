"""Tests for LShapeAMRCompareConfig (name-lock, bounds, budget, thresholds).

AQA: the single default threshold must mirror the spec's Thresholds table
(``l2_error_ratio_at_matched_dof < max_l2_ratio_at_matched_dof``).
"""

from __future__ import annotations

import pytest

from src.poc.config import MetricThreshold
from src.poc.scenarios.lshape_amr_compare_config import (
    SCENARIO_NAME,
    LShapeAMRCompareConfig,
)


def _config(**overrides: object) -> LShapeAMRCompareConfig:
    params: dict[str, object] = {"name": SCENARIO_NAME}
    params.update(overrides)
    return LShapeAMRCompareConfig(**params)  # type: ignore[arg-type]


class TestValidation:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.name == SCENARIO_NAME
        assert cfg.device == "cpu"
        assert cfg.initial_side == 4
        assert cfg.max_l2_ratio_at_matched_dof == 1.0
        assert cfg.artifact_basename == "lshape_mcts_vs_dorfler"

    def test_name_locked(self) -> None:
        with pytest.raises(ValueError, match="name must be"):
            _config(name="wrong_name")

    def test_artifact_basename_rejects_csv(self) -> None:
        with pytest.raises(ValueError, match="must not include a file extension"):
            _config(artifact_basename="foo.csv")

    def test_artifact_basename_rejects_png(self) -> None:
        with pytest.raises(ValueError, match="must not include a file extension"):
            _config(artifact_basename="foo.png")

    def test_artifact_basename_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _config(artifact_basename="")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("scale", 0.0),
            ("initial_side", 0),
            ("max_dof", 5),
            ("max_steps", 0),
            ("error_tolerance", 0.0),
            ("marking_fraction", 0.0),
            ("marking_fraction", 1.0),
            ("max_refinements", 0),
            ("n_candidate_elements", 0),
            ("n_simulations", 0),
            ("value_scale", 0.0),
            ("c_puct", 0.0),
            ("max_l2_ratio_at_matched_dof", 0.0),
        ],
    )
    def test_field_bounds(self, field: str, value: object) -> None:
        with pytest.raises(ValueError):
            _config(**{field: value})

    def test_budget_consistency_rejects_oversized_candidates(self) -> None:
        # bound is 4 * initial_side^2; initial_side=1 -> 4, so 8 is too large.
        with pytest.raises(ValueError, match="exceeds a sane bound"):
            _config(initial_side=1, n_candidate_elements=8)

    def test_budget_consistency_accepts_within_bound(self) -> None:
        cfg = _config(initial_side=4, n_candidate_elements=6)
        assert cfg.n_candidate_elements == 6


class TestThresholds:
    def test_single_threshold_shape(self) -> None:
        cfg = _config()
        thresholds = cfg.get_default_thresholds()
        assert len(thresholds) == 1
        t = thresholds[0]
        assert isinstance(t, MetricThreshold)
        assert t.name == "l2_error_ratio_at_matched_dof"
        assert t.operator == "<"
        assert t.value == 1.0

    def test_threshold_tracks_config_field(self) -> None:
        """AQA: spec <-> config agreement — threshold value follows the field."""
        cfg = _config(max_l2_ratio_at_matched_dof=0.85)
        t = cfg.get_default_thresholds()[0]
        assert t.value == pytest.approx(0.85)
        assert t.operator == "<"
        assert t.name == "l2_error_ratio_at_matched_dof"
