"""Tests for TransferBaselineCompareConfig (name-lock, bounds, validators, thresholds).

AQA: the single default threshold must mirror the spec's Thresholds table
(``transfer_mse_ratio_<t>x<t> < transfer_ratio_pass_threshold``).
"""

from __future__ import annotations

import pytest

from src.poc.config import MetricThreshold, load_config_from_dict
from src.poc.scenarios.transfer_baseline_compare_config import (
    SCENARIO_NAME,
    TransferBaselineCompareConfig,
)


def _config(**overrides: object) -> TransferBaselineCompareConfig:
    params: dict[str, object] = {"name": SCENARIO_NAME}
    params.update(overrides)
    return TransferBaselineCompareConfig(**params)  # type: ignore[arg-type]


class TestValidation:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.name == SCENARIO_NAME
        assert cfg.device == "cpu"
        assert cfg.train_resolution == 9
        assert cfg.target_resolution == 19
        assert cfg.n_seeds == 5
        assert cfg.matched_budget_mode == "grad_steps"
        assert cfg.transfer_ratio_pass_threshold == 1.0
        assert cfg.cnn_channels is None
        assert cfg.artifact_basename == "transfer_baseline_compare"

    def test_name_locked(self) -> None:
        with pytest.raises(ValueError, match="name must be"):
            _config(name="wrong_name")

    def test_odd_kernel_enforced(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            _config(cnn_kernel_size=4)

    def test_odd_kernel_accepts_odd(self) -> None:
        assert _config(cnn_kernel_size=5).cnn_kernel_size == 5

    def test_secondary_resolutions_reject_small(self) -> None:
        with pytest.raises(ValueError, match="secondary resolutions must be >= 3"):
            _config(secondary_resolutions=[9, 2])

    def test_artifact_basename_rejects_extension(self) -> None:
        with pytest.raises(ValueError, match="must not include a file extension"):
            _config(artifact_basename="foo.csv")
        with pytest.raises(ValueError, match="must not include a file extension"):
            _config(artifact_basename="foo.png")

    def test_artifact_basename_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _config(artifact_basename="")

    def test_target_must_exceed_train(self) -> None:
        with pytest.raises(ValueError, match="target_resolution must exceed train_resolution"):
            _config(train_resolution=19, target_resolution=9)
        with pytest.raises(ValueError, match="target_resolution must exceed train_resolution"):
            _config(train_resolution=13, target_resolution=13)

    def test_d_model_must_be_divisible_by_n_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible by n_heads"):
            _config(d_model=10, n_heads=4)
        # A divisible pair validates cleanly.
        assert _config(d_model=12, n_heads=4).d_model == 12

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("train_resolution", 2),
            ("target_resolution", 4),
            ("n_train_samples", 1),
            ("n_eval_samples", 1),
            ("n_charges", 0),
            ("charge_std", 0.0),
            ("batch_size", 0),
            ("n_epochs", 0),
            ("n_seeds", 0),
            ("n_seeds", 100),
            ("d_model", 1),
            ("dropout", 1.0),
            ("cnn_n_layers", -1),
            ("cnn_kernel_size", 9),
            ("cnn_dropout", 1.0),
            ("cnn_param_match_tolerance", 0.0),
            ("cnn_param_match_tolerance", 2.0),
            ("transfer_ratio_pass_threshold", 0.0),
            ("matched_budget_mode", "nonsense"),
        ],
    )
    def test_field_bounds_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ValueError):
            _config(**{field: value})


class TestDispatch:
    def test_load_config_from_dict_dispatches(self) -> None:
        cfg = load_config_from_dict({"name": SCENARIO_NAME, "target_resolution": 25})
        assert type(cfg).__name__ == "TransferBaselineCompareConfig"
        assert cfg.target_resolution == 25  # type: ignore[attr-defined]


class TestThresholdsAQA:
    """The config's get_default_thresholds() must match the spec's Thresholds table."""

    def test_single_gate_matches_spec(self) -> None:
        cfg = _config()
        thresholds = cfg.get_default_thresholds()
        assert len(thresholds) == 1
        gate = thresholds[0]
        assert isinstance(gate, MetricThreshold)
        assert gate.name == "transfer_mse_ratio_19x19"
        assert gate.operator == "<"
        assert gate.value == 1.0

    def test_gate_name_tracks_target_resolution(self) -> None:
        cfg = _config(target_resolution=25)
        assert cfg.target_metric_name == "transfer_mse_ratio_25x25"
        assert cfg.get_default_thresholds()[0].name == "transfer_mse_ratio_25x25"

    def test_gate_value_tracks_threshold_field(self) -> None:
        cfg = _config(transfer_ratio_pass_threshold=2.5)
        assert cfg.get_default_thresholds()[0].value == 2.5
