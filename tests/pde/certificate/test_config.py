"""Config schema tests — ``MetricThreshold`` reuse and ``extra='forbid'`` typo guard."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.pde.certificate import CertificateConfig
from src.pde.certificate.config import (
    DEFAULT_HEURISTIC_GRID_RESOLUTION,
    DEFAULT_TRACK_A_OVERHEAD_FRACTION,
    DEFAULT_TRACK_B_BUDGET_S_CPU,
)
from src.poc.config import MetricThreshold


def test_defaults_use_named_constants() -> None:
    """No hardcoded values discipline — defaults come from module constants."""
    cfg = CertificateConfig()
    assert cfg.track_a_overhead_fraction == DEFAULT_TRACK_A_OVERHEAD_FRACTION
    assert cfg.track_b_budget_s == DEFAULT_TRACK_B_BUDGET_S_CPU
    assert cfg.heuristic_grid_resolution == DEFAULT_HEURISTIC_GRID_RESOLUTION


def test_config_is_frozen() -> None:
    cfg = CertificateConfig()
    with pytest.raises(ValidationError):
        cfg.enabled = False  # type: ignore[misc]


def test_extra_fields_forbidden() -> None:
    """Typos in config keys must fail loudly, not silently."""
    with pytest.raises(ValidationError):
        CertificateConfig(unknown_knob=42)  # type: ignore[call-arg]


def test_thresholds_reuse_canonical_metric_threshold() -> None:
    """Peer-review correction: no parallel schema — reuse ``src.poc.config.MetricThreshold``."""
    threshold = MetricThreshold(
        name="effectivity_index", operator="<", value=3.0, description="placeholder"
    )
    cfg = CertificateConfig(thresholds=[threshold])
    assert cfg.thresholds == [threshold]
    assert isinstance(cfg.thresholds[0], MetricThreshold)


def test_positive_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        CertificateConfig(track_a_overhead_fraction=0.0)
    with pytest.raises(ValidationError):
        CertificateConfig(track_b_budget_s=-1.0)
    with pytest.raises(ValidationError):
        CertificateConfig(heuristic_grid_resolution=0)


def test_prefer_rigorous_default_true() -> None:
    """CI configs flip this to False; the default is production-oriented."""
    assert CertificateConfig().prefer_rigorous is True
