"""Property-based tests for firefighting solver invariants.

Uses Hypothesis to verify fire physics constraints hold
across random configurations.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.firefighting.config.fire import FireConfig
from src.firefighting.fire.fuel import FuelModel, FuelState
from src.firefighting.fire.radiation import RadiativeHeatTransfer
from src.firefighting.output.confidence import ConfidenceEstimator


class TestFireConservation:
    """Property tests for energy conservation in fire spread."""

    @given(
        temp=st.floats(min_value=200.0, max_value=2000.0),
        fuel_load=st.floats(min_value=0.0, max_value=10.0),
        wind=st.floats(min_value=0.0, max_value=30.0),
    )
    @settings(max_examples=50)
    def test_heat_source_non_negative(
        self,
        temp: float,
        fuel_load: float,
        wind: float,
    ) -> None:
        """Heat source from combustion should never be negative."""
        config = FireConfig(name="prop_test")
        model = FuelModel(config)
        t = np.array([[temp]])
        fuel = np.array([[fuel_load]])
        w = np.array([[wind]])
        q = model.heat_source(t, fuel, w)
        assert np.all(q >= 0)

    @given(
        temp=st.floats(min_value=200.0, max_value=2000.0),
        fuel_load=st.floats(min_value=0.0, max_value=10.0),
    )
    @settings(max_examples=50)
    def test_consumption_rate_non_negative(self, temp: float, fuel_load: float) -> None:
        """Fuel consumption rate should never be negative."""
        config = FireConfig(name="prop_test")
        model = FuelModel(config)
        t = np.array([[temp]])
        fuel = np.array([[fuel_load]])
        rate = model.fuel_consumption_rate(t, fuel)
        assert np.all(rate >= 0)

    @given(
        consumed=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=30)
    def test_fuel_remaining_bounded(self, consumed: float) -> None:
        """Fuel remaining must be in [0, 1]."""
        fuel = FuelState(
            loading=np.array([[1.0]]),
            moisture=np.array([[0.1]]),
            consumed=np.array([[consumed]]),
        )
        remaining = fuel.available
        assert np.all(remaining >= 0.0)
        assert np.all(remaining <= 1.0)


class TestRadiationProperties:
    """Property tests for radiation heat transfer."""

    @given(
        dx=st.floats(min_value=1.0, max_value=100.0),
        dy=st.floats(min_value=1.0, max_value=100.0),
    )
    @settings(max_examples=30)
    def test_radiation_zero_at_ambient(self, dx: float, dy: float) -> None:
        """No radiation from cells at ambient temperature."""
        config = FireConfig(name="prop_test")
        rad = RadiativeHeatTransfer(config)
        t = np.array([[300.0]])
        burning = np.array([[False]])
        q = rad.compute(t, burning, dx, dy)
        np.testing.assert_allclose(q, 0.0, atol=1e-10)


class TestConfidenceProperties:
    """Property tests for prediction confidence."""

    @given(
        sensor_conf=st.floats(min_value=0.0, max_value=1.0),
        age=st.floats(min_value=0.0, max_value=3600.0),
    )
    @settings(max_examples=50)
    def test_confidence_bounded(self, sensor_conf: float, age: float) -> None:
        """Confidence values must be in [0, 1]."""
        estimator = ConfidenceEstimator()
        obs = np.ones((5, 5), dtype=bool)
        conf = estimator.compute(
            sensor_confidence=sensor_conf,
            observation_mask=obs,
            prediction_age_s=age,
        )
        assert np.all(conf.values >= 0.0)
        assert np.all(conf.values <= 1.0)

    @given(
        age1=st.floats(min_value=0.0, max_value=1000.0),
        age2=st.floats(min_value=0.0, max_value=1000.0),
    )
    @settings(max_examples=30)
    def test_confidence_decreases_with_age(self, age1: float, age2: float) -> None:
        """Older predictions should have lower or equal confidence."""
        estimator = ConfidenceEstimator()
        obs = np.ones((5, 5), dtype=bool)

        c1 = estimator.compute(
            sensor_confidence=0.9,
            observation_mask=obs,
            prediction_age_s=min(age1, age2),
        )
        c2 = estimator.compute(
            sensor_confidence=0.9,
            observation_mask=obs,
            prediction_age_s=max(age1, age2),
        )
        assert c1.mean_confidence >= c2.mean_confidence - 1e-10
