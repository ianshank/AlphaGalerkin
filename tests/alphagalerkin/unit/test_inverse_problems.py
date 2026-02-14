"""Tests for the inverse problem solving framework."""

from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.planning.inverse_problems import (
    InverseAction,
    InverseActionType,
    InverseProblemSolver,
    InverseProblemState,
    SensorConfig,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def sample_sensor() -> SensorConfig:
    """A minimal sensor configuration for testing."""
    return SensorConfig(
        location=np.array([0.5, 0.5]),
        noise_level=0.01,
        measurement_type="pointwise",
        measurement=None,
    )


@pytest.fixture()
def sample_state() -> InverseProblemState:
    """A minimal inverse problem state for testing."""
    rng = np.random.default_rng(0)
    return InverseProblemState(
        parameter_estimate=rng.uniform(0, 1, size=(4,)),
        parameter_bounds=[(0.0, 1.0)] * 4,
        sensors=[
            SensorConfig(
                location=np.array([0.2, 0.3]),
                noise_level=0.01,
                measurement_type="pointwise",
                measurement=None,
            ),
            SensorConfig(
                location=np.array([0.8, 0.7]),
                noise_level=0.05,
                measurement_type="pointwise",
                measurement=None,
            ),
        ],
        uncertainty=np.array([1.0, 1.0, 1.0, 1.0]),
        total_information_gain=0.0,
        measurement_budget=50,
        measurements_taken=0,
        step=0,
    )


@pytest.fixture()
def state_with_measurements() -> InverseProblemState:
    """An inverse problem state with some measurements already taken."""
    return InverseProblemState(
        parameter_estimate=np.array([0.5, 0.5, 0.5, 0.5]),
        parameter_bounds=[(0.0, 1.0)] * 4,
        sensors=[
            SensorConfig(
                location=np.array([0.2, 0.3]),
                noise_level=0.01,
                measurement_type="pointwise",
                measurement=1.5,
            ),
            SensorConfig(
                location=np.array([0.8, 0.7]),
                noise_level=0.05,
                measurement_type="pointwise",
                measurement=2.3,
            ),
            SensorConfig(
                location=np.array([0.5, 0.5]),
                noise_level=0.01,
                measurement_type="pointwise",
                measurement=None,
            ),
        ],
        uncertainty=np.array([1.0, 1.0, 1.0, 1.0]),
        total_information_gain=5.0,
        measurement_budget=50,
        measurements_taken=2,
        step=5,
    )


@pytest.fixture()
def solver() -> InverseProblemSolver:
    """A solver configured for the [0,1]^2 domain."""
    return InverseProblemSolver(
        domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
        max_sensors=10,
        num_simulations=5,
        noise_level=0.01,
        exploration_weight=1.0,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSensorConfigDataclass:
    """SensorConfig stores sensor attributes correctly."""

    def test_sensor_config_dataclass(self, sample_sensor: SensorConfig) -> None:
        np.testing.assert_array_equal(
            sample_sensor.location,
            np.array([0.5, 0.5]),
        )
        assert sample_sensor.noise_level == 0.01
        assert sample_sensor.measurement_type == "pointwise"
        assert sample_sensor.measurement is None

    def test_sensor_config_with_measurement(self) -> None:
        sensor = SensorConfig(
            location=np.array([0.1, 0.9]),
            noise_level=0.05,
            measurement_type="averaged",
            measurement=3.14,
        )
        assert sensor.measurement == pytest.approx(3.14)
        assert sensor.measurement_type == "averaged"


class TestInverseStateClone:
    """InverseProblemState.clone produces an independent copy."""

    def test_inverse_state_clone(
        self,
        sample_state: InverseProblemState,
    ) -> None:
        cloned = sample_state.clone()

        # Must be a different object
        assert cloned is not sample_state
        assert cloned.parameter_estimate is not sample_state.parameter_estimate
        assert cloned.sensors is not sample_state.sensors
        assert cloned.uncertainty is not sample_state.uncertainty

        # Values must match
        np.testing.assert_array_equal(
            cloned.parameter_estimate,
            sample_state.parameter_estimate,
        )
        assert len(cloned.sensors) == len(sample_state.sensors)
        assert cloned.measurement_budget == sample_state.measurement_budget
        assert cloned.step == sample_state.step

    def test_clone_mutation_independence(
        self,
        sample_state: InverseProblemState,
    ) -> None:
        cloned = sample_state.clone()
        cloned.parameter_estimate[0] = -999.0
        cloned.sensors.append(
            SensorConfig(location=np.array([0.0, 0.0])),
        )
        if cloned.uncertainty is not None:
            cloned.uncertainty[0] = -1.0

        assert sample_state.parameter_estimate[0] != -999.0
        assert len(sample_state.sensors) == 2
        assert sample_state.uncertainty is not None
        assert sample_state.uncertainty[0] == 1.0

    def test_clone_with_none_uncertainty(self) -> None:
        state = InverseProblemState(
            parameter_estimate=np.array([0.5]),
            parameter_bounds=[(0.0, 1.0)],
            uncertainty=None,
        )
        cloned = state.clone()
        assert cloned.uncertainty is None


class TestInverseStateRemainingMeasurements:
    """InverseProblemState.remaining_measurements computes correctly."""

    def test_inverse_state_remaining_measurements(
        self,
        sample_state: InverseProblemState,
    ) -> None:
        assert sample_state.remaining_measurements == 50

    def test_remaining_after_measurements(
        self,
        state_with_measurements: InverseProblemState,
    ) -> None:
        assert state_with_measurements.remaining_measurements == 48


class TestInverseStateMeanUncertainty:
    """InverseProblemState.mean_uncertainty computes correctly."""

    def test_inverse_state_mean_uncertainty(
        self,
        sample_state: InverseProblemState,
    ) -> None:
        assert sample_state.mean_uncertainty == pytest.approx(1.0)

    def test_mean_uncertainty_none(self) -> None:
        state = InverseProblemState(
            parameter_estimate=np.array([0.5]),
            parameter_bounds=[(0.0, 1.0)],
            uncertainty=None,
        )
        assert state.mean_uncertainty == float("inf")

    def test_mean_uncertainty_varied(self) -> None:
        state = InverseProblemState(
            parameter_estimate=np.array([0.5, 0.5]),
            parameter_bounds=[(0.0, 1.0), (0.0, 1.0)],
            uncertainty=np.array([0.2, 0.8]),
        )
        assert state.mean_uncertainty == pytest.approx(0.5)


class TestInverseValidActions:
    """InverseProblemSolver.get_valid_actions returns correct actions."""

    def test_inverse_valid_actions(
        self,
        solver: InverseProblemSolver,
        sample_state: InverseProblemState,
    ) -> None:
        actions = solver.get_valid_actions(sample_state)
        action_types = {a.action_type for a in actions}

        # Must always contain NO_OP
        assert InverseActionType.NO_OP in action_types

        # With 2 sensors (< 10 max) and budget, PLACE_SENSOR valid
        assert InverseActionType.PLACE_SENSOR in action_types

        # With sensors present, REMOVE_SENSOR valid
        assert InverseActionType.REMOVE_SENSOR in action_types

        # With unmeasured sensors and budget, TAKE_MEASUREMENT valid
        assert InverseActionType.TAKE_MEASUREMENT in action_types

    def test_inverse_valid_actions_with_measurements(
        self,
        solver: InverseProblemSolver,
        state_with_measurements: InverseProblemState,
    ) -> None:
        actions = solver.get_valid_actions(state_with_measurements)
        action_types = {a.action_type for a in actions}

        # With measured sensors, UPDATE_PRIOR should be valid
        assert InverseActionType.UPDATE_PRIOR in action_types

        # With measured sensors and uncertainty, REFINE_ESTIMATE valid
        assert InverseActionType.REFINE_ESTIMATE in action_types


class TestInverseValidActionsBudgetExhausted:
    """Budget-exhausted state limits available actions."""

    def test_inverse_valid_actions_budget_exhausted(
        self,
        solver: InverseProblemSolver,
    ) -> None:
        state = InverseProblemState(
            parameter_estimate=np.array([0.5, 0.5]),
            parameter_bounds=[(0.0, 1.0), (0.0, 1.0)],
            sensors=[
                SensorConfig(
                    location=np.array([0.5, 0.5]),
                    noise_level=0.01,
                    measurement=None,
                ),
            ],
            measurement_budget=0,
            measurements_taken=0,
            step=10,
        )
        actions = solver.get_valid_actions(state)
        action_types = {a.action_type for a in actions}

        # NO_OP always valid
        assert InverseActionType.NO_OP in action_types

        # Cannot place new sensor or take measurement at zero budget
        assert InverseActionType.PLACE_SENSOR not in action_types
        assert InverseActionType.TAKE_MEASUREMENT not in action_types

        # Can still remove sensors
        assert InverseActionType.REMOVE_SENSOR in action_types


class TestInversePlaceSensor:
    """apply_action with PLACE_SENSOR adds a sensor."""

    def test_inverse_place_sensor(
        self,
        solver: InverseProblemSolver,
        sample_state: InverseProblemState,
    ) -> None:
        location = np.array([0.5, 0.5])
        action = InverseAction(
            action_type=InverseActionType.PLACE_SENSOR,
            location=location,
        )
        new_state = solver.apply_action(sample_state, action)

        assert len(new_state.sensors) == len(sample_state.sensors) + 1
        assert new_state.step == sample_state.step + 1
        np.testing.assert_array_equal(
            new_state.sensors[-1].location,
            location,
        )
        # Original state unmodified
        assert len(sample_state.sensors) == 2


class TestInverseTakeMeasurement:
    """apply_action with TAKE_MEASUREMENT records a measurement."""

    def test_inverse_take_measurement(
        self,
        solver: InverseProblemSolver,
        sample_state: InverseProblemState,
    ) -> None:
        action = InverseAction(
            action_type=InverseActionType.TAKE_MEASUREMENT,
            params={"sensor_index": 0},
        )
        new_state = solver.apply_action(
            sample_state,
            action,
            measurement_value=1.23,
        )

        assert new_state.sensors[0].measurement == pytest.approx(1.23)
        assert new_state.measurements_taken == sample_state.measurements_taken + 1
        assert new_state.total_information_gain > sample_state.total_information_gain
        # Original state unmodified
        assert sample_state.sensors[0].measurement is None


class TestInverseRemoveSensor:
    """apply_action with REMOVE_SENSOR removes the specified sensor."""

    def test_inverse_remove_sensor(
        self,
        solver: InverseProblemSolver,
        sample_state: InverseProblemState,
    ) -> None:
        original_count = len(sample_state.sensors)
        action = InverseAction(
            action_type=InverseActionType.REMOVE_SENSOR,
            params={"sensor_index": 1},
        )
        new_state = solver.apply_action(sample_state, action)

        assert len(new_state.sensors) == original_count - 1
        assert new_state.step == sample_state.step + 1
        # Original state unmodified
        assert len(sample_state.sensors) == original_count


class TestInverseUpdatePrior:
    """apply_action with UPDATE_PRIOR reduces uncertainty."""

    def test_inverse_update_prior(
        self,
        solver: InverseProblemSolver,
        state_with_measurements: InverseProblemState,
    ) -> None:
        action = InverseAction(
            action_type=InverseActionType.UPDATE_PRIOR,
        )
        new_state = solver.apply_action(state_with_measurements, action)

        assert new_state.uncertainty is not None
        # Uncertainty should be reduced after update
        assert new_state.mean_uncertainty < state_with_measurements.mean_uncertainty

    def test_update_prior_initialises_uncertainty(
        self,
        solver: InverseProblemSolver,
    ) -> None:
        state = InverseProblemState(
            parameter_estimate=np.array([0.5, 0.5]),
            parameter_bounds=[(0.0, 1.0), (0.0, 1.0)],
            sensors=[
                SensorConfig(
                    location=np.array([0.3, 0.3]),
                    noise_level=0.01,
                    measurement=2.0,
                ),
            ],
            uncertainty=None,
            measurements_taken=1,
        )
        action = InverseAction(
            action_type=InverseActionType.UPDATE_PRIOR,
        )
        new_state = solver.apply_action(state, action)

        # Uncertainty should now be initialised
        assert new_state.uncertainty is not None
        assert len(new_state.uncertainty) == len(state.parameter_estimate)


class TestInverseSolverPlansAction:
    """InverseProblemSolver.plan_next_action returns a valid action."""

    def test_inverse_solver_plans_action(
        self,
        solver: InverseProblemSolver,
        sample_state: InverseProblemState,
    ) -> None:
        action = solver.plan_next_action(sample_state)

        assert isinstance(action, InverseAction)
        assert isinstance(action.action_type, InverseActionType)

        # The returned action must be among the valid set
        valid_types = {a.action_type for a in solver.get_valid_actions(sample_state)}
        assert action.action_type in valid_types


class TestInverseInformationGainPositive:
    """Information gain is positive for non-NO_OP actions."""

    def test_inverse_information_gain_positive(
        self,
        solver: InverseProblemSolver,
        sample_state: InverseProblemState,
    ) -> None:
        # PLACE_SENSOR should have positive information gain
        action_place = InverseAction(
            action_type=InverseActionType.PLACE_SENSOR,
            location=np.array([0.5, 0.5]),
        )
        gain_place = solver._compute_information_gain(
            sample_state,
            action_place,
        )
        assert gain_place > 0.0

        # TAKE_MEASUREMENT should have positive information gain
        action_measure = InverseAction(
            action_type=InverseActionType.TAKE_MEASUREMENT,
            params={"sensor_index": 0},
        )
        gain_measure = solver._compute_information_gain(
            sample_state,
            action_measure,
        )
        assert gain_measure > 0.0

    def test_no_op_has_zero_information_gain(
        self,
        solver: InverseProblemSolver,
        sample_state: InverseProblemState,
    ) -> None:
        action = InverseAction(action_type=InverseActionType.NO_OP)
        gain = solver._compute_information_gain(sample_state, action)
        assert gain == pytest.approx(0.0)


class TestInverseSolverReducesUncertainty:
    """A sequence of measurement actions reduces uncertainty."""

    def test_inverse_solver_reduces_uncertainty(
        self,
        solver: InverseProblemSolver,
    ) -> None:
        # Start with high uncertainty
        state = InverseProblemState(
            parameter_estimate=np.array([0.5, 0.5]),
            parameter_bounds=[(0.0, 1.0), (0.0, 1.0)],
            sensors=[],
            uncertainty=np.array([10.0, 10.0]),
            measurement_budget=20,
            measurements_taken=0,
        )
        initial_uncertainty = state.mean_uncertainty

        # Place a sensor
        place_action = InverseAction(
            action_type=InverseActionType.PLACE_SENSOR,
            location=np.array([0.3, 0.7]),
        )
        state = solver.apply_action(state, place_action)

        # Take measurement
        measure_action = InverseAction(
            action_type=InverseActionType.TAKE_MEASUREMENT,
            params={"sensor_index": 0},
        )
        state = solver.apply_action(state, measure_action, measurement_value=1.0)

        # Update prior to reduce uncertainty
        update_action = InverseAction(
            action_type=InverseActionType.UPDATE_PRIOR,
        )
        state = solver.apply_action(state, update_action)

        # Uncertainty should have decreased
        assert state.mean_uncertainty < initial_uncertainty

        # Place another sensor and repeat
        place_action2 = InverseAction(
            action_type=InverseActionType.PLACE_SENSOR,
            location=np.array([0.8, 0.2]),
        )
        state = solver.apply_action(state, place_action2)

        measure_action2 = InverseAction(
            action_type=InverseActionType.TAKE_MEASUREMENT,
            params={"sensor_index": 1},
        )
        state = solver.apply_action(
            state,
            measure_action2,
            measurement_value=2.0,
        )

        update_action2 = InverseAction(
            action_type=InverseActionType.UPDATE_PRIOR,
        )
        state = solver.apply_action(state, update_action2)

        # Uncertainty should have decreased further
        assert state.mean_uncertainty < initial_uncertainty
        assert state.measurements_taken == 2
