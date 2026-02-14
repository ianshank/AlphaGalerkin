"""Tests for the numerical relativity AMR planning framework."""

from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.planning.numerical_relativity import (
    GaugeCondition,
    NRAction,
    NRActionType,
    NRMeshManager,
    NRMeshState,
    RefinementLevel,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def sample_refinement_level() -> RefinementLevel:
    """A single refinement level for testing."""
    return RefinementLevel(
        level=1,
        center=np.array([0.0, 0.0, 0.0]),
        extent=np.array([10.0, 10.0, 10.0]),
        resolution=1.0,
    )


@pytest.fixture()
def sample_state() -> NRMeshState:
    """A minimal NR mesh state for testing."""
    return NRMeshState(
        dimension=3,
        domain_extent=1000.0,
        base_resolution=8.0,
        refinement_levels=[
            RefinementLevel(
                level=1,
                center=np.array([5.0, 0.0, 0.0]),
                extent=np.array([20.0, 20.0, 20.0]),
                resolution=4.0,
            ),
            RefinementLevel(
                level=2,
                center=np.array([5.0, 0.0, 0.0]),
                extent=np.array([10.0, 10.0, 10.0]),
                resolution=2.0,
            ),
        ],
        lapse_gauge=GaugeCondition.BONA_MASSO,
        shift_gauge=GaugeCondition.GAMMA_DRIVER,
        extraction_radius=100.0,
        puncture_locations=[
            np.array([5.0, 0.0, 0.0]),
            np.array([-5.0, 0.0, 0.0]),
        ],
        constraint_violation=0.01,
        time=0.0,
        step=0,
        max_levels=10,
        min_resolution=0.01,
    )


@pytest.fixture()
def manager() -> NRMeshManager:
    """An NR mesh manager with fast settings for tests."""
    return NRMeshManager(
        max_levels=10,
        min_resolution=0.01,
        refinement_ratio=2,
        num_simulations=3,
        constraint_weight=1.0,
        cost_weight=0.5,
        extraction_step=50.0,
        puncture_threshold=1.0,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestRefinementLevelVolume:
    """RefinementLevel.volume computes correct box volume."""

    def test_refinement_level_volume(
        self,
        sample_refinement_level: RefinementLevel,
    ) -> None:
        # extent = [10, 10, 10], so volume = 2*10 * 2*10 * 2*10 = 8000
        assert sample_refinement_level.volume == pytest.approx(8000.0)

    def test_refinement_level_volume_asymmetric(self) -> None:
        """Volume is correct with asymmetric extents."""
        rl = RefinementLevel(
            level=0,
            center=np.array([0.0, 0.0]),
            extent=np.array([5.0, 3.0]),
            resolution=1.0,
        )
        # volume = 2*5 * 2*3 = 60
        assert rl.volume == pytest.approx(60.0)


class TestNRMeshStateClone:
    """NRMeshState.clone produces an independent copy."""

    def test_nr_mesh_state_clone(self, sample_state: NRMeshState) -> None:
        cloned = sample_state.clone()

        # Must be a different object
        assert cloned is not sample_state
        assert cloned.refinement_levels is not sample_state.refinement_levels
        assert cloned.puncture_locations is not sample_state.puncture_locations

        # Values must match
        assert cloned.dimension == sample_state.dimension
        assert cloned.domain_extent == sample_state.domain_extent
        assert cloned.base_resolution == sample_state.base_resolution
        assert cloned.lapse_gauge == sample_state.lapse_gauge
        assert cloned.shift_gauge == sample_state.shift_gauge
        assert cloned.extraction_radius == sample_state.extraction_radius
        assert cloned.constraint_violation == sample_state.constraint_violation
        assert cloned.num_levels == sample_state.num_levels
        assert cloned.step == sample_state.step

    def test_clone_mutation_independence(
        self,
        sample_state: NRMeshState,
    ) -> None:
        cloned = sample_state.clone()

        # Mutate the clone
        cloned.constraint_violation = 999.0
        cloned.puncture_locations[0][0] = -999.0
        cloned.refinement_levels.pop()

        # Original must be unaffected
        assert sample_state.constraint_violation == 0.01
        assert sample_state.puncture_locations[0][0] == pytest.approx(5.0)
        assert len(sample_state.refinement_levels) == 2


class TestNRMeshTotalGridPoints:
    """NRMeshState.total_grid_points estimates correctly."""

    def test_nr_mesh_total_grid_points(
        self,
        sample_state: NRMeshState,
    ) -> None:
        total = sample_state.total_grid_points

        # Must be a positive integer
        assert isinstance(total, int)
        assert total > 0

        # Base grid: (2*1000/8)^3 = 250^3 = 15_625_000
        base_expected = int((2.0 * 1000.0 / 8.0) ** 3)
        # Level 1: (2*20/4)^3 = 10^3 = 1000
        lvl1_expected = int((2.0 * 20.0 / 4.0) ** 3)
        # Level 2: (2*10/2)^3 = 10^3 = 1000
        lvl2_expected = int((2.0 * 10.0 / 2.0) ** 3)

        expected = base_expected + lvl1_expected + lvl2_expected
        assert total == expected

    def test_total_grid_points_no_refinement(self) -> None:
        """Total points with no refinement levels is just the base grid."""
        state = NRMeshState(
            dimension=3,
            domain_extent=100.0,
            base_resolution=10.0,
        )
        # (2*100/10)^3 = 20^3 = 8000
        assert state.total_grid_points == 8000


class TestNRValidActions:
    """NRMeshManager.get_valid_actions returns correct actions."""

    def test_nr_valid_actions(
        self,
        manager: NRMeshManager,
        sample_state: NRMeshState,
    ) -> None:
        actions = manager.get_valid_actions(sample_state)

        # Must always contain at least NO_OP
        action_types = {a.action_type for a in actions}
        assert NRActionType.NO_OP in action_types

        # With 2 punctures and < max_levels, ADD_REFINEMENT_LEVEL is valid
        assert NRActionType.ADD_REFINEMENT_LEVEL in action_types

        # With 2 existing levels, COARSEN_REGION is valid
        assert NRActionType.COARSEN_REGION in action_types

        # Gauge changes should be offered
        assert NRActionType.SET_GAUGE_LAPSE in action_types
        assert NRActionType.SET_GAUGE_SHIFT in action_types

        # Extraction radius adjustment should be offered
        assert NRActionType.ADJUST_EXTRACTION_RADIUS in action_types

    def test_no_add_level_at_max(
        self,
        manager: NRMeshManager,
        sample_state: NRMeshState,
    ) -> None:
        """ADD_REFINEMENT_LEVEL is excluded when at max level count."""
        sample_state.max_levels = 2  # Already have 2 levels
        actions = manager.get_valid_actions(sample_state)
        action_types = {a.action_type for a in actions}
        assert NRActionType.ADD_REFINEMENT_LEVEL not in action_types


class TestNRRefineRegion:
    """Applying REFINE_REGION adds a child level."""

    def test_nr_refine_region(
        self,
        manager: NRMeshManager,
        sample_state: NRMeshState,
    ) -> None:
        action = NRAction(
            action_type=NRActionType.REFINE_REGION,
            params={"target_level": 1, "center": [5.0, 0.0, 0.0]},
        )
        new_state = manager.apply_action(sample_state, action)

        # Should have one more refinement level
        assert new_state.num_levels == sample_state.num_levels + 1
        assert new_state.step == sample_state.step + 1

        # The new child level should have finer resolution
        child = new_state.refinement_levels[-1]
        parent = sample_state.refinement_levels[0]
        assert child.resolution < parent.resolution

        # Original state is unmodified
        assert sample_state.num_levels == 2


class TestNRCoarsenRegion:
    """Applying COARSEN_REGION removes a level."""

    def test_nr_coarsen_region(
        self,
        manager: NRMeshManager,
        sample_state: NRMeshState,
    ) -> None:
        action = NRAction(
            action_type=NRActionType.COARSEN_REGION,
            params={"target_level": 2},
        )
        new_state = manager.apply_action(sample_state, action)

        # Should have one fewer refinement level
        assert new_state.num_levels == sample_state.num_levels - 1
        assert new_state.step == sample_state.step + 1

        # The removed level should be gone
        remaining_levels = {rl.level for rl in new_state.refinement_levels}
        assert 2 not in remaining_levels

        # Original state is unmodified
        assert sample_state.num_levels == 2


class TestNRSetGauge:
    """Applying SET_GAUGE_LAPSE / SET_GAUGE_SHIFT changes gauge conditions."""

    def test_nr_set_gauge(
        self,
        manager: NRMeshManager,
        sample_state: NRMeshState,
    ) -> None:
        # Change lapse to harmonic
        action_lapse = NRAction(
            action_type=NRActionType.SET_GAUGE_LAPSE,
            params={"target": GaugeCondition.HARMONIC.value},
        )
        new_state = manager.apply_action(sample_state, action_lapse)
        assert new_state.lapse_gauge == GaugeCondition.HARMONIC
        assert new_state.step == sample_state.step + 1

        # Change shift to puncture
        action_shift = NRAction(
            action_type=NRActionType.SET_GAUGE_SHIFT,
            params={"target": GaugeCondition.PUNCTURE.value},
        )
        new_state2 = manager.apply_action(sample_state, action_shift)
        assert new_state2.shift_gauge == GaugeCondition.PUNCTURE

        # Original state is unmodified
        assert sample_state.lapse_gauge == GaugeCondition.BONA_MASSO
        assert sample_state.shift_gauge == GaugeCondition.GAMMA_DRIVER


class TestNRAddLevel:
    """Applying ADD_REFINEMENT_LEVEL creates a new level."""

    def test_nr_add_level(
        self,
        manager: NRMeshManager,
        sample_state: NRMeshState,
    ) -> None:
        action = NRAction(
            action_type=NRActionType.ADD_REFINEMENT_LEVEL,
            params={
                "puncture_index": 1,
                "center": [-5.0, 0.0, 0.0],
            },
        )
        new_state = manager.apply_action(sample_state, action)

        # Should have one more level
        assert new_state.num_levels == sample_state.num_levels + 1
        assert new_state.step == sample_state.step + 1

        # New level should be centred at the specified location
        new_level = new_state.refinement_levels[-1]
        np.testing.assert_array_almost_equal(
            new_level.center,
            np.array([-5.0, 0.0, 0.0]),
        )

        # New level should have finer resolution than existing finest
        finest_existing = min(rl.resolution for rl in sample_state.refinement_levels)
        assert new_level.resolution <= finest_existing

        # Original is unmodified
        assert sample_state.num_levels == 2


class TestNRManagerPlansAction:
    """NRMeshManager.plan_next_action returns a valid action."""

    def test_nr_manager_plans_action(
        self,
        manager: NRMeshManager,
        sample_state: NRMeshState,
    ) -> None:
        # A constraint function that returns a fixed value
        def constraint_fn(state: NRMeshState) -> float:
            return 0.005

        action = manager.plan_next_action(sample_state, constraint_fn)
        assert isinstance(action, NRAction)
        assert isinstance(action.action_type, NRActionType)

        # The returned action must be among the valid set
        valid_types = {a.action_type for a in manager.get_valid_actions(sample_state)}
        assert action.action_type in valid_types

    def test_nr_manager_plans_without_constraint_fn(
        self,
        manager: NRMeshManager,
        sample_state: NRMeshState,
    ) -> None:
        """Plan works with the built-in heuristic constraint estimate."""
        action = manager.plan_next_action(sample_state)
        assert isinstance(action, NRAction)
        assert isinstance(action.action_type, NRActionType)


class TestNRPunctureTracking:
    """Puncture tracking and refinement need detection."""

    def test_nr_puncture_tracking(
        self,
        manager: NRMeshManager,
    ) -> None:
        # State where puncture is NOT covered by fine enough resolution
        state = NRMeshState(
            dimension=3,
            puncture_locations=[np.array([5.0, 0.0, 0.0])],
            refinement_levels=[
                RefinementLevel(
                    level=1,
                    center=np.array([5.0, 0.0, 0.0]),
                    extent=np.array([10.0, 10.0, 10.0]),
                    resolution=2.0,  # Coarser than threshold=1.0
                ),
            ],
        )
        assert manager._needs_refinement_near_puncture(state) is True

    def test_puncture_covered_by_fine_level(
        self,
        manager: NRMeshManager,
    ) -> None:
        """Puncture covered by a fine level does not need refinement."""
        state = NRMeshState(
            dimension=3,
            puncture_locations=[np.array([5.0, 0.0, 0.0])],
            refinement_levels=[
                RefinementLevel(
                    level=1,
                    center=np.array([5.0, 0.0, 0.0]),
                    extent=np.array([10.0, 10.0, 10.0]),
                    resolution=0.5,  # Finer than threshold=1.0
                ),
            ],
        )
        assert manager._needs_refinement_near_puncture(state) is False

    def test_no_punctures_no_refinement_needed(
        self,
        manager: NRMeshManager,
    ) -> None:
        """No punctures means no puncture-driven refinement needed."""
        state = NRMeshState(dimension=3)
        assert manager._needs_refinement_near_puncture(state) is False


class TestGaugeConditionsEnum:
    """GaugeCondition enum has the expected members."""

    def test_gauge_conditions_enum(self) -> None:
        assert GaugeCondition.GEODESIC.value == "geodesic"
        assert GaugeCondition.HARMONIC.value == "harmonic"
        assert GaugeCondition.BONA_MASSO.value == "bona_masso"
        assert GaugeCondition.GAMMA_DRIVER.value == "gamma_driver"
        assert GaugeCondition.PUNCTURE.value == "puncture"

        # All members are strings
        for gauge in GaugeCondition:
            assert isinstance(gauge.value, str)

        # Exactly 5 gauge conditions
        assert len(GaugeCondition) == 5
