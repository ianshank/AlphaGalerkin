"""Tests for the plasma physics and stellarator optimization module."""
from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.planning.plasma_physics import (
    CoilAction,
    CoilActionType,
    CoilGeometry,
    ModelSelectionAction,
    ModelSelectionActionType,
    PlasmaModelSelector,
    PlasmaModelState,
    PlasmaModelType,
    PlasmaRegion,
    StellaratorOptimizer,
    StellaratorState,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def sample_coil() -> CoilGeometry:
    """A minimal coil geometry for testing."""
    rng = np.random.default_rng(0)
    return CoilGeometry(
        control_points=rng.normal(0, 1, size=(10, 3)),
        current=1.5,
        winding_number=2,
    )


@pytest.fixture()
def sample_stellarator_state() -> StellaratorState:
    """A minimal stellarator state with 3 coils."""
    rng = np.random.default_rng(0)
    coils = [
        CoilGeometry(
            control_points=rng.normal(0, 1, size=(10, 3)),
            current=1.0,
            winding_number=1,
        )
        for _ in range(3)
    ]
    return StellaratorState(
        coils=coils,
        num_field_periods=5,
        target_aspect_ratio=10.0,
        mhd_stability_metric=0.5,
        neoclassical_transport=0.3,
        fast_particle_loss=0.2,
        coil_complexity=3.0,
        step=0,
        max_coils=10,
    )


@pytest.fixture()
def optimizer() -> StellaratorOptimizer:
    """A stellarator optimizer configured for testing."""
    return StellaratorOptimizer(
        max_coils=10,
        num_simulations=3,
        stability_weight=1.0,
        transport_weight=1.0,
        complexity_weight=0.3,
        current_step=0.01,
        position_step=0.01,
    )


@pytest.fixture()
def sample_plasma_state() -> PlasmaModelState:
    """A minimal plasma model state with 3 regions."""
    return PlasmaModelState(
        regions=[
            PlasmaRegion(
                bounds=(0.0, 0.3),
                model=PlasmaModelType.MHD,
                accuracy=0.05,
                cost_per_step=1.0,
            ),
            PlasmaRegion(
                bounds=(0.3, 0.7),
                model=PlasmaModelType.KINETIC,
                accuracy=0.01,
                cost_per_step=50.0,
            ),
            PlasmaRegion(
                bounds=(0.7, 1.0),
                model=PlasmaModelType.MHD,
                accuracy=0.04,
                cost_per_step=1.0,
            ),
        ],
        total_cost=10.0,
        budget=200.0,
        accuracy_target=0.01,
        step=0,
    )


@pytest.fixture()
def selector() -> PlasmaModelSelector:
    """A plasma model selector configured for testing."""
    return PlasmaModelSelector(
        num_simulations=3,
        budget=200.0,
    )


# ------------------------------------------------------------------
# Stellarator Coil Design Tests
# ------------------------------------------------------------------


class TestCoilGeometryClone:
    """CoilGeometry.clone produces an independent copy."""

    def test_coil_geometry_clone(self, sample_coil: CoilGeometry) -> None:
        cloned = sample_coil.clone()

        # Must be a different object
        assert cloned is not sample_coil
        assert cloned.control_points is not sample_coil.control_points

        # Values must match
        np.testing.assert_array_equal(
            cloned.control_points, sample_coil.control_points,
        )
        assert cloned.current == sample_coil.current
        assert cloned.winding_number == sample_coil.winding_number

        # Mutation independence
        cloned.current = 999.0
        cloned.control_points[0, 0] = -999.0
        assert sample_coil.current == 1.5
        assert sample_coil.control_points[0, 0] != -999.0


class TestStellaratorStateClone:
    """StellaratorState.clone produces an independent copy."""

    def test_stellarator_state_clone(
        self, sample_stellarator_state: StellaratorState,
    ) -> None:
        cloned = sample_stellarator_state.clone()

        # Must be a different object
        assert cloned is not sample_stellarator_state
        assert cloned.coils is not sample_stellarator_state.coils
        assert len(cloned.coils) == len(sample_stellarator_state.coils)

        # Each coil must be independently copied
        for orig, copy in zip(
            sample_stellarator_state.coils, cloned.coils, strict=True,
        ):
            assert copy is not orig
            assert copy.control_points is not orig.control_points
            np.testing.assert_array_equal(
                copy.control_points, orig.control_points,
            )

        # Scalar values must match
        assert cloned.mhd_stability_metric == sample_stellarator_state.mhd_stability_metric
        assert cloned.neoclassical_transport == sample_stellarator_state.neoclassical_transport
        assert cloned.coil_complexity == sample_stellarator_state.coil_complexity
        assert cloned.step == sample_stellarator_state.step

        # Mutation independence
        cloned.mhd_stability_metric = 999.0
        assert sample_stellarator_state.mhd_stability_metric == 0.5


class TestStellaratorTotalObjective:
    """StellaratorState.total_objective computes weighted sum."""

    def test_stellarator_total_objective(
        self, sample_stellarator_state: StellaratorState,
    ) -> None:
        # With finite metrics: 0.5 + 0.3 + 0.2 + 3.0 = 4.0
        total = sample_stellarator_state.total_objective
        assert total == pytest.approx(4.0)

    def test_total_objective_ignores_infinite(self) -> None:
        """Infinite metrics are excluded from the sum."""
        state = StellaratorState(
            coils=[],
            mhd_stability_metric=float("inf"),
            neoclassical_transport=0.5,
            fast_particle_loss=float("inf"),
            coil_complexity=1.0,
        )
        # Only neoclassical_transport (0.5) + coil_complexity (1.0)
        assert state.total_objective == pytest.approx(1.5)


class TestStellaratorValidActions:
    """StellaratorOptimizer.get_valid_actions returns correct actions."""

    def test_stellarator_valid_actions(
        self,
        optimizer: StellaratorOptimizer,
        sample_stellarator_state: StellaratorState,
    ) -> None:
        actions = optimizer.get_valid_actions(sample_stellarator_state)
        action_types = {a.action_type for a in actions}

        # Must always contain NO_OP
        assert CoilActionType.NO_OP in action_types

        # With 3 coils (< 10 max), ADD_COIL should be valid
        assert CoilActionType.ADD_COIL in action_types

        # With 3 coils (> 1), REMOVE_COIL should be valid
        assert CoilActionType.REMOVE_COIL in action_types

        # Per-coil actions should be present
        assert CoilActionType.ADJUST_CURRENT in action_types
        assert CoilActionType.MOVE_COIL_POINT in action_types
        assert CoilActionType.ADJUST_WINDING in action_types

    def test_no_add_at_max_coils(
        self,
        optimizer: StellaratorOptimizer,
    ) -> None:
        """ADD_COIL is excluded when at max coil count."""
        rng = np.random.default_rng(1)
        coils = [
            CoilGeometry(control_points=rng.normal(0, 1, size=(5, 3)))
            for _ in range(10)
        ]
        state = StellaratorState(coils=coils, max_coils=10)
        actions = optimizer.get_valid_actions(state)
        action_types = {a.action_type for a in actions}
        assert CoilActionType.ADD_COIL not in action_types

    def test_no_remove_with_one_coil(
        self,
        optimizer: StellaratorOptimizer,
    ) -> None:
        """REMOVE_COIL is excluded when only 1 coil remains."""
        rng = np.random.default_rng(2)
        coils = [
            CoilGeometry(control_points=rng.normal(0, 1, size=(5, 3))),
        ]
        state = StellaratorState(coils=coils, max_coils=10)
        actions = optimizer.get_valid_actions(state)
        action_types = {a.action_type for a in actions}
        assert CoilActionType.REMOVE_COIL not in action_types


class TestStellaratorAdjustCurrent:
    """Applying ADJUST_CURRENT modifies the coil current."""

    def test_stellarator_adjust_current(
        self,
        optimizer: StellaratorOptimizer,
        sample_stellarator_state: StellaratorState,
    ) -> None:
        original_current = sample_stellarator_state.coils[0].current
        action = CoilAction(
            action_type=CoilActionType.ADJUST_CURRENT,
            coil_index=0,
            params={"delta": 0.05},
        )
        new_state = optimizer.apply_action(sample_stellarator_state, action)

        assert new_state.coils[0].current == pytest.approx(
            original_current + 0.05,
        )
        assert new_state.step == sample_stellarator_state.step + 1
        # Original is unmodified
        assert sample_stellarator_state.coils[0].current == original_current


class TestStellaratorAddCoil:
    """Applying ADD_COIL increases the coil count."""

    def test_stellarator_add_coil(
        self,
        optimizer: StellaratorOptimizer,
        sample_stellarator_state: StellaratorState,
    ) -> None:
        original_count = len(sample_stellarator_state.coils)
        action = CoilAction(
            action_type=CoilActionType.ADD_COIL,
            params={"num_control_points": 8},
        )
        new_state = optimizer.apply_action(sample_stellarator_state, action)

        assert len(new_state.coils) == original_count + 1
        assert new_state.coils[-1].control_points.shape == (8, 3)
        assert new_state.coil_complexity == pytest.approx(
            sample_stellarator_state.coil_complexity + 1.0,
        )
        # Original is unmodified
        assert len(sample_stellarator_state.coils) == original_count


class TestStellaratorRemoveCoil:
    """Applying REMOVE_COIL decreases the coil count."""

    def test_stellarator_remove_coil(
        self,
        optimizer: StellaratorOptimizer,
        sample_stellarator_state: StellaratorState,
    ) -> None:
        original_count = len(sample_stellarator_state.coils)
        action = CoilAction(
            action_type=CoilActionType.REMOVE_COIL,
            coil_index=1,
        )
        new_state = optimizer.apply_action(sample_stellarator_state, action)

        assert len(new_state.coils) == original_count - 1
        assert new_state.coil_complexity == pytest.approx(
            sample_stellarator_state.coil_complexity - 1.0,
        )
        assert new_state.step == sample_stellarator_state.step + 1
        # Original is unmodified
        assert len(sample_stellarator_state.coils) == original_count


class TestStellaratorOptimizerPlans:
    """StellaratorOptimizer.plan_next_action returns a valid action."""

    def test_stellarator_optimizer_plans(
        self,
        optimizer: StellaratorOptimizer,
        sample_stellarator_state: StellaratorState,
    ) -> None:
        action = optimizer.plan_next_action(sample_stellarator_state)
        assert isinstance(action, CoilAction)
        assert isinstance(action.action_type, CoilActionType)

        # The returned action must be among the valid set
        valid_types = {
            a.action_type
            for a in optimizer.get_valid_actions(sample_stellarator_state)
        }
        assert action.action_type in valid_types

    def test_stellarator_optimizer_plans_with_physics_fn(
        self,
        optimizer: StellaratorOptimizer,
        sample_stellarator_state: StellaratorState,
    ) -> None:
        """plan_next_action works with a user-supplied physics function."""
        def physics_fn(state: StellaratorState) -> dict[str, float]:
            return {
                "mhd_stability": 0.1,
                "neoclassical_transport": 0.05,
                "fast_particle_loss": 0.02,
            }

        action = optimizer.plan_next_action(
            sample_stellarator_state, physics_fn=physics_fn,
        )
        assert isinstance(action, CoilAction)
        assert isinstance(action.action_type, CoilActionType)


# ------------------------------------------------------------------
# Plasma Model Selection Tests
# ------------------------------------------------------------------


class TestPlasmaRegionDataclass:
    """PlasmaRegion holds correct data."""

    def test_plasma_region_dataclass(self) -> None:
        region = PlasmaRegion(
            bounds=(0.0, 1.0),
            model=PlasmaModelType.GYROKINETIC,
            accuracy=0.02,
            cost_per_step=10.0,
        )
        assert region.bounds == (0.0, 1.0)
        assert region.model == PlasmaModelType.GYROKINETIC
        assert region.accuracy == 0.02
        assert region.cost_per_step == 10.0

    def test_plasma_region_defaults(self) -> None:
        region = PlasmaRegion(bounds=(0.0, 0.5))
        assert region.model == PlasmaModelType.MHD
        assert region.accuracy == 0.0
        assert region.cost_per_step == 1.0


class TestPlasmaModelStateClone:
    """PlasmaModelState.clone produces an independent copy."""

    def test_plasma_model_state_clone(
        self, sample_plasma_state: PlasmaModelState,
    ) -> None:
        cloned = sample_plasma_state.clone()

        # Must be a different object
        assert cloned is not sample_plasma_state
        assert cloned.regions is not sample_plasma_state.regions
        assert len(cloned.regions) == len(sample_plasma_state.regions)

        # Each region must be independently copied
        for orig, copy in zip(
            sample_plasma_state.regions, cloned.regions, strict=True,
        ):
            assert copy is not orig
            assert copy.bounds == orig.bounds
            assert copy.model == orig.model
            assert copy.accuracy == orig.accuracy

        # Scalar values must match
        assert cloned.total_cost == sample_plasma_state.total_cost
        assert cloned.budget == sample_plasma_state.budget
        assert cloned.step == sample_plasma_state.step

        # Mutation independence
        cloned.total_cost = 999.0
        cloned.regions[0].accuracy = 999.0
        assert sample_plasma_state.total_cost == 10.0
        assert sample_plasma_state.regions[0].accuracy == 0.05


class TestModelSelectorValidActions:
    """PlasmaModelSelector.get_valid_actions returns correct actions."""

    def test_model_selector_valid_actions(
        self,
        selector: PlasmaModelSelector,
        sample_plasma_state: PlasmaModelState,
    ) -> None:
        actions = selector.get_valid_actions(sample_plasma_state)
        action_types = {a.action_type for a in actions}

        # Must always contain NO_OP
        assert ModelSelectionActionType.NO_OP in action_types

        # SET_REGION_MODEL should be available (regions have models
        # that can be changed)
        assert ModelSelectionActionType.SET_REGION_MODEL in action_types

        # SPLIT_REGION should be available (regions are wide enough)
        assert ModelSelectionActionType.SPLIT_REGION in action_types

    def test_merge_available_for_same_model_neighbors(
        self,
        selector: PlasmaModelSelector,
        sample_plasma_state: PlasmaModelState,
    ) -> None:
        """MERGE_REGIONS is available when adjacent regions share a model."""
        # Regions 0 and 2 are MHD but not adjacent; make region 1 MHD too
        sample_plasma_state.regions[1].model = PlasmaModelType.MHD
        actions = selector.get_valid_actions(sample_plasma_state)
        action_types = {a.action_type for a in actions}
        assert ModelSelectionActionType.MERGE_REGIONS in action_types


class TestModelSelectorSetModel:
    """Applying SET_REGION_MODEL changes the physics model."""

    def test_model_selector_set_model(
        self,
        selector: PlasmaModelSelector,
        sample_plasma_state: PlasmaModelState,
    ) -> None:
        action = ModelSelectionAction(
            action_type=ModelSelectionActionType.SET_REGION_MODEL,
            region_index=0,
            target_model=PlasmaModelType.HYBRID,
        )
        new_state = selector.apply_action(sample_plasma_state, action)

        assert new_state.regions[0].model == PlasmaModelType.HYBRID
        assert new_state.step == sample_plasma_state.step + 1
        # Original is unmodified
        assert sample_plasma_state.regions[0].model == PlasmaModelType.MHD

    def test_set_model_updates_cost(
        self,
        selector: PlasmaModelSelector,
        sample_plasma_state: PlasmaModelState,
    ) -> None:
        """Setting a more expensive model increases total cost."""
        action = ModelSelectionAction(
            action_type=ModelSelectionActionType.SET_REGION_MODEL,
            region_index=0,
            target_model=PlasmaModelType.GYROKINETIC,
        )
        new_state = selector.apply_action(sample_plasma_state, action)

        # Gyrokinetic costs 10.0, MHD costs 1.0, delta = +9.0
        expected_cost = sample_plasma_state.total_cost + (10.0 - 1.0)
        assert new_state.total_cost == pytest.approx(expected_cost)
        assert new_state.regions[0].cost_per_step == 10.0


class TestModelSelectorSplitRegion:
    """Applying SPLIT_REGION divides a region in half."""

    def test_model_selector_split_region(
        self,
        selector: PlasmaModelSelector,
        sample_plasma_state: PlasmaModelState,
    ) -> None:
        original_count = len(sample_plasma_state.regions)
        original_bounds = sample_plasma_state.regions[1].bounds

        action = ModelSelectionAction(
            action_type=ModelSelectionActionType.SPLIT_REGION,
            region_index=1,
        )
        new_state = selector.apply_action(sample_plasma_state, action)

        assert len(new_state.regions) == original_count + 1
        # Check that the split produced correct bounds
        mid = (original_bounds[0] + original_bounds[1]) / 2.0
        assert new_state.regions[1].bounds == pytest.approx(
            (original_bounds[0], mid),
        )
        assert new_state.regions[2].bounds == pytest.approx(
            (mid, original_bounds[1]),
        )
        # Original is unmodified
        assert len(sample_plasma_state.regions) == original_count


class TestModelCostsDict:
    """PlasmaModelSelector.DEFAULT_MODEL_COSTS contains all model types."""

    def test_model_costs_dict(self) -> None:
        costs = PlasmaModelSelector.DEFAULT_MODEL_COSTS
        # Must contain all model types
        for model_type in PlasmaModelType:
            assert model_type.value in costs, (
                f"Missing cost for model type: {model_type.value}"
            )

        # Kinetic must be the most expensive
        assert costs["kinetic"] > costs["gyrokinetic"]
        assert costs["gyrokinetic"] > costs["hybrid"]
        assert costs["hybrid"] > costs["fluid"]
        assert costs["fluid"] > costs["mhd"]

    def test_custom_model_costs(self) -> None:
        """Custom costs override defaults."""
        custom = {"mhd": 2.0, "kinetic": 100.0}
        selector = PlasmaModelSelector(model_costs=custom)
        assert selector._model_costs["mhd"] == 2.0
        assert selector._model_costs["kinetic"] == 100.0
