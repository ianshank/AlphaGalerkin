"""Comprehensive tests for the Protocol definitions in core/protocols.py.

Covers all 7 runtime-checkable Protocol classes:
1. SelectionStrategy
2. EvaluationStrategy
3. BackupStrategyProtocol
4. ActionValidator
5. PhysicsModule
6. Solver
7. FeatureExtractor

For each protocol, we verify:
- It is runtime_checkable (isinstance checks work).
- A stub class that implements the correct methods satisfies the protocol.
- A stub class missing a required method does NOT satisfy the protocol.
- Properties (name, pde_type, node_feature_dim, etc.) are checked correctly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.alphagalerkin.core.protocols import (
    ActionValidator,
    BackupStrategyProtocol,
    EvaluationStrategy,
    FeatureExtractor,
    PhysicsModule,
    SelectionStrategy,
    Solver,
)
from src.alphagalerkin.core.types import PDEType

# -------------------------------------------------------------------
# Helpers: conforming and non-conforming stubs
# -------------------------------------------------------------------


class ConformingSelectionStrategy:
    """Correctly implements SelectionStrategy."""

    def select_child(self, node: Any) -> Any:
        return node


class NonConformingSelectionStrategy:
    """Missing select_child method."""

    def pick_child(self, node: Any) -> Any:  # wrong name
        return node


class ConformingEvaluationStrategy:
    """Correctly implements EvaluationStrategy."""

    def evaluate(self, state: Any) -> float:
        return 0.5


class NonConformingEvaluationStrategy:
    """Missing evaluate method."""

    def score(self, state: Any) -> float:
        return 0.5


class ConformingBackupStrategy:
    """Correctly implements BackupStrategyProtocol."""

    def backup(self, leaf_value: float, path: Any) -> None:
        pass


class NonConformingBackupStrategy:
    """Missing backup method."""

    def propagate(self, leaf_value: float, path: Any) -> None:
        pass


class ConformingActionValidator:
    """Correctly implements ActionValidator."""

    def validate(self, action: Any, state: Any) -> bool:
        return True


class NonConformingActionValidator:
    """Missing validate method."""

    def check(self, action: Any, state: Any) -> bool:
        return True


class ConformingSolver:
    """Correctly implements Solver."""

    def solve(self, stiffness: np.ndarray, rhs: np.ndarray) -> Any:
        return {"solution": rhs}


class NonConformingSolver:
    """Missing solve method."""

    def compute(self, matrix: np.ndarray, vector: np.ndarray) -> Any:
        return {"solution": vector}


class ConformingFeatureExtractor:
    """Correctly implements FeatureExtractor (methods + properties)."""

    @property
    def node_feature_dim(self) -> int:
        return 32

    @property
    def edge_feature_dim(self) -> int:
        return 8

    @property
    def global_feature_dim(self) -> int:
        return 16

    def extract(self, state: Any) -> dict[str, Any]:
        return {"node_features": np.zeros((4, 32))}


class NonConformingFeatureExtractorMissingMethod:
    """Has properties but missing extract method."""

    @property
    def node_feature_dim(self) -> int:
        return 32

    @property
    def edge_feature_dim(self) -> int:
        return 8

    @property
    def global_feature_dim(self) -> int:
        return 16


class NonConformingFeatureExtractorMissingProperty:
    """Has extract but missing a required property."""

    @property
    def node_feature_dim(self) -> int:
        return 32

    # Missing edge_feature_dim and global_feature_dim

    def extract(self, state: Any) -> dict[str, Any]:
        return {"node_features": np.zeros((4, 32))}


class ConformingPhysicsModule:
    """Correctly implements PhysicsModule (all methods + properties)."""

    @property
    def name(self) -> str:
        return "test_pde"

    @property
    def pde_type(self) -> PDEType:
        return PDEType.ELLIPTIC

    def weak_form(self, state: Any) -> Any:
        return {}

    def boundary_conditions(self) -> list[Any]:
        return []

    def manufactured_solution(self) -> Any:
        return None

    def reward_function(
        self,
        prev_state: Any,
        action: Any,
        next_state: Any,
        solve_result: Any,
    ) -> dict[str, float]:
        return {"accuracy": 1.0}

    def state_features(self, state: Any) -> dict[str, Any]:
        return {"node_features": np.zeros((4, 8))}

    def action_validators(self) -> list[Any]:
        return []

    def default_config(self) -> Any:
        return {}


class NonConformingPhysicsModuleMissingMethod:
    """Implements properties but missing several required methods."""

    @property
    def name(self) -> str:
        return "test_pde"

    @property
    def pde_type(self) -> PDEType:
        return PDEType.ELLIPTIC

    def weak_form(self, state: Any) -> Any:
        return {}

    # Missing: boundary_conditions, manufactured_solution,
    #          reward_function, state_features, action_validators,
    #          default_config


class NonConformingPhysicsModuleMissingProperty:
    """Implements methods but missing the name property."""

    # Missing: name, pde_type properties

    def weak_form(self, state: Any) -> Any:
        return {}

    def boundary_conditions(self) -> list[Any]:
        return []

    def manufactured_solution(self) -> Any:
        return None

    def reward_function(
        self,
        prev_state: Any,
        action: Any,
        next_state: Any,
        solve_result: Any,
    ) -> dict[str, float]:
        return {}

    def state_features(self, state: Any) -> dict[str, Any]:
        return {}

    def action_validators(self) -> list[Any]:
        return []

    def default_config(self) -> Any:
        return {}


class EmptyClass:
    """A class with no methods at all."""

    pass


# -------------------------------------------------------------------
# Tests: runtime_checkable verification
# -------------------------------------------------------------------


ALL_PROTOCOLS = [
    SelectionStrategy,
    EvaluationStrategy,
    BackupStrategyProtocol,
    ActionValidator,
    PhysicsModule,
    Solver,
    FeatureExtractor,
]


class TestRuntimeCheckable:
    """Verify all protocols are runtime_checkable."""

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
    def test_protocol_is_runtime_checkable(self, protocol: type) -> None:
        """Isinstance checks should not raise TypeError."""
        # If the protocol were not runtime_checkable, isinstance()
        # would raise TypeError.
        result = isinstance(EmptyClass(), protocol)
        assert result is False


# -------------------------------------------------------------------
# Tests: SelectionStrategy
# -------------------------------------------------------------------


class TestSelectionStrategy:
    """Tests for the SelectionStrategy protocol."""

    def test_conforming_class_satisfies(self) -> None:
        obj = ConformingSelectionStrategy()
        assert isinstance(obj, SelectionStrategy)

    def test_non_conforming_class_does_not_satisfy(self) -> None:
        obj = NonConformingSelectionStrategy()
        assert not isinstance(obj, SelectionStrategy)

    def test_empty_class_does_not_satisfy(self) -> None:
        assert not isinstance(EmptyClass(), SelectionStrategy)

    def test_conforming_select_child_callable(self) -> None:
        obj = ConformingSelectionStrategy()
        result = obj.select_child("mock_node")
        assert result == "mock_node"


# -------------------------------------------------------------------
# Tests: EvaluationStrategy
# -------------------------------------------------------------------


class TestEvaluationStrategy:
    """Tests for the EvaluationStrategy protocol."""

    def test_conforming_class_satisfies(self) -> None:
        obj = ConformingEvaluationStrategy()
        assert isinstance(obj, EvaluationStrategy)

    def test_non_conforming_class_does_not_satisfy(self) -> None:
        obj = NonConformingEvaluationStrategy()
        assert not isinstance(obj, EvaluationStrategy)

    def test_empty_class_does_not_satisfy(self) -> None:
        assert not isinstance(EmptyClass(), EvaluationStrategy)

    def test_conforming_evaluate_returns_float(self) -> None:
        obj = ConformingEvaluationStrategy()
        result = obj.evaluate("mock_state")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# -------------------------------------------------------------------
# Tests: BackupStrategyProtocol
# -------------------------------------------------------------------


class TestBackupStrategyProtocol:
    """Tests for the BackupStrategyProtocol protocol."""

    def test_conforming_class_satisfies(self) -> None:
        obj = ConformingBackupStrategy()
        assert isinstance(obj, BackupStrategyProtocol)

    def test_non_conforming_class_does_not_satisfy(self) -> None:
        obj = NonConformingBackupStrategy()
        assert not isinstance(obj, BackupStrategyProtocol)

    def test_empty_class_does_not_satisfy(self) -> None:
        assert not isinstance(EmptyClass(), BackupStrategyProtocol)

    def test_conforming_backup_returns_none(self) -> None:
        obj = ConformingBackupStrategy()
        result = obj.backup(0.5, ["root", "child", "leaf"])
        assert result is None


# -------------------------------------------------------------------
# Tests: ActionValidator
# -------------------------------------------------------------------


class TestActionValidator:
    """Tests for the ActionValidator protocol."""

    def test_conforming_class_satisfies(self) -> None:
        obj = ConformingActionValidator()
        assert isinstance(obj, ActionValidator)

    def test_non_conforming_class_does_not_satisfy(self) -> None:
        obj = NonConformingActionValidator()
        assert not isinstance(obj, ActionValidator)

    def test_empty_class_does_not_satisfy(self) -> None:
        assert not isinstance(EmptyClass(), ActionValidator)

    def test_conforming_validate_returns_bool(self) -> None:
        obj = ConformingActionValidator()
        result = obj.validate("mock_action", "mock_state")
        assert result is True


# -------------------------------------------------------------------
# Tests: PhysicsModule
# -------------------------------------------------------------------


class TestPhysicsModule:
    """Tests for the PhysicsModule protocol."""

    def test_conforming_class_satisfies(self) -> None:
        obj = ConformingPhysicsModule()
        assert isinstance(obj, PhysicsModule)

    def test_missing_method_does_not_satisfy(self) -> None:
        obj = NonConformingPhysicsModuleMissingMethod()
        assert not isinstance(obj, PhysicsModule)

    def test_missing_property_does_not_satisfy(self) -> None:
        obj = NonConformingPhysicsModuleMissingProperty()
        assert not isinstance(obj, PhysicsModule)

    def test_empty_class_does_not_satisfy(self) -> None:
        assert not isinstance(EmptyClass(), PhysicsModule)

    def test_conforming_name_property(self) -> None:
        obj = ConformingPhysicsModule()
        assert obj.name == "test_pde"

    def test_conforming_pde_type_property(self) -> None:
        obj = ConformingPhysicsModule()
        assert obj.pde_type == PDEType.ELLIPTIC

    def test_conforming_methods_callable(self) -> None:
        obj = ConformingPhysicsModule()
        assert obj.boundary_conditions() == []
        assert obj.manufactured_solution() is None
        assert obj.reward_function(None, None, None, None) == {"accuracy": 1.0}
        assert "node_features" in obj.state_features(None)
        assert obj.action_validators() == []
        assert obj.default_config() == {}


# -------------------------------------------------------------------
# Tests: Solver
# -------------------------------------------------------------------


class TestSolver:
    """Tests for the Solver protocol."""

    def test_conforming_class_satisfies(self) -> None:
        obj = ConformingSolver()
        assert isinstance(obj, Solver)

    def test_non_conforming_class_does_not_satisfy(self) -> None:
        obj = NonConformingSolver()
        assert not isinstance(obj, Solver)

    def test_empty_class_does_not_satisfy(self) -> None:
        assert not isinstance(EmptyClass(), Solver)

    def test_conforming_solve_callable(self) -> None:
        obj = ConformingSolver()
        stiffness = np.eye(3)
        rhs = np.ones(3)
        result = obj.solve(stiffness, rhs)
        assert "solution" in result
        assert np.array_equal(result["solution"], rhs)


# -------------------------------------------------------------------
# Tests: FeatureExtractor
# -------------------------------------------------------------------


class TestFeatureExtractor:
    """Tests for the FeatureExtractor protocol."""

    def test_conforming_class_satisfies(self) -> None:
        obj = ConformingFeatureExtractor()
        assert isinstance(obj, FeatureExtractor)

    def test_missing_method_does_not_satisfy(self) -> None:
        obj = NonConformingFeatureExtractorMissingMethod()
        assert not isinstance(obj, FeatureExtractor)

    def test_missing_property_does_not_satisfy(self) -> None:
        obj = NonConformingFeatureExtractorMissingProperty()
        assert not isinstance(obj, FeatureExtractor)

    def test_empty_class_does_not_satisfy(self) -> None:
        assert not isinstance(EmptyClass(), FeatureExtractor)

    def test_conforming_properties(self) -> None:
        obj = ConformingFeatureExtractor()
        assert obj.node_feature_dim == 32
        assert obj.edge_feature_dim == 8
        assert obj.global_feature_dim == 16

    def test_conforming_extract_returns_dict(self) -> None:
        obj = ConformingFeatureExtractor()
        result = obj.extract("mock_state")
        assert "node_features" in result
        assert result["node_features"].shape == (4, 32)


# -------------------------------------------------------------------
# Tests: Cross-protocol non-overlap
# -------------------------------------------------------------------


class TestCrossProtocolNonOverlap:
    """Cross-protocol non-overlap verification.

    A class satisfying one protocol should not automatically
    satisfy unrelated protocols.
    """

    def test_selection_not_evaluation(self) -> None:
        obj = ConformingSelectionStrategy()
        assert isinstance(obj, SelectionStrategy)
        assert not isinstance(obj, EvaluationStrategy)

    def test_evaluation_not_action_validator(self) -> None:
        obj = ConformingEvaluationStrategy()
        assert isinstance(obj, EvaluationStrategy)
        assert not isinstance(obj, ActionValidator)

    def test_solver_not_feature_extractor(self) -> None:
        obj = ConformingSolver()
        assert isinstance(obj, Solver)
        assert not isinstance(obj, FeatureExtractor)

    def test_backup_not_selection(self) -> None:
        obj = ConformingBackupStrategy()
        assert isinstance(obj, BackupStrategyProtocol)
        assert not isinstance(obj, SelectionStrategy)


# -------------------------------------------------------------------
# Tests: Multi-protocol conformance
# -------------------------------------------------------------------


class MultiProtocolClass:
    """A class implementing both SelectionStrategy and EvaluationStrategy."""

    def select_child(self, node: Any) -> Any:
        return node

    def evaluate(self, state: Any) -> float:
        return 0.9


class TestMultiProtocol:
    """A single class can satisfy multiple protocols."""

    def test_satisfies_both(self) -> None:
        obj = MultiProtocolClass()
        assert isinstance(obj, SelectionStrategy)
        assert isinstance(obj, EvaluationStrategy)


# -------------------------------------------------------------------
# Tests: Lambda / function objects should not satisfy protocols
# -------------------------------------------------------------------


class TestNonClassObjects:
    """Verify that arbitrary objects and plain dicts do not satisfy protocols."""

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
    def test_dict_does_not_satisfy(self, protocol: type) -> None:
        assert not isinstance({}, protocol)

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
    def test_none_does_not_satisfy(self, protocol: type) -> None:
        assert not isinstance(None, protocol)

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
    def test_string_does_not_satisfy(self, protocol: type) -> None:
        assert not isinstance("hello", protocol)
