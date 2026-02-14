"""Tests for the Symbolic Equation Discovery framework."""
from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.planning.symbolic_discovery import (
    ExpressionNode,
    SymbolicActionType,
    SymbolicDiscovery,
    SymbolicState,
    SymbolType,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def x_data() -> dict[str, np.ndarray]:
    """Input data with a single variable x."""
    return {"x": np.linspace(0.1, 2.0, 50)}


@pytest.fixture()
def discovery() -> SymbolicDiscovery:
    """A symbolic discovery engine for one variable."""
    return SymbolicDiscovery(
        variables=["x"],
        max_complexity=20,
        constants=[0.0, 1.0, 2.0, -1.0, 0.5],
        binary_ops=["+", "-", "*", "/"],
        unary_ops=["sin", "cos", "exp", "sqrt"],
    )


# ------------------------------------------------------------------
# ExpressionNode tests
# ------------------------------------------------------------------

class TestExpressionEvaluateVariable:
    """Variable nodes return the corresponding input array."""

    def test_expression_evaluate_variable(
        self, x_data: dict[str, np.ndarray],
    ) -> None:
        node = ExpressionNode(
            symbol_type=SymbolType.VARIABLE,
            value="x",
        )
        result = node.evaluate(x_data)
        np.testing.assert_array_equal(result, x_data["x"])


class TestExpressionEvaluateConstant:
    """Constant nodes return a filled array."""

    def test_expression_evaluate_constant(
        self, x_data: dict[str, np.ndarray],
    ) -> None:
        node = ExpressionNode(
            symbol_type=SymbolType.CONSTANT,
            value="3.14",
            numeric_value=3.14,
        )
        result = node.evaluate(x_data)
        expected = np.full_like(x_data["x"], 3.14)
        np.testing.assert_allclose(result, expected)

    def test_constant_none_defaults_to_one(
        self, x_data: dict[str, np.ndarray],
    ) -> None:
        node = ExpressionNode(
            symbol_type=SymbolType.CONSTANT,
            value="1.0",
            numeric_value=None,
        )
        result = node.evaluate(x_data)
        expected = np.full_like(x_data["x"], 1.0)
        np.testing.assert_allclose(result, expected)


class TestExpressionEvaluateBinaryOp:
    """Binary operator nodes combine two children."""

    def test_expression_evaluate_binary_op(
        self, x_data: dict[str, np.ndarray],
    ) -> None:
        # Build: x + 2
        left = ExpressionNode(
            symbol_type=SymbolType.VARIABLE, value="x",
        )
        right = ExpressionNode(
            symbol_type=SymbolType.CONSTANT, value="2",
            numeric_value=2.0,
        )
        node = ExpressionNode(
            symbol_type=SymbolType.BINARY_OP, value="+",
            children=[left, right],
        )
        result = node.evaluate(x_data)
        expected = x_data["x"] + 2.0
        np.testing.assert_allclose(result, expected)

    def test_multiplication(
        self, x_data: dict[str, np.ndarray],
    ) -> None:
        # Build: x * x
        left = ExpressionNode(
            symbol_type=SymbolType.VARIABLE, value="x",
        )
        right = ExpressionNode(
            symbol_type=SymbolType.VARIABLE, value="x",
        )
        node = ExpressionNode(
            symbol_type=SymbolType.BINARY_OP, value="*",
            children=[left, right],
        )
        result = node.evaluate(x_data)
        expected = x_data["x"] ** 2
        np.testing.assert_allclose(result, expected)

    def test_safe_division(
        self, x_data: dict[str, np.ndarray],
    ) -> None:
        # Build: x / 0 -- should not produce Inf/NaN
        left = ExpressionNode(
            symbol_type=SymbolType.VARIABLE, value="x",
        )
        right = ExpressionNode(
            symbol_type=SymbolType.CONSTANT, value="0",
            numeric_value=0.0,
        )
        node = ExpressionNode(
            symbol_type=SymbolType.BINARY_OP, value="/",
            children=[left, right],
        )
        result = node.evaluate(x_data)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestExpressionEvaluateUnaryOp:
    """Unary operator nodes transform a single child."""

    def test_expression_evaluate_unary_op(
        self, x_data: dict[str, np.ndarray],
    ) -> None:
        # Build: sin(x)
        child = ExpressionNode(
            symbol_type=SymbolType.VARIABLE, value="x",
        )
        node = ExpressionNode(
            symbol_type=SymbolType.UNARY_OP, value="sin",
            children=[child],
        )
        result = node.evaluate(x_data)
        expected = np.sin(x_data["x"])
        np.testing.assert_allclose(result, expected)

    def test_exp_clipping(self) -> None:
        # exp should clip large values to avoid overflow
        data = {"x": np.array([100.0, -100.0, 0.0])}
        child = ExpressionNode(
            symbol_type=SymbolType.VARIABLE, value="x",
        )
        node = ExpressionNode(
            symbol_type=SymbolType.UNARY_OP, value="exp",
            children=[child],
        )
        result = node.evaluate(data)
        assert not np.any(np.isinf(result))


class TestExpressionToString:
    """ExpressionNode.to_string produces readable output."""

    def test_expression_to_string(self) -> None:
        # Build: sin((x + 2))
        x_node = ExpressionNode(
            symbol_type=SymbolType.VARIABLE, value="x",
        )
        two_node = ExpressionNode(
            symbol_type=SymbolType.CONSTANT, value="2",
            numeric_value=2.0,
        )
        add_node = ExpressionNode(
            symbol_type=SymbolType.BINARY_OP, value="+",
            children=[x_node, two_node],
        )
        sin_node = ExpressionNode(
            symbol_type=SymbolType.UNARY_OP, value="sin",
            children=[add_node],
        )
        s = sin_node.to_string()
        assert "sin" in s
        assert "x" in s
        assert "2" in s

    def test_variable_to_string(self) -> None:
        node = ExpressionNode(
            symbol_type=SymbolType.VARIABLE, value="x",
        )
        assert node.to_string() == "x"

    def test_constant_to_string(self) -> None:
        node = ExpressionNode(
            symbol_type=SymbolType.CONSTANT, value="3.14",
            numeric_value=3.14,
        )
        s = node.to_string()
        assert "3.14" in s

    def test_integer_constant_to_string(self) -> None:
        node = ExpressionNode(
            symbol_type=SymbolType.CONSTANT, value="2",
            numeric_value=2.0,
        )
        assert node.to_string() == "2"


class TestExpressionComplexity:
    """ExpressionNode.complexity counts all nodes."""

    def test_expression_complexity(self) -> None:
        # Leaf: complexity 1
        leaf = ExpressionNode(
            symbol_type=SymbolType.VARIABLE, value="x",
        )
        assert leaf.complexity() == 1

        # Unary: 1 + child
        unary = ExpressionNode(
            symbol_type=SymbolType.UNARY_OP, value="sin",
            children=[leaf.clone()],
        )
        assert unary.complexity() == 2

        # Binary: 1 + left + right
        right = ExpressionNode(
            symbol_type=SymbolType.CONSTANT, value="1",
            numeric_value=1.0,
        )
        binary = ExpressionNode(
            symbol_type=SymbolType.BINARY_OP, value="+",
            children=[unary.clone(), right.clone()],
        )
        assert binary.complexity() == 4  # + -> sin -> x, 1

    def test_deep_tree_complexity(self) -> None:
        # Build: sin(cos(x))
        x = ExpressionNode(
            symbol_type=SymbolType.VARIABLE, value="x",
        )
        cos_x = ExpressionNode(
            symbol_type=SymbolType.UNARY_OP, value="cos",
            children=[x],
        )
        sin_cos_x = ExpressionNode(
            symbol_type=SymbolType.UNARY_OP, value="sin",
            children=[cos_x],
        )
        assert sin_cos_x.complexity() == 3


# ------------------------------------------------------------------
# SymbolicDiscovery tests
# ------------------------------------------------------------------

class TestSymbolicValidActions:
    """SymbolicDiscovery.get_valid_actions returns correct actions."""

    def test_symbolic_valid_actions(
        self, discovery: SymbolicDiscovery,
    ) -> None:
        # Empty state: only leaf actions (variables + constants)
        empty_state = SymbolicState(max_complexity=20)
        actions = discovery.get_valid_actions(empty_state)

        action_types = {a.action_type for a in actions}
        assert SymbolicActionType.ADD_VARIABLE in action_types
        assert SymbolicActionType.ADD_CONSTANT in action_types
        # No binary/unary ops without an expression
        assert SymbolicActionType.ADD_BINARY_OP not in action_types
        assert SymbolicActionType.ADD_UNARY_OP not in action_types

    def test_valid_actions_with_expression(
        self, discovery: SymbolicDiscovery,
    ) -> None:
        state = SymbolicState(
            expression=ExpressionNode(
                symbol_type=SymbolType.VARIABLE, value="x",
            ),
            max_complexity=20,
        )
        actions = discovery.get_valid_actions(state)
        action_types = {a.action_type for a in actions}

        assert SymbolicActionType.ADD_BINARY_OP in action_types
        assert SymbolicActionType.ADD_UNARY_OP in action_types
        assert SymbolicActionType.NO_OP in action_types

    def test_no_actions_at_max_complexity(
        self, discovery: SymbolicDiscovery,
    ) -> None:
        # Expression already at max complexity => only NO_OP
        state = SymbolicState(
            expression=ExpressionNode(
                symbol_type=SymbolType.VARIABLE, value="x",
            ),
            max_complexity=1,  # Already at limit
        )
        actions = discovery.get_valid_actions(state)
        action_types = {a.action_type for a in actions}

        # Binary adds 2 nodes, unary adds 1 -- neither fits
        assert SymbolicActionType.ADD_BINARY_OP not in action_types
        assert SymbolicActionType.ADD_UNARY_OP not in action_types
        assert SymbolicActionType.NO_OP in action_types


class TestSymbolicDiscoverSimple:
    """discover() can find simple expressions from data."""

    def test_symbolic_discover_simple(self) -> None:
        """Discover y = 2*x from data."""
        x = np.linspace(0.1, 5.0, 100)
        target = 2.0 * x

        engine = SymbolicDiscovery(
            variables=["x"],
            max_complexity=10,
            constants=[0.0, 1.0, 2.0, -1.0, 0.5, 3.0],
            binary_ops=["+", "-", "*"],
            unary_ops=[],
            exploration_weight=0.5,
        )

        best = engine.discover(
            input_data={"x": x},
            target_data=target,
            num_iterations=200,
        )

        # The discovered expression must fit the data well
        predicted = best.evaluate({"x": x})
        mse = float(np.mean((predicted - target) ** 2))

        # We expect a reasonable fit -- MSE < 1.0 for y=2x
        assert mse < 1.0, (
            f"MSE {mse:.4f} too high for y=2*x; "
            f"best expression: {best.to_string()}"
        )

    def test_discover_constant(self) -> None:
        """Discover y = 1 from constant data."""
        x = np.linspace(0.1, 5.0, 50)
        target = np.ones_like(x)

        engine = SymbolicDiscovery(
            variables=["x"],
            max_complexity=5,
            constants=[0.0, 1.0, 2.0],
            binary_ops=["+", "-", "*"],
            unary_ops=[],
        )

        best = engine.discover(
            input_data={"x": x},
            target_data=target,
            num_iterations=50,
        )

        predicted = best.evaluate({"x": x})
        mse = float(np.mean((predicted - target) ** 2))
        assert mse < 0.01, f"MSE {mse:.6f} for constant function"
