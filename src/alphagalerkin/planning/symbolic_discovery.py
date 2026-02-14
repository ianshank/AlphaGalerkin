"""Symbolic equation discovery via MCTS over expression trees.

This module searches the combinatorial space of mathematical
expressions to find symbolic equations that fit observed data.
It uses a UCB-based exploration strategy inspired by Monte Carlo
Tree Search, building expression trees incrementally and scoring
them against target data.

The search operates over ``ExpressionNode`` trees composed of
variables, constants, binary operators (+, -, *, /), and unary
operators (sin, cos, exp, log, sqrt, neg, abs).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

from src.alphagalerkin.core.constants import DEFAULT_SEED

logger = structlog.get_logger("planning.symbolic")


# ======================================================================
# Expression tree nodes
# ======================================================================


class SymbolType(str, Enum):
    """Types of symbols in expression trees."""

    VARIABLE = "variable"
    """Named variable (e.g. x, y, t)."""

    CONSTANT = "constant"
    """Numeric constant."""

    BINARY_OP = "binary_op"
    """Binary operator (+, -, *, /)."""

    UNARY_OP = "unary_op"
    """Unary operator (sin, cos, exp, log, sqrt, neg, abs)."""


# Safe operation maps -- clipping prevents NaN / Inf blow-up.
_UNARY_OPS: dict[str, Callable[..., Any]] = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": lambda x: np.exp(np.clip(x, -10, 10)),
    "log": lambda x: np.log(np.abs(x) + 1e-10),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "neg": np.negative,
    "abs": np.abs,
}

_BINARY_OPS: dict[str, Callable[..., Any]] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": lambda a, b: np.divide(
        a,
        np.where(np.abs(b) < 1e-10, 1e-10, b),
    ),
}


@dataclass
class ExpressionNode:
    """A node in a symbolic expression tree.

    Attributes:
        symbol_type: Whether this is a variable, constant, or operator.
        value: The operator name, variable name, or string repr.
        children: Child nodes (0 for leaves, 1 for unary, 2 for binary).
        numeric_value: Numeric value for constants.

    """

    symbol_type: SymbolType
    value: str
    children: list[ExpressionNode] = field(default_factory=list)
    numeric_value: float | None = None

    def evaluate(
        self,
        variables: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Evaluate this expression tree on given variable values.

        Parameters
        ----------
        variables:
            Mapping of variable names to arrays of values.
            All arrays must broadcast to the same shape.

        Returns
        -------
        np.ndarray
            The computed output array.

        Raises
        ------
        ValueError
            If the symbol type is unknown.
        KeyError
            If a variable name is not found in *variables*.

        """
        if self.symbol_type == SymbolType.VARIABLE:
            return variables[self.value]

        if self.symbol_type == SymbolType.CONSTANT:
            val = self.numeric_value if self.numeric_value is not None else 1.0
            # Use the shape of any available variable
            ref = next(iter(variables.values()))
            return np.full_like(ref, val, dtype=float)

        if self.symbol_type == SymbolType.UNARY_OP:
            child_val = self.children[0].evaluate(variables)
            op = _UNARY_OPS.get(self.value)
            if op is None:
                raise ValueError(f"Unknown unary op: {self.value}")
            return op(child_val)

        if self.symbol_type == SymbolType.BINARY_OP:
            left = self.children[0].evaluate(variables)
            right = self.children[1].evaluate(variables)
            op = _BINARY_OPS.get(self.value)
            if op is None:
                raise ValueError(f"Unknown binary op: {self.value}")
            return op(left, right)

        raise ValueError(f"Unknown symbol type: {self.symbol_type}")

    def to_string(self) -> str:
        """Convert to human-readable infix string.

        Returns
        -------
        str
            A parenthesised string representation of the expression.

        """
        if self.symbol_type == SymbolType.VARIABLE:
            return self.value

        if self.symbol_type == SymbolType.CONSTANT:
            val = self.numeric_value if self.numeric_value is not None else 1.0
            # Display integers without decimals
            if val == int(val):
                return str(int(val))
            return f"{val:.4g}"

        if self.symbol_type == SymbolType.UNARY_OP:
            child_str = self.children[0].to_string()
            return f"{self.value}({child_str})"

        if self.symbol_type == SymbolType.BINARY_OP:
            left_str = self.children[0].to_string()
            right_str = self.children[1].to_string()
            return f"({left_str} {self.value} {right_str})"

        return f"?{self.value}?"

    def complexity(self) -> int:
        """Count the total number of nodes in this sub-tree.

        Returns
        -------
        int
            Node count (1 for a leaf, recursively accumulated).

        """
        return 1 + sum(c.complexity() for c in self.children)

    def clone(self) -> ExpressionNode:
        """Return a deep, independent copy of this node and children."""
        return ExpressionNode(
            symbol_type=self.symbol_type,
            value=self.value,
            children=[c.clone() for c in self.children],
            numeric_value=self.numeric_value,
        )


# ======================================================================
# Symbolic search state and actions
# ======================================================================


class SymbolicActionType(str, Enum):
    """Actions for building expression trees."""

    ADD_VARIABLE = "add_variable"
    """Add a variable node (leaf)."""

    ADD_CONSTANT = "add_constant"
    """Add a constant node (leaf)."""

    ADD_BINARY_OP = "add_binary_op"
    """Wrap the current expression in a binary operator with a new leaf."""

    ADD_UNARY_OP = "add_unary_op"
    """Wrap the current expression in a unary operator."""

    SIMPLIFY = "simplify"
    """Attempt to simplify the expression (placeholder)."""

    NO_OP = "no_op"
    """Do nothing."""


@dataclass
class SymbolicState:
    """State of symbolic equation discovery.

    Attributes:
        expression: The current candidate expression (None at start).
        target_data: Shape (N,) array of target output values.
        input_data: Mapping of variable names to shape (N,) arrays.
        best_fitness: Best MSE achieved so far.
        step: Number of actions taken.
        max_complexity: Maximum allowed tree node count.

    """

    expression: ExpressionNode | None = None
    target_data: np.ndarray | None = None
    input_data: dict[str, np.ndarray] | None = None
    best_fitness: float = float("inf")
    step: int = 0
    max_complexity: int = 20

    def clone(self) -> SymbolicState:
        """Return a deep, independent copy of this state."""
        return SymbolicState(
            expression=(self.expression.clone() if self.expression is not None else None),
            target_data=(self.target_data.copy() if self.target_data is not None else None),
            input_data=(
                {k: v.copy() for k, v in self.input_data.items()}
                if self.input_data is not None
                else None
            ),
            best_fitness=self.best_fitness,
            step=self.step,
            max_complexity=self.max_complexity,
        )


@dataclass
class SymbolicAction:
    """An action in symbolic discovery.

    Attributes:
        action_type: The kind of tree modification.
        value: Operator or variable name.
        numeric_value: Numeric value for ADD_CONSTANT actions.

    """

    action_type: SymbolicActionType
    value: str = ""
    numeric_value: float | None = None


# ======================================================================
# Symbolic discovery engine
# ======================================================================


class SymbolicDiscovery:
    """Discovers symbolic equations from data using MCTS-style search.

    Searches over expression trees to find equations that fit
    observed data.  Uses a UCB-based exploration of the combinatorial
    space of mathematical expressions.

    Parameters
    ----------
    variables:
        Names of input variables (e.g. ``["x", "y"]``).
    max_complexity:
        Maximum allowed tree node count.
    constants:
        Pool of numeric constants to try.
    binary_ops:
        Allowed binary operators.
    unary_ops:
        Allowed unary operators.
    exploration_weight:
        UCB exploration parameter (higher = more exploration).

    """

    def __init__(
        self,
        variables: list[str],
        max_complexity: int = 20,
        constants: list[float] | None = None,
        binary_ops: list[str] | None = None,
        unary_ops: list[str] | None = None,
        exploration_weight: float = 1.0,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._variables = variables
        self._max_complexity = max_complexity
        self._constants = constants or [0.0, 1.0, 2.0, -1.0, 0.5]
        self._binary_ops = binary_ops or ["+", "-", "*", "/"]
        self._unary_ops = unary_ops or ["sin", "cos", "exp", "sqrt"]
        self._exploration_weight = exploration_weight
        self._rng = np.random.default_rng(seed)

        # Action statistics for UCB
        self._action_visits: dict[str, int] = {}
        self._action_rewards: dict[str, float] = {}

        logger.info(
            "symbolic_discovery.init",
            variables=variables,
            max_complexity=max_complexity,
            n_constants=len(self._constants),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(
        self,
        input_data: dict[str, np.ndarray],
        target_data: np.ndarray,
        num_iterations: int = 100,
    ) -> ExpressionNode:
        """Run symbolic discovery and return the best expression.

        Parameters
        ----------
        input_data:
            Mapping of variable names to input arrays.
        target_data:
            Target output array.
        num_iterations:
            Number of search iterations.

        Returns
        -------
        ExpressionNode
            The expression with the lowest MSE to target data.

        """
        best_expr: ExpressionNode | None = None
        best_fitness = float("inf")

        for iteration in range(num_iterations):
            state = SymbolicState(
                target_data=target_data,
                input_data=input_data,
                max_complexity=self._max_complexity,
            )

            # Build an expression by taking a sequence of actions
            max_actions = self._max_complexity
            for _ in range(max_actions):
                valid = self.get_valid_actions(state)
                if not valid:
                    break

                # UCB-based action selection
                action = self._select_action(valid, iteration + 1)
                state = self.apply_action(state, action)

                if state.expression is not None:
                    fitness = self._compute_fitness(
                        state.expression,
                        input_data,
                        target_data,
                    )
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_expr = state.expression.clone()

                    # Update UCB statistics
                    key = self._action_key(action)
                    self._action_visits[key] = self._action_visits.get(key, 0) + 1
                    # Reward = inverse fitness (lower MSE = higher reward)
                    reward = 1.0 / (1.0 + fitness)
                    prev = self._action_rewards.get(key, 0.0)
                    n = self._action_visits[key]
                    self._action_rewards[key] = prev + (reward - prev) / n

            if iteration % max(1, num_iterations // 5) == 0:
                logger.debug(
                    "symbolic_discovery.progress",
                    iteration=iteration,
                    best_fitness=best_fitness,
                    best_expr=(best_expr.to_string() if best_expr else None),
                )

        if best_expr is None:
            # Fallback: return a constant
            best_expr = ExpressionNode(
                symbol_type=SymbolType.CONSTANT,
                value="1.0",
                numeric_value=1.0,
            )

        logger.info(
            "symbolic_discovery.complete",
            best_fitness=best_fitness,
            expression=best_expr.to_string(),
            complexity=best_expr.complexity(),
        )
        return best_expr

    def get_valid_actions(
        self,
        state: SymbolicState,
    ) -> list[SymbolicAction]:
        """Return valid tree-building actions for the current state.

        When the expression is None, only leaf actions (variables and
        constants) are valid.  Otherwise, the expression can be
        extended with binary or unary operators, subject to the
        complexity limit.
        """
        actions: list[SymbolicAction] = []
        current_complexity = state.expression.complexity() if state.expression else 0

        if state.expression is None:
            # Must start with a leaf node
            for var in self._variables:
                actions.append(
                    SymbolicAction(
                        action_type=SymbolicActionType.ADD_VARIABLE,
                        value=var,
                    )
                )
            for const in self._constants:
                actions.append(
                    SymbolicAction(
                        action_type=SymbolicActionType.ADD_CONSTANT,
                        value=str(const),
                        numeric_value=const,
                    )
                )
        else:
            # Binary ops add 2 nodes (the op + a new leaf)
            if current_complexity + 2 <= state.max_complexity:
                for op in self._binary_ops:
                    for var in self._variables:
                        actions.append(
                            SymbolicAction(
                                action_type=SymbolicActionType.ADD_BINARY_OP,
                                value=f"{op}:{var}",
                            )
                        )
                    for const in self._constants:
                        actions.append(
                            SymbolicAction(
                                action_type=SymbolicActionType.ADD_BINARY_OP,
                                value=f"{op}:{const}",
                                numeric_value=const,
                            )
                        )

            # Unary ops add 1 node
            if current_complexity + 1 <= state.max_complexity:
                for op in self._unary_ops:
                    actions.append(
                        SymbolicAction(
                            action_type=SymbolicActionType.ADD_UNARY_OP,
                            value=op,
                        )
                    )

            # No-op to terminate building
            actions.append(SymbolicAction(action_type=SymbolicActionType.NO_OP))

        return actions

    def apply_action(
        self,
        state: SymbolicState,
        action: SymbolicAction,
    ) -> SymbolicState:
        """Apply an action and return the new state.

        Parameters
        ----------
        state:
            Current symbolic search state.
        action:
            The tree-building action to apply.

        Returns
        -------
        SymbolicState
            Updated state with the modified expression.

        """
        new_state = state.clone()
        new_state.step = state.step + 1

        if action.action_type == SymbolicActionType.ADD_VARIABLE:
            new_state.expression = ExpressionNode(
                symbol_type=SymbolType.VARIABLE,
                value=action.value,
            )

        elif action.action_type == SymbolicActionType.ADD_CONSTANT:
            new_state.expression = ExpressionNode(
                symbol_type=SymbolType.CONSTANT,
                value=action.value,
                numeric_value=action.numeric_value,
            )

        elif action.action_type == SymbolicActionType.ADD_BINARY_OP:
            if new_state.expression is not None:
                parts = action.value.split(":", 1)
                op_name = parts[0]
                operand_str = parts[1] if len(parts) > 1 else "1.0"

                # Create the right-hand operand
                if action.numeric_value is not None:
                    right = ExpressionNode(
                        symbol_type=SymbolType.CONSTANT,
                        value=operand_str,
                        numeric_value=action.numeric_value,
                    )
                elif operand_str in (self._variables or []):
                    right = ExpressionNode(
                        symbol_type=SymbolType.VARIABLE,
                        value=operand_str,
                    )
                else:
                    # Try to parse as constant
                    try:
                        num = float(operand_str)
                        right = ExpressionNode(
                            symbol_type=SymbolType.CONSTANT,
                            value=operand_str,
                            numeric_value=num,
                        )
                    except ValueError:
                        right = ExpressionNode(
                            symbol_type=SymbolType.VARIABLE,
                            value=operand_str,
                        )

                new_state.expression = ExpressionNode(
                    symbol_type=SymbolType.BINARY_OP,
                    value=op_name,
                    children=[new_state.expression, right],
                )

        elif action.action_type == SymbolicActionType.ADD_UNARY_OP:
            if new_state.expression is not None:
                new_state.expression = ExpressionNode(
                    symbol_type=SymbolType.UNARY_OP,
                    value=action.value,
                    children=[new_state.expression],
                )

        # SIMPLIFY and NO_OP: leave expression unchanged

        # Update fitness if we have a complete expression and data
        if (
            new_state.expression is not None
            and new_state.input_data is not None
            and new_state.target_data is not None
        ):
            fitness = self._compute_fitness(
                new_state.expression,
                new_state.input_data,
                new_state.target_data,
            )
            if fitness < new_state.best_fitness:
                new_state.best_fitness = fitness

        return new_state

    def _compute_fitness(
        self,
        expression: ExpressionNode,
        input_data: dict[str, np.ndarray],
        target_data: np.ndarray,
    ) -> float:
        """Compute MSE between expression output and target.

        Returns ``float('inf')`` if evaluation fails (e.g. due to
        NaN or overflow).
        """
        try:
            predicted = expression.evaluate(input_data)
            if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
                return float("inf")
            mse = float(np.mean((predicted - target_data) ** 2))
            return mse
        except Exception:
            logger.debug("symbolic.evaluation_failed", exc_info=True)
            return float("inf")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_action(
        self,
        actions: list[SymbolicAction],
        total_visits: int,
    ) -> SymbolicAction:
        """Select an action using UCB1.

        Balances exploitation (pick high-reward actions) with
        exploration (try under-visited actions).
        """
        best_action = actions[0]
        best_ucb = -float("inf")

        for action in actions:
            key = self._action_key(action)
            n = self._action_visits.get(key, 0)
            if n == 0:
                # Unvisited actions get priority
                return action

            avg_reward = self._action_rewards.get(key, 0.0)
            ucb = avg_reward + self._exploration_weight * math.sqrt(math.log(total_visits) / n)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return best_action

    @staticmethod
    def _action_key(action: SymbolicAction) -> str:
        """Return a hashable key for an action (for UCB statistics)."""
        return f"{action.action_type.value}:{action.value}"
