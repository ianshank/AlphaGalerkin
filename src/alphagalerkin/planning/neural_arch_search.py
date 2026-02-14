"""Neural operator architecture search via MCTS.

This module applies MCTS-style search to the combinatorial space of
neural operator architectures. Current Fourier Neural Operators,
DeepONets, and Wavelet Neural Operators require extensive manual
tuning of spectral modes, layer depth, and architecture parameters.

MCTS searches operator decomposition strategies for multi-physics
problems, deciding how to compose neural operators for coupled
subproblems with learned coupling.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger("planning.nas")


# ======================================================================
# Operator building blocks
# ======================================================================


class OperatorBlockType(str, Enum):
    """Types of neural operator building blocks."""

    FOURIER_LAYER = "fourier_layer"
    """Fourier Neural Operator layer."""

    DEEPONET_BRANCH = "deeponet_branch"
    """DeepONet branch network."""

    DEEPONET_TRUNK = "deeponet_trunk"
    """DeepONet trunk network."""

    WAVELET_LAYER = "wavelet_layer"
    """Wavelet-based layer."""

    GALERKIN_ATTENTION = "galerkin_attention"
    """Galerkin attention (O(N))."""

    FNET_MIXING = "fnet_mixing"
    """FFT mixing (O(N log N))."""

    MLP_LAYER = "mlp_layer"
    """Standard MLP layer."""

    RESIDUAL_BLOCK = "residual_block"
    """Residual connection wrapper."""

    SKIP_CONNECTION = "skip_connection"
    """Skip/shortcut connection."""


class NASActionType(str, Enum):
    """Actions for neural operator architecture search."""

    ADD_LAYER = "add_layer"
    """Add a new layer at a specified position."""

    REMOVE_LAYER = "remove_layer"
    """Remove a layer from the architecture."""

    CHANGE_LAYER_TYPE = "change_layer_type"
    """Change the operator type of a layer."""

    ADJUST_WIDTH = "adjust_width"
    """Increase or decrease a layer's width."""

    ADJUST_MODES = "adjust_modes"
    """Adjust the number of Fourier modes for a layer."""

    ADD_SKIP_CONNECTION = "add_skip_connection"
    """Add a skip connection from one layer to another."""

    TOGGLE_RESIDUAL = "toggle_residual"
    """Toggle residual connection on a layer."""

    NO_OP = "no_op"
    """Do nothing -- always a valid action."""


# ======================================================================
# Layer and architecture specifications
# ======================================================================

# Base parameter cost estimates per block type (weight x width^2)
_BLOCK_PARAM_MULTIPLIERS: dict[OperatorBlockType, float] = {
    OperatorBlockType.FOURIER_LAYER: 2.0,
    OperatorBlockType.DEEPONET_BRANCH: 1.5,
    OperatorBlockType.DEEPONET_TRUNK: 1.5,
    OperatorBlockType.WAVELET_LAYER: 1.8,
    OperatorBlockType.GALERKIN_ATTENTION: 3.0,
    OperatorBlockType.FNET_MIXING: 0.5,
    OperatorBlockType.MLP_LAYER: 1.0,
    OperatorBlockType.RESIDUAL_BLOCK: 1.2,
    OperatorBlockType.SKIP_CONNECTION: 0.1,
}


@dataclass
class LayerSpec:
    """Specification for a single layer in the architecture.

    Attributes:
        block_type: The type of neural operator block.
        width: Number of channels/features in this layer.
        num_modes: Number of Fourier modes (for Fourier layers).
        activation: Activation function name.
        has_residual: Whether this layer has a residual connection.
        skip_to: Layer index for a skip connection (-1 = none).

    """

    block_type: OperatorBlockType
    width: int = 64
    num_modes: int = 12
    activation: str = "gelu"
    has_residual: bool = True
    skip_to: int = -1

    def clone(self) -> LayerSpec:
        """Return an independent copy of this layer specification."""
        return LayerSpec(
            block_type=self.block_type,
            width=self.width,
            num_modes=self.num_modes,
            activation=self.activation,
            has_residual=self.has_residual,
            skip_to=self.skip_to,
        )

    @property
    def param_count_estimate(self) -> int:
        """Estimate parameter count for this layer.

        Uses a block-type multiplier applied to ``width ** 2``.
        Fourier layers additionally scale with ``num_modes``.

        Returns
        -------
        int
            Estimated number of learnable parameters.

        """
        multiplier = _BLOCK_PARAM_MULTIPLIERS.get(self.block_type, 1.0)
        base_params = int(multiplier * self.width * self.width)

        # Fourier layers also depend on mode count
        if self.block_type == OperatorBlockType.FOURIER_LAYER:
            base_params += self.width * self.num_modes

        return base_params


@dataclass
class ArchitectureState:
    """State of an architecture search.

    Captures the full architecture specification and performance
    metrics from evaluation.

    Attributes:
        layers: Ordered list of layer specifications.
        input_dim: Dimensionality of the input space.
        output_dim: Dimensionality of the output space.
        max_layers: Upper limit on layer count.
        max_width: Maximum allowed layer width.
        min_width: Minimum allowed layer width.
        max_modes: Maximum Fourier modes per layer.
        validation_error: Most recent validation error.
        training_cost: FLOPs or wall time of most recent evaluation.
        param_count: Actual parameter count (from external evaluation).
        step: Number of search actions taken so far.

    """

    layers: list[LayerSpec]
    input_dim: int = 2
    output_dim: int = 1
    max_layers: int = 20
    max_width: int = 512
    min_width: int = 16
    max_modes: int = 64
    validation_error: float = float("inf")
    training_cost: float = 0.0
    param_count: int = 0
    step: int = 0

    @property
    def depth(self) -> int:
        """Return the number of layers in the architecture."""
        return len(self.layers)

    @property
    def total_params(self) -> int:
        """Return the estimated total parameter count across all layers."""
        return sum(layer.param_count_estimate for layer in self.layers)

    def clone(self) -> ArchitectureState:
        """Return a deep, independent copy of this state."""
        return ArchitectureState(
            layers=[layer.clone() for layer in self.layers],
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            max_layers=self.max_layers,
            max_width=self.max_width,
            min_width=self.min_width,
            max_modes=self.max_modes,
            validation_error=self.validation_error,
            training_cost=self.training_cost,
            param_count=self.param_count,
            step=self.step,
        )


@dataclass
class NASAction:
    """An architecture search action.

    Attributes:
        action_type: The kind of architecture modification.
        layer_index: Index of the layer to modify or insert at.
        params: Additional parameters for the action.

    """

    action_type: NASActionType
    layer_index: int = 0
    params: dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Neural operator architecture search engine
# ======================================================================


class NeuralOperatorNAS:
    """Searches for optimal neural operator architectures via MCTS.

    Builds architectures incrementally by adding, removing, and
    modifying layers.  Uses validation error as the reward signal
    and parameter count as a cost regularizer.

    Parameters
    ----------
    max_layers:
        Upper limit on the number of layers in an architecture.
    max_width:
        Maximum channel width per layer.
    min_width:
        Minimum channel width per layer.
    num_simulations:
        Number of look-ahead simulations per planning step.
    width_step:
        Step size for width adjustments (must be > 0).
    mode_step:
        Step size for Fourier mode adjustments (must be > 0).
    complexity_penalty:
        Penalty coefficient for parameter count in scoring.

    """

    def __init__(
        self,
        max_layers: int = 20,
        max_width: int = 512,
        min_width: int = 16,
        num_simulations: int = 50,
        width_step: int = 16,
        mode_step: int = 4,
        complexity_penalty: float = 1e-6,
    ) -> None:
        self._max_layers = max_layers
        self._max_width = max_width
        self._min_width = min_width
        self._num_simulations = num_simulations
        self._width_step = width_step
        self._mode_step = mode_step
        self._complexity_penalty = complexity_penalty
        self._rng = np.random.default_rng(42)

        logger.info(
            "neural_operator_nas.init",
            max_layers=max_layers,
            max_width=max_width,
            num_simulations=num_simulations,
            complexity_penalty=complexity_penalty,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        initial_architecture: list[LayerSpec] | None = None,
        eval_fn: Callable[..., Any] | None = None,
        max_steps: int = 100,
    ) -> list[LayerSpec]:
        """Run architecture search and return the best architecture.

        If *eval_fn* is provided, it is called with a list of
        :class:`LayerSpec` and must return a tuple
        ``(validation_error, training_cost)``.  When *eval_fn* is
        ``None``, a heuristic score based on parameter count and
        architecture diversity is used.

        Parameters
        ----------
        initial_architecture:
            Starting architecture.  If ``None``, a minimal default
            architecture is created.
        eval_fn:
            Optional callable ``(layers) -> (error, cost)`` for
            evaluating candidate architectures.
        max_steps:
            Maximum number of search steps.

        Returns
        -------
        list[LayerSpec]
            The best architecture found during the search.

        """
        layers = (
            initial_architecture
            if initial_architecture is not None
            else self._default_initial_architecture()
        )

        state = ArchitectureState(
            layers=[l.clone() for l in layers],
            max_layers=self._max_layers,
            max_width=self._max_width,
            min_width=self._min_width,
        )

        # Evaluate initial architecture
        state = self._evaluate_state(state, eval_fn)

        best_state = state.clone()
        best_score = self._score_architecture(best_state)

        for step in range(max_steps):
            valid_actions = self.get_valid_actions(state)
            if not valid_actions:
                logger.debug("neural_operator_nas.no_valid_actions", step=step)
                break

            # Evaluate each action via simulation
            best_action = valid_actions[0]
            best_action_score = -float("inf")

            for action in valid_actions:
                total_score = 0.0
                for _ in range(self._num_simulations):
                    candidate = self.apply_action(state, action)
                    candidate = self._evaluate_state(candidate, eval_fn)
                    total_score += self._score_architecture(candidate)

                avg_score = total_score / self._num_simulations
                if avg_score > best_action_score:
                    best_action_score = avg_score
                    best_action = action

            # Apply the best action
            state = self.apply_action(state, best_action)
            state = self._evaluate_state(state, eval_fn)
            current_score = self._score_architecture(state)

            if current_score > best_score:
                best_score = current_score
                best_state = state.clone()

            logger.debug(
                "neural_operator_nas.step",
                step=step,
                action=best_action.action_type.value,
                score=current_score,
                best_score=best_score,
                depth=state.depth,
            )

        logger.info(
            "neural_operator_nas.complete",
            best_score=best_score,
            depth=best_state.depth,
            total_params=best_state.total_params,
        )
        return [l.clone() for l in best_state.layers]

    def get_valid_actions(
        self,
        state: ArchitectureState,
    ) -> list[NASAction]:
        """Return all valid actions for the current architecture state.

        Enforces layer count bounds, width bounds, and mode bounds
        to prevent degenerate architectures.
        """
        actions: list[NASAction] = []

        # ADD_LAYER: allowed if below max depth
        if state.depth < state.max_layers:
            for block_type in OperatorBlockType:
                actions.append(
                    NASAction(
                        action_type=NASActionType.ADD_LAYER,
                        layer_index=state.depth,
                        params={"block_type": block_type.value},
                    )
                )

        # Per-layer modifications
        for i, layer in enumerate(state.layers):
            # REMOVE_LAYER: allowed if more than 1 layer
            if state.depth > 1:
                actions.append(
                    NASAction(
                        action_type=NASActionType.REMOVE_LAYER,
                        layer_index=i,
                    )
                )

            # CHANGE_LAYER_TYPE: change to any other type
            for block_type in OperatorBlockType:
                if block_type != layer.block_type:
                    actions.append(
                        NASAction(
                            action_type=NASActionType.CHANGE_LAYER_TYPE,
                            layer_index=i,
                            params={"block_type": block_type.value},
                        )
                    )

            # ADJUST_WIDTH: increase or decrease
            if layer.width + self._width_step <= state.max_width:
                actions.append(
                    NASAction(
                        action_type=NASActionType.ADJUST_WIDTH,
                        layer_index=i,
                        params={"delta": self._width_step},
                    )
                )
            if layer.width - self._width_step >= state.min_width:
                actions.append(
                    NASAction(
                        action_type=NASActionType.ADJUST_WIDTH,
                        layer_index=i,
                        params={"delta": -self._width_step},
                    )
                )

            # ADJUST_MODES: only for Fourier layers
            if layer.block_type == OperatorBlockType.FOURIER_LAYER:
                if layer.num_modes + self._mode_step <= state.max_modes:
                    actions.append(
                        NASAction(
                            action_type=NASActionType.ADJUST_MODES,
                            layer_index=i,
                            params={"delta": self._mode_step},
                        )
                    )
                if layer.num_modes - self._mode_step >= self._mode_step:
                    actions.append(
                        NASAction(
                            action_type=NASActionType.ADJUST_MODES,
                            layer_index=i,
                            params={"delta": -self._mode_step},
                        )
                    )

            # ADD_SKIP_CONNECTION: connect to a later layer
            for j in range(i + 2, state.depth):
                if state.layers[j].skip_to == -1:
                    actions.append(
                        NASAction(
                            action_type=NASActionType.ADD_SKIP_CONNECTION,
                            layer_index=i,
                            params={"target_index": j},
                        )
                    )

            # TOGGLE_RESIDUAL
            actions.append(
                NASAction(
                    action_type=NASActionType.TOGGLE_RESIDUAL,
                    layer_index=i,
                )
            )

        # No-op is always valid
        actions.append(NASAction(action_type=NASActionType.NO_OP))

        return actions

    def apply_action(
        self,
        state: ArchitectureState,
        action: NASAction,
    ) -> ArchitectureState:
        """Apply an action and return the resulting architecture state.

        Returns a new state with the action applied.  The original
        state is never modified.

        Parameters
        ----------
        state:
            Current architecture search state.
        action:
            The architecture modification to apply.

        Returns
        -------
        ArchitectureState
            Updated state with the action applied.

        """
        new_state = state.clone()
        new_state.step = state.step + 1

        if action.action_type == NASActionType.ADD_LAYER:
            block_type_str = action.params.get(
                "block_type", OperatorBlockType.MLP_LAYER.value,
            )
            block_type = OperatorBlockType(block_type_str)
            new_layer = LayerSpec(block_type=block_type)
            idx = min(action.layer_index, len(new_state.layers))
            new_state.layers.insert(idx, new_layer)

        elif action.action_type == NASActionType.REMOVE_LAYER:
            idx = action.layer_index
            if 0 <= idx < len(new_state.layers) and len(new_state.layers) > 1:
                # Fix skip connections that reference removed or shifted layers
                removed_skip_to = new_state.layers[idx].skip_to
                new_state.layers.pop(idx)
                for layer in new_state.layers:
                    if layer.skip_to == idx:
                        layer.skip_to = -1
                    elif layer.skip_to > idx:
                        layer.skip_to -= 1
                # Suppress unused variable warning
                _ = removed_skip_to

        elif action.action_type == NASActionType.CHANGE_LAYER_TYPE:
            idx = action.layer_index
            if 0 <= idx < len(new_state.layers):
                block_type_str = action.params.get(
                    "block_type", OperatorBlockType.MLP_LAYER.value,
                )
                new_state.layers[idx].block_type = OperatorBlockType(
                    block_type_str,
                )

        elif action.action_type == NASActionType.ADJUST_WIDTH:
            idx = action.layer_index
            if 0 <= idx < len(new_state.layers):
                delta = action.params.get("delta", self._width_step)
                new_width = new_state.layers[idx].width + delta
                new_width = max(state.min_width, min(state.max_width, new_width))
                new_state.layers[idx].width = new_width

        elif action.action_type == NASActionType.ADJUST_MODES:
            idx = action.layer_index
            if 0 <= idx < len(new_state.layers):
                delta = action.params.get("delta", self._mode_step)
                new_modes = new_state.layers[idx].num_modes + delta
                new_modes = max(self._mode_step, min(state.max_modes, new_modes))
                new_state.layers[idx].num_modes = new_modes

        elif action.action_type == NASActionType.ADD_SKIP_CONNECTION:
            idx = action.layer_index
            target = action.params.get("target_index", -1)
            if (
                0 <= idx < len(new_state.layers)
                and 0 <= target < len(new_state.layers)
            ):
                new_state.layers[target].skip_to = idx

        elif action.action_type == NASActionType.TOGGLE_RESIDUAL:
            idx = action.layer_index
            if 0 <= idx < len(new_state.layers):
                new_state.layers[idx].has_residual = (
                    not new_state.layers[idx].has_residual
                )

        # NO_OP: do nothing

        return new_state

    # ------------------------------------------------------------------
    # Scoring and evaluation
    # ------------------------------------------------------------------

    def _score_architecture(self, state: ArchitectureState) -> float:
        """Score an architecture for search ranking.

        Score = -error - complexity_penalty * param_count

        Higher scores are better.  When no evaluation has been done
        (validation_error is inf), a heuristic based on architecture
        diversity is used.
        """
        if state.validation_error < float("inf"):
            return (
                -state.validation_error
                - self._complexity_penalty * state.total_params
            )

        # Heuristic: prefer diverse architectures with moderate depth
        block_types = {l.block_type for l in state.layers}
        diversity_bonus = len(block_types) * 0.1
        depth_penalty = abs(state.depth - 5) * 0.05
        return diversity_bonus - depth_penalty - self._complexity_penalty * state.total_params

    def _evaluate_state(
        self,
        state: ArchitectureState,
        eval_fn: Callable[..., Any] | None,
    ) -> ArchitectureState:
        """Evaluate an architecture state with the provided eval_fn.

        If *eval_fn* is ``None``, the state is returned unmodified
        (heuristic scoring will be used).
        """
        if eval_fn is not None:
            error, cost = eval_fn(state.layers)
            state.validation_error = float(error)
            state.training_cost = float(cost)
        state.param_count = state.total_params
        return state

    def _default_initial_architecture(self) -> list[LayerSpec]:
        """Create a minimal starting architecture.

        Returns a three-layer architecture:
        1. A Fourier layer for spectral mixing
        2. A Galerkin attention layer for global influence
        3. An MLP layer for output projection
        """
        return [
            LayerSpec(
                block_type=OperatorBlockType.FOURIER_LAYER,
                width=64,
                num_modes=12,
            ),
            LayerSpec(
                block_type=OperatorBlockType.GALERKIN_ATTENTION,
                width=64,
            ),
            LayerSpec(
                block_type=OperatorBlockType.MLP_LAYER,
                width=64,
            ),
        ]
