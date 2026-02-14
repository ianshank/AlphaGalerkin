"""MCTS tree manager: orchestrates search."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import structlog

from src.alphagalerkin.core.config import MCTSConfig
from src.alphagalerkin.core.types import (
    ActionType,
    ElementID,
)
from src.alphagalerkin.mcts.backpropagation import backup
from src.alphagalerkin.mcts.node import MCTSNode
from src.alphagalerkin.mcts.noise import DirichletNoise
from src.alphagalerkin.mcts.selection import (
    get_selection_fn,
)
from src.alphagalerkin.mcts.temperature import TemperatureSchedule

if TYPE_CHECKING:
    from src.alphagalerkin.env.actions import Action
    from src.alphagalerkin.env.state import DiscretizationState

logger = structlog.get_logger("mcts.tree")


# Type alias for the neural-network evaluation callback.
EvalFn = Callable[
    ["DiscretizationState"],
    tuple[dict["Action", float], float],
]
"""(state) -> (action_priors, value_estimate)."""


class TreeManager:
    """Orchestrates MCTS search over discretization actions.

    The manager owns the root node, runs *N* simulations of
    select -> expand -> evaluate -> backup, then produces a
    policy distribution and a sampled action.

    Args:
        config: Full MCTS configuration (simulations, cpuct,
            noise parameters, temperature schedule, etc.).
        eval_fn: Callable that takes a
            :class:`DiscretizationState` and returns
            ``(action_priors, value)``.
        valid_actions_fn: Callable that takes a state and
            returns the list of legal actions.

    """

    def __init__(
        self,
        config: MCTSConfig,
        eval_fn: EvalFn,
        valid_actions_fn: Callable[
            [DiscretizationState], list[Action]
        ],
    ) -> None:
        self._config = config
        self._eval_fn = eval_fn
        self._valid_actions_fn = valid_actions_fn

        self._noise = DirichletNoise(
            alpha=config.dirichlet_alpha,
            epsilon=config.dirichlet_epsilon,
        )
        self._temperature = TemperatureSchedule(
            schedule_type=(
                config.temperature_schedule.schedule_type
            ),
            initial=config.temperature_schedule.initial_temp,
            final=config.temperature_schedule.final_temp,
            decay_steps=(
                config.temperature_schedule.step_threshold
            ),
        )
        self._selection_fn = get_selection_fn(
            config.selection_policy,
        )
        self._rng = np.random.default_rng()
        self._tree_size: int = 0

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def search(
        self,
        root_state: DiscretizationState,
        step: int = 0,
    ) -> tuple[Action, dict[Action, float]]:
        """Run MCTS from *root_state*.

        Args:
            root_state: Current environment state.
            step: Episode step index (used for temperature).

        Returns:
            ``(selected_action, policy_distribution)`` where the
            policy maps every expanded action to its normalised
            visit-count share.

        """
        root = MCTSNode(state=root_state, prior=1.0)
        self._tree_size = 1

        # Initial expansion with noise at root
        priors, value = self._eval_fn(root_state)

        valid = self._valid_actions_fn(root_state)
        if not valid:
            logger.warning("mcts.no_valid_actions")
            noop = self._make_noop(root_state)
            return noop, {noop: 1.0}

        filtered = self._filter_priors(priors, valid)

        if self._config.noise_at_root_only:
            filtered = self._noise.apply(
                filtered, self._rng,
            )

        root.expand(filtered)
        self._tree_size += len(filtered)

        # Simulations
        for _ in range(self._config.num_simulations):
            leaf = self._select(root)

            if not leaf.is_terminal and leaf.is_leaf:
                leaf_value = self._expand_and_evaluate(leaf)
            else:
                leaf_value = (
                    leaf.mean_value
                    if leaf.visit_count > 0
                    else 0.0
                )

            backup(
                leaf,
                leaf_value,
                self._config.backup_strategy,
            )

        # Build policy from visit counts
        visit_counts: dict[Action, int] = {
            action: child.visit_count
            for action, child in root.children.items()
        }
        total_visits = sum(visit_counts.values())
        policy: dict[Action, float] = {
            action: count / max(1, total_visits)
            for action, count in visit_counts.items()
        }

        # Select with temperature
        temperature = self._temperature.get_temperature(step)
        selected = self._temperature.select_action_with_temperature(
            visit_counts, temperature, self._rng,
        )

        logger.info(
            "mcts.search.complete",
            num_simulations=self._config.num_simulations,
            tree_size=self._tree_size,
            temperature=round(temperature, 4),
            selected_action=(
                str(selected.action_type.value)
                if hasattr(selected, "action_type")
                else str(selected)
            ),
        )

        return selected, policy

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse the tree to a leaf via selection policy."""
        current = node
        depth = 0
        while (
            not current.is_leaf
            and depth < self._config.max_tree_depth
        ):
            current = self._select_child(current)
            depth += 1
        return current

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Pick the best child of *node*.

        Uses the configured selection function to score each
        child and returns the one with the highest score.
        """
        best_score = float("-inf")
        best_child: MCTSNode | None = None
        parent_visits = node.visit_count

        for child in node.children.values():
            score = self._selection_fn(
                child,
                self._config.c_puct,
                parent_visits,
            )
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:  # pragma: no cover
            msg = "Failed to select child"
            raise RuntimeError(msg)

        return best_child

    def _expand_and_evaluate(
        self,
        leaf: MCTSNode,
    ) -> float:
        """Expand *leaf* and return the network value."""
        priors, value = self._eval_fn(leaf.state)

        valid = self._valid_actions_fn(leaf.state)
        if not valid:
            leaf.is_terminal = True
            return value

        filtered = self._filter_priors(priors, valid)

        if filtered:
            leaf.expand(filtered)
            self._tree_size += len(filtered)
        else:
            leaf.is_terminal = True

        return value

    def _filter_priors(
        self,
        priors: dict[Action, float],
        valid: list[Action],
    ) -> dict[Action, float]:
        """Restrict *priors* to *valid* actions and normalise."""
        topk = valid[: self._config.action_topk]
        filtered: dict[Action, float] = {}
        n_valid = len(valid)
        for a in topk:
            filtered[a] = priors.get(
                a, 1.0 / max(1, n_valid),
            )

        total = sum(filtered.values())
        if total > 0:
            filtered = {
                a: p / total for a, p in filtered.items()
            }
        return filtered

    @staticmethod
    def _make_noop(
        state: DiscretizationState,
    ) -> Action:
        """Create a NO_OP action for *state*."""
        from src.alphagalerkin.env.actions import Action as _Act

        eid = (
            state.mesh.element_ids[0]
            if state.mesh.element_ids
            else ElementID("e0")
        )
        return _Act(
            element_id=eid,
            action_type=ActionType.NO_OP,
        )
