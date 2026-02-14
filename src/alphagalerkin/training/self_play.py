"""Self-play episode generation via MCTS."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import structlog

from src.alphagalerkin.core.config import (
    AlphaGalerkinConfig,
)
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.environment import DiscretizationEnvironment
from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.mcts.action_masking import ActionMasker
from src.alphagalerkin.mcts.backpropagation import backup
from src.alphagalerkin.mcts.node import MCTSNode
from src.alphagalerkin.mcts.noise import DirichletNoise
from src.alphagalerkin.mcts.temperature import TemperatureSchedule
from src.alphagalerkin.training.replay_buffer import Experience

logger = structlog.get_logger("training.self_play")

# Type alias for the neural-network evaluation callback.
EvalFn = Callable[
    [DiscretizationState],
    tuple[dict[Action, float], float],
]
"""Signature: ``(state) -> (action_priors, value)``."""


# -------------------------------------------------------------------
# Episode container
# -------------------------------------------------------------------

@dataclass
class Episode:
    """A complete self-play episode.

    Stores the full trajectory so that it can be converted
    into training experiences for the replay buffer.
    """

    states: list[DiscretizationState] = field(
        default_factory=list,
    )
    policies: list[dict[Action, float]] = field(
        default_factory=list,
    )
    rewards: list[float] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)

    # ---------------------------------------------------------------
    # Derived quantities
    # ---------------------------------------------------------------

    @property
    def length(self) -> int:
        """Number of time steps in the episode."""
        return len(self.states)

    @property
    def total_reward(self) -> float:
        """Undiscounted sum of rewards."""
        return sum(self.rewards)

    # ---------------------------------------------------------------
    # Conversion
    # ---------------------------------------------------------------

    def to_experiences(
        self,
        iteration: int = 0,
    ) -> list[Experience]:
        """Convert the episode into replay-buffer experiences.

        The value target for each step is the episode-average
        reward, serving as a simple Monte-Carlo return estimate.

        Parameters
        ----------
        iteration:
            Current training iteration for staleness tracking.

        Returns
        -------
        list[Experience]
            One experience per episode step.

        """
        experiences: list[Experience] = []
        outcome = self.total_reward / max(1, self.length)

        for state, policy in zip(
            self.states, self.policies, strict=True,
        ):
            # Serialize Action keys to strings
            policy_dict: dict[str, float] = {
                f"{a.element_id}_{a.action_type.value}": p
                for a, p in policy.items()
            }
            exp = Experience(
                state_features=(
                    state.to_feature_tensor().numpy()
                ),
                policy_target=policy_dict,
                value_target=outcome,
                iteration=iteration,
            )
            experiences.append(exp)

        return experiences


# -------------------------------------------------------------------
# Self-play engine
# -------------------------------------------------------------------

class SelfPlayEngine:
    """Generates self-play episodes using MCTS.

    The engine orchestrates one-episode loops:

    1. Reset the environment.
    2. At each step run MCTS from the current state.
    3. Select an action using temperature-scaled visit counts.
    4. Record ``(state, policy, action, reward)`` tuples.
    5. Repeat until a terminal condition is met.

    Parameters
    ----------
    config:
        Root ``AlphaGalerkinConfig`` driving all sub-components.

    """

    def __init__(
        self,
        config: AlphaGalerkinConfig,
    ) -> None:
        self._config = config
        self._env_config = config.environment
        self._mcts_config = config.mcts
        self._env = DiscretizationEnvironment(
            config.environment,
        )
        self._masker = ActionMasker(config.environment)
        self._noise = DirichletNoise(
            alpha=config.mcts.dirichlet_alpha,
            epsilon=config.mcts.dirichlet_epsilon,
        )
        self._temperature = TemperatureSchedule(
            schedule_type=(
                config.mcts.temperature_schedule.schedule_type
            ),
            initial=(
                config.mcts.temperature_schedule.initial_temp
            ),
            final=(
                config.mcts.temperature_schedule.final_temp
            ),
            decay_steps=(
                config.mcts.temperature_schedule.step_threshold
            ),
        )

        # Default evaluation function (uniform policy, 0 value)
        self._eval_fn: EvalFn = self._default_eval_fn

    # ---------------------------------------------------------------
    # Evaluation function management
    # ---------------------------------------------------------------

    def set_eval_fn(self, eval_fn: EvalFn) -> None:
        """Replace the evaluation function with the neural net.

        Parameters
        ----------
        eval_fn:
            Callable mapping a ``DiscretizationState`` to
            ``(action_priors, value)``.

        """
        self._eval_fn = eval_fn

    # ---------------------------------------------------------------
    # Episode generation
    # ---------------------------------------------------------------

    def play_episode(self) -> Episode:
        """Play one complete self-play episode.

        Returns
        -------
        Episode
            Full trajectory including states, policies,
            actions, and rewards.

        """
        state = self._env.reset()
        episode = Episode()

        max_steps = self._env_config.max_steps

        for step in range(max_steps):
            # Run MCTS search
            action, policy = self._search(state, step=step)

            episode.states.append(state)
            episode.policies.append(policy)
            episode.actions.append(action)

            result = self._env.step(action)
            episode.rewards.append(result.reward)

            if result.done:
                break

            state = result.state

        logger.info(
            "self_play.episode.complete",
            length=episode.length,
            total_reward=round(episode.total_reward, 4),
            final_dof=state.dof_count,
        )

        return episode

    # ---------------------------------------------------------------
    # MCTS search
    # ---------------------------------------------------------------

    def _search(
        self,
        state: DiscretizationState,
        step: int,
    ) -> tuple[Action, dict[Action, float]]:
        """Run MCTS from *state* and return (action, policy).

        Parameters
        ----------
        state:
            Current discretization state.
        step:
            Episode step index (for temperature scheduling).

        Returns
        -------
        tuple[Action, dict[Action, float]]
            Selected action and the full MCTS visit-count
            policy over valid actions.

        """
        root = MCTSNode(state=state)

        # Initial expansion with neural-net priors
        priors, _ = self._eval_fn(state)
        if not priors:
            valid = self._masker.valid_actions(state)
            if not valid:
                # No valid actions; return a no-op
                from src.alphagalerkin.core.types import (
                    ActionType,
                    ElementID,
                )
                noop = Action(
                    element_id=ElementID("e0"),
                    action_type=ActionType.NO_OP,
                )
                return noop, {}
            uniform = 1.0 / len(valid)
            priors = dict.fromkeys(valid, uniform)

        # Apply Dirichlet noise at the root
        noised_priors = self._noise.apply(priors)
        root.expand(noised_priors)

        # Simulations
        num_sims = self._mcts_config.num_simulations
        for _ in range(num_sims):
            node = root
            # Selection
            depth = 0
            while (
                not node.is_leaf
                and depth < self._mcts_config.max_tree_depth
            ):
                node = node.select_best_child(
                    self._mcts_config.c_puct,
                )
                depth += 1

            # Expansion + evaluation
            if not node.is_terminal:
                child_priors, value = self._eval_fn(
                    node.state,
                )
                if child_priors:
                    node.expand(child_priors)
                else:
                    value = 0.0
            else:
                value = 0.0

            # Backup
            backup(
                node,
                value,
                self._mcts_config.backup_strategy,
            )

        # Build visit-count policy
        visit_policy: dict[Action, float] = {}
        total_visits = sum(
            c.visit_count for c in root.children.values()
        )
        if total_visits > 0:
            for act, child in root.children.items():
                visit_policy[act] = (
                    child.visit_count / total_visits
                )
        else:
            for act in root.children:
                visit_policy[act] = (
                    1.0 / len(root.children)
                )

        # Temperature-scaled action selection
        temperature = self._temperature.get_temperature(step)
        visit_counts = {
            act: child.visit_count
            for act, child in root.children.items()
        }
        action = self._temperature.select_action_with_temperature(
            visit_counts,
            temperature,
        )

        return action, visit_policy

    # ---------------------------------------------------------------
    # Default evaluation
    # ---------------------------------------------------------------

    def _default_eval_fn(
        self,
        state: DiscretizationState,
    ) -> tuple[dict[Action, float], float]:
        """Default evaluation: uniform policy, zero value.

        Used before a trained neural network is available.
        """
        valid = self._masker.valid_actions(state)
        if not valid:
            return {}, 0.0
        uniform = 1.0 / len(valid)
        priors = dict.fromkeys(valid, uniform)
        return priors, 0.0
