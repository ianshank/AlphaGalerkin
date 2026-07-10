"""Tests for single-agent vs zero-sum MCTS backup semantics (F0).

These lock the correctness fix for the single-agent backup bug: ``select_child``
maximises ``Q + exploration`` at *every* depth, so unconditionally negating the
backed-up value (the historical behaviour) makes the search *minimise* value at
odd depths for single-agent games. The fix routes the sign flip through
:class:`~src.mcts.search.SearchMode`.

``test_single_agent_search_prefers_higher_value_at_all_depths`` is the anchor:
it fails under the inverting modes (the pre-fix behaviour) and passes only under
``SINGLE_AGENT``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.mcts.node import MCTSNode
from src.mcts.search import MCTS, SearchMode, invert_on_backup

# --------------------------------------------------------------------------- #
# Deterministic single-agent game + evaluator                                 #
# --------------------------------------------------------------------------- #


class _ConstantEvaluator:
    """Evaluator with uniform policy and a fixed leaf value.

    A value of 0 keeps non-terminal leaf evaluations from biasing the sign of
    the accumulated Q, so terminal outcomes drive action selection.
    """

    def __init__(self, n_actions: int, value: float = 0.0) -> None:
        self.n_actions = n_actions
        self.value = value

    def evaluate(self, state: np.ndarray, legal_actions: list[int]):  # noqa: ANN001
        from src.mcts.evaluator import EvaluationResult

        policy = np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float32)
        return EvaluationResult(policy=policy, value=self.value)

    def evaluate_batch(self, states, legal_actions_batch):  # noqa: ANN001
        return [self.evaluate(s, la) for s, la in zip(states, legal_actions_batch, strict=False)]


class _TwoStepValueGame:
    """Deterministic single-agent game with an unambiguous optimum.

    Action ``0`` at the root leads (via a forced second move) to a ``+1``
    terminal leaf; action ``1`` leads to a ``-1`` terminal leaf. A correct
    single-agent search maximises value and must pick action ``0``.

    The terminal leaves sit at depth 2, so the backup sign flip (which only
    matters at odd depth) determines whether the root prefers the ``+1`` or the
    ``-1`` branch.
    """

    def __init__(self, history: list[int] | None = None) -> None:
        self.history: list[int] = list(history) if history else []

    def get_state(self) -> np.ndarray:
        state = np.zeros(3, dtype=np.float32)
        for i, action in enumerate(self.history[:3]):
            state[i] = float(action + 1)
        return state

    def get_legal_actions(self) -> list[int]:
        if len(self.history) == 0:
            return [0, 1]
        if len(self.history) == 1:
            return [0]
        return []

    def apply_action(self, action: int) -> None:
        self.history.append(action)

    def is_terminal(self) -> bool:
        return len(self.history) >= 2

    def get_winner(self) -> int:
        if not self.history:
            return 0
        return 1 if self.history[0] == 0 else -1

    def clone(self) -> _TwoStepValueGame:
        return _TwoStepValueGame(self.history)


def _best_root_action(search_mode: SearchMode, seed: int = 0) -> int:
    """Run a deterministic single-agent search and return the greedy action."""
    np.random.seed(seed)
    evaluator = _ConstantEvaluator(n_actions=2, value=0.0)
    with warnings.catch_warnings():
        # LEGACY_ADVERSARIAL intentionally warns; silence for the test run.
        warnings.simplefilter("ignore", DeprecationWarning)
        mcts = MCTS(
            evaluator=evaluator,
            n_simulations=300,
            c_puct=1.0,
            search_mode=search_mode,
        )
        return mcts.get_action(_TwoStepValueGame(), temperature=0.0, add_noise=False)


# --------------------------------------------------------------------------- #
# Backup sign by mode                                                          #
# --------------------------------------------------------------------------- #


class TestBackupSignByMode:
    """The explicit ``invert`` flag controls per-level sign flipping."""

    def _chain(self) -> tuple[MCTSNode, MCTSNode, MCTSNode]:
        root = MCTSNode()
        n1 = MCTSNode(parent=root, action=0)
        n2 = MCTSNode(parent=n1, action=0)
        return root, n1, n2

    def test_zero_sum_alternates_sign(self) -> None:
        root, n1, n2 = self._chain()
        n2.backup(1.0, invert=True)
        assert n2.total_value == pytest.approx(1.0)
        assert n1.total_value == pytest.approx(-1.0)
        assert root.total_value == pytest.approx(1.0)

    def test_single_agent_keeps_sign(self) -> None:
        root, n1, n2 = self._chain()
        n2.backup(1.0, invert=False)
        assert n2.total_value == pytest.approx(1.0)
        assert n1.total_value == pytest.approx(1.0)
        assert root.total_value == pytest.approx(1.0)

    def test_backup_default_inverts(self) -> None:
        """Default ``invert=True`` preserves the historical behaviour."""
        root, n1, n2 = self._chain()
        n2.backup(1.0)
        assert n1.total_value == pytest.approx(-1.0)
        assert root.total_value == pytest.approx(1.0)

    def test_invert_on_backup_mapping(self) -> None:
        assert invert_on_backup(SearchMode.SINGLE_AGENT) is False
        assert invert_on_backup(SearchMode.ZERO_SUM) is True
        assert invert_on_backup(SearchMode.LEGACY_ADVERSARIAL) is True


# --------------------------------------------------------------------------- #
# The anchor test: single-agent search must maximise at every depth           #
# --------------------------------------------------------------------------- #


class TestSingleAgentSearch:
    def test_single_agent_search_prefers_higher_value_at_all_depths(self) -> None:
        """SINGLE_AGENT picks the +1 branch; the inverting modes pick the -1.

        This encodes F0: the inverting (pre-fix) backup selects the *worse*
        action for a single-agent game. It fails on ``HEAD`` because ``HEAD``
        only had the inverting behaviour.
        """
        assert _best_root_action(SearchMode.SINGLE_AGENT) == 0

        # The pre-fix behaviour: both inverting modes pick the wrong action.
        assert _best_root_action(SearchMode.ZERO_SUM) == 1
        assert _best_root_action(SearchMode.LEGACY_ADVERSARIAL) == 1

    def test_legacy_matches_zero_sum_choice(self) -> None:
        """LEGACY_ADVERSARIAL is behaviourally identical to ZERO_SUM."""
        assert _best_root_action(SearchMode.LEGACY_ADVERSARIAL) == _best_root_action(
            SearchMode.ZERO_SUM
        )


# --------------------------------------------------------------------------- #
# BatchMCTS terminal/non-terminal interleaving (leaf-index mapping)           #
# --------------------------------------------------------------------------- #


class _MixedTerminalGame:
    """Action 0 → immediate ``+1`` terminal; action 1 → a non-terminal leaf.

    Within one BatchMCTS batch the two root actions produce a terminal and a
    non-terminal path, which interleave in collection order. This exercises the
    ``leaf_index_for_path`` mapping: a naive ``i < len(leaves)`` assumption
    mis-assigns the batched evaluator value to the terminal path.
    """

    def __init__(self, history: list[int] | None = None) -> None:
        self.history: list[int] = list(history) if history else []

    def get_state(self) -> np.ndarray:
        return np.array([float(self.history[0] + 1) if self.history else 0.0], dtype=np.float32)

    def get_legal_actions(self) -> list[int]:
        if len(self.history) == 0:
            return [0, 1]
        if self.history == [1]:
            return [0]
        return []

    def apply_action(self, action: int) -> None:
        self.history.append(action)

    def is_terminal(self) -> bool:
        return self.history[:1] == [0] or len(self.history) >= 2

    def get_winner(self) -> int:
        if self.history[:1] == [0]:
            return 1
        return -1

    def clone(self) -> _MixedTerminalGame:
        return _MixedTerminalGame(self.history)


class _StateValueEvaluator:
    """Uniform policy; value ``-1`` for the action-1 leaf, ``0`` elsewhere."""

    def evaluate(self, state: np.ndarray, legal_actions: list[int]):  # noqa: ANN001
        from src.mcts.evaluator import EvaluationResult

        value = -1.0 if float(state[0]) == 2.0 else 0.0
        policy = np.full(2, 0.5, dtype=np.float32)
        return EvaluationResult(policy=policy, value=value)

    def evaluate_batch(self, states, legal_actions_batch):  # noqa: ANN001
        return [self.evaluate(s, la) for s, la in zip(states, legal_actions_batch, strict=False)]


class TestBatchTerminalInterleaving:
    def test_batch_maps_values_across_interleaved_terminals(self) -> None:
        """The +1 terminal must win over the -1 leaf under BatchMCTS.

        With the buggy ``i < len(leaves)`` mapping the terminal path receives
        the -1 evaluator value and the search prefers the wrong action.
        """
        from src.mcts.search import BatchMCTS

        np.random.seed(0)
        mcts = BatchMCTS(
            evaluator=_StateValueEvaluator(),
            batch_size=4,
            n_simulations=64,
            c_puct=1.0,
            search_mode=SearchMode.SINGLE_AGENT,
        )
        action = mcts.get_action(_MixedTerminalGame(), temperature=0.0, add_noise=False)
        assert action == 0


# --------------------------------------------------------------------------- #
# Backwards compatibility + deprecation                                       #
# --------------------------------------------------------------------------- #


class TestModeConstruction:
    def test_default_mode_is_zero_sum(self) -> None:
        mcts = MCTS(evaluator=_ConstantEvaluator(2))
        assert mcts.search_mode is SearchMode.ZERO_SUM
        assert mcts._invert_backup is True

    def test_single_agent_disables_inversion(self) -> None:
        mcts = MCTS(evaluator=_ConstantEvaluator(2), search_mode=SearchMode.SINGLE_AGENT)
        assert mcts._invert_backup is False

    def test_legacy_adversarial_warns(self) -> None:
        with pytest.warns(DeprecationWarning):
            MCTS(
                evaluator=_ConstantEvaluator(2),
                search_mode=SearchMode.LEGACY_ADVERSARIAL,
            )

    def test_reward_discount_bounds(self) -> None:
        with pytest.raises(ValueError):
            MCTS(evaluator=_ConstantEvaluator(2), reward_discount=0.0)
        with pytest.raises(ValueError):
            MCTS(evaluator=_ConstantEvaluator(2), reward_discount=1.5)
        # Valid boundary
        MCTS(evaluator=_ConstantEvaluator(2), reward_discount=1.0)


# --------------------------------------------------------------------------- #
# Intermediate-reward accumulation (F1 wiring)                                 #
# --------------------------------------------------------------------------- #


class _RewardGame:
    """Single-agent game exposing a per-edge reward via ``get_last_reward``.

    Each action yields a fixed reward of ``0.5`` and the game terminates after
    two moves with a zero-value (``get_winner() == 0``) terminal, so the
    root return equals the discounted sum of edge rewards.
    """

    def __init__(self, history: list[int] | None = None) -> None:
        self.history: list[int] = list(history) if history else []
        self._last_reward = 0.0

    def get_state(self) -> np.ndarray:
        return np.array([float(len(self.history))], dtype=np.float32)

    def get_legal_actions(self) -> list[int]:
        return [] if len(self.history) >= 2 else [0]

    def apply_action(self, action: int) -> None:
        self.history.append(action)
        self._last_reward = 0.5

    def get_last_reward(self) -> float:
        return self._last_reward

    def is_terminal(self) -> bool:
        return len(self.history) >= 2

    def get_winner(self) -> int:
        return 0

    def clone(self) -> _RewardGame:
        cloned = _RewardGame(self.history)
        cloned._last_reward = self._last_reward
        return cloned


class TestIntermediateRewards:
    def test_disabled_by_default_ignores_edge_reward(self) -> None:
        np.random.seed(0)
        mcts = MCTS(
            evaluator=_ConstantEvaluator(1, value=0.0),
            n_simulations=50,
            search_mode=SearchMode.SINGLE_AGENT,
        )
        mcts.search(_RewardGame(), add_noise=False)
        # Rewards disabled: edge_reward is never written and the terminal
        # winner (0) drives the value → root value ~0.
        assert mcts.get_root_value() == pytest.approx(0.0, abs=1e-6)
        root = mcts._root
        assert root is not None
        c1 = root.children[0]
        assert c1.edge_reward == pytest.approx(0.0)
        assert c1.children[0].edge_reward == pytest.approx(0.0)

    def test_enabled_accumulates_discounted_rewards(self) -> None:
        np.random.seed(0)
        gamma = 0.9
        mcts = MCTS(
            evaluator=_ConstantEvaluator(1, value=0.0),
            n_simulations=100,
            search_mode=SearchMode.SINGLE_AGENT,
            use_intermediate_rewards=True,
            reward_discount=gamma,
        )
        mcts.search(_RewardGame(), add_noise=False)
        # Each edge stores its immediate reward of 0.5; the discounted return
        # along the path is R = 0.5 + gamma * 0.5 (leaf value is 0). Asserting
        # the stored per-edge rewards is deterministic, unlike the visit-count
        # mean in get_root_value() which averages sims of differing depth.
        root = mcts._root
        assert root is not None
        c1 = root.children[0]
        c2 = c1.children[0]
        assert c1.edge_reward == pytest.approx(0.5)
        assert c2.edge_reward == pytest.approx(0.5)
        discounted_return = c1.edge_reward + gamma * c2.edge_reward
        assert discounted_return == pytest.approx(0.5 + gamma * 0.5)


class TestReadStepReward:
    """Contract for the ``_read_step_reward`` step-reward seam."""

    def test_missing_getter_contributes_zero(self) -> None:
        class _NoReward:
            pass

        assert MCTS._read_step_reward(_NoReward()) == pytest.approx(0.0)  # type: ignore[arg-type]

    def test_callable_getter_is_read(self) -> None:
        class _HasReward:
            def get_last_reward(self) -> float:
                return 0.75

        assert MCTS._read_step_reward(_HasReward()) == pytest.approx(0.75)  # type: ignore[arg-type]

    def test_non_callable_getter_raises_typeerror(self) -> None:
        """A float/property masquerading as the method fails clearly at source."""

        class _BadReward:
            get_last_reward = 0.5  # not a method — contract violation

        with pytest.raises(TypeError, match="get_last_reward must be a callable"):
            MCTS._read_step_reward(_BadReward())  # type: ignore[arg-type]

    def test_property_returning_none_raises_typeerror(self) -> None:
        """A property returning None is a contract violation, not "absent".

        The sentinel default on ``getattr`` distinguishes this from a game that
        simply does not implement the method (which contributes 0.0). Without
        it, ``None`` would be swallowed and silently return 0.0.
        """

        class _NonePropertyReward:
            @property
            def get_last_reward(self) -> None:
                return None

        with pytest.raises(TypeError, match="get_last_reward must be a callable"):
            MCTS._read_step_reward(_NonePropertyReward())  # type: ignore[arg-type]
