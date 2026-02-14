"""Quantum chemistry active space selection and DMRG orbital ordering.

This module applies MCTS-style planning to two key quantum chemistry
problems:

1. **Active space selection** for CASSCF: choosing k orbitals from N
   candidates for a CAS(n,k) calculation. Current approaches use
   chemical intuition or AutoCAS (two-orbital entropy from DMRG).
   MCTS provides look-ahead search over configurations.

2. **DMRG orbital ordering**: mapping orbitals to a 1D lattice for
   matrix product state calculations. Optimal ordering captures
   nearest-neighbor entanglement and is NP-hard in general.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import structlog

from src.alphagalerkin.core.constants import DEFAULT_SEED

logger = structlog.get_logger("planning.quantum_chem")


# ======================================================================
# Active Space Selection
# ======================================================================

class ActiveSpaceActionType(str, Enum):
    """Actions for active space selection."""

    ADD_ORBITAL = "add_orbital"
    """Include an orbital in the active space."""

    REMOVE_ORBITAL = "remove_orbital"
    """Remove an orbital from the active space."""

    SWAP_ORBITAL = "swap_orbital"
    """Swap an active orbital with an inactive one."""

    NO_OP = "no_op"
    """Do nothing -- always a valid action."""


@dataclass
class OrbitalInfo:
    """Information about a single molecular orbital.

    Attributes:
        index: Global orbital index.
        energy: Orbital energy (Hartree).
        occupation: Natural occupation number (0.0 to 2.0).
        symmetry_label: Irreducible representation label.
        single_orbital_entropy: Single-orbital entropy from DMRG.
        mutual_info: Mutual information with other orbitals (sparse).

    """

    index: int
    energy: float
    occupation: float  # 0.0 to 2.0
    symmetry_label: str = ""
    single_orbital_entropy: float = 0.0  # From DMRG

    # Mutual information with other orbitals (sparse)
    mutual_info: dict[int, float] = field(default_factory=dict)


@dataclass
class ActiveSpaceState:
    """State of an active space selection search.

    Attributes:
        all_orbitals: List of all candidate molecular orbitals.
        active_indices: Indices of currently selected active orbitals.
        num_electrons: Number of active electrons.
        max_active_orbitals: Maximum k in CAS(n,k).
        energy_estimate: Current energy estimate for the configuration.
        step: Number of planning steps taken so far.

    """

    all_orbitals: list[OrbitalInfo]
    active_indices: list[int]  # Currently selected orbital indices
    num_electrons: int  # Number of active electrons
    max_active_orbitals: int  # Maximum k in CAS(n,k)
    energy_estimate: float = 0.0  # Current energy estimate
    step: int = 0

    @property
    def num_active(self) -> int:
        """Return the number of currently active orbitals."""
        return len(self.active_indices)

    @property
    def inactive_indices(self) -> list[int]:
        """Return orbital indices not in the active space."""
        active_set = set(self.active_indices)
        return [o.index for o in self.all_orbitals if o.index not in active_set]

    def clone(self) -> ActiveSpaceState:
        """Return a deep, independent copy of this state."""
        return ActiveSpaceState(
            all_orbitals=[
                OrbitalInfo(
                    index=o.index,
                    energy=o.energy,
                    occupation=o.occupation,
                    symmetry_label=o.symmetry_label,
                    single_orbital_entropy=o.single_orbital_entropy,
                    mutual_info=dict(o.mutual_info),
                )
                for o in self.all_orbitals
            ],
            active_indices=list(self.active_indices),
            num_electrons=self.num_electrons,
            max_active_orbitals=self.max_active_orbitals,
            energy_estimate=self.energy_estimate,
            step=self.step,
        )


@dataclass
class ActiveSpaceAction:
    """An action in active space selection.

    Attributes:
        action_type: The type of action to take.
        orbital_index: Orbital to add or remove (-1 if unused).
        swap_target: Target orbital index for SWAP_ORBITAL (-1 if unused).

    """

    action_type: ActiveSpaceActionType
    orbital_index: int = -1
    swap_target: int = -1  # For SWAP_ORBITAL


class ActiveSpaceSelector:
    """Selects optimal active space using look-ahead planning.

    Searches over CAS(n,k) configurations by adding, removing,
    and swapping orbitals. Uses orbital entropy and mutual information
    as heuristics for the value function.

    Parameters
    ----------
    max_active:
        Maximum number of orbitals in the active space.
    num_simulations:
        Number of look-ahead simulations per planning step.
    entropy_weight:
        Weight applied to single-orbital entropy in scoring.
    energy_weight:
        Weight applied to energy-based scoring.
    max_steps:
        Maximum planning steps before termination.

    """

    def __init__(
        self,
        max_active: int = 14,
        num_simulations: int = 50,
        entropy_weight: float = 1.0,
        energy_weight: float = 1.0,
        max_steps: int = 50,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._max_active = max_active
        self._num_simulations = num_simulations
        self._entropy_weight = entropy_weight
        self._energy_weight = energy_weight
        self._max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        logger.info(
            "active_space_selector.init",
            max_active=max_active,
            num_simulations=num_simulations,
            entropy_weight=entropy_weight,
            energy_weight=energy_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_active_space(
        self,
        orbitals: list[OrbitalInfo],
        num_electrons: int,
        energy_fn: Callable[[list[int]], float] | None = None,
    ) -> list[int]:
        """Select optimal active orbital indices.

        If energy_fn is provided, it computes CAS energy for a given
        orbital selection. Otherwise, entropy-based heuristics are used.

        Parameters
        ----------
        orbitals:
            List of all candidate molecular orbitals.
        num_electrons:
            Number of active electrons.
        energy_fn:
            Optional callable ``(indices: list[int]) -> float`` that
            returns the CAS energy for a given orbital selection.

        Returns
        -------
        list[int]
            Indices of the selected active orbitals.

        """
        state = ActiveSpaceState(
            all_orbitals=orbitals,
            active_indices=[],
            num_electrons=num_electrons,
            max_active_orbitals=self._max_active,
        )

        best_indices: list[int] = []
        best_score = -float("inf")

        for step in range(self._max_steps):
            valid_actions = self.get_valid_actions(state)
            if not valid_actions:
                break

            best_action = valid_actions[0]
            best_action_score = -float("inf")

            for action in valid_actions:
                total_score = 0.0
                for _ in range(self._num_simulations):
                    new_state = self.apply_action(state, action)
                    if energy_fn is not None and new_state.active_indices:
                        energy = energy_fn(new_state.active_indices)
                        # Lower energy is better -- negate for score
                        score = -energy * self._energy_weight
                    else:
                        score = self._score_configuration(new_state)
                    total_score += score

                avg_score = total_score / self._num_simulations
                if avg_score > best_action_score:
                    best_action_score = avg_score
                    best_action = action

            state = self.apply_action(state, best_action)

            current_score = self._score_configuration(state)
            if current_score > best_score and state.active_indices:
                best_score = current_score
                best_indices = list(state.active_indices)

            logger.debug(
                "active_space_selector.step",
                step=step,
                action=best_action.action_type.value,
                num_active=state.num_active,
                score=current_score,
            )

            # Stop if we hit a NO_OP
            if best_action.action_type == ActiveSpaceActionType.NO_OP:
                break

        logger.info(
            "active_space_selector.complete",
            selected_orbitals=best_indices,
            num_selected=len(best_indices),
            score=best_score,
        )
        return best_indices

    def get_valid_actions(
        self,
        state: ActiveSpaceState,
    ) -> list[ActiveSpaceAction]:
        """Return valid actions for the current state.

        Enforces active space size bounds and avoids duplicate
        orbital selections.
        """
        actions: list[ActiveSpaceAction] = []

        # ADD_ORBITAL: if below max capacity
        if state.num_active < state.max_active_orbitals:
            for idx in state.inactive_indices:
                actions.append(
                    ActiveSpaceAction(
                        action_type=ActiveSpaceActionType.ADD_ORBITAL,
                        orbital_index=idx,
                    )
                )

        # REMOVE_ORBITAL: if there are active orbitals
        if state.num_active > 0:
            for idx in state.active_indices:
                actions.append(
                    ActiveSpaceAction(
                        action_type=ActiveSpaceActionType.REMOVE_ORBITAL,
                        orbital_index=idx,
                    )
                )

        # SWAP_ORBITAL: swap an active with an inactive
        if state.num_active > 0 and state.inactive_indices:
            for active_idx in state.active_indices:
                for inactive_idx in state.inactive_indices:
                    actions.append(
                        ActiveSpaceAction(
                            action_type=ActiveSpaceActionType.SWAP_ORBITAL,
                            orbital_index=active_idx,
                            swap_target=inactive_idx,
                        )
                    )

        # NO_OP is always valid
        actions.append(
            ActiveSpaceAction(action_type=ActiveSpaceActionType.NO_OP)
        )

        return actions

    def apply_action(
        self,
        state: ActiveSpaceState,
        action: ActiveSpaceAction,
    ) -> ActiveSpaceState:
        """Apply an action to produce a new state.

        Parameters
        ----------
        state:
            Current active space state.
        action:
            The action to apply.

        Returns
        -------
        ActiveSpaceState
            Updated state with the action applied.

        """
        new_state = state.clone()
        new_state.step = state.step + 1

        if action.action_type == ActiveSpaceActionType.ADD_ORBITAL:
            if (
                action.orbital_index not in new_state.active_indices
                and new_state.num_active < new_state.max_active_orbitals
            ):
                new_state.active_indices.append(action.orbital_index)

        elif action.action_type == ActiveSpaceActionType.REMOVE_ORBITAL:
            if action.orbital_index in new_state.active_indices:
                new_state.active_indices.remove(action.orbital_index)

        elif action.action_type == ActiveSpaceActionType.SWAP_ORBITAL:
            if (
                action.orbital_index in new_state.active_indices
                and action.swap_target not in new_state.active_indices
            ):
                idx = new_state.active_indices.index(action.orbital_index)
                new_state.active_indices[idx] = action.swap_target

        # NO_OP: do nothing

        return new_state

    def _score_configuration(self, state: ActiveSpaceState) -> float:
        """Score using total single-orbital entropy + mutual info of active orbitals.

        Higher entropy orbitals contribute more to correlation -- they
        are better candidates for the active space.  Mutual information
        between active orbitals further boosts the score because it
        indicates correlated pairs that must be treated together.

        Returns
        -------
        float
            The heuristic score (higher is better).

        """
        if not state.active_indices:
            return 0.0

        active_set = set(state.active_indices)
        orbital_map = {o.index: o for o in state.all_orbitals}

        # Sum single-orbital entropy
        entropy_score = 0.0
        for idx in state.active_indices:
            orb = orbital_map.get(idx)
            if orb is not None:
                entropy_score += orb.single_orbital_entropy

        # Sum mutual information between active orbital pairs
        mutual_info_score = 0.0
        for idx in state.active_indices:
            orb = orbital_map.get(idx)
            if orb is not None:
                for other_idx, mi_val in orb.mutual_info.items():
                    if other_idx in active_set and other_idx > idx:
                        mutual_info_score += mi_val

        return (
            self._entropy_weight * entropy_score
            + self._energy_weight * mutual_info_score
        )


# ======================================================================
# DMRG Orbital Ordering
# ======================================================================

class OrderingActionType(str, Enum):
    """Actions for DMRG orbital ordering."""

    SWAP_ADJACENT = "swap_adjacent"
    """Swap two adjacent orbitals in the 1D chain."""

    SWAP_ANY = "swap_any"
    """Swap any two orbitals in the 1D chain."""

    REVERSE_SEGMENT = "reverse_segment"
    """Reverse a contiguous segment of the ordering."""

    NO_OP = "no_op"
    """Do nothing -- always a valid action."""


@dataclass
class DMRGOrderingState:
    """State of a DMRG orbital ordering search.

    Attributes:
        ordering: Current orbital permutation (list of orbital indices).
        entanglement_matrix: (N, N) mutual information matrix.
        bond_dimension_cost: Estimated total bond dimension cost.
        step: Number of planning steps taken so far.
        max_steps: Maximum planning steps before termination.

    """

    ordering: list[int]  # Current orbital permutation
    entanglement_matrix: np.ndarray  # (N,N) mutual information
    bond_dimension_cost: float = 0.0
    step: int = 0
    max_steps: int = 100

    def clone(self) -> DMRGOrderingState:
        """Return a deep, independent copy of this state."""
        return DMRGOrderingState(
            ordering=list(self.ordering),
            entanglement_matrix=self.entanglement_matrix.copy(),
            bond_dimension_cost=self.bond_dimension_cost,
            step=self.step,
            max_steps=self.max_steps,
        )

    @property
    def linear_entropy(self) -> float:
        """Compute sum of entanglement across all bonds in 1D chain.

        For each bond position i (between site i and site i+1),
        the entanglement is the sum of mutual information between
        orbitals on opposite sides of the cut.

        Returns
        -------
        float
            Total linear entanglement (lower is better for DMRG).

        """
        n = len(self.ordering)
        if n <= 1:
            return 0.0

        total = 0.0
        for cut in range(1, n):
            # Orbitals on left side of cut: ordering[0:cut]
            # Orbitals on right side: ordering[cut:]
            left = self.ordering[:cut]
            right = self.ordering[cut:]
            for l_idx in left:
                for r_idx in right:
                    total += self.entanglement_matrix[l_idx, r_idx]

        return float(total)


@dataclass
class OrderingAction:
    """An action in DMRG ordering.

    Attributes:
        action_type: The kind of reordering to perform.
        position_a: First position index in the ordering.
        position_b: Second position index (or segment end).

    """

    action_type: OrderingActionType
    position_a: int = 0
    position_b: int = 1


class DMRGOrderingOptimizer:
    """Optimizes DMRG orbital ordering via look-ahead search.

    Searches over permutations to minimize total entanglement
    across bonds in the 1D chain representation.

    Parameters
    ----------
    num_simulations:
        Number of look-ahead simulations per planning step.
    max_segment_length:
        Maximum segment length for REVERSE_SEGMENT actions.

    """

    def __init__(
        self,
        num_simulations: int = 100,
        max_segment_length: int = 5,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._num_simulations = num_simulations
        self._max_segment_length = max_segment_length
        self._rng = np.random.default_rng(seed)

        logger.info(
            "dmrg_ordering_optimizer.init",
            num_simulations=num_simulations,
            max_segment_length=max_segment_length,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize_ordering(
        self,
        entanglement_matrix: np.ndarray,
        initial_ordering: list[int] | None = None,
        max_steps: int = 100,
    ) -> list[int]:
        """Find optimal orbital ordering for DMRG.

        Returns permutation that minimizes linear entanglement.

        Parameters
        ----------
        entanglement_matrix:
            (N, N) symmetric mutual information matrix.
        initial_ordering:
            Starting permutation. If None, uses [0, 1, ..., N-1].
        max_steps:
            Maximum number of optimisation steps.

        Returns
        -------
        list[int]
            Optimal orbital ordering found by the search.

        """
        n = entanglement_matrix.shape[0]
        if initial_ordering is not None:
            ordering = list(initial_ordering)
        else:
            ordering = list(range(n))

        state = DMRGOrderingState(
            ordering=ordering,
            entanglement_matrix=entanglement_matrix,
            max_steps=max_steps,
        )

        best_ordering = list(state.ordering)
        best_entropy = state.linear_entropy

        for step in range(max_steps):
            valid_actions = self.get_valid_actions(state)
            if not valid_actions:
                break

            best_action = valid_actions[0]
            best_action_entropy = float("inf")

            for action in valid_actions:
                total_entropy = 0.0
                for _ in range(self._num_simulations):
                    new_state = self.apply_action(state, action)
                    total_entropy += new_state.linear_entropy

                avg_entropy = total_entropy / self._num_simulations
                if avg_entropy < best_action_entropy:
                    best_action_entropy = avg_entropy
                    best_action = action

            state = self.apply_action(state, best_action)

            current_entropy = state.linear_entropy
            if current_entropy < best_entropy:
                best_entropy = current_entropy
                best_ordering = list(state.ordering)

            logger.debug(
                "dmrg_ordering_optimizer.step",
                step=step,
                action=best_action.action_type.value,
                entropy=current_entropy,
                best_entropy=best_entropy,
            )

            # Stop if we hit a NO_OP
            if best_action.action_type == OrderingActionType.NO_OP:
                break

        logger.info(
            "dmrg_ordering_optimizer.complete",
            best_entropy=best_entropy,
            ordering=best_ordering,
        )
        return best_ordering

    def get_valid_actions(
        self,
        state: DMRGOrderingState,
    ) -> list[OrderingAction]:
        """Return valid ordering actions for the current state.

        Generates adjacent swaps, long-range swaps, and segment
        reversals up to the configured maximum segment length.
        """
        actions: list[OrderingAction] = []
        n = len(state.ordering)

        # SWAP_ADJACENT: every adjacent pair
        for i in range(n - 1):
            actions.append(
                OrderingAction(
                    action_type=OrderingActionType.SWAP_ADJACENT,
                    position_a=i,
                    position_b=i + 1,
                )
            )

        # SWAP_ANY: all non-adjacent pairs
        for i in range(n):
            for j in range(i + 2, n):
                actions.append(
                    OrderingAction(
                        action_type=OrderingActionType.SWAP_ANY,
                        position_a=i,
                        position_b=j,
                    )
                )

        # REVERSE_SEGMENT: segments of length 3 up to max_segment_length
        for length in range(3, min(self._max_segment_length + 1, n + 1)):
            for start in range(n - length + 1):
                actions.append(
                    OrderingAction(
                        action_type=OrderingActionType.REVERSE_SEGMENT,
                        position_a=start,
                        position_b=start + length - 1,
                    )
                )

        # NO_OP is always valid
        actions.append(
            OrderingAction(action_type=OrderingActionType.NO_OP)
        )

        return actions

    def apply_action(
        self,
        state: DMRGOrderingState,
        action: OrderingAction,
    ) -> DMRGOrderingState:
        """Apply an action to produce a new state.

        Parameters
        ----------
        state:
            Current ordering state.
        action:
            The reordering action to apply.

        Returns
        -------
        DMRGOrderingState
            Updated state with the new ordering.

        """
        new_state = state.clone()
        new_state.step = state.step + 1

        if action.action_type in (
            OrderingActionType.SWAP_ADJACENT,
            OrderingActionType.SWAP_ANY,
        ):
            a, b = action.position_a, action.position_b
            if 0 <= a < len(new_state.ordering) and 0 <= b < len(new_state.ordering):
                new_state.ordering[a], new_state.ordering[b] = (
                    new_state.ordering[b],
                    new_state.ordering[a],
                )

        elif action.action_type == OrderingActionType.REVERSE_SEGMENT:
            a, b = action.position_a, action.position_b
            if 0 <= a <= b < len(new_state.ordering):
                new_state.ordering[a : b + 1] = new_state.ordering[a : b + 1][::-1]

        # NO_OP: do nothing

        return new_state
