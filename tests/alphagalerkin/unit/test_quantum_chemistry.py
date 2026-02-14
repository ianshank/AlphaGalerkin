"""Tests for quantum chemistry active space selection and DMRG ordering."""

from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.planning.quantum_chemistry import (
    ActiveSpaceAction,
    ActiveSpaceActionType,
    ActiveSpaceSelector,
    ActiveSpaceState,
    DMRGOrderingOptimizer,
    DMRGOrderingState,
    OrbitalInfo,
    OrderingAction,
    OrderingActionType,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_orbitals(n: int = 10) -> list[OrbitalInfo]:
    """Create a list of test orbitals with synthetic entropies."""
    rng = np.random.default_rng(0)
    orbitals = []
    for i in range(n):
        mutual_info: dict[int, float] = {}
        for j in range(n):
            if j != i:
                mutual_info[j] = rng.uniform(0.0, 0.5)
        orbitals.append(
            OrbitalInfo(
                index=i,
                energy=-10.0 + i * 0.5,
                occupation=2.0 - i * 0.2 if i < 10 else 0.0,
                symmetry_label=f"a{i}",
                single_orbital_entropy=rng.uniform(0.1, 1.0),
                mutual_info=mutual_info,
            )
        )
    return orbitals


@pytest.fixture()
def orbitals() -> list[OrbitalInfo]:
    """Ten test orbitals with synthetic data."""
    return _make_orbitals(10)


@pytest.fixture()
def active_space_state(orbitals: list[OrbitalInfo]) -> ActiveSpaceState:
    """An active space state with 3 active orbitals out of 10."""
    return ActiveSpaceState(
        all_orbitals=orbitals,
        active_indices=[0, 1, 2],
        num_electrons=4,
        max_active_orbitals=6,
        energy_estimate=-75.0,
        step=0,
    )


@pytest.fixture()
def selector() -> ActiveSpaceSelector:
    """An active space selector with small simulation count for tests."""
    return ActiveSpaceSelector(
        max_active=6,
        num_simulations=2,
        entropy_weight=1.0,
        energy_weight=1.0,
        max_steps=5,
    )


@pytest.fixture()
def entanglement_matrix() -> np.ndarray:
    """A 6x6 symmetric entanglement matrix with structure."""
    rng = np.random.default_rng(42)
    n = 6
    mat = rng.uniform(0.0, 0.5, size=(n, n))
    mat = (mat + mat.T) / 2.0
    np.fill_diagonal(mat, 0.0)
    # Add strong coupling between orbitals 0-1 and 4-5
    mat[0, 1] = mat[1, 0] = 2.0
    mat[4, 5] = mat[5, 4] = 2.0
    return mat


@pytest.fixture()
def dmrg_state(entanglement_matrix: np.ndarray) -> DMRGOrderingState:
    """A DMRG ordering state with 6 orbitals."""
    return DMRGOrderingState(
        ordering=[0, 1, 2, 3, 4, 5],
        entanglement_matrix=entanglement_matrix,
        step=0,
        max_steps=50,
    )


@pytest.fixture()
def dmrg_optimizer() -> DMRGOrderingOptimizer:
    """A DMRG optimizer with small simulation count for tests."""
    return DMRGOrderingOptimizer(
        num_simulations=1,
        max_segment_length=4,
    )


# ------------------------------------------------------------------
# Active Space Selection Tests
# ------------------------------------------------------------------


class TestOrbitalInfoDataclass:
    """OrbitalInfo stores orbital data correctly."""

    def test_orbital_info_dataclass(self) -> None:
        orb = OrbitalInfo(
            index=3,
            energy=-7.5,
            occupation=1.8,
            symmetry_label="b2u",
            single_orbital_entropy=0.42,
            mutual_info={0: 0.1, 5: 0.3},
        )

        assert orb.index == 3
        assert orb.energy == pytest.approx(-7.5)
        assert orb.occupation == pytest.approx(1.8)
        assert orb.symmetry_label == "b2u"
        assert orb.single_orbital_entropy == pytest.approx(0.42)
        assert orb.mutual_info == {0: 0.1, 5: 0.3}

    def test_orbital_info_defaults(self) -> None:
        orb = OrbitalInfo(index=0, energy=-10.0, occupation=2.0)

        assert orb.symmetry_label == ""
        assert orb.single_orbital_entropy == 0.0
        assert orb.mutual_info == {}


class TestActiveSpaceStateClone:
    """ActiveSpaceState.clone produces an independent copy."""

    def test_active_space_state_clone(
        self,
        active_space_state: ActiveSpaceState,
    ) -> None:
        cloned = active_space_state.clone()

        # Must be a different object
        assert cloned is not active_space_state
        assert cloned.all_orbitals is not active_space_state.all_orbitals
        assert cloned.active_indices is not active_space_state.active_indices

        # Values must match
        assert cloned.num_electrons == active_space_state.num_electrons
        assert cloned.max_active_orbitals == active_space_state.max_active_orbitals
        assert cloned.energy_estimate == active_space_state.energy_estimate
        assert cloned.step == active_space_state.step
        assert cloned.active_indices == active_space_state.active_indices

        # Orbital data must match
        for orig, clone in zip(
            active_space_state.all_orbitals,
            cloned.all_orbitals,
        ):
            assert clone.index == orig.index
            assert clone.energy == orig.energy
            assert clone.mutual_info is not orig.mutual_info
            assert clone.mutual_info == orig.mutual_info

    def test_clone_mutation_independence(
        self,
        active_space_state: ActiveSpaceState,
    ) -> None:
        cloned = active_space_state.clone()
        cloned.active_indices.append(99)
        cloned.energy_estimate = -999.0
        cloned.all_orbitals[0].energy = -999.0

        assert 99 not in active_space_state.active_indices
        assert active_space_state.energy_estimate == -75.0
        assert active_space_state.all_orbitals[0].energy != -999.0


class TestActiveSpaceValidActions:
    """ActiveSpaceSelector.get_valid_actions returns correct actions."""

    def test_active_space_valid_actions(
        self,
        selector: ActiveSpaceSelector,
        active_space_state: ActiveSpaceState,
    ) -> None:
        actions = selector.get_valid_actions(active_space_state)
        action_types = {a.action_type for a in actions}

        # NO_OP always present
        assert ActiveSpaceActionType.NO_OP in action_types

        # With 3 active out of 6 max and 10 total: ADD, REMOVE, SWAP valid
        assert ActiveSpaceActionType.ADD_ORBITAL in action_types
        assert ActiveSpaceActionType.REMOVE_ORBITAL in action_types
        assert ActiveSpaceActionType.SWAP_ORBITAL in action_types

    def test_no_add_at_max_active(
        self,
        selector: ActiveSpaceSelector,
        active_space_state: ActiveSpaceState,
    ) -> None:
        """ADD_ORBITAL is excluded when active space is at max capacity."""
        active_space_state.active_indices = [0, 1, 2, 3, 4, 5]

        actions = selector.get_valid_actions(active_space_state)
        action_types = {a.action_type for a in actions}
        assert ActiveSpaceActionType.ADD_ORBITAL not in action_types

    def test_no_remove_when_empty(
        self,
        selector: ActiveSpaceSelector,
        active_space_state: ActiveSpaceState,
    ) -> None:
        """REMOVE_ORBITAL and SWAP_ORBITAL excluded with empty active space."""
        active_space_state.active_indices = []

        actions = selector.get_valid_actions(active_space_state)
        action_types = {a.action_type for a in actions}
        assert ActiveSpaceActionType.REMOVE_ORBITAL not in action_types
        assert ActiveSpaceActionType.SWAP_ORBITAL not in action_types


class TestActiveSpaceAddOrbital:
    """apply_action with ADD_ORBITAL adds an orbital to the active space."""

    def test_active_space_add_orbital(
        self,
        selector: ActiveSpaceSelector,
        active_space_state: ActiveSpaceState,
    ) -> None:
        action = ActiveSpaceAction(
            action_type=ActiveSpaceActionType.ADD_ORBITAL,
            orbital_index=5,
        )
        new_state = selector.apply_action(active_space_state, action)

        assert 5 in new_state.active_indices
        assert new_state.num_active == active_space_state.num_active + 1
        assert new_state.step == active_space_state.step + 1
        # Original unchanged
        assert 5 not in active_space_state.active_indices


class TestActiveSpaceRemoveOrbital:
    """apply_action with REMOVE_ORBITAL removes an orbital."""

    def test_active_space_remove_orbital(
        self,
        selector: ActiveSpaceSelector,
        active_space_state: ActiveSpaceState,
    ) -> None:
        action = ActiveSpaceAction(
            action_type=ActiveSpaceActionType.REMOVE_ORBITAL,
            orbital_index=1,
        )
        new_state = selector.apply_action(active_space_state, action)

        assert 1 not in new_state.active_indices
        assert new_state.num_active == active_space_state.num_active - 1
        assert new_state.step == active_space_state.step + 1
        # Original unchanged
        assert 1 in active_space_state.active_indices


class TestActiveSpaceSwapOrbital:
    """apply_action with SWAP_ORBITAL replaces an active with an inactive."""

    def test_active_space_swap_orbital(
        self,
        selector: ActiveSpaceSelector,
        active_space_state: ActiveSpaceState,
    ) -> None:
        # Swap active orbital 0 with inactive orbital 7
        action = ActiveSpaceAction(
            action_type=ActiveSpaceActionType.SWAP_ORBITAL,
            orbital_index=0,
            swap_target=7,
        )
        new_state = selector.apply_action(active_space_state, action)

        assert 0 not in new_state.active_indices
        assert 7 in new_state.active_indices
        assert new_state.num_active == active_space_state.num_active
        assert new_state.step == active_space_state.step + 1
        # Original unchanged
        assert 0 in active_space_state.active_indices
        assert 7 not in active_space_state.active_indices


class TestActiveSpaceSelectorReturnsIndices:
    """ActiveSpaceSelector.select_active_space returns a list of indices."""

    def test_active_space_selector_returns_indices(
        self,
        selector: ActiveSpaceSelector,
        orbitals: list[OrbitalInfo],
    ) -> None:
        result = selector.select_active_space(
            orbitals=orbitals,
            num_electrons=4,
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= selector._max_active

        # All indices must be valid orbital indices
        valid_indices = {o.index for o in orbitals}
        for idx in result:
            assert idx in valid_indices

        # No duplicates
        assert len(result) == len(set(result))


class TestActiveSpaceScoreEntropy:
    """_score_configuration uses entropy and mutual information."""

    def test_active_space_score_entropy(
        self,
        selector: ActiveSpaceSelector,
        active_space_state: ActiveSpaceState,
    ) -> None:
        score_with_orbitals = selector._score_configuration(active_space_state)

        # Score should be positive (entropy and mutual info are positive)
        assert score_with_orbitals > 0.0

        # Empty active space scores zero
        empty_state = active_space_state.clone()
        empty_state.active_indices = []
        score_empty = selector._score_configuration(empty_state)
        assert score_empty == 0.0

        # More active orbitals with high entropy => higher score
        more_active = active_space_state.clone()
        more_active.active_indices = [0, 1, 2, 3, 4]
        score_more = selector._score_configuration(more_active)
        assert score_more > score_with_orbitals


# ------------------------------------------------------------------
# DMRG Orbital Ordering Tests
# ------------------------------------------------------------------


class TestDMRGStateClone:
    """DMRGOrderingState.clone produces an independent copy."""

    def test_dmrg_state_clone(self, dmrg_state: DMRGOrderingState) -> None:
        cloned = dmrg_state.clone()

        # Must be a different object
        assert cloned is not dmrg_state
        assert cloned.ordering is not dmrg_state.ordering
        assert cloned.entanglement_matrix is not dmrg_state.entanglement_matrix

        # Values must match
        assert cloned.ordering == dmrg_state.ordering
        np.testing.assert_array_equal(
            cloned.entanglement_matrix,
            dmrg_state.entanglement_matrix,
        )
        assert cloned.step == dmrg_state.step

    def test_clone_mutation_independence(
        self,
        dmrg_state: DMRGOrderingState,
    ) -> None:
        cloned = dmrg_state.clone()
        cloned.ordering[0] = 99
        cloned.entanglement_matrix[0, 0] = -999.0

        assert dmrg_state.ordering[0] != 99
        assert dmrg_state.entanglement_matrix[0, 0] != -999.0


class TestDMRGLinearEntropy:
    """DMRGOrderingState.linear_entropy computes cross-cut entanglement."""

    def test_dmrg_linear_entropy(
        self,
        entanglement_matrix: np.ndarray,
    ) -> None:
        # With orbitals [0,1,2,3,4,5], compute expected entropy
        state = DMRGOrderingState(
            ordering=[0, 1, 2, 3, 4, 5],
            entanglement_matrix=entanglement_matrix,
        )
        entropy = state.linear_entropy
        assert entropy > 0.0
        assert np.isfinite(entropy)

    def test_single_orbital_zero_entropy(self) -> None:
        """A single orbital has zero linear entropy."""
        mat = np.zeros((1, 1))
        state = DMRGOrderingState(ordering=[0], entanglement_matrix=mat)
        assert state.linear_entropy == 0.0

    def test_ordering_affects_entropy(
        self,
        entanglement_matrix: np.ndarray,
    ) -> None:
        """Different orderings produce different entropies."""
        state_a = DMRGOrderingState(
            ordering=[0, 1, 2, 3, 4, 5],
            entanglement_matrix=entanglement_matrix,
        )
        state_b = DMRGOrderingState(
            ordering=[0, 2, 4, 1, 3, 5],
            entanglement_matrix=entanglement_matrix,
        )
        # The structured matrix means different orderings have different entropy
        assert state_a.linear_entropy != pytest.approx(state_b.linear_entropy)


class TestDMRGSwapAdjacent:
    """apply_action with SWAP_ADJACENT swaps two neighbouring orbitals."""

    def test_dmrg_swap_adjacent(
        self,
        dmrg_optimizer: DMRGOrderingOptimizer,
        dmrg_state: DMRGOrderingState,
    ) -> None:
        action = OrderingAction(
            action_type=OrderingActionType.SWAP_ADJACENT,
            position_a=1,
            position_b=2,
        )
        new_state = dmrg_optimizer.apply_action(dmrg_state, action)

        assert new_state.ordering[1] == dmrg_state.ordering[2]
        assert new_state.ordering[2] == dmrg_state.ordering[1]
        # Other positions unchanged
        assert new_state.ordering[0] == dmrg_state.ordering[0]
        assert new_state.ordering[3] == dmrg_state.ordering[3]
        assert new_state.step == dmrg_state.step + 1
        # Original unchanged
        assert dmrg_state.ordering == [0, 1, 2, 3, 4, 5]


class TestDMRGSwapAny:
    """apply_action with SWAP_ANY swaps two non-adjacent orbitals."""

    def test_dmrg_swap_any(
        self,
        dmrg_optimizer: DMRGOrderingOptimizer,
        dmrg_state: DMRGOrderingState,
    ) -> None:
        action = OrderingAction(
            action_type=OrderingActionType.SWAP_ANY,
            position_a=0,
            position_b=5,
        )
        new_state = dmrg_optimizer.apply_action(dmrg_state, action)

        assert new_state.ordering[0] == 5
        assert new_state.ordering[5] == 0
        # Middle positions unchanged
        assert new_state.ordering[1:5] == [1, 2, 3, 4]
        assert new_state.step == dmrg_state.step + 1
        # Original unchanged
        assert dmrg_state.ordering == [0, 1, 2, 3, 4, 5]


class TestDMRGReverseSegment:
    """apply_action with REVERSE_SEGMENT reverses a contiguous slice."""

    def test_dmrg_reverse_segment(
        self,
        dmrg_optimizer: DMRGOrderingOptimizer,
        dmrg_state: DMRGOrderingState,
    ) -> None:
        # Reverse positions 1 through 4 (indices 1,2,3,4)
        action = OrderingAction(
            action_type=OrderingActionType.REVERSE_SEGMENT,
            position_a=1,
            position_b=4,
        )
        new_state = dmrg_optimizer.apply_action(dmrg_state, action)

        assert new_state.ordering == [0, 4, 3, 2, 1, 5]
        assert new_state.step == dmrg_state.step + 1
        # Original unchanged
        assert dmrg_state.ordering == [0, 1, 2, 3, 4, 5]


class TestDMRGOptimizerReturnsOrdering:
    """DMRGOrderingOptimizer.optimize_ordering returns a valid permutation."""

    def test_dmrg_optimizer_returns_ordering(
        self,
        dmrg_optimizer: DMRGOrderingOptimizer,
        entanglement_matrix: np.ndarray,
    ) -> None:
        result = dmrg_optimizer.optimize_ordering(
            entanglement_matrix=entanglement_matrix,
            max_steps=3,
        )

        assert isinstance(result, list)
        assert len(result) == entanglement_matrix.shape[0]
        # Must be a permutation
        assert sorted(result) == list(range(entanglement_matrix.shape[0]))


class TestDMRGOptimizerImprovesEntropy:
    """DMRGOrderingOptimizer finds an ordering with lower entropy."""

    def test_dmrg_optimizer_improves_entropy(self) -> None:
        # Build a matrix where the optimal ordering is obvious:
        # Strong coupling between (0,1) and (2,3), weak elsewhere.
        n = 4
        mat = np.full((n, n), 0.01)
        np.fill_diagonal(mat, 0.0)
        mat[0, 1] = mat[1, 0] = 5.0
        mat[2, 3] = mat[3, 2] = 5.0

        # Start with a bad ordering: separate coupled pairs
        bad_ordering = [0, 2, 1, 3]
        bad_state = DMRGOrderingState(
            ordering=list(bad_ordering),
            entanglement_matrix=mat,
        )
        bad_entropy = bad_state.linear_entropy

        optimizer = DMRGOrderingOptimizer(
            num_simulations=1,
            max_segment_length=4,
        )
        result = optimizer.optimize_ordering(
            entanglement_matrix=mat,
            initial_ordering=bad_ordering,
            max_steps=10,
        )

        result_state = DMRGOrderingState(
            ordering=result,
            entanglement_matrix=mat,
        )
        result_entropy = result_state.linear_entropy

        # The optimizer should find a better (or equal) ordering
        assert result_entropy <= bad_entropy
