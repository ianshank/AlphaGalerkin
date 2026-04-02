"""Tests for src/pde/game_interface.py — PDEGameInterface wrapper.

Validates that PDE games can be used through the GameInterface protocol,
enabling registration in GameRegistry and use with the standard trainer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.games.interface import GamePhase, GameResult
from src.games.state import GameState
from src.pde.config import PDEConfig, PDEGameConfig, PDEType
from src.pde.game_interface import PDEGameInterface, PDEGameInterfaceConfig
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.operators import PoissonOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def poisson_operator() -> PoissonOperator:
    cfg = PDEConfig(
        name="test_poisson",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        advection_coeff=[0.0, 0.0],
    )
    return PoissonOperator(cfg)


@pytest.fixture()
def basis_game(poisson_operator: PoissonOperator) -> BasisSelectionGame:
    game_config = PDEGameConfig(
        name="test_basis",
        pde_config=PDEConfig(
            name="test_poisson",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            advection_coeff=[0.0, 0.0],
        ),
        game_mode="basis_selection",
    )
    return BasisSelectionGame(poisson_operator, game_config)


@pytest.fixture()
def pde_interface(basis_game: BasisSelectionGame) -> PDEGameInterface:
    return PDEGameInterface(pde_game=basis_game, grid_size=8)


# ---------------------------------------------------------------------------
# Construction and Properties
# ---------------------------------------------------------------------------


class TestPDEGameInterfaceConstruction:
    def test_action_space_size_positive(self, pde_interface: PDEGameInterface):
        assert pde_interface.action_space_size > 0

    def test_state_channels_positive(self, pde_interface: PDEGameInterface):
        assert pde_interface.state_channels > 0

    def test_n_players_is_one(self, pde_interface: PDEGameInterface):
        assert pde_interface.n_players == 1

    def test_grid_size_stored(self, pde_interface: PDEGameInterface):
        assert pde_interface.grid_size == 8


# ---------------------------------------------------------------------------
# GameInterface Protocol Compliance
# ---------------------------------------------------------------------------


class TestGameInterfaceProtocol:
    def test_initial_state_returns_game_state(self, pde_interface: PDEGameInterface):
        state = pde_interface.initial_state()
        assert isinstance(state, GameState)

    def test_initial_state_has_board(self, pde_interface: PDEGameInterface):
        state = pde_interface.initial_state()
        assert state.board is not None
        assert isinstance(state.board, np.ndarray)

    def test_initial_state_move_number_zero(self, pde_interface: PDEGameInterface):
        state = pde_interface.initial_state()
        assert state.move_number == 0

    def test_initial_state_metadata_has_pde_state(self, pde_interface: PDEGameInterface):
        state = pde_interface.initial_state()
        assert "_pde_state" in state.metadata
        assert "error_estimate" in state.metadata

    def test_get_legal_actions_non_empty(self, pde_interface: PDEGameInterface):
        state = pde_interface.initial_state()
        actions = pde_interface.get_legal_actions(state)
        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_get_action_mask_correct_size(self, pde_interface: PDEGameInterface):
        state = pde_interface.initial_state()
        mask = pde_interface.get_action_mask(state)
        assert mask.action_space_size == pde_interface.action_space_size
        assert mask.mask.dtype == bool

    def test_apply_action_returns_new_state(self, pde_interface: PDEGameInterface):
        state = pde_interface.initial_state()
        actions = pde_interface.get_legal_actions(state)
        new_state = pde_interface.apply_action(state, actions[0])
        assert isinstance(new_state, GameState)
        assert new_state.move_number == 1
        assert actions[0] in new_state.move_history

    def test_is_terminal_initial_state_false(self, pde_interface: PDEGameInterface):
        state = pde_interface.initial_state()
        assert not pde_interface.is_terminal(state)

    def test_to_tensor_returns_tensor(self, pde_interface: PDEGameInterface):
        state = pde_interface.initial_state()
        tensor = pde_interface.to_tensor(state)
        assert isinstance(tensor, torch.Tensor)

    def test_get_symmetries_identity(self, pde_interface: PDEGameInterface):
        state = pde_interface.initial_state()
        policy = np.ones(pde_interface.action_space_size, dtype=np.float32)
        syms = pde_interface.get_symmetries(state, policy)
        assert len(syms) == 1
        assert syms[0][0] is state

    def test_get_phase_not_terminal_initially(self, pde_interface: PDEGameInterface):
        state = pde_interface.initial_state()
        phase = pde_interface.get_phase(state)
        assert phase != GamePhase.TERMINAL


# ---------------------------------------------------------------------------
# Multi-step Game Play
# ---------------------------------------------------------------------------


class TestMultiStepPlay:
    def test_play_sequence(self, pde_interface: PDEGameInterface):
        """Play a few steps and verify state evolution."""
        state = pde_interface.initial_state()
        for step in range(3):
            if pde_interface.is_terminal(state):
                break
            actions = pde_interface.get_legal_actions(state)
            state = pde_interface.apply_action(state, actions[0])
            assert state.move_number == step + 1

    def test_error_tracked_in_metadata(self, pde_interface: PDEGameInterface):
        """Error estimate should be in metadata after each action."""
        state = pde_interface.initial_state()
        initial_error = state.metadata["error_estimate"]
        assert initial_error > 0

        actions = pde_interface.get_legal_actions(state)
        state = pde_interface.apply_action(state, actions[0])
        assert "error_estimate" in state.metadata

    def test_get_result_on_terminal(self, pde_interface: PDEGameInterface):
        """If we can reach terminal, get_result should work."""
        state = pde_interface.initial_state()
        for _ in range(50):
            if pde_interface.is_terminal(state):
                result = pde_interface.get_result(state)
                assert isinstance(result, GameResult)
                assert result.move_count >= 0
                return
            actions = pde_interface.get_legal_actions(state)
            if not actions:
                break
            state = pde_interface.apply_action(state, actions[0])
        # If we didn't reach terminal, that's OK — just verify get_result doesn't crash
        result = pde_interface.get_result(state)
        assert isinstance(result, GameResult)


# ---------------------------------------------------------------------------
# State Conversion
# ---------------------------------------------------------------------------


class TestStateConversion:
    def test_roundtrip_pde_state(self, pde_interface: PDEGameInterface):
        """PDEState -> GameState -> PDEState should be lossless."""
        state = pde_interface.initial_state()
        pde_state = pde_interface._game_to_pde_state(state)
        assert pde_state.error_estimate == state.metadata["error_estimate"]

    def test_invalid_game_state_raises(self, pde_interface: PDEGameInterface):
        """GameState without PDE metadata should raise ValueError."""
        fake_state = GameState(board=np.zeros((3, 3)))
        with pytest.raises(ValueError, match="does not contain a PDEState"):
            pde_interface._game_to_pde_state(fake_state)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestPDEGameRegistration:
    def test_register_games_importable(self):
        """Importing register_games should not crash."""
        import src.pde.register_games  # noqa: F401

    def test_pde_basis_in_registry(self):
        """pde_basis should be registered after import."""
        import src.pde.register_games  # noqa: F401
        from src.games.registry import GameRegistry

        registry = GameRegistry()
        assert "pde_basis" in registry.list_games()

    def test_pde_mesh_in_registry(self):
        """pde_mesh should be registered after import."""
        import src.pde.register_games  # noqa: F401
        from src.games.registry import GameRegistry

        registry = GameRegistry()
        assert "pde_mesh" in registry.list_games()

    def test_get_pde_basis_game(self):
        """Should be able to instantiate pde_basis from registry."""
        import src.pde.register_games  # noqa: F401
        from src.games.registry import GameRegistry

        game = GameRegistry().get("pde_basis")
        assert game is not None
        assert game.action_space_size > 0

    def test_get_pde_mesh_game(self):
        """Should be able to instantiate pde_mesh from registry."""
        import src.pde.register_games  # noqa: F401
        from src.games.registry import GameRegistry

        game = GameRegistry().get("pde_mesh")
        assert game is not None
        assert game.action_space_size > 0

    def test_pde_basis_custom_pde_type(self):
        """PDEBasisSelectionInterface should accept custom PDE type."""
        from src.pde.register_games import PDEBasisSelectionInterface

        game = PDEBasisSelectionInterface(pde_type=PDEType.POISSON)
        assert game.action_space_size > 0

    def test_pde_mesh_custom_pde_type(self):
        """PDEMeshRefinementInterface should accept custom PDE type."""
        from src.pde.register_games import PDEMeshRefinementInterface

        game = PDEMeshRefinementInterface(pde_type=PDEType.POISSON)
        assert game.action_space_size > 0


# ---------------------------------------------------------------------------
# Winner / Phase Logic
# ---------------------------------------------------------------------------


class TestWinnerAndPhaseLogic:
    def test_get_winner_non_terminal(self, pde_interface: PDEGameInterface):
        """Winner should be None or an int for non-terminal state."""
        state = pde_interface.initial_state()
        winner = pde_interface.get_winner(state)
        # Non-terminal: could be None (ambiguous) or -1 (initial error high)
        assert winner is None or isinstance(winner, int)

    def test_get_winner_with_convergence(self, basis_game: BasisSelectionGame):
        """Winner should be +1 when error < tolerance."""
        config = PDEGameInterfaceConfig(default_tolerance=999.0)
        interface = PDEGameInterface(pde_game=basis_game, interface_config=config)
        state = interface.initial_state()
        winner = interface.get_winner(state)
        assert winner == 1  # Error < very large tolerance

    def test_get_winner_failure(self, basis_game: BasisSelectionGame):
        """Winner should be -1 when error reduction is insufficient."""
        config = PDEGameInterfaceConfig(
            default_tolerance=1e-20,
            convergence_reduction=1e-15,
            failure_reduction=0.0001,
        )
        interface = PDEGameInterface(pde_game=basis_game, interface_config=config)
        state = interface.initial_state()
        # Inject initial_error so ratio > failure_reduction
        state.metadata["_initial_error"] = state.metadata["error_estimate"] * 0.9
        winner = interface.get_winner(state)
        assert winner == -1

    def test_get_phase_opening(self, basis_game: BasisSelectionGame):
        """Phase should be OPENING when error >> tolerance."""
        config = PDEGameInterfaceConfig(
            default_tolerance=1e-20,
            phase_opening_multiplier=2.0,
        )
        interface = PDEGameInterface(pde_game=basis_game, interface_config=config)
        state = interface.initial_state()
        phase = interface.get_phase(state)
        assert phase == GamePhase.OPENING

    def test_get_phase_midgame(self, basis_game: BasisSelectionGame):
        """Phase should be MIDGAME when tolerance < error < tolerance * multiplier."""
        state_init = basis_game.get_initial_state()
        error = state_init.error_estimate
        # Set tolerance so: tolerance < error < tolerance * multiplier
        config = PDEGameInterfaceConfig(
            default_tolerance=error * 0.5,
            phase_opening_multiplier=10.0,
        )
        interface = PDEGameInterface(pde_game=basis_game, interface_config=config)
        state = interface.initial_state()
        phase = interface.get_phase(state)
        assert phase == GamePhase.MIDGAME

    def test_get_phase_endgame(self, basis_game: BasisSelectionGame):
        """Phase should be ENDGAME when error < tolerance."""
        config = PDEGameInterfaceConfig(default_tolerance=999.0)
        interface = PDEGameInterface(pde_game=basis_game, interface_config=config)
        state = interface.initial_state()
        phase = interface.get_phase(state)
        assert phase == GamePhase.ENDGAME

    def test_action_mask_with_torch_tensor(self, pde_interface: PDEGameInterface):
        """Verify action mask works when underlying returns Tensor."""
        state = pde_interface.initial_state()
        mask = pde_interface.get_action_mask(state)
        assert mask.mask.dtype == bool
        assert any(mask.mask)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestPDEGameInterfaceConfig:
    def test_default_config(self):
        config = PDEGameInterfaceConfig()
        assert config.default_tolerance == 0.01
        assert config.phase_opening_multiplier == 10.0
        assert config.convergence_reduction == 0.1
        assert config.failure_reduction == 0.5

    def test_custom_config(self):
        config = PDEGameInterfaceConfig(
            default_tolerance=0.05,
            phase_opening_multiplier=5.0,
            convergence_reduction=0.2,
            failure_reduction=0.6,
        )
        assert config.default_tolerance == 0.05
        assert config.phase_opening_multiplier == 5.0

    def test_invalid_config_rejects_bad_values(self):
        with pytest.raises(Exception):
            PDEGameInterfaceConfig(default_tolerance=-1.0)
