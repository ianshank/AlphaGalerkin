"""Tests for PDE game registration in GameRegistry.

Validates that ``pde_basis`` and ``pde_mesh`` are discoverable and
produce valid ``GameInterface`` instances once ``src.pde.register_games``
has been imported.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import src.pde.register_games  # noqa: F401  — ensure PDE games are registered
from src.games.interface import GameInterface
from src.games.registry import GameRegistry
from src.games.state import GameState
from src.pde.config import PDEType

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestPDEGameDiscovery:
    """Verify PDE games are listed and retrievable by name."""

    def test_pde_basis_listed(self) -> None:
        """pde_basis should appear in GameRegistry.list_games()."""
        names = GameRegistry().list_games()
        assert "pde_basis" in names

    def test_pde_mesh_listed(self) -> None:
        """pde_mesh should appear in GameRegistry.list_games()."""
        names = GameRegistry().list_games()
        assert "pde_mesh" in names

    def test_get_pde_basis_returns_game(self) -> None:
        """GameRegistry().get('pde_basis') should return a GameInterface."""
        game = GameRegistry().get("pde_basis")
        assert game is not None
        assert isinstance(game, GameInterface)

    def test_get_pde_mesh_returns_game(self) -> None:
        """GameRegistry().get('pde_mesh') should return a GameInterface."""
        game = GameRegistry().get("pde_mesh")
        assert game is not None
        assert isinstance(game, GameInterface)

    def test_get_class_pde_basis(self) -> None:
        """get_class should return the class, not an instance."""
        cls = GameRegistry().get_class("pde_basis")
        assert cls is not None
        assert isinstance(cls, type)

    def test_is_registered_pde_basis(self) -> None:
        """is_registered should return True for pde_basis."""
        assert GameRegistry().is_registered("pde_basis")

    def test_is_registered_pde_mesh(self) -> None:
        """is_registered should return True for pde_mesh."""
        assert GameRegistry().is_registered("pde_mesh")

    def test_get_info_pde_basis(self) -> None:
        """get_info should return metadata for pde_basis."""
        info = GameRegistry().get_info("pde_basis")
        assert info is not None
        assert info["name"] == "pde_basis"
        assert info["action_space_size"] > 0
        assert info["state_channels"] > 0


# ---------------------------------------------------------------------------
# Adapter validity
# ---------------------------------------------------------------------------


class TestAdapterCreatesValidInstances:
    """Verify that registered games produce valid game instances."""

    @pytest.fixture()
    def basis_game(self) -> GameInterface:
        game = GameRegistry().get("pde_basis")
        assert game is not None
        return game

    @pytest.fixture()
    def mesh_game(self) -> GameInterface:
        game = GameRegistry().get("pde_mesh")
        assert game is not None
        return game

    def test_basis_initial_state(self, basis_game: GameInterface) -> None:
        """Initial state should be a valid GameState."""
        state = basis_game.initial_state()
        assert isinstance(state, GameState)
        assert state.move_number == 0

    def test_mesh_initial_state(self, mesh_game: GameInterface) -> None:
        """Mesh game should produce a valid initial state."""
        state = mesh_game.initial_state()
        assert isinstance(state, GameState)

    def test_basis_legal_actions(self, basis_game: GameInterface) -> None:
        """Legal actions should be a non-empty list of ints."""
        state = basis_game.initial_state()
        actions = basis_game.get_legal_actions(state)
        assert len(actions) > 0
        assert all(isinstance(a, int) for a in actions)

    def test_basis_to_tensor(self, basis_game: GameInterface) -> None:
        """to_tensor should return a finite torch.Tensor."""
        state = basis_game.initial_state()
        tensor = basis_game.to_tensor(state)
        assert isinstance(tensor, torch.Tensor)
        assert torch.isfinite(tensor).all()

    def test_basis_apply_action(self, basis_game: GameInterface) -> None:
        """apply_action should advance the move number."""
        state = basis_game.initial_state()
        actions = basis_game.get_legal_actions(state)
        new_state = basis_game.apply_action(state, actions[0])
        assert new_state.move_number == 1

    def test_basis_is_terminal_false_initially(self, basis_game: GameInterface) -> None:
        """Initial state should not be terminal."""
        state = basis_game.initial_state()
        assert not basis_game.is_terminal(state)

    def test_basis_n_players_is_one(self, basis_game: GameInterface) -> None:
        """PDE games are single-player."""
        assert basis_game.n_players == 1

    def test_basis_get_symmetries_identity(self, basis_game: GameInterface) -> None:
        """Default symmetries should return only the identity."""
        state = basis_game.initial_state()
        policy = np.ones(basis_game.action_space_size, dtype=np.float32)
        syms = basis_game.get_symmetries(state, policy)
        assert len(syms) == 1


# ---------------------------------------------------------------------------
# Custom PDE type
# ---------------------------------------------------------------------------


class TestCustomPDEType:
    """Verify PDE games can be instantiated with non-default PDE types."""

    def test_basis_with_explicit_poisson(self) -> None:
        from src.pde.register_games import PDEBasisSelectionInterface

        game = PDEBasisSelectionInterface(pde_type=PDEType.POISSON)
        assert game.action_space_size > 0
        state = game.initial_state()
        assert isinstance(state, GameState)

    def test_mesh_with_explicit_poisson(self) -> None:
        from src.pde.register_games import PDEMeshRefinementInterface

        game = PDEMeshRefinementInterface(pde_type=PDEType.POISSON)
        assert game.action_space_size > 0
        state = game.initial_state()
        assert isinstance(state, GameState)


# ---------------------------------------------------------------------------
# Auto-registration via src.pde import
# ---------------------------------------------------------------------------


class TestAutoRegistration:
    """Verify that importing src.pde triggers registration."""

    def test_import_pde_registers_games(self) -> None:
        """Importing src.pde should register pde_basis and pde_mesh."""
        import importlib

        importlib.reload(__import__("src.pde", fromlist=["register_games"]))
        registry = GameRegistry()
        assert registry.is_registered("pde_basis")
        assert registry.is_registered("pde_mesh")
