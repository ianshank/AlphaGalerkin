"""Tests for the BasisSelectionGame wired to SDF-aware helical operators.

This is the v2.2 expansion item: MCTS-guided Galerkin basis selection
running natively on a Leap 71 helical SDF rather than the rectangular
domain assumed by the original ``pde_basis`` registration.
"""

from __future__ import annotations

import pytest

import src.pde.register_games  # noqa: F401 - triggers @register_game decorators
from src.games.registry import GameRegistry
from src.pde.game_interface import PDEGameInterface
from src.pde.geometry import GeometryType
from src.pde.register_games import (
    HELICAL_OPERATOR_NAMES,
    HelicalBasisSelectionInterface,
)


class TestHelicalBasisSelectionInterface:
    def test_registered_in_game_registry(self) -> None:
        assert "pde_basis_helical" in GameRegistry().list_games()

    def test_default_constructor_uses_helical_heat(self) -> None:
        instance = HelicalBasisSelectionInterface()
        assert isinstance(instance, PDEGameInterface)
        # Underlying operator must be registered as a helical SDF operator.
        config = instance.pde_game.pde_operator.config
        assert config.geometry.geometry_type == GeometryType.PICOGK
        assert config.geometry.sdf_kind == "analytical_helix"

    @pytest.mark.parametrize("operator_name", list(HELICAL_OPERATOR_NAMES))
    def test_constructs_for_each_helical_operator(self, operator_name: str) -> None:
        instance = HelicalBasisSelectionInterface(operator_name=operator_name)
        config = instance.pde_game.pde_operator.config
        assert config.domain_dim == 3
        assert config.geometry.geometry_type == GeometryType.PICOGK

    def test_unknown_operator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported helical operator"):
            HelicalBasisSelectionInterface(operator_name="not_a_helical_op")

    def test_helix_param_overrides_propagate(self) -> None:
        # Sanity-check that the helper's defaults are used when the
        # caller doesn't override; covers the default-args path.
        instance = HelicalBasisSelectionInterface()
        cfg = instance.pde_game.pde_operator.config
        assert cfg.geometry.helix_n_turns >= 1
        assert cfg.geometry.helix_R_major > cfg.geometry.helix_r_minor
