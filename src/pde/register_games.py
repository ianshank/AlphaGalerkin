"""Register PDE games in the GameRegistry.

Importing this module triggers registration of all PDE game variants
so they can be discovered via ``GameRegistry().get("pde_basis")`` etc.

Usage::

    # Register PDE games
    import src.pde.register_games  # noqa: F401

    # Or use via GameRegistry
    from src.games.registry import GameRegistry
    game = GameRegistry().get("pde_basis")
"""

from __future__ import annotations

from typing import Literal

import structlog

from src.games.registry import register_game
from src.pde.config import PDEConfig, PDEGameConfig, PDEType
from src.pde.game_interface import PDEGameInterface
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.games.mesh_refinement import MeshRefinementGame
from src.pde.operators import PDEOperator
from src.pde.registry import PDEOperatorRegistry

logger = structlog.get_logger(__name__)


def _create_default_pde_config(pde_type: PDEType = PDEType.POISSON) -> PDEConfig:
    """Create a default PDE config (no hardcoded values — all from Pydantic defaults)."""
    return PDEConfig(name="default", pde_type=pde_type)


def _create_default_game_config(
    game_mode: Literal["basis_selection", "mesh_refinement", "collocation"] = "basis_selection",
    pde_type: PDEType = PDEType.POISSON,
) -> PDEGameConfig:
    """Create a default PDE game config."""
    pde_config = _create_default_pde_config(pde_type)
    return PDEGameConfig(name=f"default_{game_mode}", pde_config=pde_config, game_mode=game_mode)


def _create_operator(pde_type: PDEType, pde_config: PDEConfig) -> PDEOperator:
    """Create a PDE operator from the registry by type.

    Uses the PDEOperatorRegistry so new operator types are automatically
    supported without modifying this file.
    """
    registry = PDEOperatorRegistry()
    operator_cls = registry.get(pde_type.value)
    return operator_cls(pde_config)


@register_game("pde_basis")
class PDEBasisSelectionInterface(PDEGameInterface):
    """PDE basis selection game registered as a GameInterface.

    Uses Poisson operator by default. Override via config for other PDEs.
    """

    name = "pde_basis"
    description = "MCTS-guided Galerkin basis selection for PDE solving"

    def __init__(self, pde_type: PDEType = PDEType.POISSON) -> None:
        """Initialize with configurable PDE operator.

        Args:
            pde_type: Which PDE operator to use (defaults to Poisson).

        """
        pde_config = _create_default_pde_config(pde_type)
        game_config = _create_default_game_config("basis_selection", pde_type)
        operator = _create_operator(pde_type, pde_config)
        pde_game = BasisSelectionGame(operator, game_config)
        super().__init__(pde_game=pde_game)
        logger.info("pde_basis_game_registered", pde_type=pde_type.value)


@register_game("pde_mesh")
class PDEMeshRefinementInterface(PDEGameInterface):
    """PDE mesh refinement game registered as a GameInterface.

    Uses Poisson operator by default for AMR demonstration.
    """

    name = "pde_mesh"
    description = "MCTS-guided adaptive mesh refinement for PDE solving"

    def __init__(self, pde_type: PDEType = PDEType.POISSON) -> None:
        """Initialize with configurable PDE operator.

        Args:
            pde_type: Which PDE operator to use (defaults to Poisson).

        """
        pde_config = _create_default_pde_config(pde_type)
        game_config = _create_default_game_config("mesh_refinement", pde_type)
        operator = _create_operator(pde_type, pde_config)
        pde_game = MeshRefinementGame(operator, game_config)
        super().__init__(pde_game=pde_game)
        logger.info("pde_mesh_game_registered", pde_type=pde_type.value)
