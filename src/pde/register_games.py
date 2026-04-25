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
from src.pde.geometry import GeometryConfig, GeometryType
from src.pde.operators import PDEOperator
from src.pde.registry import PDEOperatorRegistry

logger = structlog.get_logger(__name__)

# Names registered in PDEOperatorRegistry that target SDF-bounded helical
# Leap 71 / Noyron geometries. Kept here (rather than hardcoded inline)
# so future helical operators register in exactly one place.
HELICAL_OPERATOR_NAMES: tuple[str, ...] = (
    "helical_heat",
    "helical_stokes",
    "helical_magnetostatics",
)


def _create_default_pde_config(pde_type: PDEType = PDEType.POISSON) -> PDEConfig:
    """Create a default PDE config (no hardcoded values — all from Pydantic defaults)."""
    return PDEConfig(name="default", pde_type=pde_type)


def _create_helical_pde_config(
    operator_name: str,
    helix_R_major: float = 0.05,  # noqa: N803 - matches GeometryConfig naming
    helix_r_minor: float = 0.012,
    helix_pitch: float = 0.02,
    helix_n_turns: int = 3,
) -> PDEConfig:
    """Create a 3D PicoGK-backed PDE config tuned for helical operators.

    Used by ``HelicalBasisSelectionInterface`` to spin up a default
    config compatible with ``HelicalHeatOperator``,
    ``HelicalStokesOperator`` and ``HelicalMagnetostaticsOperator``.
    """
    z_max = helix_pitch * helix_n_turns
    outer = helix_R_major + helix_r_minor
    pde_type = PDEType.HEAT if operator_name == "helical_heat" else PDEType.POISSON
    if operator_name == "helical_stokes":
        pde_type = PDEType.NAVIER_STOKES
    return PDEConfig(
        name=f"default_{operator_name}",
        pde_type=pde_type,
        domain_dim=3,
        domain_min=[-outer, -outer, 0.0],
        domain_max=[outer, outer, z_max],
        advection_coeff=[0.0, 0.0, 0.0],
        geometry=GeometryConfig(
            geometry_type=GeometryType.PICOGK,
            sdf_kind="analytical_helix",
            helix_R_major=helix_R_major,
            helix_r_minor=helix_r_minor,
            helix_pitch=helix_pitch,
            helix_n_turns=helix_n_turns,
        ),
    )


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
        ----
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
        ----
            pde_type: Which PDE operator to use (defaults to Poisson).

        """
        pde_config = _create_default_pde_config(pde_type)
        game_config = _create_default_game_config("mesh_refinement", pde_type)
        operator = _create_operator(pde_type, pde_config)
        pde_game = MeshRefinementGame(operator, game_config)
        super().__init__(pde_game=pde_game)
        logger.info("pde_mesh_game_registered", pde_type=pde_type.value)


@register_game("pde_basis_helical")
class HelicalBasisSelectionInterface(PDEGameInterface):
    """BasisSelectionGame wired to an SDF-aware helical PDE operator.

    Defaults to ``helical_heat`` (Noyron HX) but accepts any registered
    operator name in :data:`HELICAL_OPERATOR_NAMES` (currently
    ``helical_heat`` / ``helical_stokes`` / ``helical_magnetostatics``).

    The collocation/boundary samplers delegate to the underlying
    ``PicoGKDomain`` so MCTS basis selection runs natively on Leap 71
    geometries — the v2.2 expansion item from the integration plan.
    """

    name = "pde_basis_helical"
    description = "MCTS-guided Galerkin basis selection on a helical SDF (Leap 71 Noyron HX/RP/EA)."

    def __init__(self, operator_name: str = "helical_heat") -> None:
        if operator_name not in HELICAL_OPERATOR_NAMES:
            raise ValueError(
                f"Unsupported helical operator '{operator_name}'. "
                f"Expected one of {HELICAL_OPERATOR_NAMES}."
            )

        pde_config = _create_helical_pde_config(operator_name)
        # We deliberately reuse PDEGameConfig with game_mode='basis_selection';
        # the helical geometry is carried by PDEConfig.geometry.
        game_config = PDEGameConfig(
            name=f"default_{operator_name}_basis",
            pde_config=pde_config,
            game_mode="basis_selection",
        )
        operator_cls = PDEOperatorRegistry().get_or_raise(operator_name)
        operator = operator_cls(pde_config)
        pde_game = BasisSelectionGame(operator, game_config)
        super().__init__(pde_game=pde_game)
        logger.info(
            "helical_basis_game_registered",
            operator_name=operator_name,
            sdf_kind=pde_config.geometry.sdf_kind,
        )
