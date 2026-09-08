"""First non-test registry lookup path for substrates."""

from __future__ import annotations

import src.pde.register_refinement_games  # noqa: F401
from src.refinement.substrate_registry import RefinementSubstrateRegistry
from src.research.substrates.config import (
    SUBSTRATE_KIND_TENSOR_GRID,
    SubstrateConfig,
)
from src.research.substrates.factory import (
    build_substrate_from_config,
    ensure_substrate_registrants,
)


def test_ensure_registrants_populates_registry() -> None:
    ensure_substrate_registrants()
    registry = RefinementSubstrateRegistry()
    assert registry.get(SUBSTRATE_KIND_TENSOR_GRID) is not None


def test_build_substrate_uses_registry_lookup() -> None:
    ensure_substrate_registrants()
    config = SubstrateConfig(
        name="factory_lookup",
        kind="tensor_grid",
        initial_side=4,
    )
    substrate = build_substrate_from_config(config, operator_name="poisson")
    mesh = substrate.initial_mesh()
    result = substrate.solve(mesh)
    assert result.n_dof > 0
    assert result.l2_error >= 0.0
