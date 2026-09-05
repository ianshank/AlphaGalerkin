"""Config-driven ``RefinementSubstrate`` construction via the substrate registry.

This is the first non-test *lookup* of ``RefinementSubstrateRegistry``: callers
pass a ``SubstrateConfig.kind`` and receive a concrete substrate without
importing ``tensor_grid`` / ``skfem_tri`` directly at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import structlog

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import LShapedPoissonOperator, PoissonOperator
from src.refinement.substrate_registry import RefinementSubstrateRegistry
from src.research.lshape_amr_compare import lshape_inside_predicate
from src.research.substrates.config import (
    SUBSTRATE_KIND_TENSOR_GRID,
    SubstrateConfig,
)

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)

OperatorName = Literal["poisson", "lshape_poisson"]

# Side-effect imports: register concrete substrates so registry lookups resolve.
# Kept in a dedicated helper so importing ``factory`` is the explicit act that
# populates the registry (mirrors ``src.pde.register_games``).
def ensure_substrate_registrants() -> None:
    """Import concrete substrate modules so their ``@register_*`` decorators run."""
    import src.research.substrates.skfem_tri as _skfem_tri_registrant
    import src.research.substrates.tensor_grid as _tensor_grid_registrant

    # Keep references so the imports are not "unused" under ruff while still
    # executing the ``@register_refinement_substrate`` side effects.
    _ = (_skfem_tri_registrant, _tensor_grid_registrant)


def build_default_operator(
    operator_name: OperatorName = "poisson",
    *,
    scale: float = 1.0,
) -> PDEOperator:
    """Build a Pydantic-configured Poisson operator for substrate games.

    Args:
        operator_name: ``poisson`` (unit square) or ``lshape_poisson`` (L-shaped).
        scale: Domain scale for the L-shaped operator (ignored for rectangular).

    Returns:
        A concrete ``PDEOperator`` with an exact solution (required by substrates).

    """
    if operator_name == "poisson":
        return PoissonOperator(
            PDEConfig(
                name="substrate_game_poisson",
                pde_type=PDEType.POISSON,
                domain_dim=2,
                domain_min=[0.0, 0.0],
                domain_max=[1.0, 1.0],
            )
        )
    if operator_name == "lshape_poisson":
        return LShapedPoissonOperator(
            PDEConfig(
                name="substrate_game_lshape",
                pde_type=PDEType.POISSON,
                domain_dim=2,
                domain_min=[-scale, -scale],
                domain_max=[scale, scale],
            )
        )
    raise ValueError(
        f"unknown operator_name {operator_name!r}; expected 'poisson' or 'lshape_poisson'"
    )


def build_substrate_from_config(
    config: SubstrateConfig,
    *,
    operator: PDEOperator | None = None,
    operator_name: OperatorName = "poisson",
    scale: float = 1.0,
) -> Any:
    """Resolve ``config.kind`` through ``RefinementSubstrateRegistry`` and construct.

    Args:
        config: Typed substrate config (``kind`` selects the registrant).
        operator: Optional pre-built operator; otherwise built from ``operator_name``.
        operator_name: Used when ``operator`` is omitted.
        scale: L-shape scale when building ``lshape_poisson``.

    Returns:
        A concrete substrate instance (structural ``RefinementSubstrate``).

    """
    ensure_substrate_registrants()
    registry = RefinementSubstrateRegistry()
    # Production lookup — retires the "zero runtime lookups" charter deviation.
    substrate_cls = registry.get_or_raise(config.kind)
    op = operator if operator is not None else build_default_operator(
        operator_name, scale=scale
    )
    kwargs: dict[str, Any] = {"operator": op, "config": config}
    if config.kind == SUBSTRATE_KIND_TENSOR_GRID and operator_name == "lshape_poisson":
        kwargs["inside"] = lshape_inside_predicate(scale)
    substrate = substrate_cls(**kwargs)
    logger.info(
        "substrate_built_from_registry",
        kind=config.kind,
        operator_name=operator_name if operator is None else type(op).__name__,
        registrant=substrate_cls.__name__,
    )
    return substrate


__all__ = [
    "OperatorName",
    "build_default_operator",
    "build_substrate_from_config",
    "ensure_substrate_registrants",
]
