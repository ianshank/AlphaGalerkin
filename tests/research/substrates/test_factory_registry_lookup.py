"""First non-test registry lookup path for substrates."""

from __future__ import annotations

import src.pde.register_refinement_games  # noqa: F401
from src.refinement.substrate_registry import RefinementSubstrateRegistry
from src.research.substrates.config import (
    SUBSTRATE_KIND_SKFEM_TRI,
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


def test_ensure_registrants_re_registers_after_clear() -> None:
    """Import-only ensure cannot recover from ``RefinementSubstrateRegistry.clear``.

    The registry is a process-global singleton. Two suites ``clear()`` it
    without restoring (``tests/refinement/test_substrate.py``,
    ``tests/research/test_skfem_substrate.py``). After the modules are
    imported, a second ``import`` is a no-op, so an ensure that only
    imports leaves ``Available: []``. That is CI run 34291592000: six
    ``KeyError: 'tensor_grid' not registered`` failures in the combined
    fast lane.

    The first ``ensure`` here is load-bearing: without it this test would
    pass on the import-only body whenever it ran first in a fresh process
    (the decorator would still fire). CI's failure was the opposite order.

    Mutations: (1) restore the import-only body — this named test fails on
    the post-ensure ``get``; (2) re-register unconditionally — the second
    ``ensure`` after clear raises ``ValueError`` duplicate. Not
    ``gpu_required`` / ``fem_required``.
    """
    registry = RefinementSubstrateRegistry()
    ensure_substrate_registrants()
    assert registry.get(SUBSTRATE_KIND_TENSOR_GRID) is not None
    registry.clear()
    try:
        assert registry.get(SUBSTRATE_KIND_TENSOR_GRID) is None
        assert registry.get(SUBSTRATE_KIND_SKFEM_TRI) is None
        ensure_substrate_registrants()
        assert registry.get(SUBSTRATE_KIND_TENSOR_GRID) is not None
        assert registry.get(SUBSTRATE_KIND_SKFEM_TRI) is not None
        ensure_substrate_registrants()
        assert registry.get(SUBSTRATE_KIND_TENSOR_GRID) is not None
    finally:
        ensure_substrate_registrants()


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
