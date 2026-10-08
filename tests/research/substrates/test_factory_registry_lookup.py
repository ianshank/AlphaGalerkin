"""First non-test registry lookup path for substrates."""

from __future__ import annotations

from typing import get_args

import pytest

import src.pde.register_refinement_games  # noqa: F401
from src.pde.games.substrate_refinement_config import SubstrateRefinementConfig
from src.pde.operators import LShapedPoissonOperator, MultiCornerPoissonOperator, PoissonOperator
from src.pde.operators.multi_corner_poisson import (
    DEFAULT_ZSHAPE_PRIMARY_COEFFICIENT,
    DEFAULT_ZSHAPE_SECONDARY_COEFFICIENT,
    ZShapeMultiCornerPoissonOperator,
)
from src.refinement.substrate_registry import RefinementSubstrateRegistry
from src.research.substrates.config import (
    SUBSTRATE_KIND_SKFEM_TRI,
    SUBSTRATE_KIND_TENSOR_GRID,
    SubstrateConfig,
)
from src.research.substrates.factory import (
    ZSHAPE_SCALE,
    OperatorName,
    build_default_operator,
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
    without restoring on the failing default tip (``tests/refinement/test_substrate.py``,
    ``tests/research/test_skfem_substrate.py``); those teardowns now call
    ``ensure_substrate_registrants``. After the modules are imported, a
    second ``import`` is a no-op, so an ensure that only imports leaves
    ``Available: []``. That is CI run 34292047225 on ``a85d265``: six
    ``KeyError: 'tensor_grid' not registered`` failures in the combined
    fast lane.

    The first ``ensure`` here is load-bearing: without it this test would
    pass on the import-only body whenever it ran first in a fresh process
    (the decorator would still fire). CI's failure was the opposite order.

    Mutations: (1) restore the import-only body — this named test fails on
    the post-ensure ``get``; (2) drop the missing-kind guard — ``ensure``
    raises ``ValueError`` duplicate once the decorator has already
    registered the kind. Not ``gpu_required`` / ``fem_required``.
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


class TestBuildDefaultOperator:
    """Every ``OperatorName`` dispatches to its operator; nothing else is accepted.

    Mutation kill: the ``scale != ZSHAPE_SCALE`` check deleted ->
    ``test_a_zshape_scale_is_refused_not_ignored`` (the scale is silently dropped).
    """

    def test_operator_names_include_the_zshape(self) -> None:
        assert get_args(OperatorName) == ("poisson", "lshape_poisson", "zshape_poisson")

    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            ("poisson", PoissonOperator),
            ("lshape_poisson", LShapedPoissonOperator),
            ("zshape_poisson", ZShapeMultiCornerPoissonOperator),
        ],
    )
    def test_each_name_builds_its_operator(self, name: OperatorName, cls: type) -> None:
        assert type(build_default_operator(name)) is cls

    def test_the_zshape_uses_the_named_default_strengths(self) -> None:
        operator = build_default_operator("zshape_poisson")
        assert isinstance(operator, MultiCornerPoissonOperator)
        assert [t.coefficient for t in operator.corners] == [
            DEFAULT_ZSHAPE_PRIMARY_COEFFICIENT,
            DEFAULT_ZSHAPE_SECONDARY_COEFFICIENT,
        ]
        assert build_default_operator("zshape_poisson", scale=ZSHAPE_SCALE) is not None

    def test_a_zshape_scale_is_refused_not_ignored(self) -> None:
        with pytest.raises(ValueError, match="silently ignored"):
            build_default_operator("zshape_poisson", scale=2.0)

    def test_an_unknown_name_lists_every_valid_one(self) -> None:
        with pytest.raises(ValueError, match="zshape_poisson"):
            build_default_operator("hexagon_poisson")  # type: ignore[arg-type]

    def test_the_refinement_game_config_accepts_the_zshape(self) -> None:
        """``SubstrateRefinementConfig.operator_name`` is typed by this Literal."""
        config = SubstrateRefinementConfig(name="z_game", operator_name="zshape_poisson")
        assert config.operator_name == "zshape_poisson"
