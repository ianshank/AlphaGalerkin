"""Every registered PDE operator constructs the way the registry's callers construct it.

Defect class: a name registered in ``src.pde.registry`` that its callers cannot
build as ``cls(config)`` crashes every caller that selects it by name. The
``poisson_multi_corner`` entry was one: its constructor also required ``domain``
and ``corners``, so ``get_pde_operator("poisson_multi_corner")(config)`` raised
``TypeError``, and ``PDEBenchmarkRunner._create_operator`` catches only
``KeyError``, so a benchmark YAML naming it crashed.

How the callers construct (read 2026-10-08): ``_centaur_common.build_pde_operator``,
``PDEBenchmarkRunner._create_operator``, ``NoyronBasisScenario._build_operator`` and
``HelicalBasisSelectionInterface`` each call ``cls(config)`` with one ``PDEConfig``.
The first two pass a ``PDEConfig(name=..., pde_type=...)``-style config. The
helical two pass ``_create_helical_pde_config(name)``, because those operators
read their geometry from ``PDEConfig.geometry`` -- a pre-existing difference,
declared in :data:`CALLER_CONFIGS`, which expires itself in both directions.

The presets are reached through the registry only, never imported by name, so
this file collects on the pre-fix tree and fails there by named test.

Planted defects, each killed by a named test (``harden-a-guard``):

* the generic class registered again under ``poisson_multi_corner`` ->
  ``test_every_registered_name_constructs_from_a_config_alone[poisson_multi_corner]``
  and ``TestPolyominoPresets::test_the_generic_class_is_not_registered``;
* a preset that replaces an explicit box instead of verifying it (the D1 class) ->
  ``TestPolyominoPresets::test_an_explicit_box_that_disagrees_is_refused_not_replaced``;
* a preset that keeps the generic ``name`` -> ``test_a_registered_class_is_named_by_its_key``
  and the preset rows of the construction test;
* a ``CALLER_CONFIGS`` entry for an operator a default config already builds ->
  ``test_every_caller_config_exemption_is_still_needed``;
* the helical exemption deleted -> the helical rows of the construction test.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import MultiCornerPoissonOperator, PDEOperator
from src.pde.operators.multi_corner_poisson import (
    LSHAPE_PRESET_NAME,
    ZSHAPE_PRESET_NAME,
    build_lshape_multi_corner_operator,
    build_zshape_poisson_operator,
)
from src.pde.register_games import HELICAL_OPERATOR_NAMES, _create_helical_pde_config
from src.pde.registry import get_pde_operator, list_pde_operators
from src.research.pde_benchmarks import PDEBenchmarkRunner
from src.research.substrates.factory import build_default_operator

#: Registered names whose callers build something other than a default config,
#: and the builder those callers use.
CALLER_CONFIGS: Final[dict[str, Callable[[str], PDEConfig]]] = dict.fromkeys(
    HELICAL_OPERATOR_NAMES, _create_helical_pde_config
)

#: Built-in registrations today: the scan must see at least this many.
MIN_REGISTERED: Final[int] = 13

#: How a config-alone constructor refuses a config it cannot use.
CONSTRUCTION_REFUSALS: Final[tuple[type[Exception], ...]] = (TypeError, ValueError)

#: The two presets and the builders that must agree with them.
PRESETS: Final[list[tuple[str, Callable[[], MultiCornerPoissonOperator]]]] = [
    (LSHAPE_PRESET_NAME, build_lshape_multi_corner_operator),
    (ZSHAPE_PRESET_NAME, build_zshape_poisson_operator),
]

#: Points at which a registry-built preset must equal its builder's preset.
N_COMPARISON_POINTS: Final[int] = 64

#: The product's registrations, snapshotted at collection. The registry is a
#: process-wide singleton that tests write to (``tests/pde/test_pde_registry.py``
#: registers ``test_custom_pde_op`` and leaves it), so a class defined outside
#: ``src`` is a test's own entry, not part of the contract.
REGISTERED: Final[list[str]] = [
    name for name in list_pde_operators() if get_pde_operator(name).__module__.startswith("src.")
]


def _default_config(name: str) -> PDEConfig:
    """What a registry caller passes when it knows only the name."""
    return PDEConfig(name=name, pde_type=get_pde_operator(name).pde_type)


def _caller_config(name: str) -> PDEConfig:
    return CALLER_CONFIGS.get(name, _default_config)(name)


def _from_registry(name: str, config: PDEConfig) -> PDEOperator:
    return get_pde_operator(name)(config)


def test_the_scan_sees_every_builtin_operator() -> None:
    """Vacuity: a parametrisation over an empty registry passes forever."""
    assert len(REGISTERED) >= MIN_REGISTERED, REGISTERED


@pytest.mark.parametrize("name", REGISTERED)
def test_every_registered_name_constructs_from_a_config_alone(name: str) -> None:
    cls = get_pde_operator(name)
    operator = cls(_caller_config(name))
    assert isinstance(operator, cls)


@pytest.mark.parametrize("name", REGISTERED)
def test_a_registered_class_is_named_by_its_key(name: str) -> None:
    """``to_dict()["name"]`` must then name an entry that reconstructs it."""
    assert get_pde_operator(name).name == name


@pytest.mark.parametrize("name", sorted(CALLER_CONFIGS))
def test_every_caller_config_exemption_is_still_needed(name: str) -> None:
    """An exemption for an operator a default config builds hides nothing; drop it."""
    assert name in REGISTERED, f"{name} is no longer registered; drop its CALLER_CONFIGS entry"
    with pytest.raises(CONSTRUCTION_REFUSALS):
        _from_registry(name, _default_config(name))


class TestPolyominoPresets:
    """The multi-corner operator through the registry's public path."""

    def test_the_generic_class_is_not_registered(self) -> None:
        """It needs ``domain`` and ``corners``; only presets that fix both are."""
        registered = {get_pde_operator(name) for name in REGISTERED}
        assert MultiCornerPoissonOperator not in registered

    @pytest.mark.parametrize(("name", "builder"), PRESETS, ids=[name for name, _ in PRESETS])
    def test_a_registry_built_preset_is_the_builders_preset(
        self, name: str, builder: Callable[[], MultiCornerPoissonOperator]
    ) -> None:
        from_registry = _from_registry(name, PDEConfig(name=name, pde_type=PDEType.POISSON))
        built = builder()
        assert isinstance(from_registry, MultiCornerPoissonOperator)
        assert type(from_registry) is type(built)
        assert from_registry.corners == built.corners
        assert from_registry.geometry.cells == built.geometry.cells
        assert from_registry.domain_min.tolist() == built.domain_min.tolist()
        assert from_registry.domain_max.tolist() == built.domain_max.tolist()
        points = built.generate_collocation_points(N_COMPARISON_POINTS, seed=0)
        np.testing.assert_array_equal(
            from_registry.exact_solution(points), built.exact_solution(points)
        )

    def test_the_factorys_z_preset_is_the_registered_one(self) -> None:
        """``build_default_operator("zshape_poisson")`` keeps working, as that preset."""
        operator = build_default_operator("zshape_poisson")
        assert type(operator) is get_pde_operator(ZSHAPE_PRESET_NAME)

    def test_an_explicit_box_that_matches_is_accepted(self) -> None:
        config = PDEConfig(
            name="z", pde_type=PDEType.POISSON, domain_min=[-1.0, -1.0], domain_max=[2.0, 1.0]
        )
        assert _from_registry(ZSHAPE_PRESET_NAME, config).domain_max.tolist() == [2.0, 1.0]

    def test_an_explicit_box_that_disagrees_is_refused_not_replaced(self) -> None:
        """A box the caller set is verified, never silently swapped (the D1 class)."""
        config = PDEConfig(
            name="z", pde_type=PDEType.POISSON, domain_min=[0.0, 0.0], domain_max=[1.0, 1.0]
        )
        with pytest.raises(ValueError, match="bounding box"):
            _from_registry(ZSHAPE_PRESET_NAME, config)

    def test_only_the_unset_box_comes_from_the_domain(self) -> None:
        """The caller's other fields survive, and the caller's config is not mutated."""
        config = PDEConfig(name="mine", pde_type=PDEType.POISSON, diffusion_coeff=2.5)
        operator = _from_registry(ZSHAPE_PRESET_NAME, config)
        assert operator.config.name == "mine"
        assert operator.config.diffusion_coeff == 2.5
        assert operator.domain_min.tolist() == [-1.0, -1.0]
        assert config.domain_min == [0.0, 0.0]

    def test_a_benchmark_yaml_naming_a_preset_builds_it(self, tmp_path: Path) -> None:
        """The finding's crash site: ``_create_operator`` catches only ``KeyError``."""
        suite = tmp_path / "suite.yaml"
        suite.write_text("suite_name: registry_contract\nbenchmarks: []\nbaselines: []\n")
        bench_cfg = {
            "name": "zshape",
            "pde_type": ZSHAPE_PRESET_NAME,
            "domain": {"min": [-1.0, -1.0], "max": [2.0, 1.0]},
        }
        operator = PDEBenchmarkRunner(suite)._create_operator(bench_cfg)
        assert type(operator) is get_pde_operator(ZSHAPE_PRESET_NAME)
