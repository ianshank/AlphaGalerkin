"""``LookaheadVsGreedyConfig``: pre-registered defaults, invariants, AQA against the spec.

The pre-registration (``specs/lookahead_vs_greedy.spec.md``) fixed every budget,
arm, seed and threshold before any run. Defect classes:

* **C1 drift from the pre-registration** -- a default, a YAML value or a
  threshold differs from the spec (the AQA tests read the spec's own table).
* **C2 lost invariant** -- the config accepts an unranked legal set
  (``top_k_actions = 0``), a search with no room past one ply, non-zero
  temperature, an unreachable C3 bar, or a hand-written threshold list.
* **C3 broken dispatch** -- ``load_config_from_dict`` returns a base config
  for the shipped YAMLs, which would reject their fields.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError

from src.pde.operators.multi_corner_poisson import (
    DEFAULT_ZSHAPE_PRIMARY_COEFFICIENT,
    DEFAULT_ZSHAPE_SECONDARY_COEFFICIENT,
)
from src.poc.config import MetricThreshold, load_config_from_dict
from src.poc.scenarios.lookahead_vs_greedy_config import (
    DEFAULT_BREAK_EVEN_ALPHAS,
    DEFAULT_CLASSICAL_MAX_DOF,
    DEFAULT_DORFLER_THETAS,
    DEFAULT_LOOKAHEAD_MAX_STEPS,
    DEFAULT_LOOKAHEAD_N_SIMULATIONS,
    DEFAULT_LOOKAHEAD_TOP_K,
    DEFAULT_ROBUST_N_SEEDS,
    SCENARIO_NAME,
    TESTBED_LABELS,
    LookaheadVsGreedyConfig,
)
from src.research.substrates.config import SubstrateConfig
from src.research.substrates.factory import build_default_operator

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "specs" / "lookahead_vs_greedy.spec.md"
YAMLS = {
    "lshape_poisson": REPO_ROOT / "config" / "scenarios" / "lookahead_vs_greedy_lshape.yaml",
    "zshape_poisson": REPO_ROOT / "config" / "scenarios" / "lookahead_vs_greedy_zshape.yaml",
}
#: Fields a testbed YAML may set differently from the defaults (its identity).
TESTBED_FIELDS = frozenset(
    {"name", "description", "operator_name", "artifact_basename", "substrate", "thresholds"}
)
#: ``BaseScenarioConfig`` bookkeeping, not part of the pre-registration.
BOOKKEEPING_FIELDS = frozenset(
    {"tier", "enabled", "timeout_seconds", "retry_count", "requires_gpu",
     "estimated_duration_seconds"}
)  # fmt: skip
#: The spec's Thresholds table: | `metric` | `op` | `value` | meaning |
THRESHOLD_ROW = re.compile(
    r"^\| `(?P<name>[a-z_0-9.]+)` \| `(?P<op>[<>=]+)` \| `(?P<value>[0-9.]+)` \|"
)


def _config(**overrides: Any) -> LookaheadVsGreedyConfig:
    return LookaheadVsGreedyConfig(**overrides)


def _yaml_entry(operator_name: str) -> dict[str, Any]:
    raw = yaml.safe_load(YAMLS[operator_name].read_text(encoding="utf-8"))
    entries = [e for e in raw["scenarios"] if e["name"] == SCENARIO_NAME]
    assert len(entries) == 1
    return dict(entries[0])


def _spec_thresholds() -> list[tuple[str, str, float]]:
    text = SPEC.read_text(encoding="utf-8")
    section = text.split("\n## Thresholds\n", 1)[1].split("\n## ", 1)[0]
    rows = [THRESHOLD_ROW.match(line) for line in section.splitlines()]
    return [(m["name"], m["op"], float(m["value"])) for m in rows if m is not None]


class TestPreRegisteredDefaults:
    def test_budgets_arms_and_seeds(self) -> None:
        cfg = _config()
        assert cfg.name == SCENARIO_NAME
        assert cfg.max_steps == DEFAULT_LOOKAHEAD_MAX_STEPS == 30
        assert cfg.classical_max_dof == DEFAULT_CLASSICAL_MAX_DOF == 1000
        assert cfg.dorfler_thetas == DEFAULT_DORFLER_THETAS == (0.1, 0.3, 0.5)
        assert cfg.n_simulations == DEFAULT_LOOKAHEAD_N_SIMULATIONS == 64
        assert cfg.top_k_actions == DEFAULT_LOOKAHEAD_TOP_K == 4
        assert cfg.robust_n_seeds == DEFAULT_ROBUST_N_SEEDS == 5
        assert cfg.break_even_alphas == DEFAULT_BREAK_EVEN_ALPHAS == (1.0, 1.5)
        assert cfg.robust_seeds() == [42, 1051, 2060, 3069, 4078]
        assert (cfg.temperature, cfg.search_mode) == (0.0, "single_agent")
        assert cfg.evaluator_name == "ResidualPriorErrorValueEvaluator"
        assert (cfg.substrate.kind, cfg.substrate.element_type) == ("skfem_tri", "P1")
        assert cfg.require_adequacy_precondition is True
        assert cfg.adequacy_theta == 0.5

    def test_thresholds_equal_the_spec_table(self) -> None:
        """AQA: names, operators and values exactly as pre-registered."""
        spec = _spec_thresholds()
        assert len(spec) == 5, "the spec's Thresholds table did not parse"
        assert [(t.name, t.operator, t.value) for t in _config().get_default_thresholds()] == spec
        assert all(isinstance(t, MetricThreshold) for t in _config().get_default_thresholds())

    def test_the_z_testbed_has_the_pre_registered_corner_strengths(self) -> None:
        """T2 is c1 = 1.0, c2 = 0.25 -- the operator the YAML's name builds."""
        assert (DEFAULT_ZSHAPE_PRIMARY_COEFFICIENT, DEFAULT_ZSHAPE_SECONDARY_COEFFICIENT) == (
            1.0,
            0.25,
        )
        operator = build_default_operator("zshape_poisson")
        assert [term.coefficient for term in operator.corners] == [1.0, 0.25]  # type: ignore[attr-defined]
        assert "c1 = 1.0" in SPEC.read_text(encoding="utf-8")
        assert "c2 = 0.25" in SPEC.read_text(encoding="utf-8")


class TestShippedYamls:
    @pytest.mark.parametrize("operator_name", sorted(YAMLS))
    def test_each_yaml_restates_every_pre_registered_default(self, operator_name: str) -> None:
        """AQA: a YAML may name its testbed; it may not move a pre-registered value."""
        entry = _yaml_entry(operator_name)
        loaded = load_config_from_dict(entry)
        assert type(loaded).__name__ == LookaheadVsGreedyConfig.__name__
        cfg = cast("LookaheadVsGreedyConfig", loaded)
        defaults = _config()
        for field in LookaheadVsGreedyConfig.model_fields:
            if field in TESTBED_FIELDS | BOOKKEEPING_FIELDS:
                continue
            assert getattr(cfg, field) == getattr(defaults, field), field
        for field in ("kind", "element_type", "initial_refinements", "error_metric",
                      "marking_variant", "solve_cache_max_entries"):  # fmt: skip
            assert getattr(cfg.substrate, field) == getattr(defaults.substrate, field), field
        assert cfg.operator_name == operator_name
        assert cfg.artifact_basename == f"lookahead_vs_greedy_{operator_name.split('_')[0]}"
        assert cfg.output_dir == "results"
        assert not cfg.thresholds

    def test_the_two_testbeds_differ_only_in_their_identity(self) -> None:
        lshape, zshape = _yaml_entry("lshape_poisson"), _yaml_entry("zshape_poisson")
        differing = {k for k in lshape.keys() | zshape.keys() if lshape.get(k) != zshape.get(k)}
        assert differing == {"operator_name", "artifact_basename", "description"}


class TestInvariants:
    def test_the_name_is_locked(self) -> None:
        with pytest.raises(ValidationError, match="name must be"):
            _config(name="other")

    def test_selection_is_deterministic(self) -> None:
        with pytest.raises(ValidationError, match="temperature must be 0"):
            _config(temperature=1.0)

    def test_the_legal_set_must_be_ranked(self) -> None:
        """C2: at top_k_actions = 0 a divergence can be an index-order tie-break."""
        with pytest.raises(ValidationError, match="top_k_actions"):
            _config(top_k_actions=0)

    @pytest.mark.parametrize(("n_simulations", "top_k"), [(4, 4), (3, 4)])
    def test_the_tree_needs_room_past_one_ply(self, n_simulations: int, top_k: int) -> None:
        with pytest.raises(ValidationError, match="room to grow"):
            _config(n_simulations=n_simulations, top_k_actions=top_k)

    def test_c3_must_be_reachable(self) -> None:
        with pytest.raises(ValidationError, match="could never pass"):
            _config(robust_n_seeds=2)
        assert _config(robust_n_seeds=2, min_robust_wins=2).min_robust_wins == 2

    @pytest.mark.parametrize("thetas", [(), (0.5, 0.5), (0.0, 0.5), (0.5, 1.0)])
    def test_dorfler_thetas_are_unique_fractions(self, thetas: tuple[float, ...]) -> None:
        with pytest.raises(ValidationError, match="θ|dorfler_thetas"):
            _config(dorfler_thetas=thetas)

    @pytest.mark.parametrize("alphas", [(), (1.0, 1.0), (0.0,), (-1.5,)])
    def test_break_even_alphas_are_unique_and_positive(self, alphas: tuple[float, ...]) -> None:
        with pytest.raises(ValidationError, match="break_even_alphas"):
            _config(break_even_alphas=alphas)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"substrate": SubstrateConfig(name="tg", kind="tensor_grid", initial_side=4)},
            {"operator_name": "poisson"},
        ],
    )
    def test_adequacy_needs_skfem_and_a_corner_testbed(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="adequacy precondition"):
            _config(**overrides)

    def test_the_z_testbed_is_unit_sized(self) -> None:
        with pytest.raises(ValidationError, match="unit Z-tetromino"):
            _config(operator_name="zshape_poisson", lshape_scale=2.0)

    def test_thresholds_can_only_be_the_pre_registered_ones(self) -> None:
        """C2: the GO criteria are derived from typed fields, never hand-written."""
        loose = MetricThreshold(name="primary_l2_ratio_vs_best_classical", operator="<", value=5.0)
        with pytest.raises(ValidationError, match="pre-registered GO criteria"):
            _config(thresholds=[loose])
        cfg = _config()
        cfg.thresholds = cfg.get_default_thresholds()  # what scenario setup does
        assert cfg.thresholds == cfg.get_default_thresholds()

    def test_an_artifact_basename_carries_no_extension(self) -> None:
        with pytest.raises(ValidationError, match="extension"):
            _config(artifact_basename="lookahead.csv")


class TestDerived:
    @pytest.mark.parametrize("operator_name", sorted(TESTBED_LABELS))
    def test_testbed_labels(self, operator_name: str) -> None:
        cfg = _config(operator_name=operator_name, require_adequacy_precondition=False)
        assert cfg.testbed_label() == TESTBED_LABELS[operator_name]

    def test_the_arena_config_carries_the_game_and_keeps_the_scored_locks(self) -> None:
        cfg = _config(operator_name="zshape_poisson", max_steps=7, top_k_actions=3)
        arena = cfg.arena_config()
        assert (arena.add_noise, arena.temperature, arena.search_mode) == (
            False,
            0.0,
            "single_agent",
        )
        assert arena.substrate is cfg.substrate
        assert (arena.operator_name, arena.max_steps, arena.top_k_actions) == (
            "zshape_poisson",
            7,
            3,
        )
        assert arena.max_dof == cfg.classical_max_dof
        assert arena.max_action_space == cfg.max_action_space
        assert arena.value_scale == cfg.value_scale
        assert arena.require_adequacy_precondition is True
        assert arena.adequacy_gate().rate_fit_dof_range == (200.0, 5000.0)

    def test_dispatch_by_name(self) -> None:
        cfg = load_config_from_dict({"name": SCENARIO_NAME})
        assert type(cfg).__name__ == LookaheadVsGreedyConfig.__name__
        arena = load_config_from_dict({"name": "mcts_classical_amr_arena"})
        assert type(arena).__name__ == "MCTSClassicalAMRArenaConfig"
