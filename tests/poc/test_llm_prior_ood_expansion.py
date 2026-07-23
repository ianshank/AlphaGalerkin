"""OOD-expansion coverage for the LLM-prior ablation (helmholtz / biharmonic).

The trained FNet evaluator was never trained on the Helmholtz (oscillatory
zeroth-order term) or Biharmonic (fourth-order) residual structures, so they are
held-out OOD families. These operators already exist in the registry and in the
``ood_pde`` Literal + ``PDE_TYPE_MAP``; this suite proves they are usable as OOD
families end-to-end on CPU (no GPU / LM Studio needed) and that the shipped demo
YAMLs validate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.poc.config import load_config_from_dict
from src.poc.scenarios._centaur_common import (
    PDE_TYPE_MAP,
    build_basis_game,
    build_pde_operator,
)
from src.poc.scenarios.llm_prior_config import LLMPriorAblationConfig

_OOD_FAMILIES = ["helmholtz", "biharmonic"]
_CONFIG_DIR = Path("config/scenarios")


@pytest.mark.parametrize("pde", _OOD_FAMILIES)
class TestOODFamily:
    def test_in_pde_type_map(self, pde: str) -> None:
        assert pde in PDE_TYPE_MAP

    def test_config_accepts_ood_pde(self, pde: str) -> None:
        cfg = LLMPriorAblationConfig(name="llm_prior_ablation", ood_pde=pde)  # type: ignore[arg-type]
        assert cfg.ood_pde == pde

    def test_operator_and_game_build_with_finite_error(self, pde: str) -> None:
        """The OOD operator constructs a non-degenerate basis-selection game."""
        operator = build_pde_operator(pde)
        game = build_basis_game(
            pde,
            operator,
            max_basis_functions=4,
            n_candidate_bases=8,
            target_residual=1e-6,
        )
        error = float(game.get_initial_state().error_estimate)
        assert error > 0.0  # meaningful target (unlike homogeneous helical ops)
        assert game.action_space_size == 8

    def test_demo_yaml_validates(self, pde: str) -> None:
        path = _CONFIG_DIR / f"llm_prior_{pde}.yaml"
        assert path.exists(), f"missing OOD config {path}"
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        scenario = doc["scenarios"][0]
        cfg = load_config_from_dict(scenario)
        # Assert on class name + value rather than ``isinstance``: other suites
        # may import the config module under a non-``src.`` path, producing a
        # duplicate class object that fails identity-based ``isinstance``.
        assert type(cfg).__name__ == "LLMPriorAblationConfig"
        assert cfg.ood_pde == pde
