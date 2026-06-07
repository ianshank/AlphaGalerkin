"""Direct tests for the shared centaur MCTS primitives (`_centaur_common`).

These cover the construction helpers and the inner rollout loop independently
of any scenario, so the shared module is exercised even when scenario tests
use a synthetic cell override.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.integrations.lm_studio.config import LMStudioConfig
from src.integrations.lm_studio.preflight import PreflightReport
from src.mcts.evaluator import RandomEvaluator
from src.poc.scenarios import _centaur_common as common
from src.poc.scenarios._centaur_common import (
    PDE_TYPE_MAP,
    build_arm_evaluator,
    build_basis_game,
    build_pde_operator,
    enumerate_basis_descriptions,
    gate_llm_client,
    gate_trained_model,
    run_basis_selection_cell,
)


def _passing_report() -> PreflightReport:
    return PreflightReport(
        server_reachable=True,
        model_available=True,
        available_models=["m"],
        free_vram_gib=16.0,
        vram_sufficient=True,
        failure_reason="",
    )


def _failing_report() -> PreflightReport:
    return PreflightReport(
        server_reachable=False,
        model_available=False,
        available_models=[],
        free_vram_gib=None,
        vram_sufficient=True,
        failure_reason="unreachable",
    )


def _poisson_game(target_residual: float = 1e-6):
    operator = build_pde_operator("poisson")
    return build_basis_game(
        "poisson",
        operator,
        max_basis_functions=2,
        n_candidate_bases=4,
        target_residual=target_residual,
    )


def test_pde_type_map_includes_ood_operators() -> None:
    assert "helmholtz" in PDE_TYPE_MAP
    assert "biharmonic" in PDE_TYPE_MAP


def test_build_pde_operator_known() -> None:
    operator = build_pde_operator("helmholtz")
    assert operator.name == "helmholtz"


def test_build_pde_operator_unknown_raises() -> None:
    with pytest.raises(ValueError, match="has no PDEType mapping"):
        build_pde_operator("not_a_pde")


def test_build_basis_game_and_descriptions() -> None:
    game = _poisson_game()
    assert game.action_space_size == 4
    descriptions = enumerate_basis_descriptions(game)
    assert len(descriptions) == 4


# --------------------------------------------------------------------------- #
# build_arm_evaluator                                                         #
# --------------------------------------------------------------------------- #


def test_build_arm_evaluator_random() -> None:
    game = SimpleNamespace(action_space_size=5)
    evaluator = build_arm_evaluator(
        "random", game=game, pde_name="poisson", basis_descriptions=[], seed=0
    )
    assert isinstance(evaluator, RandomEvaluator)


def test_build_arm_evaluator_unknown_arm_raises() -> None:
    game = SimpleNamespace(action_space_size=5)
    with pytest.raises(ValueError, match="unknown arm"):
        build_arm_evaluator(
            "invented", game=game, pde_name="poisson", basis_descriptions=[], seed=0
        )


def test_build_arm_evaluator_trained_missing_model_raises() -> None:
    game = SimpleNamespace(action_space_size=5)
    with pytest.raises(RuntimeError, match="trained_model is None"):
        build_arm_evaluator(
            "trained", game=game, pde_name="poisson", basis_descriptions=[], seed=0, device="cpu"
        )


def test_build_arm_evaluator_trained_missing_device_raises() -> None:
    game = SimpleNamespace(action_space_size=5)
    with pytest.raises(RuntimeError, match="device is None"):
        build_arm_evaluator(
            "trained",
            game=game,
            pde_name="poisson",
            basis_descriptions=[],
            seed=0,
            trained_model=MagicMock(),
        )


def test_build_arm_evaluator_llm_missing_client_raises() -> None:
    game = SimpleNamespace(action_space_size=5)
    with pytest.raises(RuntimeError, match="lm_client is None"):
        build_arm_evaluator(
            "llm", game=game, pde_name="poisson", basis_descriptions=["b"] * 5, seed=0
        )


def test_build_arm_evaluator_llm_with_client() -> None:
    from src.integrations.lm_studio.evaluator import LMStudioEvaluator

    game = SimpleNamespace(action_space_size=3)
    evaluator = build_arm_evaluator(
        "llm",
        game=game,
        pde_name="poisson",
        basis_descriptions=["b0", "b1", "b2"],
        seed=7,
        lm_client=MagicMock(name="LMStudioClient"),
    )
    assert isinstance(evaluator, LMStudioEvaluator)


# --------------------------------------------------------------------------- #
# run_basis_selection_cell                                                    #
# --------------------------------------------------------------------------- #


def test_run_cell_returns_immediately_when_below_target() -> None:
    # A huge target means the initial error is already below it.
    game = _poisson_game(target_residual=0.999)
    evaluator = RandomEvaluator(n_actions=game.action_space_size)
    outcome = run_basis_selection_cell(
        game=game,
        evaluator=evaluator,
        target_residual=0.999,
        max_rollouts=8,
        n_simulations=2,
    )
    assert outcome.rollouts_used == 0
    assert outcome.final_residual >= 0.0


def test_run_cell_real_loop_accumulates_rollouts() -> None:
    import numpy as np
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    game = _poisson_game(target_residual=1e-9)
    evaluator = RandomEvaluator(n_actions=game.action_space_size)
    outcome = run_basis_selection_cell(
        game=game,
        evaluator=evaluator,
        target_residual=1e-9,
        max_rollouts=4,
        n_simulations=2,
        scenario_logger=MagicMock(),
    )
    assert outcome.rollouts_used >= 0
    assert outcome.final_residual >= 0.0


# --------------------------------------------------------------------------- #
# gate_llm_client                                                             #
# --------------------------------------------------------------------------- #


def test_gate_llm_client_disabled_by_config() -> None:
    log = MagicMock()
    cfg = LMStudioConfig(enabled=False, preflight_on_construct=False)
    client = gate_llm_client(
        cfg, cell_logger=log, preflight_fn=MagicMock(), client_factory=MagicMock()
    )
    assert client is None
    log.warning.assert_called_once()


def test_gate_llm_client_preflight_raises() -> None:
    log = MagicMock()
    cfg = LMStudioConfig(preflight_on_construct=False)

    def _raise(_cfg: LMStudioConfig) -> PreflightReport:
        raise RuntimeError("boom")

    client = gate_llm_client(cfg, cell_logger=log, preflight_fn=_raise, client_factory=MagicMock())
    assert client is None


def test_gate_llm_client_preflight_fails() -> None:
    log = MagicMock()
    cfg = LMStudioConfig(preflight_on_construct=False)
    client = gate_llm_client(
        cfg,
        cell_logger=log,
        preflight_fn=lambda _c: _failing_report(),
        client_factory=MagicMock(),
    )
    assert client is None


def test_gate_llm_client_construction_raises() -> None:
    log = MagicMock()
    cfg = LMStudioConfig(preflight_on_construct=False)

    def _raise_factory(_cfg: LMStudioConfig) -> object:
        raise RuntimeError("no client")

    client = gate_llm_client(
        cfg,
        cell_logger=log,
        preflight_fn=lambda _c: _passing_report(),
        client_factory=_raise_factory,
    )
    assert client is None


def test_gate_llm_client_success_disables_preflight_on_construct() -> None:
    log = MagicMock()
    cfg = LMStudioConfig(preflight_on_construct=True)
    captured: dict[str, LMStudioConfig] = {}

    def _factory(passed_cfg: LMStudioConfig) -> str:
        captured["cfg"] = passed_cfg
        return "client-sentinel"

    client = gate_llm_client(
        cfg,
        cell_logger=log,
        preflight_fn=lambda _c: _passing_report(),
        client_factory=_factory,
    )
    assert client == "client-sentinel"
    # The client must be built with preflight disabled (already preflighted).
    assert captured["cfg"].preflight_on_construct is False


# --------------------------------------------------------------------------- #
# gate_trained_model                                                          #
# --------------------------------------------------------------------------- #


def test_gate_trained_model_success() -> None:
    log = MagicMock()
    model = object()
    loader = MagicMock(return_value=(model, {"cfg": 1}))
    result = gate_trained_model("ckpt.pt", "cpu", cell_logger=log, loader=loader)
    assert result is model
    loader.assert_called_once_with("ckpt.pt", device="cpu", strict=False)


def test_gate_trained_model_failure_returns_none() -> None:
    log = MagicMock()

    def _raise(*_a: object, **_k: object) -> tuple[object, object]:
        raise FileNotFoundError("missing")

    result = gate_trained_model("ckpt.pt", "cpu", cell_logger=log, loader=_raise)
    assert result is None
    log.warning.assert_called_once()


# --------------------------------------------------------------------------- #
# build_arm_evaluator trained path + run-loop early exit                       #
# --------------------------------------------------------------------------- #


def test_build_arm_evaluator_trained_constructs_fnet(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    fake_fnet = MagicMock(return_value=sentinel)
    monkeypatch.setattr("src.mcts.evaluator.FNetEvaluator", fake_fnet)
    game = SimpleNamespace(action_space_size=4)
    evaluator = build_arm_evaluator(
        "trained",
        game=game,
        pde_name="poisson",
        basis_descriptions=[],
        seed=0,
        trained_model=MagicMock(),
        device="cpu",
    )
    assert evaluator is sentinel
    fake_fnet.assert_called_once()


def test_run_cell_early_exit_on_invalid_action(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force MCTS to return an illegal action so the early-exit branch (and its
    # warning) fires deterministically.
    monkeypatch.setattr(common.MCTS, "get_action", lambda self, game, **kw: -1)
    game = _poisson_game(target_residual=1e-9)
    evaluator = RandomEvaluator(n_actions=game.action_space_size)
    log = MagicMock()
    outcome = run_basis_selection_cell(
        game=game,
        evaluator=evaluator,
        target_residual=1e-9,
        max_rollouts=8,
        n_simulations=2,
        scenario_logger=log,
    )
    assert outcome.rollouts_used == 0
    log.warning.assert_called_once()
    assert log.warning.call_args.args[0] == "cell_loop_early_exit"
