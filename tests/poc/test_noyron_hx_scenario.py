"""Smoke test for the Noyron HX scenario (Leap 71 integration).

This test exercises the full scenario lifecycle on a *very* small
configuration on CPU. It does not assert convergence — only that:

- the scenario is registered under the expected name,
- ``run()`` returns a result whose status is not ``ERROR``,
- the headline metrics ``mse_low`` and ``mse_high`` are recorded.

Headline accuracy is verified by the GPU run described in the plan
file; this test exists so unrelated regressions cannot silently break
the scenario plumbing.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from src.poc.config import ScenarioStatus
from src.poc.config_noyron import NoyronHXScenarioConfig
from src.poc.registry import ScenarioRegistry


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:
    """Reset the singleton registry between tests in this module.

    Other PoC suites also clear the registry in autouse fixtures; this
    isolation keeps us robust to test ordering: clear first, then drop
    cached scenario modules so the next import re-runs the @scenario
    decorators cleanly.
    """
    ScenarioRegistry().clear()
    for mod_name in list(sys.modules):
        if mod_name.startswith("src.poc.scenarios"):
            del sys.modules[mod_name]


def _import_scenario_class() -> type:
    """Fresh-import the scenario class so registration runs in this test."""
    module = importlib.import_module("src.poc.scenarios.noyron_hx")
    return module.NoyronHXScenario


def _smoke_config() -> NoyronHXScenarioConfig:
    return NoyronHXScenarioConfig(
        name="noyron_hx",
        description="smoke",
        helix_R_major=0.05,
        helix_r_minor=0.012,
        helix_pitch=0.02,
        helix_n_turns=2,
        n_train_pts=64,
        n_train_boundary_pts=8,
        n_eval_pts=64,
        n_epochs=2,
        batch_size=1,
        d_model=16,
        n_heads=2,
        n_layers=2,
        n_fourier_features=8,
        ref_solver_kind="analytical_harmonic",
        # The smoke test forces CPU so it can run in CI without CUDA;
        # the project default is GPU-preferred.
        device="cpu",
        # Threshold relaxed because we only train for 2 epochs on
        # 64 points; we just want the plumbing to round-trip.
        mse_threshold_low=1e6,
        mse_threshold_high=1e6,
        transfer_ratio_threshold=1e6,
    )


def test_scenario_is_registered() -> None:
    expected_cls = _import_scenario_class()
    cls = ScenarioRegistry().get("noyron_hx")
    assert cls is expected_cls


def test_smoke_run_completes_without_error() -> None:
    cls = _import_scenario_class()
    scenario = cls(config=_smoke_config())
    result = scenario.run()
    # ERROR means an unexpected exception. PASSED/FAILED both indicate
    # the lifecycle completed; we only forbid ERROR here.
    assert (
        result.status != ScenarioStatus.ERROR
    ), f"Scenario errored: {result.error_message}\n{result.error_traceback}"
    assert "mse_low" in result.metrics
    assert "mse_high" in result.metrics
    assert "transfer_ratio" in result.metrics


def test_cuda_preference_fails_loud_when_unavailable() -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA available; cannot exercise the failure path.")

    cfg = _smoke_config()
    cfg = cfg.model_copy(update={"device": "cuda"})
    cls = _import_scenario_class()
    scenario = cls(config=cfg)
    # The setup() raises a clear RuntimeError; BaseScenario.run() converts
    # this into an ERROR result.
    result = scenario.run()
    assert result.status == ScenarioStatus.ERROR
    assert "CUDA" in (result.error_message or "")


def test_auto_device_falls_back_to_cpu() -> None:
    """device='auto' must complete on machines without CUDA."""
    cfg = _smoke_config().model_copy(update={"device": "auto"})
    cls = _import_scenario_class()
    scenario = cls(config=cfg)
    result = scenario.run()
    assert result.status != ScenarioStatus.ERROR, f"Scenario errored: {result.error_message}"


def test_invalid_device_preference_raises() -> None:
    """An unknown device string must raise from the resolver."""
    from src.poc.scenarios.noyron_hx import _resolve_device

    with pytest.raises(ValueError, match="Unknown device preference"):
        _resolve_device("tpu")


def test_voxel_fdm_reference_path_round_trips(tmp_path) -> None:
    """The voxel_fdm reference branch must be exercised end-to-end."""
    cfg = _smoke_config().model_copy(
        update={
            "ref_solver_kind": "voxel_fdm",
            "voxel_fdm_resolution": 16,  # tiny to keep the test fast
        }
    )
    cls = _import_scenario_class()
    scenario = cls(config=cfg)
    result = scenario.run()
    assert (
        result.status != ScenarioStatus.ERROR
    ), f"Scenario errored: {result.error_message}\n{result.error_traceback}"
    # Custom field set in execute() — voxel_fdm should round-trip through the result.
    assert getattr(result, "ref_solver_kind", None) == "voxel_fdm"


def test_model_checkpoint_artifact_recorded() -> None:
    """The trained model must be persisted as a recorded artifact."""
    from pathlib import Path

    cls = _import_scenario_class()
    scenario = cls(config=_smoke_config())
    result = scenario.run()
    assert result.status != ScenarioStatus.ERROR
    assert "model" in result.artifacts
    assert Path(result.artifacts["model"]).exists()


# ---------------------------------------------------------------------------
# Config validator tests — exercise the model_validator branches directly so
# coverage hits config_noyron.py lines 163/169/173.
# ---------------------------------------------------------------------------


class TestNoyronHXScenarioConfigValidators:
    def test_self_intersecting_helix_rejected(self) -> None:
        with pytest.raises(ValueError, match="self-intersection"):
            NoyronHXScenarioConfig(
                name="bad",
                description="x",
                helix_R_major=0.01,
                helix_r_minor=0.01,  # equals R_major -> self-intersection
                n_train_pts=64,
                n_eval_pts=64,
                device="cpu",
            )

    def test_picogk_requires_voxel_path(self) -> None:
        with pytest.raises(ValueError, match="picogk_voxel_path"):
            NoyronHXScenarioConfig(
                name="bad",
                description="x",
                use_picogk=True,
                picogk_voxel_path=None,
                n_train_pts=64,
                n_eval_pts=64,
                device="cpu",
            )

    def test_eval_pts_must_be_at_least_train_pts(self) -> None:
        with pytest.raises(ValueError, match="should be >= n_train_pts"):
            NoyronHXScenarioConfig(
                name="bad",
                description="x",
                n_train_pts=4096,
                n_eval_pts=128,  # smaller than train: degenerate
                device="cpu",
            )
