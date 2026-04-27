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
    assert result.status != ScenarioStatus.ERROR, (
        f"Scenario errored: {result.error_message}\n{result.error_traceback}"
    )
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
    assert result.status != ScenarioStatus.ERROR, (
        f"Scenario errored: {result.error_message}\n{result.error_traceback}"
    )
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

    def test_helix_n_turns_default_matches_yaml(self) -> None:
        """The dataclass default must match config/scenarios/noyron_hx.yaml.

        Mismatch was the v1 deviation noted in the gap report; locking
        this in a test keeps the YAML and Python defaults in lockstep.
        """
        cfg = NoyronHXScenarioConfig(
            name="defaults",
            description="x",
            n_train_pts=64,
            n_eval_pts=64,
            device="cpu",
        )
        assert cfg.helix_n_turns == 5


# ---------------------------------------------------------------------------
# Metrics surfaced by execute() — the gap-report fixes for the missing
# ``accept_rate`` / ``train_time_s`` / ``eval_time_s`` keys.
# ---------------------------------------------------------------------------


class TestNoyronHXScenarioMetrics:
    """Verify the result.metrics dict contains every expected key."""

    def test_accept_rate_metric_recorded(self) -> None:
        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config())
        result = scenario.run()
        assert result.status != ScenarioStatus.ERROR
        assert "accept_rate" in result.metrics
        # Helical tube fills only a few percent of its bbox.
        rate = result.metrics["accept_rate"]
        assert 0.0 < rate < 1.0

    def test_timing_metrics_recorded(self) -> None:
        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config())
        result = scenario.run()
        assert result.status != ScenarioStatus.ERROR
        assert "train_time_s" in result.metrics
        assert "eval_time_s" in result.metrics
        # Times are positive seconds.
        assert result.metrics["train_time_s"] >= 0.0
        assert result.metrics["eval_time_s"] >= 0.0

    def test_voxel_fdm_uses_fdm_supervision(self) -> None:
        """voxel_fdm mode trains on FDM samples, not the harmonic field.

        Verifies the v1 deviation is fixed by checking the cached FDM
        coordinates are returned by ``_sample_voxel_fdm_batch`` and that
        the source supplied to the network is exactly zero (matching the
        homogeneous Laplacian the FDM solver enforces).
        """
        import torch

        cls = _import_scenario_class()
        cfg = _smoke_config().model_copy(
            update={
                "ref_solver_kind": "voxel_fdm",
                "voxel_fdm_resolution": 16,
            }
        )
        scenario = cls(config=cfg)
        scenario.setup()
        try:
            interior, target, source, boundary, b_target = scenario._sample_voxel_fdm_batch(
                n_pts=8, n_boundary_pts=4
            )
            assert interior.shape == (8, 3)
            assert target.shape == (8,)
            # FDM training source must be identically zero.
            assert torch.equal(source, torch.zeros_like(source))
            # Boundary supervision matches the operator's Dirichlet condition,
            # not the harmonic reference. With ``boundary_mode='inner_dirichlet'``
            # the operator returns the configured ``config.boundary_value`` (0.0
            # by default) at every point.
            assert torch.allclose(b_target, torch.zeros_like(b_target))
            # The FDM cache must be populated and the same on a second call.
            (coords_a, u_a) = scenario._voxel_fdm_reference()
            (coords_b, u_b) = scenario._voxel_fdm_reference()
            assert coords_a is coords_b
            assert u_a is u_b
        finally:
            scenario.teardown()

    def test_voxel_fdm_cache_cleared_on_teardown(self) -> None:
        """teardown() must drop the FDM cache so re-runs do not leak state."""
        cls = _import_scenario_class()
        cfg = _smoke_config().model_copy(
            update={
                "ref_solver_kind": "voxel_fdm",
                "voxel_fdm_resolution": 16,
            }
        )
        scenario = cls(config=cfg)
        scenario.setup()
        scenario._voxel_fdm_reference()
        assert scenario._voxel_fdm_cache is not None
        scenario.teardown()
        assert scenario._voxel_fdm_cache is None


class TestNoyronHXScenarioPoolSampling:
    """Regression tests for the shared ``_draw_pool_indices`` helper."""

    def test_without_replacement_when_pool_large(self) -> None:
        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config())
        scenario.setup()
        try:
            idx = scenario._draw_pool_indices(n_pool=128, n_pts=32)
            # ``randperm`` semantics: every index unique.
            assert idx.shape == (32,)
            assert idx.unique().numel() == 32
        finally:
            scenario.teardown()

    def test_with_replacement_when_pool_smaller(self) -> None:
        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config())
        scenario.setup()
        try:
            idx = scenario._draw_pool_indices(n_pool=16, n_pts=128)
            # ``randint`` semantics: shape preserved, indices in range.
            assert idx.shape == (128,)
            assert int(idx.min().item()) >= 0
            assert int(idx.max().item()) < 16
        finally:
            scenario.teardown()

    def test_invalid_n_pool_raises(self) -> None:
        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config())
        scenario.setup()
        try:
            with pytest.raises(ValueError, match="n_pool"):
                scenario._draw_pool_indices(n_pool=0, n_pts=4)
        finally:
            scenario.teardown()

    def test_invalid_n_pts_raises(self) -> None:
        cls = _import_scenario_class()
        scenario = cls(config=_smoke_config())
        scenario.setup()
        try:
            with pytest.raises(ValueError, match="n_pts"):
                scenario._draw_pool_indices(n_pool=16, n_pts=0)
        finally:
            scenario.teardown()


class TestNoyronHXScenarioConstants:
    """Surfaced module constants must be importable and consistent."""

    def test_constants_documented(self) -> None:
        from src.poc.scenarios.noyron_hx import (
            DEFAULT_NORMALIZE_EXTENT_FLOOR,
            DEFAULT_TRANSFER_RATIO_FLOOR,
            EVAL_SEED_STRIDE,
        )

        # All numerical-stability floors must be strictly positive.
        assert DEFAULT_TRANSFER_RATIO_FLOOR > 0
        assert DEFAULT_NORMALIZE_EXTENT_FLOOR > 0
        # The seed stride must be a positive integer; it is multiplied
        # into ``int * int`` arithmetic so non-integers would silently
        # break determinism.
        assert isinstance(EVAL_SEED_STRIDE, int)
        assert EVAL_SEED_STRIDE > 0
