"""Plugin registration is idempotent and torch-free (CPU)."""

from __future__ import annotations

import sys

from eval_harness.plugins import DATASETS, SCORERS, SINKS

from src.integrations.eval_harness.plugins import register_all


def test_register_all_registers_adapters() -> None:
    register_all()
    assert "final_residual" in SCORERS
    assert "policy_topk" in SCORERS
    assert "scenario_result" in SINKS
    assert "basis_oracle" in DATASETS
    # The registered classes are constructible via the registry contract.
    assert SCORERS.create("final_residual", {"target_residual": 1e-3}) is not None


def test_register_all_is_idempotent() -> None:
    register_all()
    register_all()  # second call must not raise (membership-guarded)
    assert "final_residual" in SCORERS


def test_entrypoint_shim_registers_on_import() -> None:
    import importlib

    import src.integrations.eval_harness._entrypoint as entrypoint

    importlib.reload(entrypoint)  # re-run the module body (calls register_all)
    assert "basis_oracle" in DATASETS


def test_register_all_does_not_import_torch() -> None:
    # In a torch-free environment a stray heavy import would raise; this asserts
    # registration never pulls torch. In CI (torch already loaded) the guarantee
    # can only be checked when torch was absent beforehand.
    had_torch = "torch" in sys.modules
    register_all()
    if not had_torch:
        assert "torch" not in sys.modules
