"""Plugin registration is idempotent and torch-free (CPU).

The harness registries and the adapter's ``plugins`` module (which imports the
harness at module scope) are resolved inside fixtures so this file collects on
a base install; see ``test_contract.py`` for why (R-13).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType

import pytest

pytestmark = pytest.mark.eval_harness_required


@pytest.fixture(scope="module")
def registries() -> ModuleType:
    from eval_harness import plugins

    return plugins


@pytest.fixture(scope="module")
def register_all() -> Callable[[], None]:
    from src.integrations.eval_harness.plugins import register_all

    return register_all


def test_register_all_registers_adapters(
    registries: ModuleType, register_all: Callable[[], None]
) -> None:
    register_all()
    assert "final_residual" in registries.SCORERS
    assert "policy_topk" in registries.SCORERS
    assert "scenario_result" in registries.SINKS
    assert "basis_oracle" in registries.DATASETS
    # The registered classes are constructible via the registry contract.
    assert registries.SCORERS.create("final_residual", {"target_residual": 1e-3}) is not None


def test_register_all_is_idempotent(
    registries: ModuleType, register_all: Callable[[], None]
) -> None:
    register_all()
    register_all()  # second call must not raise (membership-guarded)
    assert "final_residual" in registries.SCORERS


def test_entrypoint_shim_registers_on_import(registries: ModuleType) -> None:
    import importlib

    import src.integrations.eval_harness._entrypoint as entrypoint

    importlib.reload(entrypoint)  # re-run the module body (calls register_all)
    assert "basis_oracle" in registries.DATASETS


def test_register_all_does_not_import_torch(register_all: Callable[[], None]) -> None:
    # In a torch-free environment a stray heavy import would raise; this asserts
    # registration never pulls torch. In CI (torch already loaded) the guarantee
    # can only be checked when torch was absent beforehand.
    had_torch = "torch" in sys.modules
    register_all()
    if not had_torch:
        assert "torch" not in sys.modules
