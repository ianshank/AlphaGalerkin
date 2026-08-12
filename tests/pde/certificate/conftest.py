"""Shared fixtures for the certificate foundation surface.

The stability registry is a *process-global* singleton — tests that mutate it
(e.g. register-duplicate raises, replace() escape hatch) must reset it between
runs. This ``autouse`` fixture reruns the built-in registration block after
each test, so mutation tests do not leak state into siblings.

The verifier registry (WS1) is likewise a singleton and needs the same
between-test reset so ``@register_verifier`` decorators do not leak sentinel
overrides.
"""

from __future__ import annotations

import pytest

from src.pde.certificate.registry import (
    VerifierRegistry,
)
from src.pde.certificate.registry import (
    _reset_for_tests as _reset_verifier_registry,
)
from src.pde.certificate.stability import (
    StabilityConstantRegistry,
    _register_builtin_stability_entries,
)


@pytest.fixture(autouse=True)
def _reset_registries() -> None:
    """Drop both singletons, then re-populate the built-in entries.

    Reset happens *before* the test so a test can inspect a fresh registry;
    tests that need an empty registry can call the ``_reset_for_tests``
    helpers again.
    """
    StabilityConstantRegistry._reset_for_tests()
    _register_builtin_stability_entries()
    _reset_verifier_registry()
    # Re-register the concrete WS1 verifier under the same class object the
    # tests hold a reference to — importlib.reload would create a *new*
    # class and break isinstance checks. Using the raw registry API keeps
    # the singleton class identity intact.
    from src.pde.certificate.verifiers.heuristic_grid import (
        HeuristicGridResidualVerifier,
    )

    reg = VerifierRegistry()
    if "heuristic_grid" not in reg:
        reg.register("heuristic_grid", HeuristicGridResidualVerifier)
