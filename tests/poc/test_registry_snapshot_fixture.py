"""Tests for the ``tests/poc/conftest.py`` registry snapshot/restore fixture.

The ``_snapshot_scenario_registry`` autouse fixture is test *infrastructure*
guarding a known order-dependence hazard: ``ScenarioRegistry`` is a
process-wide singleton, several modules in this package clear it without
restoring, and ``register_builtin_scenarios()`` cannot repair the damage
because it only imports modules that are already in ``sys.modules`` (so the
``@scenario`` decorators never re-run). Infrastructure that silently stops
working is worse than no infrastructure, so it gets its own tests.

Validates:
    - A registration made inside a test does not leak into the next test.
    - A registry a test *clears* is restored for the next test.
    - The failure mode the fixture exists to prevent is real (a cleared
      import-time registration cannot be re-created by re-importing).
    - The restore loop cannot raise on ``register``'s duplicate check,
      including when a test rebinds a snapshot name to a different class.

The ordering-sensitive tests are written as a sequence inside one class and
rely on pytest's definition-order execution; each also passes in isolation
(the baseline fixture re-captures whatever state it finds).
"""

from __future__ import annotations

import pytest

# Imported for its import-time ``@scenario`` registrations: this guarantees the
# built-ins are in ``sys.modules`` (and hence NOT re-registrable) before the
# ``register_builtin_scenarios`` regression test below runs, whether or not any
# other test module in the package has already imported them.
import src.poc.scenarios  # noqa: F401
from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.registry import BaseScenario, ScenarioRegistry

_PROBE_NAME = "conftest_snapshot_probe"
"""Registry key used by the leak probe; must never survive a test."""


class _ProbeScenario(BaseScenario):
    """Minimal concrete scenario used purely as a registry payload."""

    def execute(self) -> ScenarioResult:
        """Return an immediately-passing result."""
        return self._create_result(ScenarioStatus.PASSED)


class _OtherProbeScenario(BaseScenario):
    """A second payload class, distinguishable from :class:`_ProbeScenario`."""

    def execute(self) -> ScenarioResult:
        """Return an immediately-passing result."""
        return self._create_result(ScenarioStatus.PASSED)


@pytest.fixture(scope="module")
def registry_baseline() -> frozenset[str]:
    """Registry contents as this module found them.

    Module-scoped, so it is instantiated before the function-scoped conftest
    fixture on the first test that requests it — it therefore records the
    state the package is expected to be restored to.
    """
    return frozenset(ScenarioRegistry().list_scenarios())


class TestSnapshotRestoreAcrossTests:
    """Ordering-sensitive proof that the fixture closes the leak."""

    def test_1_registration_is_visible_within_its_own_test(
        self, registry_baseline: frozenset[str]
    ) -> None:
        """A scenario registered mid-test is live for the rest of that test."""
        registry = ScenarioRegistry()
        assert _PROBE_NAME not in registry_baseline
        registry.register(_PROBE_NAME, _ProbeScenario)
        assert registry.get(_PROBE_NAME) is _ProbeScenario

    def test_2_registration_did_not_leak_into_the_next_test(
        self, registry_baseline: frozenset[str]
    ) -> None:
        """The probe from the previous test was rolled back."""
        registry = ScenarioRegistry()
        assert registry.get(_PROBE_NAME) is None
        assert frozenset(registry.list_scenarios()) == registry_baseline

    def test_3_a_test_may_clear_the_registry_destructively(self) -> None:
        """``clear()`` inside a test really does empty the singleton."""
        registry = ScenarioRegistry()
        registry.clear()
        assert registry.list_scenarios() == []

    def test_4_cleared_registry_was_restored(self, registry_baseline: frozenset[str]) -> None:
        """The registry the previous test wiped is back, class-for-class."""
        registry = ScenarioRegistry()
        assert frozenset(registry.list_scenarios()) == registry_baseline

    def test_5_rebinding_a_snapshot_name_does_not_break_restore(
        self, registry_baseline: frozenset[str]
    ) -> None:
        """Rebind an existing key to a different class after clearing.

        This is the only shape that could plausibly trip ``register``'s
        duplicate check during teardown. It cannot, because the fixture
        ``clear()``s before replaying the snapshot — asserted by the next test.
        """
        registry = ScenarioRegistry()
        name = sorted(registry_baseline)[0]
        original = registry.get(name)
        assert original is not None
        registry.clear()
        registry.register(name, _OtherProbeScenario)
        assert registry.get(name) is _OtherProbeScenario

    def test_6_rebound_name_was_restored_to_its_original_class(
        self, registry_baseline: frozenset[str]
    ) -> None:
        """Restore replaces the rebound class, and teardown did not raise."""
        registry = ScenarioRegistry()
        name = sorted(registry_baseline)[0]
        assert registry.get(name) is not _OtherProbeScenario
        assert frozenset(registry.list_scenarios()) == registry_baseline


class TestHazardTheFixtureGuards:
    """The failure mode is real, not hypothetical."""

    def test_import_time_registration_cannot_be_redone_after_clear(self) -> None:
        """``register_builtin_scenarios()`` cannot repair a cleared registry.

        It only executes ``from src.poc.scenarios import ...``; the modules are
        already in ``sys.modules``, so the ``@scenario`` decorators do not run
        again. Without the snapshot fixture, one ``clear()`` in any test would
        permanently strip the built-ins for every later test in the process.
        """
        from src.poc.cli import register_builtin_scenarios

        registry = ScenarioRegistry()
        registry.clear()
        register_builtin_scenarios()

        assert registry.get("transfer") is None
        assert registry.list_scenarios() == []

    def test_builtins_survive_the_unrecoverable_clear(self) -> None:
        """The built-ins the previous test destroyed are back."""
        registry = ScenarioRegistry()
        assert registry.get("transfer") is not None
        assert registry.get("complexity") is not None
        assert registry.get("stability") is not None


class TestRestoreLoopSemantics:
    """Unit-level properties of the clear-then-replay restore loop."""

    @staticmethod
    def _restore(snapshot: dict[str, type[BaseScenario]]) -> None:
        """Mirror of the conftest teardown body."""
        registry = ScenarioRegistry()
        registry.clear()
        for name, scenario_cls in snapshot.items():
            registry.register(name, scenario_cls)

    def test_restore_is_idempotent(self) -> None:
        """Replaying the same snapshot twice does not raise on duplicates.

        ``ScenarioRegistry.register`` raises ``ValueError`` for a name that is
        already present, so a restore loop that forgot to ``clear()`` first
        would explode during teardown. The leading ``clear()`` makes a
        duplicate impossible: a snapshot is a ``dict``, so its keys are unique
        by construction.
        """
        snapshot = ScenarioRegistry().get_all()
        self._restore(snapshot)
        self._restore(snapshot)
        assert frozenset(ScenarioRegistry().list_scenarios()) == frozenset(snapshot)

    def test_restore_recovers_from_an_emptied_registry(self) -> None:
        """A wiped registry is fully rebuilt from the snapshot."""
        snapshot = ScenarioRegistry().get_all()
        ScenarioRegistry().clear()
        self._restore(snapshot)
        assert ScenarioRegistry().get_all() == snapshot

    def test_restore_drops_registrations_absent_from_the_snapshot(self) -> None:
        """Restore is a *replacement*, not a merge — extras are dropped."""
        snapshot = ScenarioRegistry().get_all()
        ScenarioRegistry().register(_PROBE_NAME, _ProbeScenario)
        self._restore(snapshot)
        assert ScenarioRegistry().get(_PROBE_NAME) is None

    def test_snapshot_is_a_defensive_copy(self) -> None:
        """``get_all()`` returns a copy, so later writes cannot poison it."""
        snapshot = ScenarioRegistry().get_all()
        ScenarioRegistry().register(_PROBE_NAME, _ProbeScenario)
        assert _PROBE_NAME not in snapshot
