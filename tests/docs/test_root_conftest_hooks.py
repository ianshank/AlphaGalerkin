"""The root ``conftest.py``'s collection hook may mark a test, and never hides one.

**Defect class:** a test a CI step selects that the collection hook silently drops. Every
test passes through the root hook: it skips ``gpu_required`` tests without CUDA, and gates the
two optional extras -- a counted skip when ``fem_required``'s scikit-fem or
``eval_harness_required``'s harness is missing, a usage error when it is missing under
``ALPHAGALERKIN_REQUIRE_EXTRAS=1``. That is legitimate collection-time code, so
``tests/docs/test_fem_required_visibility.py`` clause (f) cannot forbid the hook outright; it
used to exempt the whole file instead, so ``collect_ignore``, ``pytest_ignore_collect`` or an
``items[:] = ...`` added here hid a module from every step while every check stayed green
(Copilot review 2 of PR #160, reproduced on ``4c20d2a``: 65 passed with both planted).

Clause (f) now reads the root file like any other conftest, allowing only the hook by name;
this module is why that allowance is safe. For each extra's marker, in all four branches --
the extra installed or not, ``ALPHAGALERKIN_REQUIRE_EXTRAS`` set or not -- the hook is run over
items carrying the marker and must:

* never remove, add, reorder or deselect an item (any branch);
* add no marker to a test whose extra is installed;
* add exactly one (the skip) when the extra is missing and the variable is unset;
* raise ``pytest.UsageError`` when the extra is missing and the variable is set.

Hermetic except that it executes copies of the root ``conftest.py``. Planted defects
(``harden-a-guard``, 2026-10-12), each appended to a copy of the live file and reported by
:class:`TestPlantedDefects`; the unmodified copy is the control and must stay clean:

1. Fem items removed only when scikit-fem is installed -- the CI case, invisible to the
   pre-existing drive, which ran only with it missing -> ``[remove-when-installed]``.
   Appended to the live file instead, it fails ``test_the_root_hook_never_hides_a_gated_test
   [fem_required]`` while ``test_fem_required_visibility.py``, which allows the hook by name,
   stays green.
2. Fem items reported deselected and removed -> ``[deselect-and-remove]``.
3. Fem items skipped only when scikit-fem is installed -> ``[skip-when-installed]``.
4. The usage error swallowed -> ``[swallow-the-usage-error]``.
5. The usage error turned into a quiet skip -> ``[usage-error-to-skip]``.
6. Items reordered -> ``[reorder]``.

Each new check of this review was then weakened in place and a named test went red (8/8):
order ignored (``[reorder]``); markers unchecked with the extra installed
(``[skip-when-installed]``); a missing usage error accepted (``[usage-error-to-skip]``); clause
(f)'s walk stopped below the root, as before (``test_a_filter_in_the_root_conftest_is_reported``,
five cases); in ``tests/support/shell_status.py``, ``exec`` never ending the script, a command
in a compound read as top-level (the ``return``-in-a-function and conditional ``set -e``
rows), the reserved words before it left in place (``[exit-inside-if]`` and four rows), and a
redirect-only ``exec`` counted (``[exec-redirect-first]`` and its two rows).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.support.collection_hooks import (
    COLLECTION_HOOK,
    CONFTEST,
    FakeItem,
    load_module,
    rearrangements,
)
from tests.support.workflows import REPO_ROOT

ROOT_CONFTEST: Final[Path] = REPO_ROOT / CONFTEST

#: The variable that turns a missing extra into a usage error, and its "on" value.
REQUIRE_EXTRAS_ENV: Final[str] = "ALPHAGALERKIN_REQUIRE_EXTRAS"
REQUIRE_EXTRAS_ON: Final[str] = "1"

#: Each optional extra's marker -> the root conftest global that says it is installed.
EXTRA_FLAGS: Final[dict[str, str]] = {
    "fem_required": "_HAS_SKFEM",
    "eval_harness_required": "_HAS_EVAL_HARNESS",
}
FEM_MARKER: Final[str] = "fem_required"

#: A marker the hook may skip for its own reason: an extra's test can carry it too.
GPU_MARKER: Final[str] = "gpu_required"

#: The four branches: (extra installed, ``ALPHAGALERKIN_REQUIRE_EXTRAS`` set).
BRANCHES: Final[tuple[tuple[bool, bool], ...]] = (
    (True, False),
    (True, True),
    (False, False),
    (False, True),
)

#: Markers the hook adds to a gated test: none with the extra installed, one skip without.
MARKERS_WHEN_INSTALLED: Final[int] = 0
MARKERS_WHEN_MISSING: Final[int] = 1

#: A wrapper around the live hook, so each plant changes behaviour, never the file's text.
_WRAPPED: Final[str] = (
    "\n\n_live_hook = pytest_collection_modifyitems\n\n\n"
    "def pytest_collection_modifyitems(config, items):\n"
)
_AFTER_LIVE: Final[str] = f"{_WRAPPED}    _live_hook(config, items)\n"
_FEM_ITEMS: Final[str] = f"[i for i in items if i.get_closest_marker({FEM_MARKER!r})]"

#: ``name -> source appended to a copy of the root conftest``; every one must be reported.
HOOK_PLANTS: Final[dict[str, str]] = {
    "remove-when-installed": f"{_AFTER_LIVE}    if _HAS_SKFEM:\n"
    f"        items[:] = [i for i in items if i not in {_FEM_ITEMS}]\n",
    "deselect-and-remove": f"{_AFTER_LIVE}    hidden = {_FEM_ITEMS}\n"
    "    config.hook.pytest_deselected(items=hidden)\n"
    "    items[:] = [i for i in items if i not in hidden]\n",
    "skip-when-installed": f"{_AFTER_LIVE}    for item in {_FEM_ITEMS} if _HAS_SKFEM else []:\n"
    "        item.add_marker(pytest.mark.skip(reason='later'))\n",
    "swallow-the-usage-error": f"{_WRAPPED}    try:\n        _live_hook(config, items)\n"
    "    except pytest.UsageError:\n        pass\n",
    "usage-error-to-skip": f"{_WRAPPED}    try:\n        _live_hook(config, items)\n"
    f"    except pytest.UsageError:\n        for item in {_FEM_ITEMS}:\n"
    "            item.add_marker(pytest.mark.skip(reason='quiet'))\n",
    "reorder": f"{_AFTER_LIVE}    items.reverse()\n",
}


def _branch(marker: str, installed: bool, require: bool) -> str:
    state = "installed" if installed else "missing"
    return f"{marker}, extra {state}, {REQUIRE_EXTRAS_ENV} {'set' if require else 'unset'}"


def hook_violations(conftest: Path, marker: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """How the collection hook in ``conftest`` could hide or mis-gate ``marker``'s tests.

    Every other extra is reported installed, so only ``marker``'s gate is exercised.
    """
    problems: list[str] = []
    for installed, require in BRANCHES:
        if require:
            monkeypatch.setenv(REQUIRE_EXTRAS_ENV, REQUIRE_EXTRAS_ON)
        else:
            monkeypatch.delenv(REQUIRE_EXTRAS_ENV, raising=False)
        module = load_module(conftest, "_root_conftest_hooks_under_test")
        for flag in EXTRA_FLAGS.values():
            monkeypatch.setattr(module, flag, True)
        monkeypatch.setattr(module, EXTRA_FLAGS[marker], installed)
        gated = FakeItem({marker})
        items = [gated, FakeItem(set()), FakeItem({marker, GPU_MARKER})]
        branch = _branch(marker, installed, require)
        try:
            found = rearrangements(getattr(module, COLLECTION_HOOK), items)
        except pytest.UsageError:
            if installed or not require:
                problems.append(f"{branch}: raised a usage error")
            continue
        if require and not installed:
            problems.append(f"{branch}: no usage error -- a missing extra goes quietly green")
        problems.extend(f"{branch}: {problem}" for problem in found)
        expected = MARKERS_WHEN_INSTALLED if installed else MARKERS_WHEN_MISSING
        if len(gated.added) != expected:
            problems.append(f"{branch}: added {len(gated.added)} marker(s), expected {expected}")
    return problems


def _planted(tmp_path: Path, appended: str) -> Path:
    planted = tmp_path / CONFTEST
    live = ROOT_CONFTEST.read_text(encoding="utf-8")
    planted.write_text(f"{live}{appended}", encoding="utf-8")
    return planted


@pytest.mark.parametrize("marker", sorted(EXTRA_FLAGS))
def test_the_root_hook_never_hides_a_gated_test(
    marker: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Driven in all four branches: rearranging items hides them from every step."""
    problems = hook_violations(ROOT_CONFTEST, marker, monkeypatch)
    assert not problems, f"{ROOT_CONFTEST.name}:\n  " + "\n  ".join(problems)


def test_every_extra_the_hook_gates_is_driven() -> None:
    """Vacuity: a flag renamed in the conftest would leave its marker's drive testing nothing.

    ``monkeypatch.setattr`` already raises on a missing attribute; this names the cause.
    """
    module = load_module(ROOT_CONFTEST, "_root_conftest_hooks_flags")
    assert all(hasattr(module, flag) for flag in EXTRA_FLAGS.values()), EXTRA_FLAGS
    assert callable(getattr(module, COLLECTION_HOOK, None))


class TestPlantedDefects:
    """Each plant changes the hook's behaviour on a copy; each must be reported."""

    def test_the_unmodified_copy_is_clean(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert not hook_violations(_planted(tmp_path, ""), FEM_MARKER, monkeypatch)

    @pytest.mark.parametrize("plant", sorted(HOOK_PLANTS))
    def test_a_hook_that_could_hide_a_test_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plant: str
    ) -> None:
        planted = _planted(tmp_path, HOOK_PLANTS[plant])
        assert hook_violations(planted, FEM_MARKER, monkeypatch), f"{plant} went unreported"
