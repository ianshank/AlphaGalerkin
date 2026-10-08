"""``docs/architecture/components.md`` states two derivable facts about ``src/pde``; check them.

Defect class: a hand-copied enumeration of a runtime fact drifts. The PDE bullet
lists the operator registry keys "enumerated at runtime", and that list omitted
``poisson_multi_corner`` from the day the key was registered -- nothing read it.
The same bullet names the modules of ``src/pde/operators/``, and omitted
``multi_corner_poisson.py`` the same way.

Both sides are read: the backticked key list after :data:`REGISTRY_KEYS_MARKER`
against the registry, and every module on disk against the bullet.

**Registry truth is read in a subprocess.** ``PDEOperatorRegistry`` is a
process-wide singleton, and ``tests/pde/test_pde_registry.py`` registers
``test_custom_pde_op`` into it without removing it. An in-process read is
therefore order-dependent -- measured: green alone, red after that file -- the
lesson ``tests/docs/test_charter_alignment.py`` records for the scenario registry.

Planted defects, each killed by a named test (``harden-a-guard``):

* the pre-fix key list (no polyomino presets) ->
  ``test_the_documented_registry_keys_are_the_runtime_keys``;
* a documented key that is not registered -> the same test (equality, both ways);
* the marker reworded, so the scan finds nothing ->
  ``test_the_registry_keys_marker_appears_exactly_once``;
* an operator module dropped from the bullet -> ``test_every_operator_module_is_named``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
COMPONENTS_DOC: Final[Path] = REPO_ROOT / "docs" / "architecture" / "components.md"
OPERATORS_DIR: Final[Path] = REPO_ROOT / "src" / "pde" / "operators"

#: The phrase that introduces the documented key list, then the list itself.
REGISTRY_KEYS_MARKER: Final[str] = "Registry keys, enumerated at runtime: "
REGISTRY_KEYS_PATTERN: Final[re.Pattern[str]] = re.compile(
    re.escape(REGISTRY_KEYS_MARKER) + r"`([^`]*)`"
)

#: The bullet that names the operator modules.
OPERATORS_BULLET_PREFIX: Final[str] = "- **`operators/`**"

#: Modules in ``src/pde/operators/`` today, besides ``__init__.py`` (vacuity floor).
MIN_OPERATOR_MODULES: Final[int] = 11

#: Importing the registry imports torch: generous but bounded, so a hung import
#: fails this guard rather than the session.
REGISTRY_SUBPROCESS_TIMEOUT_S: Final[int] = 300

_LIST_REGISTRY = (
    "import json; from src.pde.registry import list_pde_operators; "
    "print(json.dumps(sorted(list_pde_operators())))"
)


def _doc() -> str:
    return COMPONENTS_DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def registered_keys() -> list[str]:
    """The built-in registry keys, from a fresh interpreter (see the module docstring)."""
    proc = subprocess.run(
        [sys.executable, "-c", _LIST_REGISTRY],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=REGISTRY_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(f"could not enumerate the PDE operator registry:\n{proc.stderr[-2000:]}")
    keys: list[str] = json.loads(proc.stdout.strip().splitlines()[-1])
    return keys


def test_the_registry_keys_marker_appears_exactly_once() -> None:
    """Vacuity: a reworded marker would leave the key check nothing to compare."""
    assert len(REGISTRY_KEYS_PATTERN.findall(_doc())) == 1


def test_the_documented_registry_keys_are_the_runtime_keys(registered_keys: list[str]) -> None:
    (documented,) = REGISTRY_KEYS_PATTERN.findall(_doc())
    keys = [key.strip() for key in documented.split(",")]
    assert keys == registered_keys, (
        f"{COMPONENTS_DOC.name} documents {keys}; the registry holds {registered_keys}. "
        f"Update the list after '{REGISTRY_KEYS_MARKER.strip()}'."
    )


def test_every_operator_module_is_named() -> None:
    modules = sorted(path.name for path in OPERATORS_DIR.glob("*.py") if path.name != "__init__.py")
    assert len(modules) >= MIN_OPERATOR_MODULES, modules
    bullets = [line for line in _doc().splitlines() if line.startswith(OPERATORS_BULLET_PREFIX)]
    assert len(bullets) == 1, f"expected one {OPERATORS_BULLET_PREFIX!r} bullet, got {bullets}"
    missing = [module for module in modules if f"`{module}`" not in bullets[0]]
    assert missing == [], f"{COMPONENTS_DOC.name}'s operators/ bullet omits {missing}"
