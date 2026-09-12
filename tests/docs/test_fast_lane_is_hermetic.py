"""Guards the hermetic fast lane (plan R-02).

The blocking fast lane -- CI's ``test-fast`` and ``coverage`` steps and the
Makefile's ``test-fast`` / ``coverage`` recipes -- runs under ``pytest-socket``
so that a test which reaches the network fails visibly instead of passing on
whatever egress the runner happens to have. Hygiene B35 is the case: two
"unit" tests downloaded ImageNet weights and were green on GitHub for months
while failing in every sandbox.

Three copies of the flag set exist by construction (a workflow-level ``env``
value in ``ci.yml``, a Makefile variable, and the canonical tuple below); this
module keeps them equal, checks the ``network`` marker is registered and
deselected, checks ``pytest-socket`` is an installed dev dependency, and --
because a flag that is present but inert would pass every textual check --
drives a planted test in a subprocess to prove an outbound connection is
actually blocked while loopback still works.

Mutation kills (each planted, run, reverted):

* drop ``$(HERMETIC_PYTEST_FLAGS)`` from the Makefile ``test-fast`` recipe ->
  ``test_makefile_fast_lane_recipes_apply_the_flags[test-fast]``
* drop ``pytest-socket`` from the ``dev`` extra ->
  ``test_pytest_socket_is_a_dev_dependency``
* drop ``and not network`` from the coverage step's ``-m`` ->
  ``test_ci_fast_lane_steps_deselect_network_tests[coverage]``
* change the ``env`` value to ``--allow-unix-socket`` only ->
  ``test_ci_env_flags_match_canonical``
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from tests.docs.test_marker_vocabulary import registered_markers
from tests.support.workflows import (
    REPO_ROOT,
    iter_commands,
    load_workflow,
    makefile_target_recipe,
)

CI_WORKFLOW: Final[Path] = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE: Final[Path] = REPO_ROOT / "Makefile"
PYPROJECT: Final[Path] = REPO_ROOT / "pyproject.toml"

#: The one canonical flag set. ci.yml and the Makefile carry copies.
HERMETIC_FLAGS: Final[tuple[str, ...]] = (
    "--disable-socket",
    "--allow-unix-socket",
    "--allow-hosts=127.0.0.1,localhost",
)

#: Name of the workflow-level ``env`` entry and of the Makefile variable.
FLAGS_VARIABLE: Final[str] = "HERMETIC_PYTEST_FLAGS"

#: Marker that opts a test out of the hermetic lane.
NETWORK_MARKER: Final[str] = "network"

#: ``(job, step)`` pairs that form the blocking fast lane in ``ci.yml``.
CI_FAST_LANE_STEPS: Final[tuple[tuple[str, str], ...]] = (
    ("test-fast", "Run fast unit tests"),
    ("coverage", "Run tests with coverage"),
)

_STEP_IDS: Final[list[str]] = [job for job, _ in CI_FAST_LANE_STEPS]

#: Makefile targets that mirror those steps.
MAKEFILE_FAST_LANE_TARGETS: Final[tuple[str, ...]] = ("test-fast", "coverage")

#: An address no sandbox or runner should be able to reach under the flags.
BLOCKED_PROBE_HOST: Final[str] = "203.0.113.1"  # TEST-NET-3, never routable
BLOCKED_PROBE_PORT: Final[int] = 9
PROBE_TIMEOUT_S: Final[float] = 1.0
SUBPROCESS_TIMEOUT_S: Final[float] = 120.0

#: pytest-socket's two refusal exceptions (creation-time and connect-time).
BLOCKED_ERROR_NAMES: Final[tuple[str, ...]] = ("SocketBlockedError", "SocketConnectBlockedError")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _ci_document() -> dict[str, object]:
    return load_workflow(CI_WORKFLOW)


def _ci_env_flags() -> str:
    env = _ci_document().get("env")
    assert isinstance(env, dict), "ci.yml has no workflow-level env block"
    value = env.get(FLAGS_VARIABLE)
    assert isinstance(value, str), f"ci.yml env has no {FLAGS_VARIABLE}"
    return value


def _ci_step_script(job: str, step: str) -> str:
    jobs = _ci_document().get("jobs")
    assert isinstance(jobs, dict)
    job_doc = jobs.get(job)
    assert isinstance(job_doc, dict), f"ci.yml has no job {job!r}"
    for candidate in job_doc.get("steps", []):
        if isinstance(candidate, dict) and candidate.get("name") == step:
            script = candidate.get("run")
            assert isinstance(script, str), f"{job}::{step} has no run: script"
            return script
    raise AssertionError(f"ci.yml job {job!r} has no step named {step!r}")


def _makefile_variable(name: str) -> str:
    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith((f"{name} ?=", f"{name} :=", f"{name} =")):
            return stripped.split("=", 1)[1].strip()
    raise AssertionError(f"Makefile defines no variable {name}")


#: The ``dev = [ ... ]`` list inside ``[project.optional-dependencies]``.
#: Anchored regex rather than ``tomllib``, following the sibling
#: ``tests/docs/test_marker_vocabulary.py``: ``tomllib`` is stdlib only from
#: 3.11 and this repo's declared floor is 3.10, so importing it here would be
#: a *collection* error on the oldest supported interpreter -- the exact
#: defect ``tests/docs/test_python_floor_compatibility.py`` exists to catch,
#: and it caught the first draft of this file.
_DEV_EXTRA_BLOCK: Final[re.Pattern[str]] = re.compile(r"^dev\s*=\s*\[(?P<body>.*?)^\]", re.M | re.S)
_REQUIREMENT_ENTRY: Final[re.Pattern[str]] = re.compile(r'^\s*"(?P<entry>[^"]+)",?\s*$')


def _dev_requirements() -> list[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = _DEV_EXTRA_BLOCK.search(text)
    assert match, f"could not locate the `dev` extra in {PYPROJECT.name}"
    entries = [
        entry.group("entry")
        for entry in map(_REQUIREMENT_ENTRY.match, match.group("body").splitlines())
        if entry is not None
    ]
    assert entries, "the `dev` extra parsed empty -- the dependency check would be inert"
    return entries


def _registered_markers() -> set[str]:
    return registered_markers(PYPROJECT)


def _marker_expression(script_or_command: str) -> str:
    """Return the quoted ``-m`` expression in a pytest command line."""
    text = script_or_command.replace("\\\n", " ")
    marker = ' -m "'
    start = text.find(marker)
    assert start != -1, f"no -m expression in: {text[:120]}..."
    start += len(marker)
    end = text.find('"', start)
    return text[start:end]


def _run_probe(
    tmp_path: Path, body: str, *, flags: tuple[str, ...]
) -> subprocess.CompletedProcess[str]:
    """Run one planted test file under a bare pytest with ``flags``.

    The probe runs from ``tmp_path`` with its own ``pytest.ini`` so the
    repository's ``conftest.py`` and ``addopts`` play no part; ``pytest-socket``
    registers through its entry point, so no ``-p`` is needed.
    """
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts =\n", encoding="utf-8")
    (tmp_path / "test_probe.py").write_text(body, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *flags, "test_probe.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )


# --------------------------------------------------------------------------
# Copies agree
# --------------------------------------------------------------------------


def test_canonical_flags_are_non_empty() -> None:
    """Vacuity: an empty canonical set would make every equality below trivial."""
    assert HERMETIC_FLAGS
    assert "--disable-socket" in HERMETIC_FLAGS


def test_ci_env_flags_match_canonical() -> None:
    assert tuple(_ci_env_flags().split()) == HERMETIC_FLAGS


def test_makefile_flags_match_canonical() -> None:
    assert tuple(_makefile_variable(FLAGS_VARIABLE).split()) == HERMETIC_FLAGS


def _pytest_commands(script: str) -> list[str]:
    """The pytest invocations in a step script, one logical command each.

    Comments stripped, continuations joined, separators split -- so a flag
    that appears only in an ``echo`` or a comment is not credited to the
    ``pytest`` command (Copilot review, PR #151).
    """
    return [c for c in iter_commands(script) if "pytest" in c.split()[:4]]


@pytest.mark.parametrize(("job", "step"), CI_FAST_LANE_STEPS, ids=_STEP_IDS)
def test_ci_fast_lane_steps_apply_the_flags(job: str, step: str) -> None:
    commands = _pytest_commands(_ci_step_script(job, step))
    assert commands, f"{job}::{step} runs no pytest command"
    expansion = f"${{{{ env.{FLAGS_VARIABLE} }}}}"
    for command in commands:
        assert expansion in command, (
            f"{job}::{step}: the pytest command does not expand {expansion} "
            f"(it may appear elsewhere in the step, which is not the same thing): {command[:120]}"
        )


@pytest.mark.parametrize(("job", "step"), CI_FAST_LANE_STEPS, ids=_STEP_IDS)
def test_ci_fast_lane_steps_deselect_network_tests(job: str, step: str) -> None:
    expression = _marker_expression(_ci_step_script(job, step))
    assert f"not {NETWORK_MARKER}" in expression, (
        f"{job}::{step} selects `{NETWORK_MARKER}`-marked tests into a socketless lane"
    )


@pytest.mark.parametrize("target", MAKEFILE_FAST_LANE_TARGETS)
def test_makefile_fast_lane_recipes_apply_the_flags(target: str) -> None:
    recipe = makefile_target_recipe(target)
    assert recipe, f"Makefile target {target!r} has no recipe"
    # Recipes invoke `$(PYTEST)` (the Makefile resolves tools through
    # `$(PYTHON) -m`), so match case-insensitively.
    pytest_commands = [c for c in recipe if "pytest" in c.lower()]
    assert pytest_commands, f"Makefile target {target!r} runs no pytest command"
    for command in pytest_commands:
        assert f"$({FLAGS_VARIABLE})" in command, (
            f"Makefile {target!r} does not apply $({FLAGS_VARIABLE}); `make {target}` "
            "would certify a PR with sockets open"
        )
        assert f"not {NETWORK_MARKER}" in _marker_expression(command)


# --------------------------------------------------------------------------
# Dependency and marker registration
# --------------------------------------------------------------------------


def test_pytest_socket_is_a_dev_dependency() -> None:
    names = [
        req.split(">")[0].split("=")[0].split("<")[0].strip().lower() for req in _dev_requirements()
    ]
    assert "pytest-socket" in names, (
        "pytest-socket missing from the `dev` extra: the flags would be unknown options"
    )


def test_network_marker_is_registered() -> None:
    assert NETWORK_MARKER in _registered_markers(), (
        f"`{NETWORK_MARKER}` is not in [tool.pytest.ini_options].markers; "
        "--strict-markers rejects it"
    )


# --------------------------------------------------------------------------
# The flags actually block
# --------------------------------------------------------------------------


def test_outbound_connection_is_blocked_under_the_flags(tmp_path: Path) -> None:
    """A planted test that opens an outbound socket must fail with SocketBlockedError."""
    body = (
        "import socket\n"
        "def test_probe():\n"
        f"    socket.create_connection(({BLOCKED_PROBE_HOST!r}, {BLOCKED_PROBE_PORT}), "
        f"timeout={PROBE_TIMEOUT_S})\n"
    )
    result = _run_probe(tmp_path, body, flags=HERMETIC_FLAGS)
    assert result.returncode != 0, (
        f"outbound socket was not blocked:\n{result.stdout}\n{result.stderr}"
    )
    # `--disable-socket` alone raises SocketBlockedError at socket creation;
    # with `--allow-hosts` the socket is created and the *connect* is refused
    # with SocketConnectBlockedError. Either proves the lane is closed.
    output = result.stdout + result.stderr
    assert any(name in output for name in BLOCKED_ERROR_NAMES), output


def test_loopback_bind_still_works_under_the_flags(tmp_path: Path) -> None:
    """The allow-list keeps localhost usable (gloo, free-port helpers, dashboards)."""
    body = (
        "import socket\n"
        "def test_probe():\n"
        "    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:\n"
        "        s.bind(('127.0.0.1', 0))\n"
        "        assert s.getsockname()[1] > 0\n"
    )
    result = _run_probe(tmp_path, body, flags=HERMETIC_FLAGS)
    assert result.returncode == 0, f"loopback bind was blocked:\n{result.stdout}\n{result.stderr}"
