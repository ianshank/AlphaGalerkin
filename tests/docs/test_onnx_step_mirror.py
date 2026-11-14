"""CI's blocking ONNX step and its ``make test-onnx`` mirror run the same check the same way.

Defect class: the ONNX export suites passing on nothing, or a local mirror that
certifies a different selection than CI. The step ran under ``continue-on-error``
for months while logging "24 failed" (B45), and the Makefile had no mirror at all
(B54 (c)).

The CI step is read with the repo's shared models, not a private parser:
``pytest_invocation`` (only a real entry point counts, so ``echo pytest`` does not;
``--ignore``/``--deselect``/``-m``/``-k`` are read) and ``step_failure_obstacles``
(``continue-on-error``, a skippable ``if:``, ``|| true``, ``set +e``, a pipe without
``pipefail``...). The Makefile mirror is *driven*: ``make test-onnx`` runs with stub
``PYTHON``/``PYTEST`` that record their arguments and fail, so a ``-`` prefix, a
masking ``|| true`` or ``; true``, an ``echo`` or a dropped suite fails here by
behaviour rather than by spelling (harden-a-guard Step 6). The stubs run through
``sys.executable``, so the drive works on any OS with ``make``.

Planted defects (harden-a-guard), recorded with the named killing test. The first
version of this file passed on items 5-9; the final review of PR #165 found them:

1. ``continue-on-error: true`` on the step: ``test_each_ci_command_can_fail_the
   _step``.
2. The import check deleted from the step: ``test_the_ci_step_checks_the_onnx
   _stack_before_pytest``.
3. A suite dropped from the Makefile mirror: ``test_the_makefile_mirror_runs_ci
   _s_selection_and_fails_with_it``.
4. The import check deleted from the Makefile mirror: ``test_a_failing_import
   _check_stops_the_mirror``.
5. ``|| true`` after the step's pytest: ``test_each_ci_command_can_fail_the_step``.
6. ``echo`` in front of the step's pytest: ``test_the_ci_step_runs_exactly_the
   _onnx_suites``.
7. ``--ignore=tests/integration/test_onnx_e2e.py`` on the step: ``test_the_ci
   _step_runs_exactly_the_onnx_suites``.
8. Make's ``-`` prefix on the mirror's pytest line: ``test_the_makefile_mirror
   _runs_ci_s_selection_and_fails_with_it``.
9. ``; true`` after the mirror's pytest: ``test_the_makefile_mirror_runs_ci_s
   _selection_and_fails_with_it``.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.marker_expr import is_python_program, pytest_invocation, split_environment
from tests.support.shell_status import step_failure_obstacles
from tests.support.workflows import (
    CI_WORKFLOW,
    REPO_ROOT,
    ShellCommand,
    WorkflowStep,
    blocking_jobs,
    iter_shell_commands,
    load_workflow,
    workflow_steps,
)

ONNX_JOB = "test-extras"
ONNX_STEP = "Run ONNX export/validate suites"
MAKE_TARGET = "test-onnx"
IMPORT_CHECK_MODULES = ("onnx", "onnxruntime", "onnxscript")
# The suites the step must select; written out so the step cannot agree with itself.
EXPECTED_SUITES = (
    "tests/deployment/test_export_onnx_integration.py",
    "tests/deployment/test_onnx_runtime_regressions.py",
    "tests/integration/test_onnx_e2e.py",
)
STUB_PYTEST_EXIT = 3
STUB_IMPORT_FAIL_EXIT = 4
MAKE_TIMEOUT_S = 120


def _step() -> WorkflowStep:
    steps = [s for s in workflow_steps(CI_WORKFLOW) if s.job == ONNX_JOB and s.name == ONNX_STEP]
    assert len(steps) == 1, f"expected one {ONNX_STEP!r} step in {ONNX_JOB}, found {len(steps)}"
    return steps[0]


def _commands() -> list[ShellCommand]:
    return iter_shell_commands(_step().script or "")


def _pytest_positions(commands: list[ShellCommand]) -> list[int]:
    return [i for i, c in enumerate(commands) if pytest_invocation(c.text) is not None]


def _import_check_positions(commands: list[ShellCommand]) -> list[int]:
    """Commands that run a Python interpreter on ``-c`` code importing the whole ONNX stack."""
    found = []
    for index, command in enumerate(commands):
        try:
            _, program = split_environment(shlex.split(command.text))
        except ValueError:
            continue
        if not program or not is_python_program(program[0]) or "-c" not in program:
            continue
        code_at = program.index("-c") + 1
        code = program[code_at] if code_at < len(program) else ""
        if "import" in code and all(module in code for module in IMPORT_CHECK_MODULES):
            found.append(index)
    return found


def test_each_ci_command_can_fail_the_step() -> None:
    step, commands = _step(), _commands()
    positions = _pytest_positions(commands) + _import_check_positions(commands)
    assert positions, "the step runs neither pytest nor the import check"
    for position in positions:
        obstacles = step_failure_obstacles(step, commands, position)
        assert not obstacles, f"`{commands[position].text}` cannot fail the step: {obstacles}"
    assert ONNX_JOB in blocking_jobs(load_workflow(CI_WORKFLOW)), f"{ONNX_JOB} blocks no merge"


def test_the_ci_step_runs_exactly_the_onnx_suites() -> None:
    commands = _commands()
    positions = _pytest_positions(commands)
    assert len(positions) == 1, f"expected one real pytest invocation, found {len(positions)}"
    invocation = pytest_invocation(commands[positions[0]].text)
    assert invocation is not None
    assert tuple(sorted(invocation.paths)) == tuple(sorted(EXPECTED_SUITES))
    assert not (invocation.ignores or invocation.ignore_globs or invocation.deselects), invocation
    assert not invocation.marker_expression and not invocation.keyword_expression, invocation


def test_the_ci_step_checks_the_onnx_stack_before_pytest() -> None:
    commands = _commands()
    checks = _import_check_positions(commands)
    assert checks, "no ONNX import check: a missing stack would turn every test into a skip"
    assert min(checks) < min(_pytest_positions(commands))


def _drive_make(tmp_path: Path, import_exit: int) -> tuple[int, list[str] | None]:
    """Run ``make test-onnx`` with recording stubs; return make's status and pytest's argv."""
    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    record = tmp_path / "pytest_argv.txt"
    stub_pytest = tmp_path / "stub_pytest.py"
    stub_pytest.write_text(
        "import sys\n"
        f"open({str(record)!r}, 'w').write('\\n'.join(sys.argv[1:]))\n"
        f"sys.exit({STUB_PYTEST_EXIT})\n",
        encoding="utf-8",
    )
    stub_python = tmp_path / "stub_python.py"
    stub_python.write_text(f"import sys\nsys.exit({import_exit})\n", encoding="utf-8")
    interpreter = Path(sys.executable).as_posix()
    result = subprocess.run(
        [
            "make",
            "-s",
            MAKE_TARGET,
            f'PYTEST="{interpreter}" "{stub_pytest.as_posix()}"',
            f'PYTHON="{interpreter}" "{stub_python.as_posix()}"',
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=MAKE_TIMEOUT_S,
    )
    argv = record.read_text(encoding="utf-8").splitlines() if record.exists() else None
    return result.returncode, argv


def test_the_makefile_mirror_runs_ci_s_selection_and_fails_with_it(tmp_path: Path) -> None:
    status, argv = _drive_make(tmp_path, import_exit=0)
    assert argv is not None, f"`make {MAKE_TARGET}` never ran pytest"
    assert tuple(sorted(a for a in argv if not a.startswith("-"))) == tuple(sorted(EXPECTED_SUITES))
    assert status != 0, f"`make {MAKE_TARGET}` exited 0 although pytest failed"


def test_a_failing_import_check_stops_the_mirror(tmp_path: Path) -> None:
    status, argv = _drive_make(tmp_path, import_exit=STUB_IMPORT_FAIL_EXIT)
    assert status != 0, f"`make {MAKE_TARGET}` exited 0 although the ONNX import check failed"
    assert argv is None, "pytest ran although the ONNX stack is missing: every test would skip"
