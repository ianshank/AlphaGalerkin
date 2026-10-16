"""``tests/support/shell_status.py``'s exit-status model, row by row and against the real shell.

Every guard that asks "does this command's failure fail its step" asks
:func:`~tests.support.shell_status.exit_status_obstacles`: the ``fem_required`` guard's clause
(d) and the E2E visibility guard's clause (j). A model that calls a masked command loud lets
those guards pass a step that cannot fail; each of Copilot's three reviews of PR #160 found a
new instance (``|| true``; ``exec true`` first; ``false && set -e`` and ``trap 'exit 0'
EXIT``). So :data:`STATUS_CASES` is checked twice:

* against the model, each row's verdict, masked or loud;
* against the shell GitHub runs (:data:`~tests.support.shell_status.RUNNER_ARGV`) with a
  ``pytest`` that fails. Wherever the model finds no obstacle, the script must exit non-zero
  with ``CI`` unset and set -- a direct soundness check of the model, which fails on both of
  review 3's shell findings even if a row's expectation were also wrong. And every masked row
  must exit 0 in one of those states, or carry a reason in :data:`CONSERVATIVE_ROWS`: the
  masks the model reports are real, and where it over-reports, the file says why.

Moved here from ``tests/docs/test_marker_vocabulary.py`` on 2026-10-15, which stays hermetic:
this file runs ``bash`` and ``sh`` on scripts in a temporary directory, with a stub ``pytest``
that exits :data:`STUB_STATUS` and nothing else on the path changed.

Review 3's checks weakened in place, each failing a named test here (2026-10-15):

1. A conditional ``set`` restoring options (``_surely_runs_here`` ignored) -> the
   ``false && set -e``, ``true || set -e`` and ``set -e | cat`` rows, and the soundness check.
2. ``trap`` ignored -> the three trap rows and the soundness check.
3. ``shopt`` ignored -> ``shopt -uo errexit`` and the ``shopt -so errexit`` control.
4. ``eval`` / ``source`` / ``.`` ignored -> their three rows.
5. ``builtin`` / ``command`` not seen through -> their rows and ``command exec true``.
6. ``command -v`` run as the name it describes -> the ``command -v exit`` control.
7. A function head not read -> the ``function NAME`` and ``NAME ()`` rows.
8. A function, alias or ``hash -p`` entry named like the command not reported -> the six
   shadowing rows.
9. A lone ``-`` read as an option -> the ``trap - ERR EXIT`` control.
10. ``hash`` read without its ``-p`` -> the ``hash pytest`` control, which only looks the
    name up; ``hash`` ignored -> the ``hash -p`` row.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final

import pytest

from tests.support.shell_status import RUNNER_ARGV, SHELL_PIPEFAIL, exit_status_obstacles
from tests.support.workflows import iter_shell_commands

#: ``(script, index of the command under test, shell:, whether its failure fails the step)``.
#: Copilot review, PR #160: the fem guard read only a step's metadata, so ``pytest ... || true``
#: counted as a loud test step. ``; true`` is *not* a mask under GitHub's ``bash -e`` -- the shell
#: exits at ``pytest`` -- so it is a control here, and ``set +e`` is what makes it one.
STATUS_CASES: Final[tuple[tuple[str, int, str | None, bool], ...]] = (
    ("pytest x", 0, None, True),
    ("pytest x; true", 0, None, True),
    ("pytest x || true", 0, None, False),
    ("pytest x ||\n  true", 0, None, False),
    ("set +e\npytest x\ntrue", 1, None, False),
    ("set +o errexit\npytest x; true", 1, None, False),
    ("set +e\npytest x", 1, None, True),
    ("set -- a -e\npytest x; true", 1, None, True),
    ("set +e\nset -euo pipefail\npytest x | tee log\ntrue", 2, None, True),
    ("pytest x | tee log", 0, None, False),
    ("pytest x | tee log", 0, "bash", True),
    ("echo | pytest x", 1, None, True),
    ("true || pytest x", 1, None, False),
    ("cd d && pytest x", 1, None, True),
    ("cd d && pytest x\necho done", 1, None, False),
    ("pytest x && echo ok", 0, None, True),
    ("pytest x && echo ok\necho done", 0, None, False),
    ("exit 0\npytest x", 1, None, False),
    ("exit 1\npytest x", 1, None, True),
    ('if [ -n "$CI" ]; then\n  pytest x\nfi', 2, None, False),
    ('if [ -n "$CI" ]; then\n  true\nfi\npytest x', 4, None, True),
    ("run() {\n  pytest x\n}", 1, None, False),
    ("pytest x &\nwait", 0, None, False),
    ("pytest x", 0, "sh", True),
    ("pytest x", 0, "pwsh", False),
    # Copilot review 2 (PR #160): an `exec` naming a command replaces the shell; one that only
    # redirects does not. Then the same early end, and a weakening `set`, inside a compound.
    ("exec true\npytest x", 1, None, False),
    ("exec -a name true\npytest x", 1, None, False),
    ("exec 2>&1\npytest x", 1, None, True),
    ("exec > >(tee log) 2>&1\npytest x", 1, None, True),
    ('if [ -n "$CI" ]; then\n  exit 0\nfi\npytest x', 4, None, False),
    ('if [ -n "$CI" ]; then exit 0; fi\npytest x', 3, None, False),
    ("{ exec true; }\npytest x", 2, None, False),
    ('case "$X" in\n  a) exit 0 ;;\nesac\npytest x', 3, None, False),
    ('if [ -n "$CI" ]; then\n  exit 1\nfi\npytest x', 4, None, True),
    ("die() {\n  return 0\n}\npytest x", 3, None, True),
    ('if [ -n "$CI" ]; then set +e; fi\npytest x; true', 3, None, False),
    ('set +e\nif [ -n "$CI" ]; then\n  set -e\nfi\npytest x; true', 5, None, False),
    # Copilot review 3 (PR #160): a `set -e` that may not run, or runs in a subshell, restores
    # nothing; a trap that installs an action can replace the status; and the neighbours --
    # `shopt -o`, code the model cannot read, a wrapped builtin, a function head in either
    # spelling, and a function or alias named like the command.
    ("set +e\nfalse && set -e\npytest x; true", 3, None, False),
    ("set +e\ntrue || set -e\npytest x; true", 3, None, False),
    ("set +e\ntrue && set -e\npytest x; true", 3, None, False),
    ("set +e\nset -e | cat\npytest x; true", 3, None, False),
    ("trap 'exit 0' ERR\npytest x", 1, None, False),
    ("trap 'exit 0' EXIT\npytest x", 1, None, False),
    ("trap -- 'exit 0' EXIT\npytest x", 1, None, False),
    ("trap -p\npytest x", 1, None, True),
    ("trap - ERR EXIT\npytest x", 1, None, True),
    ("trap EXIT\npytest x", 1, None, True),
    ("shopt -uo errexit\npytest x; true", 1, None, False),
    ("set +e\nshopt -so errexit\npytest x; true", 2, None, True),
    ("shopt -s nullglob\npytest x; true", 1, None, True),
    ("eval 'set +e'\npytest x; true", 1, None, False),
    (". ./env.sh\npytest x; true", 1, None, False),
    ("source ./env.sh\npytest x; true", 1, None, False),
    ("builtin set +e\npytest x; true", 1, None, False),
    ("command set +e\npytest x; true", 1, None, False),
    ("command exec true\npytest x", 1, None, False),
    ("command -v exit\npytest x", 1, None, True),
    ("function f { exit 0; }\nf\npytest x", 3, None, False),
    ("f () { exit 0; }\nf\npytest x", 3, None, False),
    ("pytest() { true; }\npytest x", 2, None, False),
    ("function pytest { true; }\npytest x", 2, None, False),
    ("pytest () { true; }\npytest x", 2, None, False),
    ("python() { true; }\npython -m pytest x", 2, None, False),
    ("alias pytest=true\npytest x", 1, "sh", False),
    ("hash -p /bin/true pytest\npytest x", 1, None, False),
    ("hash pytest\npytest x", 1, None, True),
    ('note() { echo "$1"; }\npytest x', 2, None, True),
)

#: Masked rows the shell does not mask under this file's runs, ``(script, shell) -> why the
#: model reports them anyway``. Every other masked row must exit 0 in one of those runs.
CONSERVATIVE_ROWS: Final[dict[tuple[str, str | None], str]] = {
    ("cd d && pytest x\necho done", None): (
        "masks when `cd d` fails: the `&&` list skips pytest and the step goes on"
    ),
    ('case "$X" in\n  a) exit 0 ;;\nesac\npytest x', None): (
        "masks when `$X` is `a`; the model takes every branch of a compound"
    ),
    ("set +e\ntrue && set -e\npytest x; true", None): (
        "the `set -e` runs only when the left side succeeds, so it may only switch options off"
    ),
    ("pytest x", "pwsh"): "PowerShell is not modelled, so nothing is accepted under it",
}

#: What the stub ``pytest`` exits with: any status a real failure could have.
STUB_STATUS: Final[int] = 7
#: The directory and the sourced script the rows name (``cd d``, ``. ./env.sh``).
ROW_DIRECTORY: Final[str] = "d"
SOURCED_SCRIPT: Final[str] = "env.sh"
SOURCED_BODY: Final[str] = "set +e\n"
#: ``CI`` unset, then set, so both branches of a row's ``if [ -n "$CI" ]`` run.
CI_VALUES: Final[tuple[str | None, ...]] = (None, "1")
SHELL_TIMEOUT_S: Final[float] = 30.0


def real_exit_statuses(directory: Path, script: str, shell: str | None) -> list[int]:
    """``script``'s exit status under ``shell`` as GitHub runs it, once per :data:`CI_VALUES`.

    ``directory`` holds the stub ``pytest`` (exiting :data:`STUB_STATUS`), the files the rows
    name and the script itself; the environment is ``PATH`` with the stub first, plus ``CI``.
    """
    stubs = directory / "bin"
    stubs.mkdir(exist_ok=True)
    stub = stubs / "pytest"
    stub.write_text(f"#!/bin/sh\nexit {STUB_STATUS}\n", encoding="utf-8")
    stub.chmod(0o755)
    (directory / ROW_DIRECTORY).mkdir(exist_ok=True)
    (directory / SOURCED_SCRIPT).write_text(SOURCED_BODY, encoding="utf-8")
    step = directory / "step.sh"
    step.write_text(f"{script}\n", encoding="utf-8")
    statuses: list[int] = []
    for ci in CI_VALUES:
        environment = {"PATH": f"{stubs}{os.pathsep}{os.environ.get('PATH', '')}"}
        if ci is not None:
            environment["CI"] = ci
        completed = subprocess.run(
            [*RUNNER_ARGV[shell], str(step)],
            cwd=directory,
            env=environment,
            capture_output=True,
            timeout=SHELL_TIMEOUT_S,
            check=False,
        )
        statuses.append(completed.returncode)
    return statuses


def _row_id(row: tuple[str, int, str | None, bool]) -> str:
    script, _, shell, _ = row
    return f"{shell or 'default'}:{script}"


class TestTheModel:
    """Each row's verdict, as the model gives it."""

    @pytest.mark.parametrize(("script", "position", "shell", "loud"), STATUS_CASES)
    def test_a_failure_reaches_the_step_only_when_nothing_absorbs_it(
        self, script: str, position: int, shell: str | None, loud: bool
    ) -> None:
        commands = iter_shell_commands(script)
        assert "pytest x" in commands[position].text, "the row indexes the wrong command"
        obstacles = exit_status_obstacles(commands, position, shell)
        assert (not obstacles) == loud, obstacles

    def test_both_outcomes_are_exercised(self) -> None:
        """Vacuity: a model that rejects everything passes every masked row."""
        assert {loud for *_, loud in STATUS_CASES} == {True, False}


class TestAgainstTheRealShell:
    """The model's claims, checked by running the shell it describes."""

    def test_the_runner_argv_covers_every_modelled_shell(self) -> None:
        assert set(RUNNER_ARGV) == set(SHELL_PIPEFAIL)

    @pytest.mark.parametrize(
        "row", [row for row in STATUS_CASES if row[2] in RUNNER_ARGV], ids=_row_id
    )
    def test_the_model_never_calls_a_masked_script_loud(
        self, tmp_path: Path, row: tuple[str, int, str | None, bool]
    ) -> None:
        """Soundness: no obstacle found means the real shell fails with the failing pytest."""
        script, position, shell, _ = row
        if exit_status_obstacles(iter_shell_commands(script), position, shell):
            return
        statuses = real_exit_statuses(tmp_path, script, shell)
        assert all(statuses), f"the model calls it loud, but the shell exits {statuses}"

    @pytest.mark.parametrize("row", [row for row in STATUS_CASES if not row[3]], ids=_row_id)
    def test_a_masked_row_is_a_real_mask_or_says_why_not(
        self, tmp_path: Path, row: tuple[str, int, str | None, bool]
    ) -> None:
        script, _, shell, _ = row
        reason = CONSERVATIVE_ROWS.get((script, shell))
        if shell not in RUNNER_ARGV:
            assert reason, f"`{shell}` cannot be run here; say why the row is masked"
            return
        statuses = real_exit_statuses(tmp_path, script, shell)
        if reason is None:
            assert 0 in statuses, f"never masked by the shell ({statuses}): add a reason"
        else:
            assert 0 not in statuses, f"the shell masks it ({statuses}): drop the reason"

    def test_every_conservative_row_is_a_masked_row(self) -> None:
        """An entry for a row that is gone, or loud, explains nothing."""
        masked = {(script, shell) for script, _, shell, loud in STATUS_CASES if not loud}
        assert set(CONSERVATIVE_ROWS) <= masked, set(CONSERVATIVE_ROWS) - masked
        assert all(reason.strip() for reason in CONSERVATIVE_ROWS.values())
