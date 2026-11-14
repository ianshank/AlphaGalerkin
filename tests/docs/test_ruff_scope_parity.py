"""CI's ruff scope is exactly the complement of pre-commit's ruff ``exclude:``.

Defect class: a tracked ``*.py`` that the contributor's commit hook lints but CI
never does (or the reverse) -- code that is CI-green yet rewritten or rejected
by a hook, or hook-clean yet never checked by the gate. ``ci.yml``'s lint step
stated this invariant in a comment with a shell one-liner to run by hand, and
nothing ran it; a mutation runner first written under ``.claude/`` (PR #165)
would have broken it silently, since pre-commit lints ``.claude/`` and CI does
not. Hermetic: reads ``ci.yml``, ``.pre-commit-config.yaml``, the ``Makefile``
and ``git ls-files``; runs nothing else.

Planted defects (harden-a-guard), 5/5 killed, each by the named test. The fifth was
the final review of PR #165's: the first version checked files in one direction only.

1. ``scripts/`` dropped from CI's ``ruff check`` paths:
   ``test_every_file_the_hook_lints_is_linted_by_ci``.
2. ``ruff format``'s paths drifting from ``ruff check``'s:
   ``test_ruff_check_and_format_cover_the_same_paths``.
3. The Makefile ``lint`` target drifting from CI:
   ``test_the_makefile_lint_target_mirrors_ci``.
4. A CI path inside a tree pre-commit excludes (``hf_space/`` added to CI):
   ``test_no_ci_path_lies_in_an_excluded_tree``.
5. Both hooks excluding a subtree CI lints (``src/deployment``): only
   ``test_no_file_ci_lints_is_excluded_by_the_hook`` -- the entry-level check
   above cannot see it.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.support.workflows import (
    MAKEFILE,
    REPO_ROOT,
    iter_commands,
    iter_run_scripts,
    makefile_target_recipe,
)

PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
RUFF_HOOK_IDS = ("ruff", "ruff-format")
LINT_TARGET = "lint"
CI_WORKFLOW_NAME = "ci.yml"
# Vacuity floors: below these the scan is reading the wrong thing.
MIN_TRACKED_PYTHON_FILES = 500
MIN_CI_RUFF_PATHS = 5


def _ruff_paths(command: str, subcommand: str) -> tuple[str, ...] | None:
    """The positional path arguments of a ``ruff <subcommand>`` command, else None."""
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token.endswith("ruff") and index + 1 < len(tokens) and tokens[index + 1] == subcommand:
            rest = tokens[index + 2 :]
            return tuple(sorted(t for t in rest if not t.startswith("-")))
    return None


def _ci_ruff_paths(subcommand: str) -> tuple[str, ...]:
    found = {
        paths
        for run in iter_run_scripts()
        if run.workflow == CI_WORKFLOW_NAME
        for command in iter_commands(run.script)
        if (paths := _ruff_paths(command, subcommand)) is not None
    }
    assert len(found) == 1, f"expected one CI `ruff {subcommand}` path list, found {found}"
    return found.pop()


def _makefile_ruff_paths(subcommand: str) -> tuple[str, ...]:
    commands = [c.replace("$(RUFF)", "ruff") for c in makefile_target_recipe(LINT_TARGET)]
    found = [p for c in commands if (p := _ruff_paths(c, subcommand)) is not None]
    assert len(found) == 1, f"expected one `ruff {subcommand}` in `make {LINT_TARGET}`: {found}"
    return found[0]


def _hook_excludes() -> set[str]:
    document = yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    excludes = {
        hook.get("exclude", "")
        for repo in document.get("repos", [])
        for hook in repo.get("hooks", [])
        if hook.get("id") in RUFF_HOOK_IDS
    }
    assert excludes, "no ruff hooks found in .pre-commit-config.yaml"
    return excludes


def _in_scope(path: str, scope: tuple[str, ...]) -> bool:
    return any(path == entry or (entry.endswith("/") and path.startswith(entry)) for entry in scope)


@pytest.fixture(scope="module")
def tracked_python_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files = [line for line in result.stdout.splitlines() if line]
    assert len(files) >= MIN_TRACKED_PYTHON_FILES, f"only {len(files)} tracked *.py files"
    return files


def test_both_ruff_hooks_share_one_exclude() -> None:
    excludes = _hook_excludes()
    assert len(excludes) == 1, f"ruff and ruff-format exclude different trees: {excludes}"
    assert excludes.pop(), "the ruff hooks exclude nothing, so the scan below is trivial"


def test_ruff_check_and_format_cover_the_same_paths() -> None:
    check = _ci_ruff_paths("check")
    assert len(check) >= MIN_CI_RUFF_PATHS, f"CI's ruff scope is suspiciously small: {check}"
    assert _ci_ruff_paths("format") == check


def test_the_makefile_lint_target_mirrors_ci() -> None:
    assert MAKEFILE.exists()
    for subcommand in ("check", "format"):
        assert _makefile_ruff_paths(subcommand) == _ci_ruff_paths(subcommand), subcommand


def test_no_ci_path_lies_in_an_excluded_tree() -> None:
    (exclude,) = _hook_excludes()
    overlapping = [path for path in _ci_ruff_paths("check") if re.match(exclude, path)]
    assert not overlapping, f"CI lints trees the hook excludes: {overlapping}"


def test_every_file_the_hook_lints_is_linted_by_ci(
    tracked_python_files: list[str],
) -> None:
    (exclude,) = _hook_excludes()
    scope = _ci_ruff_paths("check")
    hook_only = sorted(
        path
        for path in tracked_python_files
        if not re.match(exclude, path) and not _in_scope(path, scope)
    )
    assert not hook_only, (
        "linted by the commit hook but invisible to CI's ruff steps -- widen the CI and "
        f"Makefile scope, or move the file: {hook_only}"
    )


def test_no_file_ci_lints_is_excluded_by_the_hook(tracked_python_files: list[str]) -> None:
    """The other direction, per file: an exclude that carves into CI's scope.

    ``test_no_ci_path_lies_in_an_excluded_tree`` reads CI's path *entries*, so an
    exclude naming a subtree of one (``src/deployment``) slipped past it; the final
    review of PR #165 planted exactly that.
    """
    (exclude,) = _hook_excludes()
    scope = _ci_ruff_paths("check")
    ci_only = sorted(
        path for path in tracked_python_files if _in_scope(path, scope) and re.match(exclude, path)
    )
    assert not ci_only, f"CI lints files the commit hook excludes: {ci_only[:10]}"


def test_the_scan_sees_files_on_both_sides(tracked_python_files: list[str]) -> None:
    """Vacuity: an exclude matching nothing, or a scope covering nothing, passes the scan."""
    (exclude,) = _hook_excludes()
    scope = _ci_ruff_paths("check")
    assert any(re.match(exclude, path) for path in tracked_python_files)
    assert any(_in_scope(path, scope) for path in tracked_python_files)
    assert Path(REPO_ROOT / ".github" / "workflows" / CI_WORKFLOW_NAME).exists()
