"""Hermetic, shared access to this repo's CI configuration as *data*.

Several guards under ``tests/docs/`` need the same three things: the body of
every ``run:`` script in every GitHub Actions workflow, the recipe behind a
``Makefile`` target, and a shell-aware way to chop either into individual
commands. Each of those is a parser, and two parsers that must agree are two
parsers that will eventually disagree -- the lesson
``tests/support/import_graph.py`` was extracted for -- so they live here once.

Nothing in this module executes anything: it reads YAML and text off disk.

The helpers are deliberately small and total, so the guards that use them can
unit-test each one on synthetic input. A guard whose parser silently matches
nothing passes every assertion it makes.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
import shlex
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from tests.support.marker_expr import (
    NODEID_SEPARATOR,
    PytestInvocation,
    expression_matches,
    is_python_program,
    split_environment,
)

#: Repository root, resolved from this file's location (``tests/support/``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Directory holding every GitHub Actions workflow.
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

#: The main workflow: the one that carries the blocking ``ci-success`` gate.
CI_WORKFLOW_FILENAME = "ci.yml"

#: Absolute path to the main workflow.
CI_WORKFLOW = WORKFLOW_DIR / CI_WORKFLOW_FILENAME

#: The aggregate job whose ``needs:`` list decides what can block a merge.
CI_SUCCESS_JOB = "ci-success"

#: The developer-facing entry point that mirrors CI.
MAKEFILE = REPO_ROOT / "Makefile"


@dataclass(frozen=True)
class RunScript:
    """One ``run:`` script body, tagged with where in the workflow it lives."""

    workflow: str
    job: str
    step: str
    script: str

    def __str__(self) -> str:  # pragma: no cover - failure-message sugar only
        return f"{self.workflow}::{self.job}::{self.step}"


def load_workflow(path: Path) -> dict[str, Any]:
    """Parse one workflow file into a plain dictionary.

    Args:
        path: Absolute path to a ``.yml`` workflow file.

    Returns:
        The parsed document, or an empty dict if the file parses to a non-mapping.

    """
    document: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document if isinstance(document, dict) else {}


def iter_run_scripts(workflow_dir: Path = WORKFLOW_DIR) -> list[RunScript]:
    """Every ``run:`` script in every workflow under ``workflow_dir``.

    Args:
        workflow_dir: Directory to scan for ``*.yml`` workflow files.

    Returns:
        One :class:`RunScript` per ``run:`` key, in filename then step order.

    """
    found: list[RunScript] = []
    for path in sorted(workflow_dir.glob("*.yml")):
        document = load_workflow(path)
        jobs = document.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_name, job in jobs.items():
            steps = job.get("steps") if isinstance(job, dict) else None
            if not isinstance(steps, list):
                continue
            for index, step in enumerate(steps):
                script = step.get("run") if isinstance(step, dict) else None
                if not isinstance(script, str):
                    continue
                label = step.get("name") or f"step[{index}]"
                found.append(
                    RunScript(
                        workflow=path.name,
                        job=str(job_name),
                        step=str(label),
                        script=script,
                    )
                )
    return found


def job_needs(document: dict[str, Any], job: str) -> list[str]:
    """The ``needs:`` list of one job, normalised to a list of job names.

    GitHub accepts either a scalar or a sequence; both are returned as a list.

    Args:
        document: A parsed workflow document.
        job: The job key to read.

    Returns:
        The job names ``job`` depends on. Empty if the job or key is absent.

    """
    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        return []
    entry = jobs.get(job)
    if not isinstance(entry, dict):
        return []
    needs = entry.get("needs")
    if isinstance(needs, str):
        return [needs]
    if isinstance(needs, list):
        return [str(item) for item in needs]
    return []


#: Characters after which an unquoted ``#`` begins a comment, in addition to
#: the start of the script. Whitespace is the common case; the shell
#: metacharacters matter because ``cmd;#c`` and ``cmd &&#c`` are comments too,
#: and treating them as command text is not merely untidy -- ``iter_commands``
#: splits on those same separators, so the comment would come back as its own
#: "command". A commented-out ``pytest tests/e2e/ -m "..."`` would then be read
#: by the guards as a live step: a deleted step could look present, or a third
#: ``-k`` could appear from prose. (Copilot review, PR #144.)
COMMENT_START_PREDECESSORS: Final[frozenset[str]] = frozenset(" \t\n;&|(")


class UnbalancedQuoteError(ValueError):
    """A script ended inside an unterminated quote.

    Raised rather than returning a best-effort parse: an unbalanced quote means
    the scan's idea of "quoted" has desynchronised from the source, so every
    ``#`` after that point is classified by a state that is already wrong. A
    guard built on that would under- or over-cover silently, which is the exact
    failure this module exists to make impossible.
    """


def strip_shell_comments(script: str) -> str:
    r"""Remove ``#`` comments from a shell script without touching quoted text.

    Naive ``line.split("#")`` would truncate ``echo "a#b"`` and, worse, would
    *keep* a ``-m`` expression that only appears inside prose. Quote state is
    tracked so an apostrophe in a comment cannot desynchronise the scan --
    the comment is cut before its contents are ever examined.

    Quote state spans newlines, as it does in the shell. An earlier version
    reset it at every ``\\n``, which silently **truncated** any command carrying
    a quoted string across lines: the reopened-quote state was lost, so the next
    ``#`` inside that string was cut as a comment along with everything after
    it. Tracking across lines is what the shell actually does; the cost is that
    a genuinely unbalanced quote now desynchronises the rest of the scan, which
    is why that case raises instead of being returned quietly.

    Args:
        script: Raw shell source.

    Returns:
        ``script`` with comment text replaced by nothing, newlines preserved.

    Raises:
        UnbalancedQuoteError: The script ends inside a quote.

    """
    out: list[str] = []
    in_single = False
    in_double = False
    index = 0
    previous = "\n"  # start-of-script behaves like a fresh line
    while index < len(script):
        char = script[index]
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif (
            char == "#"
            and not in_single
            and not in_double
            and previous in COMMENT_START_PREDECESSORS
        ):
            while index < len(script) and script[index] != "\n":
                index += 1
            previous = "\n"
            continue
        previous = char
        out.append(char)
        index += 1
    if in_single or in_double:
        quote = "'" if in_single else '"'
        raise UnbalancedQuoteError(
            f"script ends inside an unterminated {quote} quote; comment stripping "
            "cannot be trusted past that point"
        )
    return "".join(out)


_CONTINUATION = re.compile(r"\\\n\s*")


def join_line_continuations(script: str) -> str:
    r"""Fold ``\``-continued shell lines into single logical lines.

    Both workflow ``run: |`` blocks and Makefile recipes wrap long pytest
    invocations across many lines, so the flags belonging to one command are
    only adjacent after this.

    Args:
        script: Shell source, possibly containing backslash continuations.

    Returns:
        The same source with each continuation collapsed to one space.

    """
    return _CONTINUATION.sub(" ", script)


_COMMAND_SEPARATOR = re.compile(r"\n|;|&&|\|\||\|")

#: Two-character separators checked before the single-character ones.
_DOUBLE_SEPARATORS: Final[tuple[str, ...]] = ("&&", "||")
_SINGLE_SEPARATORS: Final[frozenset[str]] = frozenset("\n;|")


def split_shell_commands(text: str) -> list[str]:
    r"""Split shell text into commands at separators that are *outside quotes*.

    A regex split on ``;``/``&&``/``||``/``|``/newline also fires inside a
    quoted string, so ``echo "report-only; exit 1"`` came back as two
    commands and the second read as a standalone ``exit 1`` -- a report-only
    block classified as a hard gate (Copilot review, PR #151). Quote state is
    tracked the same way :func:`strip_shell_comments` tracks it; callers pass
    text that function has already validated, so quotes are balanced here.

    Args:
        text: Shell source with comments stripped and continuations joined.

    Returns:
        Whitespace-stripped, non-empty commands in source order.

    """
    parts: list[str] = []
    buffer: list[str] = []
    in_single = False
    in_double = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if text.startswith(_DOUBLE_SEPARATORS, index):
                parts.append("".join(buffer))
                buffer = []
                index += 2
                continue
            if char in _SINGLE_SEPARATORS:
                parts.append("".join(buffer))
                buffer = []
                index += 1
                continue
        buffer.append(char)
        index += 1
    parts.append("".join(buffer))
    return [part.strip() for part in parts if part.strip()]


def iter_commands(script: str) -> list[str]:
    """Split a shell script into individual, non-empty commands.

    Comments are stripped and continuations joined first, so each returned
    string is one logical command whose first token is its program name.

    Args:
        script: Raw shell source.

    Returns:
        Whitespace-stripped commands, in source order, with blanks dropped.

    """
    normalised = join_line_continuations(strip_shell_comments(script))
    return split_shell_commands(normalised)


_MAKE_TARGET = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)\s*:(?!=)")

#: Make's recipe-line prefixes, stripped before the line is treated as a shell
#: command. ``@`` silences the echo, ``-`` ignores a non-zero exit, ``+`` forces
#: execution under ``-n``; none is part of the command Make actually runs.
#:
#: Leaving them in place is not cosmetic. ``tests/docs/test_marker_vocabulary.py``
#: decides whether an unquoted ``-m`` is a marker expression by looking at the
#: *program token*, so a recipe written ``@$(PYTEST) ... -m "..."`` would present
#: as the program ``@$(PYTEST)``, fail the pytest check, and be skipped -- a
#: guard that silently covers less than it claims, which is the exact defect
#: class this whole change exists to prevent. Two such lines are live in the
#: Makefile today (the `test-substrate` printf and the `gitleaks` if-block).
MAKE_RECIPE_PREFIXES: Final[str] = "@-+"


def strip_recipe_prefixes(line: str) -> str:
    """Remove Make's leading recipe prefixes from a recipe line.

    Make allows any combination of ``@``, ``-`` and ``+`` in any order at the
    start of a recipe line, so this strips repeatedly rather than once.

    Args:
        line: A recipe line with its leading tab already removed.

    Returns:
        The shell command Make would run.

    """
    return line.lstrip(MAKE_RECIPE_PREFIXES)


def makefile_target_recipe(target: str, makefile: Path = MAKEFILE) -> list[str]:
    """The recipe lines of one Makefile target.

    Args:
        target: The target name, e.g. ``"test-e2e"``.
        makefile: Path to the Makefile to read.

    Returns:
        One entry per logical recipe command (continuations already joined).
        Empty if the target is not defined.

    """
    lines = makefile.read_text(encoding="utf-8").splitlines()
    recipe: list[str] = []
    collecting = False
    for line in lines:
        match = _MAKE_TARGET.match(line)
        if match:
            collecting = match.group("name") == target
            continue
        if not collecting:
            continue
        if line.startswith("\t"):
            recipe.append(strip_recipe_prefixes(line[1:]))
        elif line.strip() == "" or line.lstrip().startswith("#"):
            continue
        else:
            collecting = False
    return iter_commands("\n".join(recipe))


def makefile_commands(makefile: Path = MAKEFILE) -> list[str]:
    """Every command in the Makefile, comments stripped and continuations joined.

    Args:
        makefile: Path to the Makefile to read.

    Returns:
        Whitespace-stripped commands in source order.

    """
    text = makefile.read_text(encoding="utf-8")
    dedented = "\n".join(
        strip_recipe_prefixes(line[1:]) if line.startswith("\t") else line
        for line in text.splitlines()
    )
    return iter_commands(dedented)


_IF_BLOCK = re.compile(r"if\s*\[\[(?P<cond>.*?)\]\]\s*;\s*then(?P<body>.*?)\bfi\b", re.S)
_NEEDS_RESULT = re.compile(r"needs\.(?P<job>[A-Za-z0-9_.-]+)\.result")


def hard_gate_conditions(script: str) -> list[str]:
    """The ``[[ ... ]]`` condition of every ``if`` block whose body reaches ``exit 1``.

    The raw condition text is what a guard needs when the *shape* of a gate
    matters and not only which job it names -- ``ci-success``'s ``focus`` gate,
    for example, must accept ``skipped`` on a push and reject it on an
    unlabelled pull request, and only the condition can show that.

    Args:
        script: The body of a ``run:`` step (typically ``ci-success``'s).

    Returns:
        Condition strings in source order, one per ``exit 1`` block. Blocks
        that only ``echo`` are not included.

    """
    return [
        block.group("cond")
        for block in _IF_BLOCK.finditer(script)
        if body_exits_nonzero(block.group("body"))
    ]


#: A shell ``exit`` with a non-zero literal status, as a whole command.
_EXIT_NONZERO = re.compile(r"^exit\s+(?!0+\b)\d+\b")


def body_exits_nonzero(body: str) -> bool:
    """True iff ``body`` contains a standalone ``exit <non-zero>`` *command*.

    A substring test for ``exit 1`` also matches a comment, a variable's
    contents, or ``echo "would exit 1"`` -- report-only blocks that would then
    be classified as hard gates, letting the merge-gate guard pass while the
    job cannot fail the build. Splitting into commands first (comments
    stripped, and separators inside quotes ignored by
    :func:`split_shell_commands`) means only a real ``exit`` command counts.

    Args:
        body: The text between ``then`` and ``fi``.

    Returns:
        Whether some command in ``body`` is ``exit N`` with ``N != 0``.

    """
    return any(_EXIT_NONZERO.match(command) for command in iter_commands(body))


def hard_gate_jobs(script: str) -> set[str]:
    """Job names this script *fails the build* on, as opposed to merely reporting.

    An ``echo`` of ``needs.<job>.result`` looks identical to a gate in a diff and
    blocks nothing; only an ``if`` whose body reaches ``exit 1`` does. This
    returns exactly the jobs named in such a condition.

    Args:
        script: The body of a ``run:`` step (typically ``ci-success``'s).

    Returns:
        Every job name appearing in the condition of an ``if`` block whose body
        contains ``exit 1``.

    """
    gated: set[str] = set()
    for condition in hard_gate_conditions(script):
        gated.update(_NEEDS_RESULT.findall(condition))
    return gated


# --------------------------------------------------------------------------- #
# Added for tests/docs/test_fem_required_visibility.py. Additive: nothing above #
# this banner changed behaviour.                                              #
# --------------------------------------------------------------------------- #


def blocking_jobs(document: dict[str, Any], gate_job: str = CI_SUCCESS_JOB) -> set[str]:
    """Jobs the aggregate gate both ``needs`` and hard-fails on.

    Either half alone blocks nothing: a job in ``needs`` with no ``exit 1``
    block is reported and ignored, and an ``exit 1`` block naming a job outside
    ``needs`` reads an empty result.

    Args:
        document: A parsed workflow document.
        gate_job: The aggregate job, ``ci-success`` by default.

    Returns:
        The intersection of ``gate_job``'s ``needs`` and its hard-gated jobs.

    """
    jobs = document.get("jobs")
    entry = jobs.get(gate_job) if isinstance(jobs, dict) else None
    steps = entry.get("steps") if isinstance(entry, dict) else None
    gated: set[str] = set()
    for step in steps if isinstance(steps, list) else []:
        script = step.get("run") if isinstance(step, dict) else None
        if isinstance(script, str):
            gated |= hard_gate_jobs(script)
    return set(job_needs(document, gate_job)) & gated


#: GitHub renders YAML booleans in ``env:`` as these strings.
_YAML_BOOLEAN_TEXT: Final[dict[bool, str]] = {True: "true", False: "false"}


def _env_mapping(value: object) -> dict[str, str]:
    """An ``env:`` block as the strings a process actually receives."""
    if not isinstance(value, dict):
        return {}
    return {
        str(name): _YAML_BOOLEAN_TEXT[item] if isinstance(item, bool) else str(item)
        for name, item in value.items()
    }


def _may_continue_on_error(value: object) -> bool:
    """True unless ``continue-on-error`` is absent or literally false.

    An expression (``${{ matrix.experimental }}``) is not proof that it is off.
    """
    if value is None or value is False:
        return False
    return not (isinstance(value, str) and value.strip().lower() == "false")


def _default_working_directory(mapping: object) -> str | None:
    defaults = mapping.get("defaults") if isinstance(mapping, dict) else None
    run = defaults.get("run") if isinstance(defaults, dict) else None
    directory = run.get("working-directory") if isinstance(run, dict) else None
    return str(directory) if directory is not None else None


@dataclass(frozen=True)
class WorkflowStep:
    """One step of one job, with the context that decides whether it can fail the build.

    ``env`` merges workflow, job and step ``env:`` with the later level winning,
    as GitHub does. ``continue_on_error`` is set at either the step or the job.
    ``working_directory`` falls back to the job's and then the workflow's
    ``defaults.run``.
    """

    workflow: str
    job: str
    index: int
    name: str
    script: str | None
    env: dict[str, str]
    condition: str | None
    continue_on_error: bool
    working_directory: str | None

    def __str__(self) -> str:  # pragma: no cover - failure-message sugar only
        return f"{self.workflow}::{self.job}::{self.name}"


def workflow_steps(path: Path) -> list[WorkflowStep]:
    """Every step of every job in one workflow file, with its effective context.

    Args:
        path: A workflow ``.yml`` file.

    Returns:
        Steps in job then step order. Malformed jobs and steps are skipped.

    """
    document = load_workflow(path)
    workflow_env = _env_mapping(document.get("env"))
    workflow_directory = _default_working_directory(document)
    jobs = document.get("jobs")
    found: list[WorkflowStep] = []
    for job_name, job in (jobs if isinstance(jobs, dict) else {}).items():
        if not isinstance(job, dict) or not isinstance(job.get("steps"), list):
            continue
        job_env = {**workflow_env, **_env_mapping(job.get("env"))}
        job_continues = _may_continue_on_error(job.get("continue-on-error"))
        job_directory = _default_working_directory(job) or workflow_directory
        for index, step in enumerate(job["steps"]):
            if not isinstance(step, dict):
                continue
            script = step.get("run")
            condition = step.get("if")
            directory = step.get("working-directory")
            found.append(
                WorkflowStep(
                    workflow=path.name,
                    job=str(job_name),
                    index=index,
                    name=str(step.get("name") or f"step[{index}]"),
                    script=script if isinstance(script, str) else None,
                    env={**job_env, **_env_mapping(step.get("env"))},
                    condition=None if condition is None else str(condition),
                    continue_on_error=job_continues
                    or _may_continue_on_error(step.get("continue-on-error")),
                    working_directory=str(directory) if directory is not None else job_directory,
                )
            )
    return found


#: ``pip`` as a program: ``pip``, ``pip3``, ``/usr/bin/pip3.11``.
_PIP_PROGRAM: Final[re.Pattern[str]] = re.compile(r"^(.*/)?pip[0-9.]*$")

#: The project itself, optionally with extras: ``.``, ``.[dev,fem]``, ``./[fem]``.
_PROJECT_REQUIREMENT: Final[re.Pattern[str]] = re.compile(r"^\./?(?:\[(?P<extras>[^\]]*)\])?$")

#: ``pip install`` options whose value is the next token.
_PIP_VALUE_OPTIONS: Final[frozenset[str]] = frozenset(
    {
        "-c",
        "--constraint",
        "-e",
        "--editable",
        "--extra-index-url",
        "-f",
        "--find-links",
        "-i",
        "--index-url",
        "--prefix",
        "-r",
        "--requirement",
        "--root",
        "-t",
        "--target",
        "--trusted-host",
    }
)

#: The two options above whose value is itself something installed.
_PIP_EDITABLE: Final[frozenset[str]] = frozenset({"-e", "--editable"})


@dataclass(frozen=True)
class PipInstall:
    """What one ``pip install`` command asks for."""

    project_extras: frozenset[str]
    requirements: tuple[str, ...]


def pip_install(command: str) -> PipInstall | None:
    """Read one shell command as a ``pip install``.

    Args:
        command: One logical shell command (see :func:`iter_commands`).

    Returns:
        The extras requested on the project itself (``-e ".[dev,fem]"``) and the
        other requirement tokens, or ``None`` when the command is not a
        ``pip install`` or cannot be shell-split.

    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    _, program = split_environment(tokens)
    if program[:1] and is_python_program(program[0]) and program[1:3] == ["-m", "pip"]:
        program = program[2:]
    if not program or not _PIP_PROGRAM.match(program[0]) or "install" not in program:
        return None
    arguments = program[program.index("install") + 1 :]
    extras: set[str] = set()
    requirements: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        index += 1
        if token.startswith("-"):
            name, attached, value = token.partition("=")
            if name not in _PIP_VALUE_OPTIONS or (not attached and index >= len(arguments)):
                continue
            if not attached:
                value = arguments[index]
                index += 1
            if name not in _PIP_EDITABLE:
                continue
            token = value
        project = _PROJECT_REQUIREMENT.match(token)
        if project is None:
            requirements.append(token)
            continue
        extras.update(e.strip() for e in (project.group("extras") or "").split(",") if e.strip())
    return PipInstall(frozenset(extras), tuple(requirements))


#: Prefix of the obstacle reported when no path argument reaches the target.
NOT_SELECTED_BY_PATH: Final[str] = "no path argument selects it"


def _resolve(base: str, argument: str) -> str:
    """``argument`` relative to ``base``, normalised; a ``::node`` suffix is kept."""
    path, separator, node = argument.partition(NODEID_SEPARATOR)
    return posixpath.normpath(posixpath.join(base, path)) + separator + node


def _within(path: str, directory: str) -> bool:
    return directory == "." or path == directory or path.startswith(directory + "/")


def _path_selects(selection: str, path: str, nodeid: str) -> bool:
    """Whether one resolved path argument selects all of ``path::nodeid``."""
    file, separator, node = selection.partition(NODEID_SEPARATOR)
    if not separator:
        return _within(path, file)
    if not nodeid or file != path or "[" in node:
        return False  # a node id selects part of a module, and [..] one instance
    return nodeid == node or nodeid.startswith(node + NODEID_SEPARATOR)


def _ancestors(path: str) -> list[str]:
    parts = path.split("/")
    return ["/".join(parts[:end]) for end in range(len(parts), 0, -1)]


def selection_obstacles(
    invocation: PytestInvocation,
    *,
    path: str,
    nodeid: str,
    markers: Collection[str],
    working_directory: str | None = None,
    testpaths: Sequence[str] = (),
) -> list[str]:
    """Why ``invocation`` would not run every test at ``path::nodeid`` -- empty if it would.

    ``nodeid`` empty means the whole of ``path``, which may be a directory. The
    rules are pytest's: positional paths resolve against the working directory
    (``testpaths`` apply only from the rootdir, with no paths given);
    ``--ignore`` removes a path and everything under it; ``--ignore-glob``
    matches a path or any directory above it; ``--deselect`` is a *raw string
    prefix* of the node id. Anything this cannot read with confidence -- a
    ``-k`` filter, an unknown option, a non-executing or config-swapping one --
    is an obstacle, not a pass.

    Args:
        invocation: The parsed command (``marker_expr.pytest_invocation``).
        path: Repo-relative module (or directory) path.
        nodeid: Node id inside ``path``, ``::``-joined; empty for all of it.
        markers: Every marker the test carries.
        working_directory: The step's ``working-directory``, if any.
        testpaths: ``[tool.pytest.ini_options] testpaths``.

    Returns:
        Human-readable obstacles; :data:`NOT_SELECTED_BY_PATH` first if no
        path argument reaches the target at all.

    """
    base = working_directory or "."
    if invocation.paths:
        selections = [_resolve(base, argument) for argument in invocation.paths]
    elif base == "." and testpaths:
        selections = [_resolve(base, argument) for argument in testpaths]
    else:
        selections = [_resolve(base, ".")]
    obstacles: list[str] = []
    if not any(_path_selects(selection, path, nodeid) for selection in selections):
        obstacles.append(NOT_SELECTED_BY_PATH)
    is_directory = not path.endswith(".py")
    for ignored in (_resolve(base, argument) for argument in invocation.ignores):
        if _within(path, ignored) or _within(ignored, path):
            obstacles.append(f"--ignore={ignored} removes it")
    for pattern in (_resolve(base, argument) for argument in invocation.ignore_globs):
        if is_directory or any(fnmatch.fnmatch(part, pattern) for part in _ancestors(path)):
            obstacles.append(f"--ignore-glob={pattern} may remove it")
    full = f"{path}{NODEID_SEPARATOR}{nodeid}" if nodeid else path
    for prefix in invocation.deselects:
        if full.startswith(prefix) or (
            prefix.startswith(full) and prefix[len(full) : len(full) + 1] in ("", ":", "[", "/")
        ):
            obstacles.append(f"--deselect={prefix} removes it")
    if invocation.keyword_expression:
        obstacles.append(
            f"-k {invocation.keyword_expression!r} may deselect it; select by path and -m instead"
        )
    if not expression_matches(invocation.marker_expression, markers):
        obstacles.append(
            f"-m {invocation.marker_expression!r} deselects a test carrying {sorted(markers)}"
        )
    if invocation.unknown_options:
        obstacles.append(
            f"unrecognised option(s) {list(invocation.unknown_options)}: the argv cannot be "
            "read with confidence"
        )
    if invocation.inert_options:
        obstacles.append(
            f"{list(invocation.inert_options)} collects without running, or swaps the "
            "configuration this check reads"
        )
    return obstacles
