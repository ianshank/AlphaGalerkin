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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

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


def strip_shell_comments(script: str) -> str:
    """Remove ``#`` comments from a shell script without touching quoted text.

    Naive ``line.split("#")`` would truncate ``echo "a#b"`` and, worse, would
    *keep* a ``-m`` expression that only appears inside prose. Quote state is
    tracked so an apostrophe in a comment cannot desynchronise the scan --
    the comment is cut before its contents are ever examined.

    Args:
        script: Raw shell source.

    Returns:
        ``script`` with comment text replaced by nothing, newlines preserved.

    """
    out: list[str] = []
    in_single = False
    in_double = False
    index = 0
    at_token_start = True
    while index < len(script):
        char = script[index]
        if char == "\n":
            in_single = in_double = False
            at_token_start = True
            out.append(char)
            index += 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double and at_token_start:
            while index < len(script) and script[index] != "\n":
                index += 1
            continue
        at_token_start = char.isspace()
        out.append(char)
        index += 1
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
    return [part.strip() for part in _COMMAND_SEPARATOR.split(normalised) if part.strip()]


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
    for block in _IF_BLOCK.finditer(script):
        if "exit 1" not in block.group("body"):
            continue
        gated.update(_NEEDS_RESULT.findall(block.group("cond")))
    return gated
