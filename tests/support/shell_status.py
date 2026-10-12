"""Whether one command's failure fails its GitHub Actions step.

``tests/support/workflows.py`` reads workflow files and splits a ``run:`` script into
commands (:func:`~tests.support.workflows.iter_shell_commands`); this module reads what the
runner's shell does with one of those commands' exit status. It was split out of
``workflows.py`` on 2026-10-12, when modelling ``exec`` (Copilot review, PR #160) would have
taken that module past the 1000-line budget ``tests/docs/test_module_size_budget.py`` sets.

A conservative model of the runner's shell, not an interpreter: what it does not model is
reported rather than passed.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from typing import Final

from tests.support.workflows import AND_LIST, OR_LIST, PIPE, ShellCommand

#: The runner shells whose exit semantics are modelled, mapped to whether they set
#: ``pipefail``; all three set ``errexit``. ``None`` is a step with no ``shell:``
#: (GitHub runs ``bash -e {0}``), ``bash`` runs ``bash --noprofile --norc -eo
#: pipefail {0}``, ``sh`` runs ``sh -e {0}``.
SHELL_PIPEFAIL: Final[dict[str | None, bool]] = {None: False, "bash": True, "sh": False}

#: First words that open / close a compound command or a function body.
_COMPOUND_OPENERS: Final[frozenset[str]] = frozenset(
    {"if", "while", "until", "for", "case", "select", "{", "(", "function"}
)
_COMPOUND_CLOSERS: Final[frozenset[str]] = frozenset({"fi", "done", "esac", "}", ")"})
_FUNCTION_DEFINITION_SUFFIX: Final[str] = "()"
#: What a compound puts before the command it wraps in one list element: reserved words
#: (``then exit 0``, ``{ exec true``), a ``case WORD in`` head and a case pattern (``a)``).
_LEADING_RESERVED: Final[frozenset[str]] = frozenset(
    {"if", "elif", "then", "else", "while", "until", "do", "!", "{", "("}
)
_CASE: Final[str] = "case"
_CASE_IN: Final[str] = "in"
_CASE_PATTERN_END: Final[str] = ")"
_EXIT: Final[str] = "exit"
#: Leaves a function or a sourced script, so only at the top level can it end the step.
_RETURN: Final[str] = "return"
#: Replaces the shell when it names a command (``exec true``): the script ends there, with
#: that command's status. With only redirections (``exec 2>&1``) the shell goes on.
_EXEC: Final[str] = "exec"
#: ``exec``'s one option that takes a value: ``exec -a NAME command``.
_EXEC_VALUE_OPTION: Final[str] = "-a"
#: A redirection word: an optional descriptor or ``&``, the operator, its target if attached.
_REDIRECTION: Final[re.Pattern[str]] = re.compile(
    r"^(?:\d+|&)?(?:<>|>&|<&|>\||>>?|<<?<?)(?P<target>.*)$"
)
#: A redirection target that is a process substitution runs to the word that closes it.
_PROCESS_SUBSTITUTIONS: Final[tuple[str, ...]] = ("<(", ">(")
_SUBSTITUTION_END: Final[str] = ")"
_BACKGROUND: Final[str] = "&"
_SET_BUILTIN: Final[str] = "set"
_ERREXIT_LETTER: Final[str] = "e"
_NAMED_OPTION_LETTER: Final[str] = "o"
_ERREXIT: Final[str] = "errexit"
_PIPEFAIL: Final[str] = "pipefail"
_SET_PREFIXES: Final[str] = "-+"
_END_OF_OPTIONS: Final[str] = "--"


def _words(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _command_words(words: list[str]) -> list[str]:
    """The words of the command a compound wraps: ``then exit 0`` -> ``exit 0``."""
    index = 0
    while index < len(words):
        word = words[index]
        if word == _CASE and _CASE_IN in words[index:]:
            index = words.index(_CASE_IN, index) + 1
        elif word in _LEADING_RESERVED or word.endswith(_CASE_PATTERN_END):
            index += 1
        else:
            break
    return words[index:]


def _apply_set(arguments: list[str], errexit: bool, pipefail: bool) -> tuple[bool, bool]:
    """``set``'s effect on ``errexit`` and ``pipefail`` (``+e``, ``-o pipefail``, ``-euo ...``)."""
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        index += 1
        if len(argument) < 2 or argument[0] not in _SET_PREFIXES or argument == _END_OF_OPTIONS:
            break  # positional parameters from here on
        enable = argument[0] == "-"
        letters = argument[1:]
        if _ERREXIT_LETTER in letters:
            errexit = enable
        if _NAMED_OPTION_LETTER in letters and index < len(arguments):
            name = arguments[index]
            index += 1
            errexit = enable if name == _ERREXIT else errexit
            pipefail = enable if name == _PIPEFAIL else pipefail
    return errexit, pipefail


def _may_leave_cleanly(words: list[str]) -> bool:
    """An ``exit``/``return`` that can end the script with status 0."""
    code = words[1] if len(words) > 1 else ""
    return not (code.isdigit() and int(code) != 0)


def _replaces_shell(words: list[str]) -> bool:
    """Whether an ``exec`` names a command, rather than only redirecting the shell's own streams."""
    rest = words[1:]
    while rest and rest[0].startswith("-") and rest[0] != _END_OF_OPTIONS:
        rest = rest[2:] if rest[0] == _EXEC_VALUE_OPTION else rest[1:]
    rest = rest[1:] if rest[:1] == [_END_OF_OPTIONS] else rest
    while rest:
        match = _REDIRECTION.match(rest[0])
        if match is None:
            return True
        taken = 1 if match.group("target") else 2
        target = match.group("target") or "".join(rest[1:2])
        if target.startswith(_PROCESS_SUBSTITUTIONS):
            while taken < len(rest) and not rest[taken - 1].endswith(_SUBSTITUTION_END):
                taken += 1
        rest = rest[taken:]
    return False


def _ends_script(words: list[str], nested: bool) -> bool:
    """Whether a command can end the script green before a later one runs.

    ``exit``/``return`` with no non-zero literal status, or an ``exec`` that names a command.
    Inside a compound the model does not follow which branch runs, so they count as if they
    ran -- except ``return``, which there leaves only its function.
    """
    first = words[0] if words else ""
    if first == _EXEC:
        return _replaces_shell(words)
    if first == _EXIT or (first == _RETURN and not nested):
        return _may_leave_cleanly(words)
    return False


def _list_reaches_end(commands: Sequence[ShellCommand], index: int) -> bool:
    """Whether the ``&&``/``|`` chain from ``commands[index]`` is the script's last list."""
    while commands[index].after in (AND_LIST, PIPE):
        index += 1
    return commands[index].after == ""


def exit_status_obstacles(
    commands: Sequence[ShellCommand], position: int, shell: str | None
) -> list[str]:
    """Why the step could stay green when ``commands[position]`` fails, or never run it.

    A conservative model of the runner's shell, not an interpreter: what it does not
    model (an unknown ``shell:``, a compound command around it) is reported rather than
    passed. Under errexit, ``pytest ...; true`` *does* fail the step -- the shell exits at
    ``pytest`` -- so a plain sequence is accepted; ``|| true``, a pipe without
    ``pipefail``, ``set +e``, ``&`` and an enclosing ``if`` are not (Copilot review, PR
    #160: the guard read only the step's metadata). Nor is a command before it that can
    end the script green -- ``exit 0``, or an ``exec`` that names a command (``exec
    true``; ``exec 2>&1`` only redirects) -- whether at the top level or inside a compound,
    where every branch counts as taken; there a ``set`` can only switch an option off.

    Args:
        commands: The step's commands (:func:`~tests.support.workflows.iter_shell_commands`).
        position: Index of the command whose failure must fail the step.
        shell: The step's effective ``shell:`` (:attr:`WorkflowStep.shell`).

    Returns:
        One reason per way its status can be lost; empty if it reaches the step.

    """
    if shell not in SHELL_PIPEFAIL:
        return [f"shell {shell!r}: only GitHub's default, `bash` and `sh` are modelled"]
    errexit, pipefail = True, SHELL_PIPEFAIL[shell]
    depth = 0
    obstacles: list[str] = []
    for command in commands[:position]:
        words = _words(command.text) or [""]
        first, body = words[0], _command_words(words)
        opens = first in _COMPOUND_OPENERS or first.endswith(_FUNCTION_DEFINITION_SUFFIX)
        nested = depth > 0 or opens
        if opens:
            depth += 1
        elif first in _COMPOUND_CLOSERS:
            depth = max(0, depth - 1)
        if body[:1] == [_SET_BUILTIN]:
            after = _apply_set(body[1:], errexit, pipefail)
            errexit, pipefail = (errexit and after[0], pipefail and after[1]) if nested else after
        elif _ends_script(body, nested):
            where = " inside a compound command" if nested else ""
            obstacles.append(f"`{command.text}`{where} before it can end the step green first")
    if depth:
        obstacles.append("inside a compound command (if/while/for/case/{ }/( )/function)")
    if _BACKGROUND in _words(commands[position].text):
        obstacles.append("backgrounded with `&`: the step does not wait for its status")
    start = end = position
    while commands[start].before == PIPE:
        start -= 1
    while commands[end].after == PIPE:
        end += 1
    if end > position and not pipefail:
        obstacles.append("piped on without pipefail: the pipeline's status is its last command's")
    before, after = commands[start].before, commands[end].after
    if before == OR_LIST:
        obstacles.append("after `||`: it runs only if the command before it fails")
    elif before == AND_LIST and not _list_reaches_end(commands, end):
        obstacles.append(
            "after `&&` mid-script: if that command fails, the step goes on without it"
        )
    if after == OR_LIST:
        obstacles.append("followed by `||`: the right-hand side absorbs its failure")
    elif after == AND_LIST and not _list_reaches_end(commands, end):
        obstacles.append("followed by `&&` mid-script: errexit does not fire for it")
    elif not errexit and not _list_reaches_end(commands, end):
        obstacles.append("errexit is off (`set +e`): a later command decides the step's status")
    return obstacles
