"""Whether one command's failure fails its GitHub Actions step.

``tests/support/workflows.py`` reads workflow files and splits a ``run:`` script into
commands (:func:`~tests.support.workflows.iter_shell_commands`); this module reads what the
runner's shell does with one of those commands' exit status. It was split out of
``workflows.py`` on 2026-10-12, when modelling ``exec`` (Copilot review, PR #160) would have
taken that module past the 1000-line budget ``tests/docs/test_module_size_budget.py`` sets.

A conservative model of the runner's shell, not an interpreter: what it does not model is
reported rather than passed. :func:`step_failure_obstacles` adds the step's own metadata, so
every guard that needs "this command's failure fails its step" asks one function: the
``fem_required`` guard and the E2E visibility guard's clause (j) both do.

Copilot review 3 (PR #160) found two more ways a failure went green while the model passed
it, both reproduced with a stub ``pytest`` exiting 7: a ``set -e`` that may not run (``false
&& set -e``) counted as re-enabling errexit, and a ``trap 'exit 0' ERR`` or ``EXIT`` replaced
the status. The neighbours of each were reproduced the same way and are handled with them:
``shopt -uo errexit``, ``eval``, ``source`` / ``.``, ``builtin`` / ``command set +e``, a
function definition's head (``function f`` and ``f ()``), and a function or alias named like
the command itself. ``tests/docs/test_shell_status.py`` runs every row of its table under the
real shell and fails if the model calls loud a script that shell masks.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Sequence
from typing import Final

from tests.support.marker_expr import split_environment
from tests.support.workflows import AND_LIST, OR_LIST, PIPE, ShellCommand, WorkflowStep

#: Step ``if:`` conditions that run whenever the job runs.
ALWAYS_RUN_CONDITIONS: Final[frozenset[str]] = frozenset(
    {"always()", "${{ always() }}", "success()", "${{ success() }}"}
)

#: The runner shells whose exit semantics are modelled, mapped to whether they set
#: ``pipefail``; all three set ``errexit``. ``None`` is a step with no ``shell:``
#: (GitHub runs ``bash -e {0}``), ``bash`` runs ``bash --noprofile --norc -eo
#: pipefail {0}``, ``sh`` runs ``sh -e {0}``.
SHELL_PIPEFAIL: Final[dict[str | None, bool]] = {None: False, "bash": True, "sh": False}

#: The same shells as GitHub invokes them, the script's path appended: what the model claims
#: to describe, so ``tests/docs/test_shell_status.py`` runs its rows this way.
RUNNER_ARGV: Final[dict[str | None, tuple[str, ...]]] = {
    None: ("bash", "-e"),
    "bash": ("bash", "--noprofile", "--norc", "-eo", "pipefail"),
    "sh": ("sh", "-e"),
}

#: First words that open / close a compound command, a function body included.
_COMPOUND_OPENERS: Final[frozenset[str]] = frozenset(
    {"if", "while", "until", "for", "case", "select", "{", "("}
)
_COMPOUND_CLOSERS: Final[frozenset[str]] = frozenset({"fi", "done", "esac", "}", ")"})
#: A function definition's head -- ``name()``, ``name ()``, ``function name [()]`` -- which
#: opens nothing itself: the compound command after it is the body.
_FUNCTION_HEAD: Final[re.Pattern[str]] = re.compile(
    r"^(?:function\s+(?P<keyword>[^\s(){};&|]+)(?:\s*\(\s*\))?|(?P<name>[^\s(){};&|=]+)\s*\(\s*\))"
)
#: What a compound puts before the command it wraps in one list element: reserved words
#: (``then exit 0``, ``{ exec true``), a ``case WORD in`` head and a case pattern (``a)``).
_LEADING_RESERVED: Final[frozenset[str]] = frozenset(
    {"if", "elif", "then", "else", "while", "until", "do", "!", "{", "("}
)
#: Builtins that run the builtin or command they name in this shell: ``builtin set +e``.
_PASS_THROUGH: Final[frozenset[str]] = frozenset({"builtin", "command"})
#: ``command``'s option letters that only describe a name: ``command -v pytest`` runs nothing.
_DESCRIBE_ONLY: Final[frozenset[str]] = frozenset({"v", "V"})
#: Builtins that run code the model does not read, in this shell.
_UNREAD_CODE: Final[frozenset[str]] = frozenset({"eval", "source", "."})
_TRAP: Final[str] = "trap"
#: ``trap``'s option letters that print or list, installing nothing.
_TRAP_PRINT_LETTERS: Final[frozenset[str]] = frozenset({"p", "P", "l"})
#: ``trap``'s first operand that resets the conditions after it.
_TRAP_RESET: Final[str] = "-"
#: An alias can stand in for a command name (``sh`` expands aliases in a script).
_ALIAS: Final[str] = "alias"
_ALIAS_ASSIGNMENT: Final[str] = "="
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
_SHOPT_BUILTIN: Final[str] = "shopt"
_ERREXIT_LETTER: Final[str] = "e"
_NAMED_OPTION_LETTER: Final[str] = "o"
_SHOPT_ENABLE_LETTER: Final[str] = "s"
_SHOPT_DISABLE_LETTER: Final[str] = "u"
_ERREXIT: Final[str] = "errexit"
_PIPEFAIL: Final[str] = "pipefail"
_SET_PREFIXES: Final[str] = "-+"
_END_OF_OPTIONS: Final[str] = "--"
_OPTION_PREFIX: Final[str] = "-"


def _words(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _options_end(words: list[str], start: int) -> int:
    """Index of the first operand after ``words[start:]``'s options and any ``--``.

    A lone ``-`` is an operand: ``trap - EXIT`` resets the trap.
    """
    index = start
    while index < len(words) and words[index].startswith(_OPTION_PREFIX):
        if words[index] == _OPTION_PREFIX:
            break
        index += 1
        if words[index - 1] == _END_OF_OPTIONS:
            break
    return index


def _command_words(words: list[str]) -> list[str]:
    """The words of the command a compound or a wrapper runs: ``then exit 0`` -> ``exit 0``.

    ``builtin set +e`` and ``command set +e`` run ``set +e``; ``command -v pytest`` runs
    nothing, so its words are empty.
    """
    index = 0
    while index < len(words):
        word = words[index]
        if word == _CASE and _CASE_IN in words[index:]:
            index = words.index(_CASE_IN, index) + 1
        elif word in _LEADING_RESERVED or word.endswith(_CASE_PATTERN_END):
            index += 1
        elif word in _PASS_THROUGH:
            end = _options_end(words, index + 1)
            if any(set(option[1:]) & _DESCRIBE_ONLY for option in words[index + 1 : end]):
                return []
            index = end
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


def _apply_shopt(arguments: list[str], errexit: bool, pipefail: bool) -> tuple[bool, bool]:
    """``shopt``'s effect on ``errexit`` and ``pipefail`` (``shopt -uo errexit``).

    Only ``-o`` reaches ``set``'s options, and only ``-s`` / ``-u`` change them.
    """
    end = _options_end(arguments, 0)
    letters = "".join(option[1:] for option in arguments[:end])
    if _NAMED_OPTION_LETTER not in letters:
        return errexit, pipefail
    if _SHOPT_ENABLE_LETTER in letters:
        enable = True
    elif _SHOPT_DISABLE_LETTER in letters:
        enable = False
    else:
        return errexit, pipefail  # `shopt -o` alone prints
    names = arguments[end:]
    return (
        enable if _ERREXIT in names else errexit,
        enable if _PIPEFAIL in names else pipefail,
    )


#: The builtins that switch ``errexit`` / ``pipefail``, mapped to their effect.
_OPTION_SETTERS: Final[dict[str, Callable[[list[str], bool, bool], tuple[bool, bool]]]] = {
    _SET_BUILTIN: _apply_set,
    _SHOPT_BUILTIN: _apply_shopt,
}


def _surely_runs_here(command: ShellCommand) -> bool:
    """Whether a command runs whatever came before, in this shell rather than a subshell.

    Behind ``&&`` / ``||`` it may not run at all (``false && set -e``); in a pipeline it runs
    in a subshell, whose options die with it.
    """
    return command.before not in (AND_LIST, OR_LIST, PIPE) and command.after != PIPE


def _installs_trap(arguments: list[str]) -> bool:
    """Whether ``trap``'s arguments install an action (``trap 'exit 0' EXIT``).

    Printing (``trap``, ``trap -p``, ``trap -l``) and resetting (``trap - EXIT``, ``trap
    EXIT``) install nothing. Any action does, an empty one included: what an action does
    is not read, and ``exit 0`` on ``ERR`` or ``EXIT`` replaces the failing status.
    """
    end = _options_end(arguments, 0)
    if any(set(option[1:]) & _TRAP_PRINT_LETTERS for option in arguments[:end]):
        return False
    operands = arguments[end:]
    return len(operands) > 1 and operands[0] != _TRAP_RESET


def _program(text: str) -> str:
    """The name a command runs: ``A=1 pytest x`` -> ``pytest``; ``builtin cd d`` -> ``cd``."""
    _, words = split_environment(_words(text))
    body = _command_words(words)
    return body[0] if body else ""


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
    where every branch counts as taken. A ``set`` or ``shopt -o`` switches an option back
    on only where it surely runs, in this shell: inside a compound, behind ``&&``/``||``
    or in a pipeline it can only switch one off. A ``trap`` that installs an action, code
    the model cannot read (``eval``, ``source``, ``.``) and a function or alias named like
    the command are reported wherever they appear before it (Copilot review 3).

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
    shadows: set[str] = set()
    obstacles: list[str] = []
    for command in commands[:position]:
        head = _FUNCTION_HEAD.match(command.text)
        if head:
            shadows.add(head.group("keyword") or head.group("name"))
        words = _words(command.text[head.end() :] if head else command.text) or [""]
        first, body = words[0], _command_words(words)
        verb = body[0] if body else ""
        opens = first in _COMPOUND_OPENERS
        nested = depth > 0 or opens
        if opens:
            depth += 1
        elif first in _COMPOUND_CLOSERS:
            depth = max(0, depth - 1)
        if verb in _OPTION_SETTERS:
            after = _OPTION_SETTERS[verb](body[1:], errexit, pipefail)
            switch_off_only = nested or not _surely_runs_here(command)
            weakened = (errexit and after[0], pipefail and after[1])
            errexit, pipefail = weakened if switch_off_only else after
        elif _ends_script(body, nested):
            where = " inside a compound command" if nested else ""
            obstacles.append(f"`{command.text}`{where} before it can end the step green first")
        elif verb == _TRAP and _installs_trap(body[1:]):
            obstacles.append(f"`{command.text}` before it installs a trap that can end it green")
        elif verb in _UNREAD_CODE:
            obstacles.append(f"`{command.text}` before it runs code this model does not read")
        elif verb == _ALIAS:
            defined = (word for word in body[1:] if _ALIAS_ASSIGNMENT in word)
            shadows |= {word.partition(_ALIAS_ASSIGNMENT)[0] for word in defined}
    program = _program(commands[position].text)
    if program in shadows:
        obstacles.append(f"a function or alias named `{program}` defined before it runs instead")
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


def step_failure_obstacles(
    step: WorkflowStep, commands: Sequence[ShellCommand], position: int
) -> list[str]:
    """Why ``step`` could stay green, or not run, when ``commands[position]`` fails.

    The step's own metadata -- ``continue-on-error`` at the step or its job, an ``if:``
    that can skip it -- then every way :func:`exit_status_obstacles` finds for the
    command's status to be lost. Whether the *job* can fail the build is the caller's
    question: it is ``ci-success``'s gate, not the step's.

    Args:
        step: The step running the command (:func:`~tests.support.workflows.workflow_steps`).
        commands: The step's commands (:func:`~tests.support.workflows.iter_shell_commands`).
        position: Index of the command whose failure must fail the step.

    Returns:
        One reason per way the failure can be lost; empty if it fails the step.

    """
    obstacles: list[str] = []
    if step.continue_on_error:
        obstacles.append("continue-on-error: the step cannot fail its job")
    if step.condition is not None and step.condition.strip() not in ALWAYS_RUN_CONDITIONS:
        obstacles.append(f"if: {step.condition!r} can skip the step")
    obstacles.extend(exit_status_obstacles(commands, position, step.shell))
    return obstacles
