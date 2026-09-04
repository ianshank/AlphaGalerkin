"""Parsing of pytest ``-m`` marker expressions out of shell commands.

``--strict-markers`` (pyproject.toml) rejects an unregistered marker applied to
a *test*. It says nothing about an unregistered identifier inside a ``-m``
expression: ``pytest -m "not gpu_requried"`` is accepted, matches nothing, and
therefore **deselects nothing** -- the filter silently becomes a no-op and the
job runs more than it claims. The only way to catch that is to read the
expressions as data and check their vocabulary, which is what this module
exists for.

The hard part is not the expression grammar, it is telling a marker expression
apart from ``python -m <module>``. The rules, in order:

1. In a command whose program is a Python interpreter, the **first** ``-m`` is
   the module invocation and is never a marker expression.
2. A **quoted** argument is a marker expression. Every ``-m`` filter in this
   repo is quoted, and quoting a module name is pathological.
3. An **unquoted** argument is a marker expression only when the command's
   program is ``pytest`` itself (``pytest -m e2e``). Otherwise -- notably
   ``$(COV) run ... -m pytest`` -- it is a module name and is skipped.

Known limitation, recorded rather than hidden: ``not`` binding is resolved
token-wise, so ``not (a and b)`` marks only ``a`` as negated. No expression in
this repo uses that form, and the alternative is a real precedence parser for
no present gain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Boolean connectives pytest's ``-m`` grammar accepts. Everything else that
#: looks like an identifier is a marker name.
BOOLEAN_OPERATORS = frozenset({"not", "and", "or"})

#: ``-m`` preceded by a word or dash character is part of a longer flag
#: (``--m``, ``--cov-m``), never the marker/module flag.
_DASH_M = re.compile(r"(?<![\w-])-m\s+(?P<arg>\"[^\"]*\"|'[^']*'|\S+)")

_IDENTIFIER_OR_PAREN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[()]")

_PYTHON_PROGRAM = re.compile(r"^(.*/)?python[0-9.]*$")

#: Make variables that expand to a Python interpreter invocation.
PYTHON_VARIABLES = frozenset({"$(PYTHON)", "${PYTHON}"})

#: Make variables that expand to a ``pytest`` invocation.
PYTEST_VARIABLES = frozenset({"$(PYTEST)", "${PYTEST}"})


@dataclass(frozen=True)
class MarkerTerm:
    """One marker name inside a ``-m`` expression, with its polarity."""

    name: str
    negated: bool


def is_python_program(token: str) -> bool:
    """Whether a command's first token invokes a Python interpreter.

    Args:
        token: The program token of a shell command.

    Returns:
        True for ``python``, ``python3.11``, ``/usr/bin/python`` and the
        ``$(PYTHON)`` Make variable.

    """
    return token in PYTHON_VARIABLES or bool(_PYTHON_PROGRAM.match(token))


def is_pytest_program(token: str) -> bool:
    """Whether a command's first token invokes ``pytest`` directly.

    Args:
        token: The program token of a shell command.

    Returns:
        True for ``pytest``, a path ending in ``/pytest``, and ``$(PYTEST)``.

    """
    return token in PYTEST_VARIABLES or token == "pytest" or token.endswith("/pytest")


def marker_expressions(command: str) -> list[str]:
    """Every pytest ``-m`` marker expression in one shell command.

    Args:
        command: A single logical command (see
            :func:`tests.support.workflows.iter_commands`).

    Returns:
        The expression text of each ``-m`` that is a marker filter, in source
        order, with surrounding quotes removed. Module invocations
        (``python -m pytest``) are excluded.

    """
    tokens = command.split()
    if not tokens:
        return []
    program = tokens[0]
    python_led = is_python_program(program)
    pytest_led = is_pytest_program(program)

    expressions: list[str] = []
    for index, match in enumerate(_DASH_M.finditer(command)):
        if python_led and index == 0:
            continue  # `python -m <module>`
        raw = match.group("arg")
        if raw[0] in "\"'":
            expressions.append(raw[1:-1])
        elif pytest_led:
            expressions.append(raw)
    return expressions


def parse_terms(expression: str) -> list[MarkerTerm]:
    """Split a marker expression into its named terms and their polarity.

    Args:
        expression: A pytest ``-m`` expression, e.g. ``"not slow and e2e"``.

    Returns:
        One :class:`MarkerTerm` per marker name, in source order. An expression
        containing only operators yields an empty list.

    """
    terms: list[MarkerTerm] = []
    pending_not = False
    for token in _IDENTIFIER_OR_PAREN.findall(expression):
        if token in ("(", ")"):
            continue
        if token == "not":
            pending_not = not pending_not
            continue
        if token in BOOLEAN_OPERATORS:
            pending_not = False
            continue
        terms.append(MarkerTerm(name=token, negated=pending_not))
        pending_not = False
    return terms


def marker_identifiers(expression: str) -> list[str]:
    """The distinct marker names an expression references, in source order.

    Args:
        expression: A pytest ``-m`` expression.

    Returns:
        Marker names with boolean operators and parentheses removed, deduped
        while preserving first-appearance order.

    """
    seen: dict[str, None] = {}
    for term in parse_terms(expression):
        seen.setdefault(term.name, None)
    return list(seen)


def expression_excludes(expression: str, marker: str) -> bool:
    """Whether an expression deselects tests carrying ``marker``.

    Args:
        expression: A pytest ``-m`` expression.
        marker: The marker name to test for.

    Returns:
        True if ``marker`` appears negated (``not <marker>``).

    """
    return any(term.name == marker and term.negated for term in parse_terms(expression))


def expression_requires(expression: str, marker: str) -> bool:
    """Whether an expression positively selects on ``marker``.

    Args:
        expression: A pytest ``-m`` expression.
        marker: The marker name to test for.

    Returns:
        True if ``marker`` appears un-negated.

    """
    return any(term.name == marker and not term.negated for term in parse_terms(expression))


def expression_selects_plainly(expression: str, marker: str) -> bool:
    """Whether an expression selects ordinary tests carrying ``marker``.

    "Ordinary" means: the expression neither excludes ``marker`` nor narrows the
    run to some *other* positively-required marker. ``fem_required and not
    gpu_required`` mentions no ``e2e`` term at all, yet it selects only the
    optional-extra subset -- so it cannot stand in as proof that the ``e2e``
    tier as a whole is run.

    Args:
        expression: A pytest ``-m`` expression. The empty string (no ``-m`` at
            all) selects everything and therefore returns True.
        marker: The marker whose ordinary tests must survive the filter.

    Returns:
        True if a test carrying only ``marker`` would be selected.

    """
    terms = parse_terms(expression)
    if any(term.name == marker and term.negated for term in terms):
        return False
    return not any(term.name != marker and not term.negated for term in terms)
