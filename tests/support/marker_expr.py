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

Added for ``tests/docs/test_fem_required_visibility.py`` (the section at the end
of this file), three questions the term helpers cannot answer: whether one test
survives a filter (:func:`expression_matches`, which delegates to pytest's own
compiler rather than becoming that precedence parser), where a marker is
*applied* in test source (:func:`scan_module_markers`, AST, with
:func:`attribute_access_lines` as an independent ``tokenize`` oracle), and what
one pytest command selects (:func:`pytest_invocation`; whether that reaches a
given test is ``tests.support.workflows.selection_obstacles``).
"""

from __future__ import annotations

import ast
import fnmatch
import io
import re
import shlex
import tokenize
from collections import Counter
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Final

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

#: Make variables that expand to a ``coverage`` invocation (``COV ?= $(PYTHON) -m coverage``).
COVERAGE_VARIABLES = frozenset({"$(COV)", "${COV}"})

_COVERAGE_PROGRAM = re.compile(r"^(.*/)?coverage[0-9.]*$")


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


# --------------------------------------------------------------------------- #
# Added for tests/docs/test_fem_required_visibility.py. Additive: nothing above #
# this banner changed behaviour.                                              #
# --------------------------------------------------------------------------- #


def expression_matches(expression: str, markers: Collection[str]) -> bool:
    """Whether ``pytest -m <expression>`` selects a test carrying exactly ``markers``.

    The term helpers above read an expression token by token, which is why the
    module docstring records the ``not (a and b)`` limitation. Deciding whether
    one particular test survives a filter needs the real grammar, so this
    delegates to pytest's own compiler instead of adding a parser that would
    have to agree with it. ``_pytest.mark.expression`` is private API; it is
    imported here rather than at module scope so that a pytest which moves it
    fails only the guards that call this -- loudly, with an ``ImportError``.

    Args:
        expression: A pytest ``-m`` expression. Blank selects every test.
        markers: The marker names the test carries.

    Returns:
        True if pytest would select the test.

    """
    if not expression.strip():
        return True
    from _pytest.mark.expression import Expression

    carried = frozenset(markers)

    def matcher(name: str, /, **kwargs: object) -> bool:
        # `name(key=value)` matches on marker kwargs, which no static scan can
        # see. Answering False makes a positive kwargs term select nothing.
        return name in carried and not kwargs

    return Expression.compile(expression).evaluate(matcher)


#: Owner recorded for an application outside every ``def`` and ``class``.
MODULE_OWNER: Final[str] = "<module>"

#: Separator between the parts of a pytest node id.
NODEID_SEPARATOR: Final[str] = "::"

#: The syntactic forms that attach a marker to collected tests.
FORM_DECORATOR: Final[str] = "decorator"
FORM_PYTESTMARK: Final[str] = "pytestmark"
FORM_PARAM: Final[str] = "param"
FORM_DYNAMIC: Final[str] = "dynamic"

#: pytest's built-in ``python_functions`` / ``python_classes`` defaults.
DEFAULT_FUNCTION_PATTERNS: Final[tuple[str, ...]] = ("test",)
DEFAULT_CLASS_PATTERNS: Final[tuple[str, ...]] = ("Test",)

_PYTEST: Final[str] = "pytest"
_MARK: Final[str] = "mark"
_PARAM: Final[str] = "param"
_PYTESTMARK: Final[str] = "pytestmark"
_MARKS_KEYWORD: Final[str] = "marks"
_MARKER_KEYWORD: Final[str] = "marker"
_WITH_ARGS: Final[str] = "with_args"
_CONSTRUCTOR: Final[str] = "__init__"
_DYNAMIC_METHODS: Final[frozenset[str]] = frozenset({"add_marker", "applymarker"})
_GLOB_CHARACTERS: Final[frozenset[str]] = frozenset("*?[")
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


@dataclass(frozen=True)
class MarkerApplication:
    """One place in a module's source that attaches a marker to tests.

    ``owner`` is the ``::``-joined path of the enclosing ``def``/``class``
    (``TestA::test_b``) or :data:`MODULE_OWNER`. ``line`` is the line holding
    the marker's *name*, which is what :func:`attribute_access_lines` reports.
    """

    marker: str
    form: str
    owner: str
    line: int


@dataclass(frozen=True)
class CollectableTest:
    """A test pytest's discovery would collect, with every statically visible marker.

    One entry per test function, plus one per ``pytest.param(..., marks=...)``
    in its decorators, sharing the node id. ``markers`` unions the module's
    ``pytestmark``, every enclosing class (decorators, ``pytestmark``, and the
    same-module base classes pytest reaches through the MRO) and the
    function's own decorators.
    """

    nodeid: str
    markers: frozenset[str]
    line: int


@dataclass(frozen=True)
class ModuleMarkers:
    """What :func:`scan_module_markers` reads from one module."""

    applications: tuple[MarkerApplication, ...]
    tests: tuple[CollectableTest, ...]


def name_matches_patterns(name: str, patterns: Sequence[str]) -> bool:
    """Whether ``name`` passes pytest's ``python_functions``/``python_classes`` rule.

    A pattern matches as a prefix or, when it contains a glob character,
    through ``fnmatch`` -- the order ``PyCollector`` checks them in.

    Args:
        name: A function or class name.
        patterns: The configured patterns.

    Returns:
        True if any pattern matches.

    """
    return any(
        name.startswith(pattern)
        or (bool(_GLOB_CHARACTERS & set(pattern)) and fnmatch.fnmatch(name, pattern))
        for pattern in patterns
    )


def _owner(scope: tuple[str, ...]) -> str:
    return NODEID_SEPARATOR.join(scope) or MODULE_OWNER


def _flatten_mark_values(node: ast.expr) -> list[ast.expr]:
    """The leaves of a ``pytestmark`` / ``marks=`` value, however the sequence is spelled."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [leaf for element in node.elts for leaf in _flatten_mark_values(element)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_mark_values(node.left) + _flatten_mark_values(node.right)
    if isinstance(node, ast.IfExp):
        return _flatten_mark_values(node.body) + _flatten_mark_values(node.orelse)
    if isinstance(node, ast.Starred):
        return _flatten_mark_values(node.value)
    return [node]


def assigned_names(statement: ast.stmt) -> set[str]:
    """Plain names a statement binds by ``=``, ``: T =`` or an augmented assignment.

    Args:
        statement: Any statement node.

    Returns:
        The bound names; empty for any other statement.

    """
    if isinstance(statement, ast.Assign):
        return {target.id for target in statement.targets if isinstance(target, ast.Name)}
    if isinstance(statement, (ast.AnnAssign, ast.AugAssign)) and isinstance(
        statement.target, ast.Name
    ):
        return {statement.target.id}
    return set()


def _namespace_statements(statements: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """Every statement bound in one namespace, through ``if``/``try``/``with``/loops.

    A ``def`` is yielded but not entered: its body is another namespace.
    """
    for statement in statements:
        yield statement
        if isinstance(statement, _DEFINITIONS):
            continue
        for child in ast.iter_child_nodes(statement):
            if isinstance(child, ast.stmt):
                yield from _namespace_statements([child])
            elif isinstance(child, (ast.excepthandler, ast.match_case)):
                yield from _namespace_statements(child.body)


def _definitions(statements: Sequence[ast.stmt]) -> dict[str, _Definition]:
    """Defs bound in one namespace by name; a later binding replaces an earlier one."""
    return {
        statement.name: statement
        for statement in _namespace_statements(statements)
        if isinstance(statement, _DEFINITIONS)
    }


class _MarkerScanner:
    """Resolves the ``pytest`` / ``mark`` / ``param`` spellings one module uses."""

    def __init__(self, tree: ast.Module) -> None:
        self._pytest_names = {_PYTEST}
        self._mark_names: set[str] = set()
        self._param_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self._pytest_names.update(
                    alias.asname or alias.name for alias in node.names if alias.name == _PYTEST
                )
            elif isinstance(node, ast.ImportFrom) and node.module == _PYTEST and not node.level:
                for alias in node.names:
                    if alias.name == _MARK:
                        self._mark_names.add(alias.asname or alias.name)
                    elif alias.name == _PARAM:
                        self._param_names.add(alias.asname or alias.name)
        self.applications: list[MarkerApplication] = []

    def _is_pytest_attribute(self, node: ast.expr, attribute: str) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == attribute
            and isinstance(node.value, ast.Name)
            and node.value.id in self._pytest_names
        )

    def marker_attribute(self, node: ast.expr) -> ast.Attribute | None:
        """The ``<mark>.<name>`` node behind a marker expression, or ``None``."""
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr == _WITH_ARGS:
                return self.marker_attribute(function.value)
            return self.marker_attribute(function)
        if not isinstance(node, ast.Attribute):
            return None
        generator = node.value
        if isinstance(generator, ast.Name) and generator.id in self._mark_names:
            return node
        return node if self._is_pytest_attribute(generator, _MARK) else None

    def marker_names(self, nodes: Sequence[ast.expr]) -> set[str]:
        """The marker names among ``nodes``; anything else is ignored."""
        return {
            attribute.attr
            for node in nodes
            if (attribute := self.marker_attribute(node)) is not None
        }

    def _is_param_call(self, node: ast.Call) -> bool:
        function = node.func
        if isinstance(function, ast.Name):
            return function.id in self._param_names
        return self._is_pytest_attribute(function, _PARAM)

    def _record(self, attribute: ast.Attribute, form: str, scope: tuple[str, ...]) -> None:
        line = attribute.end_lineno or attribute.lineno
        self.applications.append(MarkerApplication(attribute.attr, form, _owner(scope), line))

    def _param_marks(self, call: ast.Call) -> list[ast.expr]:
        return [
            leaf
            for keyword in call.keywords
            if keyword.arg == _MARKS_KEYWORD
            for leaf in _flatten_mark_values(keyword.value)
        ]

    def _scan_calls(self, node: ast.AST, scope: tuple[str, ...]) -> None:
        """Record ``pytest.param(marks=...)`` and ``add_marker``/``applymarker`` calls."""
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if self._is_param_call(call):
                for leaf in self._param_marks(call):
                    attribute = self.marker_attribute(leaf)
                    if attribute is not None:
                        self._record(attribute, FORM_PARAM, scope)
            elif isinstance(call.func, ast.Attribute) and call.func.attr in _DYNAMIC_METHODS:
                self._record_dynamic(call, scope)

    def _record_dynamic(self, call: ast.Call, scope: tuple[str, ...]) -> None:
        keywords = [k.value for k in call.keywords if k.arg == _MARKER_KEYWORD]
        argument = call.args[0] if call.args else (keywords[0] if keywords else None)
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            self.applications.append(
                MarkerApplication(argument.value, FORM_DYNAMIC, _owner(scope), call.lineno)
            )
            return
        attribute = None if argument is None else self.marker_attribute(argument)
        if attribute is not None:
            self._record(attribute, FORM_DYNAMIC, scope)

    def visit(self, statements: Sequence[ast.stmt], scope: tuple[str, ...], nested: bool) -> None:
        """Record every application in ``statements``.

        ``nested`` is true inside a function, where a ``pytestmark`` binding is
        a local variable that pytest never reads.
        """
        for statement in statements:
            if isinstance(statement, _DEFINITIONS):
                inner = (*scope, statement.name)
                for decorator in statement.decorator_list:
                    attribute = self.marker_attribute(decorator)
                    if attribute is not None:
                        self._record(attribute, FORM_DECORATOR, inner)
                    self._scan_calls(decorator, inner)
                if isinstance(statement, ast.ClassDef):
                    for base in (*statement.bases, *(k.value for k in statement.keywords)):
                        self._scan_calls(base, scope)
                self.visit(statement.body, inner, nested or isinstance(statement, _FUNCTIONS))
                continue
            value = getattr(statement, "value", None)
            if not nested and _PYTESTMARK in assigned_names(statement) and value is not None:
                for leaf in _flatten_mark_values(value):
                    attribute = self.marker_attribute(leaf)
                    if attribute is not None:
                        self._record(attribute, FORM_PYTESTMARK, scope)
            for child in ast.iter_child_nodes(statement):
                if isinstance(child, ast.stmt):
                    self.visit([child], scope, nested)
                elif isinstance(child, (ast.excepthandler, ast.match_case)):
                    self.visit(child.body, scope, nested)
                    for part in ast.iter_child_nodes(child):
                        if isinstance(part, ast.expr):
                            self._scan_calls(part, scope)
                else:
                    self._scan_calls(child, scope)

    # -- static model of collection ------------------------------------------------

    def pytestmark_markers(self, statements: Sequence[ast.stmt]) -> frozenset[str]:
        """Markers a namespace's own ``pytestmark`` bindings carry."""
        found: set[str] = set()
        for statement in _namespace_statements(statements):
            value = getattr(statement, "value", None)
            if _PYTESTMARK in assigned_names(statement) and value is not None:
                found |= self.marker_names(_flatten_mark_values(value))
        return frozenset(found)

    def decorator_markers(self, definition: _Definition) -> frozenset[str]:
        """Markers applied by a definition's own decorators."""
        return frozenset(self.marker_names(definition.decorator_list))

    def param_mark_groups(self, definition: _Definition) -> list[frozenset[str]]:
        """One marker set per ``pytest.param(..., marks=...)`` in the decorators."""
        groups: list[frozenset[str]] = []
        for decorator in definition.decorator_list:
            for call in ast.walk(decorator):
                if isinstance(call, ast.Call) and self._is_param_call(call):
                    names = self.marker_names(self._param_marks(call))
                    if names:
                        groups.append(frozenset(names))
        return groups


@dataclass
class _Collector:
    """pytest's discovery over one module, reduced to names and markers."""

    scanner: _MarkerScanner
    classes: dict[str, _Definition]
    function_patterns: Sequence[str]
    class_patterns: Sequence[str]
    tests: list[CollectableTest] = field(default_factory=list)

    def _same_module_bases(self, cls: ast.ClassDef, seen: frozenset[str]) -> list[ast.ClassDef]:
        return [
            base
            for node in cls.bases
            if isinstance(node, ast.Name) and node.id not in seen
            for base in [self.classes.get(node.id)]
            if isinstance(base, ast.ClassDef)
        ]

    def class_markers(self, cls: ast.ClassDef, seen: frozenset[str]) -> frozenset[str]:
        """A class's markers, including those pytest finds on its bases via the MRO."""
        found = set(self.scanner.decorator_markers(cls)) | self.scanner.pytestmark_markers(cls.body)
        for base in self._same_module_bases(cls, seen | {cls.name}):
            found |= self.class_markers(base, seen | {cls.name})
        return frozenset(found)

    def members(self, cls: ast.ClassDef, seen: frozenset[str]) -> dict[str, _Definition]:
        """A class's own defs, then inherited ones it does not override."""
        found = _definitions(cls.body)
        for base in self._same_module_bases(cls, seen | {cls.name}):
            for name, member in self.members(base, seen | {cls.name}).items():
                found.setdefault(name, member)
        return found

    def add_function(
        self, function: _Definition, prefix: tuple[str, ...], inherited: frozenset[str]
    ) -> None:
        nodeid = NODEID_SEPARATOR.join((*prefix, function.name))
        base = inherited | self.scanner.decorator_markers(function)
        self.tests.append(CollectableTest(nodeid, base, function.lineno))
        for group in self.scanner.param_mark_groups(function):
            self.tests.append(CollectableTest(nodeid, base | group, function.lineno))

    def add_class(
        self, cls: ast.ClassDef, prefix: tuple[str, ...], inherited: frozenset[str]
    ) -> None:
        members = self.members(cls, frozenset())
        if _CONSTRUCTOR in members:
            return  # pytest refuses to collect a class with an __init__
        path = (*prefix, cls.name)
        markers = inherited | self.class_markers(cls, frozenset())
        self.add_namespace(members.values(), path, markers)

    def add_namespace(
        self,
        definitions: Collection[_Definition],
        prefix: tuple[str, ...],
        inherited: frozenset[str],
    ) -> None:
        for definition in definitions:
            if isinstance(definition, ast.ClassDef):
                if name_matches_patterns(definition.name, self.class_patterns):
                    self.add_class(definition, prefix, inherited)
            elif name_matches_patterns(definition.name, self.function_patterns):
                self.add_function(definition, prefix, inherited)


def scan_module_markers(
    source: str,
    *,
    function_patterns: Sequence[str] = DEFAULT_FUNCTION_PATTERNS,
    class_patterns: Sequence[str] = DEFAULT_CLASS_PATTERNS,
) -> ModuleMarkers:
    """Where ``source`` applies markers, and the tests that end up carrying them.

    Read by AST, so a marker named in a docstring, a comment or a string literal
    is never an application. Recognised: decorators on functions and classes;
    ``pytestmark`` bound at module or class level (``=``, ``: T =``, ``+=``;
    lists, tuples, ``+`` and conditional expressions); ``pytest.param(...,
    marks=...)``; and ``add_marker`` / ``applymarker`` with a marker or its
    name. Spellings through ``import pytest as X`` and ``from pytest import
    mark``/``param`` resolve. Anything else -- an alias such as ``fem =
    pytest.mark.x`` -- is deliberately *not* an application:
    :func:`attribute_access_lines` sees it, and a guard comparing the two
    reports the difference instead of trusting a name it cannot follow.

    Args:
        source: Python source of one module.
        function_patterns: pytest's ``python_functions``.
        class_patterns: pytest's ``python_classes``.

    Returns:
        The applications, and the static model of the collected tests.

    """
    tree = ast.parse(source)
    scanner = _MarkerScanner(tree)
    scanner.visit(tree.body, (), nested=False)
    definitions = _definitions(tree.body)
    collector = _Collector(
        scanner=scanner,
        classes={name: d for name, d in definitions.items() if isinstance(d, ast.ClassDef)},
        function_patterns=function_patterns,
        class_patterns=class_patterns,
    )
    collector.add_namespace(definitions.values(), (), scanner.pytestmark_markers(tree.body))
    return ModuleMarkers(tuple(scanner.applications), tuple(dict.fromkeys(collector.tests)))


#: Tokens that may sit between an attribute's ``.`` and its name.
_TRIVIA_TOKENS: Final[frozenset[int]] = frozenset({tokenize.NL, tokenize.COMMENT})


def attribute_access_lines(source: str, attribute: str) -> Counter[int]:
    """Every ``.<attribute>`` access in *code*, as a multiset of line numbers.

    Read with :mod:`tokenize`, so comments, docstrings and string literals never
    count. Independent of the AST scan on purpose: an access the scan does not
    record as an application is a spelling it cannot follow, and comparing the
    two is how that becomes a failure rather than a gap.

    Args:
        source: Python source of one module.
        attribute: The attribute name, e.g. a marker name.

    Returns:
        Line number -> number of accesses on that line.

    """
    counts: Counter[int] = Counter()
    previous: tokenize.TokenInfo | None = None
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in _TRIVIA_TOKENS:
            continue
        if (
            token.type == tokenize.NAME
            and token.string == attribute
            and previous is not None
            and previous.type == tokenize.OP
            and previous.string == "."
        ):
            counts[token.start[0]] += 1
        previous = token
    return counts


# --------------------------------------------------------------------------- #
# What one pytest command selects                                              #
# --------------------------------------------------------------------------- #

#: Short options that take a value, attached (``-mexpr``) or as the next token.
_SHORT_VALUE_OPTIONS: Final[frozenset[str]] = frozenset("mkpcoWnr")

#: Short flags.
_SHORT_FLAGS: Final[frozenset[str]] = frozenset("qvxsl")

# The three option tables below are whitespace-separated words, not list
# literals (SIM905): as lists, ruff format puts one option per line, ~100 lines
# that would push this module past the 1000-line ceiling
# tests/docs/test_module_size_budget.py enforces on every file under tests/.

#: Long options that take a value when written without ``=``.
_LONG_VALUE_OPTIONS: Final[frozenset[str]] = frozenset(
    """
    --allow-hosts --basetemp --capture --color --config-file --confcutdir --cov --cov-config
    --cov-context --cov-fail-under --cov-report --deselect --dist --durations --durations-min
    --hypothesis-profile --hypothesis-seed --ignore --ignore-glob --import-mode --junit-xml
    --junitxml --log-cli-level --log-file --log-file-level --log-level --maxfail
    --numprocesses --override-ini --report-log --rootdir --show-capture --tb --timeout
    """.split()  # noqa: SIM905
)

#: Long flags.
_LONG_FLAGS: Final[frozenset[str]] = frozenset(
    """
    --allow-unix-socket --cache-clear --continue-on-collection-errors --cov-append
    --cov-branch --disable-pytest-warnings --disable-socket --disable-warnings
    --doctest-modules --exitfirst --failed-first --ff --force-enable-socket --full-trace
    --hypothesis-show-statistics --keep-duplicates --last-failed --lf --new-first --nf
    --no-cov --no-cov-on-fail --no-header --no-summary --quiet --runxfail --setup-show
    --showlocals --stepwise --stepwise-skip --strict --strict-config --strict-markers --sw
    --sw-skip --verbose
    """.split()  # noqa: SIM905
)

#: Options under which an argv no longer describes what runs: they collect
#: without executing, or swap the configuration, rootdir or conftests this
#: model assumes.
_INERT_OPTIONS: Final[frozenset[str]] = frozenset(
    """
    -c -h -o --co --collect-only --config-file --confcutdir --fixtures --fixtures-per-test
    --help --markers --noconftest --override-ini --pyargs --rootdir --setup-only
    --setup-plan --version
    """.split()  # noqa: SIM905
)

#: ``NAME=value`` before the program: an environment assignment for one command.
_ENV_ASSIGNMENT: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

#: ``python -m``; interpreter options after which argv is no module's, and ones taking a value.
_MODULE_OPTION: Final[str] = "-m"
_INTERPRETER_CODE_OPTIONS: Final[frozenset[str]] = frozenset({"-c", "-"})
_INTERPRETER_VALUE_OPTIONS: Final[frozenset[str]] = frozenset({"-W", "-X"})
#: ``coverage run -m pytest``, and the ``run`` options that take a value without ``=``.
_COVERAGE: Final[str] = "coverage"
_COVERAGE_RUN: Final[str] = "run"
_COVERAGE_RUN_VALUE_OPTIONS: Final[frozenset[str]] = frozenset(
    "--concurrency --context --data-file --debug --include --omit --rcfile --source".split()  # noqa: SIM905
)


@dataclass(frozen=True)
class PytestInvocation:
    """What one ``pytest`` command selects, read from its argv.

    ``marker_expression`` and ``keyword_expression`` are the effective ones
    (pytest keeps the last ``-m``/``-k``), with any ``addopts`` given to
    :func:`pytest_invocation` applied first, as pytest applies them.
    """

    env: dict[str, str]
    paths: tuple[str, ...]
    ignores: tuple[str, ...]
    ignore_globs: tuple[str, ...]
    deselects: tuple[str, ...]
    marker_expression: str
    keyword_expression: str
    unknown_options: tuple[str, ...]
    inert_options: tuple[str, ...]


def split_environment(tokens: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Separate leading ``NAME=value`` assignments from the command they prefix.

    Args:
        tokens: A shell-split command.

    Returns:
        The assignments, and the remaining tokens starting at the program.

    """
    env: dict[str, str] = {}
    index = 0
    while index < len(tokens) and _ENV_ASSIGNMENT.match(tokens[index]):
        name, _, value = tokens[index].partition("=")
        env[name] = value
        index += 1
    return env, list(tokens[index:])


def is_coverage_program(token: str) -> bool:
    """Whether a command's first token invokes ``coverage`` directly (or ``$(COV)``)."""
    return token in COVERAGE_VARIABLES or bool(_COVERAGE_PROGRAM.match(token))


def _interpreter_module(arguments: list[str]) -> tuple[str, list[str]] | None:
    """``(module, argv)`` for ``python [options] -m module``; ``None`` for a script/``-c``/``-``."""
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument.startswith(_MODULE_OPTION):  # `-m module`, or attached `-mmodule`
            rest = arguments[index + 1 :]
            module = argument[len(_MODULE_OPTION) :] or (rest.pop(0) if rest else "")
            return (module, rest) if module else None
        if argument in _INTERPRETER_CODE_OPTIONS or not argument.startswith("-"):
            return None  # what follows is that program's argv, `-m pytest` included
        index += 2 if argument in _INTERPRETER_VALUE_OPTIONS else 1
    return None


def _coverage_run_pytest(arguments: list[str]) -> list[str] | None:
    """Pytest's argv from ``coverage``'s own argv: ``run [options] -m pytest ...``."""
    if not arguments or arguments[0] != _COVERAGE_RUN:
        return None
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument == _MODULE_OPTION:
            return arguments[index + 2 :] if arguments[index + 1 : index + 2] == [_PYTEST] else None
        if not argument.startswith("-"):
            return None  # `coverage run some/program.py`: not modelled, so not counted
        index += 2 if argument in _COVERAGE_RUN_VALUE_OPTIONS else 1
    return None


def _pytest_arguments(tokens: list[str]) -> list[str] | None:
    """Pytest's own argv, or ``None`` -- ``echo -m pytest ...`` runs nothing (PR #160)."""
    if not tokens:
        return None
    program, arguments = tokens[0], tokens[1:]
    if is_pytest_program(program):
        return arguments
    if is_coverage_program(program):
        return _coverage_run_pytest(arguments)
    found = _interpreter_module(arguments) if is_python_program(program) else None
    if found is None or found[0] not in (_PYTEST, _COVERAGE):
        return None
    return found[1] if found[0] == _PYTEST else _coverage_run_pytest(found[1])


@dataclass
class _ArgvReader:
    """Mutable accumulator for :func:`pytest_invocation`."""

    arguments: list[str]
    index: int = 0
    paths: list[str] = field(default_factory=list)
    options: dict[str, list[str]] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)
    inert: list[str] = field(default_factory=list)

    def next_value(self) -> str:
        """Consume the next argument as a value -- unless it is itself an option.

        argparse never takes ``-m`` as the value of a preceding ``--cov``;
        consuming it here would lose the marker filter and read its expression
        as a path.
        """
        if self.index < len(self.arguments) and not self.arguments[self.index].startswith("-"):
            self.index += 1
            return self.arguments[self.index - 1]
        return ""

    def record(self, option: str, value: str) -> None:
        self.options.setdefault(option, []).append(value)
        if option in _INERT_OPTIONS:
            self.inert.append(option)

    def read_long(self, token: str) -> None:
        name, has_value, value = token.partition("=")
        if not has_value:
            if name in _LONG_VALUE_OPTIONS:
                value = self.next_value()
            elif name not in _LONG_FLAGS and name not in _INERT_OPTIONS:
                self.unknown.append(name)
        self.record(name, value)

    def read_short(self, token: str) -> None:
        letters = token[1:]
        for position, letter in enumerate(letters):
            option = f"-{letter}"
            if letter in _SHORT_VALUE_OPTIONS:
                self.record(option, letters[position + 1 :] or self.next_value())
                return
            if letter not in _SHORT_FLAGS and option not in _INERT_OPTIONS:
                self.unknown.append(option)
            self.record(option, "")

    def read(self) -> None:
        positional_only = False
        while self.index < len(self.arguments):
            token = self.arguments[self.index]
            self.index += 1
            if positional_only or not token.startswith("-") or token == "-":
                self.paths.append(token)
            elif token == "--":
                positional_only = True
            elif token.startswith("--"):
                self.read_long(token)
            else:
                self.read_short(token)

    def last(self, option: str) -> str:
        values = self.options.get(option)
        return values[-1] if values else ""


def pytest_invocation(command: str, *, addopts: Sequence[str] = ()) -> PytestInvocation | None:
    """Read one shell command as a pytest invocation.

    Recognised entry points: ``pytest``/``$(PYTEST)``, ``python [interpreter
    options] -m pytest``, and ``coverage run [options] -m pytest`` with
    ``coverage`` as a program, ``$(COV)`` or ``python -m coverage``. A program
    that merely carries ``-m pytest`` in its argv (``echo``, ``python
    script.py``) is not one. Leading ``NAME=value`` assignments are
    returned as ``env``. ``addopts`` are prepended to the arguments, which is
    where pytest inserts ``[tool.pytest.ini_options] addopts`` and
    ``PYTEST_ADDOPTS``.

    Args:
        command: One logical shell command (see ``iter_commands``).
        addopts: Arguments pytest would prepend.

    Returns:
        The invocation, or ``None`` when the command does not run pytest or
        cannot be shell-split.

    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    env, program = split_environment(tokens)
    arguments = _pytest_arguments(program)
    if arguments is None:
        return None
    reader = _ArgvReader([*addopts, *arguments])
    reader.read()
    return PytestInvocation(
        env=env,
        paths=tuple(reader.paths),
        ignores=tuple(reader.options.get("--ignore", ())),
        ignore_globs=tuple(reader.options.get("--ignore-glob", ())),
        deselects=tuple(reader.options.get("--deselect", ())),
        marker_expression=reader.last("-m"),
        keyword_expression=reader.last("-k"),
        unknown_options=tuple(dict.fromkeys(reader.unknown)),
        inert_options=tuple(dict.fromkeys(reader.inert)),
    )
