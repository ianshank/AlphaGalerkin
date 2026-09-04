"""Guards the vocabulary of every pytest ``-m`` expression in CI and the Makefile.

``--strict-markers`` (pyproject.toml ``addopts``) makes an unregistered marker on
a **test** a hard error. It does nothing at all for an unregistered identifier
inside a ``-m`` **expression**::

    pytest tests/ -m "not gpu_requried"     # typo -- accepted, matches nothing

pytest evaluates an unknown name as false, so ``not gpu_requried`` is true for
every test and the filter silently becomes a no-op. The job then runs *more*
than its command claims -- or, with the polarity flipped (``-m "gpu_requried"``),
selects nothing and reports green on an empty run. Both are the same defect
class this repo has hit three times through coverage gates and test directories:
a control that is reported as enforcement while enforcing nothing.

There is no way for pytest to catch this, because ``-m`` is a boolean query over
markers, not a declaration. It has to be checked as text, which this file does:
every ``-m`` expression in ``.github/workflows/*.yml`` and the ``Makefile`` is
parsed, its identifiers extracted, and each one required to be in pyproject.toml's
``[tool.pytest.ini_options] markers`` list.

Hermetic: parses YAML, TOML and Make text; runs nothing.

The parser lives in ``tests/support/marker_expr.py`` (shared with
``tests/docs/test_e2e_visibility.py``, which needs the same expressions for a
different question) and is unit-tested on synthetic input by
:class:`TestExpressionExtraction` and :class:`TestTermParsing` below -- including
the typo case, which never appears in the live data and would otherwise be a
branch nothing exercises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.support.marker_expr import (
    BOOLEAN_OPERATORS,
    MarkerTerm,
    expression_excludes,
    expression_requires,
    expression_selects_plainly,
    marker_expressions,
    marker_identifiers,
    parse_terms,
)
from tests.support.workflows import (
    MAKE_RECIPE_PREFIXES,
    MAKEFILE,
    REPO_ROOT,
    WORKFLOW_DIR,
    iter_commands,
    iter_run_scripts,
    makefile_commands,
    makefile_target_recipe,
    strip_recipe_prefixes,
)

PYPROJECT = REPO_ROOT / "pyproject.toml"

#: The TOML table declaring pytest's marker vocabulary.
MARKERS_TABLE_KEY = "markers"

#: Minimum plausible size of the parsed marker list and of the discovered
#: expression set. Both are non-vacuity floors: a regex that stops matching
#: makes every assertion in this file pass on an empty iterable.
MIN_REGISTERED_MARKERS = 5
MIN_MARKER_EXPRESSIONS = 5

_MARKERS_BLOCK = re.compile(rf"^{MARKERS_TABLE_KEY}\s*=\s*\[(?P<body>.*?)^\]", re.M | re.S)
_MARKER_ENTRY = re.compile(r'^\s*"(?P<entry>.*)",?\s*$')


def registered_markers(pyproject: Path = PYPROJECT) -> set[str]:
    """The marker names declared in ``[tool.pytest.ini_options] markers``.

    Parsed with an anchored regex rather than ``tomllib``, following the
    sibling ``tests/docs/test_version_consistency.py``: ``tomllib`` is stdlib
    only from 3.11 and this repo's declared floor is 3.10, so importing it here
    would be a *collection* error on the oldest supported interpreter -- taking
    the whole run down rather than this file.

    Args:
        pyproject: Path to the ``pyproject.toml`` to read.

    Returns:
        Marker names, i.e. the text before the first ``:`` of each entry.

    """
    text = pyproject.read_text(encoding="utf-8")
    match = _MARKERS_BLOCK.search(text)
    assert match, f"could not locate the `{MARKERS_TABLE_KEY}` list in {pyproject.name}"
    names: set[str] = set()
    for line in match.group("body").splitlines():
        entry = _MARKER_ENTRY.match(line)
        if entry is None:
            continue
        names.add(entry.group("entry").split(":", 1)[0].strip())
    return names


@dataclass(frozen=True)
class MarkerUse:
    """One ``-m`` expression, with the file and command it came from."""

    source: str
    command: str
    expression: str

    def __str__(self) -> str:  # pragma: no cover - failure-message sugar only
        return f"{self.source}: -m {self.expression!r}"


def iter_marker_uses() -> list[MarkerUse]:
    """Every pytest ``-m`` marker expression in CI workflows and the Makefile.

    Returns:
        One :class:`MarkerUse` per expression, workflows first (filename then
        step order), then the Makefile.

    """
    uses: list[MarkerUse] = []
    for run in iter_run_scripts():
        for command in iter_commands(run.script):
            for expression in marker_expressions(command):
                uses.append(
                    MarkerUse(
                        source=f"{run.workflow}::{run.job}::{run.step}",
                        command=command,
                        expression=expression,
                    )
                )
    for command in makefile_commands():
        for expression in marker_expressions(command):
            uses.append(MarkerUse(source=MAKEFILE.name, command=command, expression=expression))
    return uses


MARKER_USES = iter_marker_uses()
REGISTERED_MARKERS = registered_markers()


@pytest.mark.parametrize("use", MARKER_USES, ids=lambda use: f"{use.source}|{use.expression}")
def test_every_marker_identifier_is_registered(use: MarkerUse) -> None:
    """Every name inside a ``-m`` expression must be a declared marker.

    An unregistered name is not an error to pytest -- it evaluates to false. So
    ``-m "not slow and not e2e and not gpu_requried"`` runs the ``gpu_required``
    tests it was written to skip, and nothing anywhere says so. The failure is
    invisible in the log, in the exit code, and in the diff that introduced it.
    """
    unknown = sorted(set(marker_identifiers(use.expression)) - REGISTERED_MARKERS)
    assert not unknown, (
        f"unregistered marker name(s) {unknown} in `-m {use.expression!r}`\n"
        f"  source:  {use.source}\n"
        f"  command: {use.command}\n"
        "pytest silently evaluates an unknown name as false, so this filter does not "
        "do what it reads as. Either fix the spelling or register the marker in "
        "pyproject.toml's [tool.pytest.ini_options] markers list."
    )


class TestTheGuardItself:
    """Meta-guards: a parser that finds nothing passes every assertion above."""

    def test_the_marker_list_parses(self) -> None:
        """Pins the hand-rolled TOML read: names only, no description text."""
        assert len(REGISTERED_MARKERS) >= MIN_REGISTERED_MARKERS, (
            f"only {len(REGISTERED_MARKERS)} markers parsed: {sorted(REGISTERED_MARKERS)}"
        )
        for marker in ("slow", "e2e", "gpu_required", "fem_required"):
            assert marker in REGISTERED_MARKERS, f"{marker} missing from the parsed list"
        for name in REGISTERED_MARKERS:
            assert " " not in name, f"{name!r} looks like a description, not a marker name"
            assert ":" not in name, f"{name!r} still carries its description"

    def test_expressions_are_actually_discovered(self) -> None:
        """Non-vacuity: the parametrised guard must iterate real subjects."""
        assert len(MARKER_USES) >= MIN_MARKER_EXPRESSIONS, (
            f"only {len(MARKER_USES)} `-m` expressions found across "
            f"{WORKFLOW_DIR} and {MAKEFILE.name} -- the extractor is broken, so "
            "the vocabulary check above is passing on an empty list."
        )

    def test_both_sources_contribute(self) -> None:
        """Workflows *and* the Makefile: dropping either halves the guard silently."""
        sources = {use.source for use in MARKER_USES}
        assert any(source.endswith(".yml") or ".yml::" in source for source in sources), (
            "no workflow expression found"
        )
        assert MAKEFILE.name in sources, "no Makefile expression found"

    def test_known_expressions_are_found_verbatim(self) -> None:
        """Anchors the extractor to expressions that demonstrably exist today."""
        expressions = {use.expression for use in MARKER_USES}
        assert "not slow and not e2e and not gpu_required" in expressions
        assert "not gpu_required and not fem_required" in expressions

    def test_no_module_name_is_mistaken_for_an_expression(self) -> None:
        """``python -m pytest`` / ``-m coverage`` must never enter the vocabulary.

        This is the failure mode that would make the guard noisy rather than
        vacuous: every dotted module path in the repo's ``python -m`` commands
        would be reported as an unregistered marker, and the guard would be
        deleted rather than fixed.
        """
        identifiers = {name for use in MARKER_USES for name in marker_identifiers(use.expression)}
        for module in ("pytest", "coverage", "mypy", "pip", "src", "scripts"):
            assert module not in identifiers, (
                f"{module!r} was extracted as a marker name -- a `python -m <module>` "
                "invocation is being read as a marker expression"
            )


class TestExpressionExtraction:
    """Unit-tests ``marker_expressions`` on synthetic commands.

    Driven on literals rather than on the live workflows so the branches the
    current CI configuration never reaches -- an unquoted expression, a quoted
    one in a ``python``-led command -- are exercised at all.
    """

    def test_a_quoted_expression_after_pytest(self) -> None:
        assert marker_expressions('pytest tests/ -m "not slow"') == ["not slow"]

    def test_single_quotes(self) -> None:
        assert marker_expressions("pytest tests/ -m 'e2e and not slow'") == ["e2e and not slow"]

    def test_python_dash_m_module_is_not_an_expression(self) -> None:
        assert marker_expressions("python -m pytest tests/ -q") == []

    def test_python_dash_m_module_then_a_real_expression(self) -> None:
        """The common CI shape: the first ``-m`` is the module, the second the filter."""
        assert marker_expressions('python -m pytest tests/ -m "not gpu_required"') == [
            "not gpu_required"
        ]

    def test_coverage_runner_module_is_not_an_expression(self) -> None:
        """``python -m coverage run ... -m pytest`` carries two module ``-m``s."""
        command = "python -m coverage run --branch --include='*/src/x.py' -m pytest tests/x -q"
        assert marker_expressions(command) == []

    def test_make_variable_runner_module_is_not_an_expression(self) -> None:
        """``$(COV)`` expands to ``python -m coverage``; its ``-m pytest`` is a module."""
        assert marker_expressions("$(COV) run --branch -m pytest tests/x -q") == []

    def test_make_pytest_variable_carries_expressions(self) -> None:
        assert marker_expressions('$(PYTEST) tests/e2e/ -m "not gpu_required" -v') == [
            "not gpu_required"
        ]

    def test_an_unquoted_expression_after_pytest_is_read(self) -> None:
        """``pytest -m e2e`` is legal; the guard must not lose it."""
        assert marker_expressions("pytest tests/e2e/ -m e2e -q") == ["e2e"]

    def test_a_longer_flag_ending_in_m_is_not_matched(self) -> None:
        assert marker_expressions("pytest --tb=short --no-header -q") == []
        assert marker_expressions('pytest --custom-m "x"') == []

    def test_multiple_expressions_in_one_command(self) -> None:
        assert marker_expressions('pytest -m "a" tests/ -m "b"') == ["a", "b"]

    def test_an_empty_command(self) -> None:
        assert marker_expressions("") == []
        assert marker_expressions("   ") == []


class TestTermParsing:
    """Unit-tests the expression grammar helpers on synthetic input."""

    def test_a_simple_conjunction(self) -> None:
        assert marker_identifiers("not slow and not e2e and not gpu_required") == [
            "slow",
            "e2e",
            "gpu_required",
        ]

    def test_a_typo_is_reported_as_its_own_identifier(self) -> None:
        """The case the whole file exists for -- and one that never appears live."""
        assert marker_identifiers("not gpu_requried") == ["gpu_requried"]
        assert "gpu_requried" not in REGISTERED_MARKERS

    def test_nested_parentheses(self) -> None:
        assert marker_identifiers("(e2e or integration) and not (slow)") == [
            "e2e",
            "integration",
            "slow",
        ]

    def test_only_operators_yields_no_identifiers(self) -> None:
        """A degenerate expression must produce nothing, not the operators."""
        assert marker_identifiers("not and or") == []
        assert marker_identifiers("") == []
        assert parse_terms("not not") == []

    def test_operators_are_never_identifiers(self) -> None:
        for operator in BOOLEAN_OPERATORS:
            assert operator not in marker_identifiers(f"{operator} slow")

    def test_identifiers_are_deduped_in_source_order(self) -> None:
        assert marker_identifiers("slow and not slow and e2e") == ["slow", "e2e"]

    def test_polarity_is_recorded_per_term(self) -> None:
        assert parse_terms("not a and b") == [
            MarkerTerm(name="a", negated=True),
            MarkerTerm(name="b", negated=False),
        ]

    def test_not_binds_to_the_following_term_only(self) -> None:
        """``not a and not b`` -- both negated; the ``and`` clears the pending ``not``."""
        assert parse_terms("not a and not b") == [
            MarkerTerm(name="a", negated=True),
            MarkerTerm(name="b", negated=True),
        ]
        assert parse_terms("not a and b or c") == [
            MarkerTerm(name="a", negated=True),
            MarkerTerm(name="b", negated=False),
            MarkerTerm(name="c", negated=False),
        ]

    def test_excludes_and_requires_are_opposites_on_a_present_marker(self) -> None:
        assert expression_excludes("not e2e and not slow", "e2e")
        assert not expression_requires("not e2e and not slow", "e2e")
        assert expression_requires("fem_required and not gpu_required", "fem_required")
        assert not expression_excludes("fem_required and not gpu_required", "fem_required")

    def test_an_absent_marker_is_neither_excluded_nor_required(self) -> None:
        assert not expression_excludes("not slow", "e2e")
        assert not expression_requires("not slow", "e2e")

    def test_plain_selection_rejects_a_narrowing_positive_term(self) -> None:
        """The distinction ``test_e2e_visibility`` depends on.

        ``fem_required and not gpu_required`` mentions no ``e2e`` term and yet
        runs only the optional-extra subset, so it cannot prove the tier runs.
        """
        assert expression_selects_plainly("not gpu_required and not fem_required", "e2e")
        assert expression_selects_plainly("e2e", "e2e")
        assert expression_selects_plainly("", "e2e")
        assert not expression_selects_plainly("fem_required and not gpu_required", "e2e")
        assert not expression_selects_plainly("not slow and not e2e", "e2e")


class TestMakeRecipePrefixes:
    """``strip_recipe_prefixes`` -- Copilot review, PR #144.

    Make allows ``@`` (silence), ``-`` (ignore errors) and ``+`` (run under -n)
    in any combination at the start of a recipe line; none is part of the command
    Make runs. The parsers here removed only the leading tab, so such a line was
    handed on as e.g. ``@$(PYTEST) ... -m "..."``.

    That is not cosmetic for this file. ``_marker_terms`` decides whether an
    *unquoted* ``-m`` is a marker expression by inspecting the program token, so
    a prefixed recipe would fail that check and be skipped -- the vocabulary
    guard would silently cover less than it claims, which is precisely the defect
    class this whole change exists to prevent.
    """

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("@printf 'x'", "printf 'x'"),
            ("-rm -f coverage.xml", "rm -f coverage.xml"),
            ("+$(MAKE) sub", "$(MAKE) sub"),
            ("@-+echo combined", "echo combined"),
            ("pytest -m e2e", "pytest -m e2e"),
            ("", ""),
        ],
    )
    def test_strips_every_prefix_combination(self, line: str, expected: str) -> None:
        """Prefixes are removed; an unprefixed command is untouched."""
        assert strip_recipe_prefixes(line) == expected

    def test_no_parsed_makefile_command_retains_a_prefix(self) -> None:
        """On the real Makefile, no parsed command still starts with a prefix.

        The live inputs are the ``test-substrate`` ``@printf`` and the
        ``gitleaks`` ``@if`` block. Reverting the fix makes both reappear here.
        """
        prefixed = [
            command
            for command in makefile_commands()
            if command[:1] in set(MAKE_RECIPE_PREFIXES) and command
        ]
        assert prefixed == []

    def test_a_prefixed_pytest_recipe_is_still_scanned(self, tmp_path: Path) -> None:
        """A ``@``-prefixed pytest recipe must not vanish from the vocabulary scan.

        The regression this guards: with the prefix left on, the program token is
        ``@pytest``, the unquoted-``-m`` rule rejects it, and a typo'd marker in
        such a recipe would go unreported.
        """
        makefile = tmp_path / "Makefile"
        makefile.write_text('prefixed:\n\t@pytest tests/ -m "not slow"\n', encoding="utf-8")
        commands = makefile_target_recipe("prefixed", makefile)
        assert commands, "the prefixed recipe was not parsed at all"
        identifiers = [
            identifier
            for command in commands
            for expression in marker_expressions(command)
            for identifier in marker_identifiers(expression)
        ]
        assert "slow" in identifiers
