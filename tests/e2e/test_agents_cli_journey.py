"""E2E journeys for ``python -m src.agents.cli`` as a real process.

Guards the CLAUDE.md Regression Surface row *"Agents hardening (lifecycle hooks
+ timeout + scaffold)"* and ``docs/E2E_TEST_PLAN.md`` §6.1. ``tests/agents/
test_scaffold_cli.py`` already drives the scaffold generator in-process; what
this file adds is the process boundary — the exit code as a shell sees it, and
the fact that ``FileExistsError`` becomes a *handled* ``1`` rather than an
unhandled traceback's ``1``-by-accident.

**Device (plan §1, flow (c)):** none of these subcommands takes a device flag
and none places a tensor anywhere. ``src.agents.cli`` does import torch
transitively at startup (the registry import chain pulls in ``src.pde``), so
this is not a torch-free *process* — but it is a device-irrelevant *surface*,
so no device is forwarded and no device assertion is fabricated here.

Every output lands under ``tmp_path``: the scaffold journeys pass ``--root``,
and the read-only journeys write nothing at all.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.e2e.conftest import E2E_TRIVIAL_TIMEOUT_S

if TYPE_CHECKING:
    from tests.e2e.conftest import CLIRunnerType

pytestmark = pytest.mark.e2e

#: The entry point under test.
AGENTS_CLI_MODULE = "src.agents.cli"

#: Exit codes this file asserts, named so no bare integer appears in an
#: assertion. ``2`` is Click/argparse's usage error; ``1`` is the CLI's own
#: handled-error code (``with_error_handling`` -> ``typer.Exit(1)``).
EXIT_OK = 0
EXIT_HANDLED_ERROR = 1
EXIT_USAGE_ERROR = 2

#: The four agents ``src/agents/registry.py::_register_builtin_agents``
#: registers. Asserted as an exact *set*, so both a fifth agent and a missing
#: one fail. See :func:`test_list_agents_names_the_four_builtins` for the
#: ``research`` asymmetry this deliberately does not paper over.
EXPECTED_BUILTIN_AGENTS = frozenset({"coupling", "decomposition", "meta", "solver"})

#: Column header ``print_result_table`` renders for the ``name`` key
#: (``"name".replace("_", " ").title()``). Used as the parser's vacuity proof:
#: if this is absent, no table was parsed and every other assertion is vacuous.
AGENT_TABLE_NAME_HEADER = "Name"

#: Rich's box-drawing vertical rules. The header row is drawn with the *heavy*
#: rule and body rows with the light one, so a parser that knows only one of
#: them silently reads half the table -- which is exactly how the vacuity guard
#: in :func:`test_list_agents_names_the_four_builtins` first failed.
TABLE_CELL_SEPARATORS = ("│", "┃")

#: Name the scaffold journeys generate. Snake case, so
#: ``normalize_agent_name`` is a no-op and the test is about writing, not
#: normalising (which ``tests/agents/test_scaffold_cli.py`` already covers).
SCAFFOLD_AGENT_NAME = "demo_probe"

#: The three files ``scaffold_agent`` plans, relative to ``--root``.
SCAFFOLD_RELATIVE_PATHS: tuple[Path, ...] = (
    Path("specs") / f"{SCAFFOLD_AGENT_NAME}.spec.md",
    Path("src") / "agents" / f"{SCAFFOLD_AGENT_NAME}.py",
    Path("tests") / "agents" / f"test_{SCAFFOLD_AGENT_NAME}.py",
)

#: Prefix ``scaffold`` uses for each planned/created path line.
SCAFFOLD_PATH_LINE_PREFIX = "  - "

_WHITESPACE = re.compile(r"\s+")


def _squash(text: str) -> str:
    """Remove all whitespace from *text*.

    ``rich`` hard-wraps its error panels at the console width, so a path or a
    sentence in an error message can be split across lines mid-token. Comparing
    whitespace-squashed strings makes an assertion about *what was said*
    independent of *where it wrapped* — without weakening it to a substring of
    one fragment.

    Args:
        text: Any captured process output or expected needle.

    Returns:
        *text* with every whitespace character removed.

    """
    return _WHITESPACE.sub("", text)


def _table_first_column(output: str) -> list[str]:
    """Extract the first column of every ``rich`` table row in *output*.

    ``list-agents`` renders a table, and the registry also emits ``structlog``
    ``item_registered`` debug lines that contain the very same agent names. A
    naive ``"solver" in output`` would therefore pass even if the table were
    empty. This reads the rendered table itself.

    Wrapped description cells produce continuation rows whose first cell is
    blank; those are dropped.

    Args:
        output: Combined stdout+stderr of the CLI process.

    Returns:
        First-column cell values in row order, including the header label.

    """
    values: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        separator = next(
            (rule for rule in TABLE_CELL_SEPARATORS if stripped.startswith(rule)), None
        )
        if separator is None:
            continue
        first = stripped.strip(separator).split(separator)[0].strip()
        if first:
            values.append(first)
    return values


def _printed_scaffold_paths(output: str) -> set[Path]:
    """Parse the ``  - <path>`` lines the scaffold command prints.

    Args:
        output: Combined stdout+stderr of the CLI process.

    Returns:
        The printed paths, as a set.

    """
    return {
        Path(line[len(SCAFFOLD_PATH_LINE_PREFIX) :].strip())
        for line in output.splitlines()
        if line.startswith(SCAFFOLD_PATH_LINE_PREFIX)
    }


def test_scaffold_dry_run_writes_nothing(cli_runner: CLIRunnerType, tmp_path: Path) -> None:
    """``scaffold --dry-run`` plans three files and creates none of them.

    Guards the CLAUDE.md row *"Agents hardening (lifecycle hooks + timeout +
    scaffold)"* — specifically its "dry-run writes nothing" clause — across a
    process boundary, so a future change that writes eagerly and *then* reports
    the plan is caught by the empty-root assertion rather than by a reviewer.

    Device-irrelevant surface (plan §1, flow (c)); no device is forwarded.
    """
    root = tmp_path / "scaffold_root"
    root.mkdir()

    result = cli_runner(
        AGENTS_CLI_MODULE,
        ["scaffold", SCAFFOLD_AGENT_NAME, "--root", str(root), "--dry-run"],
        E2E_TRIVIAL_TIMEOUT_S,
        None,
    )

    assert result.returncode == EXIT_OK, result.output
    assert _printed_scaffold_paths(result.output) == {
        root / relative for relative in SCAFFOLD_RELATIVE_PATHS
    }
    assert list(root.rglob("*")) == [], "--dry-run must not create anything under --root"


def test_scaffold_then_rerun_refuses_with_exit_one(
    cli_runner: CLIRunnerType, tmp_path: Path
) -> None:
    """A second ``scaffold`` over the same root exits 1 and names the file.

    Guards the CLAUDE.md row *"Agents hardening (lifecycle hooks + timeout +
    scaffold)"* — the "overwrite-refusal" clause. ``src/agents/scaffold.py``
    raises ``FileExistsError``, which ``with_error_handling`` converts to
    ``typer.Exit(1)``; this asserts the shell-visible half of that contract,
    which the in-process ``tests/agents/test_scaffold_cli.py`` cannot see.

    The first run's assertions are load-bearing, not setup: without them a
    scaffold that silently wrote nothing would make the second run's refusal
    unreachable and the test vacuous.

    Device-irrelevant surface (plan §1, flow (c)); no device is forwarded.
    """
    root = tmp_path / "scaffold_root"
    root.mkdir()
    argv = ["scaffold", SCAFFOLD_AGENT_NAME, "--root", str(root)]

    first = cli_runner(AGENTS_CLI_MODULE, argv, E2E_TRIVIAL_TIMEOUT_S, None)
    assert first.returncode == EXIT_OK, first.output
    created = {path for path in root.rglob("*") if path.is_file()}
    assert created == {root / relative for relative in SCAFFOLD_RELATIVE_PATHS}
    contents_before = {path: path.read_bytes() for path in sorted(created)}

    second = cli_runner(AGENTS_CLI_MODULE, argv, E2E_TRIVIAL_TIMEOUT_S, None)

    assert second.returncode == EXIT_HANDLED_ERROR, second.output
    squashed = _squash(second.output)
    assert _squash("Refusing to overwrite existing file") in squashed
    assert any(_squash(str(path)) in squashed for path in sorted(created)), (
        "the refusal must name the path it refused to overwrite"
    )
    assert {path: path.read_bytes() for path in sorted(created)} == contents_before, (
        "a refused scaffold must leave every existing file byte-identical"
    )


def test_list_agents_names_the_four_builtins(cli_runner: CLIRunnerType) -> None:
    """``list-agents`` renders exactly the four registered builtins.

    Guards the CLAUDE.md row *"Agents hardening (lifecycle hooks + timeout +
    scaffold)"*: the registry's public inventory, read through the CLI rather
    than through ``AgentRegistry`` in-process (the registry is a process-global
    singleton that several suites ``clear()``; a subprocess is the only way to
    observe its shipped contents — plan §2 rule 10).

    **Recorded asymmetry, not papered over:** ``AgentType.RESEARCH`` exists and
    ``research`` is a *subcommand* of this very CLI, but
    ``src/agents/registry.py::_register_builtin_agents`` registers no agent
    under that name, so ``list-agents`` does not and must not show it. The owner
    may want to close that gap; this test asserts what the code does, and its
    exact-set comparison is what would flag the day it changes.

    Device-irrelevant surface (plan §1, flow (c)); no device is forwarded.
    """
    result = cli_runner(AGENTS_CLI_MODULE, ["list-agents"], E2E_TRIVIAL_TIMEOUT_S, None)

    assert result.returncode == EXIT_OK, result.output
    column = _table_first_column(result.output)
    assert AGENT_TABLE_NAME_HEADER in column, (
        "no rendered table was parsed -- every assertion below would be vacuous"
    )
    assert set(column) - {AGENT_TABLE_NAME_HEADER} == set(EXPECTED_BUILTIN_AGENTS)


def test_research_subcommand_rejects_a_missing_config_with_exit_two(
    cli_runner: CLIRunnerType, tmp_path: Path
) -> None:
    """``research --config <absent>`` is a usage error: exit 2, before any run.

    Guards the CLAUDE.md row *"Centaur research-loop harness (mocked CPU)"* at
    its CLI boundary. The ``--config`` option declares ``exists=True``, so Click
    rejects the path itself and the code is ``2`` (usage), not the ``1`` that
    ``load_config_file`` would produce for a *present* but unparsable file — a
    distinction a set-valued assertion would erase (plan §2 rule 2).

    The complementary positive fact, measured on this branch: ``research
    --help`` exits 0, so the subcommand is wired; this test is about its
    argument contract, not its existence.

    Device-irrelevant at this depth: the process exits before a device is
    resolved.
    """
    absent = tmp_path / "absent.yaml"
    assert not absent.exists()

    result = cli_runner(
        AGENTS_CLI_MODULE,
        ["research", "--config", str(absent)],
        E2E_TRIVIAL_TIMEOUT_S,
        None,
    )

    assert result.returncode == EXIT_USAGE_ERROR, result.output
    assert _squash("does not exist") in _squash(result.output)
