"""B8: CLAUDE.md Regression Surface coverage-gate rows ⊆ CI.

A documented ``--cov-fail-under`` / ``coverage report --fail-under`` that CI
does not run is the same invisibility class as an omit-collision gate: the
docs look enforced and nothing fails. The charter table already has this
check (``test_documented_gates_are_enforced_in_ci``). This file covers the
ungated copy — CLAUDE.md's Regression Surface.

Naive ⊆ of the *whole* table fails: GPU-smoke and mechanism rows have no
coverage threshold. Filter to coverage-gate commands only.

Mutation kill (2026-09-10): inserting

    | Planted B8 | `pytest tests/nope --cov=src/planted_b8 --cov-fail-under=99` | planted |

into the Regression Surface made ``test_coverage_gate_rows_are_enforced_in_ci``
fail (FAILED). Restored. ``test_a_planted_gate_is_not_enforced`` keeps that
kill durable against a synthetic command so the live file stays green.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tests.support.workflows import CI_WORKFLOW, iter_run_scripts

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

_SURFACE_START = "## Regression Surface"
_SURFACE_END = "## Verification Commands"

_COV = re.compile(r"--cov=(?P<target>[^\s\\'\"]+)")
_COV_FAIL = re.compile(r"--cov-fail-under=(?P<n>\d+)")
_FAIL_UNDER = re.compile(r"--fail-under=(?P<n>\d+)")
_INCLUDE = re.compile(r"--include=(?P<q>[\"'])(?P<body>.*?)(?P=q)")
_TITLE_PACKAGES = re.compile(
    r"Per-module coverage \((?P<body>[^)]+)\)",
)

#: Floor on coverage-gate rows. A parser that matches nothing would pass the
#: ⊆ check vacuously.
MIN_COVERAGE_GATE_ROWS = 20

#: Floor on *non*-gate rows. Proves the filter is not "every table row".
MIN_NON_GATE_ROWS = 20


@dataclass(frozen=True)
class SurfaceRow:
    """One Regression Surface table row."""

    title: str
    command: str


@dataclass(frozen=True)
class CovGate:
    """One coverage threshold the row claims CI enforces."""

    kind: str  # "pytest_cov" | "native_include"
    target: str
    threshold: str


def _regression_surface_text(source: str | None = None) -> str:
    text = CLAUDE_MD.read_text(encoding="utf-8") if source is None else source
    assert _SURFACE_START in text, "CLAUDE.md is missing ## Regression Surface"
    assert _SURFACE_END in text, "CLAUDE.md is missing ## Verification Commands"
    return text.split(_SURFACE_START, 1)[1].split(_SURFACE_END, 1)[0]


def parse_surface_rows(source: str | None = None) -> list[SurfaceRow]:
    """Parse Surface / Command cells from the Regression Surface table."""
    rows: list[SurfaceRow] = []
    for line in _regression_surface_text(source).splitlines():
        if not line.startswith("|"):
            continue
        stripped = line.strip()
        inner = stripped[1:-1] if stripped.endswith("|") else stripped[1:]
        cells = [c.strip() for c in inner.split("|")]
        if len(cells) < 2:
            continue
        title, command = cells[0], cells[1].strip("`")
        if title in {"Surface", ""} or set(title) <= {"-", ":"}:
            continue
        rows.append(SurfaceRow(title=title, command=command))
    return rows


def is_coverage_gate_command(command: str) -> bool:
    """True iff the Command cell documents a coverage *threshold*."""
    if "--cov-fail-under=" in command:
        return True
    return "coverage report" in command and "--fail-under=" in command


def _packages_from_title(title: str) -> list[str]:
    match = _TITLE_PACKAGES.search(title)
    if match is None:
        return []
    body = match.group("body")
    if "/" not in body:
        return []
    names: list[str] = []
    for part in body.split("/"):
        token = part.strip().split()[0] if part.strip() else ""
        token = token.strip("`").rstrip(",.")
        if token.isidentifier() and token not in {"per", "module", "coverage"}:
            names.append(token)
    return names


def coverage_gates_from_row(row: SurfaceRow) -> list[CovGate]:
    """Extract (target, threshold) pairs the row claims CI runs."""
    command = row.command
    targets = [m.group("target") for m in _COV.finditer(command)]
    cov_fails = [m.group("n") for m in _COV_FAIL.finditer(command)]
    native_fails = [m.group("n") for m in _FAIL_UNDER.finditer(command)]
    includes = [m.group("body") for m in _INCLUDE.finditer(command)]

    expanded: list[str] = []
    for target in targets:
        if target in {"src/<pkg>", "<pkg>", "src/<pkg>/"}:
            pkgs = _packages_from_title(row.title)
            expanded.extend(f"src/{name}" for name in pkgs)
        else:
            expanded.append(target)

    gates: list[CovGate] = []
    if expanded and cov_fails:
        if len(expanded) == len(cov_fails):
            paired = list(zip(expanded, cov_fails, strict=True))
        elif len(cov_fails) == 1:
            paired = [(target, cov_fails[0]) for target in expanded]
        else:
            n = min(len(expanded), len(cov_fails))
            paired = list(zip(expanded[:n], cov_fails[:n], strict=True))
        for target, threshold in paired:
            gates.append(CovGate("pytest_cov", target, threshold))

    if includes and native_fails:
        # Run+report duplicate the same --include; unique fragments.
        fragments: list[str] = []
        seen: set[str] = set()
        for blob in includes:
            for part in blob.split(","):
                piece = part.strip()
                if piece and piece not in seen:
                    seen.add(piece)
                    fragments.append(piece)
        threshold = native_fails[-1]
        for fragment in fragments:
            gates.append(CovGate("native_include", fragment, threshold))
    return gates


def _step_enforces(script: str, gate: CovGate) -> bool:
    if f"--cov-fail-under={gate.threshold}" not in script and (
        f"--fail-under={gate.threshold}" not in script
    ):
        return False
    if gate.kind == "pytest_cov":
        pattern = rf"--cov={re.escape(gate.target)}(?:\s|$)"
        return re.search(pattern, script) is not None
    return gate.target in script


def gate_is_in_ci(gate: CovGate, scripts: list[str] | None = None) -> bool:
    bodies = scripts if scripts is not None else [item.script for item in iter_run_scripts()]
    return any(_step_enforces(script, gate) for script in bodies)


def test_regression_surface_section_is_parseable() -> None:
    """Vacuity: a missing table would make every ⊆ assertion iterate nothing."""
    rows = parse_surface_rows()
    assert len(rows) >= MIN_COVERAGE_GATE_ROWS + MIN_NON_GATE_ROWS


def test_coverage_gate_filter_excludes_mechanism_rows() -> None:
    """Naive whole-table ⊆ CI is the false-positive the plan forbids.

    GPU-smoke / mechanism rows have no ``--cov-fail-under`` / native
    ``--fail-under``. The filter must drop them so this guard can exist.
    """
    rows = parse_surface_rows()
    gates = [row for row in rows if is_coverage_gate_command(row.command)]
    nongates = [row for row in rows if not is_coverage_gate_command(row.command)]
    assert len(gates) >= MIN_COVERAGE_GATE_ROWS
    assert len(nongates) >= MIN_NON_GATE_ROWS
    smoke = [row for row in nongates if "gpu" in row.title.lower() or "smoke" in row.title.lower()]
    assert smoke, "expected at least one GPU-smoke / smoke row to prove the filter"


def test_coverage_gate_rows_are_enforced_in_ci() -> None:
    """Every coverage-gate Command cell must appear as a real CI threshold.

    Mutation kill: a planted ``--cov=src/planted_b8 --cov-fail-under=99`` row
    in CLAUDE.md's Regression Surface made this test fail. Restored 2026-09-10.
    """
    scripts = [item.script for item in iter_run_scripts() if item.workflow == CI_WORKFLOW.name]
    assert scripts, "ci.yml has no run: scripts — the scan would be vacuous"
    failures: list[str] = []
    gate_rows = [row for row in parse_surface_rows() if is_coverage_gate_command(row.command)]
    assert gate_rows, "no coverage-gate rows parsed"
    for row in gate_rows:
        extracted = coverage_gates_from_row(row)
        if not extracted:
            failures.append(f"{row.title}: coverage-gate command produced no requirements")
            continue
        for gate in extracted:
            if not gate_is_in_ci(gate, scripts):
                failures.append(
                    f"{row.title}: {gate.kind} {gate.target!r} @ {gate.threshold} "
                    "not found in any ci.yml step"
                )
    assert not failures, "CLAUDE.md coverage-gate rows missing from CI:\n  " + "\n  ".join(failures)


def test_a_planted_gate_is_not_enforced() -> None:
    """Durable mutation: a threshold CI does not run must fail the matcher.

    The live-file plant (see module docstring) is restored; this assertion
    is what keeps the kill from rotting.
    """
    planted = CovGate("pytest_cov", "src/planted_b8", "99")
    assert not gate_is_in_ci(planted)


def test_placeholder_pkg_expands_from_title() -> None:
    row = SurfaceRow(
        title="Per-module coverage (analysis / tournament / prototyping)",
        command="pytest tests/<pkg>/ --cov=src/<pkg> --cov-fail-under=85",
    )
    gates = coverage_gates_from_row(row)
    targets = {gate.target for gate in gates}
    assert targets == {"src/analysis", "src/tournament", "src/prototyping"}
    assert {gate.threshold for gate in gates} == {"85"}


def test_is_coverage_gate_predicate() -> None:
    assert is_coverage_gate_command("pytest --cov=src/foo --cov-fail-under=85")
    assert is_coverage_gate_command(
        "python -m coverage report --include='*/src/foo.py' --fail-under=85"
    )
    assert not is_coverage_gate_command("pytest tests/foo -v")
    assert not is_coverage_gate_command("pytest tests/foo -m gpu_required")


def test_ci_yml_is_the_workflow_under_test() -> None:
    """Without this, iter_run_scripts could match another workflow and pass."""
    assert CI_WORKFLOW.name == "ci.yml"
    assert CI_WORKFLOW.is_file()
