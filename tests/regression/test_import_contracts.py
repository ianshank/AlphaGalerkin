"""Architectural import contracts.

Layering is the kind of property that is true on the day it is written and
false six months later, because nothing ever announces the moment it breaks --
a single `from src.pde import ...` in the wrong file is a one-line diff that
reads as convenience. `scripts/audit_abstractions.py` already guards the
*vertical* direction (an abstraction with no call site); this guards the
*horizontal* one (a layer reaching into a layer it is defined not to know
about).

Three of these contracts are scientifically load-bearing rather than stylistic:

* `src/refinement/` is the **domain-free** refinement layer. The moment it
  imports `src/pde/`, it is a PDE layer with an abstract veneer, and the claim
  that a refinement game is reusable across domains is unfalsifiable.
* `src/mcts/` is the candidate search engine. A reference baseline that imports
  it does not share an interface with the thing under test -- it shares an
  implementation, and any comparison between them measures less than it claims.
* `src/templates/` and `src/math_kernel/` are the reusable substrate. A domain
  import there inverts the dependency and makes them un-reusable by definition.

Every contract carries a `reason`, for the same purpose the charter's
deviations register does: an unexplained rule gets deleted the first time it is
inconvenient. Every contract is also asserted to be **non-vacuous** -- a
renamed package would otherwise turn a rule into a rule about nothing, and it
would still pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from tests.support.import_graph import (
    imported_modules,
    matches_module_prefix,
    python_files_under,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ImportContract:
    """One layering rule, its scope, and why it exists."""

    name: str
    #: Repo-relative directories whose files are checked.
    scope: tuple[str, ...]
    #: Module prefixes those files may not import.
    forbidden: tuple[str, ...]
    #: Why the rule exists. Never empty -- see the module docstring.
    reason: str
    #: Files inside `scope` this rule does not apply to, each with a reason.
    #: Kept as a mapping so an exemption cannot be added without explaining it.
    exemptions: tuple[tuple[str, str], ...] = ()


CONTRACTS: Final[tuple[ImportContract, ...]] = (
    ImportContract(
        name="refinement-is-domain-free",
        scope=("src/refinement",),
        forbidden=("src.pde", "src.games", "src.research"),
        reason=(
            "src/refinement/ is the domain-free refinement layer: RefinementGame, "
            "RefinementState and the adapter are defined to know nothing about PDEs "
            "or board games. A single import the other way makes 'reusable across "
            "domains' an untestable claim rather than a property."
        ),
    ),
    ImportContract(
        name="search-engine-does-not-know-its-domains",
        scope=("src/mcts",),
        forbidden=("src.pde", "src.refinement", "src.research", "src.poc", "src.games"),
        reason=(
            "The search engine talks to domains through the GameInterface and "
            "Evaluator protocols. Importing a concrete domain package would let a "
            "domain-specific fix land inside the engine, where it silently changes "
            "every other domain's behaviour. src.games is included in the forbidden "
            "list rather than quietly omitted, so the one module that does reach for "
            "it has to say so below."
        ),
        exemptions=(
            (
                "src/mcts/gumbel.py",
                "The only module under src/mcts/ that imports src.games, and it does "
                "so for src.games.interface.GameInterface and src.games.state.GameState "
                "-- a protocol and a type, not behaviour. Recorded as a real exemption "
                "rather than dropped from the rule, so the deviation is visible and a "
                "second module cannot join it silently. Moving those two names into a "
                "protocols-only module would retire this; that is a refactor, not a "
                "guard change.",
            ),
        ),
    ),
    ImportContract(
        name="reference-baselines-do-not-import-the-candidate",
        scope=("src/research/baselines.py", "src/research/fem_baseline.py"),
        forbidden=("src.mcts", "src.refinement"),
        reason=(
            "These are the classical reference implementations an experiment "
            "compares against. If a baseline imports the search engine under test, "
            "the two arms share an implementation rather than an interface, and a "
            "defect in the shared code moves both arms in the same direction -- "
            "which is invisible in a ratio. The harness that *drives* both arms "
            "(src/research/lshape_amr_compare.py) must import both and is "
            "deliberately outside this contract's scope."
        ),
    ),
    ImportContract(
        name="reusable-substrate-has-no-domain-dependencies",
        scope=("src/templates", "src/math_kernel"),
        forbidden=(
            "src.pde",
            "src.mcts",
            "src.games",
            "src.refinement",
            "src.research",
            "src.training",
            "src.poc",
        ),
        reason=(
            "src/templates/ (config, registry, logging, CLI) and src/math_kernel/ "
            "(basis functions, quadrature, spectral ops) are the layer everything "
            "else is built on. A domain import inverts that dependency and makes "
            "them un-reusable by construction -- and creates an import cycle the "
            "first time the domain imports back."
        ),
    ),
)


def _scope_files(contract: ImportContract) -> list[Path]:
    files: list[Path] = []
    for entry in contract.scope:
        target = REPO_ROOT / entry
        if target.is_dir():
            files.extend(python_files_under(target))
        elif target.is_file():
            files.append(target)
    exempt = {REPO_ROOT / path for path, _ in contract.exemptions}
    return [f for f in files if f not in exempt]


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.name)
def test_contract_is_upheld(contract: ImportContract) -> None:
    violations: list[str] = []
    for path in _scope_files(contract):
        for module in sorted(imported_modules(path, REPO_ROOT)):
            for prefix in contract.forbidden:
                if matches_module_prefix(module, prefix):
                    violations.append(
                        f"  {path.relative_to(REPO_ROOT)} imports {module} (forbidden: {prefix})"
                    )
    assert not violations, (
        f"import contract {contract.name!r} broken:\n"
        + "\n".join(violations)
        + f"\n\nWhy this rule exists: {contract.reason}"
    )


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.name)
def test_contract_is_not_vacuous(contract: ImportContract) -> None:
    """A rule about a package that no longer exists still passes.

    That is the failure mode this test exists for: rename `src/refinement/` and
    every contract above goes green while guarding nothing at all.
    """
    files = _scope_files(contract)
    assert files, (
        f"contract {contract.name!r} scans no files -- its scope {contract.scope} "
        "does not resolve. A renamed or deleted package turns this rule into a "
        "rule about nothing, which still passes."
    )


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.name)
def test_contract_states_a_reason(contract: ImportContract) -> None:
    """Every rule states why it exists.

    Same principle as the charter's deviations register: an unexplained rule is
    deleted the first time it is inconvenient.
    """
    assert contract.reason.strip(), f"contract {contract.name!r} has no reason"
    assert len(contract.reason) > 80, (
        f"contract {contract.name!r} has a reason too short to be one: {contract.reason!r}"
    )


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.name)
def test_every_exemption_names_a_real_file_and_a_reason(contract: ImportContract) -> None:
    """A stale exemption silently shrinks a contract's scope."""
    for path, reason in contract.exemptions:
        assert (REPO_ROOT / path).exists(), (
            f"contract {contract.name!r} exempts {path}, which does not exist -- "
            "the exemption is stale and is now hiding nothing while looking like "
            "it hides something"
        )
        assert len(reason) > 40, f"exemption {path} in {contract.name!r} has no real reason"


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.name)
def test_every_exemption_is_still_needed(contract: ImportContract) -> None:
    """The exemption must be doing work.

    If the exempted file no longer breaks the contract, the exemption should be
    removed rather than left as a permanent hole. The same discipline
    `tests/claude/`'s forward-reference check uses.
    """
    for path, _ in contract.exemptions:
        target = REPO_ROOT / path
        breaks = any(
            matches_module_prefix(module, prefix)
            for module in imported_modules(target, REPO_ROOT)
            for prefix in contract.forbidden
        )
        assert breaks, (
            f"contract {contract.name!r} exempts {path}, but that file no longer "
            "imports anything the contract forbids -- delete the exemption"
        )


def test_the_exemption_mechanism_actually_excludes_a_file(tmp_path: Path) -> None:
    """Proves the mechanism, not just today's use of it.

    `_scope_files` filtering by exemption is the only thing standing between a
    documented deviation and a red build; if the path comparison silently never
    matched, every exemption would be inert and the contracts would be stricter
    than they claim rather than looser -- a failure that shows up as an
    unexplained failure months later.
    """
    package = tmp_path / "src" / "refinement"
    package.mkdir(parents=True)
    (package / "clean.py").write_text("import numpy\n", encoding="utf-8")
    (package / "exempt.py").write_text("from src.pde import operators\n", encoding="utf-8")

    contract = ImportContract(
        name="synthetic",
        scope=("src/refinement",),
        forbidden=("src.pde",),
        reason="x" * 100,
        exemptions=(("src/refinement/exempt.py", "y" * 50),),
    )

    global REPO_ROOT
    original = REPO_ROOT
    try:
        REPO_ROOT = tmp_path
        scanned = {p.name for p in _scope_files(contract)}
    finally:
        REPO_ROOT = original

    assert scanned == {"clean.py"}, f"exemption did not exclude the file: {scanned}"


def test_contract_names_are_unique() -> None:
    names = [c.name for c in CONTRACTS]
    assert len(set(names)) == len(names), f"duplicate contract names: {names}"


def test_the_scanner_detects_a_planted_violation(tmp_path: Path) -> None:
    """The guard's own guard.

    Every assertion above is of the form "nothing matched". A scanner that
    silently matched nothing -- a broken glob, an AST walk that missed
    `ImportFrom` -- would pass all of them. This plants a violation and requires
    it to be found.
    """
    package = tmp_path / "src" / "refinement"
    package.mkdir(parents=True)
    offender = package / "leaky.py"
    offender.write_text("from src.pde.operators import PoissonOperator\n", encoding="utf-8")

    found = [
        module
        for module in imported_modules(offender, tmp_path)
        if matches_module_prefix(module, "src.pde")
    ]
    assert found == ["src.pde.operators"]


def test_the_scanner_resolves_relative_imports(tmp_path: Path) -> None:
    """`from ..pde import x` must be caught as `src.pde`, not skipped as relative."""
    package = tmp_path / "src" / "refinement"
    package.mkdir(parents=True)
    offender = package / "sneaky.py"
    offender.write_text("from ..pde import operators\n", encoding="utf-8")

    modules = imported_modules(offender, tmp_path)
    assert any(matches_module_prefix(m, "src.pde") for m in modules), modules
