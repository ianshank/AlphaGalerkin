"""Tests for the abstract-method audit tool."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.audit_abstractions as audit_module
from scripts.audit_abstractions import audit, main

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The roots CI's blocking `--fail-on-missing` step scans
#: (`.github/workflows/ci.yml`, "Audit abstractions (refinement surfaces)").
#: Kept here so the staleness guard below scans exactly what CI does -- a
#: narrower scan would report a cross-package reader as missing and make a
#: live exemption look correctly staged.
AUDITED_ROOTS = ("src/mcts", "src/refinement", "src/pde", "src/research")


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_flags_uncalled_abstract_method(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "mod.py",
        (
            "from abc import ABC, abstractmethod\n"
            "class Base(ABC):\n"
            "    @abstractmethod\n"
            "    def get_reward(self) -> float: ...\n"
            "class Impl(Base):\n"
            "    def get_reward(self) -> float:\n"
            "        return 1.0\n"
        ),
    )
    report = audit([tmp_path])
    names = {f.name for f in report.abstract_missing}
    assert "get_reward" in names


def test_called_abstract_method_not_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "mod.py",
        (
            "from abc import ABC, abstractmethod\n"
            "class Base(ABC):\n"
            "    @abstractmethod\n"
            "    def step(self) -> None: ...\n"
            "def run(b: Base) -> None:\n"
            "    b.step()\n"
        ),
    )
    report = audit([tmp_path])
    assert not report.abstract_missing


def test_dedup_reports_each_declaration_of_same_name(tmp_path: Path) -> None:
    """Two dead abstract methods with the same name are both reported.

    Name-only de-dup would drop the second declaration; the fully-qualified
    (file, class, name) key reports each.
    """
    _write(
        tmp_path,
        "a.py",
        (
            "from abc import ABC, abstractmethod\n"
            "class A(ABC):\n"
            "    @abstractmethod\n"
            "    def foo(self) -> None: ...\n"
        ),
    )
    _write(
        tmp_path,
        "b.py",
        (
            "from abc import ABC, abstractmethod\n"
            "class B(ABC):\n"
            "    @abstractmethod\n"
            "    def foo(self) -> None: ...\n"
        ),
    )
    report = audit([tmp_path])
    classes = {(f.cls, f.name) for f in report.abstract_missing}
    assert ("A", "foo") in classes
    assert ("B", "foo") in classes


def test_abstract_property_read_not_flagged(tmp_path: Path) -> None:
    """A property read as an attribute (no parens) must not be flagged."""
    _write(
        tmp_path,
        "mod.py",
        (
            "from abc import ABC, abstractmethod\n"
            "class Base(ABC):\n"
            "    @property\n"
            "    @abstractmethod\n"
            "    def size(self) -> int: ...\n"
            "def run(b: Base) -> int:\n"
            "    return b.size\n"
        ),
    )
    report = audit([tmp_path])
    assert not report.abstract_missing


def test_protocol_member_without_reader_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "mod.py",
        (
            "from typing import Protocol\n"
            "class Iface(Protocol):\n"
            "    def n_players(self) -> int: ...\n"
        ),
    )
    report = audit([tmp_path])
    assert {f.name for f in report.protocol_missing} == {"n_players"}


def test_dunder_never_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "mod.py",
        (
            "from abc import ABC, abstractmethod\n"
            "class Base(ABC):\n"
            "    @abstractmethod\n"
            "    def __len__(self) -> int: ...\n"
        ),
    )
    report = audit([tmp_path])
    assert not report.abstract_missing


def test_main_exit_codes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "mod.py",
        (
            "from abc import ABC, abstractmethod\n"
            "class Base(ABC):\n"
            "    @abstractmethod\n"
            "    def dead(self) -> None: ...\n"
        ),
    )
    # Report mode: exit 0 even with a finding.
    assert main([str(tmp_path)]) == 0
    # Blocking mode: exit 1 on a finding.
    assert main([str(tmp_path), "--fail-on-missing"]) == 1


def test_mcts_and_pde_surfaces_clean() -> None:
    """The F0/F1 fixes must keep the refinement surfaces audit-clean."""
    assert not audit([Path("src/mcts")]).total
    # get_reward now has a call site in the adapter (F1 resolved).
    pde_report = audit([Path("src/pde")])
    assert "get_reward" not in {f.name for f in pde_report.abstract_missing}


def test_known_live_allowlist_suppresses_verified_public_apis(tmp_path: Path) -> None:
    """Verified-live public APIs on their real classes are not flagged.

    ``BaseEngine.is_ready`` / ``GameInterface.get_symmetries`` /
    ``GameInterface.get_action_mask`` are exercised by tests or reached through
    dynamic receivers the pure-AST heuristic cannot resolve, so they are
    allowlisted by ``(class, name)`` to keep the tool's signal trustworthy.
    """
    _write(
        tmp_path,
        "mod.py",
        (
            "from abc import ABC, abstractmethod\n"
            "class GameInterface(ABC):\n"
            "    @abstractmethod\n"
            "    def get_symmetries(self) -> None: ...\n"
            "    @abstractmethod\n"
            "    def get_action_mask(self) -> None: ...\n"
            "class BaseEngine(ABC):\n"
            "    @abstractmethod\n"
            "    def is_ready(self) -> bool: ...\n"
        ),
    )
    assert not audit([tmp_path]).abstract_missing


def test_allowlist_is_class_scoped_not_name_only(tmp_path: Path) -> None:
    """A same-named abstract method on a different class is still flagged."""
    _write(
        tmp_path,
        "mod.py",
        (
            "from abc import ABC, abstractmethod\n"
            "class Unrelated(ABC):\n"
            "    @abstractmethod\n"
            "    def is_ready(self) -> bool: ...\n"
        ),
    )
    report = audit([tmp_path])
    assert ("Unrelated", "is_ready") in {(f.cls, f.name) for f in report.abstract_missing}


def test_generic_protocol_member_without_reader_flagged(tmp_path: Path) -> None:
    """A *generic* ``Protocol[T]`` base must be recognized, not silently skipped.

    ``Protocol[T]`` parses as ``ast.Subscript``, not ``ast.Name``/``ast.Attribute``
    — before the fix, ``_is_protocol_class`` returned False for any generic
    Protocol, so its members were never checked for call sites at all (a
    generic Protocol was entirely invisible to this audit, not "verified live").
    """
    _write(
        tmp_path,
        "mod.py",
        (
            "from typing import Protocol, TypeVar\n"
            "T = TypeVar('T')\n"
            "class Iface(Protocol[T]):\n"
            "    def solve(self, x: T) -> T: ...\n"
        ),
    )
    report = audit([tmp_path])
    assert {f.name for f in report.protocol_missing} == {"solve"}


def test_generic_protocol_member_with_reader_not_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "mod.py",
        (
            "from typing import Protocol, TypeVar\n"
            "T = TypeVar('T')\n"
            "class Iface(Protocol[T]):\n"
            "    def solve(self, x: T) -> T: ...\n"
            "def run(i: Iface) -> None:\n"
            "    i.solve(1)\n"
        ),
    )
    report = audit([tmp_path])
    assert not report.protocol_missing


def test_staged_allowlist_suppresses_declared_but_not_yet_consumed_members(
    tmp_path: Path,
) -> None:
    """``_STAGED_FOR_UPCOMING_TASK`` exempts a genuinely-uncalled member.

    Distinct from ``_KNOWN_LIVE`` (a real caller the AST heuristic cannot see):
    ``RefinementSubstrate.fingerprint`` has no caller anywhere yet, by design,
    pending element-local-substrate's Slice E (task 7.1), which adds the
    fingerprint-keyed solve cache that reads it.
    """
    _write(
        tmp_path,
        "mod.py",
        (
            "from typing import Protocol, TypeVar\n"
            "T = TypeVar('T')\n"
            "class RefinementSubstrate(Protocol[T]):\n"
            "    def fingerprint(self, x: T) -> bytes: ...\n"
        ),
    )
    assert not audit([tmp_path]).protocol_missing


def test_staged_allowlist_does_not_cover_members_that_gained_a_reader(
    tmp_path: Path,
) -> None:
    """A member that became live must LEAVE the allowlist, not stay exempted.

    Slice D shrank ``_STAGED_FOR_UPCOMING_TASK`` from all 8
    ``RefinementSubstrate`` members to just ``fingerprint``, because
    ``src/research/substrates/sweep.py`` became a real reader of the other
    seven. This pins that shrink: an allowlist entry covering a live member
    silently stops guarding it, which is the exact failure mode this script
    exists to prevent. ``solve`` is the representative — it is the member the
    previous version of this test used, so a careless re-widening of the
    allowlist would fail here rather than pass quietly.
    """
    _write(
        tmp_path,
        "mod.py",
        (
            "from typing import Protocol, TypeVar\n"
            "T = TypeVar('T')\n"
            "class RefinementSubstrate(Protocol[T]):\n"
            "    def solve(self, x: T) -> T: ...\n"
        ),
    )
    missing = {f.name for f in audit([tmp_path]).protocol_missing}
    assert "solve" in missing


@pytest.mark.parametrize("staged", sorted(audit_module._STAGED_FOR_UPCOMING_TASK))
def test_every_staged_exemption_is_still_forward(
    staged: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged exemption must still be *needed*, not merely present.

    ``_STAGED_FOR_UPCOMING_TASK`` means "declared ahead of its first consumer".
    The moment that consumer lands, the entry stops exempting a dead member and
    starts *hiding a live one* — silently narrowing CI's ``--fail-on-missing``
    gate by exactly the F0/F1 defect class the gate exists to catch. Asserting
    only that the exemption works (the test above) cannot detect that; nothing
    would go red, the entry would just rot.

    So: drop the allowlist entirely, re-run the audit over the same roots CI
    scans, and require every staged member to still show up as unread. This
    goes red the day a real caller lands, forcing the entry's deletion rather
    than letting it linger. Same discipline as
    ``tests/regression/test_import_contracts.py``'s "exemptions must still be
    needed" meta-guard and ``.claude``'s ``FORWARD_REFERENCES``, which CLAUDE.md
    describes as "asserted to still be forward, so a stale exemption fails
    rather than rotting".
    """
    monkeypatch.setattr(audit_module, "_STAGED_FOR_UPCOMING_TASK", frozenset())
    report = audit([REPO_ROOT / root for root in AUDITED_ROOTS])
    still_dead = {(f.cls, f.name) for f in report.protocol_missing} | {
        (f.cls, f.name) for f in report.abstract_missing
    }
    cls, name = staged
    assert staged in still_dead, (
        f"{cls}.{name} is exempted by _STAGED_FOR_UPCOMING_TASK but now HAS a reader -- "
        "delete the entry instead of leaving it to hide a live member"
    )
