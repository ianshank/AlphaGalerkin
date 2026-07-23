"""Tests for the abstract-method audit tool."""

from __future__ import annotations

from pathlib import Path

from scripts.audit_abstractions import audit, main


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
