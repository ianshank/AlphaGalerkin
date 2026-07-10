"""Audit @abstractmethod / Protocol members for missing call sites.

An abstract method that every subclass overrides but nothing ever *calls* is
dead code — a docstring describing an algorithm that does not run. That is
exactly how ``PDEGame.get_reward`` (F1) survived: abstract, universally
implemented, invoked nowhere in ``src/``. This tool enumerates such methods and
reports the ones with no call site.

Heuristic (deliberately simple, so it is trustworthy):

* An ``@abstractmethod`` named ``foo`` is *called* iff the attribute-call form
  ``.foo(`` appears anywhere under the scanned roots. Overrides are ``def foo(``
  and do not match, so only genuine call sites count.
* A ``Protocol`` member ``bar`` is *read* iff the attribute form ``.bar`` appears
  anywhere under the scanned roots.

Usage::

    python -m scripts.audit_abstractions [PATH ...] [--fail-on-missing]

With no PATH, scans ``src``. Exit code is non-zero only with
``--fail-on-missing`` (so it can run non-blocking first, blocking later).
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

# Dunder / framework hooks that are called implicitly and must never be flagged.
_IMPLICIT = frozenset(
    {
        "__init__",
        "__call__",
        "__enter__",
        "__exit__",
        "__iter__",
        "__next__",
        "__len__",
        "__getitem__",
        "__setitem__",
        "__repr__",
        "__str__",
        "__eq__",
        "__hash__",
    }
)


@dataclass
class Finding:
    """One declared abstract method or protocol member and where it lives."""

    kind: str  # "abstractmethod" | "protocol_member"
    name: str
    cls: str
    file: str
    lineno: int
    is_property: bool = False


@dataclass
class AuditReport:
    """Members with no call site / reader, split by kind."""

    abstract_missing: list[Finding] = field(default_factory=list)
    protocol_missing: list[Finding] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of flagged members."""
        return len(self.abstract_missing) + len(self.protocol_missing)


def _iter_py_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
        else:
            files.extend(sorted(root.rglob("*.py")))
    return files


def _is_abstractmethod(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "abstractmethod":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "abstractmethod":
            return True
    return False


def _is_property(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id in {"property", "cached_property"}:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr in {"property", "cached_property"}:
            return True
    return False


def _is_protocol_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "Protocol":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Protocol":
            return True
    return False


def collect_members(files: list[Path]) -> tuple[list[Finding], list[Finding]]:
    """Return (abstract methods, protocol members) declared across ``files``."""
    abstracts: list[Finding] = []
    protocols: list[Finding] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            is_proto = _is_protocol_class(cls)
            for item in cls.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    if item.name in _IMPLICIT:
                        continue
                    if _is_abstractmethod(item):
                        abstracts.append(
                            Finding(
                                "abstractmethod",
                                item.name,
                                cls.name,
                                str(path),
                                item.lineno,
                                is_property=_is_property(item),
                            )
                        )
                    elif is_proto:
                        protocols.append(
                            Finding(
                                "protocol_member",
                                item.name,
                                cls.name,
                                str(path),
                                item.lineno,
                                is_property=_is_property(item),
                            )
                        )
    return abstracts, protocols


def audit(roots: list[Path]) -> AuditReport:
    """Scan ``roots`` and return the members with no call site / reader."""
    files = _iter_py_files(roots)
    corpus = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    abstracts, protocols = collect_members(files)

    def _has_use(f: Finding) -> bool:
        # Properties are read as attributes (``.name``); methods are called
        # (``.name(``). Use the matching pattern so abstract properties are not
        # mis-flagged as dead.
        pattern = r"\." + re.escape(f.name) + (r"\b" if f.is_property else r"\(")
        return bool(re.search(pattern, corpus))

    # De-duplicate by the fully-qualified (file, class, name) declaration key,
    # not by name alone: two different ABCs/Protocols may declare the same member
    # name, and name-only de-dup would silently drop the second declaration.
    report = AuditReport()
    seen_abstract: set[tuple[str, str, str]] = set()
    for f in abstracts:
        key = (f.file, f.cls, f.name)
        if key in seen_abstract:
            continue
        seen_abstract.add(key)
        if not _has_use(f):
            report.abstract_missing.append(f)

    seen_proto: set[tuple[str, str, str]] = set()
    for f in protocols:
        key = (f.file, f.cls, f.name)
        if key in seen_proto:
            continue
        seen_proto.add(key)
        if not _has_use(f):
            report.protocol_missing.append(f)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the audit and set the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["src"], help="Roots to scan (default: src).")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit non-zero if any abstract method / protocol member has no call site.",
    )
    args = parser.parse_args(argv)

    roots = [Path(p) for p in (args.paths or ["src"])]
    report = audit(roots)

    if report.abstract_missing:
        print("Abstract methods with NO call site (dead abstractions):")
        for f in report.abstract_missing:
            print(f"  {f.cls}.{f.name}  ({f.file}:{f.lineno})")
    if report.protocol_missing:
        print("Protocol members with NO reader:")
        for f in report.protocol_missing:
            print(f"  {f.cls}.{f.name}  ({f.file}:{f.lineno})")
    if report.total == 0:
        scanned = ", ".join(str(r) for r in roots)
        print(f"OK: every abstract method / protocol member under {scanned} has a call site.")

    return 1 if (args.fail_on_missing and report.total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
