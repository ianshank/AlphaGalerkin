"""Measure the repository's *shape* and gate it shrink-only (reflection ticket R-11).

The codebase is green and well-guarded but its shape -- oversized functions,
magic values, library ``print`` calls, lazy first-party imports, ad-hoc device
resolution, orphan modules, dead ``except ImportError`` guards, silent
back-compat shims, and a diverged deploy mirror -- was measured once in prose
and gated by nothing. A number in a Markdown table drifts the moment the next
commit lands. This tool makes the numbers data:

* ``write`` measures nine metrics plus one content hash per *input* a metric
  reads and emits ``config/shape_baseline.yaml`` (values, hashes, generating
  git SHA, tool versions).
* ``check`` re-measures and compares. A metric may only *shrink*. A recorded
  value that is **larger** than the live measurement while every input that
  metric reads is **unchanged** was edited by hand (nothing can have improved
  on identical inputs), not measured, and is rejected with an instruction to
  regenerate.

``tests/docs/test_shape_baseline.py`` runs the same comparison in CI, one named
test per metric, so a regression is attributable to a metric rather than to
"the shape guard".

Provenance is tracked **per input, not per tree** (schema 2). The nine metrics
do not all read the same files: ``orphan_modules`` also reads
``importer_roots`` (``scripts/``, ``dashboard/``), ``mirror_diverged_files``
reads ``mirror_root`` and ``dead_import_error_guards`` reads ``pyproject``.
Schema 1 hashed ``src/**/*.py`` alone, with two consequences (PR #151 review):
an improvement made in ``scripts/`` or ``hf_space/`` was rejected as a hand
edit (``src/`` had not changed, so "nothing can have improved"), and any
``src/`` edit switched hand-edit detection off for every metric at once. The
alternative -- one hash over the union of every input -- fixes only the first:
it would make an edit to ``scripts/`` disable detection for all nine metrics,
where today only ``orphan_modules`` reads ``scripts/``. So the baseline records
one hash per input root (:func:`input_hashes`, the set derived from
:class:`ShapeConfig`), each metric declares the inputs it reads
(:func:`metric_inputs`), and a count is "hand-edited" only when *that
metric's* inputs are all unchanged. ``content_hash`` is kept as the
``src_root`` entry so schema-1 files migrate on load
(:func:`migrate_baseline_document`): their one hash becomes the ``src_root``
input hash and the inputs they never recorded count as changed, which keeps
``check`` on an old baseline working (growth is still gated; only hand-edit
detection on the three multi-input metrics waits for a regeneration).

Every rule set, path root, allowlist and regex is a field on
:class:`ShapeConfig` (overridable with ``--config``), and every metric is a
standalone function over a repository root, so each is unit-testable on a
synthetic tree without running ruff (the ruff runner is injectable).

Usage::

    python -m scripts.measure_shape write [--output config/shape_baseline.yaml]
    python -m scripts.measure_shape check [--baseline config/shape_baseline.yaml]

Both accept ``--root``, ``--config`` and ``--log-level``. ``check`` exits 1 on
any violation, 2 on a missing or unreadable baseline.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found, no-redef, unused-ignore]

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE: Final[Path] = REPO_ROOT / "config" / "shape_baseline.yaml"

#: Bumped when the on-disk shape of ``config/shape_baseline.yaml`` changes incompatibly.
#: 1: a single ``content_hash`` over ``src_root``. 2: adds ``input_hashes``, one per
#: input root a metric reads; ``content_hash`` stays as the ``src_root`` entry.
SHAPE_BASELINE_SCHEMA_VERSION: Final[int] = 2

#: The schema a document without a ``schema_version`` key is taken to be.
_UNVERSIONED_SCHEMA_VERSION: Final[int] = 1

#: sha256 hex digest, the form every recorded hash takes.
_HEX_DIGEST: Final[str] = r"^[0-9a-f]{64}$"

#: A hung ruff or git child in CI is indistinguishable from a wedged runner. Bound both.
RUFF_SUBPROCESS_TIMEOUT_S: Final[float] = 180.0
GIT_SUBPROCESS_TIMEOUT_S: Final[float] = 30.0

#: The nine metric keys, in report order. The baseline must carry exactly these.
METRIC_NAMES: Final[tuple[str, ...]] = (
    "complexity_findings",
    "magic_value_findings",
    "library_print_findings",
    "lazy_first_party_imports",
    "adhoc_device_resolution_sites",
    "orphan_modules",
    "dead_import_error_guards",
    "shim_files_without_deprecation_warning",
    "mirror_diverged_files",
)

#: Per-``(file, rule)`` tables recorded alongside the scalar metrics.
TABLE_NAMES: Final[tuple[str, ...]] = ("magic_values",)

#: The scalar metric each table breaks down; a table reads what its metric reads.
TABLE_METRIC: Final[dict[str, str]] = {"magic_values": "magic_value_findings"}

#: Separator between the file path and the rule code in table keys.
TABLE_KEY_SEPARATOR: Final[str] = "::"

#: What ``check`` tells a contributor who lowered a number by hand.
REGENERATE_HINT: Final[str] = (
    "run `python -m scripts.measure_shape write` to regenerate the baseline"
)

_BASELINE_HEADER: Final[str] = (
    "# Generated by `python -m scripts.measure_shape write`. Do not edit by hand:\n"
    "# tests/docs/test_shape_baseline.py rejects a lowered count whose content hash\n"
    "# is unchanged. Improve the tree, then regenerate.\n"
)

_LOG_LEVELS: Final[tuple[str, ...]] = ("DEBUG", "INFO", "WARNING", "ERROR")

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ShapeConfig(BaseModel):
    """Every root, rule set, allowlist and regex the metrics read. No hidden literals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    src_root: str = Field(default="src", description="Repo-relative root of the measured tree.")
    importer_roots: tuple[str, ...] = Field(
        default=("src", "scripts", "dashboard"),
        description="Roots whose imports count as production consumers for the orphan metric.",
    )
    mirror_root: str = Field(
        default="hf_space/src", description="Repo-relative root of the deploy mirror of src_root."
    )
    pyproject: str = Field(default="pyproject.toml", description="Where hard dependencies live.")
    first_party_package: str = Field(
        default="src", description="Top-level package whose lazy imports are counted."
    )
    complexity_rules: tuple[str, ...] = Field(
        default=("C901", "PLR0912", "PLR0913", "PLR0915", "PLR0911"),
        description="ruff codes counted as complexity findings.",
    )
    magic_value_rules: tuple[str, ...] = Field(
        default=("PLR2004",), description="ruff codes counted as magic-value findings."
    )
    print_rules: tuple[str, ...] = Field(
        default=("T201",), description="ruff codes counted as print findings."
    )
    main_block_name: str = Field(
        default="__main__",
        description=(
            "A file with an `if __name__ == <this>:` block is a CLI module; its prints are "
            "its output, not library prints. Detected by AST, not by substring: a comment "
            "mentioning __main__ must not hide a library print."
        ),
    )
    type_checking_guard: str = Field(
        default="TYPE_CHECKING",
        description="Imports under `if <this>:` are static-only and not lazy imports.",
    )
    canonical_device_modules: tuple[str, ...] = Field(
        default=("src/device.py",),
        description="Repo-relative files allowed to resolve devices ad hoc.",
    )
    torch_module_name: str = Field(default="torch", description="Name torch is bound to.")
    cuda_literal: str = Field(default="cuda", description="The device string an ad-hoc site picks.")
    cli_entry_modules: tuple[str, ...] = Field(
        default=("src.agents.cli", "src.poc.cli", "src.tools.cli"),
        description="`python -m` / [project.scripts] targets: reachable without an importer.",
    )
    package_marker_files: tuple[str, ...] = Field(
        default=("__init__.py", "__main__.py"),
        description="Filenames never counted as orphan candidates.",
    )
    import_error_names: tuple[str, ...] = Field(
        default=("ImportError", "ModuleNotFoundError"),
        description="Exception names that make a try/except an import guard.",
    )
    import_name_to_distribution: dict[str, str] = Field(
        default_factory=lambda: {
            "yaml": "pyyaml",
            "hydra": "hydra-core",
            "PIL": "pillow",
            "skfem": "scikit-fem",
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
        },
        description="Import names whose distribution name differs (beyond -/_ normalisation).",
    )
    shim_pattern: str = Field(
        default=r"back-?compat|backwards.compat",
        description="Case-insensitive regex marking a file as a back-compat shim site.",
    )
    deprecation_marker: str = Field(
        default="DeprecationWarning",
        description="A shim file containing this text warns; one without is silent.",
    )

    @field_validator("shim_pattern")
    @classmethod
    def _pattern_compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            # pydantic converts only ValueError/AssertionError into a ValidationError.
            raise ValueError(f"shim_pattern is not a valid regex: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _rule_sets_are_non_empty(self) -> ShapeConfig:
        for name in ("complexity_rules", "magic_value_rules", "print_rules", "importer_roots"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty -- an empty rule set measures nothing")
        return self

    @property
    def ruff_rules(self) -> tuple[str, ...]:
        """Every ruff code any metric reads, so one subprocess serves all three."""
        seen: dict[str, None] = {}
        for code in (*self.complexity_rules, *self.magic_value_rules, *self.print_rules):
            seen.setdefault(code, None)
        return tuple(seen)


def load_config(path: Path | None) -> ShapeConfig:
    """Load a YAML override, or the defaults when ``path`` is ``None``."""
    if path is None:
        return ShapeConfig()
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if document is None:
        return ShapeConfig()
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(document).__name__}")
    return ShapeConfig.model_validate(document)


# ---------------------------------------------------------------------------
# Shared primitives (kept in step with tests/support/import_graph.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    """One located finding: a repo-relative path, a 1-based line, and a detail string."""

    path: str
    line: int
    detail: str = ""


@dataclass(frozen=True)
class RuffFinding:
    """One entry of ``ruff check --output-format json`` reduced to what the metrics read."""

    code: str
    path: str
    line: int


#: Directory component that marks bytecode caches; never part of the measured tree.
CACHE_DIR_NAME: Final[str] = "__pycache__"


def is_cache_path(path: Path) -> bool:
    """True when ``path`` sits inside a bytecode cache directory."""
    return CACHE_DIR_NAME in path.parts


def python_files_under(root: Path) -> list[Path]:
    """All ``*.py`` under ``root``, excluding bytecode caches, sorted. Empty if absent."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if not is_cache_path(p))


def module_name_for(path: Path, repo_root: Path) -> str:
    """Dotted module name a repo-relative source file resolves to."""
    return ".".join(path.relative_to(repo_root).with_suffix("").parts)


def relative_posix(path: Path, repo_root: Path) -> str:
    """Repo-relative POSIX path, the form every Site and table key uses."""
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _iter_import_names(node: ast.AST) -> Iterator[str]:
    """Absolute module names an ``Import``/``ImportFrom`` names (relative imports skipped)."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        yield node.module


def _imports_of(path: Path, repo_root: Path) -> set[str]:
    """Modules ``path`` imports, with relative imports resolved and ``from X import y`` as ``X.y``.

    Mirrors ``tests/support/import_graph.imported_modules`` and adds the
    ``X.y`` expansion the orphan metric needs: ``from src.a import b`` reaches
    module ``src.a.b`` even though the statement names only ``src.a``.
    """
    own_package_parts = module_name_for(path, repo_root).split(".")[:-1]
    imported: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    imported.add(node.module)
                    imported.update(f"{node.module}.{alias.name}" for alias in node.names)
            else:
                base_parts = own_package_parts[: len(own_package_parts) - node.level + 1]
                base = ".".join(base_parts)
                imported.add(f"{base}.{node.module}" if node.module else base)
    return imported


# ---------------------------------------------------------------------------
# ruff-backed metrics (1)-(3)
# ---------------------------------------------------------------------------

RuffRunner = Callable[[Path, str, Sequence[str]], list[RuffFinding]]


def run_ruff(repo_root: Path, target: str, rules: Sequence[str]) -> list[RuffFinding]:
    """Run ``ruff check <target> --select <rules> --output-format json`` under ``repo_root``.

    ``--exit-zero`` makes findings exit 0, so any non-zero status is a real
    failure (bad path, broken config) and is raised rather than read as zero
    findings -- a ruff that cannot run must not look like a clean tree.
    """
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        target,
        "--select",
        ",".join(rules),
        "--output-format",
        "json",
        "--exit-zero",
    ]
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=RUFF_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ruff exited {completed.returncode}: {completed.stderr.strip() or '<no stderr>'}"
        )
    return parse_ruff_json(completed.stdout, repo_root)


def parse_ruff_json(text: str, repo_root: Path) -> list[RuffFinding]:
    """Reduce ruff's JSON to :class:`RuffFinding`.

    Code-less entries (syntax errors) are skipped with a warning, and files
    under a bytecode cache are dropped so the ruff-counted file set is the
    same one :func:`python_files_under` gives the AST metrics.

    A relative ``filename`` is resolved against ``repo_root`` -- the directory
    :func:`run_ruff` runs ruff in -- never against the caller's working
    directory, so ``measure --root /elsewhere`` from another directory
    attributes findings to the right tree (PR #151 review).
    """
    findings: list[RuffFinding] = []
    for entry in json.loads(text):
        code = entry.get("code")
        if not code:
            logger.warning("ruff_entry_without_code", filename=entry.get("filename"))
            continue
        path = Path(entry["filename"])
        if not path.is_absolute():
            path = repo_root / path
        if is_cache_path(path):
            continue
        findings.append(
            RuffFinding(
                code=str(code),
                path=relative_posix(path, repo_root),
                line=int(entry["location"]["row"]),
            )
        )
    return findings


def ruff_version() -> str:
    """``ruff --version`` as a provenance string."""
    completed = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        capture_output=True,
        text=True,
        timeout=RUFF_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def count_complexity_findings(findings: Iterable[RuffFinding], rules: Sequence[str]) -> int:
    """Metric 1: findings whose code is one of ``rules`` (C901 / PLR091x by default)."""
    wanted = set(rules)
    return sum(1 for f in findings if f.code in wanted)


def magic_value_table(findings: Iterable[RuffFinding], rules: Sequence[str]) -> dict[str, int]:
    """Metric 2, tabulated as ``{"<path>::<rule>": count}``.

    Per ``(file, rule)`` so a new finding in one file cannot hide behind a
    removal in another.
    """
    wanted = set(rules)
    table: dict[str, int] = {}
    for f in findings:
        if f.code in wanted:
            key = f"{f.path}{TABLE_KEY_SEPARATOR}{f.code}"
            table[key] = table.get(key, 0) + 1
    return dict(sorted(table.items()))


def count_magic_values(findings: Iterable[RuffFinding], rules: Sequence[str]) -> int:
    """Metric 2, scalar: the sum of :func:`magic_value_table`."""
    return sum(magic_value_table(findings, rules).values())


def has_main_block(tree: ast.AST, main_block_name: str) -> bool:
    """True when ``tree`` tests ``__name__`` against ``main_block_name``.

    Either operand order (``__name__ == "__main__"`` or the reverse) counts.
    """
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
            continue
        operands = [node.test.left, *node.test.comparators]
        names = {o.id for o in operands if isinstance(o, ast.Name)}
        constants = {o.value for o in operands if isinstance(o, ast.Constant)}
        if "__name__" in names and main_block_name in constants:
            return True
    return False


def library_print_findings(
    findings: Iterable[RuffFinding], repo_root: Path, rules: Sequence[str], main_block_name: str
) -> list[RuffFinding]:
    """Metric 3: print findings in files with no ``if __name__ == "__main__":`` block.

    A ``print`` in a CLI module is its output; one in library code is a log
    line that bypasses structlog. The reflection's Appendix A split the two by
    substring (``"__main__" in text``); the first mutation run against this
    guard defeated that with a *comment* mentioning ``__main__``, so the block
    is detected structurally.
    """
    wanted = set(rules)
    verdict_cache: dict[str, bool] = {}
    out: list[RuffFinding] = []
    for f in findings:
        if f.code not in wanted:
            continue
        if f.path not in verdict_cache:
            verdict_cache[f.path] = has_main_block(_parse(repo_root / f.path), main_block_name)
        if not verdict_cache[f.path]:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# AST metrics (4)-(7)
# ---------------------------------------------------------------------------


def lazy_first_party_imports(repo_root: Path, config: ShapeConfig) -> list[Site]:
    """Metric 4: ``import src...`` statements inside a function body, ``TYPE_CHECKING`` excluded.

    A lazy import is how a layering cycle hides from the static import graph;
    counting them is the first step to unwinding the SCCs the reflection
    named (``{data, training}``, ``{pde, experiments, research}``).

    Every alias of an ``import a, b`` statement is a site of its own: a
    first-party module hidden behind a third-party one (``import os, src.x``)
    is exactly the import this metric exists to count (PR #151 review).
    """
    package = config.first_party_package

    def is_first_party(module: str | None) -> bool:
        return bool(module) and (module == package or str(module).startswith(package + "."))

    sites: list[Site] = []
    for path in python_files_under(repo_root / config.src_root):
        tree = _parse(path)
        static_only: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == config.type_checking_guard
            ):
                static_only.update(id(sub) for sub in ast.walk(node))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, (ast.Import, ast.ImportFrom)):
                    continue
                if id(sub) in static_only:
                    continue
                modules = (
                    [sub.module]
                    if isinstance(sub, ast.ImportFrom)
                    else [alias.name for alias in sub.names]
                )
                for module in modules:
                    if is_first_party(module):
                        sites.append(Site(relative_posix(path, repo_root), sub.lineno, str(module)))
    return sites


def _is_cuda_available_call(node: ast.AST, torch_name: str) -> bool:
    """``<torch_name>.cuda.is_available()``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "is_available"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == torch_name
    )


def _is_bare_device_call(node: ast.AST, torch_name: str, literal: str) -> bool:
    """``<torch_name>.device("<literal>")``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "device"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == torch_name
        and len(node.args) >= 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == literal
    )


def adhoc_device_resolution_sites(repo_root: Path, config: ShapeConfig) -> list[Site]:
    """Metric 5: device resolution written inline instead of through the canonical resolver.

    Two shapes are matched by AST, not by grep: the conditional expression
    ``"cuda" if torch.cuda.is_available() else ...`` (also with
    ``torch.device("cuda")`` as its body) and a bare ``torch.device("cuda")``
    call. Files in ``canonical_device_modules`` are exempt -- they *are* the
    resolver.
    """
    exempt = set(config.canonical_device_modules)
    torch_name = config.torch_module_name
    literal = config.cuda_literal
    sites: list[Site] = []
    for path in python_files_under(repo_root / config.src_root):
        rel = relative_posix(path, repo_root)
        if rel in exempt:
            continue
        # ``ast.walk`` is breadth-first, so a conditional expression is seen
        # before the ``torch.device("cuda")`` call in its body; claim that call
        # so the same site is not counted twice.
        claimed: set[int] = set()
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.IfExp) and _is_cuda_available_call(node.test, torch_name):
                body = node.body
                if isinstance(body, ast.Constant) and body.value == literal:
                    sites.append(Site(rel, node.lineno, "conditional-expression"))
                elif _is_bare_device_call(body, torch_name, literal):
                    claimed.add(id(body))
                    sites.append(Site(rel, node.lineno, "conditional-expression"))
            elif (
                isinstance(node, ast.Call)
                and id(node) not in claimed
                and _is_bare_device_call(node, torch_name, literal)
            ):
                sites.append(Site(rel, node.lineno, "bare-device-call"))
    return sites


def orphan_modules(repo_root: Path, config: ShapeConfig) -> list[str]:
    """Metric 6: ``src`` modules no production root imports, CLI entry points excluded.

    Production roots are ``importer_roots`` (tests are deliberately not
    among them: a module reached only by its tests is the class this metric
    exists to surface). ``cli_entry_modules`` are reachable by ``python -m``
    without an importer and are excluded by rule, not by omission.
    """
    importers: dict[str, set[str]] = {}
    for base in config.importer_roots:
        for path in python_files_under(repo_root / base):
            importers[module_name_for(path, repo_root)] = _imports_of(path, repo_root)
    excluded = set(config.cli_entry_modules)
    markers = set(config.package_marker_files)
    orphans: list[str] = []
    for path in python_files_under(repo_root / config.src_root):
        if path.name in markers:
            continue
        module = module_name_for(path, repo_root)
        if module in excluded:
            continue
        if any(module in names for who, names in importers.items() if who != module):
            continue
        orphans.append(module)
    return orphans


def normalise_distribution_name(name: str) -> str:
    """PEP 503 normalisation, with ``_`` as the canonical separator."""
    return re.sub(r"[-_.]+", "_", name).lower()


def hard_dependencies(pyproject_path: Path) -> frozenset[str]:
    """Normalised distribution names in ``[project].dependencies`` (extras excluded)."""
    document = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    specs = document.get("project", {}).get("dependencies", [])
    names: set[str] = set()
    for spec in specs:
        match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", str(spec))
        if match is None:
            raise ValueError(f"unparseable dependency specifier: {spec!r}")
        names.add(normalise_distribution_name(match.group(1)))
    return frozenset(names)


def _import_guards(tree: ast.AST, error_names: Sequence[str]) -> Iterator[tuple[int, list[str]]]:
    """``(lineno, [modules])`` for each ``try`` whose handlers catch an import error."""
    wanted = set(error_names)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        caught: list[str] = []
        for handler in node.handlers:
            kind = handler.type
            if isinstance(kind, ast.Name):
                caught.append(kind.id)
            elif isinstance(kind, ast.Tuple):
                caught.extend(elt.id for elt in kind.elts if isinstance(elt, ast.Name))
        if not wanted.intersection(caught):
            continue
        modules: list[str] = []
        for statement in node.body:
            for sub in ast.walk(statement):
                modules.extend(_iter_import_names(sub))
        yield node.lineno, modules


def dead_import_error_guards(
    repo_root: Path, config: ShapeConfig, hard_deps: frozenset[str] | None = None
) -> list[Site]:
    """Metric 7: ``except ImportError`` guards whose every import is a declared hard dependency.

    Such a guard can never fire in a supported install and only hides real
    breakage. A guard protecting at least one genuinely optional import is
    live and is not counted, whatever else it wraps.
    """
    deps = hard_dependencies(repo_root / config.pyproject) if hard_deps is None else hard_deps
    aliases = config.import_name_to_distribution
    sites: list[Site] = []
    for path in python_files_under(repo_root / config.src_root):
        for line, modules in _import_guards(_parse(path), config.import_error_names):
            if not modules:
                continue
            tops = [m.split(".")[0] for m in modules]
            if all(normalise_distribution_name(aliases.get(t, t)) in deps for t in tops):
                sites.append(Site(relative_posix(path, repo_root), line, ", ".join(modules)))
    return sites


# ---------------------------------------------------------------------------
# Text metrics (8)-(9) and the content hash
# ---------------------------------------------------------------------------


def shim_files_without_deprecation_warning(repo_root: Path, config: ShapeConfig) -> list[str]:
    """Metric 8: files matching ``shim_pattern`` that never mention ``deprecation_marker``.

    A back-compat alias that does not warn has no removal path: nothing tells
    a caller to move, so nothing ever lets the shim go.
    """
    pattern = re.compile(config.shim_pattern, re.IGNORECASE)
    out: list[str] = []
    for path in python_files_under(repo_root / config.src_root):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text) and config.deprecation_marker not in text:
            out.append(relative_posix(path, repo_root))
    return out


@dataclass(frozen=True)
class MirrorReport:
    """Byte-comparison of the deploy mirror against the tree it mirrors."""

    identical: int
    diverged: int
    mirror_only: int
    diverged_paths: tuple[str, ...] = field(default=())


def mirror_divergence(repo_root: Path, config: ShapeConfig) -> MirrorReport:
    """Metric 9: mirror files whose bytes differ from their ``src_root`` counterpart.

    Files present only in the mirror are reported separately and not gated;
    they are a different question (should they exist?) from drift.
    """
    mirror = repo_root / config.mirror_root
    src = repo_root / config.src_root
    identical = diverged = mirror_only = 0
    diverged_paths: list[str] = []
    for path in python_files_under(mirror):
        counterpart = src / path.relative_to(mirror)
        if not counterpart.is_file():
            mirror_only += 1
        elif path.read_bytes() == counterpart.read_bytes():
            identical += 1
        else:
            diverged += 1
            diverged_paths.append(relative_posix(path, repo_root))
    return MirrorReport(identical, diverged, mirror_only, tuple(diverged_paths))


def _dedup(items: Iterable[str]) -> tuple[str, ...]:
    """Unique items in first-seen order."""
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return tuple(seen)


def input_roots(config: ShapeConfig) -> tuple[str, ...]:
    """Every repo-relative input some metric reads, derived from ``config`` (never a literal list).

    ``src_root`` (every metric), ``importer_roots`` (orphans), ``mirror_root``
    (mirror divergence) and ``pyproject`` (dead import guards), deduplicated
    in that order -- ``src`` is normally both the measured tree and an
    importer root.
    """
    return _dedup((config.src_root, *config.importer_roots, config.mirror_root, config.pyproject))


def metric_inputs(config: ShapeConfig) -> dict[str, tuple[str, ...]]:
    """The inputs each metric reads, keyed by :data:`METRIC_NAMES`.

    This is the provenance contract: a recorded count is "hand-edited" only
    when *these* inputs are all unchanged. A metric that starts reading a new
    root must be added here, and ``tests/scripts/test_measure_shape.py``
    asserts the union equals :func:`input_roots` so a root no metric declares
    (or a metric that declares nothing) fails there.
    """
    src = (config.src_root,)
    return {
        "complexity_findings": src,
        "magic_value_findings": src,
        "library_print_findings": src,
        "lazy_first_party_imports": src,
        "adhoc_device_resolution_sites": src,
        "orphan_modules": _dedup((config.src_root, *config.importer_roots)),
        "dead_import_error_guards": _dedup((config.src_root, config.pyproject)),
        "shim_files_without_deprecation_warning": src,
        "mirror_diverged_files": _dedup((config.src_root, config.mirror_root)),
    }


def _hash_input(repo_root: Path, root: str) -> str:
    """sha256 over sorted repo-relative paths and bytes of one input.

    A directory contributes every ``*.py`` under it (what the metrics read);
    a file (``pyproject.toml``) contributes itself. A missing input hashes
    to the empty digest, deterministically.
    """
    target = repo_root / root
    files = [target] if target.is_file() else python_files_under(target)
    digest = hashlib.sha256()
    for path in files:
        digest.update(relative_posix(path, repo_root).encode("utf-8"))
        digest.update(b"\n")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def content_hash(repo_root: Path, config: ShapeConfig) -> str:
    """sha256 over ``src_root/**/*.py`` -- the ``src_root`` entry of :func:`input_hashes`.

    Byte-identical to the schema-1 definition, which is what lets a schema-1
    baseline migrate: its one hash *is* this one.

    Computed from the working tree, never from ``git rev-parse HEAD:src``:
    local improvements are uncommitted when a contributor runs ``check``, and
    a pull-request checkout is a synthetic merge commit.
    """
    return _hash_input(repo_root, config.src_root)


def input_hashes(repo_root: Path, config: ShapeConfig) -> dict[str, str]:
    """One :func:`_hash_input` digest per :func:`input_roots` entry, in root order."""
    return {root: _hash_input(repo_root, root) for root in input_roots(config)}


# ---------------------------------------------------------------------------
# Baseline document
# ---------------------------------------------------------------------------


class ShapeBaseline(BaseModel):
    """Validated ``config/shape_baseline.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SHAPE_BASELINE_SCHEMA_VERSION, ge=1)
    generated_from: str = Field(min_length=1, description="git SHA the baseline was measured at.")
    git_dirty: bool = Field(
        default=False,
        description="Whether that tree had uncommitted changes, untracked files included.",
    )
    tool_versions: dict[str, str] = Field(default_factory=dict)
    content_hash: str = Field(pattern=_HEX_DIGEST, description="The src_root input hash.")
    input_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="One hash per input root a metric reads (schema 2); see metric_inputs().",
    )
    metrics: dict[str, int]
    tables: dict[str, dict[str, int]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _shape_is_complete(self) -> ShapeBaseline:
        if self.schema_version > SHAPE_BASELINE_SCHEMA_VERSION:
            raise ValueError(
                f"baseline schema_version {self.schema_version} is newer than this tool "
                f"understands ({SHAPE_BASELINE_SCHEMA_VERSION}); upgrade the tool"
            )
        if self.schema_version >= SHAPE_BASELINE_SCHEMA_VERSION and not self.input_hashes:
            raise ValueError(
                f"schema_version {self.schema_version} requires input_hashes; a schema-1 "
                f"file carries only content_hash and is migrated by load_baseline()"
            )
        bad_hashes = {k: v for k, v in self.input_hashes.items() if not re.match(_HEX_DIGEST, v)}
        if bad_hashes:
            raise ValueError(f"input_hashes must be sha256 hex digests: {bad_hashes}")
        if self.input_hashes and self.content_hash not in self.input_hashes.values():
            raise ValueError(
                "content_hash is the src_root entry of input_hashes and must appear there; "
                "one of the two was edited by hand"
            )
        expected = set(METRIC_NAMES)
        actual = set(self.metrics)
        if actual != expected:
            raise ValueError(
                f"metrics must be exactly {sorted(expected)}; "
                f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}"
            )
        negative = {k: v for k, v in self.metrics.items() if v < 0}
        if negative:
            raise ValueError(f"metric counts must be non-negative: {negative}")
        unknown_tables = set(self.tables) - set(TABLE_NAMES)
        if unknown_tables:
            raise ValueError(f"unknown tables {sorted(unknown_tables)}; known: {list(TABLE_NAMES)}")
        for name, table in self.tables.items():
            bad = {k: v for k, v in table.items() if v < 0}
            if bad:
                raise ValueError(f"table {name!r} has negative counts: {bad}")
        return self


def git_head_sha(repo_root: Path, *, exclude: Sequence[str] = ()) -> tuple[str, bool]:
    """``(HEAD sha, dirty)``; ``("unknown", False)`` when git is unavailable.

    ``dirty`` counts **untracked** files (``--untracked-files=all``), not only
    modified tracked ones: the hashes are computed from the working tree, so
    a new, not-yet-added ``src/x.py`` changes the metrics, and a baseline
    written then must not record ``git_dirty: false`` (PR #151 review).
    Ignored files stay excluded, as they are from the tracked tree.

    ``exclude`` lists repo-relative paths left out of the check -- ``write``
    passes its own output file, which is being rewritten and is not a change
    in the tree the numbers came from (a second ``write`` without a commit in
    between would otherwise always be dirty).
    """
    status_command = ["git", "status", "--porcelain", "--untracked-files=all", "--", "."]
    status_command.extend(f":(exclude){path}" for path in exclude)
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=GIT_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
        status = subprocess.run(
            status_command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=GIT_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown", False
    if head.returncode != 0:
        return "unknown", False
    return head.stdout.strip(), bool(status.stdout.strip())


def measure(
    repo_root: Path,
    config: ShapeConfig,
    *,
    ruff_runner: RuffRunner | None = None,
    ruff_version_probe: Callable[[], str] | None = None,
    git_probe: Callable[[Path], tuple[str, bool]] | None = None,
) -> ShapeBaseline:
    """Run every metric once and assemble a :class:`ShapeBaseline` for ``repo_root``.

    The three collaborators default to the real subprocess helpers, resolved
    at call time (not bound as defaults) so a test can substitute them.
    """
    runner = run_ruff if ruff_runner is None else ruff_runner
    version_probe = ruff_version if ruff_version_probe is None else ruff_version_probe
    # if/else, not a ternary: mypy joins the keyword-only-parameter function
    # and the Callable to bare ``function`` and then refuses to call it.
    sha_probe: Callable[[Path], tuple[str, bool]]
    if git_probe is None:
        sha_probe = git_head_sha
    else:
        sha_probe = git_probe
    log = logger.bind(root=str(repo_root))
    findings = runner(repo_root, config.src_root, config.ruff_rules)
    log.debug("ruff_findings", count=len(findings), rules=list(config.ruff_rules))
    prints = library_print_findings(findings, repo_root, config.print_rules, config.main_block_name)
    lazy = lazy_first_party_imports(repo_root, config)
    devices = adhoc_device_resolution_sites(repo_root, config)
    orphans = orphan_modules(repo_root, config)
    guards = dead_import_error_guards(repo_root, config)
    shims = shim_files_without_deprecation_warning(repo_root, config)
    mirror = mirror_divergence(repo_root, config)
    metrics: dict[str, int] = {
        "complexity_findings": count_complexity_findings(findings, config.complexity_rules),
        "magic_value_findings": count_magic_values(findings, config.magic_value_rules),
        "library_print_findings": len(prints),
        "lazy_first_party_imports": len(lazy),
        "adhoc_device_resolution_sites": len(devices),
        "orphan_modules": len(orphans),
        "dead_import_error_guards": len(guards),
        "shim_files_without_deprecation_warning": len(shims),
        "mirror_diverged_files": mirror.diverged,
    }
    for name, value in metrics.items():
        log.info("shape_metric", metric=name, value=value)
    log.debug("mirror_detail", identical=mirror.identical, mirror_only=mirror.mirror_only)
    sha, dirty = sha_probe(repo_root)
    hashes = input_hashes(repo_root, config)
    return ShapeBaseline(
        generated_from=sha,
        git_dirty=dirty,
        tool_versions={
            "python": platform.python_version(),
            "ruff": version_probe(),
            "measure_shape_schema": str(SHAPE_BASELINE_SCHEMA_VERSION),
        },
        content_hash=hashes[config.src_root],
        input_hashes=hashes,
        metrics=metrics,
        tables={"magic_values": magic_value_table(findings, config.magic_value_rules)},
    )


def dump_baseline(baseline: ShapeBaseline) -> str:
    """Serialise deterministically (header comment, insertion-ordered keys)."""
    body = str(
        yaml.safe_dump(baseline.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    )
    return _BASELINE_HEADER + body


def write_baseline(baseline: ShapeBaseline, path: Path) -> None:
    """Write ``baseline`` to ``path``, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_baseline(baseline), encoding="utf-8")


def migrate_baseline_document(
    document: dict[str, object], *, src_root: str = ShapeConfig().src_root
) -> dict[str, object]:
    """Lift a schema-1 document to schema 2 without touching the caller's mapping.

    Schema 1 recorded one ``content_hash`` over ``src_root``; schema 2
    records ``input_hashes`` per input. The migration records the old hash
    under ``src_root`` and nothing else, so every metric that also reads
    another input (orphans, dead guards, mirror) sees that input as *not
    recorded* -- treated as changed by :func:`inputs_unchanged`, which keeps
    growth gated and merely defers hand-edit detection on those three
    metrics until the file is regenerated. A schema-2 document is returned
    as a copy, unchanged.
    """
    migrated = dict(document)
    version = migrated.get("schema_version", _UNVERSIONED_SCHEMA_VERSION)
    if not isinstance(version, int) or version >= SHAPE_BASELINE_SCHEMA_VERSION:
        return migrated
    migrated["schema_version"] = SHAPE_BASELINE_SCHEMA_VERSION
    hashes = migrated.get("input_hashes")
    if not isinstance(hashes, dict):
        hashes = {}
    if "content_hash" in migrated:
        hashes = {src_root: migrated["content_hash"], **hashes}
    migrated["input_hashes"] = hashes
    return migrated


def load_baseline(path: Path, config: ShapeConfig | None = None) -> ShapeBaseline:
    """Load, migrate (:func:`migrate_baseline_document`) and validate a baseline file."""
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(document).__name__}")
    src_root = (ShapeConfig() if config is None else config).src_root
    return ShapeBaseline.model_validate(migrate_baseline_document(document, src_root=src_root))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """One way the live tree disagrees with the recorded baseline."""

    metric: str
    recorded: int
    actual: int
    kind: str  # one of VIOLATION_KINDS

    @property
    def message(self) -> str:
        if self.kind == "grew":
            return (
                f"{self.metric}: measured {self.actual} > recorded {self.recorded} -- the shape "
                f"grew; shrink it back or, if the growth is deliberate, {REGENERATE_HINT} "
                f"and justify it"
            )
        return (
            f"{self.metric}: recorded {self.recorded} > measured {self.actual} while every "
            f"input this metric reads is byte-identical to the tree the baseline was generated "
            f"on -- nothing can have improved, so the recorded number was edited by hand; "
            f"{REGENERATE_HINT}"
        )


#: ``grew``: the live tree exceeds the record. ``hand_edited``: the record
#: exceeds the live tree although the tree is unchanged, i.e. padding.
VIOLATION_KINDS: Final[tuple[str, ...]] = ("grew", "hand_edited")


def compare_count(
    metric: str, recorded: int, actual: int, *, src_unchanged: bool
) -> Violation | None:
    """The shrink-only rule for one count.

    * ``actual > recorded`` is a regression, whatever the tree did.
    * ``actual < recorded`` is an improvement -- unless ``src_unchanged``
      (every input *this metric* reads is unchanged; the keyword keeps its
      schema-1 name), in which case nothing can have improved and the
      recorded number was edited by hand (a lowered count that has not been
      re-measured, or padding above the measurement to buy slack).
    """
    if actual > recorded:
        return Violation(metric, recorded, actual, "grew")
    if actual < recorded and src_unchanged:
        return Violation(metric, recorded, actual, "hand_edited")
    return None


def inputs_unchanged(
    recorded: ShapeBaseline, current: ShapeBaseline, inputs: Iterable[str]
) -> bool:
    """True iff every named input is recorded on both sides with the same hash.

    An input the recorded baseline never hashed (a schema-1 file, or a root
    added to the config since) is *not* provably unchanged, so it counts as
    changed: the rule then cannot call an improvement a hand edit, which is
    the safe direction -- growth is gated by :func:`compare_count` regardless.
    """
    return all(
        name in recorded.input_hashes
        and recorded.input_hashes[name] == current.input_hashes.get(name)
        for name in inputs
    )


def compare(
    recorded: ShapeBaseline, current: ShapeBaseline, config: ShapeConfig | None = None
) -> list[Violation]:
    """Apply :func:`compare_count` to every metric and every table entry.

    Each metric's ``src_unchanged`` is :func:`inputs_unchanged` over the
    inputs :func:`metric_inputs` declares for it under ``config`` (defaults
    when ``None``); a table inherits its metric's (:data:`TABLE_METRIC`).
    """
    reads = metric_inputs(ShapeConfig() if config is None else config)
    violations: list[Violation] = []
    for name in METRIC_NAMES:
        unchanged = inputs_unchanged(recorded, current, reads[name])
        found = compare_count(
            name, recorded.metrics[name], current.metrics[name], src_unchanged=unchanged
        )
        if found is not None:
            violations.append(found)
    for table in TABLE_NAMES:
        unchanged = inputs_unchanged(recorded, current, reads[TABLE_METRIC[table]])
        old = recorded.tables.get(table, {})
        new = current.tables.get(table, {})
        for key in sorted(set(old) | set(new)):
            found = compare_count(
                f"{table}[{key}]", old.get(key, 0), new.get(key, 0), src_unchanged=unchanged
            )
            if found is not None:
                violations.append(found)
    return violations


def format_report(
    recorded: ShapeBaseline | None,
    current: ShapeBaseline,
    violations: Sequence[Violation],
    config: ShapeConfig | None = None,
) -> str:
    """Human-readable summary of a measurement and (optionally) its comparison."""
    roots = input_roots(ShapeConfig() if config is None else config)
    lines = ["=== Repository shape ==="]
    for name in METRIC_NAMES:
        actual = current.metrics[name]
        if recorded is None:
            lines.append(f"{name:40s} {actual:6d}")
        else:
            lines.append(f"{name:40s} {actual:6d}  (recorded {recorded.metrics[name]})")
    lines.append(f"content_hash {current.content_hash}")
    if recorded is not None:
        for root in roots:
            if root not in recorded.input_hashes:
                state = "not recorded"
            elif inputs_unchanged(recorded, current, (root,)):
                state = "unchanged"
            else:
                state = "changed"
            label = root if Path(root).suffix else f"{root}/"  # pyproject.toml vs src/
            lines.append(
                f"{label} is {state} relative to the baseline ({recorded.generated_from[:12]})"
            )
    if violations:
        lines.append("")
        lines.append(f"{len(violations)} violation(s):")
        lines.extend(f"  - {v.message}" for v in violations)
    elif recorded is not None:
        lines.append("")
        lines.append("OK: no metric grew, and no recorded count was lowered by hand.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def configure_logging(level: str) -> None:
    """Route structlog to stderr at ``level`` so stdout stays the report."""
    numeric = int(getattr(logging, level.upper()))
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="measure_shape",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML overriding ShapeConfig fields."
    )
    parser.add_argument(
        "--log-level", choices=_LOG_LEVELS, default="INFO", help="structlog threshold (stderr)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write", help="Measure and emit the baseline file.")
    write.add_argument("--output", type=Path, default=DEFAULT_BASELINE, help="Baseline to write.")
    check = commands.add_parser("check", help="Measure and compare against the baseline file.")
    check.add_argument(
        "--baseline", type=Path, default=DEFAULT_BASELINE, help="Baseline to compare against."
    )
    return parser


def output_pathspec(output: Path, repo_root: Path) -> tuple[str, ...]:
    """``output`` as a repo-relative path to exclude from the dirtiness check; ``()`` if outside."""
    try:
        return (output.resolve().relative_to(repo_root.resolve()).as_posix(),)
    except ValueError:
        return ()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. ``write`` exits 0; ``check`` exits 1 on violations, 2 on a bad baseline."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    config = load_config(args.config)
    root: Path = args.root.resolve()
    exclude = output_pathspec(args.output, root) if args.command == "write" else ()
    git_probe = (lambda r: git_head_sha(r, exclude=exclude)) if exclude else None
    current = measure(root, config, git_probe=git_probe)
    if args.command == "write":
        write_baseline(current, args.output)
        print(format_report(None, current, ()))
        logger.info("baseline_written", path=str(args.output), sha=current.generated_from)
        return 0
    try:
        recorded = load_baseline(args.baseline, config)
    except (OSError, ValueError) as exc:
        logger.error("baseline_unreadable", path=str(args.baseline), error=str(exc))
        print(f"cannot read baseline {args.baseline}: {exc}", file=sys.stderr)
        return 2
    violations = compare(recorded, current, config)
    print(format_report(recorded, current, violations, config))
    return 1 if violations else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
