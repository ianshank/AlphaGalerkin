"""Artifact-freeze manifest for the committed benchmark artifacts (R-05).

Every headline number in the charter's evidence register traces to a file under
``results/`` or ``config/baselines/``. Those files are committed, cited, and --
until this manifest -- unguarded: an edit to ``results/transfer_baseline_compare.csv``
changed a number the charter quotes, and nothing in the tree could tell that the
bytes had moved. Three headline claims have already been retracted here because
the number was written down before, or instead of, the artifact behind it. A
manifest makes the *artifact* the thing that cannot drift silently.

``results/MANIFEST.sha256`` is regenerated **only** by the ``claims-ledger`` and
``run-provenance`` skills, i.e. when an artifact is deliberately produced or
replaced. Every other change to a frozen artifact is a defect, and
``tests/docs/test_artifact_manifest.py`` fails on it.

Format -- ``sha256sum -c`` compatible::

    # <header comment lines>
    # presence-only: results/<name>.png
    <64 hex>  config/baselines/<name>_ci.json
    <64 hex>  results/<name>.csv

Hashed entries are the two-space form ``sha256sum`` itself emits. Presence-only
entries (PNGs: a re-rendered plot legitimately changes bytes without changing
any number) are written as ``#`` comment lines, which GNU ``sha256sum -c``
skips, so the whole file verifies with plain coreutils from the repo root::

    sha256sum -c results/MANIFEST.sha256

Which files are tracked is **configuration, not code**: the patterns, the
exclusions, the presence marker and the manifest path live in
:class:`ArtifactManifestConfig` with documented defaults and an optional
``--config`` YAML override.

Usage::

    python -m scripts.artifact_manifest [--config X.yaml] [--root DIR] [--log-level L] write
    python -m scripts.artifact_manifest [--config X.yaml] [--root DIR] [--log-level L] check

(Options are global and precede the subcommand.)

Exit codes: ``0`` manifest agrees with the tree, ``1`` mismatch (stale hash,
unlisted artifact, entry without a file, wrong entry kind, too few entries),
``2`` usage or input error (bad flags, unreadable config, malformed or absent
manifest, not a git checkout).

Where it runs: CI's ``lint`` job (``.github/workflows/ci.yml``) and
``make artifact-manifest`` both run ``check``. The ``lint`` job installs a
minimal dependency set -- ruff, mypy, and a handful of packages named
explicitly in its install step -- so this module's **module-scope** imports
are limited to the standard library plus ``structlog`` and ``pydantic``, both
of which that step installs by name. ``yaml`` is imported lazily inside
:func:`load_config`, because only the optional ``--config`` override needs it
and PyYAML reaches the lint job only transitively. The guard
``tests/docs/test_artifact_manifest.py`` asserts this footprint against the
workflow, so adding an import here that the lint job does not install fails a
test rather than the build.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import logging
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Default location of the manifest, relative to the repository root.
DEFAULT_MANIFEST_PATH: Final[str] = "results/MANIFEST.sha256"

#: Exit codes -- the contract callers (CI, the guard test) rely on.
EXIT_OK: Final[int] = 0
EXIT_MISMATCH: Final[int] = 1
EXIT_USAGE: Final[int] = 2

#: A hung ``git ls-files`` in CI is indistinguishable from a wedged runner. Bound it.
GIT_SUBPROCESS_TIMEOUT_S: Final[float] = 60.0

#: Read files in chunks so a large CSV does not need to fit in memory at once.
_HASH_CHUNK_BYTES: Final[int] = 1 << 20

#: One hashed line, as ``sha256sum`` writes and reads it. The second character
#: is `` `` (text mode) or ``*`` (binary mode); both are accepted on read.
_HASHED_LINE: Final[re.Pattern[str]] = re.compile(r"^(?P<digest>[0-9a-f]{64}) [ *](?P<path>\S.*)$")

_COMMENT_PREFIX: Final[str] = "#"

_LOG = structlog.get_logger(__name__)


class EntryKind(str, Enum):
    """How a manifest entry constrains its file."""

    HASHED = "hashed"
    PRESENCE = "presence"


class ArtifactManifestConfig(BaseModel):
    """Which committed files the manifest freezes, and how.

    Every value here is a *decision* about what counts as a frozen artifact.
    They are fields rather than literals so that adding a new artifact
    directory is a one-line, reviewable config change and so the guard test can
    read the same source of truth the CLI uses.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scan_roots: tuple[str, ...] = Field(
        default=("results", "config/baselines"),
        min_length=1,
        description=(
            "Repo-relative directories handed to `git ls-files`. Only tracked files "
            "under these roots are candidates; an untracked file is not an artifact."
        ),
    )
    hash_patterns: tuple[str, ...] = Field(
        default=("results/*.csv", "results/*.run.json", "config/baselines/*.json"),
        min_length=1,
        description=(
            "Segment-wise globs (`*` never crosses `/`) for files whose *bytes* are "
            "frozen: the CSVs the charter quotes, their provenance sidecars, and the "
            "baseline JSONs that gate regressions. `config/baselines/*.json` rather "
            "than `*_ci.json` on purpose: fail closed, so a future baseline that is "
            "not named `_ci` is frozen by default, and the template exclusion below "
            "is load-bearing instead of inert (the guard's exemption meta-test "
            "caught the narrower pattern making it so)."
        ),
    )
    presence_patterns: tuple[str, ...] = Field(
        default=("results/*.png",),
        description=(
            "Globs for files whose *existence* is frozen but whose bytes are not: a "
            "plot re-rendered from unchanged data is a legitimate byte change."
        ),
    )
    exclude: tuple[str, ...] = Field(
        default=("config/baselines/poc_headline.example.json",),
        description=(
            "Exact repo-relative paths that match a pattern but are not artifacts. "
            "The `.example.json` is a template users copy, not a measured baseline."
        ),
    )
    manifest_path: str = Field(
        default=DEFAULT_MANIFEST_PATH,
        min_length=1,
        description="Repo-relative path of the manifest. Never hashed, even if it matches.",
    )
    presence_marker: str = Field(
        default="# presence-only: ",
        min_length=2,
        description=(
            "Prefix of a presence-only line. Must start with `#` so `sha256sum -c` "
            "treats it as a comment and the file stays coreutils-verifiable."
        ),
    )
    min_hashed_entries: int = Field(
        default=10,
        ge=0,
        description=(
            "Vacuity floor: a manifest with fewer hashed entries than this is refused "
            "on `write` and fails `check`. A scanner matching no files passes everything."
        ),
    )

    @field_validator("presence_marker")
    @classmethod
    def _marker_is_a_comment(cls, value: str) -> str:
        if not value.startswith(_COMMENT_PREFIX):
            raise ValueError(
                f"presence_marker must start with {_COMMENT_PREFIX!r} so sha256sum -c "
                f"skips it; got {value!r}"
            )
        return value

    @field_validator("scan_roots", "hash_patterns", "presence_patterns", "exclude")
    @classmethod
    def _normalise(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(_normalise_path(entry) for entry in value)
        if any(not entry for entry in cleaned):
            raise ValueError(f"empty path entry in {value!r}")
        return cleaned

    @field_validator("manifest_path")
    @classmethod
    def _normalise_manifest(cls, value: str) -> str:
        cleaned = _normalise_path(value)
        if not cleaned:
            raise ValueError("manifest_path must not be empty")
        return cleaned

    def classify(self, path: str) -> EntryKind | None:
        """Return the entry kind ``path`` should have, or ``None`` if it is not frozen.

        The manifest itself and every ``exclude`` entry are never frozen. Hash
        patterns win over presence patterns when both match, because freezing
        bytes is the stricter contract and a file that matches both was almost
        certainly meant to be hashed.
        """
        if path == self.manifest_path or path in self.exclude:
            return None
        if any(_glob_match(path, pattern) for pattern in self.hash_patterns):
            return EntryKind.HASHED
        if any(_glob_match(path, pattern) for pattern in self.presence_patterns):
            return EntryKind.PRESENCE
        return None


@dataclass(frozen=True)
class ManifestEntry:
    """One line of the manifest that constrains a file."""

    path: str
    kind: EntryKind
    digest: str | None = None

    def __post_init__(self) -> None:
        if (self.kind is EntryKind.HASHED) != (self.digest is not None):
            raise ValueError(f"{self.path}: kind {self.kind.value} inconsistent with digest")


@dataclass(frozen=True)
class CheckReport:
    """Every way the tree and the manifest can disagree, kept separately.

    Kept as named categories rather than one boolean so the failure message
    says *what* moved -- a stale hash is an edited artifact; an unlisted file is
    an artifact that landed without the skill; a missing file is a deletion.
    """

    stale: tuple[str, ...] = ()
    unlisted: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    wrong_kind: tuple[str, ...] = ()
    hashed_count: int = 0
    presence_count: int = 0
    min_hashed_entries: int = 0
    duplicates: tuple[str, ...] = field(default=())

    @property
    def too_few(self) -> bool:
        return self.hashed_count < self.min_hashed_entries

    @property
    def ok(self) -> bool:
        return not (
            self.stale
            or self.unlisted
            or self.missing
            or self.wrong_kind
            or self.duplicates
            or self.too_few
        )

    def format(self, manifest_path: str) -> str:
        """Render the verdict with each category named and every path listed."""
        lines = [f"=== Artifact-freeze manifest check: {manifest_path} ==="]
        lines.append(f"hashed entries:   {self.hashed_count} (floor {self.min_hashed_entries})")
        lines.append(f"presence entries: {self.presence_count}")
        categories = (
            ("STALE (bytes changed since the manifest was written)", self.stale),
            ("UNLISTED (tracked artifact with no manifest entry)", self.unlisted),
            ("MISSING (manifest entry with no tracked file)", self.missing),
            ("WRONG KIND (hashed vs presence-only disagrees with the config)", self.wrong_kind),
            ("DUPLICATE (path listed more than once)", self.duplicates),
        )
        for title, paths in categories:
            if paths:
                lines.append(f"{title}:")
                lines.extend(f"  {path}" for path in paths)
        if self.too_few:
            lines.append(
                f"TOO FEW: {self.hashed_count} hashed entries < floor {self.min_hashed_entries} "
                f"-- the scanner matched (almost) nothing, which passes everything"
            )
        lines.append("")
        if self.ok:
            lines.append("OK: every frozen artifact matches the manifest.")
        else:
            lines.append(
                "MISMATCH: a frozen artifact moved. If this is a deliberate regeneration, "
                "follow the claims-ledger / run-provenance skill and run "
                "`python -m scripts.artifact_manifest write`; otherwise revert the artifact."
            )
        return "\n".join(lines)


class ManifestFormatError(ValueError):
    """The manifest file cannot be interpreted (as opposed to disagreeing with the tree)."""


#: Signature of the tracked-file lister so tests can substitute a fake for git.
TrackedLister = Callable[[Path, Sequence[str]], list[str]]


def _normalise_path(entry: str) -> str:
    """Strip the spellings that would silently stop an exact comparison from matching."""
    cleaned = entry.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.strip("/")


def _glob_match(path: str, pattern: str) -> bool:
    """Segment-wise glob: ``results/*.csv`` matches ``results/a.csv``, not ``results/x/a.csv``.

    :func:`fnmatch.fnmatch` lets ``*`` cross ``/``, which would silently widen
    every pattern to arbitrary depth; :meth:`pathlib.PurePath.match` anchors at
    the *right*, so ``results/*.csv`` would match ``outputs/results/a.csv``.
    Neither is the intended meaning, so this is spelled out.
    """
    path_parts = path.split("/")
    pattern_parts = pattern.split("/")
    if len(path_parts) != len(pattern_parts):
        return False
    return all(fnmatch.fnmatchcase(p, q) for p, q in zip(path_parts, pattern_parts, strict=True))


def git_tracked_files(root: Path, scan_roots: Sequence[str]) -> list[str]:
    """``git ls-files`` under ``scan_roots``, repo-relative, sorted.

    Tracked-ness is the criterion on purpose: a file the author forgot to
    ``git add`` is not a committed artifact and must not enter the manifest --
    that is how a manifest would come to reference a file CI cannot see.
    """
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", *scan_roots],
        cwd=str(root),
        capture_output=True,
        timeout=GIT_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed in {root} (exit {completed.returncode}): "
            f"{completed.stderr.decode('utf-8', 'replace').strip() or '<no stderr>'}"
        )
    raw = completed.stdout.decode("utf-8").split("\0")
    return sorted(_normalise_path(entry) for entry in raw if entry)


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_entries(
    config: ArtifactManifestConfig,
    root: Path,
    lister: TrackedLister = git_tracked_files,
) -> list[ManifestEntry]:
    """The entries the manifest *should* contain for the tree at ``root``."""
    entries: list[ManifestEntry] = []
    for path in lister(root, config.scan_roots):
        kind = config.classify(path)
        if kind is None:
            continue
        file_path = root / path
        if not file_path.is_file():
            # Tracked in the index but gone from the working tree: a staged
            # deletion. Surfaced by `check` as MISSING rather than crashing here.
            continue
        digest = sha256_of(file_path) if kind is EntryKind.HASHED else None
        entries.append(ManifestEntry(path=path, kind=kind, digest=digest))
    return sorted(entries, key=lambda entry: entry.path)


def render_manifest(entries: Sequence[ManifestEntry], config: ArtifactManifestConfig) -> str:
    """Serialise entries in ``sha256sum -c`` form. Deterministic: no timestamps."""
    lines = [
        "# AlphaGalerkin artifact-freeze manifest (R-05). Do not edit by hand.",
        "# Regenerate ONLY when an artifact is deliberately produced or replaced, via the",
        "# claims-ledger / run-provenance skill:  python -m scripts.artifact_manifest write",
        "# Verify from the repo root:  sha256sum -c " + config.manifest_path,
        "#                        or:  python -m scripts.artifact_manifest check",
        "# Presence-only lines are comments so plain sha256sum still verifies the rest.",
    ]
    for entry in sorted(entries, key=lambda e: e.path):
        if entry.kind is EntryKind.PRESENCE:
            lines.append(f"{config.presence_marker}{entry.path}")
        else:
            lines.append(f"{entry.digest}  {entry.path}")
    return "\n".join(lines) + "\n"


def parse_manifest(text: str, config: ArtifactManifestConfig) -> list[ManifestEntry]:
    """Parse manifest text. Raises :class:`ManifestFormatError` on any line it cannot read.

    Strict on purpose: a line that is neither a comment, a presence marker, nor
    a well-formed hash line is a corrupted manifest, and a lenient parser would
    turn that corruption into a silently *shorter* manifest.
    """
    entries: list[ManifestEntry] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        if line.startswith(config.presence_marker):
            path = _normalise_path(line[len(config.presence_marker) :])
            if not path:
                raise ManifestFormatError(f"line {number}: presence-only line names no path")
            entries.append(ManifestEntry(path=path, kind=EntryKind.PRESENCE))
            continue
        if line.startswith(_COMMENT_PREFIX):
            continue
        match = _HASHED_LINE.match(line)
        if match is None:
            raise ManifestFormatError(f"line {number}: not a sha256sum line: {line!r}")
        entries.append(
            ManifestEntry(
                path=_normalise_path(match.group("path")),
                kind=EntryKind.HASHED,
                digest=match.group("digest"),
            )
        )
    return entries


def compare(
    listed: Sequence[ManifestEntry],
    expected: Sequence[ManifestEntry],
    config: ArtifactManifestConfig,
) -> CheckReport:
    """Diff manifest entries against the tree's expected entries."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in listed:
        if entry.path in seen:
            duplicates.append(entry.path)
        seen.add(entry.path)
    listed_by_path = {entry.path: entry for entry in listed}
    expected_by_path = {entry.path: entry for entry in expected}

    stale: list[str] = []
    wrong_kind: list[str] = []
    missing: list[str] = []
    for path, entry in sorted(listed_by_path.items()):
        actual = expected_by_path.get(path)
        if actual is None:
            missing.append(path)
        elif actual.kind is not entry.kind:
            wrong_kind.append(path)
        elif entry.kind is EntryKind.HASHED and entry.digest != actual.digest:
            stale.append(path)
    unlisted = sorted(path for path in expected_by_path if path not in listed_by_path)

    return CheckReport(
        stale=tuple(stale),
        unlisted=tuple(unlisted),
        missing=tuple(missing),
        wrong_kind=tuple(wrong_kind),
        duplicates=tuple(sorted(set(duplicates))),
        hashed_count=sum(1 for e in listed if e.kind is EntryKind.HASHED),
        presence_count=sum(1 for e in listed if e.kind is EntryKind.PRESENCE),
        min_hashed_entries=config.min_hashed_entries,
    )


def load_config(path: Path | None) -> ArtifactManifestConfig:
    """Defaults, or the validated YAML mapping at ``path`` (``extra="forbid"``)."""
    if path is None:
        return ArtifactManifestConfig()
    # Lazy on purpose: the CI `check` never passes `--config`, and PyYAML is
    # not named in the lint job's install step (see the module docstring).
    import yaml

    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if document is None:
        document = {}
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(document).__name__}")
    return ArtifactManifestConfig.model_validate(document)


def write_manifest(
    config: ArtifactManifestConfig,
    root: Path,
    lister: TrackedLister = git_tracked_files,
) -> CheckReport:
    """Regenerate the manifest from the tree. Refuses a vacuous manifest.

    Returns the report of the freshly written manifest against the tree, which
    is ``ok`` unless the vacuity floor was violated -- in which case nothing is
    written, because a manifest that freezes nothing is worse than none.
    """
    entries = expected_entries(config, root, lister)
    report = compare(entries, entries, config)
    log = _LOG.bind(
        manifest=config.manifest_path,
        hashed=report.hashed_count,
        presence=report.presence_count,
    )
    if report.too_few:
        log.error("artifact_manifest.write_refused", floor=config.min_hashed_entries)
        return report
    target = root / config.manifest_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_manifest(entries, config), encoding="utf-8", newline="\n")
    log.info("artifact_manifest.written")
    return report


def check_manifest(
    config: ArtifactManifestConfig,
    root: Path,
    lister: TrackedLister = git_tracked_files,
) -> CheckReport:
    """Compare the manifest on disk with the tree. Raises on an unreadable manifest."""
    target = root / config.manifest_path
    if not target.is_file():
        raise ManifestFormatError(
            f"{config.manifest_path} does not exist under {root}; "
            f"run `python -m scripts.artifact_manifest write` first"
        )
    listed = parse_manifest(target.read_text(encoding="utf-8"), config)
    report = compare(listed, expected_entries(config, root, lister), config)
    log = _LOG.bind(
        manifest=config.manifest_path,
        hashed=report.hashed_count,
        presence=report.presence_count,
    )
    if report.ok:
        log.info("artifact_manifest.check_ok")
    else:
        log.warning(
            "artifact_manifest.check_mismatch",
            stale=len(report.stale),
            unlisted=len(report.unlisted),
            missing=len(report.missing),
            wrong_kind=len(report.wrong_kind),
            duplicates=len(report.duplicates),
            too_few=report.too_few,
        )
    return report


class _CurrentStderrLoggerFactory:
    """Build a ``PrintLogger`` on the ``sys.stderr`` of the *call*, not of configure time.

    ``structlog.PrintLoggerFactory(file=sys.stderr)`` captures the stream
    object once, when :func:`configure_logging` runs. Anything that swaps
    ``sys.stderr`` afterwards -- pytest's per-test capture, a caller
    redirecting into a log file -- leaves the logger writing to the *old*
    stream; under pytest that stream is closed by the next test, and the
    first ``check`` that logs a mismatch dies with ``I/O operation on closed
    file`` instead of reporting the mismatch. Resolving the stream per call
    (``cache_logger_on_first_use=False`` below makes every ``bind`` call this)
    keeps the logger attached to whatever ``sys.stderr`` currently is.
    """

    def __call__(self, *args: object) -> structlog.PrintLogger:
        return structlog.PrintLogger(file=sys.stderr)


def configure_logging(level: str) -> None:
    """Minimal structlog setup: stderr, level-filtered, no timestamps.

    Local rather than ``src.poc.logging.configure_logging`` because importing
    ``src.poc`` pulls the scenario runner (and torch) into a guard that only
    hashes files; the docs-tier test importing this module should stay hermetic.
    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level {level!r}")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=_CurrentStderrLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.artifact_manifest",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML overriding ArtifactManifestConfig fields (extra keys are rejected).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root the paths are relative to (default: this checkout).",
    )
    parser.add_argument("--log-level", default="INFO", help="structlog level (default INFO).")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write", help="Regenerate the manifest from the tracked tree.")
    subparsers.add_parser("check", help="Verify the tree against the manifest; exit 1 on drift.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. See the module docstring for the exit-code contract."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configure_logging(args.log_level)
        config = load_config(args.config)
        root = Path(args.root).resolve()
        if args.command == "write":
            report = write_manifest(config, root)
        else:
            report = check_manifest(config, root)
    except (OSError, ValueError, ValidationError, RuntimeError) as exc:
        # ManifestFormatError is a ValueError: an unreadable manifest is a usage
        # error (2), not a mismatch (1) -- the check could not run at all.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(report.format(config.manifest_path))
    return EXIT_OK if report.ok else EXIT_MISMATCH


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
