"""Scope-containment guard: frozen tracks must not be co-modified with core work.

An owner decision froze two tracks for this implementation cycle (the codec
model-zoo, and the dashboard + its HuggingFace deploy mirror). A freeze written
only in prose is a suggestion; this makes it a check.

The rule, stated precisely:

* A changeset may touch a frozen track. It may touch the core solver surface.
* It may **not** make a *substantive* change to a frozen track in the same
  changeset as a core change -- because that is what a split attention span
  looks like in a diff.
* "Substantive" is a line budget, not a file count: incidental edits (a version
  shim, an import fix, a tree-wide rename) stay under
  ``incidental_line_budget`` and never trip the gate.

Why a budget rather than an exemption list: an exemption list only ever grows,
and each entry silently narrows the gate. A budget states the actual intent
("feature work is never seven lines") in one auditable number, and it lives in
``config/focus.yaml`` so changing it is a visible scope decision.

Usage::

    python -m scripts.check_focus --base origin/main [--fail-on-violation]
    python -m scripts.check_focus --numstat-file - < numstat.txt

Exit code is non-zero only with ``--fail-on-violation``, so the check can run
report-only first and become blocking later -- the convention this repository
already uses for ``audit_abstractions`` and the transfer-baseline tripwire.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG: Final[Path] = REPO_ROOT / "config" / "focus.yaml"

#: Bumped when the on-disk shape of ``config/focus.yaml`` changes incompatibly.
FOCUS_CONFIG_SCHEMA_VERSION: Final[int] = 1

#: ``git diff`` on a large tree is fast, but a hung subprocess in CI is not
#: distinguishable from a wedged runner. Bound it.
GIT_SUBPROCESS_TIMEOUT_S: Final[float] = 60.0

#: ``git diff --numstat`` writes ``-`` in both count columns for binary files.
_BINARY_MARKER: Final[str] = "-"


class FrozenTrack(BaseModel):
    """One paused workstream and the paths that constitute it."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str = Field(min_length=1, description="Stable identifier, also used in docs/FOCUS.md.")
    reason: str = Field(min_length=1, description="Why this track is paused. Never empty.")
    paths: tuple[str, ...] = Field(
        min_length=1,
        description="Repo-relative paths; a trailing '/' means directory prefix.",
    )

    @field_validator("paths", mode="after")
    @classmethod
    def _normalise_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_normalise_path(entry) for entry in value)


class FocusConfig(BaseModel):
    """Validated ``config/focus.yaml``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: int = Field(default=FOCUS_CONFIG_SCHEMA_VERSION, ge=1)
    incidental_line_budget: int = Field(
        ge=0,
        description="Changed lines a single frozen track may carry before counting as substantive.",
    )
    frozen_tracks: tuple[FrozenTrack, ...] = Field(min_length=1)
    core_paths: tuple[str, ...] = Field(min_length=1)

    @field_validator("core_paths", mode="after")
    @classmethod
    def _normalise_core(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_normalise_path(entry) for entry in value)

    @model_validator(mode="after")
    def _tracks_and_core_are_disjoint(self) -> FocusConfig:
        """A path that is both frozen and core would make the gate incoherent.

        Not a stylistic check: the classifier would count the same file on both
        sides and report a violation against a single-file diff.
        """
        if self.schema_version > FOCUS_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"focus config schema_version {self.schema_version} is newer than this "
                f"tool understands ({FOCUS_CONFIG_SCHEMA_VERSION}); upgrade the tool"
            )
        seen: dict[str, str] = {}
        for track in self.frozen_tracks:
            for path in track.paths:
                if path in seen:
                    raise ValueError(
                        f"path {path!r} claimed by both {seen[path]!r} and {track.name!r}"
                    )
                seen[path] = track.name
        for core in self.core_paths:
            for frozen, owner in seen.items():
                if _prefix_overlaps(core, frozen):
                    raise ValueError(
                        f"core path {core!r} overlaps frozen path {frozen!r} (track {owner!r})"
                    )
        names = [track.name for track in self.frozen_tracks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate frozen-track names: {names}")
        return self


@dataclass(frozen=True)
class ChangedFile:
    """One entry of ``git diff --numstat``."""

    path: str
    lines_changed: int


@dataclass(frozen=True)
class FocusReport:
    """The classification of one changeset against a :class:`FocusConfig`."""

    budget: int
    core_hits: tuple[ChangedFile, ...]
    frozen_hits: tuple[tuple[str, tuple[ChangedFile, ...]], ...]

    @property
    def frozen_lines(self) -> dict[str, int]:
        return {name: sum(f.lines_changed for f in files) for name, files in self.frozen_hits}

    @property
    def substantive_tracks(self) -> tuple[str, ...]:
        return tuple(sorted(n for n, total in self.frozen_lines.items() if total > self.budget))

    @property
    def violated(self) -> bool:
        return bool(self.core_hits) and bool(self.substantive_tracks)


def _normalise_path(entry: str) -> str:
    """Strip the spellings that would silently stop a prefix from matching."""
    cleaned = entry.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def _prefix_overlaps(left: str, right: str) -> bool:
    """True when either path is a directory prefix of the other, or they are equal."""
    if left == right:
        return True
    if left.endswith("/") and right.startswith(left):
        return True
    return right.endswith("/") and left.startswith(right)


def _matches(path: str, pattern: str) -> bool:
    return path.startswith(pattern) if pattern.endswith("/") else path == pattern


def load_focus_config(path: Path = DEFAULT_CONFIG) -> FocusConfig:
    """Load and validate the focus configuration."""
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(document).__name__}")
    return FocusConfig.model_validate(document)


def parse_numstat(text: str) -> tuple[ChangedFile, ...]:
    """Parse ``git diff --numstat`` output.

    Binary files report ``-`` for both counts; they are kept (the path still
    matters for classification) with a zero line count, which is the honest
    reading -- a binary blob has no line delta to budget against.
    """
    files: list[ChangedFile] = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added, deleted, path = parts[0], parts[1], "\t".join(parts[2:])
        if added == _BINARY_MARKER or deleted == _BINARY_MARKER:
            count = 0
        else:
            try:
                count = int(added) + int(deleted)
            except ValueError:
                continue
        files.append(ChangedFile(path=_normalise_path(path), lines_changed=count))
    return tuple(files)


def collect_changed_files(
    base: str, head: str = "HEAD", cwd: Path = REPO_ROOT
) -> tuple[ChangedFile, ...]:
    """Run ``git diff --numstat <base>...<head>`` and parse it.

    Three-dot form on purpose: a pull request's diff is against the merge base,
    not against the tip of the target branch, so two-dot would attribute every
    commit landed on the base since the branch forked to this changeset.
    """
    completed = subprocess.run(
        ["git", "diff", "--numstat", f"{base}...{head}"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=GIT_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git diff {base}...{head} failed (exit {completed.returncode}): "
            f"{completed.stderr.strip() or '<no stderr>'}"
        )
    return parse_numstat(completed.stdout)


def classify(config: FocusConfig, files: Sequence[ChangedFile]) -> FocusReport:
    """Split a changeset into core hits and per-track frozen hits."""
    core_hits = tuple(f for f in files if any(_matches(f.path, p) for p in config.core_paths))
    frozen_hits: list[tuple[str, tuple[ChangedFile, ...]]] = []
    for track in config.frozen_tracks:
        hits = tuple(f for f in files if any(_matches(f.path, p) for p in track.paths))
        if hits:
            frozen_hits.append((track.name, hits))
    return FocusReport(
        budget=config.incidental_line_budget,
        core_hits=core_hits,
        frozen_hits=tuple(frozen_hits),
    )


def format_report(report: FocusReport, config: FocusConfig) -> str:
    """Render a human-readable summary. Always explains *why*, not just pass/fail."""
    lines: list[str] = ["=== Focus / scope-containment check ==="]
    lines.append(f"incidental line budget: {report.budget}")
    lines.append(f"core files changed:     {len(report.core_hits)}")
    for path in sorted(f.path for f in report.core_hits):
        lines.append(f"  core   {path}")
    if not report.frozen_hits:
        lines.append("frozen tracks touched:  none")
    reasons = {track.name: track.reason for track in config.frozen_tracks}
    for name, hits in report.frozen_hits:
        total = sum(f.lines_changed for f in hits)
        verdict = "SUBSTANTIVE" if total > report.budget else "incidental"
        lines.append(
            f"frozen track {name!r}: {total} changed lines across {len(hits)} file(s) [{verdict}]"
        )
        for hit in sorted(hits, key=lambda f: f.path):
            lines.append(f"  frozen {hit.path} ({hit.lines_changed} lines)")
    if report.violated:
        lines.append("")
        lines.append(
            "VIOLATION: this changeset mixes core work with substantive frozen-track work."
        )
        for name in report.substantive_tracks:
            lines.append(f"  - {name}: {reasons.get(name, '<no reason recorded>')}")
        lines.append("")
        lines.append("Split it into two changesets, or -- if the coupling is real -- record why in")
        lines.append("the pull request and add the `focus-override` label, which skips this check.")
    else:
        lines.append("")
        lines.append("OK: no substantive frozen-track change alongside core work.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to focus.yaml.")
    parser.add_argument("--base", default=None, help="Base ref for the diff (e.g. origin/main).")
    parser.add_argument("--head", default="HEAD", help="Head ref for the diff.")
    parser.add_argument(
        "--numstat-file",
        type=str,
        default=None,
        help="Read `git diff --numstat` output from a file instead of running git ('-' = stdin).",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit non-zero when a violation is found (otherwise report only).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check. Returns 1 only on a violation *and* ``--fail-on-violation``."""
    args = build_parser().parse_args(argv)
    config = load_focus_config(args.config)

    if args.numstat_file is not None:
        text = (
            sys.stdin.read()
            if args.numstat_file == "-"
            else Path(args.numstat_file).read_text(encoding="utf-8")
        )
        files = parse_numstat(text)
    elif args.base is not None:
        files = collect_changed_files(args.base, args.head)
    else:
        build_parser().error("one of --base or --numstat-file is required")
        raise AssertionError("unreachable")  # pragma: no cover - argparse exits

    report = classify(config, files)
    print(format_report(report, config))
    return 1 if (report.violated and args.fail_on_violation) else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
