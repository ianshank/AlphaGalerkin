"""Run provenance sidecars for committed benchmark artifacts.

The project charter requires every numeric headline claim to cite a committed
artifact. It does not require the artifact to say *how it was produced*, and the
gap is not theoretical: ``results/lshape_mcts_vs_dorfler.csv`` carries exactly
one provenance column, ``seed``. Not the search mode, not the marking fraction,
not a git SHA. You cannot tell from that file whether it predates or postdates
the 2026-08-16 single-agent backup fix -- and the same harness still exposes a
``legacy_adversarial`` mode that reproduces the retracted number.

A :class:`RunManifest` is written alongside an artifact as
``<artifact-stem>.run.json`` and records what the CSV cannot: the config hash,
the git SHA and dirty flag, package versions, resolved seeds, per-arm parameters
and counters, and the thresholds that were actually gated.

Schema versioning follows ``src/poc/baselines``: an integer module constant,
``extra="ignore"`` for forward compatibility, and an explicit migration function
with a documented table.

Design note, load-bearing: :func:`collect_git_provenance` and
:func:`collect_package_versions` **never raise**. A provenance collector that
throws inside a benchmark destroys the run it exists to document, so every
failure degrades to a default-populated object and the fields say ``unknown``
rather than the call propagating.
"""

from __future__ import annotations

import json
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Any, Final

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)

#: Current manifest schema. Bump when a field's *meaning* changes; additive
#: fields do not need a bump because readers use ``extra="ignore"``.
RUN_MANIFEST_SCHEMA_VERSION: Final[int] = 1

#: Wall-clock ceiling for each ``git`` subprocess. Provenance collection must not
#: hang a benchmark on a slow or wedged repository.
GIT_SUBPROCESS_TIMEOUT_S: Final[float] = 5.0

#: Value recorded when a field genuinely cannot be determined. Preferred over
#: omitting the field or guessing: "we do not know" is itself provenance.
UNKNOWN: Final[str] = "unknown"

#: Architecture label when ``platform.machine()`` returns an empty string.
#: Distinct from :data:`UNKNOWN`, which means "the whole probe failed" -- a
#: reader must be able to tell "we could not identify the CPU architecture" from
#: "we collected nothing at all".
UNKNOWN_ARCH: Final[str] = "unknown-arch"

#: Which CUDA device the tag names. Device 0 by convention; the device *count*
#: is recorded alongside it so a multi-GPU host is not mistaken for a single-card
#: one. Named rather than inlined so a caller reading the tag knows which card
#: the name refers to.
CUDA_PROBE_DEVICE_INDEX: Final[int] = 0

#: Fields excluded from :meth:`RunManifest.stable_fields`, because they change
#: between two runs of identical code and would make any comparison flaky.
_VOLATILE_FIELDS: Final[frozenset[str]] = frozenset({"created_at_utc", "hardware_tag", "arms"})

#: Distributions recorded on every manifest. Absent ones are recorded as ``None``
#: rather than omitted, so "not installed" is distinguishable from "not checked".
_TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "numpy",
    "scipy",
    "torch",
    "pydantic",
    "scikit-fem",
    "matplotlib",
)


class GitProvenance(BaseModel):
    """Repository state at run time. Every field degrades to a default."""

    model_config = ConfigDict(extra="ignore")

    sha: str = Field(default=UNKNOWN, description="Full commit SHA of HEAD.")
    branch: str = Field(default=UNKNOWN, description="Abbreviated branch name.")
    dirty: bool | None = Field(
        default=None,
        description=(
            "True when the working tree had uncommitted changes. None means the "
            "question could not be answered (not a git worktree, or git absent) "
            "-- deliberately distinct from False."
        ),
    )


class PackageVersions(BaseModel):
    """Installed versions of the distributions a result can depend on."""

    model_config = ConfigDict(extra="ignore")

    python: str = Field(default=UNKNOWN, description="CPython version.")
    packages: dict[str, str | None] = Field(
        default_factory=dict,
        description="Distribution name to version; None when not installed.",
    )


class ArmProvenance(BaseModel):
    """One experimental arm: what it was, and what it cost."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Arm identifier as written in the artifact.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Resolved arm parameters.")
    counters: dict[str, float] = Field(
        default_factory=dict,
        description="Compute counters (solves, wall seconds, evaluator calls…).",
    )


class RunManifest(BaseModel):
    """Everything a committed artifact cannot say about itself."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(
        default=RUN_MANIFEST_SCHEMA_VERSION,
        ge=1,
        description="Manifest schema version.",
    )
    run_id: str = Field(description="Identifier for this run.")
    created_at_utc: str = Field(default=UNKNOWN, description="RFC3339 UTC timestamp, or 'unknown'.")
    harness: str = Field(description="Dotted module path of the code that produced the artifact.")
    config_hash: str = Field(
        default=UNKNOWN,
        description="Hash of the validated config (see BaseModuleConfig.compute_hash).",
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Full resolved config, JSON-safe."
    )
    git: GitProvenance = Field(default_factory=GitProvenance)
    packages: PackageVersions = Field(default_factory=PackageVersions)
    hardware_tag: str = Field(default=UNKNOWN, description="Free-form host/accelerator label.")
    seeds: list[int] = Field(default_factory=list, description="Resolved seeds.")
    arms: list[ArmProvenance] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(
        default_factory=dict, description="Role to repo-relative path."
    )
    notes: str = Field(
        default="",
        description="Free text for what the numbers do and do not establish.",
    )

    def stable_fields(self) -> dict[str, Any]:
        """The subset a golden comparison may assert on.

        Excludes timestamps, hardware labels and per-arm counters -- everything
        that differs between two runs of identical code on different machines.
        """
        return {
            key: value
            for key, value in self.model_dump(mode="json").items()
            if key not in _VOLATILE_FIELDS
        }


def _git(args: list[str], repo_root: Path) -> str | None:
    """Run a git command, returning stripped stdout or None on any failure."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=GIT_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def collect_git_provenance(repo_root: Path | None = None) -> GitProvenance:
    """Best-effort repository state. Never raises.

    Args:
        repo_root: Directory to inspect. Defaults to this file's repository.

    Returns:
        A :class:`GitProvenance`; fields are ``unknown``/``None`` when the
        question could not be answered.

    """
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    sha = _git(["rev-parse", "HEAD"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _git(["status", "--porcelain"], root)
    return GitProvenance(
        sha=sha or UNKNOWN,
        branch=branch or UNKNOWN,
        dirty=None if status is None else bool(status),
    )


def collect_package_versions() -> PackageVersions:
    """Installed versions of the tracked distributions. Never raises."""
    import sys

    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = _package_version(name)
        except (PackageNotFoundError, ValueError, OSError):
            versions[name] = None
    return PackageVersions(
        python=".".join(str(part) for part in sys.version_info[:3]),
        packages=versions,
    )


def migrate_run_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a manifest document to the current schema.

    Migration table::

        +---------------+----+------------------------------+
        | from          | to | change                       |
        +===============+====+==============================+
        | (unversioned) | 1  | add ``schema_version`` field |
        +---------------+----+------------------------------+

    Args:
        raw: Parsed manifest document. Not mutated.

    Returns:
        A new document at :data:`RUN_MANIFEST_SCHEMA_VERSION`.

    Raises:
        ValueError: If the document is newer than this code understands.

    """
    document = dict(raw)
    version = int(document.get("schema_version", 0))
    if version > RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"run manifest schema version {version} is newer than this code "
            f"understands ({RUN_MANIFEST_SCHEMA_VERSION}); upgrade AlphaGalerkin"
        )
    if version < 1:
        document["schema_version"] = RUN_MANIFEST_SCHEMA_VERSION
    return document


def write_run_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Write ``manifest`` as JSON, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_run_manifest(path: str | Path) -> RunManifest:
    """Read a manifest, migrating it to the current schema first."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunManifest.model_validate(migrate_run_manifest(raw))


def manifest_path_for(artifact: str | Path) -> Path:
    """The sidecar path for ``artifact``: ``<stem>.run.json`` beside it."""
    target = Path(artifact)
    return target.with_suffix(".run.json")


def collect_hardware_tag() -> str:
    """Best-effort host/accelerator label for a run manifest. Never raises.

    ``RunManifest.hardware_tag`` defaults to ``UNKNOWN`` and every committed
    sidecar carried that default, because no harness ever set it. A provenance
    record that cannot say what the numbers were measured on is missing the one
    field a reader needs to judge whether a wall-clock or accelerator-sensitive
    result transfers to their machine.

    The tag is deliberately free-form and deliberately in ``_VOLATILE_FIELDS``
    (excluded from ``stable_fields``): it changes with the host, so a golden
    comparison must not assert on it.

    Returns:
        ``"<machine>-<n>cpu"``, plus ``-<k>x<device name>`` when ``k`` CUDA
        devices are visible, else :data:`UNKNOWN` if even the platform probe
        fails.

    """
    import os
    import platform

    try:
        machine = platform.machine() or UNKNOWN_ARCH
        # os.cpu_count() can return None on exotic platforms.
        n_cpu = os.cpu_count() or 0
        tag = f"{machine}-{n_cpu}cpu"
    except OSError:
        logger.debug("hardware_tag_platform_probe_failed", exc_info=True)
        return UNKNOWN

    try:
        import torch

        if torch.cuda.is_available():
            # Device *count* as well as name: the reference rig is dual-GPU
            # (cuda:0 RTX 5060 Ti + cuda:1 RTX 5060), and a tag naming only
            # device 0 would record a single-card host for a two-card sweep.
            n_cuda = torch.cuda.device_count()
            tag = f"{tag}-{n_cuda}x{torch.cuda.get_device_name(CUDA_PROBE_DEVICE_INDEX)}"
    # Broad by intent: a provenance label must never be the reason a benchmark
    # run fails. A missing/broken torch, a driver mismatch, or a CUDA call that
    # raises all degrade to the CPU-only tag rather than aborting the harness.
    # Logged at debug so "why does the sidecar say CPU on the GPU rig?" is
    # answerable without reproducing the run.
    except Exception:
        logger.debug("hardware_tag_cuda_probe_failed", cpu_tag=tag, exc_info=True)

    return tag


__all__ = [
    "GIT_SUBPROCESS_TIMEOUT_S",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "UNKNOWN",
    "ArmProvenance",
    "GitProvenance",
    "PackageVersions",
    "RunManifest",
    "collect_git_provenance",
    "collect_hardware_tag",
    "collect_package_versions",
    "load_run_manifest",
    "manifest_path_for",
    "migrate_run_manifest",
    "write_run_manifest",
]
