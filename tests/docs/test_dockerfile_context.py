"""Hermetic guard: the Dockerfile's build context is not undercut by ``.dockerignore``.

``docker/Dockerfile`` has existed since ``61c1e93`` (2026-08-16) and is built by
**nothing** -- no CI job, no Makefile target until 2026-09-02, no test. It was
also, until 2026-09-02, described by ``CLAUDE.md``'s Next Steps as not existing
at all ("no ``Dockerfile`` or ``docker-compose.yml`` anywhere in the tree,
verified 2026-08-21" -- five days *after* it landed). Untested, unbuilt, and
misdescribed is how a `.dockerignore` line silently breaks an image.

This file does not need a Docker daemon and does not build anything. It parses
both files and asserts the one property that is cheap to check and expensive to
discover in a broken image: **every path the Dockerfile copies in must survive
the ignore file**, and every path its default command runs must be one of them.

Verified once by hand on 2026-09-02 by materialising the real context (git-tracked
files minus ``.dockerignore``) and running the ``CMD`` inside it: 510 passed,
6 skipped. This test is what keeps that true without repeating the experiment.
"""

from __future__ import annotations

import fnmatch
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"

#: Sources a ``COPY`` may name that are not repo paths (build-stage artefacts).
#: Empty today; present so a future ``COPY --from=builder /usr/local/lib ...``
#: is an explicit decision rather than a silent skip.
_NON_CONTEXT_SOURCES: frozenset[str] = frozenset()


def _dockerignore_patterns() -> list[str]:
    return [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _is_excluded(path: str, patterns: list[str]) -> str | None:
    """Return the ``.dockerignore`` pattern that drops ``path``, or ``None``.

    Docker matches a pattern against the whole path *and* against each path
    component, so ``__pycache__/`` excludes ``src/foo/__pycache__``. Both
    directions are checked here for the same reason ``_is_omitted`` in
    ``test_coverage_gate_integrity.py`` checks both: a one-directional match
    passes on the case that actually bites.
    """
    parts = path.rstrip("/").split("/")
    for pattern in patterns:
        bare = pattern.rstrip("/").lstrip("./")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            if fnmatch.fnmatch(prefix, bare) or fnmatch.fnmatch(parts[i - 1], bare):
                return pattern
    return None


def _copy_sources() -> list[str]:
    """Every context path named as a source of a ``COPY`` instruction.

    Handles line continuations, ``--chown=``/``--from=`` flags, and the
    "last token is the destination" rule.
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    # Join continuations before parsing: a COPY split across lines otherwise
    # parses as a COPY with no destination and silently contributes nothing.
    joined = text.replace("\\\n", " ")
    sources: list[str] = []
    for raw in joined.splitlines():
        line = raw.strip()
        if not line.upper().startswith("COPY "):
            continue
        tokens = [t for t in shlex.split(line)[1:] if not t.startswith("--")]
        if len(tokens) < 2:  # pragma: no cover - malformed Dockerfile
            pytest.fail(f"COPY with no destination: {line!r}")
        sources.extend(tokens[:-1])
    return sources


def _cmd_arguments() -> list[str]:
    """The tokens of the Dockerfile's ``CMD``, JSON-array form."""
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.upper().startswith("CMD "):
            body = line[4:].strip()
            if body.startswith("["):
                return [t.strip().strip('"').strip("'") for t in body[1:-1].split(",")]
            return shlex.split(body)
    return []


def test_dockerfile_exists() -> None:
    """Pins the fact ``CLAUDE.md`` got wrong for two weeks.

    Not a tautology: if the Dockerfile is ever deleted, every other test in this
    file would vacuously pass (no COPY lines to check), and the Makefile target
    and CLAUDE.md row would be left pointing at nothing.
    """
    assert DOCKERFILE.is_file(), f"{DOCKERFILE} is missing"
    assert DOCKERIGNORE.is_file(), f"{DOCKERIGNORE} is missing"


def test_the_dockerfile_actually_copies_something() -> None:
    """Guards this file against vacuity, the way the import contracts do.

    A Dockerfile whose COPY lines stopped parsing (a syntax change, a new flag
    form) would make every assertion below iterate an empty list and pass.
    """
    assert len(_copy_sources()) >= 2


@pytest.mark.parametrize("source", _copy_sources())
def test_no_copied_path_is_excluded_by_dockerignore(source: str) -> None:
    """A COPY of an ignored path fails the build -- or worse, copies nothing.

    This is the failure the file exists for: adding ``docs/`` or ``config/`` to
    ``.dockerignore`` is a one-line diff that reads like housekeeping and breaks
    the image, and nothing else in this repo would notice.
    """
    if source in _NON_CONTEXT_SOURCES:
        pytest.skip(f"{source} is a build-stage artefact, not a context path")
    pattern = _is_excluded(source, _dockerignore_patterns())
    assert pattern is None, (
        f"docker/Dockerfile copies {source!r}, but .dockerignore excludes it via "
        f"{pattern!r}. The build would fail or silently produce an image missing "
        f"that path."
    )


@pytest.mark.parametrize("source", _copy_sources())
def test_every_copied_path_exists(source: str) -> None:
    """A COPY of a path that no longer exists fails the build at that line."""
    if source in _NON_CONTEXT_SOURCES:
        pytest.skip(f"{source} is a build-stage artefact, not a context path")
    assert (REPO_ROOT / source).exists(), f"docker/Dockerfile copies missing path {source!r}"


def test_default_command_runs_paths_that_are_in_the_image() -> None:
    """The ``CMD``'s test directories must be copied *and* not ignored.

    ``CMD ["pytest", "tests/sanity", ...]`` is only meaningful if those trees
    reach the image. They arrive via ``COPY tests/ tests/``, so this asserts the
    two halves agree rather than trusting that they do.
    """
    patterns = _dockerignore_patterns()
    copied = _copy_sources()
    path_args = [token for token in _cmd_arguments() if "/" in token and not token.startswith("-")]
    assert path_args, "the Dockerfile CMD names no paths; update this guard if that is intended"
    for arg in path_args:
        assert (REPO_ROOT / arg).exists(), f"CMD runs {arg!r}, which does not exist in the repo"
        assert _is_excluded(arg, patterns) is None, f"CMD runs {arg!r}, excluded by .dockerignore"
        assert any(arg == c or arg.startswith(c.rstrip("/") + "/") for c in copied), (
            f"CMD runs {arg!r}, but no COPY brings it into the image"
        )
