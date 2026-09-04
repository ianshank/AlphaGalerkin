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
    """Patterns from ``.dockerignore``, comments and blanks stripped.

    Deliberately **strict** -- a missing file raises rather than returning
    ``[]``. A Copilot review suggested guarding it "the same way
    ``_copy_sources()`` guards the Dockerfile, so other assertions either
    no-op or skip cleanly"; the no-op half is wrong. With ``[]``,
    ``_is_excluded(source, [])`` returns ``None`` for every source and
    ``test_no_copied_path_is_excluded_by_dockerignore`` goes green on vacuous
    input -- the exact false-pass shape ``test_coverage_gate_integrity.py``
    exists to catch. So the *callers* skip (pointing at ``test_dockerfile_exists``,
    which reports the missing file clearly) and this helper stays honest.
    """
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

    Returns ``[]`` if the Dockerfile is missing rather than raising: this
    function feeds ``@pytest.mark.parametrize`` below, which runs at
    *collection* time -- a ``FileNotFoundError`` here would crash collection
    of this entire file with a traceback, burying the one test that exists to
    report a missing Dockerfile clearly (``test_dockerfile_exists``). An empty
    return just yields zero parametrized cases, which pytest collects fine.
    """
    if not DOCKERFILE.is_file():
        return []
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
    if not DOCKERFILE.is_file():
        pytest.skip("no Dockerfile -- see test_dockerfile_exists for the real failure")
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
    if not DOCKERIGNORE.is_file():
        pytest.skip("no .dockerignore -- see test_dockerfile_exists for the real failure")
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
    if not DOCKERFILE.is_file():
        pytest.skip("no Dockerfile -- see test_dockerfile_exists for the real failure")
    if not DOCKERIGNORE.is_file():
        pytest.skip("no .dockerignore -- see test_dockerfile_exists for the real failure")
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


# ============================================================================ #
# The image must not be told to run a suite whose inputs it does not contain   #
# ============================================================================ #
#
# The clauses above assert that every path the `CMD` *names* survives
# `.dockerignore` and is brought in by a `COPY`. That is necessary and not
# sufficient: `COPY tests/ tests/` brings in **every** suite, including two that
# read repo-root files the image deliberately does not carry.
#
# `tests/claude/` reads `.claude/`, which `.dockerignore` excludes on purpose
# (agent configuration is not runtime material). `tests/docs/` reads `.github/`,
# `CLAUDE.md` and `ARCHITECTURE.md`, none of which is `COPY`ed. Both are
# *present* in the image and would fail there.
#
# Today's `CMD` names neither, so nothing is broken. This guard exists so that
# stays true by construction: adding `tests/docs` to the `CMD` -- a one-word
# edit that reads like broadening coverage -- fails here, with the missing input
# named, instead of failing inside a built image with an import or fixture error.

#: Suites that read repo-root paths, and the paths they read. A suite may only
#: appear in the ``CMD`` if every path listed here reaches the image.
#:
#: Deliberately hand-maintained rather than derived: deriving it would mean
#: importing the suites, which is what this check exists to avoid needing to do.
#: The mapping is small and its entries are asserted to be real directories, so
#: a renamed suite fails loudly rather than dropping out of the check.
SUITE_REPO_ROOT_INPUTS: Final[dict[str, tuple[str, ...]]] = {
    "tests/claude": (".claude",),
    "tests/docs": (".github", "CLAUDE.md", "ARCHITECTURE.md"),
}


def test_the_input_map_names_real_suites() -> None:
    """Vacuity guard: a renamed suite must fail, not silently stop being checked."""
    for suite in SUITE_REPO_ROOT_INPUTS:
        assert (REPO_ROOT / suite).is_dir(), (
            f"{suite} is listed in SUITE_REPO_ROOT_INPUTS but does not exist; "
            f"the entry now checks nothing"
        )


@pytest.mark.parametrize("suite", sorted(SUITE_REPO_ROOT_INPUTS))
def test_the_cmd_does_not_run_a_suite_whose_inputs_are_absent(suite: str) -> None:
    """A suite in the ``CMD`` must have every repo-root input it reads.

    Fails with the *cause* -- the named missing path -- rather than leaving a
    built image to fail with a fixture error, which is the same principle as
    making the two `[fem]` coverage gates fail on the missing extra rather than
    on the coverage number.
    """
    if suite not in _cmd_arguments():
        pytest.skip(f"{suite} is not named in the Dockerfile CMD, so its inputs are not required")

    patterns = _dockerignore_patterns()
    copied = _copy_sources()
    missing = [
        required
        for required in SUITE_REPO_ROOT_INPUTS[suite]
        if _is_excluded(required, patterns) is not None
        or not any(required.rstrip("/") == source.rstrip("/") for source in copied)
    ]
    assert not missing, (
        f"the Dockerfile CMD runs {suite}, which reads {missing} -- and those "
        f"paths are either excluded by .dockerignore or never COPYed, so the "
        f"suite is present in the image but cannot pass there. Either COPY them "
        f"(and un-ignore where needed) or drop {suite} from the CMD."
    )
