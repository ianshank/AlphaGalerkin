"""No file may declare a version string that disagrees with ``pyproject.toml``.

``src/__init__.py`` already reads the installed distribution's metadata, so the
importable ``__version__`` cannot drift. Nothing enforced that for the *other*
places a version is written down, and they had drifted: ``README.md`` declared
the version twice with two different values (``0.4.0-dev`` under
``## Project status`` and ``0.1.0`` in an orphaned duplicate of that paragraph
stranded at the end of the roadmap section), ``src/templates/cli.py`` defaulted
``create_cli_app(version=...)`` to a hardcoded ``"0.1.0"`` that every CLI built
on it inherited, ``src/tools/gtp.py`` reported ``"0.1.0"`` to GTP controllers,
and ``hf_space/src/__init__.py`` hardcoded the same stale literal.

Version strings are compared after PEP 440 normalisation, because
``importlib.metadata`` reports ``0.4.0.dev0`` for a ``pyproject.toml`` that
reads ``0.4.0-dev`` -- they are the same version, and a guard that could not
see that would be unusable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest
import tomllib

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
PYPROJECT: Final[Path] = REPO_ROOT / "pyproject.toml"

#: Files scanned for version declarations. Deliberately narrow: these are the
#: places a version is *declared*, not every file that happens to contain a
#: version-shaped string (a dependency pin is not a project version).
SCANNED_FILES: Final[tuple[str, ...]] = (
    "README.md",
    "hf_space/src/__init__.py",
)

#: ``x.y.z`` optionally followed by a pre/dev suffix, inside backticks (Markdown)
#: or quotes (Python). Anchored on the delimiters so dependency pins such as
#: ``">=9.0,<13"`` and prose like "Python 3.10" cannot match.
_VERSION_IN_BACKTICKS: Final[re.Pattern[str]] = re.compile(
    r"`(\d+\.\d+\.\d+(?:[.-]?(?:dev|a|b|rc|alpha|beta)\d*)?)`"
)
_DUNDER_VERSION: Final[re.Pattern[str]] = re.compile(
    r"^__version__\s*=\s*[\"'](.+?)[\"']", re.MULTILINE
)


def _normalise(version: str) -> str:
    """Collapse PEP 440 spellings so ``0.4.0-dev`` == ``0.4.0.dev0``."""
    normalised = version.strip().lower().replace("-", ".").replace("_", ".")
    normalised = re.sub(r"\.(dev|a|b|rc|alpha|beta)$", r".\g<1>0", normalised)
    return normalised


def _project_version() -> str:
    with PYPROJECT.open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _declared_versions(path: Path) -> list[tuple[int, str]]:
    """``(line_number, version)`` for every version declaration in ``path``."""
    text = path.read_text(encoding="utf-8")
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        for match in _VERSION_IN_BACKTICKS.finditer(line):
            found.append((number, match.group(1)))
    for match in _DUNDER_VERSION.finditer(text):
        line_number = text[: match.start()].count("\n") + 1
        found.append((line_number, match.group(1)))
    return found


class TestVersionConsistency:
    def test_pyproject_declares_a_version(self) -> None:
        assert _project_version(), "pyproject.toml [project].version is empty"

    def test_importable_version_matches_pyproject(self) -> None:
        from src import __version__

        assert _normalise(__version__) == _normalise(_project_version()), (
            f"src.__version__ ({__version__!r}) disagrees with pyproject.toml "
            f"({_project_version()!r}); is the package installed with -e?"
        )

    @pytest.mark.parametrize("relative_path", SCANNED_FILES)
    def test_declared_versions_match_pyproject(self, relative_path: str) -> None:
        path = REPO_ROOT / relative_path
        assert path.exists(), f"{relative_path} does not exist"
        expected = _normalise(_project_version())
        offenders = [
            f"{relative_path}:{number}: {declared!r}"
            for number, declared in _declared_versions(path)
            if _normalise(declared) != expected
        ]
        assert not offenders, (
            f"version string(s) disagreeing with pyproject.toml ({_project_version()!r}):\n  "
            + "\n  ".join(offenders)
            + "\n\nSingle-source it (see src/__init__.py) or update the literal."
        )

    def test_scan_actually_finds_declarations(self) -> None:
        """A scanner that matches nothing passes everything."""
        for relative_path in SCANNED_FILES:
            found = _declared_versions(REPO_ROOT / relative_path)
            assert found, (
                f"{relative_path} is scanned for version strings but the pattern "
                "matched none -- the guard would pass no matter what it contained"
            )


class TestNoHardcodedVersionsInCode:
    """The CLI/GTP version reporters must resolve, not hardcode."""

    def test_create_cli_app_defaults_to_package_version(self) -> None:
        import inspect

        from src.templates.cli import create_cli_app

        signature = inspect.signature(create_cli_app)
        default = signature.parameters["version"].default
        assert default is None, (
            "create_cli_app's version default must be None (resolved to the package "
            f"version at call time), not the literal {default!r} -- a hardcoded default "
            "is inherited by every CLI built on it and outlived the 0.4.0-dev bump"
        )

    def test_gtp_reports_the_package_version(self) -> None:
        from src import __version__
        from src.tools.gtp import GTPEngine

        engine = GTPEngine()
        assert __version__ in engine.process_command("version")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
