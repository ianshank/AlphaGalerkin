"""Every src/ package ships an AGENT.md (hygiene B12).

Root AGENT.md describes per-module documentation as universal. The count
drifted (14 of 26 → 14 of 28) because it was hand-maintained. This guard
compares ``src/*/__init__.py`` to ``src/*/AGENT.md``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

MIN_PACKAGES = 20

#: B19 extras that must appear in getting-started (skip dashboard).
REQUIRED_EXTRAS = (
    "dev",
    "viz",
    "test-extras",
    "fem",
    "jax",
    "picogk",
    "lm-studio",
    "docs",
)

B10_PACKAGES = ("prototyping", "analysis", "curriculum", "tournament")
B10_KEEP_PHRASE = "test-held, not production-wired, not a 2026-07-22-style cut"


def _packages() -> set[str]:
    return {path.parent.name for path in SRC.glob("*/__init__.py")}


def _documented() -> set[str]:
    return {path.parent.name for path in SRC.glob("*/AGENT.md")}


def test_src_has_packages() -> None:
    """Vacuity: an empty glob would make the equality below pass on nothing."""
    assert len(_packages()) >= MIN_PACKAGES


def test_every_src_package_has_agent_md() -> None:
    packages = _packages()
    documented = _documented()
    missing = sorted(packages - documented)
    extra = sorted(documented - packages)
    assert not missing, (
        "src/ packages without AGENT.md (charter B12 / CLAUDE.md Next Steps): " + ", ".join(missing)
    )
    assert not extra, "AGENT.md files whose package has no __init__.py: " + ", ".join(extra)


def test_b10_agent_md_states_the_keep_reason() -> None:
    for name in B10_PACKAGES:
        text = (SRC / name / "AGENT.md").read_text(encoding="utf-8")
        assert B10_KEEP_PHRASE in text, f"src/{name}/AGENT.md missing B10 keep-reason"


def test_b19_extras_are_documented() -> None:
    getting_started = (REPO_ROOT / "docs" / "getting-started.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for extra in REQUIRED_EXTRAS:
        token = f"`{extra}`"
        assert token in getting_started, f"docs/getting-started.md missing {token}"
        assert token in readme, f"README.md missing {token}"
    assert "no `dashboard` extra" in getting_started
    assert "no `dashboard` extra" in readme
