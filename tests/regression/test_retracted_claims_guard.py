"""Retracted claims must not be stated live in the outward-facing documents.

Three guards already police retracted claims, and between them they leave the
highest-stakes surface uncovered:

* ``tests/docs/test_charter_alignment.py`` scans the charter itself.
* ``tests/regression/test_related_work_guard.py`` scans ``docs/related-work.md``
  and ``README.md``.
* the charter's *UI Claim Fidelity* guard scans ``dashboard/`` and ``hf_space/``.

**None scans the outward-facing SBIR material under ``docs/business/``.** On
2026-08-23 an audit found the retracted "MCTS beats Doerfler at matched DOF"
headline stated live in ``docs/business/proposals/PRIOR_ART_REVIEW.md`` -- the
2026-07-22 transfer correction had been propagated to those documents via an
enumerated nine-file sweep, but the later 2026-08-16 AMR retraction never was.
A guard, not another manual sweep, is the fix.

Escape hatches, both deliberate:

* a line may *describe* a retraction if it carries a marker word -- the same
  convention ``test_charter_alignment.py`` uses, so a correction note reads as a
  correction rather than tripping the guard;
* a file may be exempted wholesale via :data:`EXEMPTIONS`, which is asserted to
  stay *needed* -- a stale exemption fails the suite rather than rotting. This
  mirrors ``tests/claude/test_harness_validation.py``'s ``FORWARD_REFERENCES``,
  the repo's proven pattern for a deliberate exception that must not decay.

``docs/archive/`` is deliberately out of scope: archived PR reviews legitimately
quote the fabricated figure under a banner, and a guard that reverts on false
positives is worse than no guard -- the lesson of ``check_doc_links.py``'s
inline-span attempt (105 false positives across 21 files).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.support.cut_modules import (
    FABRICATED_FIGURE,
    RETRACTED_AMR_WIN_PHRASES,
    RETRACTED_BLANKET_CLAIM,
    RETRACTED_UNIFORMLY_SINGLE_STEP,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: Roots scanned for live retracted claims. ``docs/archive/`` is excluded by
#: construction (see the module docstring).
SCAN_ROOTS: Final[tuple[Path, ...]] = (
    REPO_ROOT / "docs" / "business",
    REPO_ROOT / "docs" / "doe_genesis",
)

#: Single files scanned alongside the roots.
SCAN_FILES: Final[tuple[Path, ...]] = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "CLAUDE.md",
)

#: A *block* carrying any of these may state a retracted claim, because it is
#: describing the retraction. The charter's own guard matches markers per line;
#: that granularity is wrong for prose, where a correction note is inherently a
#: multi-line blockquote and the marker word cannot appear on every line. Blocks
#: are runs of contiguous non-blank lines.
RETRACTION_MARKERS: Final[tuple[str, ...]] = (
    "fabricat",
    "retract",
    "correct",
    "must not",
    "no longer",
    "inadmissible",
    "previously",
    "do not overclaim",
)

#: Repo-relative path -> why the whole file is exempt. Every entry is asserted
#: to still be needed, so a stale exemption fails rather than silently rotting.
EXEMPTIONS: Final[dict[str, str]] = {}

#: The claim strings this guard policices, as ``(label, needle)`` pairs. Needles
#: are matched case-insensitively.
GUARDED_CLAIMS: Final[tuple[tuple[str, str], ...]] = (
    ("fabricated transfer figure", FABRICATED_FIGURE),
    ("retracted blanket MCTS+Galerkin claim", RETRACTED_BLANKET_CLAIM),
    ("retracted 'uniformly single-step' claim", RETRACTED_UNIFORMLY_SINGLE_STEP),
    *(("retracted AMR win headline", phrase) for phrase in RETRACTED_AMR_WIN_PHRASES),
)


def _scanned_files() -> list[Path]:
    """Every markdown file this guard covers, exemptions removed."""
    found: list[Path] = []
    for root in SCAN_ROOTS:
        if root.exists():
            found.extend(sorted(root.rglob("*.md")))
    found.extend(path for path in SCAN_FILES if path.exists())
    return [path for path in found if path.relative_to(REPO_ROOT).as_posix() not in EXEMPTIONS]


def _blocks(text: str) -> list[tuple[int, list[str]]]:
    """Split ``text`` into ``(first_line_number, lines)`` runs of non-blank lines."""
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start = 1
    for number, line in enumerate(text.splitlines(), 1):
        if line.strip():
            if not current:
                start = number
            current.append(line)
        elif current:
            blocks.append((start, current))
            current = []
    if current:
        blocks.append((start, current))
    return blocks


def _live_mentions(path: Path, needle: str) -> list[tuple[int, str]]:
    """Lines stating ``needle`` inside a block that carries no retraction marker."""
    found: list[tuple[int, str]] = []
    for start, lines in _blocks(path.read_text(encoding="utf-8")):
        block = "\n".join(lines).lower()
        if any(marker in block for marker in RETRACTION_MARKERS):
            continue
        for offset, line in enumerate(lines):
            if needle.lower() in line.lower():
                found.append((start + offset, line.strip()))
    return found


class TestNoLiveRetractedClaims:
    def test_scan_covers_the_outward_facing_documents(self) -> None:
        """The guard must actually reach docs/business — an empty scan passes everything."""
        scanned = _scanned_files()
        assert scanned, "the retracted-claims guard scanned zero files"
        business = [path for path in scanned if "docs/business" in path.as_posix()]
        assert len(business) >= 10, (
            "docs/business is the surface this guard exists for; only "
            f"{len(business)} file(s) were scanned"
        )

    @pytest.mark.parametrize(("label", "needle"), GUARDED_CLAIMS)
    def test_claim_is_not_stated_live(self, label: str, needle: str) -> None:
        offenders: list[str] = []
        for path in _scanned_files():
            for number, line in _live_mentions(path, needle):
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders.append(f"{rel}:{number}: {line[:160]}")
        assert not offenders, (
            f"{label} ({needle!r}) is stated live, without a retraction marker, in:\n  "
            + "\n  ".join(offenders)
            + "\n\nEither correct the claim or mark the line as describing a retraction. "
            "The committed L-shape result is that MCTS loses at matched DOF (1.0996) and "
            "at matched compute (2.04); see openspec/specs/project-charter/spec.md."
        )


class TestExemptionsStayHonest:
    """A stale exemption is a silent hole; assert every one is still load-bearing.

    ``EXEMPTIONS`` is empty today, and these tests are what keep it that way
    honestly: an entry added later must name a real reason and must still be
    needed, so it fails rather than rotting into a permanent blind spot.
    """

    def test_every_exemption_is_still_needed(self) -> None:
        stale = []
        for relative_path in sorted(EXEMPTIONS):
            path = REPO_ROOT / relative_path
            assert path.exists(), f"exempted file {relative_path} no longer exists"
            if not any(_live_mentions(path, needle) for _, needle in GUARDED_CLAIMS):
                stale.append(relative_path)
        assert not stale, (
            f"exempted from the retracted-claims guard but no longer containing any "
            f"unmarked retracted claim: {stale}. Remove the EXEMPTIONS entries so those "
            "files are guarded again."
        )

    def test_every_exemption_states_a_reason(self) -> None:
        thin = [path for path, reason in EXEMPTIONS.items() if len(reason.strip()) < 40]
        assert not thin, f"exemptions needing a real reason: {thin}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
