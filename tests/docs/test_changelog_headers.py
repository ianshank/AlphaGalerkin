"""Guard R-08: one preamble, one ``[Unreleased]`` block, an ordered release ladder.

Defect class: an append-only changelog whose *structure* is duplicated by merges
-- two preambles, two ``[Unreleased]`` blocks each with their own ``### Added`` /
``### Changed`` / ``### Fixed`` groups, a version label reused or out of order --
reads as two changelogs, and a release cut by renaming "the" ``[Unreleased]``
heading renames one of them. That was the live state on 2026-09-11 (the format
sentence sat at line ~212 between two ``[Unreleased]`` headers, and ``## [0.3.0]``
heads two blocks), found by a person reading the file
(``docs/ENGINEERING_REFLECTION_2026-09-11.md`` R-08), not by a check.

Contract (Keep a Changelog 1.0.0 + PEP 440):

(a) exactly one ``## [Unreleased]`` header, and it is the first ``##`` header;
(b) exactly one preamble: the "Keep a Changelog" format sentence appears once
    in the file, before the first ``##`` header;
(c) release versions parse under PEP 440, are non-increasing top-to-bottom, and
    are unique except for ``DUPLICATE_VERSIONS`` -- an allowlist whose entries
    are asserted to *still* head more than one block (self-expiring) and to
    carry a reason;
(d) every ``###`` group under ``[Unreleased]`` is one of Added / Changed /
    Deprecated / Removed / Fixed / Security and appears at most once there
    (sub-sections use ``####``, which this guard does not read);
(e) every release header carries an ISO ``YYYY-MM-DD`` date, non-increasing
    top-to-bottom; ``[Unreleased]`` carries none.

Hermetic: reads ``CHANGELOG.md`` as text. Runs nothing. The file has no fenced
code blocks, and a vacuity test pins that so a future fence containing ``## ``
cannot be mistaken for a header without this docstring being wrong first.

PEP 440 comparison uses ``packaging.version.Version``. ``packaging`` is not a
declared dependency of this project, but it is a hard dependency of ``pytest``
itself, so it is importable in every environment that can collect this file.

Mutations (named tests) -- each planted, seen red, reverted, green again:

1. A second ``## [Unreleased]`` inserted above ``## [0.3.0] - 2026-04-01`` --
   ``test_exactly_one_unreleased_header_and_it_is_first`` fails.
2. The preamble's format sentence pasted again between two ``[Unreleased]``
   groups (the pre-R-08 state) --
   ``test_the_format_sentence_appears_once_in_the_preamble`` fails.
3. ``## [0.2.0] - 2026-01-26`` renamed ``## [0.5.0] - 2026-01-26`` --
   ``test_release_versions_are_non_increasing`` fails.
4. ``DUPLICATE_VERSIONS["0.2.0"] = "..."`` for a version that heads one block --
   ``test_every_allowlisted_duplicate_is_still_a_duplicate`` fails.
5. ``## [0.1.0]`` renamed ``## [0.2.0]`` (a duplicate not on the allowlist) --
   ``test_release_versions_are_unique_except_the_allowlist`` fails.
6. A second ``### Added`` inserted under ``[Unreleased]`` --
   ``test_unreleased_groups_are_canonical_and_unique`` fails.
7. ``## [0.2.0] - 2026-01-26`` redated ``2026-02-30`` --
   ``test_release_dates_are_iso_and_non_increasing`` fails.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Final, NamedTuple

from packaging.version import InvalidVersion, Version

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CHANGELOG: Final[Path] = REPO_ROOT / "CHANGELOG.md"

UNRELEASED: Final[str] = "Unreleased"
FORMAT_SENTENCE: Final[str] = "The format is based on [Keep a Changelog]"
CANONICAL_GROUPS: Final[frozenset[str]] = frozenset(
    {"Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"}
)

#: Versions that legitimately head more than one release block, with the reason.
#: Self-expiring in both directions: an entry whose version heads only one block
#: (or none) fails ``test_every_allowlisted_duplicate_is_still_a_duplicate``.
DUPLICATE_VERSIONS: Final[dict[str, str]] = {
    "0.3.0": (
        "`## [0.3.0] - 2026-07-22` and `## [0.3.0] - 2026-04-01` both exist: the "
        "July block reused the label instead of bumping. Left in place because "
        "released entries are append-only history (RELEASING.md); the next cut "
        "must use a fresh version rather than repair this one."
    ),
}

#: Vacuity floors: a changelog with fewer headers than this is not the file
#: this guard was written for, and every ladder assertion would pass on it.
MIN_RELEASE_HEADERS: Final[int] = 3
MIN_UNRELEASED_BULLETS: Final[int] = 1

_H2: Final[re.Pattern[str]] = re.compile(r"^## \[(?P<label>[^\]]+)\](?:\s+-\s+(?P<date>.*?))?\s*$")
_H3: Final[re.Pattern[str]] = re.compile(r"^### (?P<name>.+?)\s*$")
_ISO_DATE: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FENCE: Final[str] = "```"


class Header(NamedTuple):
    line_number: int
    label: str
    date: str | None


def _lines() -> list[str]:
    return CHANGELOG.read_text(encoding="utf-8").splitlines()


def _h2_headers(lines: list[str]) -> list[Header]:
    found: list[Header] = []
    for number, line in enumerate(lines, 1):
        match = _H2.match(line)
        if match:
            found.append(Header(number, match.group("label"), match.group("date")))
    return found


def _release_headers(lines: list[str]) -> list[Header]:
    return [h for h in _h2_headers(lines) if h.label != UNRELEASED]


def _unreleased_group_names(lines: list[str]) -> list[tuple[int, str]]:
    """``(line_number, name)`` for every ``###`` between ``[Unreleased]`` and the next ``##``."""
    headers = _h2_headers(lines)
    unreleased = [h for h in headers if h.label == UNRELEASED]
    if not unreleased:
        return []
    start = unreleased[0].line_number
    later = [h.line_number for h in headers if h.line_number > start]
    end = later[0] if later else len(lines) + 1
    groups: list[tuple[int, str]] = []
    for number in range(start + 1, end):
        match = _H3.match(lines[number - 1])
        if match:
            groups.append((number, match.group("name")))
    return groups


def version_ladder_violations(labels: list[str]) -> list[str]:
    """Adjacent pairs where a lower block outranks the one above it under PEP 440.

    Equal versions are *not* a violation here (uniqueness is a separate rule),
    and a final release ranks above its own ``-dev`` pre-release, so
    ``["0.4.0", "0.4.0-dev"]`` is clean while ``["0.4.0-dev", "0.4.0"]`` is not.
    """
    violations: list[str] = []
    for above, below in zip(labels, labels[1:]):
        if Version(below) > Version(above):
            violations.append(f"{below!r} sits below {above!r} but ranks above it")
    return violations


def date_ladder_violations(dates: list[str]) -> list[str]:
    """Adjacent pairs where a lower block is dated after the one above it."""
    violations: list[str] = []
    parsed = [dt.date.fromisoformat(d) for d in dates]
    for (above_s, above), (below_s, below) in zip(zip(dates, parsed), zip(dates[1:], parsed[1:])):
        if below > above:
            violations.append(f"{below_s} sits below {above_s} but is later")
    return violations


# --------------------------------------------------------------------------- vacuity


def test_changelog_has_enough_structure_to_guard() -> None:
    """Without this, every ladder assertion below could pass on an empty file."""
    lines = _lines()
    assert not any(line.startswith(_FENCE) for line in lines), (
        "CHANGELOG.md now contains a fenced code block; this guard reads headers "
        "line-by-line and would treat a '## ' inside a fence as a release header -- "
        "teach it to skip fences before relying on it again"
    )
    releases = _release_headers(lines)
    assert len(releases) >= MIN_RELEASE_HEADERS, (
        f"expected >= {MIN_RELEASE_HEADERS} release headers, found {len(releases)}"
    )
    groups = _unreleased_group_names(lines)
    assert groups, "no ### group under [Unreleased] -- nothing for rule (d) to check"
    start = groups[0][0]
    bullets = sum(1 for line in lines[start:] if line.startswith("- "))
    assert bullets >= MIN_UNRELEASED_BULLETS


# --------------------------------------------------------------------------- (a)


def test_exactly_one_unreleased_header_and_it_is_first() -> None:
    """Two ``[Unreleased]`` blocks means a release cut renames only one of them."""
    headers = _h2_headers(_lines())
    unreleased = [h for h in headers if h.label == UNRELEASED]
    assert len(unreleased) == 1, (
        "expected exactly one '## [Unreleased]' header, found "
        f"{[h.line_number for h in unreleased]}"
    )
    assert headers[0].label == UNRELEASED, (
        f"'## [Unreleased]' must be the first '##' header; first is "
        f"{headers[0].label!r} at line {headers[0].line_number}"
    )
    assert unreleased[0].date is None, (
        f"'[Unreleased]' must carry no date (line {unreleased[0].line_number})"
    )


# --------------------------------------------------------------------------- (b)


def test_the_format_sentence_appears_once_in_the_preamble() -> None:
    """A second preamble mid-file is the fingerprint of a merged-in second changelog."""
    lines = _lines()
    hits = [n for n, line in enumerate(lines, 1) if FORMAT_SENTENCE in line]
    assert len(hits) == 1, (
        f"the Keep a Changelog format sentence must appear exactly once; found at lines {hits}"
    )
    first_h2 = _h2_headers(lines)[0].line_number
    assert hits[0] < first_h2, (
        f"the format sentence (line {hits[0]}) must precede the first '##' header "
        f"(line {first_h2}) -- it is the preamble, not an entry"
    )


# --------------------------------------------------------------------------- (c)


def test_release_versions_parse_under_pep440() -> None:
    bad: list[str] = []
    for header in _release_headers(_lines()):
        try:
            Version(header.label)
        except InvalidVersion:
            bad.append(f"line {header.line_number}: {header.label!r}")
    assert not bad, f"release labels that are not PEP 440 versions: {bad}"


def test_release_versions_are_non_increasing() -> None:
    """Newest first: no block may outrank the block above it."""
    labels = [h.label for h in _release_headers(_lines())]
    assert not version_ladder_violations(labels), version_ladder_violations(labels)


def test_release_versions_are_unique_except_the_allowlist() -> None:
    """A reused label is history that cannot be cited; a new one needs an allowlist reason."""
    seen: dict[str, list[int]] = {}
    for header in _release_headers(_lines()):
        seen.setdefault(str(Version(header.label)), []).append(header.line_number)
    unexplained = {
        label: at
        for label, at in seen.items()
        if len(at) > 1 and label not in {str(Version(v)) for v in DUPLICATE_VERSIONS}
    }
    assert not unexplained, (
        "version labels heading more than one block with no DUPLICATE_VERSIONS entry: "
        f"{unexplained}"
    )


def test_every_allowlisted_duplicate_is_still_a_duplicate() -> None:
    """Self-expiring: an entry for a version that heads one block (or none) is stale."""
    seen: dict[str, int] = {}
    for header in _release_headers(_lines()):
        key = str(Version(header.label))
        seen[key] = seen.get(key, 0) + 1
    stale: list[str] = []
    for label, reason in DUPLICATE_VERSIONS.items():
        assert reason.strip(), f"DUPLICATE_VERSIONS[{label!r}] has no reason"
        if seen.get(str(Version(label)), 0) < 2:
            stale.append(f"{label!r} heads {seen.get(str(Version(label)), 0)} block(s)")
    assert not stale, f"stale DUPLICATE_VERSIONS entries (delete them): {stale}"


# --------------------------------------------------------------------------- (d)


def test_unreleased_groups_are_canonical_and_unique() -> None:
    """One ``### Added`` etc. under ``[Unreleased]``; titled sections belong at ``####``."""
    groups = _unreleased_group_names(_lines())
    non_canonical = [(n, name) for n, name in groups if name not in CANONICAL_GROUPS]
    assert not non_canonical, (
        "'###' headings under [Unreleased] must be exactly one of "
        f"{sorted(CANONICAL_GROUPS)}; use '####' for a titled sub-section: {non_canonical}"
    )
    counts: dict[str, list[int]] = {}
    for n, name in groups:
        counts.setdefault(name, []).append(n)
    repeated = {name: at for name, at in counts.items() if len(at) > 1}
    assert not repeated, f"groups appearing more than once under [Unreleased]: {repeated}"


# --------------------------------------------------------------------------- (e)


def test_release_dates_are_iso_and_non_increasing() -> None:
    lines = _lines()
    releases = _release_headers(lines)
    bad: list[str] = []
    for header in releases:
        if header.date is None or not _ISO_DATE.match(header.date):
            bad.append(f"line {header.line_number}: {header.label!r} date={header.date!r}")
            continue
        try:
            dt.date.fromisoformat(header.date)
        except ValueError:
            bad.append(f"line {header.line_number}: {header.date!r} is not a calendar date")
    assert not bad, f"release headers without a valid ISO YYYY-MM-DD date: {bad}"
    dates = [h.date for h in releases if h.date is not None]
    assert not date_ladder_violations(dates), date_ladder_violations(dates)


# --------------------------------------------------------------------------- predicates


class TestLadderPredicates:
    """Pin the ordering rules on synthetic input, independent of the live file."""

    def test_equal_versions_are_not_an_ordering_violation(self) -> None:
        assert version_ladder_violations(["0.4.0-dev", "0.3.0", "0.3.0", "0.2.0"]) == []

    def test_final_above_its_own_dev_prerelease_is_clean(self) -> None:
        assert version_ladder_violations(["0.4.0", "0.4.0-dev"]) == []

    def test_dev_prerelease_above_its_final_is_a_violation(self) -> None:
        assert version_ladder_violations(["0.4.0-dev", "0.4.0"])

    def test_higher_version_below_is_a_violation(self) -> None:
        assert version_ladder_violations(["0.3.0", "0.5.0"])

    def test_equal_dates_are_clean(self) -> None:
        assert date_ladder_violations(["2026-01-26", "2026-01-26"]) == []

    def test_later_date_below_is_a_violation(self) -> None:
        assert date_ladder_violations(["2026-01-26", "2026-02-01"])
