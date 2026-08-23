"""Tests for the scope-containment guard (``scripts/check_focus.py``).

Three things need guarding, and they fail in different ways: 1. **The classifier.** A gate
that never fires is worse than no gate, because it reads as coverage. Every test that
asserts a clean diff has a paired test asserting the same shape *does* fire once it crosses
the budget. 2. **The shipped config.** ``config/focus.yaml`` is the live policy; it must
validate, and its tracks must be the tracks ``docs/FOCUS.md`` describes. A frozen track
nobody wrote down is indistinguishable from an accident. 3. **The config schema.**
Overlapping or duplicated paths would make the classifier count one file on both sides and
report a violation against a single-file diff, so they are rejected at load time rather
than debugged later from a confusing CI failure.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import ValidationError

from scripts.check_focus import (
    DEFAULT_CONFIG,
    FOCUS_CONFIG_SCHEMA_VERSION,
    REPO_ROOT,
    ChangedFile,
    FocusConfig,
    classify,
    collect_changed_files,
    format_report,
    load_focus_config,
    main,
    parse_numstat,
)

FOCUS_DOC: Final[Path] = REPO_ROOT / "docs" / "FOCUS.md"

_MINIMAL: Final[dict[str, object]] = {
    "schema_version": 1,
    "incidental_line_budget": 20,
    "frozen_tracks": [{"name": "frozen", "reason": "because", "paths": ["frozen/"]}],
    "core_paths": ["core/"],
}


def _config(**overrides: object) -> FocusConfig:
    document = {**_MINIMAL, **overrides}
    return FocusConfig.model_validate(document)


# --------------------------------------------------------------------------
# numstat parsing
# --------------------------------------------------------------------------


def test_parse_numstat_sums_added_and_deleted() -> None:
    files = parse_numstat("10\t5\tsrc/a.py\n")
    assert files == (ChangedFile(path="src/a.py", lines_changed=15),)


def test_parse_numstat_treats_binary_files_as_zero_lines() -> None:
    """A binary blob has no line delta.

    Keeping the path but zeroing the count is the honest reading, and stops an image from
    being budgeted as feature work.
    """
    files = parse_numstat("-\t-\tresults/plot.png\n")
    assert files == (ChangedFile(path="results/plot.png", lines_changed=0),)


def test_parse_numstat_ignores_blank_and_malformed_lines() -> None:
    files = parse_numstat("\n  \nnot-a-numstat-row\n1\t1\tok.py\n")
    assert files == (ChangedFile(path="ok.py", lines_changed=2),)


def test_parse_numstat_ignores_non_integer_counts() -> None:
    files = parse_numstat("x\ty\tweird.py\n2\t3\tfine.py\n")
    assert files == (ChangedFile(path="fine.py", lines_changed=5),)


def test_parse_numstat_keeps_paths_containing_tabs() -> None:
    """``git`` can emit a path containing a tab; splitting on the first two only."""
    files = parse_numstat("1\t1\tdir/with\ttab.py\n")
    assert files == (ChangedFile(path="dir/with\ttab.py", lines_changed=2),)


def test_parse_numstat_normalises_windows_separators() -> None:
    files = parse_numstat("1\t1\tsrc\\mcts\\search.py\n")
    assert files[0].path == "src/mcts/search.py"


# --------------------------------------------------------------------------
# classification -- each clean case paired with the violation it must catch
# --------------------------------------------------------------------------


def test_core_only_change_is_clean() -> None:
    report = classify(_config(), [ChangedFile("core/a.py", 500)])
    assert not report.violated
    assert report.substantive_tracks == ()


def test_frozen_only_change_is_clean_however_large() -> None:
    """A freeze is not a ban. Work on a frozen track alone is allowed."""
    report = classify(_config(), [ChangedFile("frozen/a.py", 5000)])
    assert not report.violated
    assert report.substantive_tracks == ("frozen",)


def test_incidental_frozen_change_alongside_core_is_clean() -> None:
    report = classify(_config(), [ChangedFile("frozen/a.py", 20), ChangedFile("core/b.py", 300)])
    assert not report.violated


def test_substantive_frozen_change_alongside_core_violates() -> None:
    report = classify(_config(), [ChangedFile("frozen/a.py", 21), ChangedFile("core/b.py", 1)])
    assert report.violated
    assert report.substantive_tracks == ("frozen",)


def test_budget_is_summed_across_a_tracks_files_not_applied_per_file() -> None:
    """Sum the budget over a track, not per file.

    Otherwise the gate is trivially evaded by splitting one change into twenty files of
    one line each.
    """
    files = [ChangedFile(f"frozen/f{i}.py", 3) for i in range(10)] + [ChangedFile("core/b.py", 1)]
    report = classify(_config(), files)
    assert report.frozen_lines == {"frozen": 30}
    assert report.violated


def test_budgets_are_independent_per_track() -> None:
    config = _config(
        frozen_tracks=[
            {"name": "one", "reason": "r", "paths": ["one/"]},
            {"name": "two", "reason": "r", "paths": ["two/"]},
        ]
    )
    files = [ChangedFile("one/a.py", 15), ChangedFile("two/b.py", 15), ChangedFile("core/c.py", 1)]
    report = classify(config, files)
    assert report.frozen_lines == {"one": 15, "two": 15}
    assert not report.violated, "two independently-incidental tracks must not sum into a violation"


@pytest.mark.parametrize(
    ("pattern", "changed"),
    [
        # A directory written without its trailing slash. This is the real
        # configuration hazard: `src/pde` must not silently swallow
        # `src/pde_extras/`, which plain prefix matching would.
        ("src/pde", "src/pde_extras/a.py"),
        # A file path extended by a suffix. `train_compression.py` must not
        # also match a stub or a backup beside it.
        ("scripts/train_compression.py", "scripts/train_compression.py.bak"),
        ("scripts/train_compression.py", "scripts/train_compression.pyi"),
    ],
)
def test_pattern_without_a_trailing_slash_never_matches_by_prefix(
    pattern: str, changed: str
) -> None:
    config = _config(frozen_tracks=[{"name": "f", "reason": "r", "paths": [pattern]}])
    report = classify(config, [ChangedFile(changed, 999), ChangedFile("core/a.py", 1)])
    assert report.frozen_hits == (), f"{pattern!r} must not match {changed!r} by prefix"
    assert not report.violated


def test_exact_file_pattern_matches_that_file() -> None:
    config = _config(frozen_tracks=[{"name": "f", "reason": "r", "paths": ["deploy_space.py"]}])
    report = classify(config, [ChangedFile("deploy_space.py", 999), ChangedFile("core/a.py", 1)])
    assert report.violated


def test_directory_pattern_does_not_match_a_sibling_with_a_shared_prefix() -> None:
    config = _config(core_paths=["src/pde/"])
    report = classify(config, [ChangedFile("src/pde_extras/a.py", 1)])
    assert report.core_hits == ()


def test_zero_budget_makes_every_frozen_touch_substantive() -> None:
    report = classify(
        _config(incidental_line_budget=0),
        [ChangedFile("frozen/a.py", 1), ChangedFile("core/b.py", 1)],
    )
    assert report.violated


def test_binary_only_frozen_change_stays_incidental() -> None:
    report = classify(_config(), [ChangedFile("frozen/x.png", 0), ChangedFile("core/b.py", 1)])
    assert not report.violated


# --------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------


def test_paths_are_normalised_on_load() -> None:
    config = _config(
        frozen_tracks=[{"name": "f", "reason": "r", "paths": ["./frozen/"]}],
        core_paths=["/core/"],
    )
    assert config.frozen_tracks[0].paths == ("frozen/",)
    assert config.core_paths == ("core/",)


def test_overlapping_core_and_frozen_paths_are_rejected() -> None:
    with pytest.raises(ValidationError, match="overlaps frozen path"):
        _config(
            frozen_tracks=[{"name": "f", "reason": "r", "paths": ["src/"]}],
            core_paths=["src/mcts/"],
        )


def test_identical_core_and_frozen_paths_are_rejected() -> None:
    with pytest.raises(ValidationError, match="overlaps frozen path"):
        _config(
            frozen_tracks=[{"name": "f", "reason": "r", "paths": ["src/mcts/"]}],
            core_paths=["src/mcts/"],
        )


def test_a_path_claimed_by_two_tracks_is_rejected() -> None:
    with pytest.raises(ValidationError, match="claimed by both"):
        _config(
            frozen_tracks=[
                {"name": "one", "reason": "r", "paths": ["shared/"]},
                {"name": "two", "reason": "r", "paths": ["shared/"]},
            ]
        )


def test_duplicate_track_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate frozen-track names"):
        _config(
            frozen_tracks=[
                {"name": "same", "reason": "r", "paths": ["a/"]},
                {"name": "same", "reason": "r", "paths": ["b/"]},
            ]
        )


def test_empty_reason_is_rejected() -> None:
    """A reason is mandatory.

    Same principle as the charter's deviations register: an undisclosed reason is
    indistinguishable from an accident.
    """
    with pytest.raises(ValidationError):
        _config(frozen_tracks=[{"name": "f", "reason": "", "paths": ["frozen/"]}])


def test_no_frozen_tracks_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _config(frozen_tracks=[])


def test_negative_budget_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _config(incidental_line_budget=-1)


def test_future_schema_version_is_rejected_with_an_actionable_message() -> None:
    with pytest.raises(ValidationError, match="newer than this tool understands"):
        _config(schema_version=FOCUS_CONFIG_SCHEMA_VERSION + 1)


def test_unknown_keys_are_ignored_for_forward_compatibility() -> None:
    config = _config(some_future_key="value")
    assert config.incidental_line_budget == 20


def test_load_focus_config_rejects_a_non_mapping_document(tmp_path: Path) -> None:
    path = tmp_path / "focus.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_focus_config(path)


# --------------------------------------------------------------------------
# the shipped config, and its agreement with the doc
# --------------------------------------------------------------------------


def test_shipped_config_validates() -> None:
    config = load_focus_config(DEFAULT_CONFIG)
    assert config.frozen_tracks
    assert config.core_paths


def test_shipped_config_paths_exist_on_disk() -> None:
    """A frozen path that has been renamed away silently stops being frozen."""
    config = load_focus_config(DEFAULT_CONFIG)
    missing = [
        path
        for group in ([t.paths for t in config.frozen_tracks] + [config.core_paths])
        for path in group
        if not (REPO_ROOT / path.rstrip("/")).exists()
    ]
    assert not missing, f"focus.yaml names paths that do not exist: {missing}"


def test_shipped_config_directory_paths_end_with_a_slash() -> None:
    """Directory entries need their trailing slash.

    Without it the entry means "the file of that name", which matches nothing and freezes
    nothing -- a silent no-op rather than an error.

    The classifier cannot detect this (it has no disk access); the live config can be checked
    directly, so it is checked here.
    """
    config = load_focus_config(DEFAULT_CONFIG)
    unslashed = [
        path
        for group in ([t.paths for t in config.frozen_tracks] + [config.core_paths])
        for path in group
        if not path.endswith("/") and (REPO_ROOT / path).is_dir()
    ]
    assert not unslashed, f"directory entries missing a trailing '/': {unslashed}"


def test_every_shipped_track_is_described_in_the_focus_doc() -> None:
    config = load_focus_config(DEFAULT_CONFIG)
    doc = FOCUS_DOC.read_text(encoding="utf-8")
    undocumented = [t.name for t in config.frozen_tracks if f"`{t.name}`" not in doc]
    assert not undocumented, f"frozen tracks missing from docs/FOCUS.md: {undocumented}"


def test_the_focus_doc_names_no_track_the_config_does_not_freeze() -> None:
    """The reverse direction.

    A doc that claims a freeze the gate does not enforce is the failure mode this whole file
    exists to prevent.
    """
    config = load_focus_config(DEFAULT_CONFIG)
    known = {t.name for t in config.frozen_tracks}
    doc_table_rows = [
        line
        for line in FOCUS_DOC.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `")
    ]
    claimed = {row.split("`")[1] for row in doc_table_rows}
    assert claimed <= known, f"docs/FOCUS.md claims unenforced freezes: {sorted(claimed - known)}"
    assert claimed, "the doc's frozen-track table parsed empty -- the guard would be inert"


def test_shipped_config_yaml_is_the_document_the_model_validates() -> None:
    """Guards against the model silently ignoring a renamed top-level key."""
    document = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert set(document) == {
        "schema_version",
        "incidental_line_budget",
        "frozen_tracks",
        "core_paths",
    }


# --------------------------------------------------------------------------
# report rendering and the CLI
# --------------------------------------------------------------------------


def test_report_names_the_offending_track_and_its_reason() -> None:
    config = _config()
    report = classify(config, [ChangedFile("frozen/a.py", 100), ChangedFile("core/b.py", 1)])
    text = format_report(report, config)
    assert "VIOLATION" in text
    assert "frozen" in text
    assert "because" in text, "the report must explain why the track is frozen, not just that it is"


def test_report_says_ok_and_names_no_track_when_clean() -> None:
    config = _config()
    text = format_report(classify(config, [ChangedFile("core/b.py", 1)]), config)
    assert "OK:" in text
    assert "VIOLATION" not in text
    assert "frozen tracks touched:  none" in text


def test_cli_exit_codes_match_the_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    violation = tmp_path / "violation.numstat"
    violation.write_text(
        "300\t50\tsrc/video_compression/codec.py\n10\t2\tsrc/mcts/search.py\n", encoding="utf-8"
    )
    clean = tmp_path / "clean.numstat"
    clean.write_text("10\t2\tsrc/mcts/search.py\n", encoding="utf-8")

    assert main(["--numstat-file", str(violation)]) == 0, "report-only must not fail the build"
    assert main(["--numstat-file", str(violation), "--fail-on-violation"]) == 1
    assert main(["--numstat-file", str(clean), "--fail-on-violation"]) == 0
    assert "VIOLATION" in capsys.readouterr().out


def test_cli_requires_a_source_of_changed_files() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_accepts_an_alternate_config(tmp_path: Path) -> None:
    config_path = tmp_path / "focus.yaml"
    config_path.write_text(yaml.safe_dump(_MINIMAL), encoding="utf-8")
    numstat = tmp_path / "n.numstat"
    numstat.write_text("100\t0\tfrozen/a.py\n1\t0\tcore/b.py\n", encoding="utf-8")
    assert (
        main(["--config", str(config_path), "--numstat-file", str(numstat), "--fail-on-violation"])
        == 1
    )


# --------------------------------------------------------------------------
# git integration -- exercised against a real repository, not a mock
# --------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)


@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "base")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed")
    return repo


def test_collect_changed_files_reads_a_real_diff(tiny_repo: Path) -> None:
    _git(tiny_repo, "checkout", "-q", "-b", "work")
    (tiny_repo / "core").mkdir()
    (tiny_repo / "core" / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    _git(tiny_repo, "add", "-A")
    _git(tiny_repo, "commit", "-qm", "work")

    files = collect_changed_files("base", "HEAD", cwd=tiny_repo)
    assert files == (ChangedFile(path="core/a.py", lines_changed=2),)


def test_collect_changed_files_uses_the_merge_base_not_the_branch_tip(tiny_repo: Path) -> None:
    """Three-dot semantics.

    Commits landed on the base *after* the fork must not be attributed to this changeset --
    two-dot would attribute them all.
    """
    _git(tiny_repo, "checkout", "-q", "-b", "work")
    (tiny_repo / "mine.py").write_text("mine\n", encoding="utf-8")
    _git(tiny_repo, "add", "-A")
    _git(tiny_repo, "commit", "-qm", "mine")

    _git(tiny_repo, "checkout", "-q", "base")
    (tiny_repo / "theirs.py").write_text("theirs\n" * 50, encoding="utf-8")
    _git(tiny_repo, "add", "-A")
    _git(tiny_repo, "commit", "-qm", "theirs")

    _git(tiny_repo, "checkout", "-q", "work")
    paths = {f.path for f in collect_changed_files("base", "HEAD", cwd=tiny_repo)}
    assert paths == {"mine.py"}


def test_collect_changed_files_raises_on_an_unknown_ref(tiny_repo: Path) -> None:
    with pytest.raises(RuntimeError, match="git diff"):
        collect_changed_files("no-such-ref", "HEAD", cwd=tiny_repo)


def test_this_repository_is_currently_clean_against_its_own_base() -> None:
    """The live check, on the live tree.

    If this fails, the branch it runs on is itself violating the freeze -- which is exactly
    what the gate is for.
    """
    base = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/claude/alphagalerkin-implementation-4zGEN"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if base.returncode != 0:
        pytest.skip("base branch not fetched in this environment")
    report = classify(load_focus_config(DEFAULT_CONFIG), collect_changed_files(base.stdout.strip()))
    assert not report.violated, format_report(report, load_focus_config(DEFAULT_CONFIG))
