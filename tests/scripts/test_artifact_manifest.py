"""Unit tests for ``scripts/artifact_manifest.py`` on a throwaway git tree.

The docs-tier guard (``tests/docs/test_artifact_manifest.py``) checks the
*committed* manifest against the *committed* tree. This file checks the tool
itself: every exit code, every mismatch category, config override and
rejection, format strictness, and that untracked / excluded files never enter
the manifest. Each test builds its own tiny repository under ``tmp_path`` so
nothing here can touch ``results/``.

Mutation kill recorded here (``harden-a-guard``, 2026-09-11):

- **Bind ``sys.stderr`` at configure time** -- restore
  ``logger_factory=structlog.PrintLoggerFactory(file=sys.stderr)`` in
  ``configure_logging`` -> ``TestCli::test_logging_writes_to_the_current_stderr``
  fails (1 failed, 46 passed). That test is the only killer *because* the
  autouse ``_restore_structlog_config`` fixture now stops the CLI's global
  configuration leaking between tests; before it existed, the same defect
  surfaced as five ``TestWriteAndCheck`` tests dying with ``ValueError: I/O
  operation on closed file`` -- the symptom, not the cause, which is why a
  test pinning the cause was added rather than the symptom being fixed by the
  fixture alone.
"""

from __future__ import annotations

import io
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

import pytest
import structlog
import yaml
from pydantic import ValidationError
from structlog.testing import capture_logs

from scripts.artifact_manifest import (
    EXIT_MISMATCH,
    EXIT_OK,
    EXIT_USAGE,
    ArtifactManifestConfig,
    CheckReport,
    EntryKind,
    ManifestEntry,
    ManifestFormatError,
    _glob_match,
    check_manifest,
    compare,
    configure_logging,
    expected_entries,
    git_tracked_files,
    load_config,
    main,
    parse_manifest,
    render_manifest,
    sha256_of,
    write_manifest,
)

#: Small tree: two hashed CSVs, a sidecar, a baseline, a PNG, the excluded
#: template, and one *untracked* CSV that must never appear.
_TRACKED: Final[dict[str, str]] = {
    "results/a.csv": "x,y\n1,2\n",
    "results/b.csv": "x,y\n3,4\n",
    "results/a.run.json": '{"run_id": "a"}\n',
    "results/a.png": "not really a png\n",
    "config/baselines/x_ci.json": '{"entries": []}\n',
    "config/baselines/poc_headline.example.json": '{"template": true}\n',
}
_UNTRACKED: Final[dict[str, str]] = {"results/untracked.csv": "never,listed\n"}

#: The shipped floor is 10; a six-file fixture needs a lower one to be testable.
_FLOOR: Final[int] = 3


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, timeout=60)


@pytest.fixture(autouse=True)
def _restore_structlog_config() -> Iterator[None]:
    """``main()`` calls ``structlog.configure``, which is process-global.

    Snapshot and restore it around every test so the CLI's console renderer
    does not leak into whichever test module happens to run next.
    """
    saved = structlog.get_config()
    yield
    structlog.configure(**saved)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repository with the fixture files tracked (added, not committed)."""
    _git(tmp_path, "init", "-q")
    for rel, body in {**_TRACKED, **_UNTRACKED}.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    _git(tmp_path, "add", "--", *_TRACKED)
    return tmp_path


@pytest.fixture
def config() -> ArtifactManifestConfig:
    return ArtifactManifestConfig(min_hashed_entries=_FLOOR)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "manifest_config.yaml"
    path.write_text(yaml.safe_dump({"min_hashed_entries": _FLOOR}), encoding="utf-8")
    return path


def _run(*argv: str) -> int:
    return main(list(argv))


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_defaults_are_the_documented_ones(self) -> None:
        cfg = ArtifactManifestConfig()
        assert cfg.manifest_path == "results/MANIFEST.sha256"
        assert "results/*.csv" in cfg.hash_patterns
        assert "results/*.run.json" in cfg.hash_patterns
        assert "config/baselines/*.json" in cfg.hash_patterns
        assert cfg.presence_patterns == ("results/*.png",)
        assert cfg.exclude == ("config/baselines/poc_headline.example.json",)
        assert cfg.min_hashed_entries == 10
        assert cfg.presence_marker.startswith("#")

    def test_extra_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            ArtifactManifestConfig.model_validate({"hash_pattern": ["results/*.csv"]})

    def test_presence_marker_must_be_a_comment(self) -> None:
        """A non-comment marker would break ``sha256sum -c`` compatibility."""
        with pytest.raises(ValidationError, match="must start with '#'"):
            ArtifactManifestConfig(presence_marker="presence: ")
        assert ArtifactManifestConfig(presence_marker="# keep: ").presence_marker == "# keep: "

    def test_empty_path_entries_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty path entry"):
            ArtifactManifestConfig(hash_patterns=("results/*.csv", "  "))

    def test_empty_manifest_path_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            ArtifactManifestConfig(manifest_path="./")

    def test_paths_are_normalised(self) -> None:
        cfg = ArtifactManifestConfig(scan_roots=("./results/", "config\\baselines"))
        assert cfg.scan_roots == ("results", "config/baselines")

    def test_classify_precedence_and_exclusions(self) -> None:
        cfg = ArtifactManifestConfig(
            hash_patterns=("results/*.csv",),
            presence_patterns=("results/*.csv", "results/*.png"),
            exclude=("results/skip.csv",),
        )
        assert cfg.classify("results/a.csv") is EntryKind.HASHED  # hash wins over presence
        assert cfg.classify("results/a.png") is EntryKind.PRESENCE
        assert cfg.classify("results/skip.csv") is None
        assert cfg.classify(cfg.manifest_path) is None
        assert cfg.classify("results/notes.txt") is None

    def test_glob_never_crosses_a_slash(self) -> None:
        assert _glob_match("results/a.csv", "results/*.csv")
        assert not _glob_match("results/sub/a.csv", "results/*.csv")
        assert not _glob_match("outputs/results/a.csv", "results/*.csv")
        assert _glob_match("results/a.run.json", "results/*.run.json")
        assert not _glob_match("results/a.json", "results/*.run.json")

    def test_load_config_none_gives_defaults(self) -> None:
        assert load_config(None) == ArtifactManifestConfig()

    def test_load_config_empty_file_gives_defaults(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        assert load_config(empty) == ArtifactManifestConfig()

    def test_load_config_rejects_a_non_mapping(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- results/*.csv\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(bad)


# --------------------------------------------------------------------------- #
# Entries, rendering, parsing                                                  #
# --------------------------------------------------------------------------- #


class TestEntries:
    def test_entry_kind_and_digest_must_agree(self) -> None:
        with pytest.raises(ValueError, match="inconsistent"):
            ManifestEntry(path="a", kind=EntryKind.HASHED)
        with pytest.raises(ValueError, match="inconsistent"):
            ManifestEntry(path="a", kind=EntryKind.PRESENCE, digest="0" * 64)

    def test_expected_entries_skip_untracked_excluded_and_the_manifest(
        self, repo: Path, config: ArtifactManifestConfig
    ) -> None:
        # A stray manifest on disk that is *tracked* must still not hash itself.
        (repo / config.manifest_path).write_text("# stray\n", encoding="utf-8")
        _git(repo, "add", "--", config.manifest_path)
        entries = expected_entries(config, repo)
        paths = [e.path for e in entries]
        assert paths == sorted(paths)
        assert "results/untracked.csv" not in paths
        assert "config/baselines/poc_headline.example.json" not in paths
        assert config.manifest_path not in paths
        kinds = {e.path: e.kind for e in entries}
        assert kinds["results/a.csv"] is EntryKind.HASHED
        assert kinds["results/a.run.json"] is EntryKind.HASHED
        assert kinds["config/baselines/x_ci.json"] is EntryKind.HASHED
        assert kinds["results/a.png"] is EntryKind.PRESENCE
        assert kinds["results/a.csv"] is EntryKind.HASHED
        digest = next(e.digest for e in entries if e.path == "results/a.csv")
        assert digest == sha256_of(repo / "results/a.csv")

    def test_expected_entries_skip_a_staged_deletion(
        self, repo: Path, config: ArtifactManifestConfig
    ) -> None:
        """Tracked in the index, gone from disk: not crashed on, surfaced by check."""
        (repo / "results/b.csv").unlink()
        paths = [e.path for e in expected_entries(config, repo)]
        assert "results/b.csv" not in paths

    def test_render_then_parse_round_trips(self, config: ArtifactManifestConfig) -> None:
        entries = [
            ManifestEntry(path="results/z.png", kind=EntryKind.PRESENCE),
            ManifestEntry(path="results/a.csv", kind=EntryKind.HASHED, digest="a" * 64),
        ]
        text = render_manifest(entries, config)
        assert text.endswith("\n")
        assert "\r" not in text
        lines = text.splitlines()
        assert lines[0].startswith("#")
        body = [line for line in lines if not line.startswith("#") or "presence-only" in line]
        assert body == ["a" * 64 + "  results/a.csv", "# presence-only: results/z.png"]
        assert parse_manifest(text, config) == sorted(entries, key=lambda e: e.path)

    def test_parse_accepts_the_binary_marker_and_crlf(self, config: ArtifactManifestConfig) -> None:
        text = "b" * 64 + " *results/a.csv\r\n\r\n# comment\r\n"
        assert parse_manifest(text, config) == [
            ManifestEntry(path="results/a.csv", kind=EntryKind.HASHED, digest="b" * 64)
        ]

    def test_parse_rejects_a_malformed_line(self, config: ArtifactManifestConfig) -> None:
        with pytest.raises(ManifestFormatError, match="line 1: not a sha256sum line"):
            parse_manifest("deadbeef  results/a.csv\n", config)

    def test_parse_rejects_an_empty_presence_line(self, config: ArtifactManifestConfig) -> None:
        with pytest.raises(ManifestFormatError, match="names no path"):
            parse_manifest(config.presence_marker + "   \n", config)


class TestCompare:
    _A = ManifestEntry(path="results/a.csv", kind=EntryKind.HASHED, digest="a" * 64)
    _B = ManifestEntry(path="results/b.csv", kind=EntryKind.HASHED, digest="b" * 64)
    _P = ManifestEntry(path="results/a.png", kind=EntryKind.PRESENCE)

    def test_identical_is_ok(self) -> None:
        cfg = ArtifactManifestConfig(min_hashed_entries=2)
        report = compare([self._A, self._B, self._P], [self._A, self._B, self._P], cfg)
        assert report.ok
        assert (report.hashed_count, report.presence_count) == (2, 1)
        assert "OK:" in report.format(cfg.manifest_path)

    def test_every_category_is_reported_separately(self) -> None:
        cfg = ArtifactManifestConfig(min_hashed_entries=0)
        stale_a = ManifestEntry(path="results/a.csv", kind=EntryKind.HASHED, digest="c" * 64)
        demoted_b = ManifestEntry(path="results/b.csv", kind=EntryKind.PRESENCE)
        ghost = ManifestEntry(path="results/ghost.csv", kind=EntryKind.HASHED, digest="d" * 64)
        listed = [stale_a, demoted_b, ghost, ghost]
        report = compare(listed, [self._A, self._B, self._P], cfg)
        assert report.stale == ("results/a.csv",)
        assert report.wrong_kind == ("results/b.csv",)
        assert report.missing == ("results/ghost.csv",)
        assert report.unlisted == ("results/a.png",)
        assert report.duplicates == ("results/ghost.csv",)
        assert not report.ok
        text = report.format(cfg.manifest_path)
        for title in ("STALE", "UNLISTED", "MISSING", "WRONG KIND", "DUPLICATE", "MISMATCH"):
            assert title in text

    def test_too_few_is_a_failure_even_when_everything_matches(self) -> None:
        cfg = ArtifactManifestConfig(min_hashed_entries=5)
        report = compare([self._A], [self._A], cfg)
        assert report.too_few
        assert not report.ok
        assert "TOO FEW" in report.format(cfg.manifest_path)

    def test_check_report_defaults(self) -> None:
        assert CheckReport().ok


# --------------------------------------------------------------------------- #
# write / check on a real tree                                                 #
# --------------------------------------------------------------------------- #


class TestWriteAndCheck:
    def test_write_then_check_round_trip_and_idempotence(
        self, repo: Path, config: ArtifactManifestConfig
    ) -> None:
        report = write_manifest(config, repo)
        assert report.ok
        manifest = repo / config.manifest_path
        first = manifest.read_bytes()
        assert check_manifest(config, repo).ok
        write_manifest(config, repo)
        assert manifest.read_bytes() == first, "write is not deterministic"
        text = first.decode("utf-8")
        assert "results/untracked.csv" not in text
        assert "poc_headline.example.json" not in text
        assert config.manifest_path not in text.replace("sha256sum -c " + config.manifest_path, "")

    def test_write_refuses_a_vacuous_manifest(self, repo: Path) -> None:
        cfg = ArtifactManifestConfig(min_hashed_entries=50)
        with capture_logs() as logs:
            report = write_manifest(cfg, repo)
        assert report.too_few and not report.ok
        assert not (repo / cfg.manifest_path).exists(), "a vacuous manifest was written"
        assert any(entry["event"] == "artifact_manifest.write_refused" for entry in logs)

    def test_flipped_byte_is_stale(self, repo: Path, config: ArtifactManifestConfig) -> None:
        write_manifest(config, repo)
        target = repo / "results/a.csv"
        original = target.read_bytes()
        target.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
        assert target.read_bytes() != original, "mutation did not apply"
        report = check_manifest(config, repo)
        assert report.stale == ("results/a.csv",)
        assert not report.ok

    def test_deleted_entry_is_unlisted(self, repo: Path, config: ArtifactManifestConfig) -> None:
        write_manifest(config, repo)
        manifest = repo / config.manifest_path
        lines = manifest.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if not line.endswith("  results/b.csv")]
        assert len(kept) == len(lines) - 1, "mutation did not apply"
        manifest.write_text("\n".join(kept) + "\n", encoding="utf-8")
        report = check_manifest(config, repo)
        assert report.unlisted == ("results/b.csv",)

    def test_unlisted_tracked_file_is_reported(
        self, repo: Path, config: ArtifactManifestConfig
    ) -> None:
        write_manifest(config, repo)
        (repo / "results/new.csv").write_text("late,arrival\n", encoding="utf-8")
        assert check_manifest(config, repo).ok, "an untracked file must not count"
        _git(repo, "add", "--", "results/new.csv")
        report = check_manifest(config, repo)
        assert report.unlisted == ("results/new.csv",)

    def test_entry_without_a_file_is_missing(
        self, repo: Path, config: ArtifactManifestConfig
    ) -> None:
        write_manifest(config, repo)
        (repo / "results/a.png").unlink()
        _git(repo, "rm", "-q", "--cached", "--", "results/a.png")
        report = check_manifest(config, repo)
        assert report.missing == ("results/a.png",)

    def test_demoting_a_hashed_file_to_presence_is_wrong_kind(
        self, repo: Path, config: ArtifactManifestConfig
    ) -> None:
        """The adjacent weaker defect: ``sha256sum -c`` would still pass."""
        write_manifest(config, repo)
        manifest = repo / config.manifest_path
        text = manifest.read_text(encoding="utf-8")
        line = next(ln for ln in text.splitlines() if ln.endswith("  results/a.csv"))
        mutated = text.replace(line, config.presence_marker + "results/a.csv")
        assert mutated != text, "mutation did not apply"
        manifest.write_text(mutated, encoding="utf-8")
        report = check_manifest(config, repo)
        assert report.wrong_kind == ("results/a.csv",)

    def test_check_without_a_manifest_is_a_format_error(
        self, repo: Path, config: ArtifactManifestConfig
    ) -> None:
        with pytest.raises(ManifestFormatError, match="does not exist"):
            check_manifest(config, repo)

    def test_check_logs_a_mismatch_event(self, repo: Path, config: ArtifactManifestConfig) -> None:
        write_manifest(config, repo)
        (repo / "results/a.csv").write_text("changed\n", encoding="utf-8")
        with capture_logs() as logs:
            check_manifest(config, repo)
        events = {entry["event"]: entry for entry in logs}
        assert "artifact_manifest.check_mismatch" in events
        assert events["artifact_manifest.check_mismatch"]["stale"] == 1
        assert events["artifact_manifest.check_mismatch"]["manifest"] == config.manifest_path

    def test_injected_lister_replaces_git(self, tmp_path: Path) -> None:
        """The lister is a seam: a fake proves nothing here depends on a real repo."""
        (tmp_path / "results").mkdir()
        (tmp_path / "results/only.csv").write_text("1\n", encoding="utf-8")
        cfg = ArtifactManifestConfig(min_hashed_entries=1)

        def fake(root: Path, roots: Sequence[str]) -> list[str]:
            assert root == tmp_path and tuple(roots) == cfg.scan_roots
            return ["results/only.csv", "results/absent.csv"]

        assert write_manifest(cfg, tmp_path, lister=fake).ok
        assert check_manifest(cfg, tmp_path, lister=fake).ok

    def test_git_lister_fails_loudly_outside_a_repo(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="git ls-files failed"):
            git_tracked_files(tmp_path, ("results",))


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


class TestCli:
    def test_write_then_check_exit_0(
        self, repo: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run("--config", str(config_file), "--root", str(repo), "write") == EXIT_OK
        assert _run("--config", str(config_file), "--root", str(repo), "check") == EXIT_OK
        assert "OK: every frozen artifact matches" in capsys.readouterr().out

    def test_mismatch_exit_1(
        self, repo: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run("--config", str(config_file), "--root", str(repo), "write")
        (repo / "results/a.csv").write_text("edited\n", encoding="utf-8")
        assert _run("--config", str(config_file), "--root", str(repo), "check") == EXIT_MISMATCH
        out = capsys.readouterr().out
        assert "STALE" in out and "results/a.csv" in out

    def test_write_refusal_exit_1(self, repo: Path) -> None:
        # Default floor is 10; the fixture has 4 hashed files.
        assert _run("--root", str(repo), "write") == EXIT_MISMATCH

    def test_missing_manifest_exit_2(
        self, repo: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run("--config", str(config_file), "--root", str(repo), "check") == EXIT_USAGE
        assert "does not exist" in capsys.readouterr().err

    def test_malformed_manifest_exit_2(
        self, repo: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest = repo / "results/MANIFEST.sha256"
        manifest.write_text("not a manifest\n", encoding="utf-8")
        assert _run("--config", str(config_file), "--root", str(repo), "check") == EXIT_USAGE
        assert "not a sha256sum line" in capsys.readouterr().err

    def test_bad_config_exit_2(
        self, repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("no_such_field: 1\n", encoding="utf-8")
        assert _run("--config", str(bad), "--root", str(repo), "check") == EXIT_USAGE
        assert "no_such_field" in capsys.readouterr().err

    def test_absent_config_file_exit_2(self, repo: Path, tmp_path: Path) -> None:
        assert _run("--config", str(tmp_path / "nope.yaml"), "--root", str(repo), "check") == 2

    def test_outside_a_git_repo_exit_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run("--root", str(tmp_path), "write") == EXIT_USAGE
        assert "git ls-files failed" in capsys.readouterr().err

    def test_unknown_log_level_exit_2(self, repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert _run("--log-level", "LOUD", "--root", str(repo), "check") == EXIT_USAGE
        assert "unknown log level" in capsys.readouterr().err

    def test_missing_subcommand_is_a_usage_error(self, repo: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _run("--root", str(repo))
        assert excinfo.value.code == EXIT_USAGE

    def test_config_override_changes_what_is_frozen(self, repo: Path, tmp_path: Path) -> None:
        """A YAML override is honoured end-to-end, not just parsed."""
        (repo / "results/notes.txt").write_text("frozen by override\n", encoding="utf-8")
        _git(repo, "add", "--", "results/notes.txt")
        override = tmp_path / "override.yaml"
        override.write_text(
            yaml.safe_dump(
                {
                    "hash_patterns": ["results/*.txt"],
                    "presence_patterns": [],
                    "manifest_path": "results/OTHER.sha256",
                    "min_hashed_entries": 1,
                }
            ),
            encoding="utf-8",
        )
        assert _run("--config", str(override), "--root", str(repo), "write") == EXIT_OK
        text = (repo / "results/OTHER.sha256").read_text(encoding="utf-8")
        assert "results/notes.txt" in text
        assert "results/a.csv" not in text
        assert not (repo / "results/MANIFEST.sha256").exists()

    def test_log_level_filters_info(
        self, repo: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run("--log-level", "WARNING", "--config", str(config_file), "--root", str(repo), "write")
        assert "artifact_manifest.written" not in capsys.readouterr().err
        _run("--log-level", "INFO", "--config", str(config_file), "--root", str(repo), "write")
        assert "artifact_manifest.written" in capsys.readouterr().err

    def test_configure_logging_accepts_lowercase(self) -> None:
        configure_logging("debug")

    def test_logging_writes_to_the_current_stderr(
        self, monkeypatch: pytest.MonkeyPatch, repo: Path, config: ArtifactManifestConfig
    ) -> None:
        """The stream is resolved when a line is logged, not when logging was configured.

        Defect class: a logger bound to the ``sys.stderr`` object of configure
        time keeps writing there after the stream is swapped -- under pytest
        the old stream is closed by the next test and the first mismatch
        report dies with ``I/O operation on closed file``. Configure against
        one stream, swap in another, log, and require the line on the second.
        """
        write_manifest(config, repo)
        (repo / "results/a.csv").write_text("changed\n", encoding="utf-8")
        at_configure_time = io.StringIO()
        monkeypatch.setattr(sys, "stderr", at_configure_time)
        configure_logging("WARNING")
        at_log_time = io.StringIO()
        monkeypatch.setattr(sys, "stderr", at_log_time)
        check_manifest(config, repo)  # logs artifact_manifest.check_mismatch
        assert "artifact_manifest.check_mismatch" in at_log_time.getvalue()
        assert at_configure_time.getvalue() == ""
