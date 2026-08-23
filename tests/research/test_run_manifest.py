"""Tests for src/research/run_manifest.py.

The module exists because committed artifacts cannot say how they were produced
-- ``results/lshape_mcts_vs_dorfler.csv`` carries only a ``seed`` column, so it
is impossible to tell from the file whether it predates the 2026-08-16 backup
fix. These tests pin the two properties that make a provenance collector
trustworthy: that it never raises inside the run it documents, and that a
migration cannot silently mangle an older document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.research.run_manifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    UNKNOWN,
    ArmProvenance,
    GitProvenance,
    RunManifest,
    collect_git_provenance,
    collect_package_versions,
    load_run_manifest,
    manifest_path_for,
    migrate_run_manifest,
    write_run_manifest,
)


def _manifest(**overrides: Any) -> RunManifest:
    defaults: dict[str, Any] = {
        "run_id": "test-run",
        "harness": "src.research.example",
        "seeds": [7961],
        "metrics": {"ratio": 1.5},
    }
    defaults.update(overrides)
    return RunManifest(**defaults)


class TestSchema:
    def test_defaults_are_unknown_not_absent(self) -> None:
        """'We do not know' is provenance; an omitted field is not."""
        manifest = _manifest()
        assert manifest.config_hash == UNKNOWN
        assert manifest.git.sha == UNKNOWN
        assert manifest.git.dirty is None, "None must be distinct from False"

    def test_schema_version_defaults_to_current(self) -> None:
        assert _manifest().schema_version == RUN_MANIFEST_SCHEMA_VERSION

    def test_unknown_fields_are_ignored_for_forward_compatibility(self) -> None:
        manifest = RunManifest.model_validate(
            {
                "run_id": "r",
                "harness": "h",
                "a_field_from_the_future": {"nested": True},
            }
        )
        assert manifest.run_id == "r"
        assert not hasattr(manifest, "a_field_from_the_future")

    def test_schema_version_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            _manifest(schema_version=0)


class TestStableFields:
    def test_excludes_every_volatile_field(self) -> None:
        stable = _manifest(
            created_at_utc="2026-08-23T00:00:00Z",
            hardware_tag="some-host",
            arms=[ArmProvenance(name="a", counters={"solves": 9.0})],
        ).stable_fields()
        for volatile in ("created_at_utc", "hardware_tag", "arms"):
            assert volatile not in stable, f"{volatile} would make goldens flaky"

    def test_keeps_the_fields_a_golden_needs(self) -> None:
        stable = _manifest().stable_fields()
        for required in ("run_id", "harness", "config_hash", "seeds", "metrics"):
            assert required in stable

    def test_two_runs_differing_only_in_volatiles_compare_equal(self) -> None:
        a = _manifest(created_at_utc="2026-01-01T00:00:00Z", hardware_tag="host-a")
        b = _manifest(created_at_utc="2026-12-31T23:59:59Z", hardware_tag="host-b")
        assert a.stable_fields() == b.stable_fields()

    def test_stable_fields_is_json_safe(self) -> None:
        json.dumps(_manifest().stable_fields())


class TestMigration:
    def test_unversioned_document_migrates_to_current(self) -> None:
        migrated = migrate_run_manifest({"run_id": "r", "harness": "h"})
        assert migrated["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION

    def test_future_schema_raises_rather_than_guessing(self) -> None:
        with pytest.raises(ValueError, match="newer than this code"):
            migrate_run_manifest({"schema_version": RUN_MANIFEST_SCHEMA_VERSION + 1, "run_id": "r"})

    def test_migration_does_not_mutate_the_caller_document(self) -> None:
        original = {"run_id": "r", "harness": "h"}
        snapshot = dict(original)
        migrate_run_manifest(original)
        assert original == snapshot, "migration mutated its input"

    @given(
        st.dictionaries(
            st.sampled_from(["run_id", "harness", "config_hash", "notes"]),
            st.text(max_size=16),
            max_size=4,
        )
    )
    def test_migration_is_idempotent(self, document: dict[str, str]) -> None:
        once = migrate_run_manifest(document)
        assert migrate_run_manifest(once) == once


class TestCollectorsNeverRaise:
    """A provenance collector that throws destroys the run it documents."""

    def test_outside_a_git_worktree(self, tmp_path: Path) -> None:
        provenance = collect_git_provenance(tmp_path)
        assert provenance.sha == UNKNOWN
        assert provenance.branch == UNKNOWN
        assert provenance.dirty is None

    def test_when_git_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _no_git(*args: Any, **kwargs: Any) -> Any:
            raise FileNotFoundError("git")

        monkeypatch.setattr("src.research.run_manifest.subprocess.run", _no_git)
        assert collect_git_provenance().sha == UNKNOWN

    def test_when_git_times_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        def _slow(*args: Any, **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd="git", timeout=1.0)

        monkeypatch.setattr("src.research.run_manifest.subprocess.run", _slow)
        assert collect_git_provenance().dirty is None

    def test_inside_this_repository(self) -> None:
        provenance = collect_git_provenance()
        assert len(provenance.sha) == 40, f"expected a full SHA, got {provenance.sha!r}"
        assert provenance.dirty in (True, False)

    def test_dirty_flag_is_true_when_the_tree_is_dirty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.research.run_manifest._git",
            lambda args, root: " M some/file.py" if args[0] == "status" else "abc",
        )
        assert collect_git_provenance().dirty is True

    def test_package_versions_records_a_missing_distribution_as_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An uninstalled optional extra must be None, not a crash and not omitted."""
        from importlib.metadata import PackageNotFoundError

        def _absent(name: str) -> str:
            raise PackageNotFoundError(name)

        monkeypatch.setattr("src.research.run_manifest._package_version", _absent)
        versions = collect_package_versions()
        assert versions.packages, "tracked packages must still be enumerated"
        assert all(value is None for value in versions.packages.values())

    def test_package_versions_records_absent_as_none(self) -> None:
        versions = collect_package_versions()
        assert versions.python.count(".") == 2
        assert "numpy" in versions.packages
        assert all(value is None or isinstance(value, str) for value in versions.packages.values())


class TestRoundTrip:
    def test_write_then_load_preserves_stable_fields(self, tmp_path: Path) -> None:
        manifest = _manifest(
            git=GitProvenance(sha="a" * 40, branch="main", dirty=False),
            arms=[ArmProvenance(name="dorfler", counters={"solves": 9.0})],
        )
        path = write_run_manifest(manifest, tmp_path / "nested" / "x.run.json")
        assert path.exists(), "parent directories must be created"
        assert load_run_manifest(path).stable_fields() == manifest.stable_fields()

    def test_load_migrates_an_unversioned_file(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.run.json"
        path.write_text(json.dumps({"run_id": "r", "harness": "h"}), encoding="utf-8")
        assert load_run_manifest(path).schema_version == RUN_MANIFEST_SCHEMA_VERSION

    def test_written_json_is_deterministic(self, tmp_path: Path) -> None:
        manifest = _manifest()
        first = write_run_manifest(manifest, tmp_path / "a.run.json").read_text()
        second = write_run_manifest(manifest, tmp_path / "b.run.json").read_text()
        assert first == second, "sorted keys make the sidecar diffable"


class TestSidecarsAreCommittable:
    """A sidecar silently swallowed by .gitignore defeats the whole mechanism.

    ``.gitignore`` carries a blanket ``*.json`` with a handful of negations. Without
    ``!results/**/*.json`` the provenance sidecar is never committed and **nothing
    errors** -- the artifact just lands alone, exactly as if the module did not exist.
    This is the failure mode a guard is most needed for, because it is invisible.
    """

    @staticmethod
    def _is_ignored(relative_path: str) -> bool:
        """True when git would ignore ``relative_path``."""
        import subprocess

        repo_root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["git", "check-ignore", "-q", relative_path],
            cwd=repo_root,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
        return completed.returncode == 0

    def test_a_results_sidecar_is_not_ignored(self) -> None:
        assert not self._is_ignored("results/example.run.json"), (
            "results/*.run.json is gitignored, so run provenance would never be "
            "committed and nothing would error. Restore the `!results/**/*.json` "
            "negation in .gitignore."
        )

    def test_the_committed_sidecar_is_tracked(self) -> None:
        """The mechanism is only real if a real sidecar is actually in the tree."""
        repo_root = Path(__file__).resolve().parents[2]
        sidecar = repo_root / "results" / "lshape_adaptive_vs_uniform.run.json"
        assert sidecar.exists(), "the committed adaptive-vs-uniform sidecar is missing"
        assert not self._is_ignored("results/lshape_adaptive_vs_uniform.run.json")

    def test_the_blanket_json_rule_still_applies_elsewhere(self) -> None:
        """The negation must be narrow: scratch JSON stays ignored."""
        assert self._is_ignored("outputs/scratch.run.json"), (
            "the .gitignore negation is too broad -- outputs/ JSON should stay ignored"
        )


class TestManifestPath:
    @pytest.mark.parametrize(
        ("artifact", "expected"),
        [
            ("results/x.csv", "results/x.run.json"),
            ("results/x.png", "results/x.run.json"),
            ("/abs/dir/y.csv", "/abs/dir/y.run.json"),
        ],
    )
    def test_sidecar_sits_beside_the_artifact(self, artifact: str, expected: str) -> None:
        assert manifest_path_for(artifact) == Path(expected)

    def test_csv_and_png_share_one_sidecar(self) -> None:
        assert manifest_path_for("results/x.csv") == manifest_path_for("results/x.png")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
