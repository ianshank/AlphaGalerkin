"""Unit tests for the shared baseline-CLI helper.

``src/poc/baselines/cli_support.py`` is the single implementation behind
``--record-baseline`` / ``--baseline`` on the three harness scripts. It is tested
here rather than only through those scripts because each of them costs a full
benchmark run, and the behaviour that matters most -- refusing to write a
baseline that would gate nothing -- is cheapest to pin directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import pytest

from src.poc.baselines.cli_support import (
    DEFAULT_CLI_TOLERANCE_PCT,
    EXIT_OK,
    EXIT_REGRESSION,
    add_baseline_arguments,
    handle_baseline_flags,
)

SCENARIO: Final[str] = "demo_scenario"


def _args(**overrides: object) -> argparse.Namespace:
    """Build a Namespace shaped like the one the harness parsers produce."""
    base: dict[str, object] = {
        "record_baseline": None,
        "baseline": None,
        "tolerance_pct": DEFAULT_CLI_TOLERANCE_PCT,
        "git_sha": "",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestArgumentWiring:
    """The three flags exist with the documented defaults."""

    def test_defaults_match_the_documented_contract(self) -> None:
        """Both paths default to off, and the tolerance to the shared constant."""
        parser = add_baseline_arguments(argparse.ArgumentParser())
        parsed = parser.parse_args([])
        assert parsed.record_baseline is None
        assert parsed.baseline is None
        assert parsed.tolerance_pct == DEFAULT_CLI_TOLERANCE_PCT

    def test_tolerance_default_is_overridable_per_harness(self) -> None:
        """A noisier harness can widen its own default without editing the helper."""
        parser = add_baseline_arguments(argparse.ArgumentParser(), default_tolerance_pct=99.0)
        assert parser.parse_args([]).tolerance_pct == 99.0


class TestNeitherFlagGiven:
    """Returning ``None`` is what keeps the caller's own verdict reachable."""

    def test_returns_none_so_the_caller_decides(self) -> None:
        """Collapsing ``None`` into ``EXIT_OK`` would make every run exit 0."""
        assert (
            handle_baseline_flags(
                _args(),
                observed={SCENARIO: {"metric": 1.0}},
                scenario_name=SCENARIO,
                stable_filter=dict,
            )
            is None
        )


class TestEmptyBaselineIsRefused:
    """Copilot review, PR #144.

    An empty baseline is worse than no baseline: ``compare`` has nothing to
    check, so every later ``--baseline`` run exits 0 and the gate reports green
    while measuring nothing. It happens silently the moment a metric is renamed
    and the harness's ``stable_filter`` stops matching -- the same vacuity class
    as a coverage gate whose target is swallowed by ``omit``, which this repo has
    now hit four times.

    ``poc.cli record-baseline`` already refused this case; the shared helper must
    not become the way around it.
    """

    def test_records_nothing_and_reports_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A filter that selects nothing must not produce a file."""
        destination = tmp_path / "empty.json"
        code = handle_baseline_flags(
            _args(record_baseline=str(destination)),
            observed={SCENARIO: {"a": 1.0, "b": 2.0}},
            scenario_name=SCENARIO,
            stable_filter=lambda _metrics: {},
        )
        assert code == EXIT_REGRESSION
        assert not destination.exists()
        assert "refusing to write an empty baseline" in capsys.readouterr().out

    def test_a_matching_filter_still_records(self, tmp_path: Path) -> None:
        """The refusal must be conditional, or recording would be broken outright.

        Without this, deleting the whole record branch would also pass the test
        above.
        """
        destination = tmp_path / "populated.json"
        code = handle_baseline_flags(
            _args(record_baseline=str(destination)),
            observed={SCENARIO: {"kept": 1.0, "dropped": 2.0}},
            scenario_name=SCENARIO,
            stable_filter=lambda metrics: {k: v for k, v in metrics.items() if k == "kept"},
        )
        assert code == EXIT_OK
        document = json.loads(destination.read_text(encoding="utf-8"))
        assert [entry["metric_name"] for entry in document["entries"]] == ["kept"]


class TestDiff:
    """The diff path's exit codes."""

    def _record(self, tmp_path: Path, value: float) -> Path:
        destination = tmp_path / "baseline.json"
        handle_baseline_flags(
            _args(record_baseline=str(destination), tolerance_pct=0.0),
            observed={SCENARIO: {"metric": value}},
            scenario_name=SCENARIO,
            stable_filter=dict,
        )
        return destination

    def test_self_diff_is_clean(self, tmp_path: Path) -> None:
        """Diffing a run against a baseline recorded from it exits 0."""
        baseline = self._record(tmp_path, 1.0)
        code = handle_baseline_flags(
            _args(baseline=str(baseline)),
            observed={SCENARIO: {"metric": 1.0}},
            scenario_name=SCENARIO,
            stable_filter=dict,
        )
        assert code == EXIT_OK

    def test_a_worse_value_is_reported(self, tmp_path: Path) -> None:
        """A lower-better metric that doubled at zero tolerance exits 1."""
        baseline = self._record(tmp_path, 1.0)
        code = handle_baseline_flags(
            _args(baseline=str(baseline)),
            observed={SCENARIO: {"metric": 2.0}},
            scenario_name=SCENARIO,
            stable_filter=dict,
        )
        assert code == EXIT_REGRESSION
