import argparse
import csv
import json
from unittest.mock import MagicMock, patch

import pytest

from src.poc.cli import cmd_run
from src.poc.config import ScenarioResult


@pytest.fixture
def mock_results() -> list[ScenarioResult]:
    # Need to create some dummy ScenarioResults
    # Mocking it out with simple MagicMocks
    res1 = MagicMock()
    res1.passed = True
    res1.scenario_name = "scenario_a"
    res1.duration_seconds = 1.23
    res1.metrics = {"score": 0.95, "error": 0.01}

    res2 = MagicMock()
    res2.passed = False
    res2.scenario_name = "scenario_b"
    res2.duration_seconds = 0.45
    res2.metrics = {"score": 0.10}

    return [res1, res2]


@pytest.fixture
def mock_runner(mock_results: list[ScenarioResult]):
    with patch("src.poc.cli.ScenarioRunner") as MockRunner:
        instance = MockRunner.return_value
        instance.run_all.return_value = mock_results
        yield instance


@pytest.fixture
def default_args() -> argparse.Namespace:
    args = argparse.Namespace()
    args.output_dir = "outputs/test"
    args.parallel = 1
    args.fail_fast = False
    args.config = None
    args.scenario = None
    args.tier = None
    args.demo = False
    args.export_results = None
    args.log_level = "INFO"
    return args


def test_demo_flag(mock_runner, default_args, capsys):
    default_args.demo = True
    cmd_run(default_args)
    captured = capsys.readouterr()

    assert "Scenario Name" in captured.out
    assert "scenario_a" in captured.out
    assert "scenario_b" in captured.out
    assert "✅ PASS" in captured.out
    assert "❌ FAIL" in captured.out
    assert "1.23" in captured.out


def test_export_results_json(mock_runner, default_args, tmp_path):
    export_file = tmp_path / "results.json"
    default_args.export_results = str(export_file)

    cmd_run(default_args)

    assert export_file.exists()
    with open(export_file) as f:
        data = json.load(f)

    assert len(data) == 2
    assert data[0]["scenario_name"] == "scenario_a"
    assert data[0]["passed"] is True
    assert data[0]["metrics"]["score"] == 0.95


def test_export_results_csv(mock_runner, default_args, tmp_path):
    export_file = tmp_path / "results.csv"
    default_args.export_results = str(export_file)

    cmd_run(default_args)

    assert export_file.exists()
    with open(export_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["scenario_name"] == "scenario_a"
    assert rows[0]["passed"] == "True"
    assert rows[0]["metric_score"] == "0.95"
    assert rows[0]["metric_error"] == "0.01"

    assert rows[1]["scenario_name"] == "scenario_b"
    assert rows[1]["passed"] == "False"
    assert rows[1]["metric_score"] == "0.1"
    assert rows[1]["metric_error"] == ""


def test_both_flags(mock_runner, default_args, tmp_path, capsys):
    export_file = tmp_path / "results.json"
    default_args.demo = True
    default_args.export_results = str(export_file)

    cmd_run(default_args)

    captured = capsys.readouterr()
    assert "Scenario Name" in captured.out
    assert export_file.exists()


def test_empty_results(mock_runner, default_args, tmp_path, capsys):
    mock_runner.run_all.return_value = []
    export_file = tmp_path / "empty.csv"
    default_args.demo = True
    default_args.export_results = str(export_file)

    cmd_run(default_args)

    captured = capsys.readouterr()
    assert "Scenario Name" in captured.out

    with open(export_file) as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0] == ["scenario_name", "passed", "duration_seconds"]


def test_all_pass(mock_runner, default_args):
    # Fix the mock results to all pass
    res1 = MagicMock()
    res1.passed = True
    res1.scenario_name = "scenario_a"
    res1.duration_seconds = 1.0
    res1.metrics = {}
    mock_runner.run_all.return_value = [res1]

    exit_code = cmd_run(default_args)
    assert exit_code == 0


def test_demo_flag_with_debug_log_level_skips_error_level_override(
    mock_runner, default_args, capsys
):
    """When ``--log-level`` is DEBUG/WARNING, ``--demo`` must not force ERROR.

    ``cmd_run`` only calls ``logging.getLogger().setLevel(logging.ERROR)``
    when the caller's log level is *not* one of DEBUG/WARNING -- this proves
    the guard's False branch (an explicit DEBUG/WARNING run keeps demo mode's
    presentation table without silencing the caller's chosen verbosity).
    """
    default_args.demo = True
    default_args.log_level = "DEBUG"
    ret = cmd_run(default_args)
    captured = capsys.readouterr()

    assert ret == 1  # one of the two mock results fails
    assert "Scenario Name" in captured.out


def test_demo_flag_truncates_long_metrics_string(mock_runner, default_args, capsys):
    """Metric summaries longer than 22 chars are truncated with an ellipsis."""
    long_metrics_result = mock_runner.run_all.return_value[0]
    long_metrics_result.metrics = {
        "residual_fit_r2": 0.987654321,
        "solved_fraction": 0.5,
        "extra_metric_name": 1.2345,
    }
    default_args.demo = True

    cmd_run(default_args)
    captured = capsys.readouterr()

    assert "..." in captured.out


def test_all_fail(mock_runner, default_args):
    # Fix the mock results to all fail
    res1 = MagicMock()
    res1.passed = False
    res1.scenario_name = "scenario_a"
    res1.duration_seconds = 1.0
    res1.metrics = {}
    mock_runner.run_all.return_value = [res1]

    exit_code = cmd_run(default_args)
    assert exit_code == 1
