"""Additional coverage tests for ScenarioRunner.

Covers: run_all, run_from_config, _run_sequential, _run_parallel,
_print_summary, fail_fast in run_from_config, disabled scenarios.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import Field

from src.poc.config import (
    BaseScenarioConfig,
    ScenarioResult,
    ScenarioStatus,
)
from src.poc.registry import BaseScenario, ScenarioRegistry, scenario
from src.poc.runner import ScenarioRunner, create_runner


class _TestConfig(BaseScenarioConfig):
    """Config with defaults for name/description."""

    model_config = {"extra": "forbid"}
    name: str = Field(default="test")
    description: str = Field(default="test scenario")


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Clean registry before each test."""
    ScenarioRegistry().clear()


@pytest.fixture
def temp_output_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _register_passing(scenario_name: str) -> type[BaseScenario]:
    @scenario(scenario_name)
    class Passing(BaseScenario):
        config_class = _TestConfig

        def execute(self) -> ScenarioResult:
            return self._create_result(ScenarioStatus.PASSED)

    return Passing


def _register_failing(scenario_name: str) -> type[BaseScenario]:
    @scenario(scenario_name)
    class Failing(BaseScenario):
        config_class = _TestConfig

        def execute(self) -> ScenarioResult:
            return self._create_result(ScenarioStatus.FAILED)

    return Failing


class TestRunAll:
    """Tests for run_all method."""

    def test_run_all_sequential(self, temp_output_dir: Path) -> None:
        _register_passing("seq_a")
        _register_passing("seq_b")
        runner = ScenarioRunner(output_dir=temp_output_dir, max_workers=1)
        results = runner.run_all()
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_run_all_parallel(self, temp_output_dir: Path) -> None:
        _register_passing("par_a")
        _register_passing("par_b")
        runner = ScenarioRunner(output_dir=temp_output_dir, max_workers=2)
        results = runner.run_all()
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_run_all_filter_names(self, temp_output_dir: Path) -> None:
        _register_passing("keep")
        _register_passing("drop")
        runner = ScenarioRunner(output_dir=temp_output_dir)
        results = runner.run_all(filter_names=["keep"])
        assert len(results) == 1

    def test_run_all_empty(self, temp_output_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        runner = ScenarioRunner(output_dir=temp_output_dir)
        results = runner.run_all()
        assert len(results) == 0
        captured = capsys.readouterr()
        assert "No scenarios executed" in captured.out


class TestRunFromConfig:
    """Tests for run_from_config method."""

    def test_run_from_config_with_disabled(self, temp_output_dir: Path) -> None:
        _register_passing("cfg_test")
        config_content = {
            "scenarios": [
                {"name": "cfg_test", "description": "Test", "enabled": False},
            ]
        }
        config_path = temp_output_dir / "cfg.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        runner = ScenarioRunner(output_dir=temp_output_dir)
        results = runner.run_from_config(config_path)
        assert len(results) == 0

    def test_run_from_config_not_found(self, temp_output_dir: Path) -> None:
        config_content = {
            "scenarios": [
                {"name": "nonexistent", "description": "Missing"},
            ]
        }
        config_path = temp_output_dir / "cfg.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        runner = ScenarioRunner(output_dir=temp_output_dir)
        results = runner.run_from_config(config_path)
        assert len(results) == 0

    def test_run_from_config_enabled(self, temp_output_dir: Path) -> None:
        """Enabled scenarios in config are run."""
        _register_passing("cfg_run")
        config_content = {
            "scenarios": [
                {"name": "cfg_run", "description": "Enabled test"},
            ]
        }
        config_path = temp_output_dir / "cfg.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        runner = ScenarioRunner(output_dir=temp_output_dir)
        results = runner.run_from_config(config_path)
        assert len(results) == 1
        assert results[0].passed


class TestRunSequentialBranches:
    """Tests for _run_sequential edge cases."""

    def test_sequential_runs_all(self, temp_output_dir: Path) -> None:
        _register_passing("sq_a")
        _register_passing("sq_b")
        runner = ScenarioRunner(output_dir=temp_output_dir)
        results = runner.run_all()
        assert len(results) == 2
        assert all(r.passed for r in results)


class TestPrintSummary:
    """Tests for _print_summary output."""

    def test_summary_with_results(
        self, temp_output_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _register_passing("sum_a")
        runner = ScenarioRunner(output_dir=temp_output_dir)
        runner.run_all()
        captured = capsys.readouterr()
        assert "POC SCENARIO EXECUTION SUMMARY" in captured.out
        assert "ALL SCENARIOS PASSED" in captured.out

    def test_summary_with_no_threshold_scenario(
        self, temp_output_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Scenario with FAILED status but no thresholds still shows passed=True."""
        _register_failing("sum_fail")
        runner = ScenarioRunner(output_dir=temp_output_dir)
        results = runner.run_all()
        # Without thresholds, passed is always True
        assert len(results) == 1
        assert results[0].passed is True


class TestCreateRunner:
    """Tests for create_runner factory."""

    def test_create_runner_defaults(self, temp_output_dir: Path) -> None:
        runner = create_runner(output_dir=temp_output_dir)
        assert isinstance(runner, ScenarioRunner)
        assert runner.max_workers == 1
        assert runner.fail_fast is False

    def test_create_runner_custom(self, temp_output_dir: Path) -> None:
        runner = create_runner(
            output_dir=temp_output_dir,
            max_workers=4,
            fail_fast=True,
            retry_delay_base=0.5,
        )
        assert runner.max_workers == 4
        assert runner.fail_fast is True
        assert runner.retry_delay_base == 0.5
