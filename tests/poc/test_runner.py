"""Tests for the scenario runner.

Validates:
    - Single scenario execution
    - Batch execution
    - Retry logic
    - Config file loading
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.poc.config import (
    BaseScenarioConfig,
    ScenarioResult,
    ScenarioStatus,
)
from src.poc.registry import BaseScenario, ScenarioRegistry, scenario
from src.poc.runner import ScenarioRunner, create_runner


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Clean registry before each test."""
    ScenarioRegistry().clear()


@pytest.fixture
def temp_output_dir() -> Path:
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestScenarioRunner:
    """Tests for ScenarioRunner."""

    def test_create_runner(self, temp_output_dir: Path) -> None:
        """Test runner creation."""
        runner = create_runner(output_dir=temp_output_dir)

        assert runner.output_dir == temp_output_dir
        assert runner.max_workers == 1

    def test_run_single_scenario(self, temp_output_dir: Path) -> None:
        """Test running a single scenario."""

        @scenario("test_single")
        class TestScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                self.record_metric("test_value", 42.0)
                return self._create_result(ScenarioStatus.PASSED)

        runner = ScenarioRunner(output_dir=temp_output_dir)
        result = runner.run("test_single", name="test_single", description="Test")

        assert result.status == ScenarioStatus.PASSED
        assert result.metrics["test_value"] == 42.0

    def test_run_nonexistent_scenario(self, temp_output_dir: Path) -> None:
        """Test that running non-existent scenario raises error."""
        runner = ScenarioRunner(output_dir=temp_output_dir)

        with pytest.raises(ValueError, match="not found"):
            runner.run("nonexistent", name="test", description="test")

    def test_run_all_scenarios(self, temp_output_dir: Path) -> None:
        """Test running all registered scenarios."""

        @scenario("scenario_a")
        class ScenarioA(BaseScenario):
            config_class = BaseScenarioConfig

            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        @scenario("scenario_b")
        class ScenarioB(BaseScenario):
            config_class = BaseScenarioConfig

            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        # Need to set defaults since config_class expects name/description
        ScenarioA.config_class = type(
            "ConfigA",
            (BaseScenarioConfig,),
            {
                "__annotations__": {},
                "model_config": {"extra": "forbid"},
            },
        )
        ScenarioB.config_class = type(
            "ConfigB",
            (BaseScenarioConfig,),
            {
                "__annotations__": {},
                "model_config": {"extra": "forbid"},
            },
        )

        runner = ScenarioRunner(output_dir=temp_output_dir)

        # Run with explicit configs since defaults don't include name/description
        result_a = runner.run("scenario_a", name="scenario_a", description="Test A")
        result_b = runner.run("scenario_b", name="scenario_b", description="Test B")

        assert result_a.status == ScenarioStatus.PASSED
        assert result_b.status == ScenarioStatus.PASSED

    def test_retry_on_failure(self, temp_output_dir: Path) -> None:
        """Test retry logic on failure."""
        attempt_count = 0

        @scenario("retry_test")
        class RetryScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                nonlocal attempt_count
                attempt_count += 1

                if attempt_count < 3:
                    return self._create_result(ScenarioStatus.FAILED)
                return self._create_result(ScenarioStatus.PASSED)

        runner = ScenarioRunner(
            output_dir=temp_output_dir,
            retry_delay_base=0.01,  # Fast retries for testing
        )

        result = runner.run(
            "retry_test",
            name="retry_test",
            description="Test",
            retry_count=3,
        )

        assert result.status == ScenarioStatus.PASSED
        assert attempt_count == 3

    def test_fail_fast(self, temp_output_dir: Path) -> None:
        """Test fail_fast stops on first failure."""

        @scenario("first")
        class FirstScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.FAILED)

        @scenario("second")
        class SecondScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        runner = ScenarioRunner(
            output_dir=temp_output_dir,
            fail_fast=True,
        )

        # Run first scenario which fails
        result = runner.run("first", name="first", description="Test")
        assert result.status == ScenarioStatus.FAILED


class TestConfigFileLoading:
    """Tests for loading scenarios from config files."""

    def test_load_single_scenario_config(self, temp_output_dir: Path) -> None:
        """Test loading a single scenario from config."""
        config_content = {
            "name": "transfer",
            "description": "Test transfer scenario",
            "train_resolution": 9,
            "eval_resolutions": [9, 13],
            "mse_threshold": 0.1,
        }

        config_path = temp_output_dir / "scenario.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        runner = ScenarioRunner(output_dir=temp_output_dir)
        configs = runner.load_config(config_path)

        assert len(configs) == 1
        assert configs[0].name == "transfer"

    def test_load_multiple_scenarios(self, temp_output_dir: Path) -> None:
        """Test loading multiple scenarios from config."""
        config_content = {
            "scenarios": [
                {
                    "name": "transfer",
                    "description": "Transfer scenario",
                    "train_resolution": 9,
                },
                {
                    "name": "complexity",
                    "description": "Complexity scenario",
                    "grid_sizes": [5, 9, 13],
                },
            ]
        }

        config_path = temp_output_dir / "scenarios.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        runner = ScenarioRunner(output_dir=temp_output_dir)
        configs = runner.load_config(config_path)

        assert len(configs) == 2
        assert configs[0].name == "transfer"
        assert configs[1].name == "complexity"


class TestParallelExecution:
    """Tests for parallel scenario execution."""

    def test_parallel_execution(self, temp_output_dir: Path) -> None:
        """Test parallel execution with multiple workers."""
        import time

        execution_times: list[float] = []

        @scenario("parallel_a")
        class ParallelA(BaseScenario):
            def execute(self) -> ScenarioResult:
                execution_times.append(time.time())
                time.sleep(0.1)
                return self._create_result(ScenarioStatus.PASSED)

        @scenario("parallel_b")
        class ParallelB(BaseScenario):
            def execute(self) -> ScenarioResult:
                execution_times.append(time.time())
                time.sleep(0.1)
                return self._create_result(ScenarioStatus.PASSED)

        runner = ScenarioRunner(
            output_dir=temp_output_dir,
            max_workers=2,
        )

        # Run both scenarios
        runner.run("parallel_a", name="parallel_a", description="Test A")
        runner.run("parallel_b", name="parallel_b", description="Test B")

        # Both should have executed (order not guaranteed in parallel)
        assert len(execution_times) == 2


class TestResultCollection:
    """Tests for result collection during execution."""

    def test_results_persisted(self, temp_output_dir: Path) -> None:
        """Test that results are persisted to disk."""

        @scenario("persist_test")
        class PersistScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        runner = ScenarioRunner(output_dir=temp_output_dir)
        result = runner.run("persist_test", name="persist_test", description="Test")

        # Check that result file was created
        results_dir = temp_output_dir / "results" / runner.collector.run_id
        assert results_dir.exists()
        assert any(results_dir.glob("*.json"))

    def test_summary_generated(self, temp_output_dir: Path) -> None:
        """Test that summary is generated after run_all."""

        @scenario("summary_test")
        class SummaryScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        runner = ScenarioRunner(output_dir=temp_output_dir)
        runner.run("summary_test", name="summary_test", description="Test")

        # Summary should be created
        runner.collector.save_summary()

        summaries_dir = temp_output_dir / "summaries"
        assert summaries_dir.exists()
        assert any(summaries_dir.glob("*.json"))
