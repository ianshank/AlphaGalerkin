"""Additional coverage tests for the scenario runner.

Targets uncovered lines in src/poc/runner.py:
    - run_all with filter_names
    - run_from_config with disabled/fail_fast/missing scenarios
    - _run_with_retry with class-based invocation
    - _run_parallel including timeout handling
    - _print_summary edge cases
    - create_runner factory
"""

from __future__ import annotations

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
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory."""
    return tmp_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scenario(name: str, passed: bool = True):
    """Register and return a trivial scenario class with default config."""
    # Create a config class that provides default name/description
    config_cls = type(
        f"Config_{name}",
        (BaseScenarioConfig,),
        {
            "__annotations__": {
                "name": str,
                "description": str,
            },
            "name": name,
            "description": f"Auto-generated for {name}",
            "model_config": {"extra": "forbid"},
        },
    )

    status = ScenarioStatus.PASSED if passed else ScenarioStatus.FAILED

    @scenario(name)
    class _Scenario(BaseScenario):
        config_class = config_cls

        def execute(self) -> ScenarioResult:
            from datetime import datetime as dt

            end = dt.now()
            assert self._start_time is not None
            dur = (end - self._start_time).total_seconds()
            return ScenarioResult(
                scenario_name=self.name,
                config_hash=self.config.compute_hash(),
                status=status,
                passed=passed,
                metrics={},
                start_time=self._start_time,
                end_time=end,
                duration_seconds=dur,
            )

    return _Scenario


# ---------------------------------------------------------------------------
# Tests: ScenarioRunner init and factory
# ---------------------------------------------------------------------------


class TestRunnerInit:
    def test_default_init(self, tmp_dir: Path) -> None:
        runner = ScenarioRunner(output_dir=tmp_dir)
        assert runner.output_dir == tmp_dir
        assert runner.max_workers == 1
        assert runner.fail_fast is False
        assert runner.retry_delay_base == 1.0

    def test_custom_init(self, tmp_dir: Path) -> None:
        runner = ScenarioRunner(
            output_dir=tmp_dir,
            max_workers=4,
            retry_delay_base=0.5,
            fail_fast=True,
        )
        assert runner.max_workers == 4
        assert runner.fail_fast is True
        assert runner.retry_delay_base == 0.5

    def test_create_runner_factory(self, tmp_dir: Path) -> None:
        runner = create_runner(output_dir=tmp_dir, max_workers=2)
        assert isinstance(runner, ScenarioRunner)
        assert runner.max_workers == 2

    def test_output_dir_created(self, tmp_dir: Path) -> None:
        nested = tmp_dir / "nested" / "output"
        runner = ScenarioRunner(output_dir=nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Tests: run_all with filters
# ---------------------------------------------------------------------------


class TestRunAll:
    def test_run_all_filter_names(self, tmp_dir: Path) -> None:
        _make_scenario("alpha")
        _make_scenario("beta")
        _make_scenario("gamma")

        runner = ScenarioRunner(output_dir=tmp_dir)
        results = runner.run_all(filter_names=["alpha", "gamma"])

        names = {r.scenario_name for r in results}
        assert "alpha" in names
        assert "gamma" in names
        assert "beta" not in names

    def test_run_all_empty_registry(self, tmp_dir: Path) -> None:
        runner = ScenarioRunner(output_dir=tmp_dir)
        results = runner.run_all()
        assert results == []

    def test_run_all_sequential_fail_fast(self, tmp_dir: Path) -> None:
        _make_scenario("aaa_pass_first")
        _make_scenario("bbb_fail_second", passed=False)
        _make_scenario("ccc_never_run")

        runner = ScenarioRunner(output_dir=tmp_dir, fail_fast=True)
        results = runner.run_all()

        # At least one should have failed, triggering fail_fast
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1
        # Should have stopped - max 2 results (pass + fail), not 3
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Tests: run_from_config
# ---------------------------------------------------------------------------


class TestRunFromConfig:
    def test_disabled_scenario_skipped(self, tmp_dir: Path) -> None:
        _make_scenario("transfer")

        config_content = {
            "scenarios": [
                {
                    "name": "transfer",
                    "description": "Disabled scenario",
                    "enabled": False,
                    "train_resolution": 9,
                },
            ]
        }
        cfg_path = tmp_dir / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config_content, f)

        runner = ScenarioRunner(output_dir=tmp_dir)
        results = runner.run_from_config(cfg_path)
        assert len(results) == 0

    def test_missing_scenario_logged(self, tmp_dir: Path) -> None:
        config_content = {
            "scenarios": [
                {
                    "name": "nonexistent_scenario",
                    "description": "Does not exist",
                },
            ]
        }
        cfg_path = tmp_dir / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config_content, f)

        runner = ScenarioRunner(output_dir=tmp_dir)
        # Should not raise, just log error
        results = runner.run_from_config(cfg_path)
        assert len(results) == 0

    def test_fail_fast_in_config_run(self, tmp_dir: Path) -> None:
        _make_scenario("complexity", passed=False)

        config_content = {
            "scenarios": [
                {
                    "name": "complexity",
                    "description": "Will fail",
                    "grid_sizes": [5, 9, 13],
                },
            ]
        }
        cfg_path = tmp_dir / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config_content, f)

        runner = ScenarioRunner(output_dir=tmp_dir, fail_fast=True)
        results = runner.run_from_config(cfg_path)
        assert len(results) == 1
        assert not results[0].passed


# ---------------------------------------------------------------------------
# Tests: _run_with_retry
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_retry_with_class_arg(self, tmp_dir: Path) -> None:
        """Test _run_with_retry when given a class instead of instance."""
        call_count = 0

        @scenario("retry_cls")
        class RetryCls(BaseScenario):
            def execute(self) -> ScenarioResult:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    return self._create_result(ScenarioStatus.FAILED)
                return self._create_result(ScenarioStatus.PASSED)

        runner = ScenarioRunner(output_dir=tmp_dir, retry_delay_base=0.001)

        config = BaseScenarioConfig(
            name="retry_cls", description="test", retry_count=2
        )
        result = runner._run_with_retry(RetryCls, config=config)
        assert result.passed

    def test_skipped_returns_immediately(self, tmp_dir: Path) -> None:
        @scenario("skip_me")
        class SkipScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.SKIPPED)

        runner = ScenarioRunner(output_dir=tmp_dir, retry_delay_base=0.001)
        instance = SkipScenario(name="skip_me", description="test")
        result = runner._run_with_retry(instance)
        assert result.status == ScenarioStatus.SKIPPED


# ---------------------------------------------------------------------------
# Tests: _run_parallel
# ---------------------------------------------------------------------------


class TestParallelExecution:
    def test_parallel_run_all(self, tmp_dir: Path) -> None:
        _make_scenario("p1")
        _make_scenario("p2")

        runner = ScenarioRunner(output_dir=tmp_dir, max_workers=2)
        results = runner.run_all()

        assert len(results) >= 2
        assert all(r.passed for r in results)


# ---------------------------------------------------------------------------
# Tests: _print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def test_empty_summary(self, tmp_dir: Path, capsys) -> None:
        runner = ScenarioRunner(output_dir=tmp_dir)
        runner._print_summary([])
        captured = capsys.readouterr()
        assert "No scenarios executed" in captured.out

    def test_mixed_results_summary(self, tmp_dir: Path, capsys) -> None:
        from datetime import datetime

        results = [
            ScenarioResult(
                scenario_name="ok",
                config_hash="abc",
                status=ScenarioStatus.PASSED,
                passed=True,
                metrics={"m": 1.0},
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.5,
            ),
            ScenarioResult(
                scenario_name="nok",
                config_hash="def",
                status=ScenarioStatus.FAILED,
                passed=False,
                metrics={},
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.5,
            ),
        ]

        runner = ScenarioRunner(output_dir=tmp_dir)
        runner._print_summary(results)
        captured = capsys.readouterr()
        assert "1 passed" in captured.out
        assert "1 failed" in captured.out
        assert "FAILED" in captured.out

    def test_all_passed_summary(self, tmp_dir: Path, capsys) -> None:
        from datetime import datetime

        results = [
            ScenarioResult(
                scenario_name="good",
                config_hash="abc",
                status=ScenarioStatus.PASSED,
                passed=True,
                metrics={},
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.1,
            ),
        ]

        runner = ScenarioRunner(output_dir=tmp_dir)
        runner._print_summary(results)
        captured = capsys.readouterr()
        assert "ALL SCENARIOS PASSED" in captured.out
