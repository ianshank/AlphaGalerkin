"""Tests for the PoC CLI module (src/poc/cli.py).

Validates:
    - `list` command output formatting and scenario listing
    - `info` command for existing and non-existing scenarios
    - `run` command with mocked scenario execution
    - `compare` command with mocked result data
    - `main()` argument parsing and dispatch
    - Edge cases: no subcommand, unknown scenario, empty registry
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.poc.config import (
    BaseScenarioConfig,
    ScenarioResult,
    ScenarioStatus,
)
from src.poc.registry import BaseScenario, ScenarioRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean the scenario registry before each test.

    We also remove cached scenario module imports so that
    re-importing them re-triggers the @scenario decorator.
    """
    ScenarioRegistry().clear()
    # Remove scenario module cache entries so @scenario decorators
    # re-fire on next import.
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src.poc.scenarios"):
            del sys.modules[mod_name]


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _make_result(
    name: str = "test_scenario",
    status: ScenarioStatus = ScenarioStatus.PASSED,
    passed: bool = True,
    metrics: dict[str, float] | None = None,
    duration: float = 1.0,
) -> ScenarioResult:
    """Helper to create a ScenarioResult for testing."""
    now = datetime.now()
    return ScenarioResult(
        scenario_name=name,
        config_hash="abc123",
        status=status,
        passed=passed,
        metrics=metrics or {},
        threshold_results={},
        artifacts={},
        start_time=now - timedelta(seconds=duration),
        end_time=now,
        duration_seconds=duration,
        device="cpu",
        python_version="3.11.0",
        torch_version="2.0.0",
    )


class _DummyConfig(BaseScenarioConfig):
    """Config with defaults for name/description.

    Dummy scenarios can be instantiated with no arguments
    (required by ScenarioRunner internals).
    """

    name: str = "dummy"
    description: str = "dummy scenario for testing"


def _make_dummy_config_cls(name: str) -> type[_DummyConfig]:
    """Create a Pydantic config subclass with proper annotations."""
    return type(
        f"DummyConfig_{name}",
        (_DummyConfig,),
        {
            "__annotations__": {"name": str, "description": str},
            "name": name,
            "description": f"dummy scenario {name}",
        },
    )


def _register_dummy_scenario(
    name: str = "dummy",
    *,
    pass_scenario: bool = True,
) -> type[BaseScenario]:
    """Register a minimal dummy scenario for testing CLI commands.

    The scenario uses a config subclass with defaults for required fields,
    allowing ScenarioRunner to create temp instances with ``scenario_cls()``.
    """

    def _execute_pass(self: BaseScenario) -> ScenarioResult:
        return self._create_result(ScenarioStatus.PASSED)

    def _execute_fail(self: BaseScenario) -> ScenarioResult:
        self.record_metric("dummy_fail", 1.0)
        import sys as _sys

        import torch as _torch

        end = datetime.now()
        assert self._start_time is not None
        dur = (end - self._start_time).total_seconds()
        return ScenarioResult(
            scenario_name=self.name,
            config_hash=self.config.compute_hash(),
            status=ScenarioStatus.FAILED,
            passed=False,
            metrics=dict(self._metrics),
            threshold_results={},
            artifacts={},
            start_time=self._start_time,
            end_time=end,
            duration_seconds=dur,
            device="cpu",
            python_version=_sys.version,
            torch_version=_torch.__version__,
        )

    config_cls = _make_dummy_config_cls(name)

    cls = type(
        f"DummyScenario_{name}",
        (BaseScenario,),
        {
            "config_class": config_cls,
            "execute": _execute_pass if pass_scenario else _execute_fail,
        },
    )
    ScenarioRegistry().register(name, cls)
    return cls


# ---------------------------------------------------------------------------
# Tests for cmd_list
# ---------------------------------------------------------------------------


class TestCmdList:
    """Tests for the `list` command."""

    def test_list_empty_registry(self, capsys: pytest.CaptureFixture[str]) -> None:
        """List with no scenarios registered prints 'No scenarios registered'."""
        from src.poc.cli import cmd_list

        # Mock register_builtin_scenarios to keep the registry empty
        with patch("src.poc.cli.register_builtin_scenarios"):
            args = argparse.Namespace()
            ret = cmd_list(args)

        assert ret == 0
        captured = capsys.readouterr()
        assert "No scenarios registered" in captured.out

    def test_list_with_scenarios(self, capsys: pytest.CaptureFixture[str]) -> None:
        """List with registered scenarios shows names and descriptions."""
        from src.poc.cli import cmd_list

        _register_dummy_scenario("alpha")
        _register_dummy_scenario("beta")

        # Mock register_builtin_scenarios so only our two dummies appear
        with patch("src.poc.cli.register_builtin_scenarios"):
            args = argparse.Namespace()
            ret = cmd_list(args)

        assert ret == 0
        captured = capsys.readouterr()
        assert "Available Scenarios" in captured.out
        assert "alpha" in captured.out
        assert "beta" in captured.out
        assert "Total: 2 scenarios" in captured.out

    def test_list_shows_tier(self, capsys: pytest.CaptureFixture[str]) -> None:
        """List command displays the tier of each scenario."""
        from src.poc.cli import cmd_list

        _register_dummy_scenario("gamma")

        args = argparse.Namespace()
        cmd_list(args)

        captured = capsys.readouterr()
        # BaseScenarioConfig defaults to FUNCTIONAL tier
        assert "functional" in captured.out

    def test_list_handles_scenario_init_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """List gracefully handles scenario that fails to initialize."""
        from src.poc.cli import cmd_list

        # Create a scenario class whose __init__ raises
        class BadScenario(BaseScenario):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("init failed")

            def execute(self) -> ScenarioResult:
                raise NotImplementedError

        ScenarioRegistry().register("bad_scenario", BadScenario)

        args = argparse.Namespace()
        ret = cmd_list(args)

        assert ret == 0
        captured = capsys.readouterr()
        assert "bad_scenario" in captured.out
        assert "(no description)" in captured.out


# ---------------------------------------------------------------------------
# Tests for cmd_info
# ---------------------------------------------------------------------------


class TestCmdInfo:
    """Tests for the `info` command."""

    def test_info_existing_scenario(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Info command shows details for a registered scenario."""
        from src.poc.cli import cmd_info

        _register_dummy_scenario("my_scenario")

        args = argparse.Namespace(scenario="my_scenario")
        ret = cmd_info(args)

        assert ret == 0
        captured = capsys.readouterr()
        assert "Scenario: my_scenario" in captured.out
        assert "Configuration Fields" in captured.out

    def test_info_nonexistent_scenario(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Info command returns 1 and prints error for unknown scenario."""
        from src.poc.cli import cmd_info

        args = argparse.Namespace(scenario="nonexistent")
        ret = cmd_info(args)

        assert ret == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_info_shows_config_fields(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Info prints config field names and defaults."""
        from src.poc.cli import cmd_info

        _register_dummy_scenario("field_test")

        args = argparse.Namespace(scenario="field_test")
        cmd_info(args)

        captured = capsys.readouterr()
        # BaseScenarioConfig has these fields
        assert "seed" in captured.out
        assert "timeout_seconds" in captured.out

    def test_info_scenario_init_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Info handles scenario init error gracefully."""
        from src.poc.cli import cmd_info

        class BrokenScenario(BaseScenario):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ValueError("bad init")

            def execute(self) -> ScenarioResult:
                raise NotImplementedError

        ScenarioRegistry().register("broken", BrokenScenario)

        args = argparse.Namespace(scenario="broken")
        ret = cmd_info(args)

        assert ret == 1
        captured = capsys.readouterr()
        assert "Error creating scenario" in captured.out


# ---------------------------------------------------------------------------
# Tests for cmd_run
# ---------------------------------------------------------------------------


class TestCmdRun:
    """Tests for the `run` command."""

    def test_run_specific_scenario(self, temp_output_dir: Path) -> None:
        """Run a specific named scenario returns 0 on success."""
        from src.poc.cli import cmd_run

        _register_dummy_scenario("run_test")

        args = argparse.Namespace(
            scenario="run_test",
            config=None,
            tier=None,
            parallel=1,
            fail_fast=False,
            output_dir=str(temp_output_dir),
        )

        ret = cmd_run(args)
        assert ret == 0

    def test_run_specific_scenario_failure(self, temp_output_dir: Path) -> None:
        """Run returns 1 when the scenario fails."""
        from src.poc.cli import cmd_run

        _register_dummy_scenario("fail_test", pass_scenario=False)

        args = argparse.Namespace(
            scenario="fail_test",
            config=None,
            tier=None,
            parallel=1,
            fail_fast=False,
            output_dir=str(temp_output_dir),
        )

        ret = cmd_run(args)
        assert ret == 1

    def test_run_all_scenarios(self, temp_output_dir: Path) -> None:
        """Run all registered scenarios when no --scenario is specified."""
        from src.poc.cli import cmd_run

        _register_dummy_scenario("all_a")
        _register_dummy_scenario("all_b")

        # Mock register_builtin_scenarios to avoid registering heavy builtins
        with patch("src.poc.cli.register_builtin_scenarios"):
            args = argparse.Namespace(
                scenario=None,
                config=None,
                tier=None,
                parallel=1,
                fail_fast=False,
                output_dir=str(temp_output_dir),
            )

            ret = cmd_run(args)
        assert ret == 0

    def test_run_from_config_file(self, temp_output_dir: Path) -> None:
        """Run from a YAML config file."""
        from src.poc.cli import cmd_run

        _register_dummy_scenario("from_config")

        # Write a minimal config YAML
        config_path = temp_output_dir / "test_config.yaml"
        config_data = {
            "scenarios": [
                {
                    "name": "from_config",
                    "description": "test scenario from config",
                    "tier": "functional",
                }
            ]
        }
        config_path.write_text(__import__("yaml").dump(config_data), encoding="utf-8")

        args = argparse.Namespace(
            scenario=None,
            config=str(config_path),
            tier=None,
            parallel=1,
            fail_fast=False,
            output_dir=str(temp_output_dir),
        )

        ret = cmd_run(args)
        assert ret == 0

    def test_run_with_tier_filter(self, temp_output_dir: Path) -> None:
        """Run with a tier filter runs only matching scenarios."""
        from src.poc.cli import cmd_run

        _register_dummy_scenario("tier_test")

        # Mock register_builtin_scenarios to avoid registering heavy builtins
        with patch("src.poc.cli.register_builtin_scenarios"):
            args = argparse.Namespace(
                scenario=None,
                config=None,
                tier="functional",
                parallel=1,
                fail_fast=False,
                output_dir=str(temp_output_dir),
            )

            # Should succeed (_DummyConfig default tier is functional)
            ret = cmd_run(args)
        assert ret == 0


# ---------------------------------------------------------------------------
# Tests for cmd_compare
# ---------------------------------------------------------------------------


class TestCmdCompare:
    """Tests for the `compare` command."""

    def test_compare_with_results(
        self, temp_output_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Compare two runs that have result files."""
        from src.poc.cli import cmd_compare
        from src.poc.results import ResultCollector

        # Create two runs with results
        collector_a = ResultCollector(output_dir=temp_output_dir, run_id="run_a")
        collector_a.collect(_make_result("scenario_x", metrics={"mse": 0.05}, duration=2.0))

        collector_b = ResultCollector(output_dir=temp_output_dir, run_id="run_b")
        collector_b.collect(_make_result("scenario_x", metrics={"mse": 0.03}, duration=1.5))

        args = argparse.Namespace(
            run_a="run_a",
            run_b="run_b",
            output_dir=str(temp_output_dir),
        )

        ret = cmd_compare(args)

        assert ret == 0
        captured = capsys.readouterr()
        assert "Comparing: run_a vs run_b" in captured.out
        assert "scenario_x" in captured.out
        assert "mse" in captured.out

    def test_compare_empty_runs(
        self, temp_output_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Compare two runs that have no result files."""
        from src.poc.cli import cmd_compare

        args = argparse.Namespace(
            run_a="empty_a",
            run_b="empty_b",
            output_dir=str(temp_output_dir),
        )

        ret = cmd_compare(args)

        assert ret == 0
        captured = capsys.readouterr()
        assert "Comparing: empty_a vs empty_b" in captured.out

    def test_compare_shows_status_change(
        self, temp_output_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Compare highlights status changes between runs."""
        from src.poc.cli import cmd_compare
        from src.poc.results import ResultCollector

        collector_a = ResultCollector(output_dir=temp_output_dir, run_id="change_a")
        collector_a.collect(_make_result("sc1", status=ScenarioStatus.FAILED, passed=False))

        collector_b = ResultCollector(output_dir=temp_output_dir, run_id="change_b")
        collector_b.collect(_make_result("sc1", status=ScenarioStatus.PASSED, passed=True))

        args = argparse.Namespace(
            run_a="change_a",
            run_b="change_b",
            output_dir=str(temp_output_dir),
        )

        ret = cmd_compare(args)
        assert ret == 0
        captured = capsys.readouterr()
        assert "failed" in captured.out
        assert "passed" in captured.out

    def test_compare_only_in_one_run(
        self, temp_output_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Compare shows 'Only in run X' for scenarios present in one run only."""
        from src.poc.cli import cmd_compare
        from src.poc.results import ResultCollector

        collector_a = ResultCollector(output_dir=temp_output_dir, run_id="one_a")
        collector_a.collect(_make_result("only_a_scenario"))

        collector_b = ResultCollector(output_dir=temp_output_dir, run_id="one_b")
        collector_b.collect(_make_result("only_b_scenario"))

        args = argparse.Namespace(
            run_a="one_a",
            run_b="one_b",
            output_dir=str(temp_output_dir),
        )

        ret = cmd_compare(args)
        assert ret == 0
        captured = capsys.readouterr()
        assert "Only in run" in captured.out


# ---------------------------------------------------------------------------
# Tests for main() entry point
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() CLI entry point and argument parsing."""

    def test_main_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main() with no subcommand prints help and returns 0."""
        from src.poc.cli import main

        with patch("sys.argv", ["cli"]):
            ret = main()

        assert ret == 0

    def test_main_list_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main() dispatches to list command."""
        from src.poc.cli import main

        with patch("sys.argv", ["cli", "list"]):
            ret = main()

        assert ret == 0
        captured = capsys.readouterr()
        # register_builtin_scenarios is called internally, so builtins appear
        assert "Available Scenarios" in captured.out

    def test_main_info_command_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main() dispatches info for a non-existent scenario."""
        from src.poc.cli import main

        with patch("sys.argv", ["cli", "info", "missing"]):
            ret = main()

        assert ret == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_main_run_command(self, temp_output_dir: Path) -> None:
        """main() dispatches run command with a specific scenario."""
        from src.poc.cli import main

        _register_dummy_scenario("main_test")

        with patch(
            "sys.argv",
            [
                "cli",
                "run",
                "--scenario",
                "main_test",
                "--output-dir",
                str(temp_output_dir),
            ],
        ):
            ret = main()

        assert ret == 0

    def test_main_compare_command(
        self, temp_output_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() dispatches compare command."""
        from src.poc.cli import main

        with patch(
            "sys.argv",
            [
                "cli",
                "compare",
                "run_x",
                "run_y",
                "--output-dir",
                str(temp_output_dir),
            ],
        ):
            ret = main()

        assert ret == 0
        captured = capsys.readouterr()
        assert "Comparing: run_x vs run_y" in captured.out

    def test_main_log_level_option(self) -> None:
        """main() accepts --log-level argument."""
        from src.poc.cli import main

        with patch("sys.argv", ["cli", "--log-level", "DEBUG", "list"]):
            ret = main()

        assert ret == 0


# ---------------------------------------------------------------------------
# Tests for register_builtin_scenarios
# ---------------------------------------------------------------------------


class TestRegisterBuiltinScenarios:
    """Tests for the register_builtin_scenarios helper."""

    def test_registers_all_builtins(self) -> None:
        """Calling register_builtin_scenarios populates the registry."""
        from src.poc.cli import register_builtin_scenarios

        register_builtin_scenarios()

        registry = ScenarioRegistry()
        names = registry.list_scenarios()

        assert "transfer" in names
        assert "complexity" in names
        assert "stability" in names

    def test_idempotent_registration(self) -> None:
        """Calling register_builtin_scenarios twice does not raise.

        The first call imports the scenario modules and triggers the
        @scenario decorators. The second call is a no-op because
        Python's import system returns the cached module.
        """
        from src.poc.cli import register_builtin_scenarios

        register_builtin_scenarios()

        # Second call should not raise; modules are already cached so
        # the @scenario decorators don't re-fire.
        register_builtin_scenarios()

        registry = ScenarioRegistry()
        assert len(registry.list_scenarios()) >= 3
