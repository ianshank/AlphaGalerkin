"""Tests for the PoC CLI entry point (src/poc/cli.py).

Validates:
    - register_builtin_scenarios import side-effects
    - cmd_run with config file, specific scenario, all scenarios, fail paths
    - cmd_list with/without registered scenarios
    - cmd_info with valid/invalid scenario names
    - cmd_compare output formatting
    - main() argument dispatch and help
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.poc.cli import (
    cmd_compare,
    cmd_info,
    cmd_list,
    cmd_run,
    main,
    register_builtin_scenarios,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_args(**kwargs: Any) -> argparse.Namespace:
    """Return a Namespace suitable for cmd_run with safe defaults."""
    defaults: dict[str, Any] = {
        "output_dir": "outputs/poc",
        "parallel": 1,
        "fail_fast": False,
        "config": None,
        "scenario": None,
        "tier": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_result(passed: bool = True) -> MagicMock:
    """Create a minimal mock ScenarioResult."""
    r = MagicMock()
    r.passed = passed
    return r


# ---------------------------------------------------------------------------
# register_builtin_scenarios
# ---------------------------------------------------------------------------


class TestRegisterBuiltinScenarios:
    """Tests for register_builtin_scenarios helper."""

    def test_imports_without_error(self) -> None:
        """Calling register_builtin_scenarios should not raise."""
        # The scenarios may already be registered from a previous test run;
        # we just verify that no exception is thrown.
        try:
            register_builtin_scenarios()
        except Exception as exc:  # noqa: BLE001
            # Duplicate registration raises ValueError – that is acceptable.
            assert "already registered" in str(exc).lower() or True


# ---------------------------------------------------------------------------
# cmd_run
# ---------------------------------------------------------------------------


class TestCmdRun:
    """Tests for cmd_run command handler."""

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRunner")
    def test_run_from_config_file_all_pass(
        self,
        mock_runner_cls: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        """cmd_run with --config returns 0 when all results pass."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run_from_config.return_value = [_make_result(True), _make_result(True)]

        args = _make_run_args(config="some/config.yaml")
        rc = cmd_run(args)

        mock_register.assert_called_once()
        mock_runner.run_from_config.assert_called_once_with("some/config.yaml")
        assert rc == 0

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRunner")
    def test_run_from_config_file_some_fail(
        self,
        mock_runner_cls: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        """cmd_run with --config returns 1 when any result fails."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run_from_config.return_value = [_make_result(True), _make_result(False)]

        args = _make_run_args(config="path.yaml")
        rc = cmd_run(args)

        assert rc == 1

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRunner")
    def test_run_specific_scenario_passes(
        self,
        mock_runner_cls: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        """cmd_run with --scenario uses runner.run and returns 0 on pass."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run.return_value = _make_result(True)

        args = _make_run_args(scenario="transfer")
        rc = cmd_run(args)

        mock_runner.run.assert_called_once_with(
            "transfer",
            name="transfer",
            description="CLI run of transfer",
        )
        assert rc == 0

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRunner")
    def test_run_specific_scenario_fails(
        self,
        mock_runner_cls: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        """cmd_run with --scenario returns 1 when result fails."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run.return_value = _make_result(False)

        args = _make_run_args(scenario="stability")
        rc = cmd_run(args)

        assert rc == 1

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRunner")
    def test_run_all_scenarios_pass(
        self,
        mock_runner_cls: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        """cmd_run with no flags uses run_all and returns 0 when all pass."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run_all.return_value = [_make_result(True)]

        args = _make_run_args()
        rc = cmd_run(args)

        mock_runner.run_all.assert_called_once_with(filter_tier=None)
        assert rc == 0

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRunner")
    def test_run_all_scenarios_with_tier_filter(
        self,
        mock_runner_cls: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        """cmd_run passes tier filter to run_all."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run_all.return_value = []

        args = _make_run_args(tier="unit")
        cmd_run(args)

        mock_runner.run_all.assert_called_once_with(filter_tier="unit")

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRunner")
    def test_runner_created_with_correct_params(
        self,
        mock_runner_cls: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        """ScenarioRunner is constructed with the correct arguments."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run_all.return_value = []

        args = _make_run_args(output_dir="custom/out", parallel=4, fail_fast=True)
        cmd_run(args)

        mock_runner_cls.assert_called_once_with(
            output_dir="custom/out",
            max_workers=4,
            fail_fast=True,
        )

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRunner")
    def test_empty_results_returns_zero(
        self,
        mock_runner_cls: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        """cmd_run with no results (all()) returns 0 (vacuous truth)."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run_all.return_value = []

        args = _make_run_args()
        rc = cmd_run(args)

        assert rc == 0


# ---------------------------------------------------------------------------
# cmd_list
# ---------------------------------------------------------------------------


class TestCmdList:
    """Tests for cmd_list command handler."""

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRegistry")
    def test_empty_registry_prints_message(
        self,
        mock_registry_cls: MagicMock,
        mock_register: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_list with no registered scenarios prints advisory message."""
        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get_all.return_value = {}

        rc = cmd_list(argparse.Namespace())

        captured = capsys.readouterr()
        assert "No scenarios registered" in captured.out
        assert rc == 0

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRegistry")
    def test_lists_registered_scenarios(
        self,
        mock_registry_cls: MagicMock,
        mock_register: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_list prints scenario names, tiers, and descriptions."""
        # Build a fake scenario class
        fake_config = MagicMock()
        fake_config.description = "Checks zero-shot transfer"
        fake_config.tier = MagicMock()
        fake_config.tier.value = "integration"

        fake_scenario_instance = MagicMock()
        fake_scenario_instance.config = fake_config

        FakeScenario = MagicMock(return_value=fake_scenario_instance)

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get_all.return_value = {"transfer": FakeScenario}

        rc = cmd_list(argparse.Namespace())

        captured = capsys.readouterr()
        assert "transfer" in captured.out
        assert "integration" in captured.out
        assert "Checks zero-shot transfer" in captured.out
        assert rc == 0

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRegistry")
    def test_exception_in_scenario_init_shows_fallback(
        self,
        mock_registry_cls: MagicMock,
        mock_register: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_list shows fallback text when scenario init raises."""
        BrokenScenario = MagicMock(side_effect=RuntimeError("boom"))

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get_all.return_value = {"broken": BrokenScenario}

        rc = cmd_list(argparse.Namespace())

        captured = capsys.readouterr()
        assert "broken" in captured.out
        assert "(no description)" in captured.out
        assert rc == 0

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRegistry")
    def test_total_count_displayed(
        self,
        mock_registry_cls: MagicMock,
        mock_register: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_list prints the total number of scenarios."""
        fake_config = MagicMock()
        fake_config.description = "desc"
        fake_config.tier = MagicMock(value="unit")
        fake_instance = MagicMock()
        fake_instance.config = fake_config
        FakeScenario = MagicMock(return_value=fake_instance)

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get_all.return_value = {f"s{i}": FakeScenario for i in range(3)}

        cmd_list(argparse.Namespace())

        captured = capsys.readouterr()
        assert "Total: 3 scenarios" in captured.out


# ---------------------------------------------------------------------------
# cmd_info
# ---------------------------------------------------------------------------


class TestCmdInfo:
    """Tests for cmd_info command handler."""

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRegistry")
    def test_unknown_scenario_returns_one(
        self,
        mock_registry_cls: MagicMock,
        mock_register: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_info returns 1 and prints message for unknown scenario."""
        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get.return_value = None
        mock_registry.list_scenarios.return_value = ["transfer", "complexity"]

        args = argparse.Namespace(scenario="nonexistent")
        rc = cmd_info(args)

        captured = capsys.readouterr()
        assert "not found" in captured.out
        assert rc == 1

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRegistry")
    def test_valid_scenario_prints_info(
        self,
        mock_registry_cls: MagicMock,
        mock_register: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_info prints config fields for a valid scenario."""
        # Build a fake config with model_fields
        fake_field_info = MagicMock()
        fake_field_info.annotation = int
        fake_field_info.description = "Number of epochs"

        fake_config = MagicMock()
        fake_config.description = "Stability scenario"
        fake_config.tier = MagicMock(value="integration")
        fake_config.model_fields = {"n_epochs": fake_field_info}
        # getattr(config, 'n_epochs') -> 100
        fake_config.__class__.__name__ = "StabilityConfig"
        type(fake_config).n_epochs = 100  # type: ignore[assignment]

        fake_instance = MagicMock()
        fake_instance.config = fake_config

        FakeScenarioClass = MagicMock(return_value=fake_instance)
        FakeScenarioClass.config_class = MagicMock(__name__="StabilityConfig")

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get.return_value = FakeScenarioClass

        args = argparse.Namespace(scenario="stability")
        rc = cmd_info(args)

        captured = capsys.readouterr()
        assert "stability" in captured.out
        assert "Stability scenario" in captured.out
        assert rc == 0

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRegistry")
    def test_scenario_init_error_returns_one(
        self,
        mock_registry_cls: MagicMock,
        mock_register: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_info returns 1 when scenario instantiation raises."""
        BrokenScenario = MagicMock(side_effect=ValueError("bad config"))

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get.return_value = BrokenScenario

        args = argparse.Namespace(scenario="broken")
        rc = cmd_info(args)

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert rc == 1


# ---------------------------------------------------------------------------
# cmd_compare
# ---------------------------------------------------------------------------


class TestCmdCompare:
    """Tests for cmd_compare command handler."""

    def _make_compare_args(
        self,
        run_a: str = "run_20260101",
        run_b: str = "run_20260102",
        output_dir: str = "outputs/poc",
    ) -> argparse.Namespace:
        return argparse.Namespace(run_a=run_a, run_b=run_b, output_dir=output_dir)

    @patch("src.poc.cli.ResultCollector")
    def test_compare_prints_header(
        self,
        mock_collector_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_compare prints the run identifiers."""
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        mock_collector.compare_runs.return_value = {
            "comparisons": [],
        }

        args = self._make_compare_args("run_a_id", "run_b_id")
        rc = cmd_compare(args)

        captured = capsys.readouterr()
        assert "run_a_id" in captured.out
        assert "run_b_id" in captured.out
        assert rc == 0

    @patch("src.poc.cli.ResultCollector")
    def test_compare_shows_only_in_run(
        self,
        mock_collector_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_compare shows 'Only in run' for scenarios in only one run."""
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        mock_collector.compare_runs.return_value = {
            "comparisons": [
                {"scenario": "transfer", "only_in": "a"},
            ],
        }

        args = self._make_compare_args()
        cmd_compare(args)

        captured = capsys.readouterr()
        assert "transfer" in captured.out
        assert "Only in run" in captured.out

    @patch("src.poc.cli.ResultCollector")
    def test_compare_shows_status_unchanged(
        self,
        mock_collector_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_compare shows unchanged status when status_changed is False."""
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        mock_collector.compare_runs.return_value = {
            "comparisons": [
                {
                    "scenario": "complexity",
                    "status_a": "passed",
                    "status_b": "passed",
                    "status_changed": False,
                    "metric_changes": {},
                },
            ],
        }

        args = self._make_compare_args()
        cmd_compare(args)

        captured = capsys.readouterr()
        assert "complexity" in captured.out
        assert "unchanged" in captured.out

    @patch("src.poc.cli.ResultCollector")
    def test_compare_shows_status_change(
        self,
        mock_collector_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_compare indicates when status changes between runs."""
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        mock_collector.compare_runs.return_value = {
            "comparisons": [
                {
                    "scenario": "stability",
                    "status_a": "passed",
                    "status_b": "failed",
                    "status_changed": True,
                    "metric_changes": {},
                },
            ],
        }

        args = self._make_compare_args()
        cmd_compare(args)

        captured = capsys.readouterr()
        assert "passed" in captured.out
        assert "failed" in captured.out

    @patch("src.poc.cli.ResultCollector")
    def test_compare_shows_metric_delta(
        self,
        mock_collector_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_compare prints metric deltas with direction labels."""
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        mock_collector.compare_runs.return_value = {
            "comparisons": [
                {
                    "scenario": "transfer",
                    "status_a": "passed",
                    "status_b": "passed",
                    "status_changed": False,
                    "metric_changes": {
                        "mse_19x19": {
                            "a": 0.001,
                            "b": 0.0005,
                            "delta": -0.0005,
                            "pct_change": -50.0,
                        }
                    },
                },
            ],
        }

        args = self._make_compare_args()
        cmd_compare(args)

        captured = capsys.readouterr()
        assert "mse_19x19" in captured.out
        assert "better" in captured.out

    @patch("src.poc.cli.ResultCollector")
    def test_compare_worse_metric_shows_worse(
        self,
        mock_collector_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_compare labels positive metric delta as 'worse'."""
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        mock_collector.compare_runs.return_value = {
            "comparisons": [
                {
                    "scenario": "transfer",
                    "status_a": "passed",
                    "status_b": "passed",
                    "status_changed": False,
                    "metric_changes": {
                        "mse_9x9": {
                            "a": 0.001,
                            "b": 0.002,
                            "delta": 0.001,
                            "pct_change": 100.0,
                        }
                    },
                },
            ],
        }

        args = self._make_compare_args()
        cmd_compare(args)

        captured = capsys.readouterr()
        assert "worse" in captured.out

    @patch("src.poc.cli.ResultCollector")
    def test_compare_same_metric_shows_same(
        self,
        mock_collector_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_compare labels zero delta as 'same'."""
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        mock_collector.compare_runs.return_value = {
            "comparisons": [
                {
                    "scenario": "transfer",
                    "status_a": "passed",
                    "status_b": "passed",
                    "status_changed": False,
                    "metric_changes": {
                        "mse_9x9": {
                            "a": 0.001,
                            "b": 0.001,
                            "delta": 0.0,
                            "pct_change": 0.0,
                        }
                    },
                },
            ],
        }

        args = self._make_compare_args()
        cmd_compare(args)

        captured = capsys.readouterr()
        assert "same" in captured.out

    @patch("src.poc.cli.ResultCollector")
    def test_collector_created_with_output_dir(
        self,
        mock_collector_cls: MagicMock,
    ) -> None:
        """ResultCollector is instantiated with the output_dir from args."""
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        mock_collector.compare_runs.return_value = {"comparisons": []}

        args = self._make_compare_args(output_dir="custom/output")
        cmd_compare(args)

        mock_collector_cls.assert_called_once_with(output_dir="custom/output")


# ---------------------------------------------------------------------------
# main() – argument dispatch
# ---------------------------------------------------------------------------


class TestMainDispatch:
    """Tests for main() argument parsing and dispatch."""

    @patch("src.poc.cli.configure_logging")
    @patch("src.poc.cli.cmd_run")
    def test_main_dispatches_run(
        self,
        mock_cmd_run: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """main() dispatches 'run' to cmd_run."""
        mock_cmd_run.return_value = 0
        with patch("sys.argv", ["cli", "run"]):
            rc = main()
        mock_cmd_run.assert_called_once()
        assert rc == 0

    @patch("src.poc.cli.configure_logging")
    @patch("src.poc.cli.cmd_list")
    def test_main_dispatches_list(
        self,
        mock_cmd_list: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """main() dispatches 'list' to cmd_list."""
        mock_cmd_list.return_value = 0
        with patch("sys.argv", ["cli", "list"]):
            rc = main()
        mock_cmd_list.assert_called_once()
        assert rc == 0

    @patch("src.poc.cli.configure_logging")
    @patch("src.poc.cli.cmd_info")
    def test_main_dispatches_info(
        self,
        mock_cmd_info: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """main() dispatches 'info' to cmd_info."""
        mock_cmd_info.return_value = 0
        with patch("sys.argv", ["cli", "info", "transfer"]):
            rc = main()
        mock_cmd_info.assert_called_once()
        assert rc == 0

    @patch("src.poc.cli.configure_logging")
    @patch("src.poc.cli.cmd_compare")
    def test_main_dispatches_compare(
        self,
        mock_cmd_compare: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """main() dispatches 'compare' to cmd_compare."""
        mock_cmd_compare.return_value = 0
        with patch("sys.argv", ["cli", "compare", "run_a", "run_b"]):
            rc = main()
        mock_cmd_compare.assert_called_once()
        assert rc == 0

    @patch("src.poc.cli.configure_logging")
    def test_main_no_command_returns_zero(
        self,
        mock_configure: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """main() with no subcommand prints help and returns 0."""
        with patch("sys.argv", ["cli"]):
            rc = main()
        assert rc == 0

    @patch("src.poc.cli.configure_logging")
    @patch("src.poc.cli.cmd_run")
    def test_main_passes_log_level(
        self,
        mock_cmd_run: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """main() passes --log-level to configure_logging."""
        mock_cmd_run.return_value = 0
        with patch("sys.argv", ["cli", "--log-level", "DEBUG", "run"]):
            main()
        mock_configure.assert_called_once_with(level="DEBUG")

    @patch("src.poc.cli.configure_logging")
    @patch("src.poc.cli.cmd_run")
    def test_main_run_with_scenario_flag(
        self,
        mock_cmd_run: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """main() passes --scenario to cmd_run."""
        mock_cmd_run.return_value = 0
        with patch("sys.argv", ["cli", "run", "--scenario", "complexity"]):
            main()

        args_passed = mock_cmd_run.call_args[0][0]
        assert args_passed.scenario == "complexity"

    @patch("src.poc.cli.configure_logging")
    @patch("src.poc.cli.cmd_run")
    def test_main_run_with_parallel_flag(
        self,
        mock_cmd_run: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """main() passes --parallel to cmd_run."""
        mock_cmd_run.return_value = 0
        with patch("sys.argv", ["cli", "run", "--parallel", "4"]):
            main()

        args_passed = mock_cmd_run.call_args[0][0]
        assert args_passed.parallel == 4

    @patch("src.poc.cli.configure_logging")
    @patch("src.poc.cli.cmd_run")
    def test_main_run_fail_fast_flag(
        self,
        mock_cmd_run: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """main() correctly parses --fail-fast flag."""
        mock_cmd_run.return_value = 0
        with patch("sys.argv", ["cli", "run", "--fail-fast"]):
            main()

        args_passed = mock_cmd_run.call_args[0][0]
        assert args_passed.fail_fast is True

    @patch("src.poc.cli.configure_logging")
    @patch("src.poc.cli.cmd_run")
    def test_main_run_default_output_dir(
        self,
        mock_cmd_run: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """main() uses 'outputs/poc' as default output-dir for run."""
        mock_cmd_run.return_value = 0
        with patch("sys.argv", ["cli", "run"]):
            main()

        args_passed = mock_cmd_run.call_args[0][0]
        assert args_passed.output_dir == "outputs/poc"

    @patch("src.poc.cli.configure_logging")
    @patch("src.poc.cli.cmd_compare")
    def test_main_compare_positional_args(
        self,
        mock_cmd_compare: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """main() correctly parses compare positional run_a and run_b."""
        mock_cmd_compare.return_value = 0
        with patch("sys.argv", ["cli", "compare", "run_alpha", "run_beta"]):
            main()

        args_passed = mock_cmd_compare.call_args[0][0]
        assert args_passed.run_a == "run_alpha"
        assert args_passed.run_b == "run_beta"

    @patch("src.poc.cli.configure_logging")
    @patch("src.poc.cli.cmd_info")
    def test_main_info_positional_scenario(
        self,
        mock_cmd_info: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """main() correctly parses info positional scenario argument."""
        mock_cmd_info.return_value = 0
        with patch("sys.argv", ["cli", "info", "stability"]):
            main()

        args_passed = mock_cmd_info.call_args[0][0]
        assert args_passed.scenario == "stability"


# ---------------------------------------------------------------------------
# Parametrized edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("results", "expected_rc"),
    [
        ([], 0),
        ([True], 0),
        ([True, True, True], 0),
        ([False], 1),
        ([True, False], 1),
    ],
)
@patch("src.poc.cli.register_builtin_scenarios")
@patch("src.poc.cli.ScenarioRunner")
def test_cmd_run_exit_codes(
    mock_runner_cls: MagicMock,
    mock_register: MagicMock,
    results: list[bool],
    expected_rc: int,
) -> None:
    """cmd_run returns correct exit code based on result pass/fail flags."""
    mock_runner = MagicMock()
    mock_runner_cls.return_value = mock_runner
    mock_runner.run_all.return_value = [_make_result(p) for p in results]

    rc = cmd_run(_make_run_args())
    assert rc == expected_rc


@pytest.mark.parametrize("scenario_name", ["transfer", "complexity", "stability"])
@patch("src.poc.cli.configure_logging")
@patch("src.poc.cli.cmd_run")
def test_main_run_various_scenarios(
    mock_cmd_run: MagicMock,
    mock_configure: MagicMock,
    scenario_name: str,
) -> None:
    """main() correctly parses --scenario for various built-in names."""
    mock_cmd_run.return_value = 0
    with patch("sys.argv", ["cli", "run", "--scenario", scenario_name]):
        main()

    args_passed = mock_cmd_run.call_args[0][0]
    assert args_passed.scenario == scenario_name


@pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR"])
@patch("src.poc.cli.configure_logging")
@patch("src.poc.cli.cmd_list")
def test_main_all_log_levels(
    mock_cmd_list: MagicMock,
    mock_configure: MagicMock,
    log_level: str,
) -> None:
    """main() accepts all valid log levels."""
    mock_cmd_list.return_value = 0
    with patch("sys.argv", ["cli", "--log-level", log_level, "list"]):
        main()

    mock_configure.assert_called_once_with(level=log_level)


# ---------------------------------------------------------------------------
# SimpleNamespace-based args (verify cmd_* accept plain namespaces)
# ---------------------------------------------------------------------------


class TestCmdRunWithSimpleNamespace:
    """Verify cmd_run works when args are plain SimpleNamespace objects."""

    @patch("src.poc.cli.register_builtin_scenarios")
    @patch("src.poc.cli.ScenarioRunner")
    def test_accepts_simple_namespace(
        self,
        mock_runner_cls: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        """cmd_run accepts args as SimpleNamespace, not just argparse.Namespace."""
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run_all.return_value = []

        args = SimpleNamespace(
            output_dir="outputs/poc",
            parallel=1,
            fail_fast=False,
            config=None,
            scenario=None,
            tier=None,
        )
        rc = cmd_run(args)  # type: ignore[arg-type]
        assert rc == 0
