"""Tests for the templates.cli module (src/templates/cli.py).

Validates:
    - create_cli_app returns a configured Typer app with version support
    - add_common_options decorator wires logging configuration correctly
    - load_config_file handles YAML, JSON, missing files, and parse errors
    - print_result_table renders tables and handles empty results
    - print_status_panel renders success/failure panels
    - create_progress_context returns a Rich Progress object
    - confirm_action delegates to typer.confirm
    - handle_keyboard_interrupt catches KeyboardInterrupt
    - with_error_handling catches exceptions and re-raises typer.Exit
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from pydantic import BaseModel
from typer.testing import CliRunner

from src.templates.cli import (
    add_common_options,
    confirm_action,
    create_cli_app,
    create_progress_context,
    handle_keyboard_interrupt,
    load_config_file,
    print_result_table,
    print_status_panel,
    with_error_handling,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli_runner() -> CliRunner:
    """Return a Typer test runner."""
    return CliRunner()


@pytest.fixture()
def simple_config_class() -> type[BaseModel]:
    """Return a simple Pydantic model for config loading tests."""

    class _Config(BaseModel):
        name: str = "default"
        value: int = 42

    return _Config


@pytest.fixture()
def json_config_file(simple_config_class: type[BaseModel], tmp_path: Path) -> Path:
    """Write a JSON config file and return its path."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"name": "from_json", "value": 100}))
    return config_path


@pytest.fixture()
def yaml_config_file(tmp_path: Path) -> Path:
    """Write a YAML config file and return its path."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: from_yaml\nvalue: 99\n")
    return config_path


@pytest.fixture()
def yml_config_file(tmp_path: Path) -> Path:
    """Write a .yml config file and return its path."""
    config_path = tmp_path / "config.yml"
    config_path.write_text("name: from_yml\nvalue: 77\n")
    return config_path


# ---------------------------------------------------------------------------
# create_cli_app
# ---------------------------------------------------------------------------


class TestCreateCliApp:
    """Tests for create_cli_app factory function."""

    def test_returns_typer_app(self) -> None:
        """create_cli_app returns a Typer instance."""
        app = create_cli_app("myapp", "My application")
        assert isinstance(app, typer.Typer)

    def test_version_callback_prints_version(self, cli_runner: CliRunner) -> None:
        """--version flag prints name and version then exits."""
        app = create_cli_app("myapp", "My application", version="2.3.4")
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "myapp" in result.output
        assert "2.3.4" in result.output

    def test_default_version_is_used(self, cli_runner: CliRunner) -> None:
        """Default version string is applied when version is not given."""
        app = create_cli_app("tool", "A tool")
        result = cli_runner.invoke(app, ["--version"])
        assert "0.1.0" in result.output

    def test_help_text_is_set(self, cli_runner: CliRunner) -> None:
        """Help text appears in the app's help output."""
        app = create_cli_app("myapp", "Unique help text here", version="1.0.0")

        @app.command()
        def dummy_cmd() -> None:
            pass

        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Unique help text here" in result.output

    def test_different_names_produce_different_apps(self) -> None:
        """Each call to create_cli_app returns a separate Typer instance."""
        app_a = create_cli_app("app_a", "App A")
        app_b = create_cli_app("app_b", "App B")
        assert app_a is not app_b

    @pytest.mark.parametrize(
        ("name", "version"),
        [
            ("alpha", "0.1.0"),
            ("beta", "1.2.3"),
            ("gamma", "10.0.0"),
        ],
    )
    def test_version_output_matches_input(
        self, cli_runner: CliRunner, name: str, version: str
    ) -> None:
        """Version callback output contains the exact name and version strings."""
        app = create_cli_app(name, "Description", version=version)
        result = cli_runner.invoke(app, ["--version"])
        assert name in result.output
        assert version in result.output


# ---------------------------------------------------------------------------
# add_common_options
# ---------------------------------------------------------------------------


class TestAddCommonOptions:
    """Tests for the add_common_options decorator."""

    def _make_decorated_func(self) -> Any:
        """Return a simple function wrapped with add_common_options."""

        @add_common_options
        def func(
            verbose: bool = False,
            debug: bool = False,
            quiet: bool = False,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {"verbose": verbose, "debug": debug, "quiet": quiet}

        return func

    @patch("src.templates.cli.configure_module_logging")
    def test_verbose_sets_info_level(self, mock_log: MagicMock) -> None:
        """verbose=True sets logging level to INFO."""
        func = self._make_decorated_func()
        func(verbose=True, debug=False, quiet=False, log_format="console")
        mock_log.assert_called_once_with(level="INFO", json_format=False)

    @patch("src.templates.cli.configure_module_logging")
    def test_debug_sets_debug_level(self, mock_log: MagicMock) -> None:
        """debug=True sets logging level to DEBUG."""
        func = self._make_decorated_func()
        func(verbose=False, debug=True, quiet=False, log_format="console")
        mock_log.assert_called_once_with(level="DEBUG", json_format=False)

    @patch("src.templates.cli.configure_module_logging")
    def test_quiet_sets_error_level(self, mock_log: MagicMock) -> None:
        """quiet=True sets logging level to ERROR."""
        func = self._make_decorated_func()
        func(verbose=False, debug=False, quiet=True, log_format="console")
        mock_log.assert_called_once_with(level="ERROR", json_format=False)

    @patch("src.templates.cli.configure_module_logging")
    def test_default_sets_warning_level(self, mock_log: MagicMock) -> None:
        """With no flags set, logging level is WARNING."""
        func = self._make_decorated_func()
        func(verbose=False, debug=False, quiet=False, log_format="console")
        mock_log.assert_called_once_with(level="WARNING", json_format=False)

    @patch("src.templates.cli.configure_module_logging")
    def test_json_format_flag(self, mock_log: MagicMock) -> None:
        """log_format='json' sets json_format=True."""
        func = self._make_decorated_func()
        func(verbose=False, debug=False, quiet=False, log_format="json")
        mock_log.assert_called_once_with(level="WARNING", json_format=True)

    @patch("src.templates.cli.configure_module_logging")
    def test_debug_takes_priority_over_verbose(self, mock_log: MagicMock) -> None:
        """When debug=True, level is DEBUG regardless of verbose."""
        func = self._make_decorated_func()
        func(verbose=True, debug=True, quiet=False, log_format="console")
        # debug branch is evaluated first
        mock_log.assert_called_once_with(level="DEBUG", json_format=False)

    @patch("src.templates.cli.configure_module_logging")
    def test_wraps_preserves_function_name(self, mock_log: MagicMock) -> None:
        """Decorated function retains original __name__ via functools.wraps."""

        @add_common_options
        def my_special_func(verbose: bool = False, **kw: Any) -> None:
            pass

        assert my_special_func.__name__ == "my_special_func"

    def test_wrapped_function_via_typer_app(self, cli_runner: CliRunner) -> None:
        """Decorator works correctly when registered as a Typer command."""
        app = create_cli_app("testapp", "Test app")

        @app.command()
        @add_common_options
        def run_cmd(verbose: bool = False, debug: bool = False, quiet: bool = False) -> None:
            pass

        with patch("src.templates.cli.configure_module_logging"):
            result = cli_runner.invoke(app, ["run-cmd", "--verbose"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# load_config_file
# ---------------------------------------------------------------------------


class TestLoadConfigFile:
    """Tests for load_config_file utility."""

    def test_load_json_file(
        self,
        json_config_file: Path,
        simple_config_class: type[BaseModel],
    ) -> None:
        """JSON config file is parsed into the config class."""
        config = load_config_file(json_config_file, simple_config_class)
        assert config.name == "from_json"  # type: ignore[attr-defined]
        assert config.value == 100  # type: ignore[attr-defined]

    def test_load_yaml_file(
        self,
        yaml_config_file: Path,
        simple_config_class: type[BaseModel],
    ) -> None:
        """YAML config file is parsed into the config class."""
        config = load_config_file(yaml_config_file, simple_config_class)
        assert config.name == "from_yaml"  # type: ignore[attr-defined]
        assert config.value == 99  # type: ignore[attr-defined]

    def test_load_yml_extension(
        self,
        yml_config_file: Path,
        simple_config_class: type[BaseModel],
    ) -> None:
        """.yml extension is treated the same as .yaml."""
        config = load_config_file(yml_config_file, simple_config_class)
        assert config.name == "from_yml"  # type: ignore[attr-defined]

    def test_missing_file_raises_exit(
        self,
        tmp_path: Path,
        simple_config_class: type[BaseModel],
    ) -> None:
        """Non-existent file raises typer.Exit(1)."""
        missing = tmp_path / "nonexistent.json"
        with pytest.raises(typer.Exit) as exc_info:
            load_config_file(missing, simple_config_class)
        assert exc_info.value.exit_code == 1

    def test_unsupported_extension_raises_exit(
        self,
        tmp_path: Path,
        simple_config_class: type[BaseModel],
    ) -> None:
        """Unsupported file extension raises typer.Exit(1)."""
        bad_file = tmp_path / "config.toml"
        bad_file.write_text("name = 'test'\n")
        with pytest.raises(typer.Exit) as exc_info:
            load_config_file(bad_file, simple_config_class)
        assert exc_info.value.exit_code == 1

    def test_overrides_are_applied(
        self,
        json_config_file: Path,
        simple_config_class: type[BaseModel],
    ) -> None:
        """Overrides dict is merged into the loaded data."""
        config = load_config_file(
            json_config_file,
            simple_config_class,
            overrides={"value": 999},
        )
        assert config.value == 999  # type: ignore[attr-defined]
        assert config.name == "from_json"  # type: ignore[attr-defined]

    def test_overrides_none_is_safe(
        self,
        json_config_file: Path,
        simple_config_class: type[BaseModel],
    ) -> None:
        """Passing overrides=None does not raise."""
        config = load_config_file(json_config_file, simple_config_class, overrides=None)
        assert config.value == 100  # type: ignore[attr-defined]

    def test_invalid_json_raises_exit(
        self,
        tmp_path: Path,
        simple_config_class: type[BaseModel],
    ) -> None:
        """Malformed JSON raises typer.Exit(1)."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json}")
        with pytest.raises(typer.Exit) as exc_info:
            load_config_file(bad_json, simple_config_class)
        assert exc_info.value.exit_code == 1

    def test_validation_error_raises_exit(
        self,
        tmp_path: Path,
    ) -> None:
        """Data that fails Pydantic validation raises typer.Exit(1)."""

        class _StrictConfig(BaseModel):
            count: int

        bad_json = tmp_path / "bad_data.json"
        bad_json.write_text(json.dumps({"count": "not_an_int_that_pydantic_cannot_coerce"}))
        with pytest.raises(typer.Exit) as exc_info:
            load_config_file(bad_json, _StrictConfig)
        assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# print_result_table
# ---------------------------------------------------------------------------


class TestPrintResultTable:
    """Tests for print_result_table helper."""

    def test_empty_results_prints_no_results_message(self) -> None:
        """Empty results list prints advisory message."""
        with patch("src.templates.cli.console") as mock_console:
            print_result_table("My Table", [])
            mock_console.print.assert_called_once()
            arg = mock_console.print.call_args[0][0]
            assert "No results" in arg

    def test_non_empty_results_prints_table(self) -> None:
        """Non-empty results causes console.print to be called with a Table."""
        from rich.table import Table

        with patch("src.templates.cli.console") as mock_console:
            print_result_table(
                "Results",
                [{"scenario": "transfer", "mse": 0.001, "passed": True}],
            )
            mock_console.print.assert_called_once()
            table_arg = mock_console.print.call_args[0][0]
            assert isinstance(table_arg, Table)

    def test_explicit_columns_respected(self) -> None:
        """Only specified columns appear in the table."""
        from rich.table import Table

        with patch("src.templates.cli.console") as mock_console:
            print_result_table(
                "Results",
                [{"name": "a", "score": 0.9, "extra": "ignored"}],
                columns=["name", "score"],
            )
            table_arg = mock_console.print.call_args[0][0]
            assert isinstance(table_arg, Table)
            col_names = [col.header for col in table_arg.columns]
            assert "Name" in col_names
            assert "Score" in col_names

    def test_float_values_are_formatted(self) -> None:
        """Float values in results are formatted to 4 decimal places."""
        from io import StringIO

        from rich.console import Console
        from rich.table import Table

        captured_table: list[Table] = []

        def capture_print(obj: Any) -> None:
            if isinstance(obj, Table):
                captured_table.append(obj)

        with patch("src.templates.cli.console") as mock_console:
            mock_console.print.side_effect = capture_print
            print_result_table("T", [{"loss": 0.123456789}])

        assert captured_table, "Expected a Table to be printed"
        sio = StringIO()
        console = Console(file=sio, no_color=True)
        console.print(captured_table[0])
        assert "0.1235" in sio.getvalue()

    def test_bool_values_are_formatted(self) -> None:
        """Bool values are rendered as Yes/No markup strings."""
        from io import StringIO

        from rich.console import Console
        from rich.table import Table

        captured_table: list[Table] = []

        def capture_print(obj: Any) -> None:
            if isinstance(obj, Table):
                captured_table.append(obj)

        with patch("src.templates.cli.console") as mock_console:
            mock_console.print.side_effect = capture_print
            print_result_table(
                "T",
                [{"passed": True}, {"passed": False}],
                columns=["passed"],
            )

        assert captured_table, "Expected a Table to be printed"
        sio = StringIO()
        console = Console(file=sio, no_color=True)
        console.print(captured_table[0])
        output = sio.getvalue()
        assert "Yes" in output
        assert "No" in output

    def test_missing_column_value_uses_empty_string(self) -> None:
        """If a row is missing a column key, an empty string is used."""
        from rich.table import Table

        with patch("src.templates.cli.console") as mock_console:
            print_result_table(
                "T",
                [{"name": "only_name"}],
                columns=["name", "missing_col"],
            )
            table_arg = mock_console.print.call_args[0][0]
            assert isinstance(table_arg, Table)


# ---------------------------------------------------------------------------
# print_status_panel
# ---------------------------------------------------------------------------


class TestPrintStatusPanel:
    """Tests for print_status_panel helper."""

    def test_success_uses_green_border(self) -> None:
        """Success panel uses green border style."""
        from rich.panel import Panel

        with patch("src.templates.cli.console") as mock_console:
            print_status_panel("Title", "All good", success=True)
            panel_arg = mock_console.print.call_args[0][0]
            assert isinstance(panel_arg, Panel)
            assert panel_arg.border_style == "green"

    def test_failure_uses_red_border(self) -> None:
        """Failure panel uses red border style."""
        from rich.panel import Panel

        with patch("src.templates.cli.console") as mock_console:
            print_status_panel("Error", "Something failed", success=False)
            panel_arg = mock_console.print.call_args[0][0]
            assert isinstance(panel_arg, Panel)
            assert panel_arg.border_style == "red"

    def test_title_is_set(self) -> None:
        """Panel title is set from the title argument."""
        from rich.panel import Panel

        with patch("src.templates.cli.console") as mock_console:
            print_status_panel("My Panel Title", "status", success=True)
            panel_arg = mock_console.print.call_args[0][0]
            assert isinstance(panel_arg, Panel)
            assert "My Panel Title" in str(panel_arg.title)

    def test_details_with_float_values(self) -> None:
        """Float values in details are formatted to 4 decimal places."""
        from rich.panel import Panel

        with patch("src.templates.cli.console") as mock_console:
            print_status_panel(
                "Result",
                "done",
                details={"mse": 0.123456},
                success=True,
            )
            panel_arg = mock_console.print.call_args[0][0]
            assert isinstance(panel_arg, Panel)
            content_str = str(panel_arg.renderable)
            assert "0.1235" in content_str

    def test_details_with_non_float_values(self) -> None:
        """Non-float values in details are rendered as strings."""
        from rich.panel import Panel

        with patch("src.templates.cli.console") as mock_console:
            print_status_panel(
                "Result",
                "done",
                details={"name": "my_run", "count": 5},
                success=True,
            )
            panel_arg = mock_console.print.call_args[0][0]
            assert isinstance(panel_arg, Panel)
            content_str = str(panel_arg.renderable)
            assert "my_run" in content_str
            assert "5" in content_str

    def test_no_details_renders_status_only(self) -> None:
        """Panel without details still renders the status message."""
        from rich.panel import Panel

        with patch("src.templates.cli.console") as mock_console:
            print_status_panel("T", "Status message here", details=None)
            panel_arg = mock_console.print.call_args[0][0]
            assert isinstance(panel_arg, Panel)
            content_str = str(panel_arg.renderable)
            assert "Status message here" in content_str


# ---------------------------------------------------------------------------
# create_progress_context
# ---------------------------------------------------------------------------


class TestCreateProgressContext:
    """Tests for create_progress_context helper."""

    def test_returns_progress_instance(self) -> None:
        """create_progress_context returns a Rich Progress object."""
        from rich.progress import Progress

        prog = create_progress_context("Loading")
        assert isinstance(prog, Progress)

    def test_custom_description_accepted(self) -> None:
        """Custom description string is accepted without error."""
        from rich.progress import Progress

        prog = create_progress_context("Custom task description")
        assert isinstance(prog, Progress)

    def test_default_description_accepted(self) -> None:
        """Default description 'Processing' is used when none is given."""
        from rich.progress import Progress

        prog = create_progress_context()
        assert isinstance(prog, Progress)

    def test_progress_context_is_usable(self) -> None:
        """Progress context can be used as a context manager."""
        prog = create_progress_context("Working")
        with prog:
            task = prog.add_task("Testing", total=10)
            prog.update(task, advance=5)


# ---------------------------------------------------------------------------
# confirm_action
# ---------------------------------------------------------------------------


class TestConfirmAction:
    """Tests for confirm_action utility."""

    def test_returns_true_when_confirmed(self) -> None:
        """confirm_action returns True when typer.confirm returns True."""
        with patch("src.templates.cli.typer.confirm", return_value=True) as mock_confirm:
            result = confirm_action("Are you sure?")
            assert result is True
            mock_confirm.assert_called_once_with("Are you sure?", default=False)

    def test_returns_false_when_declined(self) -> None:
        """confirm_action returns False when typer.confirm returns False."""
        with patch("src.templates.cli.typer.confirm", return_value=False):
            result = confirm_action("Continue?")
            assert result is False

    def test_custom_default_passed_through(self) -> None:
        """Default argument is passed to typer.confirm."""
        with patch("src.templates.cli.typer.confirm", return_value=True) as mock_confirm:
            confirm_action("Proceed?", default=True)
            mock_confirm.assert_called_once_with("Proceed?", default=True)

    @pytest.mark.parametrize(
        ("confirm_return", "expected"),
        [
            (True, True),
            (False, False),
            (1, True),
            (0, False),
        ],
    )
    def test_result_cast_to_bool(
        self, confirm_return: Any, expected: bool
    ) -> None:
        """Result is always cast to bool."""
        with patch("src.templates.cli.typer.confirm", return_value=confirm_return):
            result = confirm_action("Ok?")
            assert result is expected


# ---------------------------------------------------------------------------
# handle_keyboard_interrupt
# ---------------------------------------------------------------------------


class TestHandleKeyboardInterrupt:
    """Tests for handle_keyboard_interrupt decorator."""

    def test_normal_function_executes(self) -> None:
        """Non-interrupted functions execute normally."""

        @handle_keyboard_interrupt
        def normal_func() -> str:
            return "ok"

        assert normal_func() == "ok"

    def test_keyboard_interrupt_raises_typer_exit_130(self) -> None:
        """KeyboardInterrupt is caught and re-raised as typer.Exit(130)."""

        @handle_keyboard_interrupt
        def interrupted_func() -> None:
            raise KeyboardInterrupt

        with patch("src.templates.cli.console"):
            with pytest.raises(typer.Exit) as exc_info:
                interrupted_func()
        assert exc_info.value.exit_code == 130

    def test_other_exceptions_propagate(self) -> None:
        """Non-KeyboardInterrupt exceptions are not caught."""

        @handle_keyboard_interrupt
        def bad_func() -> None:
            raise ValueError("not a keyboard interrupt")

        with pytest.raises(ValueError, match="not a keyboard interrupt"):
            bad_func()

    def test_wraps_preserves_function_name(self) -> None:
        """Decorated function retains original __name__."""

        @handle_keyboard_interrupt
        def my_long_running_task() -> None:
            pass

        assert my_long_running_task.__name__ == "my_long_running_task"

    def test_cancel_message_printed(self) -> None:
        """Cancellation message is printed to console on KeyboardInterrupt."""

        @handle_keyboard_interrupt
        def interrupted() -> None:
            raise KeyboardInterrupt

        with patch("src.templates.cli.console") as mock_console:
            with pytest.raises(typer.Exit):
                interrupted()
            mock_console.print.assert_called_once()
            msg = mock_console.print.call_args[0][0]
            assert "cancelled" in msg.lower() or "cancel" in msg.lower()

    def test_return_value_passed_through(self) -> None:
        """Return value of the wrapped function is passed through."""

        @handle_keyboard_interrupt
        def func_with_return() -> int:
            return 42

        assert func_with_return() == 42


# ---------------------------------------------------------------------------
# with_error_handling
# ---------------------------------------------------------------------------


class TestWithErrorHandling:
    """Tests for with_error_handling decorator."""

    def test_normal_function_executes(self) -> None:
        """Normal functions execute and return their value."""

        @with_error_handling
        def good_func() -> str:
            return "success"

        assert good_func() == "success"

    def test_exception_raises_exit_code_1(self) -> None:
        """Generic exceptions are caught and re-raised as typer.Exit(1)."""

        @with_error_handling
        def bad_func() -> None:
            raise RuntimeError("something went wrong")

        with patch("src.templates.cli.console"), patch("src.templates.cli.error_console"):
            with pytest.raises(typer.Exit) as exc_info:
                bad_func()
        assert exc_info.value.exit_code == 1

    def test_typer_exit_is_reraised(self) -> None:
        """typer.Exit is re-raised without modification."""

        @with_error_handling
        def func_that_exits() -> None:
            raise typer.Exit(5)

        with pytest.raises(typer.Exit) as exc_info:
            func_that_exits()
        assert exc_info.value.exit_code == 5

    def test_keyboard_interrupt_raises_exit_130(self) -> None:
        """KeyboardInterrupt is caught and raises typer.Exit(130)."""

        @with_error_handling
        def interrupted_func() -> None:
            raise KeyboardInterrupt

        with patch("src.templates.cli.console"):
            with pytest.raises(typer.Exit) as exc_info:
                interrupted_func()
        assert exc_info.value.exit_code == 130

    def test_error_message_printed(self) -> None:
        """Error message is printed to error_console on exception."""

        @with_error_handling
        def bad_func() -> None:
            raise ValueError("user-visible error")

        with patch("src.templates.cli.console"), patch(
            "src.templates.cli.error_console"
        ) as mock_err_console:
            with pytest.raises(typer.Exit):
                bad_func()
        mock_err_console.print.assert_called()
        printed = str(mock_err_console.print.call_args)
        assert "user-visible error" in printed

    def test_wraps_preserves_function_name(self) -> None:
        """Decorated function retains original __name__ via functools.wraps."""

        @with_error_handling
        def my_risky_command() -> None:
            pass

        assert my_risky_command.__name__ == "my_risky_command"

    def test_debug_flag_triggers_traceback_print(self) -> None:
        """When --debug is in sys.argv, console.print_exception is called."""
        import sys

        @with_error_handling
        def bad_func() -> None:
            raise RuntimeError("debug error")

        original_argv = sys.argv[:]
        try:
            sys.argv = ["prog", "--debug"]
            with patch("src.templates.cli.console") as mock_console, patch(
                "src.templates.cli.error_console"
            ):
                with pytest.raises(typer.Exit):
                    bad_func()
            mock_console.print_exception.assert_called_once()
        finally:
            sys.argv = original_argv

    def test_no_debug_flag_no_traceback_print(self) -> None:
        """Without --debug in sys.argv, console.print_exception is not called."""
        import sys

        @with_error_handling
        def bad_func() -> None:
            raise RuntimeError("normal error")

        original_argv = sys.argv[:]
        try:
            sys.argv = ["prog"]
            with patch("src.templates.cli.console") as mock_console, patch(
                "src.templates.cli.error_console"
            ):
                with pytest.raises(typer.Exit):
                    bad_func()
            mock_console.print_exception.assert_not_called()
        finally:
            sys.argv = original_argv

    @pytest.mark.parametrize(
        "exception_type",
        [ValueError, TypeError, KeyError, OSError],
    )
    def test_various_exceptions_produce_exit_code_1(
        self, exception_type: type[Exception]
    ) -> None:
        """Multiple exception types are all caught and produce Exit(1)."""

        @with_error_handling
        def bad_func() -> None:
            raise exception_type("error")

        with patch("src.templates.cli.console"), patch("src.templates.cli.error_console"):
            with pytest.raises(typer.Exit) as exc_info:
                bad_func()
        assert exc_info.value.exit_code == 1

    def test_kwargs_are_passed_through(self) -> None:
        """Keyword arguments are forwarded to the wrapped function."""

        @with_error_handling
        def func_with_kwargs(x: int = 0, y: int = 0) -> int:
            return x + y

        result = func_with_kwargs(x=3, y=4)
        assert result == 7

    def test_positional_args_passed_through(self) -> None:
        """Positional arguments are forwarded to the wrapped function."""

        @with_error_handling
        def add(a: int, b: int) -> int:
            return a + b

        assert add(5, 6) == 11


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleLevelConstants:
    """Verify that module-level CLI argument type objects are importable."""

    def test_config_path_is_defined(self) -> None:
        """ConfigPath is defined and importable."""
        from src.templates.cli import ConfigPath

        assert ConfigPath is not None

    def test_output_dir_is_defined(self) -> None:
        """OutputDir is defined and importable."""
        from src.templates.cli import OutputDir

        assert OutputDir is not None

    def test_dry_run_is_defined(self) -> None:
        """DryRun is defined and importable."""
        from src.templates.cli import DryRun

        assert DryRun is not None

    def test_force_is_defined(self) -> None:
        """Force is defined and importable."""
        from src.templates.cli import Force

        assert Force is not None
