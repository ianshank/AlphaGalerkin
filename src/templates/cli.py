"""CLI template utilities for AlphaGalerkin modules.

This module provides reusable CLI patterns using Typer:
- Standard CLI app setup with common options
- Configuration loading from YAML/CLI
- Progress bars and output formatting
- Error handling and exit codes

Example:
    from src.templates.cli import create_cli_app, add_common_options

    app = create_cli_app("my_module", "Description of my module")

    @app.command()
    @add_common_options
    def run(
        config: Path = typer.Option(..., help="Config file"),
        verbose: bool = False,
        debug: bool = False,
    ):
        # Your command implementation
        pass

    if __name__ == "__main__":
        app()

"""

from __future__ import annotations

import functools
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.templates.logging import configure_module_logging

# Type for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Rich console for output
console = Console()
error_console = Console(stderr=True)


def create_cli_app(
    name: str,
    help_text: str,
    version: str = "0.1.0",
) -> typer.Typer:
    """Create a Typer CLI app with standard configuration.

    Args:
        name: Name of the CLI app.
        help_text: Help text for the app.
        version: Version string.

    Returns:
        Configured Typer app.

    Example:
        app = create_cli_app(
            "my_module",
            "My module CLI for doing things",
            version="1.0.0",
        )

    """
    app = typer.Typer(
        name=name,
        help=help_text,
        add_completion=True,
        no_args_is_help=True,
        rich_markup_mode="rich",
    )

    # Add version callback
    def version_callback(value: bool) -> None:
        if value:
            console.print(f"{name} version {version}")
            raise typer.Exit()

    @app.callback()
    def main(
        version: bool = typer.Option(
            False,
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ) -> None:
        """CLI callback for global options."""
        pass

    return app


def add_common_options(func: F) -> F:
    """Decorator to add common CLI options to a command.

    Adds:
    - --verbose / -V: Enable verbose output
    - --debug / -d: Enable debug mode
    - --quiet / -q: Suppress non-essential output
    - --log-format: Log format (console/json)

    Example:
        @app.command()
        @add_common_options
        def my_command(verbose: bool, debug: bool, quiet: bool):
            pass

    """

    @functools.wraps(func)
    def wrapper(
        *args: Any,
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-V",
            help="Enable verbose output.",
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            "-d",
            help="Enable debug mode with detailed logging.",
        ),
        quiet: bool = typer.Option(
            False,
            "--quiet",
            "-q",
            help="Suppress non-essential output.",
        ),
        log_format: str = typer.Option(
            "console",
            "--log-format",
            help="Log format: console or json.",
        ),
        **kwargs: Any,
    ) -> Any:
        # Configure logging based on options
        level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
        if quiet:
            level = "ERROR"

        configure_module_logging(
            level=level,
            json_format=(log_format == "json"),
        )

        return func(*args, verbose=verbose, debug=debug, quiet=quiet, **kwargs)

    return cast(F, wrapper)


def load_config_file(
    path: Path,
    config_class: type,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """Load configuration from a YAML or JSON file.

    Args:
        path: Path to configuration file.
        config_class: Pydantic model class to parse into.
        overrides: Optional dict of overrides to apply.

    Returns:
        Parsed configuration object.

    Raises:
        typer.Exit: If file not found or parsing fails.

    """
    import yaml

    if not path.exists():
        error_console.print(f"[red]Error:[/red] Config file not found: {path}")
        raise typer.Exit(1)

    try:
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                error_console.print(f"[red]Error:[/red] Unsupported config format: {path.suffix}")
                raise typer.Exit(1)

        # Apply overrides
        if overrides:
            data = {**data, **overrides}

        return config_class(**data)

    except Exception as e:
        error_console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)


def print_result_table(
    title: str,
    results: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> None:
    """Print results as a formatted table.

    Args:
        title: Table title.
        results: List of dictionaries with result data.
        columns: Optional list of column names to show.

    """
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return

    table = Table(title=title, show_header=True, header_style="bold cyan")

    # Determine columns
    if columns is None:
        columns = list(results[0].keys())

    for col in columns:
        table.add_column(col.replace("_", " ").title())

    for result in results:
        row = []
        for col in columns:
            value = result.get(col, "")
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            elif isinstance(value, bool):
                row.append("[green]Yes[/green]" if value else "[red]No[/red]")
            else:
                row.append(str(value))
        table.add_row(*row)

    console.print(table)


def print_status_panel(
    title: str,
    status: str,
    details: dict[str, Any] | None = None,
    success: bool = True,
) -> None:
    """Print a status panel with optional details.

    Args:
        title: Panel title.
        status: Status message.
        details: Optional details dictionary.
        success: If True, use green border; else red.

    """
    color = "green" if success else "red"

    content_lines = [f"[bold]{status}[/bold]"]
    if details:
        content_lines.append("")
        for key, value in details.items():
            if isinstance(value, float):
                content_lines.append(f"  {key}: {value:.4f}")
            else:
                content_lines.append(f"  {key}: {value}")

    content = "\n".join(content_lines)
    console.print(Panel(content, title=title, border_style=color))


def create_progress_context(
    description: str = "Processing",
) -> Progress:
    """Create a progress context for long-running operations.

    Args:
        description: Default task description.

    Returns:
        Rich Progress context manager.

    Example:
        with create_progress_context("Training") as progress:
            task = progress.add_task("Training model", total=100)
            for i in range(100):
                progress.update(task, advance=1)

    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


def confirm_action(
    message: str,
    default: bool = False,
) -> bool:
    """Prompt user for confirmation.

    Args:
        message: Confirmation message.
        default: Default value if user presses Enter.

    Returns:
        True if confirmed, False otherwise.

    """
    return bool(typer.confirm(message, default=default))


def handle_keyboard_interrupt(func: F) -> F:
    """Decorator to handle keyboard interrupts gracefully.

    Example:
        @app.command()
        @handle_keyboard_interrupt
        def long_running_command():
            pass

    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            raise typer.Exit(130)

    return cast(F, wrapper)


def with_error_handling(func: F) -> F:
    """Decorator to handle errors and provide user-friendly messages.

    Example:
        @app.command()
        @with_error_handling
        def risky_command():
            pass

    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = structlog.get_logger(__name__)
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise  # Re-raise Typer exits
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            raise typer.Exit(130)
        except Exception as e:
            logger.exception("command_failed", error=str(e))
            error_console.print(f"[red]Error:[/red] {e}")

            # In debug mode, show full traceback
            if "--debug" in sys.argv or "-d" in sys.argv:
                console.print_exception()

            raise typer.Exit(1)

    return cast(F, wrapper)


# Common CLI argument types
ConfigPath = typer.Option(
    ...,
    "--config",
    "-c",
    help="Path to configuration file (YAML or JSON).",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
)

OutputDir = typer.Option(
    Path("output"),
    "--output",
    "-o",
    help="Output directory for results.",
    file_okay=False,
    dir_okay=True,
)

DryRun = typer.Option(
    False,
    "--dry-run",
    help="Simulate execution without making changes.",
)

Force = typer.Option(
    False,
    "--force",
    "-f",
    help="Force operation without confirmation.",
)
