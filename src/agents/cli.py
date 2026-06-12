"""CLI entry point for the agent-physics integration module.

Commands:
    list-agents     List all registered agent types.
    info            Show details for a specific agent type.
    run             Run a multi-physics solve from a YAML config.

Example:
    python -m src.agents.cli list-agents
    python -m src.agents.cli info solver
    python -m src.agents.cli run --config config/agents/poisson.yaml

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from src.templates.cli import (
    add_common_options,
    create_cli_app,
    load_config_file,
    print_result_table,
    print_status_panel,
    with_error_handling,
)

app = create_cli_app(
    name="agents",
    help_text="Agent-physics integration for multi-physics PDE solving.",
    version="0.1.0",
)


@app.command()
@add_common_options
@with_error_handling
def list_agents(
    verbose: bool = False,
    debug: bool = False,
    quiet: bool = False,
    log_format: str = "text",
) -> None:
    """List all registered agent types."""
    from src.agents.registry import AgentRegistry, _register_builtin_agents

    _register_builtin_agents()
    registry = AgentRegistry()
    agents = registry.list_items()

    if not agents:
        typer.echo("No agents registered.")
        return

    results = []
    for name in sorted(agents):
        cls = registry.get(name)
        desc = cls.__doc__.split("\n")[0] if cls and cls.__doc__ else "No description"
        results.append({"name": name, "description": desc})

    print_result_table(
        title="Registered Agents",
        results=results,
        columns=["name", "description"],
    )


@app.command()
@add_common_options
@with_error_handling
def info(
    agent_type: str = typer.Argument(..., help="Agent type to show info for"),
    verbose: bool = False,
    debug: bool = False,
    quiet: bool = False,
    log_format: str = "text",
) -> None:
    """Show details for a specific agent type."""
    from src.agents.registry import AgentRegistry, _register_builtin_agents

    _register_builtin_agents()
    registry = AgentRegistry()

    cls = registry.get(agent_type)
    if cls is None:
        available = registry.list_items()
        typer.echo(
            f"Agent type '{agent_type}' not found. Available: {', '.join(sorted(available))}"
        )
        raise typer.Exit(code=1)

    typer.echo(f"Agent: {agent_type}")
    typer.echo(f"Class: {cls.__module__}.{cls.__qualname__}")
    if cls.__doc__:
        typer.echo(f"\n{cls.__doc__}")


_config_option = typer.Option(
    ...,
    "--config",
    "-c",
    help="Path to YAML configuration file",
    exists=True,
)


@app.command()
@add_common_options
@with_error_handling
def run(
    config: Path = _config_option,
    verbose: bool = False,
    debug: bool = False,
    quiet: bool = False,
    log_format: str = "text",
) -> None:
    """Run a multi-physics solve from a YAML configuration."""
    from src.agents.config import OrchestratorConfig
    from src.agents.orchestrator import AgentOrchestrator

    orch_config = load_config_file(config, OrchestratorConfig)
    orchestrator = AgentOrchestrator(orch_config)
    result = orchestrator.run()

    success = result.is_success()
    print_status_panel(
        title="Agent Orchestration Result",
        status=result.status.value,
        details={
            "duration": f"{result.duration_seconds:.2f}s"
            if result.duration_seconds is not None
            else "N/A",
            "total_steps": str(result.metrics.get("total_steps", 0)),
            "budget_used": f"{result.metrics.get('budget_used', 0):.4f}",
        },
        success=success,
    )

    if result.metrics:
        metric_results = [
            {"metric": k, "value": f"{v:.6f}" if isinstance(v, float) else str(v)}
            for k, v in sorted(result.metrics.items())
        ]
        print_result_table(
            title="Metrics",
            results=metric_results,
            columns=["metric", "value"],
        )

    if not success and result.error:
        typer.echo(f"\nError: {result.error}")
        raise typer.Exit(code=1)


_research_output_dir_option = typer.Option(
    "outputs/agents/research",
    "--output-dir",
    "-o",
    help="Directory to persist the research-loop result JSON under <dir>/<run_id>/result.json.",
)


def _persist_research_result(result: Any, output_dir: Path) -> Path:
    """Write a research-loop ExecutionResult to ``<output_dir>/<run_id>/result.json``.

    Mirrors ``src/poc/results.py`` persistence so research-loop runs are
    durable and diffable by the baseline harness. Returns the written path.
    """
    run_dir = output_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "result.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
    return path


@app.command()
@add_common_options
@with_error_handling
def research(
    config: Path = _config_option,
    output_dir: Path = _research_output_dir_option,
    verbose: bool = False,
    debug: bool = False,
    quiet: bool = False,
    log_format: str = "text",
) -> None:
    """Run the centaur research-loop harness from a YAML configuration."""
    from src.agents.config import ResearchLoopConfig
    from src.agents.research_loop import ResearchLoopOrchestrator

    loop_config = load_config_file(config, ResearchLoopConfig)
    orchestrator = ResearchLoopOrchestrator(loop_config)
    result = orchestrator.run()

    result_path = _persist_research_result(result, output_dir)
    typer.echo(f"Result written to {result_path}")

    success = result.is_success()
    print_status_panel(
        title="Research-Loop Result",
        status=result.status.value,
        details={
            "duration": f"{result.duration_seconds:.2f}s"
            if result.duration_seconds is not None
            else "N/A",
            "n_problems": str(result.metrics.get("n_problems", 0)),
            "solved_fraction": f"{result.metrics.get('solved_fraction', 0):.3f}",
        },
        success=success,
    )

    if result.metrics:
        metric_results = [
            {"metric": k, "value": f"{v:.6f}" if isinstance(v, float) else str(v)}
            for k, v in sorted(result.metrics.items())
        ]
        print_result_table(
            title="Metrics",
            results=metric_results,
            columns=["metric", "value"],
        )

    if not success and result.error:
        typer.echo(f"\nError: {result.error}")
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for ``python -m src.agents.cli``."""
    app()


if __name__ == "__main__":
    main()
