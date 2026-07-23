"""Tests for the agent scaffolding helper and CLI command."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.agents.scaffold import (
    ScaffoldPlan,
    class_name_for,
    normalize_agent_name,
    render_agent_module,
    render_agent_test,
    scaffold_agent,
)


class TestNameNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("my_agent", "my_agent"),
            ("My Agent", "my_agent"),
            ("my-agent", "my_agent"),
            ("  Spaced  Name ", "spaced_name"),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert normalize_agent_name(raw) == expected

    @pytest.mark.parametrize("bad", ["1agent", "", "!!", "9"])
    def test_invalid_names_raise(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Invalid agent name"):
            normalize_agent_name(bad)

    def test_class_name(self) -> None:
        assert class_name_for("my_agent") == "MyAgentAgent"
        assert class_name_for("solver") == "SolverAgent"


class TestRenderers:
    def test_rendered_module_is_valid_python(self) -> None:
        src = render_agent_module("my_agent", "MyAgentAgent")
        # Parses without SyntaxError and defines the class + factory.
        tree = ast.parse(src)
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        assert "MyAgentAgent" in names

    def test_rendered_test_is_valid_python(self) -> None:
        ast.parse(render_agent_test("my_agent", "MyAgentAgent"))


class TestScaffoldAgent:
    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        plan = scaffold_agent("probe_agent", root=tmp_path, dry_run=True)
        assert isinstance(plan, ScaffoldPlan)
        assert plan.name == "probe_agent"
        assert plan.class_name == "ProbeAgentAgent"
        for path in plan.paths:
            assert not path.exists()

    def test_writes_three_mirrored_files(self, tmp_path: Path) -> None:
        plan = scaffold_agent("probe_agent", root=tmp_path)
        created = plan.paths
        assert (tmp_path / "src" / "agents" / "probe_agent.py") in created
        assert (tmp_path / "tests" / "agents" / "test_probe_agent.py") in created
        assert (tmp_path / "specs" / "probe_agent.spec.md") in created
        for path in created:
            assert path.exists() and path.read_text(encoding="utf-8")

    def test_refuses_to_overwrite(self, tmp_path: Path) -> None:
        scaffold_agent("probe_agent", root=tmp_path)
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            scaffold_agent("probe_agent", root=tmp_path)

    def test_generated_agent_imports_and_runs(self, tmp_path: Path) -> None:
        """The generated module + test are internally consistent."""
        plan = scaffold_agent("probe_agent", root=tmp_path)
        module_src = plan.files[tmp_path / "src" / "agents" / "probe_agent.py"]
        namespace: dict[str, object] = {}
        # Execute the rendered module against the real package deps.
        exec(compile(module_src, "probe_agent.py", "exec"), namespace)  # noqa: S102
        build = namespace["build_default_config"]
        agent_cls = namespace["ProbeAgentAgent"]
        config = build()  # type: ignore[operator]
        config.max_steps = 2
        agent = agent_cls(config)  # type: ignore[operator]
        state = agent.run()
        assert state.step == 2


class TestScaffoldCLI:
    def test_cli_dry_run(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from src.agents.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["scaffold", "cli_probe", "--root", str(tmp_path), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Would create" in result.stdout
        assert not (tmp_path / "src" / "agents" / "cli_probe.py").exists()
