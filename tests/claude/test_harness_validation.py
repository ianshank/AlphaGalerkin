"""Deterministic validation of the `.claude/` agentic harness.

The harness — 9 skills, 5 subagents, 4 slash commands, a SessionStart hook and
`settings.json` — had **no tests at all**. It is executable configuration: a
skill that cites a deleted path, an agent declaring a tool that does not exist,
or a permission entry naming a module that was renamed all fail *at the moment
someone relies on them*, which is the worst possible time to find out.

Design notes, because they are what make this suite worth having:

- **Data-driven, not enumerated.** Every test parametrizes over files discovered
  on disk, so a new skill is validated the moment it is added. A hardcoded list
  would drift exactly like the doc claims this repo keeps having to correct.
- **Deterministic and hermetic.** No network, no subprocess against the model,
  no wall-clock. Every assertion is a pure function of files in the tree, so a
  failure means the tree changed, never that a runner was slow.
- **Forward references are allowed but must be declared.** The
  `certificate-validation` skill instructs the reader to *create*
  `src/pde/certificate/`, so a naive existence check flags it. Rather than drop
  the check (which is what makes `check_doc_links.py` unusable as a gate today —
  see CLAUDE.md Next Steps, 105 false positives), each exemption is listed with
  a reason and is itself asserted to still be a forward reference.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAUDE_DIR = REPO_ROOT / ".claude"

SKILLS = sorted(CLAUDE_DIR.glob("skills/*/SKILL.md"))
AGENTS = sorted(CLAUDE_DIR.glob("agents/*.md"))
COMMANDS = sorted(CLAUDE_DIR.glob("commands/*.md"))
ALL_MARKDOWN = sorted(CLAUDE_DIR.rglob("*.md"))

# Tool names Claude Code actually provides. An agent declaring anything else
# silently gets no such tool at runtime.
VALID_TOOLS = frozenset(
    {
        "Bash",
        "Edit",
        "Glob",
        "Grep",
        "Read",
        "Write",
        "NotebookEdit",
        "WebFetch",
        "WebSearch",
        "Task",
        "TodoWrite",
        "SlashCommand",
        "Skill",
    }
)

# Repo paths a `.claude` file may cite without them existing yet. Each entry
# must carry a reason, and `test_forward_references_are_still_forward` asserts
# the exemption is still needed -- so a stale entry fails rather than rotting.
FORWARD_REFERENCES: dict[str, str] = {
    "src/pde/certificate/": (
        "certificate-validation is a KICKOFF skill: step 2 is literally "
        "'Scaffold the module additively: src/pde/certificate/'. The path is "
        "the skill's output, not its dependency."
    ),
}

_PATH_IN_BACKTICKS = re.compile(
    r"`((?:src|tests|scripts|docs|config|specs|openspec|dashboard|\.github|results)"
    r"/[A-Za-z0-9_./\-]+)`"
)


def _frontmatter(path: Path) -> dict[str, Any]:
    """Parse a file's YAML frontmatter, or fail with a legible message."""
    text = path.read_text()
    assert text.startswith("---\n"), f"{path.name}: no YAML frontmatter"
    _, raw, _ = text.split("---", 2)
    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict), f"{path.name}: frontmatter is not a mapping"
    return parsed


def _cited_paths(path: Path) -> set[str]:
    """Repo-relative paths cited in backticks, minus globs and `::member` suffixes."""
    out: set[str] = set()
    for raw in _PATH_IN_BACKTICKS.findall(path.read_text()):
        candidate = raw.split("::")[0].rstrip(".,;:")
        if any(ch in candidate for ch in "*?[]"):
            continue
        out.add(candidate)
    return out


class TestHarnessIsDiscoverable:
    """The suite is worthless if it silently parametrizes over nothing."""

    def test_every_artifact_kind_is_present(self) -> None:
        """Guards the failure mode where a glob typo makes 0 tests run and pass."""
        assert SKILLS, "no skills discovered -- the glob or the directory moved"
        assert AGENTS, "no agents discovered"
        assert COMMANDS, "no commands discovered"
        assert (CLAUDE_DIR / "settings.json").is_file()
        assert (CLAUDE_DIR / "hooks" / "session_start.sh").is_file()


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
class TestSkills:
    def test_frontmatter_has_required_keys(self, path: Path) -> None:
        fm = _frontmatter(path)
        assert set(fm) >= {"name", "description"}, f"{path}: missing name/description"
        assert fm["description"].strip(), f"{path}: empty description"

    def test_name_matches_its_directory(self, path: Path) -> None:
        """The directory name is the invocation name; a mismatch is unreachable."""
        assert _frontmatter(path)["name"] == path.parent.name


@pytest.mark.parametrize("path", AGENTS, ids=lambda p: p.stem)
class TestAgents:
    def test_frontmatter_has_required_keys(self, path: Path) -> None:
        fm = _frontmatter(path)
        assert set(fm) >= {"name", "description", "tools"}

    def test_name_matches_its_filename(self, path: Path) -> None:
        assert _frontmatter(path)["name"] == path.stem

    def test_declares_only_real_tools(self, path: Path) -> None:
        """A misspelled tool is not an error at load time -- the agent just lacks it."""
        declared = {t.strip() for t in _frontmatter(path)["tools"].split(",")}
        unknown = declared - VALID_TOOLS
        assert not unknown, f"{path.name} declares unknown tool(s): {sorted(unknown)}"


@pytest.mark.parametrize("path", COMMANDS, ids=lambda p: p.stem)
class TestCommands:
    def test_frontmatter_has_a_description(self, path: Path) -> None:
        fm = _frontmatter(path)
        assert fm.get("description", "").strip()

    def test_declaring_an_argument_hint_means_using_the_argument(self, path: Path) -> None:
        """`argument-hint` promises the command consumes `$ARGUMENTS`."""
        fm = _frontmatter(path)
        if "argument-hint" in fm:
            body = path.read_text().split("---", 2)[2]
            assert "$ARGUMENTS" in body, (
                f"{path.name} advertises argument-hint={fm['argument-hint']!r} "
                "but never references $ARGUMENTS"
            )


@pytest.mark.parametrize("path", ALL_MARKDOWN, ids=lambda p: str(p.relative_to(CLAUDE_DIR)))
def test_cited_repo_paths_exist(path: Path) -> None:
    """A skill citing a path that was deleted sends its reader somewhere empty."""
    missing = sorted(
        p
        for p in _cited_paths(path)
        if not (REPO_ROOT / p).exists() and p not in FORWARD_REFERENCES
    )
    assert not missing, f"{path.name} cites non-existent path(s): {missing}"


def test_forward_references_are_still_forward() -> None:
    """A declared exemption that now exists is stale and must be removed.

    Without this the allowlist only ever grows, and an entry outliving its
    reason silently weakens `test_cited_repo_paths_exist`.
    """
    stale = sorted(p for p in FORWARD_REFERENCES if (REPO_ROOT / p).exists())
    assert not stale, f"these paths now exist and must be dropped from FORWARD_REFERENCES: {stale}"


class TestSettings:
    def test_is_valid_json_with_a_schema(self) -> None:
        data = json.loads((CLAUDE_DIR / "settings.json").read_text())
        assert data.get("$schema", "").startswith("https://")

    def test_every_module_permission_resolves(self) -> None:
        """`Bash(python -m src.foo.bar:*)` must name an importable module.

        A renamed entry point leaves a permission that silently never matches,
        so the agent gets a permission prompt for a command the repo intended to
        pre-approve.
        """
        data = json.loads((CLAUDE_DIR / "settings.json").read_text())
        missing: list[str] = []
        for entry in data["permissions"]["allow"]:
            m = re.match(r"Bash\(python -m ((?:scripts|src)\.[A-Za-z0-9_.]+):", entry)
            if not m:
                continue
            base = REPO_ROOT / m.group(1).replace(".", "/")
            if not (base.with_suffix(".py").exists() or (base / "__main__.py").exists()):
                missing.append(m.group(1))
        assert not missing, f"permissions name non-existent module(s): {missing}"

    def test_coverage_core_is_pinned_to_pytrace(self) -> None:
        """Not cosmetic: the installed torch wheel crashes coverage's C tracer.

        The failure is silent UNDER-measurement, so a session without this env
        var produces coverage numbers that are simply wrong -- CLAUDE.md's
        Regression Surface calls this out on every gate.
        """
        data = json.loads((CLAUDE_DIR / "settings.json").read_text())
        assert data["env"]["COVERAGE_CORE"] == "pytrace"


class TestSessionStartHook:
    HOOK = CLAUDE_DIR / "hooks" / "session_start.sh"

    def test_is_syntactically_valid_shell(self) -> None:
        """`bash -n` parses without executing -- safe and deterministic."""
        proc = subprocess.run(
            ["bash", "-n", str(self.HOOK)], capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 0, f"shell syntax error: {proc.stderr}"

    def test_is_registered_in_settings(self) -> None:
        """A hook file nothing references is dead configuration."""
        data = json.loads((CLAUDE_DIR / "settings.json").read_text())
        commands = [h["command"] for entry in data["hooks"]["SessionStart"] for h in entry["hooks"]]
        assert any("session_start.sh" in c for c in commands)


class TestNamesAreUnique:
    """Two artifacts sharing a name means one is unreachable."""

    @pytest.mark.parametrize(
        "kind,paths,key",
        [
            ("skill", SKILLS, lambda p: p.parent.name),
            ("agent", AGENTS, lambda p: p.stem),
            ("command", COMMANDS, lambda p: p.stem),
        ],
        ids=["skills", "agents", "commands"],
    )
    def test_names_do_not_collide(self, kind: str, paths: list[Path], key: Any) -> None:
        names = [key(p) for p in paths]
        assert len(names) == len(set(names)), f"duplicate {kind} name in {names}"


def test_parsing_is_deterministic() -> None:
    """Same bytes -> same parse, twice, for every artifact.

    Cheap insurance that nothing here depends on dict ordering, hash seeding or
    filesystem iteration order. `PYTHONHASHSEED=0` is set in settings.json, so a
    regression that only shows up under randomized hashing would otherwise be
    invisible in exactly the environment this harness runs in.
    """
    for path in SKILLS + AGENTS + COMMANDS:
        first, second = _frontmatter(path), _frontmatter(path)
        assert first == second
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_every_python_snippet_in_the_harness_parses() -> None:
    """A skill whose example does not compile teaches a broken pattern."""
    fence = re.compile(r"```python\n(.*?)```", re.S)
    failures: list[str] = []
    for path in ALL_MARKDOWN:
        for i, snippet in enumerate(fence.findall(path.read_text())):
            if "..." in snippet or snippet.strip().startswith("#"):
                continue  # elided example, not meant to compile
            try:
                ast.parse(snippet)
            except SyntaxError as exc:
                failures.append(f"{path.relative_to(CLAUDE_DIR)} block {i}: {exc}")
    assert not failures, "non-parsing python snippets:\n" + "\n".join(failures)
