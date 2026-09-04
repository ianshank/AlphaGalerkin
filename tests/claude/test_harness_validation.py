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
import os
import re
import subprocess
from collections.abc import Callable
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
    # ``evidence`` added 2026-08-23: the harness now cites spike write-ups under
    # ``evidence/spikes/``, and a root absent from this alternation is not "allowed",
    # it is *unchecked* -- a typo there would have gone unnoticed.
    r"`((?:src|tests|scripts|docs|config|specs|openspec|dashboard|evidence|\.github|results)"
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

    def test_no_coverage_core_tracer_pin(self) -> None:
        """The inverse of what this test asserted until 2026-09-02.

        History, kept rather than deleted -- the pin was real and its removal
        was evidence-driven, not a cleanup. Two commits justified it:

        * ``cb645ad`` (2026-07-10): the installed torch wheel crashed coverage's
          default C tracer at collection.
        * ``cfe7f22`` (2026-08-15): the C tracer silently UNDER-measured --
          ``src/training`` cited at 89.53% under ``pytrace`` vs 82.45% without,
          with ``base_trainer.py`` reported at 46%.

        Both were re-verified in 2026-09 and neither reproduces. The identical
        ``src/training`` gate measures 88.25% under *both* tracers with a
        byte-identical per-file breakdown, and a minimal direct reproduction of
        the documented crash (``coverage run --branch`` over a script importing
        ``torch._C`` and running a tensor op) exits 0 under both cores. CI's own
        logs then confirmed it at scale: removing the pin from the ``coverage``
        job cut pytest execution from 1967.30s to 604.32s (3.26x) with
        byte-identical totals (``TOTAL 29971 2232 7150 91%``, 90.70%), zero
        crash signatures, and all 43 per-module gates green.

        So this now asserts *absence*: the pin costs ~3x wall-clock on every
        coverage job and buys nothing. A reintroduced ``COVERAGE_CORE`` here
        would silently slow CI back down and re-assert a false invariant, so it
        fails until someone re-establishes the crash with fresh evidence.
        """
        data = json.loads((CLAUDE_DIR / "settings.json").read_text())
        assert "COVERAGE_CORE" not in data.get("env", {})


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


class TestEnforcementIsWired:
    """AQA: the suite above is worthless if nothing runs it.

    Every test in this file passes happily on a developer's machine while the CI
    step that runs them has been deleted. That is the failure mode this repo
    has hit repeatedly and at scale — `tests/demos/` and `tests/notebooks/` (226
    tests) were green locally and executed by NO CI step for months, and
    `.gitleaks.toml` shipped alongside a `make gitleaks` target while nothing
    ever invoked either.

    So the acceptance criterion is not "the harness is valid" but "the harness
    is valid AND something enforces it." These assertions read the workflow and
    the Makefile as data.
    """

    CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    MAKEFILE = REPO_ROOT / "Makefile"

    def test_ci_runs_the_harness_suite(self) -> None:
        assert "pytest tests/claude/" in self.CI.read_text(), (
            "no CI step runs tests/claude/ -- these 71 tests would be green and "
            "unexecuted, exactly like tests/demos/ was"
        )

    def test_ci_runs_gitleaks(self) -> None:
        """The scanner, not merely its config, must be invoked."""
        ci = self.CI.read_text()
        assert "gitleaks/gitleaks-action" in ci, "gitleaks config exists but CI never runs it"
        assert "GITLEAKS_CONFIG: .gitleaks.toml" in ci, "CI runs gitleaks with the wrong config"

    def test_gitleaks_config_exists_for_that_step_to_use(self) -> None:
        assert (REPO_ROOT / ".gitleaks.toml").is_file()

    @pytest.mark.parametrize(
        "target",
        # `test-e2e` was added 2026-09-04: the branch that created that target
        # chained it into `pre-pr` and asserted the chaining nowhere, so deleting
        # the line from the Makefile left every guard in the repo green -- the
        # "pre-pr is narrower than CI" defect the target exists to prevent.
        ["test-claude", "test-demos", "gitleaks", "test-e2e"],
    )
    def test_make_pre_pr_chains_the_local_equivalents(self, target: str) -> None:
        """`make pre-pr` narrower than CI is how CI-invisible tests happen."""
        text = self.MAKEFILE.read_text()
        pre_pr = next((ln for ln in text.splitlines() if ln.startswith("pre-pr:")), "")
        assert pre_pr, "no pre-pr target found in Makefile"
        assert target in pre_pr, f"make pre-pr does not chain {target}"
        assert f"\n{target}:" in text, f"{target} is chained but not defined"

    def test_gitleaks_target_does_not_pass_silently_when_uninstalled(self) -> None:
        """It is chained into pre-pr, so it must not hard-fail without the binary.

        But a skipped scan reported as success is the "check described but not
        executed" failure mode. The target must say so out loud.
        """
        body = self.MAKEFILE.read_text().split("\ngitleaks:", 1)[1].split("\n\n", 1)[0]
        assert "command -v gitleaks" in body, "target does not detect a missing binary"
        assert "SKIPPED" in body, "a skipped scan must announce itself, not pass quietly"


class TestTheDocumentedInventoryMatchesDisk:
    """CLAUDE.md's harness counts must equal what is on disk.

    This row has drifted **twice** and records both drifts in its own prose: a
    "9 subagents" transcription slip corrected on 2026-09-02, and a "103 tests"
    figure that was never a measured count. The row explains why -- the suite is
    data-driven, so a new skill is validated on sight and the hand-maintained
    number in the row is free to rot unnoticed.

    Correcting it a third time by hand would repeat the mistake. These assertions
    make the claim self-maintaining: adding a skill, an agent or a command fails
    here until the row is updated, which is the only mechanism that has ever kept
    a number in this file honest.

    The test count is deliberately **not** asserted exactly -- it changes with
    every test added anywhere under ``tests/claude/``, which would make this a
    tax rather than a guard. A floor is asserted instead, so the row can never
    claim more coverage than exists.
    """

    #: The Regression Surface row that states the inventory.
    ROW_PREFIX = "| Agentic harness (`.claude/`) |"

    @staticmethod
    def _row() -> str:
        text = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        rows = [
            line
            for line in text.splitlines()
            if line.startswith(TestTheDocumentedInventoryMatchesDisk.ROW_PREFIX)
        ]
        assert len(rows) == 1, (
            f"expected exactly one Agentic-harness Regression Surface row, found "
            f"{len(rows)}; a renamed row would make every assertion below vacuous"
        )
        return rows[0]

    def test_the_row_exists(self) -> None:
        """Vacuity guard for the assertions below."""
        assert self._row()

    @pytest.mark.parametrize(
        ("noun", "counted"),
        [
            (
                "skills",
                lambda: len([p for p in (REPO_ROOT / ".claude/skills").iterdir() if p.is_dir()]),
            ),
            ("subagents", lambda: len(list((REPO_ROOT / ".claude/agents").glob("*.md")))),
            ("slash commands", lambda: len(list((REPO_ROOT / ".claude/commands").glob("*.md")))),
        ],
    )
    def test_the_stated_count_matches_the_directory(
        self, noun: str, counted: Callable[[], int]
    ) -> None:
        """Each ``N <noun>`` claim in the row equals the files on disk."""
        actual = counted()
        assert actual > 0, f"no {noun} found on disk; the count would be vacuous"
        expected = f"**{actual} skills, " if noun == "skills" else f"{actual} {noun}"
        row = self._row()
        assert expected in row or f"{actual} {noun}" in row, (
            f"CLAUDE.md's Agentic-harness row does not state {actual} {noun}; "
            f"the directory holds {actual}. This number has drifted twice before -- "
            f"update the row rather than the assertion."
        )


#: Wall-clock ceiling for running one hook in a test. Every invocation here
#: is a dry-run, so this is a hang detector rather than a work budget; named
#: so widening it is a visible edit rather than a silent one.
HOOK_RUN_TIMEOUT_S: int = 30


class TestPostToolUseHooks:
    """The two path-gated hooks parse, are registered, and actually gate.

    "Deterministic validation of the harness" means more than "the file exists".
    A hook that fires on every edit is a hook someone disables, and a disabled
    hook checks nothing -- so the *gating* is the property worth asserting, and
    it is asserted by **running** each hook against synthetic payloads rather
    than by reading its source.

    Every case below is hermetic: the non-matching payloads exercise the early
    exit, which runs no pytest and touches no network.
    """

    #: Hook scripts registered under ``PostToolUse``, and the tool payload key
    #: they read. Discovered from settings.json rather than listed, so a third
    #: hook is validated on sight.
    HOOKS_DIR = CLAUDE_DIR / "hooks"

    #: Paths each hook must *decline* -- ordinary source edits, which are the
    #: overwhelming majority and must cost nothing.
    IGNORED_PATHS: tuple[str, ...] = (
        "src/pde/trainer.py",
        "tests/e2e/test_untested_entry_points.py",
        "README.rst",
        "config/scenarios/poc_quick.yaml",
    )

    @staticmethod
    def _post_tool_use_commands() -> list[str]:
        settings = json.loads((CLAUDE_DIR / "settings.json").read_text(encoding="utf-8"))
        entries = settings.get("hooks", {}).get("PostToolUse", [])
        return [
            hook["command"]
            for entry in entries
            for hook in entry.get("hooks", [])
            if isinstance(hook, dict) and "command" in hook
        ]

    @staticmethod
    def _run_hook(script: Path, file_path: str) -> subprocess.CompletedProcess[str]:
        """Feed a synthetic PostToolUse payload to *script*, in dry-run.

        Dry-run is what makes this hermetic and fast: the property under test is
        the *gating decision* (does this path fire the hook?), not the command it
        then runs. Running the real command would cost ~13 s per matching
        parametrisation and make this suite something a developer skips -- and a
        skipped harness test is the failure mode the suite exists to prevent.
        """
        payload = json.dumps({"tool_input": {"file_path": file_path}})
        return subprocess.run(
            ["bash", str(script)],
            input=payload,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "ALPHAGALERKIN_HOOK_DRY_RUN": "1"},
            timeout=HOOK_RUN_TIMEOUT_S,
            check=False,
        )

    def test_post_tool_use_is_registered(self) -> None:
        """Vacuity guard: without a registration every assertion below is moot."""
        commands = self._post_tool_use_commands()
        assert commands, "settings.json declares no PostToolUse hooks"

    @pytest.mark.parametrize("script_name", ["guard_build_config.sh", "check_doc_links.sh"])
    def test_the_hook_exists_parses_and_is_registered(self, script_name: str) -> None:
        """A hook that is not registered runs never; one that is not executable errors."""
        script = self.HOOKS_DIR / script_name
        assert script.is_file(), f"{script_name} is registered or expected but absent"
        parsed = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert parsed.returncode == 0, parsed.stderr
        assert any(script_name in command for command in self._post_tool_use_commands()), (
            f"{script_name} exists but no PostToolUse entry invokes it, so it never runs"
        )

    @pytest.mark.parametrize("script_name", ["guard_build_config.sh", "check_doc_links.sh"])
    @pytest.mark.parametrize("ignored", IGNORED_PATHS)
    def test_the_hook_declines_paths_outside_its_scope(
        self, script_name: str, ignored: str
    ) -> None:
        """An ordinary source edit must cost nothing and print nothing.

        This is the assertion that keeps the hook cheap enough to survive. If it
        ever starts firing on `src/**.py`, it fires hundreds of times a session
        and gets removed -- taking its protection with it.
        """
        result = self._run_hook(self.HOOKS_DIR / script_name, ignored)
        assert result.returncode == 0
        assert not result.stdout.strip(), (
            f"{script_name} produced output for {ignored!r}, which is outside its "
            f"declared scope; a hook that fires on every edit will be disabled"
        )

    @pytest.mark.parametrize(
        ("script_name", "matching"),
        [
            ("guard_build_config.sh", ".github/workflows/ci.yml"),
            ("guard_build_config.sh", "Makefile"),
            ("guard_build_config.sh", "pyproject.toml"),
            ("guard_build_config.sh", "conftest.py"),
            ("check_doc_links.sh", "CLAUDE.md"),
        ],
    )
    def test_the_hook_acts_on_paths_inside_its_scope(self, script_name: str, matching: str) -> None:
        """The conditional half: a gate that declines *everything* is not a gate.

        Without this, deleting the whole ``case`` body -- or narrowing it to a
        path that never occurs -- would pass every declining test above.
        """
        result = self._run_hook(self.HOOKS_DIR / script_name, matching)
        assert result.stdout.strip(), (
            f"{script_name} produced no output for {matching!r}, which it is "
            f"supposed to guard; the path gate matches nothing"
        )

    @pytest.mark.parametrize("script_name", ["guard_build_config.sh", "check_doc_links.sh"])
    def test_the_hook_never_blocks(self, script_name: str) -> None:
        """Hooks here report; CI gates. A blocking hook gets switched off."""
        result = self._run_hook(self.HOOKS_DIR / script_name, ".github/workflows/ci.yml")
        assert result.returncode == 0, (
            f"{script_name} exited {result.returncode}; these hooks must always "
            f"exit 0 -- the build gate is CI, not the editor"
        )

    @pytest.mark.parametrize("script_name", ["guard_build_config.sh", "check_doc_links.sh"])
    def test_the_hook_survives_a_malformed_payload(self, script_name: str) -> None:
        """A hook that crashes on unexpected stdin is noise on every edit."""
        result = subprocess.run(
            ["bash", str(self.HOOKS_DIR / script_name)],
            input="not json at all",
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "ALPHAGALERKIN_HOOK_DRY_RUN": "1"},
            timeout=HOOK_RUN_TIMEOUT_S,
            check=False,
        )
        assert result.returncode == 0
