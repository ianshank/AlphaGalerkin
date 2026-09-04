"""E2E test fixtures and configuration.

Provides shared fixtures for end-to-end testing of CLI commands and user
journeys, plus the tier's **device contract** (see ``E2E_DEVICE`` below).

Two environment variables configure this tier, and nothing else does:

- ``E2E_TIMEOUT_SCALE`` -- multiplies the three subprocess budgets.
- ``E2E_DEVICE`` -- selects the device every journey runs on.

Both default to values that reproduce the historical behaviour exactly.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from src.poc.device import resolve_device

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.terminal import TerminalReporter

# Subprocess budgets for the CLI journeys, scaled by one env var.
#
# These were six bare literals across two modules (30 / 60 / 120). The 120 s one
# was long blamed for `test_quick_validation_journey.py::test_train_physics_minimal`
# failing on a loaded machine -- including in the `ci.yml` comment that kept this
# whole directory out of CI. That attribution was wrong and is recorded here so
# it is not repeated: the run was never minimal (`--train-size` is the grid side,
# not the sample count, so it built the default 5000 samples and needed ~1.7 h),
# and `returncode in [0, 1]` cannot hold for a run that does not finish under
# *any* budget. Passing `--n-train-samples/--n-eval-samples` fixed it; the test
# now takes ~15 s. The scale factor remains useful for genuinely slow or
# contended runners, but it is not a workaround for an unbounded run.
#
# The three tiers are ordered by what the subprocess actually does, and that
# ordering is the part worth preserving if these are ever retuned:
#   TRIVIAL  -- argument parsing / --help; process startup dominates.
#   BENCH    -- a bounded measurement loop; work is real but capped by argv.
#   TRAINING -- a real training run; the only one whose cost tracks host speed.
E2E_TIMEOUT_SCALE: float = float(os.environ.get("E2E_TIMEOUT_SCALE", "1.0"))

E2E_TRIVIAL_TIMEOUT_S: int = int(30 * E2E_TIMEOUT_SCALE)
E2E_BENCHMARK_TIMEOUT_S: int = int(60 * E2E_TIMEOUT_SCALE)
E2E_TRAINING_TIMEOUT_S: int = int(120 * E2E_TIMEOUT_SCALE)

# --------------------------------------------------------------------------- #
# Device contract                                                              #
# --------------------------------------------------------------------------- #
#
# The tier runs unchanged on a CPU-only runner and on a CUDA host. That is three
# properties, not one, and before this contract the repo satisfied none of them
# for an E2E test:
#
#   1. The test picks the device from the environment, never from a literal.
#      Every GPU smoke in the repo hardcodes `device="cuda"` and leans on the
#      `gpu_required` auto-skip -- and since every workflow is `ubuntu-latest`,
#      all of them have skipped on every CI run ever.
#   2. The artifact can prove where the run executed. `ScenarioResult.device`
#      used to record host CUDA *availability*, so a `device: cpu` config on a
#      CUDA host persisted "cuda" -- the inverse of the field's meaning. Fixed in
#      `src/poc/registry.py::BaseScenario.execution_device_label`.
#   3. No silent fallback between "asked for" and "ran on". The repo has five
#      device-resolution policies with different fallback semantics; forwarding a
#      *concrete* device string to every child means none of them is ever handed
#      an ambiguous input.
#
# `E2E_DEVICE` takes exactly `resolve_device`'s four forms -- "cuda", "cuda:N",
# "cpu", "auto" -- and no fifth vocabulary. It is resolved ONCE, here at conftest
# import, which is what gives the required semantics for free:
#
#   auto (default) -> CPU CI runs unchanged; a CUDA host resolves to "cuda".
#   cuda           -> `resolve_device` RAISES when CUDA is absent, so this is a
#                     COLLECTION ERROR for the whole directory, not a skip. "cuda"
#                     *is* the require form (see `src/poc/device.py`), so no
#                     separate E2E_REQUIRE_GPU var is needed. Same outcome class
#                     as ALPHAGALERKIN_REQUIRE_EXTRAS=1 for the [fem] extra.
#   cpu            -> forces CPU on a CUDA host, for bisecting.
#
# Tests forward `E2E_RESOLVED_DEVICE` (concrete) to children, never `E2E_DEVICE`
# (which may be the ambiguous "auto").
E2E_DEVICE: str = os.environ.get("E2E_DEVICE", "auto")

E2E_RESOLVED_DEVICE: str = str(resolve_device(E2E_DEVICE, context="tests/e2e"))

#: Device *type* without any index, for comparing against ``torch.device.type``
#: (``"cuda:0"`` -> ``"cuda"``). Journeys asserting where a tensor landed use
#: this; journeys asserting what was requested use ``E2E_RESOLVED_DEVICE``.
E2E_DEVICE_TYPE: str = E2E_RESOLVED_DEVICE.split(":")[0]

#: Env mapping that makes a child process see no CUDA device, whatever the host
#: has. This is what lets the fail-loud negative tests ("--device cuda must raise")
#: run identically on a CPU-only runner and on a GPU box -- on the latter it also
#: proves the child actually honoured the flag rather than quietly using the GPU.
NO_CUDA_ENV: dict[str, str] = {"CUDA_VISIBLE_DEVICES": ""}

# Type alias for the CLI runner fixture
CLIRunnerType = Callable[
    [str, list[str] | None, int, dict[str, str] | None],
    "CLIResult",
]

#: Type alias for the inline-Python runner fixture (see ``py_runner``).
PyRunnerType = Callable[
    [str, int, dict[str, str] | None],
    "CLIResult",
]

#: Type alias for the scenario-YAML pinning helper (see ``pin_scenario_yaml``).
ScenarioYamlPinnerType = Callable[..., Path]


#: Repository root, resolved once. Every child process is launched with this as
#: its cwd so relative paths in shipped configs resolve the same way they do for
#: a developer running the command by hand.
PROJECT_ROOT: Path = Path(__file__).parents[2]


@dataclass
class CLIResult:
    """Result from running a CLI command."""

    returncode: int
    stdout: str
    stderr: str
    command: list[str]

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.returncode == 0

    @property
    def output(self) -> str:
        """Combined stdout + stderr.

        Journeys that search for a message need both streams: ``structlog``
        writes to stdout while ``argparse`` and tracebacks go to stderr, and
        which one carries a given line is not a stable contract.
        """
        return self.stdout + self.stderr


def _run_subprocess(cmd: list[str], timeout: int, env: dict[str, str] | None) -> CLIResult:
    """Run *cmd* to completion and capture its output.

    Shared by :func:`cli_runner` and :func:`py_runner` so the two cannot drift in
    how they set cwd, merge the environment, or report a timeout.

    The parent environment is inherited (then updated with *env*), which is what
    lets a caller pass ``NO_CUDA_ENV`` to hide the GPU from a child. It also means
    anything set in the parent -- ``LM_STUDIO_URL``, ``PYTHONHASHSEED`` -- reaches
    the child; that is deliberate, and tests that must not see a variable should
    pass it explicitly as empty rather than assume a clean environment.

    Args:
        cmd: Full argv of the child process.
        timeout: Seconds before the child is killed; must come from one of the
            three tier constants, never a bare literal.
        env: Extra environment variables layered over the parent's.

    Returns:
        CLIResult; a timeout is reported as ``returncode == -1`` rather than
        raising, so a test can assert on it.

    """
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
            cwd=PROJECT_ROOT,
        )
    except subprocess.TimeoutExpired:
        return CLIResult(
            returncode=-1,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            command=cmd,
        )
    return CLIResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=cmd,
    )


@pytest.fixture(scope="session")
def e2e_device() -> str:
    """The concrete device every journey in this tier runs on.

    Resolved once at conftest import from ``E2E_DEVICE`` (default ``auto``); see
    the module docstring for the semantics. Always a concrete string --
    ``"cpu"``, ``"cuda"``, ``"cuda:0"`` -- never ``"auto"``, so forwarding it to
    a child can never reach one of the repo's silent-fallback branches.

    Returns:
        The resolved device string.

    """
    return E2E_RESOLVED_DEVICE


@pytest.fixture(scope="session")
def e2e_device_type() -> str:
    """Device *type* of :data:`E2E_RESOLVED_DEVICE` (``"cuda:0"`` -> ``"cuda"``).

    For comparing against ``torch.device.type`` / ``tensor.device.type``, which
    never carry the index in the form a config string does.

    Returns:
        ``"cpu"`` or ``"cuda"``.

    """
    return E2E_DEVICE_TYPE


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs.

    Yields:
        Path to temporary directory (cleaned up after test).

    """
    with tempfile.TemporaryDirectory(prefix="alphagalerkin_e2e_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def cli_runner() -> CLIRunnerType:
    """Create a CLI command runner.

    Session-scoped: a stateless factory closing over nothing mutable, so one
    instance is safe to share -- and sharing it is what lets a test module build
    a module-scoped fixture that runs an expensive harness once rather than once
    per test.

    Returns:
        Function to run CLI commands and capture output.

    """

    def run_command(
        module: str,
        args: list[str] | None = None,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> CLIResult:
        """Run a Python module command.

        Args:
            module: Module to run (e.g., "src.poc.cli").
            args: Command-line arguments.
            timeout: Timeout in seconds.
            env: Additional environment variables.

        Returns:
            CLIResult with command output.

        """
        cmd = [sys.executable, "-m", module]
        if args:
            cmd.extend(args)
        return _run_subprocess(cmd, timeout, env)

    return run_command


@pytest.fixture(scope="session")
def py_runner() -> PyRunnerType:
    """Run a snippet of Python in a fresh interpreter.

    ``cli_runner`` can only build ``python -m <module>``, so journeys that must
    exercise a *library* entry point across a process boundary -- the substrate
    registry, which is a process-global singleton that two other suites
    ``clear()`` -- have no way to use it. This is the sibling for those: same
    cwd, environment merging and timeout handling, different argv shape.

    Returns:
        Function taking ``(code, timeout, env)`` and returning a CLIResult.

    """

    def run_code(
        code: str,
        timeout: int = E2E_BENCHMARK_TIMEOUT_S,
        env: dict[str, str] | None = None,
    ) -> CLIResult:
        """Run *code* via ``python -c`` from the project root.

        Args:
            code: Python source to execute.
            timeout: Timeout in seconds (use a tier constant).
            env: Additional environment variables.

        Returns:
            CLIResult with the child's output.

        """
        return _run_subprocess([sys.executable, "-c", code], timeout, env)

    return run_code


class ScenarioYamlKeyError(KeyError):
    """A scenario YAML lacked a key the pinning helper was asked to overwrite.

    Raised rather than silently adding the key: ``pin_scenario_yaml`` exists to
    *override* a value the shipped config already declares. Writing a key the
    config does not have would produce a config that validates (Pydantic fills
    defaults) while pinning nothing -- a silent no-op, which is precisely the
    failure mode the helper is meant to make impossible.
    """


def _scenario_mapping(document: dict[str, Any], path: Path) -> dict[str, Any]:
    """Return the single scenario mapping inside a loaded scenario document.

    Shipped configs come in two shapes -- a ``scenarios:`` list, or a bare
    mapping -- mirroring what ``src/poc/config.py`` accepts.

    Args:
        document: Parsed YAML document.
        path: Source path, for error messages.

    Returns:
        The scenario mapping (a live reference into *document*).

    Raises:
        ScenarioYamlKeyError: The document has neither shape, or the
            ``scenarios`` list does not hold exactly one entry (the helper
            refuses to guess which one the caller meant).

    """
    scenarios = document.get("scenarios")
    if scenarios is None:
        if "name" in document:
            return document
        raise ScenarioYamlKeyError(
            f"{path}: expected a 'scenarios' list or a bare scenario mapping with a 'name'"
        )
    if not isinstance(scenarios, list) or len(scenarios) != 1:
        raise ScenarioYamlKeyError(
            f"{path}: expected exactly one entry under 'scenarios', "
            f"found {len(scenarios) if isinstance(scenarios, list) else type(scenarios).__name__}"
        )
    mapping = scenarios[0]
    if not isinstance(mapping, dict):
        raise ScenarioYamlKeyError(f"{path}: scenarios[0] is not a mapping")
    return mapping


@pytest.fixture(scope="session")
def pin_scenario_yaml(tmp_path_factory: pytest.TempPathFactory) -> ScenarioYamlPinnerType:
    """Copy a shipped scenario YAML, overriding declared keys.

    ``python -m src.poc.cli run`` has no ``--device`` (and no per-scenario
    override flags at all), so the only way to steer it is the config file.
    Adding such a flag was considered and rejected: not every scenario config
    carries a ``device`` field, and a flag that applies to some scenarios is a
    new silent path. Copying is also what lets a journey shrink budgets without
    editing a committed file.

    Every override must name a key the shipped config **already declares**, so a
    renamed or removed field fails here instead of pinning nothing.

    Session-scoped (via ``tmp_path_factory`` rather than ``tmp_path``) so a
    module-scoped fixture can pin a config once for an expensive run; a
    session-scoped fixture is still requestable from function scope, so this is
    strictly more general. Each call gets its own directory, so two pins of the
    same shipped file cannot overwrite one another.

    Returns:
        ``pin(src, /, **overrides) -> Path``.

    """

    def pin(src: str | Path, /, **overrides: Any) -> Path:
        """Write a copy of *src* into a fresh temp directory with *overrides* applied.

        Args:
            src: Path to the shipped YAML (absolute, or relative to the repo root).
            **overrides: Scenario keys to overwrite; each must already exist.

        Returns:
            Path to the written copy.

        Raises:
            ScenarioYamlKeyError: An override names a key the config lacks.

        """
        source = Path(src)
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        document = copy.deepcopy(document)
        mapping = _scenario_mapping(document, source)

        missing = sorted(key for key in overrides if key not in mapping)
        if missing:
            raise ScenarioYamlKeyError(
                f"{source}: cannot pin {missing} -- not declared by this scenario. "
                f"Declared keys: {sorted(mapping)}"
            )
        mapping.update(overrides)

        # A fresh directory per call: two pins of the same shipped file (a
        # module-scoped one and a function-scoped one, say) must not collide on
        # a shared session path.
        destination = tmp_path_factory.mktemp("pinned") / source.name
        destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return destination

    return pin


def pytest_terminal_summary(
    terminalreporter: TerminalReporter, exitstatus: int, config: Config
) -> None:
    """Report which device this tier ran on -- visibly, not silently.

    A GPU host whose driver disappeared resolves ``auto`` to ``cpu`` and every
    test still passes; without this line that degradation is invisible.
    """
    if E2E_DEVICE == "auto" and E2E_DEVICE_TYPE == "cpu":
        detail = " (E2E_DEVICE=auto, CUDA unavailable)"
    elif E2E_DEVICE != E2E_RESOLVED_DEVICE:
        detail = f" (E2E_DEVICE={E2E_DEVICE})"
    else:
        detail = ""
    terminalreporter.write_line(f"e2e device: {E2E_RESOLVED_DEVICE}{detail}", yellow=True)


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root.

    """
    return Path(__file__).parents[2]


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Get the config directory.

    Returns:
        Path to config directory.

    """
    return project_root / "config"
