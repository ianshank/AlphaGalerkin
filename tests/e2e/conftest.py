"""E2E test fixtures and configuration.

Provides shared fixtures for end-to-end testing of CLI commands,
user journeys, and browser-based dashboard interaction (Playwright).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

import pytest

# Type alias for the CLI runner fixture
CLIRunnerType = Callable[
    [str, list[str] | None, int, dict[str, str] | None],
    "CLIResult",
]


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


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs.

    Yields:
        Path to temporary directory (cleaned up after test).

    """
    with tempfile.TemporaryDirectory(prefix="alphagalerkin_e2e_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cli_runner() -> CLIRunnerType:
    """Create a CLI command runner.

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

        import os

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
                cwd=Path(__file__).parents[2],  # Project root
            )
            return CLIResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                command=cmd,
            )
        except subprocess.TimeoutExpired:
            return CLIResult(
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                command=cmd,
            )

    return run_command


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


# ---------------------------------------------------------------------------
# Playwright / Dashboard browser fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dashboard_server():
    """Launch the Gradio dashboard on a random port for E2E testing.

    Yields:
        The local URL (e.g., "http://127.0.0.1:XXXXX") where the app is running.

    """
    # Add project paths for imports
    root = Path(__file__).parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    hf_space = root / "hf_space"
    if str(hf_space) not in sys.path:
        sys.path.insert(0, str(hf_space))

    import matplotlib

    matplotlib.use("Agg")

    from dashboard.app import build_app

    app = build_app()

    # Launch on a specific port to avoid port=0 issues
    import socket

    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        show_error=True,
        quiet=True,
    )

    local_url = f"http://127.0.0.1:{port}"

    # Wait for server to be ready by polling
    import httpx

    max_wait = 15
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = httpx.get(local_url, timeout=2)
            if resp.status_code == 200:
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            time.sleep(0.5)
    else:
        raise RuntimeError(f"Dashboard server not ready after {max_wait}s at {local_url}")

    yield local_url

    app.close()


@pytest.fixture(scope="session")
def browser_context_args():
    """Configure Playwright browser context for dashboard tests."""
    return {
        "viewport": {"width": 1280, "height": 800},
        "ignore_https_errors": True,
    }


@pytest.fixture
def dashboard_page(dashboard_server, page):
    """Navigate to the dashboard and wait for it to load.

    Provides a Playwright Page pre-navigated to the running dashboard.

    Args:
        dashboard_server: The running dashboard URL fixture.
        page: Playwright page fixture from pytest-playwright.

    Returns:
        Playwright Page object at the dashboard URL.

    """
    page.goto(dashboard_server)
    # Wait for DOM content (don't use networkidle — Gradio keeps WebSocket open)
    page.wait_for_load_state("domcontentloaded")
    # Wait for Gradio's JS framework to render the app
    page.wait_for_selector("h1, .gradio-container", timeout=15000)
    # Small extra buffer for dynamic components
    page.wait_for_timeout(1000)
    return page
