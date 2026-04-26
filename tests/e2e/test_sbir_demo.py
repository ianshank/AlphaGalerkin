"""End-to-end tests for the SBIR benchmark demo script.

Verifies that scripts/run_sbir_demo.py --dry-run completes successfully,
writes a valid JSON report, and includes all required SBIR metric keys.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_sbir_demo.py"


def _run_sbir_demo(
    extra_args: list[str],
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run run_sbir_demo as a subprocess and return the result."""
    cmd = [sys.executable, str(SCRIPT_PATH), *extra_args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestSBIRDemoDryRun:
    """End-to-end tests for run_sbir_demo --dry-run mode."""

    def test_dry_run_exits_zero(self, tmp_path: Path) -> None:
        """Verify the script exits with code 0 in dry-run mode."""
        result = _run_sbir_demo(
            ["--dry-run", "--output-dir", str(tmp_path), "--formats", "json"],
        )
        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_dry_run_creates_json_file(self, tmp_path: Path) -> None:
        """Verify results.json is created in the output directory."""
        result = _run_sbir_demo(
            ["--dry-run", "--output-dir", str(tmp_path), "--formats", "json"],
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        json_path = tmp_path / "results.json"
        assert json_path.exists(), f"results.json not found in {tmp_path}"

    def test_dry_run_json_is_valid(self, tmp_path: Path) -> None:
        """Verify results.json contains parseable JSON."""
        result = _run_sbir_demo(
            ["--dry-run", "--output-dir", str(tmp_path), "--formats", "json"],
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        json_path = tmp_path / "results.json"
        content = json_path.read_text(encoding="utf-8")
        data = json.loads(content)  # raises on invalid JSON
        assert isinstance(data, dict), "JSON root should be a dict"

    def test_dry_run_json_contains_required_keys(self, tmp_path: Path) -> None:
        """Verify the JSON report contains all required SBIR metric keys."""
        result = _run_sbir_demo(
            ["--dry-run", "--output-dir", str(tmp_path), "--formats", "json"],
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        json_path = tmp_path / "results.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))

        # Standard report structure keys
        assert "suite_name" in data, "Missing 'suite_name'"
        assert "config_path" in data, "Missing 'config_path'"
        assert "n_results" in data, "Missing 'n_results'"
        assert "results" in data, "Missing 'results'"

        # SBIR-specific summary metric keys
        assert "transfer_mse" in data, "Missing 'transfer_mse'"
        assert "complexity_timing" in data, "Missing 'complexity_timing'"
        assert "lbb_sigma_min" in data, "Missing 'lbb_sigma_min'"

    def test_dry_run_json_results_list_non_empty(self, tmp_path: Path) -> None:
        """Verify the results list is non-empty and has expected structure."""
        result = _run_sbir_demo(
            ["--dry-run", "--output-dir", str(tmp_path), "--formats", "json"],
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        json_path = tmp_path / "results.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))

        results = data["results"]
        assert isinstance(results, list), "'results' should be a list"
        assert len(results) > 0, "'results' list should not be empty"

        # Each result entry should have benchmark metadata
        for entry in results:
            assert "benchmark_name" in entry, f"Entry missing 'benchmark_name': {entry}"
            assert "method_name" in entry, f"Entry missing 'method_name': {entry}"
            assert "l2_error" in entry, f"Entry missing 'l2_error': {entry}"

    def test_dry_run_transfer_mse_value(self, tmp_path: Path) -> None:
        """Verify transfer_mse is a positive float below 0.05 threshold."""
        result = _run_sbir_demo(
            ["--dry-run", "--output-dir", str(tmp_path), "--formats", "json"],
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        json_path = tmp_path / "results.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))

        mse = data["transfer_mse"]
        assert isinstance(mse, float | int), f"transfer_mse should be numeric, got {type(mse)}"
        assert mse > 0, f"transfer_mse should be positive, got {mse}"
        # Physics PoC milestone: MSE < 0.05 success criterion
        assert mse < 0.05, f"transfer_mse {mse} exceeds 0.05 threshold"

    def test_dry_run_lbb_sigma_min_value(self, tmp_path: Path) -> None:
        """Verify lbb_sigma_min is a positive float (LBB stability condition)."""
        result = _run_sbir_demo(
            ["--dry-run", "--output-dir", str(tmp_path), "--formats", "json"],
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        json_path = tmp_path / "results.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))

        sigma_min = data["lbb_sigma_min"]
        assert isinstance(
            sigma_min, float | int
        ), f"lbb_sigma_min should be numeric, got {type(sigma_min)}"
        assert sigma_min > 0, f"lbb_sigma_min should be positive (LBB stable), got {sigma_min}"

    def test_dry_run_complexity_timing_structure(self, tmp_path: Path) -> None:
        """Verify complexity_timing dict has expected keys."""
        result = _run_sbir_demo(
            ["--dry-run", "--output-dir", str(tmp_path), "--formats", "json"],
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        json_path = tmp_path / "results.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))

        timing = data["complexity_timing"]
        assert isinstance(timing, dict), "'complexity_timing' should be a dict"
        assert "speedup_factor" in timing, "Missing 'speedup_factor' in complexity_timing"
        assert timing["speedup_factor"] > 1.0, "FNet should be faster than softmax (speedup > 1)"

    def test_dry_run_completes_quickly(self, tmp_path: Path) -> None:
        """Verify dry-run completes in under 5 seconds."""
        import time

        t0 = time.perf_counter()
        result = _run_sbir_demo(
            ["--dry-run", "--output-dir", str(tmp_path), "--formats", "json"],
        )
        elapsed = time.perf_counter() - t0

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert elapsed < 5.0, f"Dry-run took {elapsed:.2f}s, expected < 5s"

    def test_dry_run_output_mentions_dry_run(self, tmp_path: Path) -> None:
        """Verify console output indicates this is a dry run."""
        result = _run_sbir_demo(
            ["--dry-run", "--output-dir", str(tmp_path), "--formats", "json"],
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        combined = result.stdout + result.stderr
        assert (
            "dry" in combined.lower() or "DRY" in combined
        ), "Expected 'dry' mention in output to distinguish from a real run"

    def test_dry_run_does_not_require_config_file(self, tmp_path: Path) -> None:
        """Verify dry-run works even if config file path doesn't exist.

        The config file is not needed during dry-run since no computation occurs.
        The config path is only stored in the report, not read.
        """
        nonexistent_config = str(tmp_path / "does_not_exist.yaml")
        result = _run_sbir_demo(
            [
                "--dry-run",
                "--config",
                nonexistent_config,
                "--output-dir",
                str(tmp_path),
                "--formats",
                "json",
            ],
        )
        # Dry-run should succeed regardless of config file existence
        assert result.returncode == 0, (
            f"Dry-run should not require config file, got exit {result.returncode}.\n"
            f"stderr: {result.stderr}"
        )

    def test_dry_run_with_markdown_format(self, tmp_path: Path) -> None:
        """Verify results.md is created when markdown format is requested."""
        result = _run_sbir_demo(
            [
                "--dry-run",
                "--output-dir",
                str(tmp_path),
                "--formats",
                "json",
                "markdown",
            ],
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        md_path = tmp_path / "results.md"
        assert md_path.exists(), f"results.md not found in {tmp_path}"
        content = md_path.read_text(encoding="utf-8")
        assert len(content) > 0, "results.md should not be empty"

    def test_dry_run_with_tempfile(self) -> None:
        """Verify dry-run works with an OS-allocated temp directory."""
        with tempfile.TemporaryDirectory(prefix="sbir_e2e_") as tmpdir:
            result = _run_sbir_demo(
                ["--dry-run", "--output-dir", tmpdir, "--formats", "json"],
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            json_path = Path(tmpdir) / "results.json"
            assert json_path.exists()
            data = json.loads(json_path.read_text(encoding="utf-8"))
            assert "transfer_mse" in data
