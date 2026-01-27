"""Tests for result collection and persistence.

Validates:
    - Result collection
    - JSON persistence
    - Summary generation
    - Run comparison
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.results import ResultCollector, create_collector


@pytest.fixture
def temp_output_dir() -> Path:
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_result() -> ScenarioResult:
    """Create a sample result for testing."""
    return ScenarioResult(
        scenario_name="test_scenario",
        config_hash="abc123",
        status=ScenarioStatus.PASSED,
        passed=True,
        metrics={"mse": 0.03, "mae": 0.02},
        threshold_results={"mse": True},
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration_seconds=10.0,
        device="cpu",
        python_version="3.11.0",
        torch_version="2.0.0",
    )


class TestResultCollector:
    """Tests for ResultCollector."""

    def test_create_collector(self, temp_output_dir: Path) -> None:
        """Test collector creation."""
        collector = create_collector(output_dir=temp_output_dir)

        assert collector.output_dir == temp_output_dir
        assert collector.run_id is not None

    def test_collect_result(
        self, temp_output_dir: Path, sample_result: ScenarioResult
    ) -> None:
        """Test collecting a result."""
        collector = ResultCollector(output_dir=temp_output_dir)
        collector.collect(sample_result)

        assert len(collector.results) == 1
        assert collector.results[0].scenario_name == "test_scenario"

    def test_result_persisted(
        self, temp_output_dir: Path, sample_result: ScenarioResult
    ) -> None:
        """Test that result is persisted to JSON."""
        collector = ResultCollector(output_dir=temp_output_dir)
        collector.collect(sample_result)

        # Check file exists
        results_dir = temp_output_dir / "results" / collector.run_id
        assert results_dir.exists()

        # Verify content
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) == 1

        with open(result_files[0]) as f:
            data = json.load(f)

        assert data["scenario_name"] == "test_scenario"
        assert data["metrics"]["mse"] == 0.03


class TestSummaryGeneration:
    """Tests for summary generation."""

    def test_save_summary(
        self, temp_output_dir: Path, sample_result: ScenarioResult
    ) -> None:
        """Test saving summary."""
        collector = ResultCollector(output_dir=temp_output_dir)
        collector.collect(sample_result)

        filepath = collector.save_summary()

        assert filepath.exists()
        assert filepath.suffix == ".json"

    def test_summary_content(
        self, temp_output_dir: Path, sample_result: ScenarioResult
    ) -> None:
        """Test summary content."""
        collector = ResultCollector(output_dir=temp_output_dir)
        collector.collect(sample_result)

        filepath = collector.save_summary()

        with open(filepath) as f:
            summary = json.load(f)

        assert summary["total"] == 1
        assert summary["passed"] == 1
        assert summary["failed"] == 0
        assert summary["pass_rate"] == 1.0
        assert "mse" in summary["metric_summaries"]

    def test_summary_multiple_results(self, temp_output_dir: Path) -> None:
        """Test summary with multiple results."""
        collector = ResultCollector(output_dir=temp_output_dir)

        # Add passed result
        collector.collect(
            ScenarioResult(
                scenario_name="passed_test",
                config_hash="hash1",
                status=ScenarioStatus.PASSED,
                passed=True,
                metrics={"accuracy": 0.95},
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=5.0,
            )
        )

        # Add failed result
        collector.collect(
            ScenarioResult(
                scenario_name="failed_test",
                config_hash="hash2",
                status=ScenarioStatus.FAILED,
                passed=False,
                metrics={"accuracy": 0.5},
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=3.0,
            )
        )

        filepath = collector.save_summary()

        with open(filepath) as f:
            summary = json.load(f)

        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5


class TestResultLoading:
    """Tests for loading saved results."""

    def test_load_results(
        self, temp_output_dir: Path, sample_result: ScenarioResult
    ) -> None:
        """Test loading results from disk."""
        # Save results
        collector1 = ResultCollector(output_dir=temp_output_dir, run_id="test_run")
        collector1.collect(sample_result)

        # Load results
        collector2 = ResultCollector(output_dir=temp_output_dir)
        loaded = collector2.load_results("test_run")

        assert len(loaded) == 1
        assert loaded[0].scenario_name == "test_scenario"

    def test_load_nonexistent_run(self, temp_output_dir: Path) -> None:
        """Test loading non-existent run returns empty list."""
        collector = ResultCollector(output_dir=temp_output_dir)
        loaded = collector.load_results("nonexistent")

        assert loaded == []


class TestRunComparison:
    """Tests for comparing runs."""

    def test_compare_runs(self, temp_output_dir: Path) -> None:
        """Test comparing two runs."""
        # Run A
        collector_a = ResultCollector(output_dir=temp_output_dir, run_id="run_a")
        collector_a.collect(
            ScenarioResult(
                scenario_name="test",
                config_hash="hash1",
                status=ScenarioStatus.PASSED,
                passed=True,
                metrics={"mse": 0.05},
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=10.0,
            )
        )

        # Run B (improved mse)
        collector_b = ResultCollector(output_dir=temp_output_dir, run_id="run_b")
        collector_b.collect(
            ScenarioResult(
                scenario_name="test",
                config_hash="hash1",
                status=ScenarioStatus.PASSED,
                passed=True,
                metrics={"mse": 0.03},
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=8.0,
            )
        )

        # Compare
        collector = ResultCollector(output_dir=temp_output_dir)
        comparison = collector.compare_runs("run_a", "run_b")

        assert comparison["scenarios_compared"] == 1
        assert len(comparison["comparisons"]) == 1

        comp = comparison["comparisons"][0]
        assert comp["scenario"] == "test"
        assert comp["metric_changes"]["mse"]["a"] == pytest.approx(0.05)
        assert comp["metric_changes"]["mse"]["b"] == pytest.approx(0.03)
        assert comp["metric_changes"]["mse"]["delta"] == pytest.approx(-0.02)

    def test_compare_with_new_scenario(self, temp_output_dir: Path) -> None:
        """Test comparing runs with scenario added in second run."""
        # Run A
        collector_a = ResultCollector(output_dir=temp_output_dir, run_id="run_a")
        collector_a.collect(
            ScenarioResult(
                scenario_name="existing",
                config_hash="hash1",
                status=ScenarioStatus.PASSED,
                passed=True,
                metrics={},
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.0,
            )
        )

        # Run B (with new scenario)
        collector_b = ResultCollector(output_dir=temp_output_dir, run_id="run_b")
        collector_b.collect(
            ScenarioResult(
                scenario_name="existing",
                config_hash="hash1",
                status=ScenarioStatus.PASSED,
                passed=True,
                metrics={},
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.0,
            )
        )
        collector_b.collect(
            ScenarioResult(
                scenario_name="new_scenario",
                config_hash="hash2",
                status=ScenarioStatus.PASSED,
                passed=True,
                metrics={},
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.0,
            )
        )

        collector = ResultCollector(output_dir=temp_output_dir)
        comparison = collector.compare_runs("run_a", "run_b")

        assert comparison["scenarios_compared"] == 2

        # Find the new scenario comparison
        new_comp = next(
            c for c in comparison["comparisons"] if c["scenario"] == "new_scenario"
        )
        assert new_comp["only_in"] == "b"


class TestDataFrameConversion:
    """Tests for DataFrame conversion."""

    def test_to_dataframe_without_pandas(
        self, temp_output_dir: Path, sample_result: ScenarioResult
    ) -> None:
        """Test conversion returns list when pandas not available."""
        collector = ResultCollector(output_dir=temp_output_dir)
        collector.collect(sample_result)

        result = collector.to_dataframe()

        # Either returns DataFrame or list of dicts
        if isinstance(result, list):
            assert len(result) == 1
            assert result[0]["scenario"] == "test_scenario"
        else:
            # pandas is available
            assert len(result) == 1
            assert result.iloc[0]["scenario"] == "test_scenario"
