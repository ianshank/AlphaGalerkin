"""Result collection and persistence for PoC scenarios.

This module handles:
    - Collecting results from scenario executions
    - Persisting results to JSON/Parquet
    - Generating reports and visualizations
    - Comparing results across runs
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.poc.config import ScenarioResult, ScenarioStatus

logger = structlog.get_logger(__name__)


def _generate_run_id() -> str:
    """Generate a unique run ID combining timestamp and UUID.

    This ensures uniqueness even with concurrent collectors while
    maintaining human-readable timestamp prefix.

    Returns:
        Unique run ID string.

    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{unique_suffix}"


class ResultCollector:
    """Collects and persists scenario results.

    Provides:
        - In-memory result storage
        - JSON/Parquet persistence
        - Summary generation
        - Result comparison
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs/poc",
        run_id: str | None = None,
    ) -> None:
        """Initialize collector.

        Args:
            output_dir: Directory for result files.
            run_id: Unique identifier for this run. Auto-generated if None.

        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or _generate_run_id()
        self.results: list[ScenarioResult] = []

        self._logger = structlog.get_logger(
            __name__,
            run_id=self.run_id,
        )

    def collect(self, result: ScenarioResult) -> None:
        """Add a result to the collection.

        Args:
            result: Scenario result to collect.

        """
        self.results.append(result)

        # Persist immediately for fault tolerance
        self._save_result(result)

        self._logger.info(
            "result_collected",
            scenario=result.scenario_name,
            status=result.status.value,
            passed=result.passed,
        )

    def _save_result(self, result: ScenarioResult) -> Path:
        """Save individual result to JSON.

        Args:
            result: Result to save.

        Returns:
            Path to saved file.

        """
        results_dir = self.output_dir / "results" / self.run_id
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{result.scenario_name}_{result.config_hash}.json"
        filepath = results_dir / filename

        with open(filepath, "w") as f:
            json.dump(
                result.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

        return filepath

    def save_summary(self, results: list[ScenarioResult] | None = None) -> Path:
        """Save summary of all results.

        Args:
            results: Optional results list. Uses collected if None.

        Returns:
            Path to summary file.

        """
        results = results or self.results

        summary = self._generate_summary(results)

        summary_dir = self.output_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)

        filepath = summary_dir / f"summary_{self.run_id}.json"

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self._logger.info("summary_saved", path=str(filepath))

        return filepath

    def _generate_summary(self, results: list[ScenarioResult]) -> dict[str, Any]:
        """Generate summary statistics.

        Args:
            results: List of results.

        Returns:
            Summary dictionary.

        """
        if not results:
            return {
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "scenarios": [],
            }

        status_counts = {
            "passed": sum(1 for r in results if r.status == ScenarioStatus.PASSED),
            "failed": sum(1 for r in results if r.status == ScenarioStatus.FAILED),
            "errors": sum(1 for r in results if r.status == ScenarioStatus.ERROR),
            "skipped": sum(1 for r in results if r.status == ScenarioStatus.SKIPPED),
        }

        total_duration = sum(r.duration_seconds for r in results)

        # Aggregate metrics across all scenarios
        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for name, value in result.metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)

        metric_summaries = {}
        for name, values in all_metrics.items():
            metric_summaries[name] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "count": len(values),
            }

        return {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "total_duration_seconds": total_duration,
            **status_counts,
            "pass_rate": status_counts["passed"] / len(results) if results else 0.0,
            "metric_summaries": metric_summaries,
            "scenarios": [
                {
                    "name": r.scenario_name,
                    "status": r.status.value,
                    "passed": r.passed,
                    "duration_seconds": r.duration_seconds,
                    "metrics": r.metrics,
                }
                for r in results
            ],
        }

    def load_results(self, run_id: str) -> list[ScenarioResult]:
        """Load results from a previous run.

        Args:
            run_id: Run identifier.

        Returns:
            List of ScenarioResult.

        """
        results_dir = self.output_dir / "results" / run_id

        if not results_dir.exists():
            self._logger.warning("results_not_found", run_id=run_id)
            return []

        results = []
        for filepath in results_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    results.append(ScenarioResult(**data))
            except json.JSONDecodeError as e:
                self._logger.warning(
                    "json_decode_error",
                    file=str(filepath),
                    error=str(e),
                )
            except Exception as e:
                self._logger.warning(
                    "result_load_error",
                    file=str(filepath),
                    error=str(e),
                )

        self._logger.info("results_loaded", run_id=run_id, count=len(results))

        return results

    def compare_runs(
        self,
        run_id_a: str,
        run_id_b: str,
    ) -> dict[str, Any]:
        """Compare results between two runs.

        Args:
            run_id_a: First run ID.
            run_id_b: Second run ID.

        Returns:
            Comparison dictionary.

        """
        results_a = self.load_results(run_id_a)
        results_b = self.load_results(run_id_b)

        # Index by scenario name
        index_a = {r.scenario_name: r for r in results_a}
        index_b = {r.scenario_name: r for r in results_b}

        all_scenarios = set(index_a.keys()) | set(index_b.keys())

        comparisons = []
        for name in sorted(all_scenarios):
            result_a = index_a.get(name)
            result_b = index_b.get(name)

            comparison: dict[str, Any] = {"scenario": name}

            if result_a and result_b:
                comparison["status_a"] = result_a.status.value
                comparison["status_b"] = result_b.status.value
                comparison["status_changed"] = result_a.status != result_b.status

                # Compare metrics
                metric_changes = {}
                all_metrics = set(result_a.metrics.keys()) | set(result_b.metrics.keys())

                for metric in all_metrics:
                    val_a = result_a.metrics.get(metric)
                    val_b = result_b.metrics.get(metric)

                    if val_a is not None and val_b is not None:
                        delta = val_b - val_a
                        pct_change = (delta / val_a * 100) if val_a != 0 else 0
                        metric_changes[metric] = {
                            "a": val_a,
                            "b": val_b,
                            "delta": delta,
                            "pct_change": pct_change,
                        }

                comparison["metric_changes"] = metric_changes

            elif result_a:
                comparison["only_in"] = "a"
                comparison["status_a"] = result_a.status.value
            else:
                comparison["only_in"] = "b"
                comparison["status_b"] = result_b.status.value if result_b else None

            comparisons.append(comparison)

        return {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "timestamp": datetime.now().isoformat(),
            "scenarios_compared": len(comparisons),
            "comparisons": comparisons,
        }

    def to_dataframe(self, results: list[ScenarioResult] | None = None) -> Any:
        """Convert results to pandas DataFrame (if pandas available).

        Args:
            results: Results to convert. Uses collected if None.

        Returns:
            pandas DataFrame or dict if pandas not available.

        """
        results = results or self.results

        rows = []
        for r in results:
            row = {
                "scenario": r.scenario_name,
                "status": r.status.value,
                "passed": r.passed,
                "duration_seconds": r.duration_seconds,
                "config_hash": r.config_hash,
                "timestamp": r.start_time.isoformat(),
            }
            # Flatten metrics
            for name, value in r.metrics.items():
                row[f"metric_{name}"] = value
            rows.append(row)

        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            self._logger.debug("pandas_not_available")
            return rows


def create_collector(
    output_dir: str | Path = "outputs/poc",
    run_id: str | None = None,
) -> ResultCollector:
    """Factory function to create a configured collector.

    Args:
        output_dir: Output directory.
        run_id: Optional run identifier.

    Returns:
        Configured ResultCollector.

    """
    return ResultCollector(output_dir=output_dir, run_id=run_id)
