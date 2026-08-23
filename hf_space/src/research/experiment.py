"""Experiment management and tracking.

Provides:
- Experiment lifecycle management
- Run tracking and metrics logging
- Artifact persistence
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.research.config import ExperimentConfig


@dataclass
class ExperimentRun:
    """A single run of an experiment.

    Tracks metrics, artifacts, and status for one configuration.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    config_hash: str = ""
    start_time: str | None = None
    end_time: str | None = None
    status: str = "pending"  # pending, running, completed, failed
    metrics: dict[str, list[float]] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    hyperparams: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark run as started."""
        self.status = "running"
        self.start_time = datetime.now(timezone.utc).isoformat()

    def complete(self) -> None:
        """Mark run as completed."""
        self.status = "completed"
        self.end_time = datetime.now(timezone.utc).isoformat()

        # Compute final metrics from history
        for name, values in self.metrics.items():
            if values:
                self.final_metrics[f"{name}_final"] = values[-1]
                self.final_metrics[f"{name}_best"] = (
                    min(values) if "loss" in name.lower() else max(values)
                )
                self.final_metrics[f"{name}_mean"] = sum(values) / len(values)

    def fail(self, error: str) -> None:
        """Mark run as failed."""
        self.status = "failed"
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.metadata["error"] = error

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a metric value.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step number.

        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

        if step is not None:
            if "steps" not in self.metadata:
                self.metadata["steps"] = {}
            self.metadata["steps"][name] = self.metadata["steps"].get(name, []) + [step]

    def log_artifact(self, name: str, path: str) -> None:
        """Log an artifact path.

        Args:
            name: Artifact name.
            path: Path to artifact.

        """
        self.artifacts[name] = path

    @property
    def duration_seconds(self) -> float | None:
        """Get run duration."""
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "final_metrics": self.final_metrics,
            "artifacts": self.artifacts,
            "hyperparams": self.hyperparams,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRun:
        """Create from dictionary."""
        run = cls(
            run_id=data.get("run_id", str(uuid.uuid4())[:8]),
            config_hash=data.get("config_hash", ""),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            status=data.get("status", "pending"),
        )
        run.metrics = data.get("metrics", {})
        run.final_metrics = data.get("final_metrics", {})
        run.artifacts = data.get("artifacts", {})
        run.hyperparams = data.get("hyperparams", {})
        run.metadata = data.get("metadata", {})
        return run


class Experiment:
    """Manages a research experiment.

    Coordinates multiple runs with different configurations.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize experiment.

        Args:
            config: Experiment configuration.
            logger: Optional structured logger.

        """
        self.config = config
        self._logger = logger or structlog.get_logger(__name__).bind(
            experiment=config.name,
        )
        self._runs: list[ExperimentRun] = []
        self._current_run: ExperimentRun | None = None
        self._output_dir = Path(config.output_dir) / config.name

        # Callbacks
        self._on_run_start: list[Callable[[ExperimentRun], None]] = []
        self._on_run_end: list[Callable[[ExperimentRun], None]] = []
        self._on_metric: list[Callable[[str, float, int | None], None]] = []

    @property
    def runs(self) -> list[ExperimentRun]:
        """Get all runs."""
        return self._runs

    @property
    def current_run(self) -> ExperimentRun | None:
        """Get current run."""
        return self._current_run

    @property
    def is_running(self) -> bool:
        """Check if experiment is running."""
        return self._current_run is not None and self._current_run.status == "running"

    def start_run(
        self,
        hyperparams: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentRun:
        """Start a new run.

        Args:
            hyperparams: Run hyperparameters.
            metadata: Additional metadata.

        Returns:
            New ExperimentRun.

        """
        if self._current_run and self._current_run.status == "running":
            self._logger.warning("ending_previous_run")
            self._current_run.complete()

        run = ExperimentRun(
            config_hash=self.config.compute_hash(),
            hyperparams=hyperparams or {},
            metadata=metadata or {},
        )
        run.start()

        self._current_run = run
        self._runs.append(run)

        self._logger.info(
            "run_started",
            run_id=run.run_id,
            hyperparams=run.hyperparams,
        )

        # Fire callbacks
        for callback in self._on_run_start:
            callback(run)

        return run

    def end_run(self, success: bool = True, error: str | None = None) -> None:
        """End the current run.

        Args:
            success: Whether run succeeded.
            error: Error message if failed.

        """
        if not self._current_run:
            return

        if success:
            self._current_run.complete()
        else:
            self._current_run.fail(error or "Unknown error")

        self._logger.info(
            "run_ended",
            run_id=self._current_run.run_id,
            status=self._current_run.status,
            duration=self._current_run.duration_seconds,
        )

        # Fire callbacks
        for callback in self._on_run_end:
            callback(self._current_run)

        self._current_run = None

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a metric to current run.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step number.

        """
        if self._current_run:
            self._current_run.log_metric(name, value, step)

            # Fire callbacks
            for callback in self._on_metric:
                callback(name, value, step)

    def log_artifact(self, name: str, path: str | Path) -> None:
        """Log an artifact to current run.

        Args:
            name: Artifact name.
            path: Path to artifact.

        """
        if self._current_run:
            self._current_run.log_artifact(name, str(path))

    def get_best_run(self, metric: str, minimize: bool = True) -> ExperimentRun | None:
        """Get the best run based on a metric.

        Args:
            metric: Metric name.
            minimize: Whether lower is better.

        Returns:
            Best run or None.

        """
        completed_runs = [r for r in self._runs if r.status == "completed"]
        if not completed_runs:
            return None

        def get_value(run: ExperimentRun) -> float:
            if metric in run.final_metrics:
                return run.final_metrics[metric]
            if f"{metric}_final" in run.final_metrics:
                return run.final_metrics[f"{metric}_final"]
            return float("inf") if minimize else float("-inf")

        return (
            min(completed_runs, key=get_value) if minimize else max(completed_runs, key=get_value)
        )

    def save(self, path: Path | str | None = None) -> Path:
        """Save experiment state.

        Args:
            path: Optional save path.

        Returns:
            Path to saved file.

        """
        if path is None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            path = self._output_dir / "experiment.json"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": self.config.model_dump(mode="json"),
            "runs": [run.to_dict() for run in self._runs],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path

    @classmethod
    def load(cls, path: Path | str) -> Experiment:
        """Load experiment from file.

        Args:
            path: Path to experiment file.

        Returns:
            Loaded Experiment.

        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        config = ExperimentConfig(**data["config"])
        experiment = cls(config=config)
        experiment._runs = [ExperimentRun.from_dict(r) for r in data.get("runs", [])]

        return experiment

    def on_run_start(self, callback: Callable[[ExperimentRun], None]) -> None:
        """Register callback for run start.

        Args:
            callback: Function to call.

        """
        self._on_run_start.append(callback)

    def on_run_end(self, callback: Callable[[ExperimentRun], None]) -> None:
        """Register callback for run end.

        Args:
            callback: Function to call.

        """
        self._on_run_end.append(callback)

    def on_metric(self, callback: Callable[[str, float, int | None], None]) -> None:
        """Register callback for metric logging.

        Args:
            callback: Function to call.

        """
        self._on_metric.append(callback)

    def get_summary(self) -> dict[str, Any]:
        """Get experiment summary.

        Returns:
            Summary dictionary.

        """
        completed = [r for r in self._runs if r.status == "completed"]
        failed = [r for r in self._runs if r.status == "failed"]

        return {
            "name": self.config.name,
            "description": self.config.description,
            "config_hash": self.config.compute_hash(),
            "total_runs": len(self._runs),
            "completed_runs": len(completed),
            "failed_runs": len(failed),
            "is_running": self.is_running,
        }


class ExperimentTracker:
    """Tracks multiple experiments.

    Provides centralized experiment management.
    """

    def __init__(
        self,
        base_dir: Path | str = "outputs/research",
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize tracker.

        Args:
            base_dir: Base directory for experiments.
            logger: Optional structured logger.

        """
        self._base_dir = Path(base_dir)
        self._logger = logger or structlog.get_logger(__name__)
        self._experiments: dict[str, Experiment] = {}

    @property
    def experiments(self) -> dict[str, Experiment]:
        """Get all tracked experiments."""
        return self._experiments

    def create_experiment(
        self,
        name: str,
        config: ExperimentConfig | None = None,
        **kwargs: Any,
    ) -> Experiment:
        """Create and track a new experiment.

        Args:
            name: Experiment name.
            config: Optional configuration.
            **kwargs: Config overrides.

        Returns:
            Created Experiment.

        """
        if config is None:
            config = ExperimentConfig(name=name, **kwargs)

        experiment = Experiment(config=config)
        self._experiments[name] = experiment

        self._logger.info("experiment_created", name=name)

        return experiment

    def get_experiment(self, name: str) -> Experiment | None:
        """Get experiment by name.

        Args:
            name: Experiment name.

        Returns:
            Experiment or None.

        """
        return self._experiments.get(name)

    def list_experiments(self) -> list[str]:
        """List all experiment names.

        Returns:
            List of names.

        """
        return list(self._experiments.keys())

    def load_experiments(self) -> int:
        """Load experiments from base directory.

        Returns:
            Number of experiments loaded.

        """
        count = 0
        if self._base_dir.exists():
            for exp_dir in self._base_dir.iterdir():
                if exp_dir.is_dir():
                    exp_file = exp_dir / "experiment.json"
                    if exp_file.exists():
                        experiment = Experiment.load(exp_file)
                        self._experiments[experiment.config.name] = experiment
                        count += 1

        self._logger.info("experiments_loaded", count=count)
        return count

    def save_all(self) -> int:
        """Save all experiments.

        Returns:
            Number saved.

        """
        count = 0
        for experiment in self._experiments.values():
            experiment.save()
            count += 1

        return count

    def get_best_across_experiments(
        self,
        metric: str,
        minimize: bool = True,
    ) -> tuple[str, ExperimentRun] | None:
        """Find best run across all experiments.

        Args:
            metric: Metric name.
            minimize: Whether lower is better.

        Returns:
            Tuple of (experiment_name, best_run) or None.

        """
        best_exp = None
        best_run = None
        best_value = float("inf") if minimize else float("-inf")

        for name, experiment in self._experiments.items():
            run = experiment.get_best_run(metric, minimize)
            if run:
                value = run.final_metrics.get(metric) or run.final_metrics.get(f"{metric}_final")
                if value is not None:
                    if (minimize and value < best_value) or (not minimize and value > best_value):
                        best_value = value
                        best_exp = name
                        best_run = run

        if best_exp and best_run:
            return (best_exp, best_run)
        return None

    def iter_runs(self) -> Iterator[tuple[str, ExperimentRun]]:
        """Iterate through all runs across experiments.

        Yields:
            Tuples of (experiment_name, run).

        """
        for name, experiment in self._experiments.items():
            for run in experiment.runs:
                yield name, run


def create_experiment(
    name: str,
    experiment_type: str = "custom",
    **kwargs: Any,
) -> Experiment:
    """Factory function to create an experiment.

    Args:
        name: Experiment name.
        experiment_type: Type of experiment.
        **kwargs: Additional configuration.

    Returns:
        Experiment instance.

    """
    from src.research.config import ExperimentType

    config = ExperimentConfig(
        name=name,
        experiment_type=ExperimentType(experiment_type),
        **kwargs,
    )
    return Experiment(config=config)
