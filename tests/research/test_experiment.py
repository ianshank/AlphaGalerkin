"""Tests for experiment management."""

from __future__ import annotations

import tempfile
from pathlib import Path


from src.research.experiment import (
    Experiment,
    ExperimentRun,
    ExperimentTracker,
    create_experiment,
)


class TestExperimentRun:
    """Tests for ExperimentRun dataclass."""

    def test_default_values(self, experiment_run: ExperimentRun) -> None:
        """Test default run values."""
        assert experiment_run.run_id == "test123"
        assert experiment_run.status == "pending"
        assert len(experiment_run.metrics) == 0

    def test_start(self, experiment_run: ExperimentRun) -> None:
        """Test starting a run."""
        experiment_run.start()
        assert experiment_run.status == "running"
        assert experiment_run.start_time is not None

    def test_complete(self, experiment_run: ExperimentRun) -> None:
        """Test completing a run."""
        experiment_run.start()
        experiment_run.log_metric("loss", 0.5)
        experiment_run.log_metric("loss", 0.3)
        experiment_run.complete()

        assert experiment_run.status == "completed"
        assert experiment_run.end_time is not None
        assert "loss_final" in experiment_run.final_metrics
        assert "loss_best" in experiment_run.final_metrics

    def test_fail(self, experiment_run: ExperimentRun) -> None:
        """Test failing a run."""
        experiment_run.start()
        experiment_run.fail("Out of memory")

        assert experiment_run.status == "failed"
        assert experiment_run.metadata["error"] == "Out of memory"

    def test_log_metric(self, experiment_run: ExperimentRun) -> None:
        """Test logging metrics."""
        experiment_run.log_metric("loss", 0.5)
        experiment_run.log_metric("loss", 0.3)
        experiment_run.log_metric("accuracy", 0.8)

        assert len(experiment_run.metrics["loss"]) == 2
        assert experiment_run.metrics["loss"][1] == 0.3
        assert experiment_run.metrics["accuracy"][0] == 0.8

    def test_log_artifact(self, experiment_run: ExperimentRun) -> None:
        """Test logging artifacts."""
        experiment_run.log_artifact("model", "/path/to/model.pt")
        assert experiment_run.artifacts["model"] == "/path/to/model.pt"

    def test_duration_seconds(self, experiment_run: ExperimentRun) -> None:
        """Test duration calculation."""
        assert experiment_run.duration_seconds is None

        experiment_run.start()
        experiment_run.complete()

        assert experiment_run.duration_seconds is not None
        assert experiment_run.duration_seconds >= 0

    def test_to_dict(self, experiment_run: ExperimentRun) -> None:
        """Test serialization to dict."""
        experiment_run.log_metric("loss", 0.5)
        data = experiment_run.to_dict()

        assert data["run_id"] == "test123"
        assert "loss" in data["metrics"]

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "run_id": "loaded123",
            "status": "completed",
            "metrics": {"loss": [0.5, 0.3]},
        }

        run = ExperimentRun.from_dict(data)
        assert run.run_id == "loaded123"
        assert run.status == "completed"
        assert len(run.metrics["loss"]) == 2


class TestExperiment:
    """Tests for Experiment."""

    def test_initialization(self, experiment: Experiment) -> None:
        """Test experiment initialization."""
        assert experiment.config.name == "test_experiment"
        assert len(experiment.runs) == 0
        assert not experiment.is_running

    def test_start_run(self, experiment: Experiment) -> None:
        """Test starting a run."""
        run = experiment.start_run()

        assert run.status == "running"
        assert experiment.current_run == run
        assert len(experiment.runs) == 1
        assert experiment.is_running

    def test_start_run_with_hyperparams(self, experiment: Experiment) -> None:
        """Test starting run with hyperparameters."""
        run = experiment.start_run(
            hyperparams={"lr": 0.001},
            metadata={"version": "1.0"},
        )

        assert run.hyperparams["lr"] == 0.001
        assert run.metadata["version"] == "1.0"

    def test_end_run_success(self, experiment: Experiment) -> None:
        """Test ending run successfully."""
        run = experiment.start_run()
        experiment.log_metric("loss", 0.5)
        experiment.end_run(success=True)

        assert run.status == "completed"
        assert experiment.current_run is None
        assert not experiment.is_running

    def test_end_run_failure(self, experiment: Experiment) -> None:
        """Test ending run with failure."""
        run = experiment.start_run()
        experiment.end_run(success=False, error="GPU OOM")

        assert run.status == "failed"
        assert run.metadata["error"] == "GPU OOM"

    def test_log_metric(self, experiment: Experiment) -> None:
        """Test logging metric."""
        run = experiment.start_run()
        experiment.log_metric("loss", 0.5, step=1)
        experiment.log_metric("loss", 0.3, step=2)

        assert len(run.metrics["loss"]) == 2

    def test_log_artifact(self, experiment: Experiment) -> None:
        """Test logging artifact."""
        run = experiment.start_run()
        experiment.log_artifact("model", "/path/to/model")

        assert run.artifacts["model"] == "/path/to/model"

    def test_get_best_run(self, experiment: Experiment) -> None:
        """Test getting best run."""
        # First run
        run1 = experiment.start_run()
        run1.log_metric("loss", 0.5)
        experiment.end_run()

        # Second run
        run2 = experiment.start_run()
        run2.log_metric("loss", 0.3)
        experiment.end_run()

        best = experiment.get_best_run("loss", minimize=True)
        assert best is not None
        assert best.run_id == run2.run_id

    def test_save_and_load(self, experiment: Experiment) -> None:
        """Test saving and loading experiment."""
        experiment.start_run()
        experiment.log_metric("loss", 0.5)
        experiment.end_run()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = experiment.save(Path(tmpdir) / "exp.json")
            assert path.exists()

            loaded = Experiment.load(path)
            assert loaded.config.name == "test_experiment"
            assert len(loaded.runs) == 1

    def test_callbacks(self, experiment: Experiment) -> None:
        """Test callbacks."""
        start_called = []
        end_called = []
        metric_called = []

        experiment.on_run_start(lambda r: start_called.append(r.run_id))
        experiment.on_run_end(lambda r: end_called.append(r.run_id))
        experiment.on_metric(lambda n, v, s: metric_called.append((n, v)))

        run = experiment.start_run()
        assert len(start_called) == 1

        experiment.log_metric("loss", 0.5)
        assert len(metric_called) == 1

        experiment.end_run()
        assert len(end_called) == 1

    def test_get_summary(self, experiment: Experiment) -> None:
        """Test getting summary."""
        experiment.start_run()
        experiment.end_run()
        experiment.start_run()
        experiment.end_run(success=False, error="test")

        summary = experiment.get_summary()

        assert summary["name"] == "test_experiment"
        assert summary["total_runs"] == 2
        assert summary["completed_runs"] == 1
        assert summary["failed_runs"] == 1


class TestExperimentTracker:
    """Tests for ExperimentTracker."""

    def test_initialization(
        self, experiment_tracker: ExperimentTracker
    ) -> None:
        """Test tracker initialization."""
        assert len(experiment_tracker.experiments) == 0

    def test_create_experiment(
        self, experiment_tracker: ExperimentTracker
    ) -> None:
        """Test creating experiment."""
        exp = experiment_tracker.create_experiment("test")

        assert exp.config.name == "test"
        assert "test" in experiment_tracker.experiments

    def test_get_experiment(
        self, experiment_tracker: ExperimentTracker
    ) -> None:
        """Test getting experiment."""
        experiment_tracker.create_experiment("test1")
        experiment_tracker.create_experiment("test2")

        assert experiment_tracker.get_experiment("test1") is not None
        assert experiment_tracker.get_experiment("nonexistent") is None

    def test_list_experiments(
        self, experiment_tracker: ExperimentTracker
    ) -> None:
        """Test listing experiments."""
        experiment_tracker.create_experiment("exp1")
        experiment_tracker.create_experiment("exp2")

        names = experiment_tracker.list_experiments()
        assert "exp1" in names
        assert "exp2" in names

    def test_iter_runs(
        self, experiment_tracker: ExperimentTracker
    ) -> None:
        """Test iterating runs."""
        exp1 = experiment_tracker.create_experiment("exp1")
        exp1.start_run()
        exp1.end_run()

        exp2 = experiment_tracker.create_experiment("exp2")
        exp2.start_run()
        exp2.end_run()

        runs = list(experiment_tracker.iter_runs())
        assert len(runs) == 2


class TestCreateExperiment:
    """Tests for create_experiment factory."""

    def test_create_default(self) -> None:
        """Test creating default experiment."""
        exp = create_experiment(name="test")
        assert exp.config.name == "test"

    def test_create_with_type(self) -> None:
        """Test creating with type."""
        exp = create_experiment(name="ablation", experiment_type="ablation")
        assert exp.config.experiment_type.value == "ablation"
