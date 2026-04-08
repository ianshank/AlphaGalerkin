"""Additional coverage tests for research/experiment.py.

Covers: get_best_run, save/load, ExperimentTracker, callbacks, end_run failure.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.research.config import ExperimentConfig
from src.research.experiment import Experiment, ExperimentRun, ExperimentTracker


@pytest.fixture
def tmp_dir() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def experiment(tmp_dir: Path) -> Experiment:
    cfg = ExperimentConfig(
        name="test_exp",
        description="Test experiment",
        output_dir=str(tmp_dir),
    )
    return Experiment(config=cfg)


class TestExperimentRunComplete:
    """Tests for ExperimentRun.complete final_metrics."""

    def test_complete_computes_final_metrics(self) -> None:
        run = ExperimentRun()
        run.start()
        run.log_metric("loss", 1.0)
        run.log_metric("loss", 0.5)
        run.log_metric("accuracy", 0.8)
        run.log_metric("accuracy", 0.9)
        run.complete()
        assert run.final_metrics["loss_final"] == 0.5
        assert run.final_metrics["loss_best"] == 0.5
        assert run.final_metrics["accuracy_final"] == 0.9
        assert run.final_metrics["accuracy_best"] == 0.9

    def test_fail_sets_error(self) -> None:
        run = ExperimentRun()
        run.start()
        run.fail("boom")
        assert run.status == "failed"
        assert run.metadata["error"] == "boom"

    def test_duration_seconds(self) -> None:
        run = ExperimentRun()
        run.start()
        run.complete()
        dur = run.duration_seconds
        assert dur is not None
        assert dur >= 0

    def test_duration_none_without_end(self) -> None:
        run = ExperimentRun()
        assert run.duration_seconds is None

    def test_from_dict_roundtrip(self) -> None:
        run = ExperimentRun()
        run.start()
        run.log_metric("x", 1.0)
        run.complete()
        d = run.to_dict()
        run2 = ExperimentRun.from_dict(d)
        assert run2.run_id == run.run_id
        assert run2.status == "completed"
        assert run2.metrics == run.metrics

    def test_log_metric_with_step(self) -> None:
        run = ExperimentRun()
        run.log_metric("lr", 0.01, step=1)
        run.log_metric("lr", 0.005, step=2)
        assert run.metadata["steps"]["lr"] == [1, 2]


class TestExperimentCallbacks:
    """Tests for Experiment callback system."""

    def test_on_run_start_callback(self, experiment: Experiment) -> None:
        started = []
        experiment.on_run_start(lambda r: started.append(r.run_id))
        run = experiment.start_run()
        assert len(started) == 1
        experiment.end_run()

    def test_on_run_end_callback(self, experiment: Experiment) -> None:
        ended = []
        experiment.on_run_end(lambda r: ended.append(r.status))
        experiment.start_run()
        experiment.end_run()
        assert ended == ["completed"]

    def test_on_metric_callback(self, experiment: Experiment) -> None:
        metrics = []
        experiment.on_metric(lambda n, v, s: metrics.append((n, v)))
        experiment.start_run()
        experiment.log_metric("x", 42.0)
        assert metrics == [("x", 42.0)]
        experiment.end_run()


class TestExperimentEndRunFailure:
    """Tests for end_run with failure."""

    def test_end_run_failure(self, experiment: Experiment) -> None:
        experiment.start_run()
        experiment.end_run(success=False, error="test error")
        run = experiment.runs[-1]
        assert run.status == "failed"

    def test_end_run_no_current(self, experiment: Experiment) -> None:
        # Should not raise
        experiment.end_run()

    def test_start_run_replaces_running(self, experiment: Experiment) -> None:
        run1 = experiment.start_run()
        run2 = experiment.start_run()
        assert run1.status == "completed"  # auto-completed
        assert experiment.current_run is run2
        experiment.end_run()


class TestExperimentGetBestRun:
    """Tests for get_best_run."""

    def test_get_best_run_minimize(self, experiment: Experiment) -> None:
        run1 = experiment.start_run()
        experiment.log_metric("loss", 0.5)
        experiment.end_run()

        run2 = experiment.start_run()
        experiment.log_metric("loss", 0.1)
        experiment.end_run()

        best = experiment.get_best_run("loss", minimize=True)
        assert best is not None
        assert best.final_metrics["loss_final"] == 0.1

    def test_get_best_run_maximize(self, experiment: Experiment) -> None:
        experiment.start_run()
        experiment.log_metric("acc", 0.8)
        experiment.end_run()

        experiment.start_run()
        experiment.log_metric("acc", 0.95)
        experiment.end_run()

        best = experiment.get_best_run("acc", minimize=False)
        assert best is not None
        assert best.final_metrics["acc_final"] == 0.95

    def test_get_best_run_no_completed(self, experiment: Experiment) -> None:
        assert experiment.get_best_run("loss") is None

    def test_get_best_run_metric_not_found(self, experiment: Experiment) -> None:
        experiment.start_run()
        experiment.log_metric("x", 1.0)
        experiment.end_run()
        # "missing" metric doesn't exist; should still return a run (inf)
        best = experiment.get_best_run("missing", minimize=True)
        assert best is not None


class TestExperimentSaveLoad:
    """Tests for save and load."""

    def test_save_and_load(self, experiment: Experiment, tmp_dir: Path) -> None:
        experiment.start_run(hyperparams={"lr": 0.01})
        experiment.log_metric("loss", 0.5)
        experiment.end_run()

        path = experiment.save()
        assert path.exists()

        loaded = Experiment.load(path)
        assert loaded.config.name == "test_exp"
        assert len(loaded.runs) == 1
        assert loaded.runs[0].hyperparams["lr"] == 0.01

    def test_save_custom_path(self, experiment: Experiment, tmp_dir: Path) -> None:
        custom = tmp_dir / "custom" / "exp.json"
        path = experiment.save(path=custom)
        assert path == custom
        assert custom.exists()


class TestExperimentLogArtifact:
    """Tests for log_artifact."""

    def test_log_artifact(self, experiment: Experiment) -> None:
        experiment.start_run()
        experiment.log_artifact("model", "/tmp/model.pt")
        assert experiment.current_run is not None
        assert experiment.current_run.artifacts["model"] == "/tmp/model.pt"
        experiment.end_run()

    def test_log_artifact_no_run(self, experiment: Experiment) -> None:
        experiment.log_artifact("model", "/tmp/x.pt")  # no-op


class TestExperimentTracker:
    """Tests for ExperimentTracker."""

    def test_create_experiment(self, tmp_dir: Path) -> None:
        tracker = ExperimentTracker(base_dir=tmp_dir)
        exp = tracker.create_experiment("test", description="d")
        assert isinstance(exp, Experiment)
        assert "test" in tracker.list_experiments()

    def test_get_experiment(self, tmp_dir: Path) -> None:
        tracker = ExperimentTracker(base_dir=tmp_dir)
        tracker.create_experiment("exp1", description="d")
        assert tracker.get_experiment("exp1") is not None
        assert tracker.get_experiment("missing") is None

    def test_load_experiments(self, tmp_dir: Path) -> None:
        # Create experiment with output_dir matching base_dir
        cfg = ExperimentConfig(name="e1", description="d", output_dir=str(tmp_dir))
        exp = Experiment(config=cfg)
        exp.start_run()
        exp.end_run()
        exp.save()

        # Load from same base_dir
        tracker = ExperimentTracker(base_dir=tmp_dir)
        count = tracker.load_experiments()
        assert count == 1
        assert "e1" in tracker.list_experiments()

    def test_load_experiments_empty_dir(self, tmp_dir: Path) -> None:
        tracker = ExperimentTracker(base_dir=tmp_dir / "empty")
        assert tracker.load_experiments() == 0

    def test_save_all(self, tmp_dir: Path) -> None:
        tracker = ExperimentTracker(base_dir=tmp_dir)
        tracker.create_experiment("a", description="d")
        tracker.create_experiment("b", description="d")
        count = tracker.save_all()
        assert count == 2

    def test_get_best_across_experiments(self, tmp_dir: Path) -> None:
        tracker = ExperimentTracker(base_dir=tmp_dir)

        exp1 = tracker.create_experiment("e1", description="d")
        exp1.start_run()
        exp1.log_metric("loss", 0.5)
        exp1.end_run()

        exp2 = tracker.create_experiment("e2", description="d")
        exp2.start_run()
        exp2.log_metric("loss", 0.1)
        exp2.end_run()

        result = tracker.get_best_across_experiments("loss", minimize=True)
        assert result is not None
        name, run = result
        assert name == "e2"

    def test_get_best_across_no_data(self, tmp_dir: Path) -> None:
        tracker = ExperimentTracker(base_dir=tmp_dir)
        assert tracker.get_best_across_experiments("loss") is None

    def test_get_summary(self, experiment: Experiment) -> None:
        experiment.start_run()
        experiment.end_run()
        summary = experiment.get_summary()
        assert summary["total_runs"] == 1
        assert summary["completed_runs"] == 1
        assert summary["is_running"] is False

    def test_experiments_property(self, tmp_dir: Path) -> None:
        tracker = ExperimentTracker(base_dir=tmp_dir)
        tracker.create_experiment("x", description="d")
        assert "x" in tracker.experiments
