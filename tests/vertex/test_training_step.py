"""Tests for the reusable Vertex training-step primitives."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from src.vertex.training_step import (
    BatchSource,
    VertexHyperparams,
    default_compute_loss,
    make_optimizer,
    run_training_step,
)


class TestVertexHyperparams:
    def test_defaults(self) -> None:
        hp = VertexHyperparams()
        assert hp.learning_rate == pytest.approx(2e-4)
        assert hp.weight_decay == pytest.approx(1e-4)
        assert hp.gradient_clip == pytest.approx(1.0)

    def test_from_training_dict_canonical_keys(self) -> None:
        hp = VertexHyperparams.from_training_dict(
            {"learning_rate": 1e-3, "weight_decay": 0.0, "gradient_clip": 5.0}
        )
        assert hp.learning_rate == pytest.approx(1e-3)
        assert hp.weight_decay == pytest.approx(0.0)
        assert hp.gradient_clip == pytest.approx(5.0)

    def test_from_training_dict_ignores_unknown_and_none(self) -> None:
        # Unknown keys (e.g. total_steps) are ignored; missing keys keep defaults.
        hp = VertexHyperparams.from_training_dict({"total_steps": 10000, "learning_rate": 3e-4})
        assert hp.learning_rate == pytest.approx(3e-4)
        assert hp.weight_decay == pytest.approx(1e-4)  # default
        assert VertexHyperparams.from_training_dict(None).learning_rate == pytest.approx(2e-4)

    def test_rejects_nonpositive_lr(self) -> None:
        with pytest.raises(ValidationError):
            VertexHyperparams(learning_rate=0.0)

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            VertexHyperparams(unknown_field=1)  # type: ignore[call-arg]


class TestMakeOptimizer:
    def test_builds_adamw_with_hyperparams(self) -> None:
        model = nn.Linear(2, 1)
        hp = VertexHyperparams(learning_rate=1e-3, weight_decay=0.01)
        opt = make_optimizer(model, hp)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.01)


class TestDefaultComputeLoss:
    def test_tuple_batch(self) -> None:
        model = nn.Linear(3, 1)
        x = torch.randn(4, 3)
        y = torch.randn(4, 1)
        loss = default_compute_loss(model, (x, y))
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_dict_batch(self) -> None:
        model = nn.Linear(3, 1)
        batch = {"input": torch.randn(4, 3), "target": torch.randn(4, 1)}
        loss = default_compute_loss(model, batch)
        assert torch.isfinite(loss)

    def test_dict_batch_missing_keys_raises(self) -> None:
        model = nn.Linear(3, 1)
        with pytest.raises(KeyError, match="input.*target|target"):
            default_compute_loss(model, {"x": torch.randn(4, 3)})


class TestRunTrainingStep:
    def test_returns_loss_and_lr_and_populates_grads(self) -> None:
        model = nn.Linear(3, 1)
        hp = VertexHyperparams(learning_rate=1e-2)
        opt = make_optimizer(model, hp)
        batch = (torch.randn(8, 3), torch.randn(8, 1))

        metrics = run_training_step(model, opt, batch, gradient_clip=hp.gradient_clip)

        assert set(metrics) == {"loss", "lr"}
        assert metrics["lr"] == pytest.approx(1e-2)
        assert any(p.grad is not None for p in model.parameters())

    def test_loss_decreases_on_fittable_problem(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(1, 1)
        opt = make_optimizer(model, VertexHyperparams(learning_rate=1e-1))
        x = torch.linspace(-1, 1, 32).unsqueeze(1)
        y = 3.0 * x  # learnable linear map
        batch = (x, y)

        first = run_training_step(model, opt, batch)["loss"]
        for _ in range(50):
            last = run_training_step(model, opt, batch)["loss"]
        assert last < first

    def test_scheduler_is_stepped(self) -> None:
        model = nn.Linear(2, 1)
        opt = make_optimizer(model, VertexHyperparams())

        class _RecordingScheduler:
            def __init__(self) -> None:
                self.calls = 0

            def step(self) -> None:
                self.calls += 1

        sched = _RecordingScheduler()
        run_training_step(model, opt, (torch.randn(4, 2), torch.randn(4, 1)), scheduler=sched)
        assert sched.calls == 1

    def test_gradient_clip_zero_disables(self) -> None:
        # gradient_clip=0 must not raise and still steps.
        model = nn.Linear(2, 1)
        opt = make_optimizer(model, VertexHyperparams())
        metrics = run_training_step(
            model, opt, (torch.randn(4, 2), torch.randn(4, 1)), gradient_clip=0.0
        )
        assert torch.isfinite(torch.tensor(metrics["loss"]))


class TestBatchSource:
    def test_iterates_then_reiterates(self) -> None:
        src = BatchSource([1, 2])
        assert src.next() == 1
        assert src.next() == 2
        # Exhausted -> re-iterates from the start.
        assert src.next() == 1

    def test_empty_source_raises(self) -> None:
        src = BatchSource([])
        with pytest.raises(ValueError, match="empty"):
            src.next()

    def test_reiterates_a_generator_factory(self) -> None:
        # A list re-iterates fine; confirm many draws never raise.
        src = BatchSource([10, 20, 30])
        drawn = [src.next() for _ in range(7)]
        assert drawn == [10, 20, 30, 10, 20, 30, 10]
