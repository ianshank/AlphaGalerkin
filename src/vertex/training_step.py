"""Reusable, typed training-step primitives for :class:`VertexTrainer`.

``VertexTrainer._train_step`` historically returned a hardcoded ``{"loss": 0.0}``
no-op when no custom ``train_step_fn`` was injected, so the cloud training loop did
no learning by default. This module provides the real, testable building blocks the
trainer delegates to:

* :class:`VertexHyperparams` — a typed/validated hyperparameter config (no hardcoded
  values; maps the canonical ``TrainingConfig`` keys ``learning_rate`` /
  ``weight_decay`` / ``gradient_clip``).
* :func:`make_optimizer` — AdamW from a model + hyperparams.
* :func:`default_compute_loss` — MSE over a ``(input, target)`` tuple or
  ``{"input","target"}`` dict, with device handling.
* :func:`run_training_step` — one forward/backward/optimizer step.
* :class:`BatchSource` — draws batches indefinitely by re-iterating its source
  (no ``itertools.cycle``: that caches every batch, leaking memory and freezing
  DataLoader shuffling).

This logic lives in its own module (rather than inside the CI-omitted legacy
``trainer.py``) so it can be unit-tested and coverage-gated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import torch
from pydantic import Field
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from src.templates.config import BaseModuleConfig

# A batch is either a (input, target) pair or a mapping with those keys.
Batch = Any
ComputeLoss = Callable[[nn.Module, Any], Tensor]


class VertexHyperparams(BaseModuleConfig):
    """Typed training hyperparameters for the Vertex default training step."""

    name: str = Field(default="vertex_hyperparams", min_length=1)
    learning_rate: float = Field(
        default=2e-4,
        gt=0.0,
        le=1.0,
        description="AdamW initial learning rate.",
    )
    weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        le=1.0,
        description="AdamW weight decay (L2 regularization).",
    )
    gradient_clip: float = Field(
        default=1.0,
        ge=0.0,
        description="Max gradient norm; 0 disables clipping.",
    )

    @classmethod
    def from_training_dict(cls, training: Mapping[str, Any] | None) -> VertexHyperparams:
        """Build from a ``config['training']`` dict, reading canonical keys only.

        Uses the same field names as ``config/schemas.py::TrainingConfig``
        (``learning_rate`` / ``weight_decay`` / ``gradient_clip``); unknown keys
        (e.g. ``total_steps``) are ignored. Missing keys fall back to the typed
        defaults above.
        """
        training = training or {}
        known = ("learning_rate", "weight_decay", "gradient_clip")
        kwargs = {k: training[k] for k in known if k in training}
        return cls(**kwargs)


def make_optimizer(model: nn.Module, hp: VertexHyperparams) -> AdamW:
    """Construct an AdamW optimizer from a model and hyperparameters."""
    return AdamW(
        model.parameters(),
        lr=hp.learning_rate,
        weight_decay=hp.weight_decay,
    )


def _unpack_batch(batch: Batch) -> tuple[Tensor, Tensor]:
    """Extract ``(input, target)`` tensors from a tuple or mapping batch."""
    if isinstance(batch, Mapping):
        try:
            return batch["input"], batch["target"]
        except KeyError as exc:
            raise KeyError(
                f"dict batch must contain 'input' and 'target' keys; got {sorted(batch.keys())}"
            ) from exc
    inputs, target = batch  # (input, target) pair
    return inputs, target


def default_compute_loss(model: nn.Module, batch: Batch) -> Tensor:
    """Default loss: MSE between ``model(input)`` and ``target``.

    Moves the batch tensors to the model's device and forwards through *model*
    (not ``model.module``) so DDP gradient synchronization is preserved.
    """
    inputs, target = _unpack_batch(batch)
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    target = target.to(device)
    prediction = model(inputs)
    return torch.nn.functional.mse_loss(prediction, target)


def run_training_step(
    model: nn.Module,
    optimizer: Optimizer,
    batch: Batch,
    *,
    compute_loss: ComputeLoss = default_compute_loss,
    gradient_clip: float = 0.0,
    scheduler: Any | None = None,
) -> dict[str, float]:
    """Execute a single forward/backward/optimizer step.

    Args:
        model: Model to train (put into ``train()`` mode here).
        optimizer: Optimizer holding the model's parameters.
        batch: A ``(input, target)`` pair or ``{"input","target"}`` mapping.
        compute_loss: ``(model, batch) -> loss`` callable.
        gradient_clip: Max gradient norm; ``0`` (or negative) disables clipping.
        scheduler: Optional **step-wise** LR scheduler (e.g. OneCycle/warmup-cosine).
            ReduceLROnPlateau-style (metric-arg / epoch-wise) schedulers are not
            supported and must not be passed here.

    Returns:
        ``{"loss": float, "lr": float}`` for the step.

    """
    model.train()
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()  # type: ignore[no-untyped-call]
    if gradient_clip and gradient_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    lr = float(optimizer.param_groups[0]["lr"])
    return {"loss": float(loss.detach().item()), "lr": lr}


class BatchSource:
    """Draws batches indefinitely from an iterable by re-iterating it.

    Unlike :func:`itertools.cycle`, this does not cache yielded batches — it calls
    ``iter(source)`` again at the end of each pass, so a ``DataLoader`` re-shuffles
    each epoch and large tensors are not pinned in memory.
    """

    def __init__(self, source: Iterable[Any]) -> None:
        self._source = source
        self._iter: Iterator[Any] = iter(source)

    def next(self) -> Any:
        """Return the next batch, re-iterating the source when exhausted.

        Raises:
            ValueError: If the source is empty (yields nothing on a fresh pass).

        """
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._source)
            try:
                return next(self._iter)
            except StopIteration as exc:
                raise ValueError("data_source is empty; cannot draw a training batch") from exc
