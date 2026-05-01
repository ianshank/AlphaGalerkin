from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from src.video_compression.config import CodecConfig
from src.video_compression.training import ZooTrainer, build_training_config
from src.video_compression.training.trainer import TrainingMetrics
from src.video_compression.training.zoo_trainer import (
    make_default_loader_factory,
    resolve_dataset_spec,
)
from src.video_compression.zoo.config import (
    ModelZooEntryConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from src.video_compression.zoo.dataset_spec import DatasetSpec
from src.video_compression.zoo.storage import VideoCodecZoo


def _entry(**overrides: object) -> ModelZooEntryConfig:
    base: dict[str, object] = {
        "entry_id": "lambda_0.0016",
        "lambda_rd": 0.0016,
        "target_bpp": 0.25,
        "target_psnr_db": 35.0,
        "train_steps": 4,
        "batch_size": 2,
        "optimizer": OptimizerConfig(name="optimizer"),
        "scheduler": SchedulerConfig(name="scheduler", warmup_steps=1, min_lr_ratio=0.05),
    }
    base.update(overrides)
    return ModelZooEntryConfig(**base)


class _FakeCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


class _FakeTrainer:
    def __init__(self, codec: nn.Module, config, output_dir: Path) -> None:  # noqa: ANN001
        self.codec = codec
        self.config = config
        self.output_dir = output_dir
        self.state = type(
            "State",
            (),
            {"step": 0, "epoch": 0, "best_rd_loss": float("inf")},
        )()

    def train(
        self,
        train_loader: Iterator[Tensor],
        val_loader: Iterator[Tensor] | None = None,
    ) -> None:
        next(train_loader)
        if val_loader is not None:
            next(val_loader)
        self.state.step = self.config.total_steps
        self.state.best_rd_loss = 0.2

    def eval_step(self, batch: Tensor) -> TrainingMetrics:
        return TrainingMetrics(
            loss=0.2,
            rate=0.24,
            distortion=0.01,
            psnr=35.2,
            ms_ssim=0.98,
            lr=1.0e-4,
        )


def _loader_factory(entry: ModelZooEntryConfig, split: str) -> Iterator[Tensor]:
    del entry, split
    while True:
        yield torch.zeros((2, 3, 16, 16), dtype=torch.float32)


def _codec_factory(config: CodecConfig, device: str) -> nn.Module:
    del config, device
    return _FakeCodec()


def test_build_training_config_preserves_entry_settings() -> None:
    entry = _entry(device="cuda:1", use_amp=True)

    config = build_training_config(entry, device="cuda:1")

    assert config.device == "cuda:1"
    assert config.lambda_values == [entry.lambda_rd]
    assert config.learning_rate == entry.optimizer.learning_rate
    assert config.weight_decay == entry.optimizer.weight_decay
    assert config.gradient_clip == entry.grad_clip_norm
    assert config.total_steps == entry.train_steps
    assert config.min_lr_ratio == entry.scheduler.min_lr_ratio
    assert config.use_amp is True


def test_build_training_config_rejects_unsupported_optimizer() -> None:
    entry = _entry(optimizer=OptimizerConfig(name="optimizer", optimizer_type="sgd"))

    with pytest.raises(ValueError, match="adamw"):
        build_training_config(entry, device="cpu")


def test_build_training_config_disables_amp_on_cpu() -> None:
    entry = _entry(use_amp=True)

    config = build_training_config(entry, device="cpu")

    assert config.use_amp is False


def test_zoo_trainer_run_persists_report(tmp_path: Path) -> None:
    entry = _entry(parent_entry_id=None)
    zoo = VideoCodecZoo(tmp_path / "zoo")
    trainer = ZooTrainer(
        entry,
        zoo,
        codec_config=CodecConfig(name="codec"),
        device="cpu",
        output_root=tmp_path / "outputs",
        codec_factory=_codec_factory,
        loader_factory=_loader_factory,
        trainer_factory=_FakeTrainer,
        max_eval_batches=2,
    )

    report = trainer.run()

    assert report.entry_id == entry.entry_id
    assert report.device == "cpu"
    assert report.tolerance_passed is True
    assert report.checkpoint_path.exists()
    metrics = zoo.load_metrics(entry.entry_id)
    assert metrics["rate_bpp"] == pytest.approx(0.24)
    assert metrics["psnr_db"] == pytest.approx(35.2)
    bundle = zoo.load_state_dict(entry.entry_id)
    assert "model_state" in bundle


def test_resolve_dataset_spec_prefers_entry() -> None:
    entry_spec = DatasetSpec(name="entry_spec", kind="synthetic", seed=7)
    default_spec = DatasetSpec(name="default_spec", kind="synthetic", seed=99)
    entry = _entry(dataset_spec=entry_spec)

    resolved = resolve_dataset_spec(entry, default_dataset_spec=default_spec)

    assert resolved is entry_spec


def test_resolve_dataset_spec_falls_back_to_default() -> None:
    default_spec = DatasetSpec(name="default_spec", kind="synthetic", seed=99)
    entry = _entry()

    resolved = resolve_dataset_spec(entry, default_dataset_spec=default_spec)

    assert resolved is default_spec


def test_resolve_dataset_spec_synthetic_fallback() -> None:
    entry = _entry(seed=12345)

    resolved = resolve_dataset_spec(entry)

    assert resolved.kind == "synthetic"
    assert resolved.seed == 12345


def test_default_loader_factory_dispatches_via_registry() -> None:
    entry = _entry(
        dataset_spec=DatasetSpec(
            name="ds",
            kind="synthetic",
            height=16,
            width=16,
            seed=3,
        ),
        batch_size=2,
    )
    factory = make_default_loader_factory()
    loader = factory(entry, "train")

    batch = next(loader)

    assert batch.shape == (2, 3, 16, 16)


def test_default_loader_factory_rejects_invalid_split() -> None:
    factory = make_default_loader_factory()
    with pytest.raises(ValueError, match="split"):
        factory(_entry(), "test")
