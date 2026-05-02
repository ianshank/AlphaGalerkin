"""Per-entry training adapter for the video-compression model zoo.

Phase 2-C keeps the existing :class:`VideoCompressionTrainer` intact and
wraps it with per-entry zoo semantics: one lambda point, optional
warm-start, structured logging, tolerance checks, and persistence through
``VideoCodecZoo``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TypeAlias, cast

from torch import Tensor, nn

from src.templates.logging import create_logger_class
from src.video_compression.codec.codec import VideoCodec
from src.video_compression.config import CodecConfig, TrainingConfig
from src.video_compression.training.trainer import TrainingMetrics, VideoCompressionTrainer
from src.video_compression.zoo.config import ModelZooEntryConfig
from src.video_compression.zoo.dataset_spec import (
    DatasetSpec,
    DatasetSplit,
    build_loader,
)
from src.video_compression.zoo.storage import VideoCodecZoo

_Logger = create_logger_class("video_compression.zoo_trainer")

CodecFactory: TypeAlias = Callable[[CodecConfig, str], nn.Module]
LoaderFactory: TypeAlias = Callable[[ModelZooEntryConfig, str], Iterator[Tensor]]
TrainerFactory: TypeAlias = Callable[
    [nn.Module, TrainingConfig, Path | str],
    VideoCompressionTrainer,
]
_DEFAULT_TRAINER_FACTORY: TrainerFactory = cast(TrainerFactory, VideoCompressionTrainer)


def resolve_dataset_spec(
    entry: ModelZooEntryConfig,
    *,
    default_dataset_spec: DatasetSpec | None = None,
) -> DatasetSpec:
    """Resolve the effective :class:`DatasetSpec` for a zoo entry.

    Precedence is ``entry.dataset_spec`` > ``default_dataset_spec`` >
    a synthetic fallback keyed off ``entry.seed``. The fallback exists
    so smoke runs and pre-Phase-2-C manifests keep working without any
    schema changes.
    """
    if entry.dataset_spec is not None:
        return entry.dataset_spec
    if default_dataset_spec is not None:
        return default_dataset_spec
    return DatasetSpec(
        name=f"dataset_spec_{entry.entry_id}",
        kind="synthetic",
        seed=entry.seed,
    )


@dataclass(frozen=True)
class ZooTrainingReport:
    """Outcome of training a single zoo entry."""

    entry_id: str
    lambda_rd: float
    target_bpp: float
    target_psnr_db: float
    realized_bpp: float
    realized_psnr_db: float
    realized_ms_ssim: float | None
    final_loss: float
    step_count: int
    device: str
    checkpoint_path: Path
    tolerance_passed: bool
    bpp_relative_error: float
    psnr_absolute_error_db: float
    train_wallclock_s: float
    eval_wallclock_s: float
    parent_entry_id: str | None = None


def build_training_config(
    entry: ModelZooEntryConfig,
    *,
    device: str,
) -> TrainingConfig:
    """Translate one zoo entry into the existing training config.

    The adapter is intentionally narrow: it only accepts optimizer and
    scheduler families that the current ``VideoCompressionTrainer``
    actually implements. Unsupported settings fail loud rather than being
    silently ignored.
    """
    if entry.optimizer.optimizer_type != "adamw":
        raise ValueError(
            "ZooTrainer currently supports optimizer_type='adamw' only; "
            f"got {entry.optimizer.optimizer_type!r}",
        )
    if entry.scheduler.scheduler_type not in {"cosine", "warmup_cosine"}:
        raise ValueError(
            "ZooTrainer currently supports scheduler_type 'cosine' and "
            f"'warmup_cosine' only; got {entry.scheduler.scheduler_type!r}",
        )

    use_amp = entry.use_amp and device.startswith("cuda")
    return TrainingConfig(
        name=f"zoo_training_{entry.entry_id}",
        seed=entry.seed,
        learning_rate=entry.optimizer.learning_rate,
        weight_decay=entry.optimizer.weight_decay,
        batch_size=entry.batch_size,
        gradient_clip=entry.grad_clip_norm,
        total_steps=entry.train_steps,
        warmup_steps=entry.scheduler.warmup_steps,
        device=device,
        use_amp=use_amp,
        lambda_rd=entry.lambda_rd,
        lambda_values=[entry.lambda_rd],
        distortion_metric=entry.distortion_metric,
        ms_ssim_weight=entry.ms_ssim_weight,
        min_lr_ratio=entry.scheduler.min_lr_ratio,
    )


def _default_codec_factory(config: CodecConfig, device: str) -> VideoCodec:
    return VideoCodec(config=config, device=device)


def make_default_loader_factory(
    *,
    default_dataset_spec: DatasetSpec | None = None,
) -> LoaderFactory:
    """Build a loader factory bound to a manifest-level default spec.

    The returned closure dispatches per-entry through
    :func:`resolve_dataset_spec` and then the dataset registry, so all
    spec kinds (``synthetic``, ``image_folder``, future kinds) flow
    through the same code path.
    """

    def _factory(entry: ModelZooEntryConfig, split: str) -> Iterator[Tensor]:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val'; got {split!r}")
        spec = resolve_dataset_spec(
            entry,
            default_dataset_spec=default_dataset_spec,
        )
        return build_loader(spec, entry.batch_size, cast(DatasetSplit, split))

    return _factory


#: Module-level default loader factory — retained as the import-stable
#: ``default_loader_factory`` symbol so existing test seams keep working.
default_loader_factory: LoaderFactory = make_default_loader_factory()


def _extract_model_state(bundle: dict[str, object]) -> dict[str, Tensor]:
    """Normalize a saved bundle to the raw model state dict."""
    model_state = bundle.get("model_state")
    if model_state is not None:
        if not isinstance(model_state, dict):
            raise TypeError(
                "checkpoint bundle key 'model_state' must be a dict; got "
                f"{type(model_state).__name__}",
            )
        return cast(dict[str, Tensor], model_state)
    return cast(dict[str, Tensor], bundle)


def _evaluate_loader(
    trainer: VideoCompressionTrainer,
    val_loader: Iterator[Tensor],
    *,
    max_batches: int,
) -> TrainingMetrics:
    totals = {
        "loss": 0.0,
        "rate": 0.0,
        "distortion": 0.0,
        "psnr": 0.0,
        "ms_ssim": 0.0,
    }
    n_ms_ssim = 0
    count = 0

    for batch in val_loader:
        if count >= max_batches:
            break
        metrics = trainer.eval_step(batch)
        totals["loss"] += metrics.loss
        totals["rate"] += metrics.rate
        totals["distortion"] += metrics.distortion
        totals["psnr"] += metrics.psnr
        if metrics.ms_ssim is not None:
            totals["ms_ssim"] += metrics.ms_ssim
            n_ms_ssim += 1
        count += 1

    if count == 0:
        raise ValueError("validation loader produced no batches")

    return TrainingMetrics(
        loss=totals["loss"] / count,
        rate=totals["rate"] / count,
        distortion=totals["distortion"] / count,
        psnr=totals["psnr"] / count,
        ms_ssim=(totals["ms_ssim"] / n_ms_ssim) if n_ms_ssim else None,
        lr=None,
    )


class ZooTrainer:
    """Train and publish a single zoo entry."""

    def __init__(
        self,
        entry: ModelZooEntryConfig,
        zoo: VideoCodecZoo,
        *,
        codec_config: CodecConfig,
        device: str | None = None,
        output_root: str | Path = "outputs/video_compression/zoo",
        codec_factory: CodecFactory = _default_codec_factory,
        loader_factory: LoaderFactory | None = None,
        default_dataset_spec: DatasetSpec | None = None,
        trainer_factory: TrainerFactory = _DEFAULT_TRAINER_FACTORY,
        max_eval_batches: int = 10,
    ) -> None:
        self.entry = entry
        self.zoo = zoo
        self.codec_config = codec_config
        self.device = device or entry.device or "auto"
        self.output_root = Path(output_root)
        self.codec_factory = codec_factory
        self.loader_factory = loader_factory or make_default_loader_factory(
            default_dataset_spec=default_dataset_spec,
        )
        self.default_dataset_spec = default_dataset_spec
        self.trainer_factory = trainer_factory
        self.max_eval_batches = max_eval_batches
        self.output_dir = self.output_root / entry.entry_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log = _Logger(
            "ZooTrainer",
            entry_id=entry.entry_id,
            lambda_rd=entry.lambda_rd,
            device=self.device,
        )

    def _warm_start_if_needed(self, codec: nn.Module) -> None:
        if self.entry.parent_entry_id is None:
            return
        bundle = self.zoo.load_state_dict(
            self.entry.parent_entry_id,
            map_location=self.device if self.device != "auto" else None,
            weights_only=True,
        )
        codec.load_state_dict(_extract_model_state(bundle), strict=False)
        self._log.info(
            "zoo_trainer.warm_start.loaded",
            parent_entry_id=self.entry.parent_entry_id,
        )

    def _build_state_bundle(
        self,
        trainer: VideoCompressionTrainer,
        training_config: TrainingConfig,
    ) -> dict[str, object]:
        return {
            "model_state": trainer.codec.state_dict(),
            "training_state": {
                "step": trainer.state.step,
                "epoch": trainer.state.epoch,
                "best_rd_loss": trainer.state.best_rd_loss,
            },
            # Keep the checkpoint bundle compatible with
            # ``weights_only=True`` torch.load by storing only tensors and
            # plain JSON-friendly primitives here; ``model_dump()`` would
            # include ``created_at`` as a datetime object.
            "training_config": training_config.to_yaml_dict(),
            "entry_id": self.entry.entry_id,
            "lambda_rd": self.entry.lambda_rd,
            "parent_entry_id": self.entry.parent_entry_id,
        }

    def run(self) -> ZooTrainingReport:
        training_config = build_training_config(self.entry, device=self.device)
        codec = self.codec_factory(self.codec_config, self.device)
        self._warm_start_if_needed(codec)

        trainer = self.trainer_factory(codec, training_config, self.output_dir)
        train_loader = self.loader_factory(self.entry, "train")
        val_loader = self.loader_factory(self.entry, "val")

        self._log.info(
            "zoo_trainer.run.started",
            output_dir=str(self.output_dir),
        )
        train_start = perf_counter()
        trainer.train(train_loader, val_loader)
        train_wallclock_s = perf_counter() - train_start

        eval_start = perf_counter()
        final_metrics = _evaluate_loader(
            trainer,
            self.loader_factory(self.entry, "val"),
            max_batches=self.max_eval_batches,
        )
        eval_wallclock_s = perf_counter() - eval_start

        bpp_relative_error = abs(final_metrics.rate - self.entry.target_bpp) / self.entry.target_bpp
        psnr_absolute_error_db = abs(final_metrics.psnr - self.entry.target_psnr_db)
        tolerance_passed = (
            bpp_relative_error <= self.entry.bpp_tolerance
            and psnr_absolute_error_db <= self.entry.psnr_tolerance_db
        )

        persisted_metrics: dict[str, float] = {
            "loss": float(final_metrics.loss),
            "rate_bpp": float(final_metrics.rate),
            "distortion": float(final_metrics.distortion),
            "psnr_db": float(final_metrics.psnr),
            "bpp_relative_error": float(bpp_relative_error),
            "psnr_absolute_error_db": float(psnr_absolute_error_db),
            "tolerance_passed": 1.0 if tolerance_passed else 0.0,
            "step_count": float(trainer.state.step),
            "lambda_rd": float(self.entry.lambda_rd),
            "train_wallclock_s": float(train_wallclock_s),
            "eval_wallclock_s": float(eval_wallclock_s),
        }
        if final_metrics.ms_ssim is not None:
            persisted_metrics["ms_ssim"] = float(final_metrics.ms_ssim)

        artifacts = self.zoo.save_entry(
            self.entry,
            self._build_state_bundle(trainer, training_config),
            metrics=persisted_metrics,
        )
        self._log.info(
            "zoo_trainer.run.completed",
            tolerance_passed=tolerance_passed,
            checkpoint_path=str(artifacts.checkpoint_path),
        )
        return ZooTrainingReport(
            entry_id=self.entry.entry_id,
            lambda_rd=self.entry.lambda_rd,
            target_bpp=self.entry.target_bpp,
            target_psnr_db=self.entry.target_psnr_db,
            realized_bpp=float(final_metrics.rate),
            realized_psnr_db=float(final_metrics.psnr),
            realized_ms_ssim=(
                float(final_metrics.ms_ssim) if final_metrics.ms_ssim is not None else None
            ),
            final_loss=float(final_metrics.loss),
            step_count=trainer.state.step,
            device=self.device,
            checkpoint_path=artifacts.checkpoint_path,
            tolerance_passed=tolerance_passed,
            bpp_relative_error=float(bpp_relative_error),
            psnr_absolute_error_db=float(psnr_absolute_error_db),
            train_wallclock_s=train_wallclock_s,
            eval_wallclock_s=eval_wallclock_s,
            parent_entry_id=self.entry.parent_entry_id,
        )
