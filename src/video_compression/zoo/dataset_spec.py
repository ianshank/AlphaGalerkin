"""Dataset specifications and factory registry for the video-codec model zoo.

Phase 2-C.4 — replaces the synthetic-only fallback in
:mod:`src.video_compression.training.zoo_trainer` with a typed,
extensible dataset spec.

A :class:`DatasetSpec` is a Pydantic config that fully describes how a
single zoo entry's training and evaluation batches are produced. The
spec is resolved into an iterator-of-tensors via the dataset
:func:`registry.get` lookup; new dataset kinds (``image_folder``,
``video_folder``, future ``webdataset``) plug in without touching
``ZooTrainer``.

No paths or numerical defaults live in code — every knob is a Pydantic
field with a validated default and a docstring describing its effect.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import structlog
import torch
from pydantic import ConfigDict, Field, field_validator, model_validator
from torch import Tensor
from torch.utils.data import DataLoader

from src.templates.config import BaseModuleConfig
from src.video_compression.data.dataset import DatasetConfig, ImageDataset
from src.video_compression.data.synthetic import (
    SyntheticPattern,
    SyntheticVideoConfig,
    SyntheticVideoGenerator,
)

logger = structlog.get_logger(__name__)

#: Allowed values for :attr:`DatasetSpec.kind`. Adding a new kind here
#: must be paired with a ``register_dataset_factory`` call below.
DatasetKind = Literal["synthetic", "image_folder"]

#: Split labels accepted by dataset factories. Factories may use the
#: split to seed differently or pick a different file subset.
DatasetSplit = Literal["train", "val"]


class DatasetSpec(BaseModuleConfig):
    """Declarative spec for a zoo entry's data pipeline.

    The spec is resolved to a :class:`collections.abc.Iterator` of
    ``(N, C, H, W)`` float tensors via the factory registry below.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: DatasetKind = Field(
        ...,
        description=(
            "Dataset family. ``synthetic`` uses the in-process "
            "SyntheticVideoGenerator (zero external deps, deterministic). "
            "``image_folder`` reads images from disk via ImageDataset."
        ),
    )

    #: Filesystem root for ``image_folder`` (and future folder-backed
    #: kinds). Required when ``kind`` references a folder; ignored for
    #: the synthetic kind.
    root: str | None = Field(
        default=None,
        description="Filesystem root for folder-backed datasets.",
    )

    # Frame geometry — used by synthetic generator and as the crop size
    # for folder-backed datasets so a zoo run produces shape-stable
    # batches across kinds.
    height: int = Field(
        default=64,
        ge=16,
        le=8192,
        description="Frame height in pixels.",
    )
    width: int = Field(
        default=64,
        ge=16,
        le=8192,
        description="Frame width in pixels.",
    )
    channels: int = Field(
        default=3,
        ge=1,
        le=4,
        description="Number of color channels.",
    )

    # Sampling and reproducibility
    num_workers: int = Field(
        default=0,
        ge=0,
        le=64,
        description="DataLoader worker count for folder-backed datasets.",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether folder-backed datasets shuffle each epoch.",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Deterministic seed for the data pipeline.",
    )

    # Synthetic-only knobs — nested rather than re-typed so the
    # downstream generator stays the single source of truth for
    # pattern semantics.
    synthetic_pattern: SyntheticPattern = Field(
        default=SyntheticPattern.GRADIENT,
        description="Synthetic pattern (only used when kind='synthetic').",
    )
    synthetic_num_frames: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Synthetic sequence length (only used when kind='synthetic').",
    )

    @field_validator("root")
    @classmethod
    def _normalize_root(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("root must be a non-empty string when set")
        return stripped

    @model_validator(mode="after")
    def _validate_kind_root(self) -> DatasetSpec:
        if self.kind == "image_folder":
            if self.root is None:
                raise ValueError(
                    "DatasetSpec(kind='image_folder') requires 'root' to be set",
                )
            # ImageDataset's DatasetConfig enforces patch_size >= 32, so
            # folder-backed kinds must declare resolutions large enough
            # to feed it. Synthetic kinds remain free to go smaller.
            if self.height < 32 or self.width < 32:
                raise ValueError(
                    "DatasetSpec(kind='image_folder') requires height>=32 "
                    f"and width>=32; got ({self.height}, {self.width})",
                )
        return self


#: Factory signature: ``(spec, batch_size, split) -> iterator of float tensors``.
DatasetFactory = Callable[[DatasetSpec, int, DatasetSplit], Iterator[Tensor]]


_REGISTRY: dict[str, DatasetFactory] = {}


def register_dataset_factory(kind: str, factory: DatasetFactory) -> None:
    """Register a dataset factory for a :attr:`DatasetSpec.kind` value.

    Re-registering an existing kind raises ``ValueError`` so collisions
    fail loud rather than silently overriding an active factory.
    """
    if kind in _REGISTRY:
        raise ValueError(
            f"dataset factory already registered for kind={kind!r}; "
            f"existing={_REGISTRY[kind]!r}",
        )
    _REGISTRY[kind] = factory


def get_dataset_factory(kind: str) -> DatasetFactory:
    """Return the factory registered for ``kind`` or raise ``KeyError``."""
    if kind not in _REGISTRY:
        raise KeyError(
            f"no dataset factory registered for kind={kind!r}; "
            f"known kinds={sorted(_REGISTRY)!r}",
        )
    return _REGISTRY[kind]


def list_dataset_kinds() -> list[str]:
    """Return the sorted list of registered dataset kinds."""
    return sorted(_REGISTRY)


def build_loader(
    spec: DatasetSpec,
    batch_size: int,
    split: DatasetSplit,
) -> Iterator[Tensor]:
    """Resolve a :class:`DatasetSpec` to a tensor iterator.

    Thin convenience wrapper over the registry for callers that only
    need the default dispatch.
    """
    factory = get_dataset_factory(spec.kind)
    return factory(spec, batch_size, split)


def _split_seed_offset(split: DatasetSplit) -> int:
    """Stable, non-zero offset between train and val seeds."""
    return 0 if split == "train" else 10_000


def _synthetic_factory(
    spec: DatasetSpec,
    batch_size: int,
    split: DatasetSplit,
) -> Iterator[Tensor]:
    """Synthetic factory: deterministic, repeats one batch forever."""
    config = SyntheticVideoConfig(
        pattern=spec.synthetic_pattern,
        num_frames=spec.synthetic_num_frames,
        height=spec.height,
        width=spec.width,
        channels=spec.channels,
        seed=spec.seed + _split_seed_offset(split),
    )
    generator = SyntheticVideoGenerator(config)
    sequence = generator.generate()
    if sequence.shape[0] >= batch_size:
        batch = sequence[:batch_size]
    else:
        repeats = (batch_size + sequence.shape[0] - 1) // sequence.shape[0]
        batch = sequence.repeat((repeats, 1, 1, 1))[:batch_size]

    def _iter() -> Iterator[Tensor]:
        while True:
            yield batch.clone()

    return _iter()


def _image_folder_factory(
    spec: DatasetSpec,
    batch_size: int,
    split: DatasetSplit,
) -> Iterator[Tensor]:
    """Image-folder factory backed by :class:`ImageDataset`.

    Cycles the underlying loader so the iterator never raises
    ``StopIteration``; ``ZooTrainer`` consumes a fixed step count.
    """
    if spec.root is None:  # pragma: no cover - validator guarantees this
        raise ValueError("DatasetSpec.root must be set for kind='image_folder'")
    root = Path(spec.root)
    if not root.exists():
        raise FileNotFoundError(
            f"DatasetSpec(kind='image_folder', root={spec.root!r}) "
            f"does not exist",
        )

    patch_size = min(spec.height, spec.width)
    dataset_config = DatasetConfig(
        root_dir=str(root),
        patch_size=patch_size,
        random_crop=True,
        random_flip=True,
        num_workers=spec.num_workers,
    )
    dataset = ImageDataset(root=root, config=dataset_config)

    generator = torch.Generator()
    generator.manual_seed(spec.seed + _split_seed_offset(split))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=spec.shuffle,
        num_workers=spec.num_workers,
        drop_last=True,
        generator=generator,
    )

    def _iter() -> Iterator[Tensor]:
        # ImageDataset returns (C, H, W) per item; default collate
        # stacks to (N, C, H, W). Loop forever because ZooTrainer
        # consumes a fixed step count, not epochs.
        while True:
            yield from loader

    return _iter()


# Default kinds are registered at import time so the registry is
# usable without explicit bootstrapping by the caller.
register_dataset_factory("synthetic", _synthetic_factory)
register_dataset_factory("image_folder", _image_folder_factory)


__all__ = [
    "DatasetFactory",
    "DatasetKind",
    "DatasetSpec",
    "DatasetSplit",
    "build_loader",
    "get_dataset_factory",
    "list_dataset_kinds",
    "register_dataset_factory",
]
