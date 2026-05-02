"""Tests for the Phase 2-C dataset spec and registry."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import Tensor

from src.video_compression.zoo import dataset_spec as dataset_spec_module
from src.video_compression.zoo.dataset_spec import (
    DatasetSpec,
    build_loader,
    get_dataset_factory,
    list_dataset_kinds,
    register_dataset_factory,
)


def _save_dummy_image(path: Path, color: int) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    Image.new("RGB", (32, 32), color=(color, color, color)).save(path)


def test_synthetic_spec_validates_with_defaults() -> None:
    spec = DatasetSpec(name="synth", kind="synthetic")
    assert spec.kind == "synthetic"
    assert spec.root is None
    assert spec.height >= 16


def test_image_folder_spec_requires_root() -> None:
    with pytest.raises(ValueError, match="root"):
        DatasetSpec(name="img", kind="image_folder")


def test_image_folder_spec_accepts_root() -> None:
    spec = DatasetSpec(name="img", kind="image_folder", root="/tmp/foo")
    assert spec.root == "/tmp/foo"


def test_image_folder_spec_rejects_blank_root() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        DatasetSpec(name="img", kind="image_folder", root="   ")


def test_synthetic_factory_yields_repeating_batch() -> None:
    spec = DatasetSpec(name="s", kind="synthetic", height=16, width=16)
    loader = build_loader(spec, batch_size=2, split="train")

    a = next(loader)
    b = next(loader)

    assert isinstance(a, Tensor)
    assert a.shape == (2, 3, 16, 16)
    assert torch.equal(a, b)


def test_synthetic_factory_split_seeds_differ() -> None:
    spec = DatasetSpec(name="s", kind="synthetic", height=16, width=16, seed=7)
    train_batch = next(build_loader(spec, batch_size=2, split="train"))
    val_batch = next(build_loader(spec, batch_size=2, split="val"))
    # The split seed offset is non-zero, so the two splits must produce
    # distinct samples for any non-degenerate generator pattern.
    assert not torch.equal(train_batch, val_batch)


def test_image_folder_factory_loads_images(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    for i in range(4):
        _save_dummy_image(tmp_path / f"img_{i}.png", color=20 * i)

    spec = DatasetSpec(
        name="img",
        kind="image_folder",
        root=str(tmp_path),
        height=32,
        width=32,
        num_workers=0,
    )
    loader = build_loader(spec, batch_size=2, split="train")
    batch = next(loader)
    assert batch.shape == (2, 3, 32, 32)


def test_image_folder_factory_missing_root_raises() -> None:
    spec = DatasetSpec(
        name="img",
        kind="image_folder",
        root="/nonexistent/path/abc123",
        height=32,
        width=32,
    )
    with pytest.raises(FileNotFoundError):
        next(build_loader(spec, batch_size=1, split="train"))


def test_registry_lookup_unknown_kind() -> None:
    with pytest.raises(KeyError):
        get_dataset_factory("not_a_real_kind")


def test_registry_rejects_duplicate_registration() -> None:
    def _dummy(
        spec: DatasetSpec, batch_size: int, split: str
    ) -> Iterator[Tensor]:
        del spec, batch_size, split
        yield torch.zeros(1)

    with pytest.raises(ValueError, match="already registered"):
        register_dataset_factory("synthetic", _dummy)


def test_list_dataset_kinds_includes_defaults() -> None:
    kinds = list_dataset_kinds()
    assert "synthetic" in kinds
    assert "image_folder" in kinds


def test_register_unregister_roundtrip() -> None:
    kind_name = "custom_test_only"

    def _factory(
        spec: DatasetSpec, batch_size: int, split: str
    ) -> Iterator[Tensor]:
        del spec, split
        while True:
            yield torch.zeros((batch_size, 1, 4, 4))

    register_dataset_factory(kind_name, _factory)
    try:
        assert kind_name in list_dataset_kinds()
        assert get_dataset_factory(kind_name) is _factory
    finally:
        # Manual cleanup so the test does not pollute global state for
        # other tests in the suite.
        dataset_spec_module._REGISTRY.pop(kind_name, None)
