"""Pydantic schema tests for the model-zoo config layer."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.video_compression.zoo.config import (
    PERF_ZOO_MANIFEST_SCHEMA_VERSION,
    DeviceAssignmentStrategy,
    ModelZooEntryConfig,
    ModelZooManifestConfig,
    OptimizerConfig,
    SchedulerConfig,
)


def _entry(**overrides: object) -> ModelZooEntryConfig:
    base: dict[str, object] = {
        "entry_id": "e1",
        "lambda_rd": 0.01,
        "target_bpp": 0.5,
        "target_psnr_db": 33.0,
        "train_steps": 1000,
    }
    base.update(overrides)
    return ModelZooEntryConfig(**base)  # type: ignore[arg-type]


class TestOptimizerConfig:
    def test_default_betas_validate(self) -> None:
        cfg = OptimizerConfig(name="opt")
        assert cfg.betas == (0.9, 0.999)

    @pytest.mark.parametrize("bad", [(0.0, 0.999), (0.9, 1.0), (-0.1, 0.5), (0.9, 0.0)])
    def test_invalid_betas_rejected(self, bad: tuple[float, float]) -> None:
        with pytest.raises(ValidationError):
            OptimizerConfig(name="opt", betas=bad)

    def test_lr_bounds(self) -> None:
        with pytest.raises(ValidationError):
            OptimizerConfig(name="opt", learning_rate=0.0)
        with pytest.raises(ValidationError):
            OptimizerConfig(name="opt", learning_rate=2.0)


class TestSchedulerConfig:
    def test_defaults(self) -> None:
        cfg = SchedulerConfig(name="sched")
        assert cfg.scheduler_type == "warmup_cosine"
        assert cfg.warmup_steps == 500


class TestModelZooEntryConfig:
    def test_basic(self) -> None:
        entry = _entry()
        # name is auto-promoted from entry_id.
        assert entry.name == "e1"
        assert entry.lambda_rd == 0.01
        assert entry.use_amp is True

    def test_entry_id_pattern(self) -> None:
        with pytest.raises(ValidationError):
            _entry(entry_id="bad id!")

    @pytest.mark.parametrize("traversal_id", [".", ".."])
    def test_entry_id_path_traversal_rejected(self, traversal_id: str) -> None:
        # Bare ``.`` and ``..`` pass the regex but would resolve outside
        # ``storage_root`` when fed to ``VideoCodecZoo.entry_dir`` -
        # the post-validator must reject them explicitly.
        with pytest.raises(ValidationError, match="path traversal"):
            _entry(entry_id=traversal_id)

    def test_duplicate_entry_id_reports_all_duplicates(self) -> None:
        # Multiple duplicates must all be reported in one ValidationError
        # rather than fix-one-at-a-time.
        with pytest.raises(ValidationError) as excinfo:
            ModelZooManifestConfig(
                name="m",
                storage_root="./zoo",
                entries=[
                    _entry(entry_id="a"),
                    _entry(entry_id="a"),
                    _entry(entry_id="b"),
                    _entry(entry_id="b"),
                ],
            )
        msg = str(excinfo.value)
        assert "'a'" in msg
        assert "'b'" in msg

    def test_warmup_must_not_exceed_steps(self) -> None:
        with pytest.raises(ValidationError):
            _entry(
                train_steps=10,
                scheduler=SchedulerConfig(name="sched", warmup_steps=20),
            )

    def test_lambda_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _entry(lambda_rd=0.0)
        with pytest.raises(ValidationError):
            _entry(lambda_rd=11.0)

    def test_explicit_name_preserved(self) -> None:
        entry = _entry(name="custom_name")
        assert entry.name == "custom_name"

    def test_explicit_device(self) -> None:
        entry = _entry(device="cuda:1")
        assert entry.device == "cuda:1"

    def test_compute_hash_shape(self) -> None:
        # ``compute_hash`` returns the BaseModuleConfig 16-char hex
        # digest. Determinism across instances is bounded by
        # BaseModuleConfig (nested ``created_at`` timestamps differ);
        # we only assert the contract here.
        h = _entry().compute_hash()
        assert isinstance(h, str)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_estimated_vram_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _entry(estimated_vram_mib=0.0)


class TestModelZooManifestConfig:
    def test_basic(self) -> None:
        m = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[_entry(entry_id="a"), _entry(entry_id="b")],
        )
        assert m.schema_version == PERF_ZOO_MANIFEST_SCHEMA_VERSION
        assert m.device_assignment_strategy is DeviceAssignmentStrategy.VRAM_AWARE
        assert len(m.entries) == 2

    def test_duplicate_entry_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelZooManifestConfig(
                name="m",
                storage_root="./zoo",
                entries=[_entry(entry_id="x"), _entry(entry_id="x")],
            )

    def test_unknown_parent_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelZooManifestConfig(
                name="m",
                storage_root="./zoo",
                entries=[_entry(entry_id="child", parent_entry_id="missing")],
            )

    def test_known_parent_accepted(self) -> None:
        m = ModelZooManifestConfig(
            name="m",
            storage_root="./zoo",
            entries=[
                _entry(entry_id="parent"),
                _entry(entry_id="child", parent_entry_id="parent"),
            ],
        )
        assert m.entries[1].parent_entry_id == "parent"

    def test_extra_fields_ignored(self) -> None:
        # Forward-compat: unknown fields silently dropped.
        m = ModelZooManifestConfig.model_validate(
            {
                "name": "m",
                "storage_root": "./zoo",
                "entries": [_entry(entry_id="a").model_dump()],
                "future_field": 42,
            }
        )
        assert not hasattr(m, "future_field")

    def test_min_one_entry(self) -> None:
        with pytest.raises(ValidationError):
            ModelZooManifestConfig(name="m", storage_root="./zoo", entries=[])

    def test_parallel_workers_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ModelZooManifestConfig(
                name="m",
                storage_root="./zoo",
                entries=[_entry()],
                parallel_workers_per_device=0,
            )
