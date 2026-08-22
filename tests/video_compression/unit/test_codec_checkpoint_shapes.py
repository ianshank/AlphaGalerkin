"""`load_codec` must read every checkpoint shape this repo writes.

Three defects lived in one function, and each was invisible to the existing
suite because every fixture in it wrote the *one* shape no in-repo trainer
produces (``"model_state_dict"`` plus a whole ``CodecConfig`` dump):

1. ``CodecConfig(**checkpoint["config"])`` raised ``ValidationError`` (22
   errors) on a trainer-written checkpoint, whose ``"config"`` is a
   ``TrainingConfig`` dump -- a *sub*-config of ``CodecConfig``.
2. Weights were read only from ``"model_state_dict"``. Both real writers
   (``VideoCompressionTrainer``, ``ZooTrainer``) use ``"model_state"``, so the
   bare-payload fallback ran and ``strict=False`` loaded *nothing* -- returning
   an untrained codec with no error at all.
3. ``use_mcts`` was probed as ``"rate_controller" in
   checkpoint["model_state_dict"]``. That can never be True: state-dict keys are
   dotted paths, ``MCTSRateController`` is not an ``nn.Module`` so it contributes
   no entries at all, and real writers use ``"model_state"`` anyway. It is now an
   explicit parameter defaulting to the ``False`` that line always produced.

Between them, `load_codec` could not read any checkpoint the repo writes, which
is what made `scripts/decode_video.py`'s reduced fallback the ordinary path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from src.video_compression.codec.codec import (
    CODEC_STATE_DICT_KEYS,
    codec_config_from_checkpoint,
    codec_state_dict_from_checkpoint,
    create_codec,
    load_codec,
)
from src.video_compression.config import CodecConfig, TrainingConfig


@pytest.fixture(scope="module")
def reference_state_dict() -> dict[str, Any]:
    """The weights every shape below carries, so a load can be checked for real."""
    return create_codec(CodecConfig(name="reference"), device="cpu").state_dict()


def _save(payload: dict[str, Any], path: Path) -> Path:
    torch.save(payload, path)
    return path


def _assert_all_weights_loaded(path: Path, reference: dict[str, Any]) -> None:
    """Every reference tensor must come back bit-identical.

    Asserting the *count* matters: ``load_state_dict(..., strict=False)`` returns
    happily after matching zero keys, so `load_codec` "succeeding" proves nothing
    on its own. That is exactly how defect 2 stayed invisible.
    """
    loaded = load_codec(path, device="cpu").state_dict()
    matched = sum(1 for k, v in reference.items() if k in loaded and torch.equal(loaded[k], v))
    assert matched == len(reference), f"only {matched}/{len(reference)} tensors loaded"


class TestEveryWriterShapeLoads:
    """One test per shape a writer in this repo actually produces."""

    def test_trainer_written_checkpoint(
        self, tmp_path: Path, reference_state_dict: dict[str, Any]
    ) -> None:
        """Field-for-field ``VideoCompressionTrainer._save_checkpoint``.

        Before the fix this raised ``ValidationError`` -- so *no* checkpoint from
        the repo's own codec trainer could be loaded by its own codec loader.
        """
        path = _save(
            {
                "step": 3,
                "epoch": 1,
                "best_rd_loss": 0.5,
                "model_state": reference_state_dict,
                "optimizer_state": {},
                "scheduler_state": {},
                "config": TrainingConfig(name="run").model_dump(),
            },
            tmp_path / "trainer.pt",
        )
        _assert_all_weights_loaded(path, reference_state_dict)

    def test_zoo_trainer_written_checkpoint(
        self, tmp_path: Path, reference_state_dict: dict[str, Any]
    ) -> None:
        """``ZooTrainer`` writes ``"model_state"`` with no ``"config"`` at all."""
        path = _save(
            {"step": 1, "lambda_rd": 0.01, "model_state": reference_state_dict},
            tmp_path / "zoo.pt",
        )
        _assert_all_weights_loaded(path, reference_state_dict)

    def test_fixture_shaped_checkpoint(
        self, tmp_path: Path, reference_state_dict: dict[str, Any]
    ) -> None:
        """The shape the pre-existing suite already covered -- must not regress."""
        path = _save(
            {
                "step": 7,
                "model_state_dict": reference_state_dict,
                "config": CodecConfig(name="genuine").model_dump(),
            },
            tmp_path / "fixture.pt",
        )
        _assert_all_weights_loaded(path, reference_state_dict)

    def test_bare_state_dict(self, tmp_path: Path, reference_state_dict: dict[str, Any]) -> None:
        """``torch.save(codec.state_dict(), ...)`` with no envelope."""
        _assert_all_weights_loaded(
            _save(reference_state_dict, tmp_path / "bare.pt"), reference_state_dict
        )


class TestConfigReconstruction:
    """``codec_config_from_checkpoint`` -- one branch per payload shape."""

    def test_whole_codec_config_round_trips(self) -> None:
        cfg = CodecConfig(name="whole")
        assert codec_config_from_checkpoint({"config": cfg.model_dump()}).name == "whole"

    def test_training_config_is_nested_not_discarded(self) -> None:
        """A ``TrainingConfig`` dump carries real values; keep them.

        Falling straight through to a default ``CodecConfig`` would load the
        weights but silently lose every training hyperparameter the checkpoint
        recorded, so the nesting is the point -- not just "does not raise".
        """
        training = TrainingConfig(name="run", batch_size=7, learning_rate=1.5e-4)
        rebuilt = codec_config_from_checkpoint({"config": training.model_dump()})
        assert rebuilt.name == "run"
        assert rebuilt.training.batch_size == 7
        assert rebuilt.training.learning_rate == pytest.approx(1.5e-4)

    def test_unrecognised_payload_degrades_instead_of_raising(self) -> None:
        """Config metadata we cannot parse must not reject loadable weights."""
        assert codec_config_from_checkpoint({"config": {"nonsense": 1}}).name == "loaded"

    @pytest.mark.parametrize("payload", [{}, {"config": None}, {"config": "not-a-dict"}])
    def test_missing_or_non_dict_config(self, payload: dict[str, Any]) -> None:
        assert codec_config_from_checkpoint(payload).name == "loaded"


class TestStateDictResolution:
    """``codec_state_dict_from_checkpoint`` -- key precedence and fallback."""

    def test_precedence_order_is_documented_and_honoured(self) -> None:
        """``model_state_dict`` wins, so the pre-existing shape is unaffected."""
        assert CODEC_STATE_DICT_KEYS == ("model_state_dict", "model_state")
        both = {"model_state_dict": {"a": 1}, "model_state": {"b": 2}}
        assert codec_state_dict_from_checkpoint(both) == {"a": 1}

    def test_falls_back_to_model_state(self) -> None:
        assert codec_state_dict_from_checkpoint({"model_state": {"b": 2}}) == {"b": 2}

    def test_falls_back_to_the_whole_payload(self) -> None:
        bare = {"conv.weight": torch.zeros(1)}
        assert codec_state_dict_from_checkpoint(bare) is bare

    def test_non_dict_under_a_known_key_is_skipped(self) -> None:
        """A ``None`` placeholder must not shadow a real state dict below it."""
        assert codec_state_dict_from_checkpoint(
            {"model_state_dict": None, "model_state": {"b": 2}}
        ) == {"b": 2}


class TestMCTSRateControlIsExplicit:
    """Defect 3: the removed probe was not mis-keyed, it was impossible."""

    def test_a_checkpoint_carries_no_rate_controller_signal(self) -> None:
        """Pins *why* the parameter is explicit, so it cannot be "fixed" back.

        If this ever fails, ``MCTSRateController`` has become an ``nn.Module``
        and auto-detection is worth revisiting -- which is the only condition
        under which reintroducing a state-dict probe would be correct.
        """
        mcts_codec = create_codec(CodecConfig(name="m"), use_mcts_rate_control=True, device="cpu")
        assert mcts_codec.rate_controller is not None, "precondition: MCTS is on"
        assert not isinstance(mcts_codec.rate_controller, torch.nn.Module)
        assert not [k for k in mcts_codec.state_dict() if "rate_controller" in k]

    def test_default_preserves_the_historical_behaviour(self, tmp_path: Path) -> None:
        """Every existing caller got ``use_mcts=False``; that must not change."""
        state = create_codec(CodecConfig(name="m"), device="cpu").state_dict()
        path = _save({"model_state": state}, tmp_path / "default.pt")
        assert load_codec(path, device="cpu").rate_controller is None

    def test_opting_in_attaches_the_controller(self, tmp_path: Path) -> None:
        """The branch the old probe made unreachable."""
        state = create_codec(CodecConfig(name="m"), device="cpu").state_dict()
        path = _save({"model_state": state}, tmp_path / "optin.pt")
        codec = load_codec(path, device="cpu", use_mcts_rate_control=True)
        assert codec.rate_controller is not None
