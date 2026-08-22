"""Security tests for the video-compression checkpoint loaders.

Companion to :mod:`tests.security.test_checkpoint_safety`, which covers
``src/training/checkpoint.py``. This file covers the *second* instance of the
same arbitrary-code-execution shape, found in the codec loaders.

The defect, for anyone re-reading this after it is long fixed:

``load_codec`` deserialized with a bare ``torch.load``. From torch 2.6 that
means ``weights_only=True``, whose unpickler rejects any global it was not told
about -- and ``CodecConfig.model_dump()``, which every codec checkpoint embeds,
carries three ``Enum`` members and a ``datetime``. So ``load_codec`` failed on
*valid* input. ``scripts/decode_video.py`` caught that failure and retried with
``weights_only=False``, meaning the unsafe path was the **normal** path for real
checkpoints, and a malicious file was executed precisely because it had failed
the safe check.

Both halves are asserted here: the safe path must *work* (otherwise callers are
pushed back onto an unsafe one), and the unsafe path must not run on rejection.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any

import pytest
import torch

from src.training.checkpoint import SAFE_CHECKPOINT_GLOBALS, load_torch_checkpoint
from src.video_compression import config as vc_config
from src.video_compression.config import SAFE_CODEC_GLOBALS, CodecConfig

pytestmark = pytest.mark.security


class CodecMarkerPickle:
    """Payload that leaves filesystem evidence when unpickled.

    Mirrors ``tests.security.test_checkpoint_safety.MarkerPickle``: the point of
    a marker rather than a no-op payload is that "the load raised" and "the load
    did not execute anything" are different claims, and only the second one is
    the security property. A fallback that unpickles after the safe load
    rejected the file raises nothing at all.
    """

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, ...]:
        return (Path.write_text, (self.marker, "PAYLOAD_EXECUTED"))


@pytest.fixture
def genuine_codec_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint shaped the way ``load_codec`` expects to read one.

    The ``config`` entry is a ``CodecConfig`` dump, because that is what
    ``load_codec`` reconstructs (``CodecConfig(**checkpoint["config"])``), and
    its enum members are what make this non-trivial to load safely.

    Corrected after review: this said "field-for-field from
    ``VideoCompressionTrainer.save_checkpoint``". It is not -- that trainer
    writes a ``TrainingConfig`` dump under different keys, and its round trip is
    covered separately in
    ``tests/security/test_checkpoint_roundtrip.py::TestVideoCompressionTrainerRoundTrip``.
    """
    path = tmp_path / "genuine.pt"
    torch.save(
        {
            "step": 7,
            "epoch": 2,
            "best_rd_loss": 0.25,
            "model_state": {"w": torch.zeros(2)},
            "optimizer_state": {"state": {}, "param_groups": [{"lr": 1e-4}]},
            "scheduler_state": {"last_epoch": 0},
            "config": CodecConfig(name="genuine").model_dump(),
        },
        path,
    )
    return path


@pytest.fixture
def malicious_codec_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """A codec-shaped checkpoint carrying an executable payload.

    Returns:
        ``(checkpoint_path, marker_path)``. The marker does not exist yet; if it
        exists after a load, the payload executed.

    """
    marker = tmp_path / "PAYLOAD_EXECUTED"
    path = tmp_path / "malicious.pt"
    torch.save(
        {
            "model_state_dict": {"w": torch.zeros(2)},
            "config": CodecConfig(name="evil").model_dump(),
            "payload": CodecMarkerPickle(marker),
        },
        path,
    )
    return path, marker


class TestSafeCodecGlobals:
    """``SAFE_CODEC_GLOBALS`` must stay complete and stay pure data."""

    def test_lists_every_enum_the_config_module_defines(self) -> None:
        """Reflect over the module rather than restating the list.

        A new enum added to ``CodecConfig`` without being added here would
        reintroduce exactly the bare-load failure this tuple exists to prevent,
        and would do it silently -- the symptom is a loader that fails on valid
        input, which reads like a corrupt checkpoint rather than a code defect.
        """
        defined = {
            obj
            for name in dir(vc_config)
            if isinstance(obj := getattr(vc_config, name), type)
            and issubclass(obj, enum.Enum)
            and obj.__module__ == vc_config.__name__
        }
        assert defined == set(SAFE_CODEC_GLOBALS), (
            "SAFE_CODEC_GLOBALS is out of sync with the enums in "
            f"{vc_config.__name__}: missing={defined - set(SAFE_CODEC_GLOBALS)}, "
            f"stale={set(SAFE_CODEC_GLOBALS) - defined}"
        )

    def test_covers_every_enum_actually_reachable_in_a_checkpoint(self) -> None:
        """Walk the real payload, not just the module namespace.

        The reflection test above filters on ``obj.__module__``, so an enum that
        reaches ``CodecConfig`` from *another* module would escape it — narrower
        than the promise made where ``SAFE_CODEC_GLOBALS`` is defined. This walks
        the nested ``model_dump()`` a checkpoint actually stores, so the
        assertion matches the claim regardless of where a type is declared.

        (All three are same-module today, so this is currently redundant with the
        reflection test. It is here for the case that stops being true.)
        """
        found: set[type] = set()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
            elif isinstance(node, enum.Enum):
                found.add(type(node))

        walk(CodecConfig(name="reachability").model_dump())
        assert found, "no enum found in the dump — the walk is not exercising anything"
        missing = found - set(SAFE_CODEC_GLOBALS)
        assert not missing, f"enums reachable in a checkpoint but not allowlisted: {missing}"

    def test_every_entry_is_a_pure_data_enum(self) -> None:
        """The allowlist rule is 'pure data constructors only'.

        An ``Enum`` deserializes by looking up a member and can invoke nothing
        else. Anything with a custom ``__reduce__`` would not qualify, and must
        go through ``allow_unsafe_pickle`` instead so it carries a warning and
        an audit trail.
        """
        for entry in SAFE_CODEC_GLOBALS:
            assert issubclass(entry, enum.Enum), f"{entry!r} is not an Enum"
            assert "__reduce__" not in vars(entry), f"{entry!r} defines a custom __reduce__"

    def test_does_not_overlap_the_base_allowlist(self) -> None:
        """Extras widen the window; they should not silently duplicate it."""
        assert not set(SAFE_CODEC_GLOBALS) & set(SAFE_CHECKPOINT_GLOBALS)


class TestCodecCheckpointLoadsSafely:
    """The safe path must *work* -- a broken safe path is how this defect began."""

    def test_genuine_checkpoint_loads_with_extras(self, genuine_codec_checkpoint: Path) -> None:
        loaded = load_torch_checkpoint(
            genuine_codec_checkpoint,
            extra_safe_globals=SAFE_CODEC_GLOBALS,
        )
        assert loaded["step"] == 7
        assert CodecConfig(**loaded["config"]).name == "genuine"

    def test_genuine_checkpoint_is_rejected_without_extras(
        self,
        genuine_codec_checkpoint: Path,
    ) -> None:
        """Pins *why* the extras are needed, so the fix cannot be quietly undone.

        If this ever starts passing, either the config stopped embedding enums or
        the default allowlist was widened -- both of which mean the
        ``extra_safe_globals`` plumbing should be re-examined rather than left in
        place as cargo cult.
        """
        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_torch_checkpoint(genuine_codec_checkpoint)

    def test_load_codec_succeeds_on_a_genuine_checkpoint(
        self,
        genuine_codec_checkpoint: Path,
    ) -> None:
        """The regression that made the unsafe fallback routine.

        Before the fix this raised ``UnpicklingError`` on a checkpoint the codec
        trainer had just written, which is what pushed ``decode_video.py`` onto
        ``weights_only=False`` as its ordinary path.
        """
        from src.video_compression.codec.codec import load_codec

        codec = load_codec(genuine_codec_checkpoint, device="cpu")
        assert codec is not None


class TestCodecCheckpointRejectsPayloads:
    """The unsafe path must never run as a consequence of the safe path failing."""

    def test_malicious_payload_does_not_execute(
        self,
        malicious_codec_checkpoint: tuple[Path, Path],
    ) -> None:
        path, marker = malicious_codec_checkpoint

        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_torch_checkpoint(path, extra_safe_globals=SAFE_CODEC_GLOBALS)

        assert not marker.exists(), "payload executed despite the safe load rejecting it"

    def test_load_codec_does_not_execute_a_payload(
        self,
        malicious_codec_checkpoint: tuple[Path, Path],
    ) -> None:
        """Same assertion through the entry point ``decode_video.py`` calls."""
        from src.video_compression.codec.codec import load_codec

        path, marker = malicious_codec_checkpoint

        with pytest.raises(RuntimeError):
            load_codec(path, device="cpu")

        assert not marker.exists(), "payload executed inside load_codec"

    def test_extras_do_not_admit_arbitrary_globals(
        self,
        malicious_codec_checkpoint: tuple[Path, Path],
    ) -> None:
        """Widening the window for enums must not widen it for anything else.

        The payload's blocked global is ``Path.write_text``, which is not in
        either allowlist -- so admitting the codec enums must leave it rejected.
        """
        path, marker = malicious_codec_checkpoint

        with pytest.raises(RuntimeError):
            load_torch_checkpoint(
                path,
                extra_safe_globals=(*SAFE_CODEC_GLOBALS, *SAFE_CHECKPOINT_GLOBALS),
            )

        assert not marker.exists()

    def test_unsafe_pickle_remains_an_explicit_opt_in(
        self,
        malicious_codec_checkpoint: tuple[Path, Path],
    ) -> None:
        """The hatch still exists and still requires being asked for by name.

        Asserting the payload *does* run here is deliberate: it proves the
        preceding tests are measuring the guard rather than a checkpoint that
        happened to be inert.
        """
        path, marker = malicious_codec_checkpoint

        load_torch_checkpoint(path, allow_unsafe_pickle=True)

        assert marker.exists(), "the opt-in path did not deserialize the payload"
