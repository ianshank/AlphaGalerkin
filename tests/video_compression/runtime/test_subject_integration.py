"""Subject-integration tests for the new ``RuntimeBackedDecoderSubject``.

Verifies that the new decode subject plugs into the existing perf
benchmark without breaking the legacy FORWARD path.
"""

from __future__ import annotations

import pytest
import torch

from src.video_compression.data.synthetic import SyntheticPattern
from src.video_compression.perf.config import BenchmarkPhase
from src.video_compression.perf.subjects import (
    DEFAULT_DECODE_RUNTIME_NAME,
    BenchmarkSubject,
    CodecForwardSubject,
    RuntimeBackedDecoderSubject,
    create_subject,
)


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


# ----------------------------------------------------------- DEFAULT_NAME


def test_default_decode_runtime_name_is_eager() -> None:
    # Frozen as a stable contract; later iterations may choose to
    # change it but the default must remain *some* registered runtime.
    from src.video_compression.runtime import RuntimeRegistry

    assert RuntimeRegistry().is_registered(DEFAULT_DECODE_RUNTIME_NAME)


# ------------------------------------------------- create_subject dispatch


class TestCreateSubject:
    def test_forward_returns_codec_forward(self, tiny_codec_config, cpu_device) -> None:
        subj = create_subject(
            BenchmarkPhase.FORWARD,
            tiny_codec_config,
            device=cpu_device,
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        assert isinstance(subj, CodecForwardSubject)
        assert isinstance(subj, BenchmarkSubject)

    def test_decode_returns_runtime_backed(self, tiny_codec_config, cpu_device) -> None:
        subj = create_subject(
            BenchmarkPhase.DECODE,
            tiny_codec_config,
            device=cpu_device,
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        assert isinstance(subj, RuntimeBackedDecoderSubject)
        assert isinstance(subj, BenchmarkSubject)

    def test_decode_uses_default_runtime(self, tiny_codec_config, cpu_device) -> None:
        subj = create_subject(
            BenchmarkPhase.DECODE,
            tiny_codec_config,
            device=cpu_device,
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        # name encodes the runtime so reports stay distinguishable
        assert DEFAULT_DECODE_RUNTIME_NAME in subj.name

    def test_decode_with_explicit_runtime_name(self, tiny_codec_config, cpu_device) -> None:
        subj = create_subject(
            BenchmarkPhase.DECODE,
            tiny_codec_config,
            device=cpu_device,
            pattern=SyntheticPattern.MOTION,
            seed=1,
            runtime_name="pytorch-eager",
        )
        assert "pytorch-eager" in subj.name

    def test_encode_still_unimplemented(self, tiny_codec_config, cpu_device) -> None:
        with pytest.raises(NotImplementedError, match="encode"):
            create_subject(
                BenchmarkPhase.ENCODE,
                tiny_codec_config,
                device=cpu_device,
                pattern=SyntheticPattern.MOTION,
                seed=1,
            )


# ----------------------------------------------- RuntimeBackedDecoderSubject


class TestRuntimeBackedDecoderSubjectLifecycle:
    def test_step_before_prepare_raises(self, tiny_codec_config, cpu_device) -> None:
        subj = RuntimeBackedDecoderSubject(
            tiny_codec_config,
            device=cpu_device,
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        with pytest.raises(RuntimeError, match="before prepare"):
            subj.step()

    def test_resolution_must_match_downsample(self, tiny_codec_config, cpu_device) -> None:
        subj = RuntimeBackedDecoderSubject(
            tiny_codec_config,
            device=cpu_device,
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        # downsample=4 in tiny config; 17 must fail
        with pytest.raises(ValueError, match="downsample_factor"):
            subj.prepare(batch_size=1, height=17, width=17)

    def test_full_prepare_step_teardown_cycle(self, tiny_codec_config, cpu_device) -> None:
        subj = RuntimeBackedDecoderSubject(
            tiny_codec_config,
            device=cpu_device,
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        subj.prepare(batch_size=1, height=16, width=16)
        subj.step()
        subj.teardown()
        # After teardown, step() must raise again
        with pytest.raises(RuntimeError, match="before prepare"):
            subj.step()

    def test_can_re_prepare_for_new_cell(self, tiny_codec_config, cpu_device) -> None:
        subj = RuntimeBackedDecoderSubject(
            tiny_codec_config,
            device=cpu_device,
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        subj.prepare(batch_size=1, height=16, width=16)
        subj.step()
        subj.teardown()
        subj.prepare(batch_size=1, height=32, width=32)
        subj.step()
        subj.teardown()

    def test_unknown_runtime_name_raises_at_prepare(self, tiny_codec_config, cpu_device) -> None:
        subj = RuntimeBackedDecoderSubject(
            tiny_codec_config,
            device=cpu_device,
            pattern=SyntheticPattern.MOTION,
            seed=1,
            runtime_name="non-existent-runtime",
        )
        with pytest.raises(KeyError, match="not registered"):
            subj.prepare(batch_size=1, height=16, width=16)
