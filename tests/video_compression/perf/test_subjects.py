"""Tests for benchmark subjects."""

from __future__ import annotations

import pytest
import torch

from src.video_compression.data.synthetic import SyntheticPattern
from src.video_compression.perf.config import BenchmarkPhase
from src.video_compression.perf.subjects import (
    BenchmarkSubject,
    CodecForwardSubject,
    create_subject,
)


class TestCodecForwardSubject:
    def test_implements_protocol(self, tiny_codec_config) -> None:
        subj = CodecForwardSubject(
            tiny_codec_config,
            device=torch.device("cpu"),
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        assert isinstance(subj, BenchmarkSubject)

    def test_step_before_prepare_raises(self, tiny_codec_config) -> None:
        subj = CodecForwardSubject(
            tiny_codec_config,
            device=torch.device("cpu"),
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        with pytest.raises(RuntimeError, match="prepare"):
            subj.step()

    def test_prepare_step_teardown_cycle(self, tiny_codec_config) -> None:
        subj = CodecForwardSubject(
            tiny_codec_config,
            device=torch.device("cpu"),
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        subj.prepare(batch_size=1, height=16, width=16)
        subj.step()
        subj.teardown()
        # After teardown, step() should raise again
        with pytest.raises(RuntimeError, match="prepare"):
            subj.step()

    def test_resolution_must_match_downsample(self, tiny_codec_config) -> None:
        # downsample_factor=4 in tiny codec; 17x17 should fail
        subj = CodecForwardSubject(
            tiny_codec_config,
            device=torch.device("cpu"),
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        with pytest.raises(ValueError, match="downsample_factor"):
            subj.prepare(batch_size=1, height=17, width=17)

    def test_can_reprepare_for_new_cell(self, tiny_codec_config) -> None:
        subj = CodecForwardSubject(
            tiny_codec_config,
            device=torch.device("cpu"),
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        subj.prepare(batch_size=1, height=16, width=16)
        subj.step()
        subj.teardown()
        subj.prepare(batch_size=1, height=32, width=32)
        subj.step()
        subj.teardown()


class TestCreateSubjectFactory:
    def test_forward_phase_returns_codec_forward(self, tiny_codec_config) -> None:
        subj = create_subject(
            phase=BenchmarkPhase.FORWARD,
            codec_config=tiny_codec_config,
            device=torch.device("cpu"),
            pattern=SyntheticPattern.MOTION,
            seed=1,
        )
        assert isinstance(subj, CodecForwardSubject)

    @pytest.mark.parametrize("phase", [BenchmarkPhase.ENCODE, BenchmarkPhase.DECODE])
    def test_unimplemented_phases_raise(self, phase, tiny_codec_config) -> None:
        with pytest.raises(NotImplementedError, match="Phase"):
            create_subject(
                phase=phase,
                codec_config=tiny_codec_config,
                device=torch.device("cpu"),
                pattern=SyntheticPattern.MOTION,
                seed=1,
            )
