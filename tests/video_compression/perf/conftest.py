"""Shared fixtures for perf-benchmark tests."""

from __future__ import annotations

import pytest

from src.video_compression.config import (
    CodecConfig,
    DecoderConfig,
    EncoderConfig,
    EntropyConfig,
    MCTSRateControlConfig,
    QuantizerConfig,
    TrainingConfig,
)
from src.video_compression.perf import (
    PerfBenchmarkConfig,
    ResolutionSpec,
    RuntimeProfile,
)
from src.video_compression.perf.config import BenchmarkPhase


@pytest.fixture
def tiny_codec_config() -> CodecConfig:
    """Small codec config that exercises every component without burning CPU.

    All channels/dimensions are deliberately tiny; this is for harness
    correctness tests, not realistic codec measurements.
    """
    return CodecConfig(
        name="tiny",
        encoder=EncoderConfig(
            name="enc",
            in_channels=3,
            latent_channels=32,
            n_layers=1,
            d_model=64,
            n_heads=2,
            d_ffn=128,
            downsample_factor=4,
        ),
        decoder=DecoderConfig(
            name="dec",
            latent_channels=32,
            out_channels=3,
            n_layers=1,
            d_model=64,
            n_heads=2,
            d_ffn=128,
            upsample_factor=4,
        ),
        quantizer=QuantizerConfig(name="q"),
        entropy=EntropyConfig(
            name="ent",
            hyper_channels=32,
            num_filters=32,
        ),
        mcts=MCTSRateControlConfig(name="mcts", state_dim=64),
        training=TrainingConfig(name="train"),
    )


@pytest.fixture
def tiny_perf_config(tmp_path) -> PerfBenchmarkConfig:
    """Minimal config that the benchmark can run end-to-end on CPU."""
    return PerfBenchmarkConfig(
        name="tiny_perf",
        resolutions=[
            ResolutionSpec(name="r16", label="16x16", height=16, width=16),
        ],
        batch_sizes=[1],
        runtime_profiles=[RuntimeProfile(name="default")],
        phases=[BenchmarkPhase.FORWARD],
        n_warmup=0,
        n_repeats=2,
        n_frames_per_iter=1,
        device_preference="cpu",
        track_gpu_memory=False,
        pattern="motion",
        data_seed=12345,
        regression_tolerance_pct=50.0,
    )
