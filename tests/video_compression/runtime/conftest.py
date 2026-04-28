"""Shared fixtures for the runtime test suite.

Builds on Phase 0's tiny codec config (which respected the codec's
field-validator floors) so test cycles stay fast on CPU.
"""

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
from src.video_compression.runtime import DecoderRuntimeContext


@pytest.fixture
def tiny_codec_config() -> CodecConfig:
    """Smallest codec config that satisfies field validators.

    Mirrors the Phase 0 fixture: latent_channels >= 32, d_model >= 64,
    d_ffn >= 128. ``downsample_factor=4`` keeps test resolutions tiny.
    """
    return CodecConfig(
        name="runtime_tiny",
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
def tiny_context() -> DecoderRuntimeContext:
    """Per-cell context matching ``tiny_codec_config`` on CPU."""
    return DecoderRuntimeContext(
        name="tiny_ctx",
        batch_size=1,
        latent_channels=32,
        # 16x16 frame, downsample=4 -> 4x4 latent
        latent_height=4,
        latent_width=4,
        dtype="float32",
        device="cpu",
        model_hash="tinyhash",
    )
