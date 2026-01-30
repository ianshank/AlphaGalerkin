"""Tests for video compression configuration schemas."""

import pytest
from pydantic import ValidationError

from src.video_compression.config import (
    EncoderConfig,
    DecoderConfig,
    QuantizerConfig,
    EntropyConfig,
    MCTSRateControlConfig,
    TrainingConfig,
    CodecConfig,
    QuantizationMode,
    EntropyModelType,
    RateControlMode,
)


class TestEncoderConfig:
    """Tests for EncoderConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EncoderConfig(name="test")

        assert config.in_channels == 3
        assert config.latent_channels == 192
        assert config.n_layers == 4
        assert config.d_model == 256
        assert config.n_heads == 8
        assert config.downsample_factor == 16

    def test_valid_downsample_factor(self) -> None:
        """Test valid downsample factors (powers of 2)."""
        for ds in [4, 8, 16, 32, 64]:
            config = EncoderConfig(name="test", downsample_factor=ds)
            assert config.downsample_factor == ds

    def test_invalid_downsample_factor(self) -> None:
        """Test invalid downsample factor raises error."""
        with pytest.raises(ValidationError):
            EncoderConfig(name="test", downsample_factor=12)

    def test_channel_constraints(self) -> None:
        """Test channel constraints."""
        with pytest.raises(ValidationError):
            EncoderConfig(name="test", in_channels=0)

        with pytest.raises(ValidationError):
            EncoderConfig(name="test", latent_channels=20)  # Below min


class TestDecoderConfig:
    """Tests for DecoderConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DecoderConfig(name="test")

        assert config.latent_channels == 192
        assert config.out_channels == 3
        assert config.upsample_factor == 16

    def test_matches_encoder_defaults(self) -> None:
        """Test decoder defaults match encoder for symmetry."""
        encoder = EncoderConfig(name="enc")
        decoder = DecoderConfig(name="dec")

        assert encoder.latent_channels == decoder.latent_channels
        assert encoder.in_channels == decoder.out_channels
        assert encoder.d_model == decoder.d_model


class TestQuantizerConfig:
    """Tests for QuantizerConfig."""

    def test_default_mode(self) -> None:
        """Test default quantization mode."""
        config = QuantizerConfig(name="test")
        assert config.mode == QuantizationMode.NOISE

    def test_all_modes_valid(self) -> None:
        """Test all quantization modes can be configured."""
        for mode in QuantizationMode:
            config = QuantizerConfig(name="test", mode=mode)
            assert config.mode == mode

    def test_temperature_constraints(self) -> None:
        """Test temperature constraints."""
        with pytest.raises(ValidationError):
            QuantizerConfig(name="test", temperature=-1.0)


class TestEntropyConfig:
    """Tests for EntropyConfig."""

    def test_default_model_type(self) -> None:
        """Test default entropy model type."""
        config = EntropyConfig(name="test")
        assert config.model_type == EntropyModelType.HYPERPRIOR

    def test_all_model_types_valid(self) -> None:
        """Test all entropy model types can be configured."""
        for model_type in EntropyModelType:
            config = EntropyConfig(name="test", model_type=model_type)
            assert config.model_type == model_type


class TestMCTSRateControlConfig:
    """Tests for MCTSRateControlConfig."""

    def test_default_values(self) -> None:
        """Test default MCTS configuration."""
        config = MCTSRateControlConfig(name="test")

        assert config.num_simulations == 50
        assert config.c_puct == 1.25
        assert config.gop_size == 16

    def test_qp_range_validation(self) -> None:
        """Test QP range validation."""
        # Valid range
        config = MCTSRateControlConfig(name="test", qp_min=10, qp_max=40)
        assert config.qp_min == 10
        assert config.qp_max == 40

        # Invalid range (min > max)
        with pytest.raises(ValidationError):
            MCTSRateControlConfig(name="test", qp_min=40, qp_max=10)

    def test_rate_control_modes(self) -> None:
        """Test all rate control modes."""
        for mode in RateControlMode:
            config = MCTSRateControlConfig(name="test", rate_control_mode=mode)
            assert config.rate_control_mode == mode


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_lambda_values(self) -> None:
        """Test default lambda values for R-D training."""
        config = TrainingConfig(name="test")

        assert len(config.lambda_values) == 8
        assert config.lambda_values[0] < config.lambda_values[-1]

    def test_resolution_range_validation(self) -> None:
        """Test resolution range validation."""
        # Valid range
        config = TrainingConfig(
            name="test",
            min_resolution=128,
            max_resolution=512,
        )
        assert config.min_resolution == 128

        # Invalid range
        with pytest.raises(ValidationError):
            TrainingConfig(
                name="test",
                min_resolution=512,
                max_resolution=128,
            )


class TestCodecConfig:
    """Tests for complete CodecConfig."""

    def test_default_construction(self) -> None:
        """Test default codec configuration."""
        config = CodecConfig(name="test")

        assert config.encoder.name == "encoder"
        assert config.decoder.name == "decoder"
        assert config.quantizer.name == "quantizer"

    def test_channel_consistency_validation(self) -> None:
        """Test encoder/decoder channel consistency."""
        # Valid: matching channels
        config = CodecConfig(
            name="test",
            encoder=EncoderConfig(name="enc", latent_channels=192),
            decoder=DecoderConfig(name="dec", latent_channels=192),
        )
        assert config.encoder.latent_channels == config.decoder.latent_channels

        # Invalid: mismatched latent channels
        with pytest.raises(ValidationError):
            CodecConfig(
                name="test",
                encoder=EncoderConfig(name="enc", latent_channels=192),
                decoder=DecoderConfig(name="dec", latent_channels=256),
            )

    def test_hash_reproducibility(self) -> None:
        """Test configuration hashing is reproducible."""
        config1 = CodecConfig(name="test", seed=42)
        config2 = CodecConfig(name="test", seed=42)

        assert config1.compute_hash() == config2.compute_hash()

    def test_hash_changes_with_params(self) -> None:
        """Test hash changes when parameters change."""
        config1 = CodecConfig(name="test", seed=42)
        config2 = CodecConfig(name="test", seed=43)

        assert config1.compute_hash() != config2.compute_hash()
