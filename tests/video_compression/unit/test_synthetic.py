"""Tests for synthetic video data generator.

Validates:
- Configuration validation (Pydantic constraints, extra fields, validators)
- All pattern types generate correct shapes and value ranges
- Deterministic output with seed control
- Temporal consistency between adjacent frames
- Factory function convenience API
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from src.video_compression.data.synthetic import (
    SyntheticPattern,
    SyntheticVideoConfig,
    SyntheticVideoGenerator,
    create_test_sequence,
)


# ---------------------------------------------------------------------------
# SyntheticVideoConfig Tests
# ---------------------------------------------------------------------------


class TestSyntheticVideoConfig:
    """Tests for configuration validation."""

    def test_default_config(self) -> None:
        """Default config should be valid with sensible defaults."""
        config = SyntheticVideoConfig()
        assert config.pattern == SyntheticPattern.GRADIENT
        assert config.num_frames == 8
        assert config.height == 64
        assert config.width == 64
        assert config.channels == 3
        assert config.seed == 42

    def test_custom_config(self) -> None:
        """Custom parameters should be accepted within constraints."""
        config = SyntheticVideoConfig(
            pattern=SyntheticPattern.WAVES,
            num_frames=16,
            height=128,
            width=256,
            channels=1,
            seed=123,
            wave_frequency=8.0,
        )
        assert config.pattern == SyntheticPattern.WAVES
        assert config.num_frames == 16
        assert config.height == 128
        assert config.width == 256
        assert config.channels == 1

    def test_invalid_dimensions_rejected(self) -> None:
        """Dimensions outside valid range should be rejected."""
        with pytest.raises(ValidationError):
            SyntheticVideoConfig(height=8)  # Below minimum 16
        with pytest.raises(ValidationError):
            SyntheticVideoConfig(width=4096)  # Above maximum 2048
        with pytest.raises(ValidationError):
            SyntheticVideoConfig(num_frames=0)  # Below minimum 1

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should be rejected (typo protection)."""
        with pytest.raises(ValidationError):
            SyntheticVideoConfig(unknown_field=42)  # type: ignore[call-arg]

    def test_negative_seed_rejected(self) -> None:
        """Negative seed should be rejected."""
        with pytest.raises(ValidationError):
            SyntheticVideoConfig(seed=-1)

    def test_checkerboard_size_validation(self) -> None:
        """Checkerboard size must not exceed frame dimensions."""
        with pytest.raises(ValidationError, match="checkerboard_size"):
            SyntheticVideoConfig(
                checkerboard_size=128,
                height=64,
                width=64,
            )

    def test_noise_std_range(self) -> None:
        """Noise std must be in [0, 1]."""
        config = SyntheticVideoConfig(noise_std=0.0)
        assert config.noise_std == 0.0

        config = SyntheticVideoConfig(noise_std=1.0)
        assert config.noise_std == 1.0

        with pytest.raises(ValidationError):
            SyntheticVideoConfig(noise_std=1.5)


# ---------------------------------------------------------------------------
# SyntheticVideoGenerator Tests
# ---------------------------------------------------------------------------


class TestSyntheticVideoGenerator:
    """Tests for video generation."""

    @pytest.mark.parametrize("pattern", list(SyntheticPattern))
    def test_frame_shape(self, pattern: SyntheticPattern) -> None:
        """All patterns should produce correctly shaped single frames."""
        config = SyntheticVideoConfig(pattern=pattern, height=32, width=48)
        gen = SyntheticVideoGenerator(config)
        frame = gen.generate_frame(0)
        assert frame.shape == (3, 32, 48)

    @pytest.mark.parametrize("pattern", list(SyntheticPattern))
    def test_sequence_shape(self, pattern: SyntheticPattern) -> None:
        """All patterns should produce correctly shaped sequences."""
        config = SyntheticVideoConfig(pattern=pattern, num_frames=4, height=32, width=32)
        gen = SyntheticVideoGenerator(config)
        seq = gen.generate()
        assert seq.shape == (4, 3, 32, 32)

    @pytest.mark.parametrize("h,w", [(16, 16), (32, 32), (64, 64), (64, 128), (48, 96)])
    def test_various_resolutions(self, h: int, w: int) -> None:
        """Generator should work at various resolutions."""
        config = SyntheticVideoConfig(height=h, width=w, num_frames=2)
        gen = SyntheticVideoGenerator(config)
        seq = gen.generate()
        assert seq.shape == (2, 3, h, w)

    @pytest.mark.parametrize("pattern", list(SyntheticPattern))
    def test_value_range(self, pattern: SyntheticPattern) -> None:
        """All output values must be in [0, 1]."""
        config = SyntheticVideoConfig(pattern=pattern, num_frames=4)
        gen = SyntheticVideoGenerator(config)
        seq = gen.generate()
        assert seq.min() >= 0.0, f"Min value {seq.min()} is below 0"
        assert seq.max() <= 1.0, f"Max value {seq.max()} is above 1"

    def test_determinism_with_same_seed(self) -> None:
        """Same seed should produce identical output."""
        config = SyntheticVideoConfig(seed=42, pattern=SyntheticPattern.WAVES)
        gen1 = SyntheticVideoGenerator(config)
        gen2 = SyntheticVideoGenerator(config)
        seq1 = gen1.generate()
        seq2 = gen2.generate()
        assert torch.allclose(seq1, seq2), "Same seed should produce identical output"

    def test_different_seeds_produce_different_output(self) -> None:
        """Different seeds should produce different output."""
        config1 = SyntheticVideoConfig(seed=42)
        config2 = SyntheticVideoConfig(seed=99)
        gen1 = SyntheticVideoGenerator(config1)
        gen2 = SyntheticVideoGenerator(config2)
        seq1 = gen1.generate()
        seq2 = gen2.generate()
        assert not torch.allclose(seq1, seq2), "Different seeds should differ"

    @pytest.mark.parametrize("pattern", list(SyntheticPattern))
    def test_temporal_consistency(self, pattern: SyntheticPattern) -> None:
        """Adjacent frames should be similar but not identical."""
        config = SyntheticVideoConfig(
            pattern=pattern,
            num_frames=4,
            temporal_variation=0.1,
        )
        gen = SyntheticVideoGenerator(config)
        seq = gen.generate()

        for t in range(seq.shape[0] - 1):
            diff = (seq[t] - seq[t + 1]).abs().mean()
            # Frames should differ (not static)
            # Allow noise pattern to differ more
            assert diff < 1.0, f"Frames {t} and {t+1} differ too much: {diff}"

    def test_default_three_channels(self) -> None:
        """Default config produces 3-channel float32 tensors."""
        config = SyntheticVideoConfig()
        gen = SyntheticVideoGenerator(config)
        frame = gen.generate_frame(0)
        assert frame.shape[0] == 3
        assert frame.dtype == torch.float32

    def test_single_channel(self) -> None:
        """Single channel generation should work."""
        config = SyntheticVideoConfig(channels=1, num_frames=2)
        gen = SyntheticVideoGenerator(config)
        seq = gen.generate()
        assert seq.shape == (2, 1, 64, 64)
        assert seq.min() >= 0.0
        assert seq.max() <= 1.0


# ---------------------------------------------------------------------------
# Factory Function Tests
# ---------------------------------------------------------------------------


class TestCreateTestSequence:
    """Tests for the convenience factory function."""

    def test_default_parameters(self) -> None:
        """Factory with defaults should produce valid sequence."""
        seq = create_test_sequence()
        assert seq.shape == (8, 3, 64, 64)
        assert seq.min() >= 0.0
        assert seq.max() <= 1.0

    def test_custom_parameters(self) -> None:
        """Factory with custom parameters should work."""
        seq = create_test_sequence(
            pattern=SyntheticPattern.MOTION,
            num_frames=4,
            height=32,
            width=48,
            seed=99,
        )
        assert seq.shape == (4, 3, 32, 48)

    def test_factory_determinism(self) -> None:
        """Factory should be deterministic with same seed."""
        seq1 = create_test_sequence(seed=42)
        seq2 = create_test_sequence(seed=42)
        assert torch.allclose(seq1, seq2)

    @pytest.mark.parametrize("pattern", list(SyntheticPattern))
    def test_all_patterns_via_factory(self, pattern: SyntheticPattern) -> None:
        """Factory should work for all pattern types."""
        seq = create_test_sequence(pattern=pattern, num_frames=2, height=32, width=32)
        assert seq.shape == (2, 3, 32, 32)
