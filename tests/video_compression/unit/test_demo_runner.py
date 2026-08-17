"""Tests for compression demo runner.

Validates:
- DemoConfig construction and validation
- Codec creation from demo config
- Single-lambda encoding/decoding
- R-D curve generation
- Resolution independence testing
- DemoResult serialization
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.video_compression.data.synthetic import SyntheticPattern, create_test_sequence
from src.video_compression.demo.config import DemoConfig
from src.video_compression.demo.runner import (
    CompressionDemoRunner,
    DemoResult,
    FrameResult,
    LambdaResult,
    ResolutionResult,
)

# ---------------------------------------------------------------------------
# Small config fixture for fast tests
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config(tmp_path: Path) -> DemoConfig:
    """Create minimal config for fast test execution."""
    return DemoConfig(
        num_frames=4,
        height=32,
        width=32,
        patterns=[SyntheticPattern.GRADIENT],
        latent_channels=32,
        d_model=64,
        n_heads=2,
        d_ffn=128,
        n_layers=1,
        downsample_factor=4,
        lambda_values=[0.01],
        resolution_sizes=[(32, 32)],
        resolution_lambda=0.01,
        device="cpu",
        seed=42,
        output_dir=str(tmp_path / "demo_output"),
        write_bitstream=True,
    )


@pytest.fixture
def runner(small_config: DemoConfig) -> CompressionDemoRunner:
    """Create demo runner with small config."""
    return CompressionDemoRunner(small_config)


# ---------------------------------------------------------------------------
# DemoConfig Tests
# ---------------------------------------------------------------------------


class TestDemoConfig:
    """Tests for demo configuration validation."""

    def test_default_config(self) -> None:
        """Default config should be valid."""
        config = DemoConfig()
        assert config.num_frames == 8
        assert config.height == 64
        assert config.width == 64
        assert len(config.patterns) == 2
        assert len(config.lambda_values) == 4

    def test_custom_config(self, tmp_path: Path) -> None:
        """Custom config should be accepted."""
        config = DemoConfig(
            num_frames=4,
            height=32,
            width=32,
            patterns=[SyntheticPattern.WAVES],
            latent_channels=64,
            d_model=128,
            n_heads=4,
            d_ffn=256,
            n_layers=2,
            downsample_factor=8,
            lambda_values=[0.01, 0.05],
            resolution_sizes=[(32, 32), (64, 64)],
            device="cpu",
            output_dir=str(tmp_path),
        )
        assert config.num_frames == 4
        assert len(config.lambda_values) == 2

    def test_invalid_downsample_factor(self) -> None:
        """Non-power-of-2 downsample factor should be rejected."""
        with pytest.raises(ValidationError, match="power of 2"):
            DemoConfig(downsample_factor=6)

    def test_d_model_divisible_by_n_heads(self) -> None:
        """d_model must be divisible by n_heads."""
        with pytest.raises(ValidationError, match="divisible"):
            DemoConfig(
                d_model=128,
                n_heads=3,
                downsample_factor=8,
                height=64,
                width=64,
            )

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should be rejected."""
        with pytest.raises(ValidationError):
            DemoConfig(unknown_field=42)  # type: ignore[call-arg]

    def test_empty_lambda_values_rejected(self) -> None:
        """Empty lambda values list should be rejected."""
        with pytest.raises(ValidationError):
            DemoConfig(lambda_values=[])

    def test_height_divisible_by_downsample(self) -> None:
        """Height must be divisible by downsample_factor."""
        with pytest.raises(ValidationError, match="divisible"):
            DemoConfig(height=50, downsample_factor=8, width=64)

    def test_to_summary_dict(self) -> None:
        """Summary dict should contain expected keys."""
        config = DemoConfig()
        summary = config.to_summary_dict()
        assert "video" in summary
        assert "codec" in summary
        assert "rd_sweep" in summary
        assert "resolution_test" in summary
        assert "runtime" in summary


# ---------------------------------------------------------------------------
# CompressionDemoRunner Tests
# ---------------------------------------------------------------------------


class TestCompressionDemoRunner:
    """Tests for the demo runner functionality."""

    def test_codec_creation(self, runner: CompressionDemoRunner) -> None:
        """Runner should successfully create a codec."""
        codec = runner._create_codec()
        assert codec is not None
        # Verify codec is on correct device
        param = next(codec.parameters())
        assert param.device.type == "cpu"

    def test_run_single_lambda(self, runner: CompressionDemoRunner) -> None:
        """Single lambda encoding should produce valid results."""
        frames = create_test_sequence(
            SyntheticPattern.GRADIENT,
            num_frames=2,
            height=32,
            width=32,
            seed=42,
        )
        result = runner.run_single_lambda(
            frames,
            lambda_rd=0.01,
            pattern_name="test_gradient",
            write_bitstream=False,
        )
        assert isinstance(result, LambdaResult)
        assert result.num_frames == 2
        assert len(result.frame_results) == 2
        assert result.total_bits > 0

    def test_single_lambda_result_structure(self, runner: CompressionDemoRunner) -> None:
        """Result should have all expected fields populated."""
        frames = create_test_sequence(num_frames=2, height=32, width=32, seed=42)
        result = runner.run_single_lambda(frames, 0.01, "test")

        # Check frame results
        for fr in result.frame_results:
            assert isinstance(fr, FrameResult)
            assert fr.frame_idx >= 0
            assert fr.frame_type in ("I", "P", "B")
            assert fr.rate_bits >= 0
            assert fr.encode_time_ms >= 0

        # Check aggregates
        assert result.avg_bpp >= 0
        assert result.avg_psnr_db > 0  # PSNR should be positive
        assert 0 <= result.avg_ssim <= 1  # SSIM in [0, 1]

    def test_psnr_positive(self, runner: CompressionDemoRunner) -> None:
        """PSNR should be positive (signal exists)."""
        frames = create_test_sequence(num_frames=2, height=32, width=32)
        result = runner.run_single_lambda(frames, 0.01, "test")
        assert result.avg_psnr_db > 0

    def test_run_rd_curve(self, runner: CompressionDemoRunner) -> None:
        """R-D curve should produce one result per lambda."""
        frames = create_test_sequence(num_frames=2, height=32, width=32)
        lambdas = [0.005, 0.01]
        results = runner.run_rd_curve(frames, lambdas, "test")
        assert len(results) == 2
        assert results[0].lambda_rd == 0.005
        assert results[1].lambda_rd == 0.01

    def test_run_resolution_test(self, runner: CompressionDemoRunner) -> None:
        """Resolution test should produce results for each size."""
        results = runner.run_resolution_test(
            SyntheticPattern.GRADIENT,
            lambda_rd=0.01,
        )
        assert len(results) == len(runner.config.resolution_sizes)
        for res in results:
            assert isinstance(res, ResolutionResult)
            assert res.avg_psnr_db > 0

    def test_bitstream_writing(self, small_config: DemoConfig) -> None:
        """Bitstream should be written when configured."""
        runner = CompressionDemoRunner(small_config)
        frames = create_test_sequence(num_frames=2, height=32, width=32)
        result = runner.run_single_lambda(frames, 0.01, "test", write_bitstream=True)
        assert result.bitstream_path is not None
        assert result.bitstream_size_bytes > 0
        assert Path(result.bitstream_path).exists()


# ---------------------------------------------------------------------------
# DemoResult Tests
# ---------------------------------------------------------------------------


class TestDemoResult:
    """Tests for demo result serialization."""

    def test_to_dict(self) -> None:
        """Result should convert to valid dict."""
        result = DemoResult(
            lambda_results=[
                LambdaResult(
                    lambda_rd=0.01,
                    pattern="test",
                    height=64,
                    width=64,
                    num_frames=4,
                    avg_bpp=0.5,
                    avg_psnr_db=25.0,
                    avg_ssim=0.9,
                )
            ],
            total_time_s=1.23,
            device="cpu",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert len(d["lambda_results"]) == 1
        assert d["total_time_s"] == 1.23
        assert d["device"] == "cpu"

    def test_to_json_valid(self) -> None:
        """Result should produce valid JSON."""
        result = DemoResult(
            lambda_results=[],
            resolution_results=[],
            total_time_s=0.5,
            device="cpu",
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["total_time_s"] == 0.5

    def test_empty_result(self) -> None:
        """Empty result should be serializable."""
        result = DemoResult()
        d = result.to_dict()
        assert len(d["lambda_results"]) == 0
        assert len(d["resolution_results"]) == 0
