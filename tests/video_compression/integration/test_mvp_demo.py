"""Integration tests for the video compression MVP demo.

End-to-end tests that exercise the complete pipeline:
- Full demo run with all components
- Bitstream write/read roundtrip
- Resolution independence across sizes
- Multiple pattern support
- JSON output validation
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.video_compression.data.synthetic import SyntheticPattern, create_test_sequence
from src.video_compression.demo.config import DemoConfig
from src.video_compression.demo.runner import CompressionDemoRunner, DemoResult
from src.video_compression.utils.bitstream import load_bitstream

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_config(tmp_path: Path) -> DemoConfig:
    """Create a small demo config for integration testing."""
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


# ---------------------------------------------------------------------------
# End-to-End Tests
# ---------------------------------------------------------------------------


class TestMVPDemoEndToEnd:
    """Full pipeline integration tests."""

    def test_full_demo_runs(self, demo_config: DemoConfig) -> None:
        """Full demo should complete without errors."""
        runner = CompressionDemoRunner(demo_config)
        result = runner.run_full_demo()
        assert isinstance(result, DemoResult)

    def test_full_demo_produces_results(self, demo_config: DemoConfig) -> None:
        """Full demo should produce non-empty results."""
        runner = CompressionDemoRunner(demo_config)
        result = runner.run_full_demo()

        # R-D results
        assert len(result.lambda_results) > 0
        for lr in result.lambda_results:
            assert lr.num_frames == 4
            assert lr.total_bits > 0
            assert lr.avg_psnr_db > 0
            assert 0 <= lr.avg_ssim <= 1

        # Resolution results
        assert len(result.resolution_results) > 0

        # Timing
        assert result.total_time_s > 0

    def test_json_output_written(self, demo_config: DemoConfig) -> None:
        """Demo should write JSON results file."""
        runner = CompressionDemoRunner(demo_config)
        runner.run_full_demo()

        json_path = Path(demo_config.output_dir) / "demo_results.json"
        assert json_path.exists()

    def test_json_output_valid(self, demo_config: DemoConfig) -> None:
        """JSON output should be valid and parseable."""
        runner = CompressionDemoRunner(demo_config)
        result = runner.run_full_demo()

        json_path = Path(demo_config.output_dir) / "demo_results.json"
        with open(json_path) as f:
            parsed = json.load(f)

        assert isinstance(parsed, dict)
        assert "lambda_results" in parsed
        assert "resolution_results" in parsed
        assert "total_time_s" in parsed
        assert "device" in parsed

        # Verify content matches result object
        assert len(parsed["lambda_results"]) == len(result.lambda_results)


# ---------------------------------------------------------------------------
# Bitstream Roundtrip Tests
# ---------------------------------------------------------------------------


class TestBitstreamRoundTripDemo:
    """Tests for bitstream I/O during demo."""

    def test_bitstream_write_read(self, demo_config: DemoConfig) -> None:
        """Written bitstream should be readable."""
        runner = CompressionDemoRunner(demo_config)
        frames = create_test_sequence(
            SyntheticPattern.GRADIENT,
            num_frames=4,
            height=32,
            width=32,
        )
        result = runner.run_single_lambda(frames, 0.01, "test", write_bitstream=True)

        assert result.bitstream_path is not None
        bitstream_path = Path(result.bitstream_path)
        assert bitstream_path.exists()

        # Read back and verify
        header, loaded_frames = load_bitstream(bitstream_path)
        assert header.num_frames == 4
        assert header.width == 32
        assert header.height == 32

    def test_bitstream_frame_count_matches(self, demo_config: DemoConfig) -> None:
        """Bitstream should contain correct number of frames."""
        runner = CompressionDemoRunner(demo_config)
        frames = create_test_sequence(
            SyntheticPattern.GRADIENT,
            num_frames=4,
            height=32,
            width=32,
        )
        result = runner.run_single_lambda(frames, 0.01, "test", write_bitstream=True)

        assert result.bitstream_path is not None
        _, loaded_frames = load_bitstream(result.bitstream_path)
        assert len(loaded_frames) == 4

    def test_bitstream_size_positive(self, demo_config: DemoConfig) -> None:
        """Bitstream file should have positive size."""
        runner = CompressionDemoRunner(demo_config)
        frames = create_test_sequence(
            SyntheticPattern.GRADIENT,
            num_frames=4,
            height=32,
            width=32,
        )
        result = runner.run_single_lambda(frames, 0.01, "test", write_bitstream=True)
        assert result.bitstream_size_bytes > 0


# ---------------------------------------------------------------------------
# Resolution Independence Tests
# ---------------------------------------------------------------------------


class TestResolutionIndependenceDemo:
    """Tests for resolution independence."""

    def test_multiple_resolutions(self, tmp_path: Path) -> None:
        """Codec should work at multiple resolutions with same architecture."""
        config = DemoConfig(
            num_frames=2,
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
            resolution_sizes=[(32, 32), (64, 64)],
            resolution_lambda=0.01,
            device="cpu",
            output_dir=str(tmp_path),
            write_bitstream=False,
        )

        runner = CompressionDemoRunner(config)
        results = runner.run_resolution_test(SyntheticPattern.GRADIENT, 0.01)
        assert len(results) == 2
        assert results[0].height == 32
        assert results[1].height == 64

    def test_quality_across_resolutions(self, tmp_path: Path) -> None:
        """Quality metrics should be valid at all resolutions."""
        config = DemoConfig(
            num_frames=2,
            height=32,
            width=32,
            patterns=[SyntheticPattern.WAVES],
            latent_channels=32,
            d_model=64,
            n_heads=2,
            d_ffn=128,
            n_layers=1,
            downsample_factor=4,
            lambda_values=[0.01],
            resolution_sizes=[(32, 32), (64, 64)],
            resolution_lambda=0.01,
            device="cpu",
            output_dir=str(tmp_path),
            write_bitstream=False,
        )

        runner = CompressionDemoRunner(config)
        results = runner.run_resolution_test(SyntheticPattern.WAVES, 0.01)
        for res in results:
            assert res.avg_psnr_db > 0, f"PSNR should be positive at {res.height}x{res.width}"
            assert 0 <= res.avg_ssim <= 1, f"SSIM out of range at {res.height}x{res.width}"


# ---------------------------------------------------------------------------
# Multiple Pattern Tests
# ---------------------------------------------------------------------------


class TestMultiplePatterns:
    """Tests for pattern variety."""

    @pytest.mark.parametrize(
        "pattern",
        [SyntheticPattern.GRADIENT, SyntheticPattern.WAVES, SyntheticPattern.NOISE],
    )
    def test_pattern_demo(self, pattern: SyntheticPattern, tmp_path: Path) -> None:
        """Each pattern should produce valid demo results."""
        config = DemoConfig(
            num_frames=2,
            height=32,
            width=32,
            patterns=[pattern],
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
            output_dir=str(tmp_path),
            write_bitstream=False,
        )

        runner = CompressionDemoRunner(config)
        result = runner.run_full_demo()
        assert len(result.lambda_results) > 0
        assert result.lambda_results[0].avg_psnr_db > 0
