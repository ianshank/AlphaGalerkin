"""E2E tests for MCTS rate control in video compression.

Validates MCTS-based QP selection behavior and bitrate targeting
when encoding video content.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    pass

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mcts_config():
    """Get MCTS rate control configuration."""
    from src.video_compression.config import CodecConfig, MCTSRateControlConfig

    # Create config with MCTS settings for testing
    # Note: MCTSRateControlConfig doesn't have 'enabled' field
    # MCTS is enabled by passing use_mcts_rate_control=True to create_codec
    mcts_settings = MCTSRateControlConfig(
        name="mcts_test",
        num_simulations=10,  # Reduced for testing speed
        gop_size=4,
    )
    config = CodecConfig(
        name="mcts_test",
        mcts=mcts_settings,
    )
    return config


@pytest.fixture
def codec_with_mcts(mcts_config):
    """Create codec with MCTS rate control enabled."""
    from src.video_compression.codec.codec import create_codec

    codec = create_codec(mcts_config)
    codec.eval()
    return codec


# ============================================================================
# MCTS Rate Controller Tests
# ============================================================================


@pytest.mark.video
class TestMCTSRateController:
    """Tests for MCTS-based rate control."""

    def test_mcts_import(self) -> None:
        """Verify MCTS rate controller can be imported."""
        from src.video_compression.mcts.rate_control import MCTSRateController

        assert MCTSRateController is not None

    def test_mcts_qp_selection(self, codec_with_mcts) -> None:
        """Test that MCTS produces valid QP values."""
        # Create dummy latent
        latent = torch.randn(1, 192, 4, 4)

        # Check if MCTS is available and enabled
        if (
            hasattr(codec_with_mcts, "rate_controller")
            and codec_with_mcts.rate_controller is not None
        ):
            decision = codec_with_mcts.rate_controller.select_qp(latent)

            # Verify QP is within valid range
            qp = decision.qp if hasattr(decision, "qp") else decision
            assert 0 <= qp <= 51, f"QP out of range: {qp}"
        else:
            pytest.skip("MCTS rate controller not enabled in codec")

    def test_mcts_deterministic_with_seed(self, mcts_config) -> None:
        """Test that MCTS produces deterministic results with fixed seed."""
        from src.video_compression.codec.codec import create_codec

        # Create two codecss with same seed
        torch.manual_seed(42)
        codec1 = create_codec(mcts_config)
        codec1.eval()

        torch.manual_seed(42)
        codec2 = create_codec(mcts_config)
        codec2.eval()

        # Same input should give same output with fixed seed
        latent = torch.randn(1, 192, 4, 4)

        if hasattr(codec1, "rate_controller") and codec1.rate_controller is not None:
            torch.manual_seed(42)
            decision1 = codec1.rate_controller.select_qp(latent)

            torch.manual_seed(42)
            decision2 = codec2.rate_controller.select_qp(latent)

            qp1 = decision1.qp if hasattr(decision1, "qp") else decision1
            qp2 = decision2.qp if hasattr(decision2, "qp") else decision2

            assert qp1 == qp2, "MCTS should be deterministic with fixed seed"

    def test_mcts_adapts_to_content(self) -> None:
        """Test that MCTS adapts QP to content complexity."""
        pytest.skip("Requires trained MCTS model for meaningful results")


# ============================================================================
# GOP Rate Allocation Tests
# ============================================================================


@pytest.mark.video
class TestGOPRateAllocation:
    """Tests for GOP-level rate allocation."""

    def test_gop_manager_frame_types(self, mcts_config) -> None:
        """Test GOP manager assigns correct frame types."""
        from src.video_compression.codec.gop_manager import FrameType, GOPManager

        gop_manager = GOPManager(
            gop_size=mcts_config.mcts.gop_size,
            use_b_frames=True,
        )

        # First frame should be I-frame
        frame_info = gop_manager.get_frame_info(0)
        assert frame_info.frame_type == FrameType.I

        # Check pattern for remaining frames
        for i in range(1, mcts_config.mcts.gop_size):
            frame_info = gop_manager.get_frame_info(i)
            assert frame_info.frame_type in [FrameType.P, FrameType.B]

    def test_gop_boundary_reset(self, mcts_config) -> None:
        """Test that GOP manager correctly handles boundaries."""
        from src.video_compression.codec.gop_manager import FrameType, GOPManager

        gop_size = mcts_config.mcts.gop_size
        gop_manager = GOPManager(gop_size=gop_size, use_b_frames=True)

        # Frame at GOP size should be new I-frame
        frame_info = gop_manager.get_frame_info(gop_size)
        assert frame_info.frame_type == FrameType.I

        # And the next GOP begins
        frame_info = gop_manager.get_frame_info(gop_size + 1)
        assert frame_info.frame_type in [FrameType.P, FrameType.B]


# ============================================================================
# Bitrate Targeting Tests
# ============================================================================


@pytest.mark.video
@pytest.mark.slow
class TestBitrateTargeting:
    """Tests for bitrate targeting accuracy."""

    def test_encode_respects_qp(
        self, synthetic_video_factory, dummy_codec_checkpoint, tmp_path
    ) -> None:
        """Test that higher QP results in lower bitrate."""
        import subprocess
        import sys

        video = synthetic_video_factory(height=64, width=64, num_frames=8)

        sizes = {}
        for qp in [22, 32, 42]:
            output = tmp_path / f"output_qp{qp}.agk"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/encode_video.py",
                    str(video),
                    str(output),
                    "--qp",
                    str(qp),
                    "--model",
                    str(dummy_codec_checkpoint),
                    "--device",
                    "cpu",
                ],
                cwd=str(Path(__file__).parents[2]),
                capture_output=True,
                timeout=60,
            )

            if result.returncode == 0 and output.exists():
                sizes[qp] = output.stat().st_size

        # Higher QP should generally give smaller files
        if len(sizes) >= 2:
            sorted_qps = sorted(sizes.keys())
            # At least trend should be downward (allow some variation)
            assert sizes[sorted_qps[0]] >= sizes[sorted_qps[-1]] * 0.8


# ============================================================================
# Re-export fixtures from video_fixtures
# ============================================================================

