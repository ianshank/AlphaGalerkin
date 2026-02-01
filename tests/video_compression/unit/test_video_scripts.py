
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).parents[3]))

from scripts.decode_video import parse_args as parse_decode_args
from scripts.decode_video import write_video_frames

class TestDecodeVideoCLI:
    def test_parse_args_prores(self) -> None:
        """Test that --prores flag is parsed correctly."""
        # Note: we need to ensure we provide all required args
        test_args = ["scripts/decode_video.py", "--input", "in.agk", "--output", "out.mov", "--prores", "--checkpoint", "model.pt"]
        with patch.object(sys, "argv", test_args):
            args = parse_decode_args()
            assert args.prores is True
            assert args.input == Path("in.agk")
            assert args.output == Path("out.mov")

    def test_parse_args_defaults(self) -> None:
        """Test default arguments."""
        test_args = ["scripts/decode_video.py", "--input", "in.agk", "--output", "out.mp4", "--checkpoint", "model.pt"]
        with patch.object(sys, "argv", test_args):
            args = parse_decode_args()
            assert args.prores is False

class TestWriteVideoFrames:
    def test_write_video_frames_prores(self) -> None:
        """Test that ProRes codec is selected for .mov when requested."""
        # Create a dummy tensor frame (1, 3, 64, 64)
        frame = torch.zeros(1, 3, 64, 64)
        frames = [frame]
        output_path = Path("output.mov")
        
        mock_cv2 = MagicMock()
        mock_cv2.VideoWriter_fourcc.return_value = 12345
        
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            write_video_frames(iter(frames), output_path, fps=30.0, width=64, height=64, use_prores=True)
            
            # Check that 'apcn' (ProRes) was correctly converted to fourcc
            mock_cv2.VideoWriter_fourcc.assert_called_with('a', 'p', 'c', 'n')

    def test_write_video_frames_default_mov(self) -> None:
        """Test default codec for .mov when prores is False."""
        frame = torch.zeros(1, 3, 64, 64)
        frames = [frame]
        output_path = Path("output.mov")
        
        mock_cv2 = MagicMock()
        mock_cv2.VideoWriter_fourcc.return_value = 12345

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            write_video_frames(iter(frames), output_path, fps=30.0, width=64, height=64, use_prores=False)
            
            # Should default to mp4v
            mock_cv2.VideoWriter_fourcc.assert_called_with('m', 'p', '4', 'v')
