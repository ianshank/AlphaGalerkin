import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# Ensure src is in pythonpath
sys.path.append(str(Path(__file__).parents[2]))

from scripts.decode_video import main as decode_main
from scripts.encode_video import main as encode_main
from src.video_compression.config import CodecConfig


cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")


@pytest.fixture
def temp_video_file(tmp_path):
    """Create a dummy stored video file."""
    import numpy as np

    path = tmp_path / "test_video.mp4"
    height, width = 64, 64
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for _ in range(10):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


@pytest.mark.integration
def test_end_to_end_workflow(temp_video_file: Path, tmp_path: Path) -> None:
    """Test the full encode-decode cycle with a dummy codec."""
    # Output paths
    bitstream_path = tmp_path / "output.agk"
    decoded_path = tmp_path / "decoded.mp4"
    model_path = tmp_path / "dummy_model.pt"

    # Create a dummy model checkpoint
    # We need to save a state dict that matches the codec structure
    # Since we can't easily train one here, we'll try to rely on the script
    # instantiating a random model if no checkpoint is provided,
    # OR we mock the model loading to do nothing if we pass 'None' but the script expects a path.
    # Actually, let's just create a random state dict matching the default config.

    from src.video_compression.codec.codec import create_codec

    config = CodecConfig(name="test_codec")
    model = create_codec(config)
    torch.save({"model_state_dict": model.state_dict()}, model_path)

    # 1. Encode
    # We mock sys.argv
    test_args = [
        "encode_video.py",
        str(temp_video_file),
        str(bitstream_path),
        "--qp",
        "32",
        "--model",
        str(model_path),
        "--device",
        "cpu",  # Force CPU for CI/test env compatibility
        "--gop-size",
        "4",
    ]

    with patch.object(sys, "argv", test_args):
        encode_main()

    assert bitstream_path.exists()
    assert bitstream_path.stat().st_size > 0

    # 2. Decode
    test_decode_args = [
        "decode_video.py",
        "--input",
        str(bitstream_path),
        "--output",
        str(decoded_path),
        "--checkpoint",
        str(model_path),
        "--device",
        "cpu",
    ]

    with patch.object(sys, "argv", test_decode_args):
        ret = decode_main()
        assert ret == 0

    assert decoded_path.exists()
    assert decoded_path.stat().st_size > 0

    # 3. Check extension preservation logic indirectly
    # The output filename above was explicitly .mp4.
    # Let's try .mov
    decoded_mov = tmp_path / "decoded.mov"
    test_decode_args_mov = [
        "decode_video.py",
        "--input",
        str(bitstream_path),
        "--output",
        str(decoded_mov),
        "--checkpoint",
        str(model_path),
        "--device",
        "cpu",
    ]

    with patch.object(sys, "argv", test_decode_args_mov):
        ret = decode_main()
        assert ret == 0
    assert decoded_mov.exists()
