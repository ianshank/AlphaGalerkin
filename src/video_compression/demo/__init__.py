"""Video compression MVP demo package.

Provides a self-contained demo of the AlphaGalerkin neural video codec:
- Synthetic video generation (no external dependencies)
- Encode/decode pipeline with bitstream I/O
- Rate-distortion curve evaluation
- Resolution independence verification
- Structured logging and JSON output

Usage:
    from src.video_compression.demo import CompressionDemoRunner, DemoConfig

    config = DemoConfig(num_frames=8, height=64, width=64)
    runner = CompressionDemoRunner(config)
    result = runner.run_full_demo()
    print(result.to_json())
"""

from src.video_compression.demo.config import DemoConfig
from src.video_compression.demo.runner import CompressionDemoRunner, DemoResult

__all__ = [
    "DemoConfig",
    "DemoResult",
    "CompressionDemoRunner",
]
