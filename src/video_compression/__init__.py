"""AlphaGalerkin Neural Video Compression System.

A resolution-independent neural video codec combining:
- FNet mixing layers for O(N log N) frequency analysis
- Galerkin attention for resolution-independent feature extraction
- MCTS planning for GOP-level bit allocation decisions
- Hyperprior entropy model for variational compression
"""

from src.video_compression.config import (
    CodecConfig,
    DecoderConfig,
    EncoderConfig,
    EntropyConfig,
    MCTSRateControlConfig,
    QuantizerConfig,
    TrainingConfig,
)

__all__ = [
    "CodecConfig",
    "EncoderConfig",
    "DecoderConfig",
    "EntropyConfig",
    "MCTSRateControlConfig",
    "QuantizerConfig",
    "TrainingConfig",
]
