"""MCTS-based rate control for video compression.

Uses MuZero-style learned world model for GOP-level bit allocation:
- Policy network: Predicts QP/mode distribution
- Value network: Estimates R-D cost
- Dynamics network: Predicts next frame state
"""

from src.video_compression.mcts.rate_control import (
    MCTSRateController,
    RateControlDecision,
    GOPPlanner,
)
from src.video_compression.mcts.networks import (
    PolicyNetwork,
    ValueNetwork,
    DynamicsNetwork,
)

__all__ = [
    "MCTSRateController",
    "RateControlDecision",
    "GOPPlanner",
    "PolicyNetwork",
    "ValueNetwork",
    "DynamicsNetwork",
]
