"""Evaluation utilities for AlphaGalerkin training.

This module provides:
- EloTracker: Track Elo ratings across checkpoints
- MultiResolutionEvaluator: Evaluate at multiple board sizes
"""

from src.training.eval_utils.elo_tracker import EloRating, EloTracker

__all__ = ["EloRating", "EloTracker"]
