"""Hyperparameter tuning infrastructure for PoC framework.

This module provides automated hyperparameter tuning capabilities
for finding optimal configurations for scenarios.

Key Components:
    - TuningConfig: Configuration for tuning runs
    - SearchSpace: Parameter search space definitions
    - HyperparameterTuner: Main tuning orchestrator
    - Samplers: Grid, random, and Bayesian samplers

Usage:
    from src.poc.tuning import TuningConfig, HyperparameterTuner

    config = TuningConfig(
        n_trials=100,
        sampler="tpe",
        search_space={...},
    )
    tuner = HyperparameterTuner(config, scenario)
    result = tuner.tune()
"""

from src.poc.tuning.config import SearchSpace, TuningConfig
from src.poc.tuning.sampler import create_sampler
from src.poc.tuning.tuner import HyperparameterTuner, TuningResult

__all__ = [
    "HyperparameterTuner",
    "SearchSpace",
    "TuningConfig",
    "TuningResult",
    "create_sampler",
]
