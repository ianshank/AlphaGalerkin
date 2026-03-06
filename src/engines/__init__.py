"""External chess engine integration for AlphaGalerkin.

This module provides infrastructure for communicating with external
chess engines (e.g., Stockfish) via the UCI protocol, enabling
benchmarking and evaluation against established engines.

Key Components:
    - BaseEngine: Abstract engine interface
    - UCIEngine: UCI protocol implementation via subprocess
    - EngineEvaluator: Adapter bridging engines to MCTS Evaluator protocol
    - EngineMatch: Match orchestration framework
    - EloCalculator: Elo rating estimation
    - EngineRegistry: Engine protocol registry

Usage:
    from src.engines import UCIEngine, UCIConfig

    config = UCIConfig(
        name="stockfish",
        engine_path=Path("/usr/bin/stockfish"),
        depth_limit=20,
    )
    with UCIEngine(config) as engine:
        engine.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        best_move, info = engine.go(depth=20)
"""

from __future__ import annotations

from src.engines.config import (
    EloConfig,
    EngineConfig,
    MatchConfig,
    TimeControl,
    UCIConfig,
)
from src.engines.elo import EloCalculator, EloEstimate
from src.engines.protocol import BaseEngine, EngineInfo, EngineProtocol
from src.engines.registry import EngineRegistry, create_engine
from src.engines.uci import UCIEngine

__all__ = [
    "BaseEngine",
    "EloCalculator",
    "EloConfig",
    "EloEstimate",
    "EngineConfig",
    "EngineInfo",
    "EngineProtocol",
    "EngineRegistry",
    "MatchConfig",
    "TimeControl",
    "UCIConfig",
    "UCIEngine",
    "create_engine",
]
