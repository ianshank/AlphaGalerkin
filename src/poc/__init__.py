"""Proof of Concept (PoC) Scenario Framework.

This module provides a configuration-driven framework for defining,
executing, and validating proof-of-concept scenarios for AlphaGalerkin.

Architecture:
    - config.py: Pydantic models for scenario configuration
    - registry.py: Scenario registration and discovery
    - runner.py: Scenario execution engine
    - results.py: Result collection and persistence
    - logging.py: Structured logging utilities

Usage:
    from src.poc import ScenarioRunner, scenario

    # Define a scenario
    @scenario("my_scenario")
    class MyScenario(BaseScenario):
        def execute(self, config: ScenarioConfig) -> ScenarioResult:
            ...

    # Run scenarios
    runner = ScenarioRunner()
    results = runner.run_all()
"""

from src.poc.config import (
    BaseScenarioConfig,
    ComplexityScenarioConfig,
    ScenarioResult,
    ScenarioStatus,
    ScenarioTier,
    StabilityScenarioConfig,
    TransferScenarioConfig,
)
from src.poc.logging import ScenarioLogger, get_scenario_logger
from src.poc.registry import BaseScenario, ScenarioRegistry, scenario
from src.poc.results import ResultCollector
from src.poc.runner import ScenarioRunner

__all__ = [
    # Config
    "BaseScenarioConfig",
    "TransferScenarioConfig",
    "ComplexityScenarioConfig",
    "StabilityScenarioConfig",
    "ScenarioResult",
    "ScenarioStatus",
    "ScenarioTier",
    # Registry
    "ScenarioRegistry",
    "BaseScenario",
    "scenario",
    # Runner
    "ScenarioRunner",
    # Results
    "ResultCollector",
    # Logging
    "ScenarioLogger",
    "get_scenario_logger",
]
