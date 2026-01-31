"""Built-in scenario implementations.

This module contains the core PoC scenarios for AlphaGalerkin:
    - TransferScenario: Zero-shot resolution transfer validation
    - ComplexityScenario: O(N) Galerkin / O(N log N) FNet benchmarks
    - StabilityScenario: LBB stability monitoring
"""

from src.poc.scenarios.complexity import ComplexityScenario
from src.poc.scenarios.stability import StabilityScenario
from src.poc.scenarios.transfer import TransferScenario

__all__ = [
    "TransferScenario",
    "ComplexityScenario",
    "StabilityScenario",
]
