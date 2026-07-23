"""Built-in scenario implementations.

This module contains the core PoC scenarios for AlphaGalerkin:
    - TransferScenario: Zero-shot resolution transfer validation
    - ComplexityScenario: O(N) Galerkin / O(N log N) FNet benchmarks
    - StabilityScenario: LBB stability monitoring
    - NoyronHXScenario: Zero-shot 3D heat-equation transfer on a Leap 71
      helical heat exchanger SDF (Leap 71 / PicoGK integration).
    - LLMPriorAblationScenario: MCTS basis selection with random / trained /
      LM-Studio-Qwen evaluators on ID + OOD PDEs ([lm-studio] extra).
    - NoyronBasisScenario: MCTS-guided Galerkin basis selection on a Leap 71
      helical SDF operator (v2.2 — first MCTS-on-Noyron result).
    - LShapeAMRCompareScenario: L-shaped Poisson AMR head-to-head — MCTS
      refinement policy vs classical Dörfler marking on a shared solver.
    - TransferBaselineCompareScenario: Honest zero-shot transfer — the
      AlphaGalerkin operator (zero-shot) vs a retrained discrete CNN.
"""

from src.poc.scenarios.complexity import ComplexityScenario
from src.poc.scenarios.llm_prior_ablation import LLMPriorAblationScenario
from src.poc.scenarios.lshape_amr_compare import LShapeAMRCompareScenario
from src.poc.scenarios.noyron_basis import NoyronBasisScenario
from src.poc.scenarios.noyron_hx import NoyronHXScenario
from src.poc.scenarios.scaling_law import ScalingLawScenario
from src.poc.scenarios.stability import StabilityScenario
from src.poc.scenarios.transfer import TransferScenario
from src.poc.scenarios.transfer_baseline_compare import TransferBaselineCompareScenario

__all__ = [
    "ComplexityScenario",
    "LLMPriorAblationScenario",
    "LShapeAMRCompareScenario",
    "NoyronBasisScenario",
    "NoyronHXScenario",
    "ScalingLawScenario",
    "StabilityScenario",
    "TransferBaselineCompareScenario",
    "TransferScenario",
]
