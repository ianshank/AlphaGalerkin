"""λ-window scheduling ablation — the first non-PDE ``RefinementGame``.

A falsification test for the generality of ``src.refinement``: does single-agent
MCTS planning beat greedy variance-weighted allocation for free-energy (BAR/FEP)
λ-window sample scheduling, and does it survive surrogate-model error? Not a
chemistry product — a controlled ablation with a binding kill criterion (see
``specs/lambda_scheduling.spec.md``).
"""

from src.thermo.config import (
    HardnessProfileConfig,
    LambdaSchedulingConfig,
    SchedulingParams,
)
from src.thermo.game import LambdaSchedulingGame
from src.thermo.surrogate import (
    AnalyticSurrogate,
    MismatchedSurrogate,
    OperatorSurrogate,
    RecordedSurrogate,
    VarianceSurrogate,
)

__all__ = [
    "AnalyticSurrogate",
    "HardnessProfileConfig",
    "LambdaSchedulingConfig",
    "LambdaSchedulingGame",
    "MismatchedSurrogate",
    "OperatorSurrogate",
    "RecordedSurrogate",
    "SchedulingParams",
    "VarianceSurrogate",
]
