"""Variance surrogates for λ-window scheduling.

A free-energy (BAR / FEP) calculation partitions ``λ ∈ [0, 1]`` into windows.
Window *i* run with ``n_i`` samples has standard error ``σ_i = c_i / n_i**0.5``
where ``c_i`` is a per-window *variance coefficient*. The total ΔG standard error
is ``sqrt(Σ σ_i**2) = sqrt(Σ c_i**2 / n_i)``. Scheduling is the problem of
choosing the windows and their sample counts to minimise that under a budget.

A ``VarianceSurrogate`` supplies ``c(window)``. Four implementations:

* ``AnalyticSurrogate`` — closed-form ``c`` from a known hardness profile. This is
  *ground truth* in the ablation.
* ``MismatchedSurrogate`` — analytic ``c`` perturbed by a bias (and optional
  deterministic, λ-keyed noise). Models the real case where ``c`` is estimated
  from finite samples and is always wrong. **Mandatory** to the honest test.
* ``RecordedSurrogate`` — replays a committed table of ``(lo, hi) -> c`` (an MD
  fixture stand-in). CPU / CI safe.
* ``OperatorSurrogate`` — ``c(λ)`` from a Fourier-feature operator (the research
  question, P5). Not run in CI.

``BAR_VARIANCE_EXPONENT = 0.5`` is fixed by the ``σ ∝ n**(-1/2)`` scaling of a
Monte-Carlo standard error (variance ``∝ 1/n``); it is a named constant, not a
tunable, and is documented here rather than surfaced as a config field.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.thermo.config import HardnessProfileConfig

# σ_i = c_i / n_i**BAR_VARIANCE_EXPONENT. Fixed by the central-limit scaling of a
# Monte-Carlo standard error; see module docstring.
BAR_VARIANCE_EXPONENT = 0.5


@runtime_checkable
class VarianceSurrogate(Protocol):
    """Supplies the variance coefficient ``c`` for a λ-window ``[lo, hi]``."""

    def variance_coeff(self, lam_lo: float, lam_hi: float) -> float:
        """Return the (non-negative) variance coefficient for the window."""
        ...


def _hardness(lam: float, profile: HardnessProfileConfig) -> float:
    """Baseline hardness plus a Gaussian bump (a mock phase transition)."""
    gap = lam - profile.peak_center
    bump = profile.peak_amplitude * math.exp(-0.5 * (gap / profile.peak_width) ** 2)
    return profile.baseline + bump


class AnalyticSurrogate:
    """Closed-form ``c(window) = sqrt(width * hardness(midpoint))``.

    ``c**2 = width * hardness`` makes the per-window *variance* scale with both
    the window width and the local hardness, so a peaked hardness profile makes
    some windows far noisier — the regime where allocation matters.
    """

    def __init__(self, profile: HardnessProfileConfig) -> None:
        self.profile = profile

    def variance_coeff(self, lam_lo: float, lam_hi: float) -> float:
        width = max(lam_hi - lam_lo, 0.0)
        mid = 0.5 * (lam_lo + lam_hi)
        return math.sqrt(width * _hardness(mid, self.profile))


class MismatchedSurrogate:
    """Analytic ``c`` scaled by ``(1 + bias)`` plus deterministic λ-keyed noise.

    The noise is a fixed trigonometric function of the window midpoint, so the
    surrogate is deterministic (no RNG) yet mis-shaped relative to the truth —
    the planner sees this while the world is scored on the analytic truth.
    """

    def __init__(
        self,
        truth: AnalyticSurrogate,
        bias: float,
        noise_amplitude: float = 0.0,
        noise_frequency: float = 7.0,
    ) -> None:
        self.truth = truth
        self.bias = bias
        self.noise_amplitude = noise_amplitude
        self.noise_frequency = noise_frequency

    def variance_coeff(self, lam_lo: float, lam_hi: float) -> float:
        base = self.truth.variance_coeff(lam_lo, lam_hi)
        mid = 0.5 * (lam_lo + lam_hi)
        noise = self.noise_amplitude * math.sin(self.noise_frequency * math.pi * mid)
        return max(base * (1.0 + self.bias) + noise, 0.0)


class RecordedSurrogate:
    """Replays a committed table of ``(lo, hi) -> c`` (an MD fixture stand-in).

    For a query window it returns the coefficient of the recorded window whose
    midpoint is nearest — a piecewise-constant surrogate over the fixture grid.
    """

    def __init__(self, table: list[tuple[float, float, float]]) -> None:
        if not table:
            raise ValueError("RecordedSurrogate requires a non-empty table")
        self.table = table

    def variance_coeff(self, lam_lo: float, lam_hi: float) -> float:
        mid = 0.5 * (lam_lo + lam_hi)
        best = min(self.table, key=lambda row: abs(0.5 * (row[0] + row[1]) - mid))
        return max(best[2], 0.0)


class OperatorSurrogate:
    """``c(λ)`` from a Fourier-feature operator (P5 research question).

    Not exercised in CI. Constructing it without a fitted operator raises so the
    ablation cannot silently fall back to a hand-tuned model.
    """

    def __init__(self, predict_fn: object | None = None) -> None:
        if predict_fn is None:
            raise NotImplementedError(
                "OperatorSurrogate needs a fitted c(λ) operator (P5); no default. "
                "Pass a callable predict_fn(lo, hi) -> float."
            )
        self._predict_fn = predict_fn

    def variance_coeff(self, lam_lo: float, lam_hi: float) -> float:
        return max(float(self._predict_fn(lam_lo, lam_hi)), 0.0)  # type: ignore[operator]
