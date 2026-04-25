"""Comparison against reference data (flight data, solver results).

Provides structured comparison of computed results against
reference data for validation reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ComparisonMetric:
    """Single comparison metric between computed and reference."""

    name: str
    computed: float
    reference: float
    unit: str
    tolerance_percent: float

    @property
    def error_absolute(self) -> float:
        return abs(self.computed - self.reference)

    @property
    def error_percent(self) -> float:
        if abs(self.reference) < 1e-30:
            return 0.0 if abs(self.computed) < 1e-30 else float("inf")
        return abs(self.computed - self.reference) / abs(self.reference) * 100.0

    @property
    def passes(self) -> bool:
        return self.error_percent <= self.tolerance_percent


@dataclass
class ValidationReport:
    """Complete validation report with multiple metrics."""

    case_name: str
    metrics: list[ComparisonMetric]

    @property
    def all_pass(self) -> bool:
        return all(m.passes for m in self.metrics)

    @property
    def n_pass(self) -> int:
        return sum(1 for m in self.metrics if m.passes)

    @property
    def n_fail(self) -> int:
        return sum(1 for m in self.metrics if not m.passes)

    def summary(self) -> str:
        lines = [f"Validation: {self.case_name}"]
        lines.append(
            f"  Result: {'PASS' if self.all_pass else 'FAIL'} "
            f"({self.n_pass}/{len(self.metrics)} metrics pass)"
        )
        for m in self.metrics:
            status = "PASS" if m.passes else "FAIL"
            lines.append(
                f"  [{status}] {m.name}: computed={m.computed:.6g} "
                f"ref={m.reference:.6g} {m.unit} "
                f"(err={m.error_percent:.2f}%, tol={m.tolerance_percent}%)"
            )
        return "\n".join(lines)


def compare_stagnation_heat_flux(
    computed_q: float,
    reference_q: float,
    tolerance_percent: float = 15.0,
) -> ComparisonMetric:
    """Compare computed stagnation heat flux against reference."""
    return ComparisonMetric(
        name="stagnation_heat_flux",
        computed=computed_q,
        reference=reference_q,
        unit="W/m^2",
        tolerance_percent=tolerance_percent,
    )


def compare_surface_pressure(
    computed_p: NDArray[np.float64],
    reference_p: NDArray[np.float64],
    tolerance_percent: float = 5.0,
) -> ComparisonMetric:
    """Compare surface pressure distributions (L2 norm)."""
    min_len = min(len(computed_p), len(reference_p))
    c = computed_p[:min_len]
    r = reference_p[:min_len]
    l2_computed = float(np.sqrt(np.mean(c**2)))
    l2_ref = float(np.sqrt(np.mean(r**2)))
    return ComparisonMetric(
        name="surface_pressure_L2",
        computed=l2_computed,
        reference=l2_ref,
        unit="Pa",
        tolerance_percent=tolerance_percent,
    )


# FIRE II flight data reference values (Cauchon 1966, NASA TN D-3646)
FIRE2_FLIGHT_DATA = {
    "t1634s": {"q_stag_W_m2": 1.14e6, "p_stag_Pa": 2.83e4},
    "t1636s": {"q_stag_W_m2": 1.15e6, "p_stag_Pa": 3.35e4},
    "t1637.5s": {"q_stag_W_m2": 1.08e6, "p_stag_Pa": 3.90e4},
    "t1643s": {"q_stag_W_m2": 0.58e6, "p_stag_Pa": 5.67e4},
    "t1645s": {"q_stag_W_m2": 0.40e6, "p_stag_Pa": 6.30e4},
}
