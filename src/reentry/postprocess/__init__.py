"""Post-processing and validation for reentry simulations."""

from src.reentry.postprocess.comparison import (
    ComparisonMetric,
    ValidationReport,
    compare_stagnation_heat_flux,
)
from src.reentry.postprocess.stagnation import (
    StagnationResult,
    find_stagnation_point,
    sutton_graves_heat_flux,
)
from src.reentry.postprocess.surface import SurfaceData, extract_surface

__all__ = [
    "SurfaceData",
    "extract_surface",
    "StagnationResult",
    "find_stagnation_point",
    "sutton_graves_heat_flux",
    "ComparisonMetric",
    "ValidationReport",
    "compare_stagnation_heat_flux",
]
