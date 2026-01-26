"""Statistical analysis utilities for PoC framework.

This module provides tools for statistical significance testing
and effect size calculation to rigorously compare experimental results.

Key Components:
    - SignificanceTest: Configuration for statistical tests
    - StatisticalAnalyzer: Main analysis class
    - Effect size calculations (Cohen's d, etc.)
    - Multiple comparison corrections

Usage:
    from src.poc.statistics import StatisticalAnalyzer

    analyzer = StatisticalAnalyzer()
    result = analyzer.compare_runs(baseline_values, treatment_values)
    print(f"p-value: {result.p_value}, significant: {result.is_significant}")
"""

from src.poc.statistics.significance import (
    ComparisonResult,
    EffectSizeResult,
    SignificanceTest,
    StatisticalAnalyzer,
)

__all__ = [
    "ComparisonResult",
    "EffectSizeResult",
    "SignificanceTest",
    "StatisticalAnalyzer",
]
