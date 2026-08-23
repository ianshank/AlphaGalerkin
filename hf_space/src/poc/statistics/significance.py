"""Statistical significance testing utilities.

This module provides tools for comparing experimental results
using various statistical tests with proper corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class SignificanceTest(BaseModel):
    """Configuration for statistical significance testing.

    Attributes:
        test_type: Type of statistical test.
        alpha: Significance level.
        correction: Multiple comparison correction method.
        alternative: Alternative hypothesis.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    test_type: Literal["t_test", "mann_whitney", "bootstrap", "permutation"] = Field(
        default="bootstrap",
        description="Type of statistical test",
    )
    alpha: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Significance level",
    )
    correction: Literal["none", "bonferroni", "holm", "fdr"] = Field(
        default="bonferroni",
        description="Multiple comparison correction",
    )
    alternative: Literal["two-sided", "less", "greater"] = Field(
        default="two-sided",
        description="Alternative hypothesis",
    )

    # Bootstrap-specific settings
    n_bootstrap: int = Field(
        default=10000,
        ge=1000,
        description="Number of bootstrap samples",
    )
    confidence_level: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description="Confidence level for intervals",
    )


@dataclass
class ComparisonResult:
    """Result from a statistical comparison."""

    test_type: str
    statistic: float
    p_value: float
    p_value_corrected: float | None
    is_significant: bool
    confidence_interval: tuple[float, float] | None
    n_baseline: int
    n_treatment: int
    mean_baseline: float
    mean_treatment: float
    std_baseline: float
    std_treatment: float
    mean_difference: float
    effect_size: float | None = None


@dataclass
class EffectSizeResult:
    """Effect size calculation result."""

    cohens_d: float
    hedges_g: float
    cliff_delta: float | None
    interpretation: str  # "small", "medium", "large"


class StatisticalAnalyzer:
    """Analyzes experimental results for statistical significance.

    Provides various statistical tests for comparing baseline and
    treatment groups with proper handling of multiple comparisons.

    Attributes:
        test_config: Default test configuration.

    """

    def __init__(
        self,
        test_config: SignificanceTest | None = None,
    ) -> None:
        """Initialize analyzer.

        Args:
            test_config: Default test configuration.

        """
        self.test_config = test_config or SignificanceTest()
        self._logger = structlog.get_logger(__name__)

    def compare_runs(
        self,
        baseline: list[float] | np.ndarray,
        treatment: list[float] | np.ndarray,
        test: SignificanceTest | None = None,
    ) -> ComparisonResult:
        """Compare two sets of experimental results.

        Args:
            baseline: Baseline run results.
            treatment: Treatment run results.
            test: Test configuration (uses default if None).

        Returns:
            ComparisonResult with statistical analysis.

        """
        test = test or self.test_config

        baseline = np.array(baseline)
        treatment = np.array(treatment)

        # Basic statistics
        mean_baseline = np.mean(baseline)
        mean_treatment = np.mean(treatment)
        std_baseline = np.std(baseline, ddof=1)
        std_treatment = np.std(treatment, ddof=1)

        # Run statistical test
        if test.test_type == "t_test":
            statistic, p_value = self._t_test(baseline, treatment, test)
        elif test.test_type == "mann_whitney":
            statistic, p_value = self._mann_whitney(baseline, treatment, test)
        elif test.test_type == "bootstrap":
            statistic, p_value, ci = self._bootstrap_test(baseline, treatment, test)
        elif test.test_type == "permutation":
            statistic, p_value = self._permutation_test(baseline, treatment, test)
        else:
            raise ValueError(f"Unknown test type: {test.test_type}")

        # Confidence interval
        ci = None
        if test.test_type == "bootstrap":
            ci = self._bootstrap_ci(baseline, treatment, test)
        else:
            ci = self._normal_ci(baseline, treatment, test)

        # Apply correction if needed
        p_corrected = None  # Only relevant for multiple comparisons

        # Determine significance
        is_significant = p_value < test.alpha

        # Effect size
        effect_size = self._cohens_d(baseline, treatment)

        return ComparisonResult(
            test_type=test.test_type,
            statistic=statistic,
            p_value=p_value,
            p_value_corrected=p_corrected,
            is_significant=is_significant,
            confidence_interval=ci,
            n_baseline=len(baseline),
            n_treatment=len(treatment),
            mean_baseline=mean_baseline,
            mean_treatment=mean_treatment,
            std_baseline=std_baseline,
            std_treatment=std_treatment,
            mean_difference=mean_treatment - mean_baseline,
            effect_size=effect_size,
        )

    def effect_size(
        self,
        baseline: list[float] | np.ndarray,
        treatment: list[float] | np.ndarray,
    ) -> EffectSizeResult:
        """Calculate effect sizes.

        Args:
            baseline: Baseline values.
            treatment: Treatment values.

        Returns:
            EffectSizeResult with various effect size measures.

        """
        baseline = np.array(baseline)
        treatment = np.array(treatment)

        # Cohen's d
        cohens_d = self._cohens_d(baseline, treatment)

        # Hedges' g (corrected for small sample sizes)
        n = len(baseline) + len(treatment)
        hedges_g = cohens_d * (1 - 3 / (4 * n - 9))

        # Cliff's delta (non-parametric)
        cliff_delta = self._cliff_delta(baseline, treatment)

        # Interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return EffectSizeResult(
            cohens_d=cohens_d,
            hedges_g=hedges_g,
            cliff_delta=cliff_delta,
            interpretation=interpretation,
        )

    def _t_test(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
        test: SignificanceTest,
    ) -> tuple[float, float]:
        """Perform independent samples t-test.

        Args:
            baseline: Baseline values.
            treatment: Treatment values.
            test: Test configuration.

        Returns:
            Tuple of (statistic, p-value).

        """
        try:
            from scipy import stats

            result = stats.ttest_ind(
                baseline,
                treatment,
                alternative=test.alternative,
            )
            return result.statistic, result.pvalue
        except ImportError:
            # Fallback to manual calculation
            return self._manual_t_test(baseline, treatment)

    def _manual_t_test(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
    ) -> tuple[float, float]:
        """Manual t-test calculation.

        Args:
            baseline: Baseline values.
            treatment: Treatment values.

        Returns:
            Tuple of (statistic, p-value).

        """
        n1, n2 = len(baseline), len(treatment)
        mean1, mean2 = np.mean(baseline), np.mean(treatment)
        var1, var2 = np.var(baseline, ddof=1), np.var(treatment, ddof=1)

        # Pooled standard error
        se = np.sqrt(var1 / n1 + var2 / n2)

        # T-statistic
        t_stat = (mean1 - mean2) / se

        # Degrees of freedom (Welch-Satterthwaite)
        df = (var1 / n1 + var2 / n2) ** 2 / (
            (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        )

        # P-value (approximation)
        try:
            from scipy import stats

            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        except ImportError:
            # Very rough approximation
            p_value = 2 * np.exp(-0.5 * t_stat**2) if abs(t_stat) < 3 else 0.001

        return t_stat, p_value

    def _mann_whitney(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
        test: SignificanceTest,
    ) -> tuple[float, float]:
        """Perform Mann-Whitney U test.

        Args:
            baseline: Baseline values.
            treatment: Treatment values.
            test: Test configuration.

        Returns:
            Tuple of (statistic, p-value).

        """
        try:
            from scipy import stats

            result = stats.mannwhitneyu(
                baseline,
                treatment,
                alternative=test.alternative,
            )
            return result.statistic, result.pvalue
        except ImportError:
            self._logger.warning(
                "scipy_not_available",
                message="Falling back to bootstrap test",
            )
            return self._bootstrap_test(baseline, treatment, test)[:2]

    def _bootstrap_test(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
        test: SignificanceTest,
    ) -> tuple[float, float, tuple[float, float]]:
        """Perform bootstrap hypothesis test.

        Args:
            baseline: Baseline values.
            treatment: Treatment values.
            test: Test configuration.

        Returns:
            Tuple of (observed_diff, p-value, confidence_interval).

        """
        observed_diff = np.mean(treatment) - np.mean(baseline)

        # Combine for null hypothesis
        combined = np.concatenate([baseline, treatment])
        n_baseline = len(baseline)

        # Bootstrap under null
        bootstrap_diffs = []
        for _ in range(test.n_bootstrap):
            np.random.shuffle(combined)
            boot_baseline = combined[:n_baseline]
            boot_treatment = combined[n_baseline:]
            bootstrap_diffs.append(np.mean(boot_treatment) - np.mean(boot_baseline))

        bootstrap_diffs = np.array(bootstrap_diffs)

        # P-value
        if test.alternative == "two-sided":
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        elif test.alternative == "greater":
            p_value = np.mean(bootstrap_diffs >= observed_diff)
        else:  # less
            p_value = np.mean(bootstrap_diffs <= observed_diff)

        # Confidence interval
        ci = self._bootstrap_ci(baseline, treatment, test)

        return observed_diff, p_value, ci

    def _permutation_test(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
        test: SignificanceTest,
    ) -> tuple[float, float]:
        """Perform permutation test.

        Args:
            baseline: Baseline values.
            treatment: Treatment values.
            test: Test configuration.

        Returns:
            Tuple of (statistic, p-value).

        """
        observed_diff = np.mean(treatment) - np.mean(baseline)
        combined = np.concatenate([baseline, treatment])
        n_baseline = len(baseline)

        n_permutations = min(test.n_bootstrap, 10000)
        count_extreme = 0

        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_baseline = combined[:n_baseline]
            perm_treatment = combined[n_baseline:]
            perm_diff = np.mean(perm_treatment) - np.mean(perm_baseline)

            if test.alternative == "two-sided":
                if abs(perm_diff) >= abs(observed_diff):
                    count_extreme += 1
            elif test.alternative == "greater":
                if perm_diff >= observed_diff:
                    count_extreme += 1
            else:
                if perm_diff <= observed_diff:
                    count_extreme += 1

        p_value = count_extreme / n_permutations

        return observed_diff, p_value

    def _bootstrap_ci(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
        test: SignificanceTest,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for mean difference.

        Args:
            baseline: Baseline values.
            treatment: Treatment values.
            test: Test configuration.

        Returns:
            Tuple of (lower, upper) bounds.

        """
        bootstrap_diffs = []

        for _ in range(test.n_bootstrap):
            boot_baseline = np.random.choice(baseline, size=len(baseline), replace=True)
            boot_treatment = np.random.choice(treatment, size=len(treatment), replace=True)
            bootstrap_diffs.append(np.mean(boot_treatment) - np.mean(boot_baseline))

        alpha = 1 - test.confidence_level
        lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

        return (lower, upper)

    def _normal_ci(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
        test: SignificanceTest,
    ) -> tuple[float, float]:
        """Compute normal-based confidence interval.

        Args:
            baseline: Baseline values.
            treatment: Treatment values.
            test: Test configuration.

        Returns:
            Tuple of (lower, upper) bounds.

        """
        mean_diff = np.mean(treatment) - np.mean(baseline)
        se = np.sqrt(
            np.var(baseline, ddof=1) / len(baseline) + np.var(treatment, ddof=1) / len(treatment)
        )

        # Z-score for confidence level
        try:
            from scipy import stats

            z = stats.norm.ppf(1 - (1 - test.confidence_level) / 2)
        except ImportError:
            z = 1.96  # Approximation for 95%

        return (mean_diff - z * se, mean_diff + z * se)

    def _cohens_d(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
    ) -> float:
        """Calculate Cohen's d effect size.

        Args:
            baseline: Baseline values.
            treatment: Treatment values.

        Returns:
            Cohen's d value.

        """
        n1, n2 = len(baseline), len(treatment)
        var1, var2 = np.var(baseline, ddof=1), np.var(treatment, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(treatment) - np.mean(baseline)) / pooled_std

    def _cliff_delta(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
    ) -> float:
        """Calculate Cliff's delta (non-parametric effect size).

        Args:
            baseline: Baseline values.
            treatment: Treatment values.

        Returns:
            Cliff's delta value in [-1, 1].

        """
        n1, n2 = len(baseline), len(treatment)
        count = 0

        for x in treatment:
            for y in baseline:
                if x > y:
                    count += 1
                elif x < y:
                    count -= 1

        return count / (n1 * n2)

    def apply_correction(
        self,
        p_values: list[float],
        method: str = "bonferroni",
    ) -> list[float]:
        """Apply multiple comparison correction.

        Args:
            p_values: List of p-values.
            method: Correction method.

        Returns:
            Corrected p-values.

        """
        n = len(p_values)

        if method == "none":
            return p_values

        elif method == "bonferroni":
            return [min(1.0, p * n) for p in p_values]

        elif method == "holm":
            # Holm-Bonferroni step-down
            sorted_idx = np.argsort(p_values)
            corrected = [0.0] * n

            for i, idx in enumerate(sorted_idx):
                corrected[idx] = min(1.0, p_values[idx] * (n - i))

            # Enforce monotonicity
            for i in range(1, n):
                idx = sorted_idx[i]
                prev_idx = sorted_idx[i - 1]
                corrected[idx] = max(corrected[idx], corrected[prev_idx])

            return corrected

        elif method == "fdr":
            # Benjamini-Hochberg
            sorted_idx = np.argsort(p_values)
            corrected = [0.0] * n

            for i, idx in enumerate(sorted_idx):
                corrected[idx] = p_values[idx] * n / (i + 1)

            # Enforce monotonicity (reverse direction)
            for i in range(n - 2, -1, -1):
                idx = sorted_idx[i]
                next_idx = sorted_idx[i + 1]
                corrected[idx] = min(corrected[idx], corrected[next_idx])

            return [min(1.0, p) for p in corrected]

        else:
            raise ValueError(f"Unknown correction method: {method}")


def create_analyzer(
    test_type: str = "bootstrap",
    alpha: float = 0.05,
    **kwargs: Any,
) -> StatisticalAnalyzer:
    """Factory function to create statistical analyzer.

    Args:
        test_type: Type of statistical test.
        alpha: Significance level.
        **kwargs: Additional test configuration.

    Returns:
        Configured StatisticalAnalyzer instance.

    """
    config = SignificanceTest(test_type=test_type, alpha=alpha, **kwargs)
    return StatisticalAnalyzer(test_config=config)
