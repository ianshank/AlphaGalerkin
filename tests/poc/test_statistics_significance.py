"""Tests for statistical significance testing utilities.

Validates:
    - T-test with known distributions
    - Mann-Whitney U test
    - Cohen's d effect size
    - Hedges' g and Cliff's delta
    - Multiple comparison corrections (Bonferroni, Holm, FDR)
    - Bootstrap confidence intervals
    - StatisticalAnalyzer.compare_runs result structure
"""

from __future__ import annotations

import numpy as np
import pytest

from src.poc.statistics.significance import (
    ComparisonResult,
    EffectSizeResult,
    SignificanceTest,
    StatisticalAnalyzer,
    create_analyzer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(SEED)


@pytest.fixture()
def identical_samples(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two samples drawn from the same distribution (no effect)."""
    data = rng.normal(loc=5.0, scale=1.0, size=50)
    return data[:25], data[25:]


@pytest.fixture()
def different_samples(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two samples from clearly different distributions (large effect)."""
    baseline = rng.normal(loc=0.0, scale=1.0, size=50)
    treatment = rng.normal(loc=3.0, scale=1.0, size=50)
    return baseline, treatment


@pytest.fixture()
def small_effect_samples(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two samples with a small but present effect."""
    baseline = rng.normal(loc=0.0, scale=1.0, size=100)
    treatment = rng.normal(loc=0.3, scale=1.0, size=100)
    return baseline, treatment


@pytest.fixture()
def default_analyzer() -> StatisticalAnalyzer:
    """Analyzer with default configuration."""
    return StatisticalAnalyzer()


@pytest.fixture()
def t_test_analyzer() -> StatisticalAnalyzer:
    """Analyzer configured for t-test."""
    config = SignificanceTest(test_type="t_test", alpha=0.05)
    return StatisticalAnalyzer(test_config=config)


@pytest.fixture()
def mann_whitney_analyzer() -> StatisticalAnalyzer:
    """Analyzer configured for Mann-Whitney."""
    config = SignificanceTest(test_type="mann_whitney", alpha=0.05)
    return StatisticalAnalyzer(test_config=config)


@pytest.fixture()
def bootstrap_analyzer() -> StatisticalAnalyzer:
    """Analyzer configured for bootstrap test with fewer iterations for speed."""
    config = SignificanceTest(
        test_type="bootstrap",
        alpha=0.05,
        n_bootstrap=2000,
    )
    return StatisticalAnalyzer(test_config=config)


# ---------------------------------------------------------------------------
# SignificanceTest config validation
# ---------------------------------------------------------------------------


class TestSignificanceTestConfig:
    """Tests for SignificanceTest Pydantic model."""

    def test_defaults(self) -> None:
        config = SignificanceTest()
        assert config.test_type == "bootstrap"
        assert config.alpha == 0.05
        assert config.correction == "bonferroni"
        assert config.alternative == "two-sided"
        assert config.n_bootstrap == 10000
        assert config.confidence_level == 0.95

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.5])
    def test_valid_alpha(self, alpha: float) -> None:
        config = SignificanceTest(alpha=alpha)
        assert config.alpha == alpha

    def test_alpha_zero_invalid(self) -> None:
        with pytest.raises(Exception):
            SignificanceTest(alpha=0.0)

    def test_alpha_one_invalid(self) -> None:
        with pytest.raises(Exception):
            SignificanceTest(alpha=1.0)

    def test_n_bootstrap_minimum(self) -> None:
        with pytest.raises(Exception):
            SignificanceTest(n_bootstrap=500)


# ---------------------------------------------------------------------------
# T-test
# ---------------------------------------------------------------------------


class TestTTest:
    """Tests for t-test via StatisticalAnalyzer."""

    def test_identical_distributions_not_significant(
        self,
        t_test_analyzer: StatisticalAnalyzer,
        identical_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Samples from the same distribution should not be significant."""
        baseline, treatment = identical_samples
        result = t_test_analyzer.compare_runs(baseline, treatment)
        assert result.p_value > 0.05
        assert not result.is_significant

    def test_different_distributions_significant(
        self,
        t_test_analyzer: StatisticalAnalyzer,
        different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Clearly different distributions should be significant."""
        baseline, treatment = different_samples
        result = t_test_analyzer.compare_runs(baseline, treatment)
        assert result.p_value < 0.01
        assert result.is_significant

    def test_result_structure(
        self,
        t_test_analyzer: StatisticalAnalyzer,
        different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Result should contain all expected fields."""
        baseline, treatment = different_samples
        result = t_test_analyzer.compare_runs(baseline, treatment)

        assert isinstance(result, ComparisonResult)
        assert result.test_type == "t_test"
        assert result.n_baseline == len(baseline)
        assert result.n_treatment == len(treatment)
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.effect_size is not None

    def test_mean_difference_sign(
        self,
        t_test_analyzer: StatisticalAnalyzer,
    ) -> None:
        """mean_difference should be treatment - baseline."""
        np.random.seed(SEED)
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        treatment = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        result = t_test_analyzer.compare_runs(baseline, treatment)
        assert result.mean_difference > 0
        assert result.mean_difference == pytest.approx(
            np.mean(treatment) - np.mean(baseline)
        )


# ---------------------------------------------------------------------------
# Mann-Whitney U test
# ---------------------------------------------------------------------------


class TestMannWhitney:
    """Tests for Mann-Whitney U test."""

    def test_identical_distributions_not_significant(
        self,
        mann_whitney_analyzer: StatisticalAnalyzer,
        identical_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Same-distribution samples should not be significant."""
        baseline, treatment = identical_samples
        result = mann_whitney_analyzer.compare_runs(baseline, treatment)
        assert result.p_value > 0.05

    def test_different_distributions_significant(
        self,
        mann_whitney_analyzer: StatisticalAnalyzer,
        different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Clearly separated distributions should be significant."""
        baseline, treatment = different_samples
        result = mann_whitney_analyzer.compare_runs(baseline, treatment)
        assert result.p_value < 0.01
        assert result.is_significant

    def test_result_type(
        self,
        mann_whitney_analyzer: StatisticalAnalyzer,
        different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        result = mann_whitney_analyzer.compare_runs(*different_samples)
        assert result.test_type == "mann_whitney"


# ---------------------------------------------------------------------------
# Bootstrap test
# ---------------------------------------------------------------------------


class TestBootstrapTest:
    """Tests for bootstrap hypothesis test."""

    def test_different_distributions_significant(
        self,
        bootstrap_analyzer: StatisticalAnalyzer,
        different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Clearly different distributions should be detected."""
        np.random.seed(SEED)
        baseline, treatment = different_samples
        result = bootstrap_analyzer.compare_runs(baseline, treatment)
        assert result.p_value < 0.05
        assert result.is_significant

    def test_confidence_interval_contains_diff(
        self,
        bootstrap_analyzer: StatisticalAnalyzer,
        different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """CI should contain the observed mean difference (approximately)."""
        np.random.seed(SEED)
        baseline, treatment = different_samples
        result = bootstrap_analyzer.compare_runs(baseline, treatment)

        assert result.confidence_interval is not None
        lower, upper = result.confidence_interval
        assert lower < upper
        # The true effect is ~3.0; CI should bracket it
        assert lower > 0  # treatment > baseline


# ---------------------------------------------------------------------------
# Cohen's d effect size
# ---------------------------------------------------------------------------


class TestCohensD:
    """Tests for Cohen's d effect size computation."""

    def test_no_effect(
        self,
        default_analyzer: StatisticalAnalyzer,
        identical_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Identical distributions should yield negligible effect size."""
        baseline, treatment = identical_samples
        es = default_analyzer.effect_size(baseline, treatment)
        assert abs(es.cohens_d) < 0.5

    def test_large_effect(
        self,
        default_analyzer: StatisticalAnalyzer,
        different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Clearly separated distributions should yield large effect."""
        baseline, treatment = different_samples
        es = default_analyzer.effect_size(baseline, treatment)
        assert abs(es.cohens_d) > 0.8
        assert es.interpretation == "large"

    def test_known_effect_size(self, default_analyzer: StatisticalAnalyzer) -> None:
        """Test with distributions having known Cohen's d ~ 1.0."""
        np.random.seed(SEED)
        baseline = np.zeros(1000)  # mean=0, sd=0 won't work; use normal
        baseline = np.random.normal(0, 1, 1000)
        treatment = np.random.normal(1, 1, 1000)

        es = default_analyzer.effect_size(baseline, treatment)
        assert abs(es.cohens_d - 1.0) < 0.2  # approximate

    def test_effect_size_sign(self, default_analyzer: StatisticalAnalyzer) -> None:
        """Cohen's d should be positive when treatment > baseline."""
        baseline = np.array([1.0, 2.0, 3.0])
        treatment = np.array([4.0, 5.0, 6.0])
        es = default_analyzer.effect_size(baseline, treatment)
        assert es.cohens_d > 0

    def test_zero_variance_returns_zero(
        self, default_analyzer: StatisticalAnalyzer
    ) -> None:
        """Constant samples should return d=0."""
        baseline = np.array([5.0, 5.0, 5.0])
        treatment = np.array([5.0, 5.0, 5.0])
        es = default_analyzer.effect_size(baseline, treatment)
        assert es.cohens_d == 0.0


# ---------------------------------------------------------------------------
# Hedges' g and Cliff's delta
# ---------------------------------------------------------------------------


class TestEffectSizes:
    """Tests for Hedges' g and Cliff's delta."""

    def test_hedges_g_smaller_than_cohens_d(
        self,
        default_analyzer: StatisticalAnalyzer,
        different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Hedges' g applies small-sample correction, should be slightly smaller."""
        baseline, treatment = different_samples
        es = default_analyzer.effect_size(baseline, treatment)
        # Hedges' g = d * (1 - 3/(4n-9)), so |g| < |d| for finite n
        assert abs(es.hedges_g) <= abs(es.cohens_d)

    def test_cliff_delta_range(
        self,
        default_analyzer: StatisticalAnalyzer,
        different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Cliff's delta should be in [-1, 1]."""
        baseline, treatment = different_samples
        es = default_analyzer.effect_size(baseline, treatment)
        assert es.cliff_delta is not None
        assert -1.0 <= es.cliff_delta <= 1.0

    def test_cliff_delta_sign(self, default_analyzer: StatisticalAnalyzer) -> None:
        """Positive cliff delta when treatment is systematically larger."""
        baseline = np.array([1.0, 2.0, 3.0])
        treatment = np.array([4.0, 5.0, 6.0])
        es = default_analyzer.effect_size(baseline, treatment)
        assert es.cliff_delta is not None
        assert es.cliff_delta > 0

    @pytest.mark.parametrize(
        "d_expected,label",
        [
            (0.1, "negligible"),
            (0.3, "small"),
            (0.6, "medium"),
            (1.5, "large"),
        ],
    )
    def test_effect_interpretation(
        self,
        default_analyzer: StatisticalAnalyzer,
        d_expected: float,
        label: str,
    ) -> None:
        """Effect size interpretation should match standard thresholds."""
        np.random.seed(SEED)
        n = 10000
        baseline = np.random.normal(0, 1, n)
        treatment = np.random.normal(d_expected, 1, n)
        es = default_analyzer.effect_size(baseline, treatment)
        assert es.interpretation == label

    def test_effect_size_result_fields(
        self, default_analyzer: StatisticalAnalyzer
    ) -> None:
        """EffectSizeResult should contain all expected fields."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([2.0, 3.0, 4.0, 5.0])
        es = default_analyzer.effect_size(baseline, treatment)

        assert isinstance(es, EffectSizeResult)
        assert isinstance(es.cohens_d, float)
        assert isinstance(es.hedges_g, float)
        assert isinstance(es.cliff_delta, float)
        assert es.interpretation in ("negligible", "small", "medium", "large")


# ---------------------------------------------------------------------------
# Multiple comparison corrections
# ---------------------------------------------------------------------------


class TestMultipleComparisons:
    """Tests for apply_correction."""

    def test_bonferroni(self, default_analyzer: StatisticalAnalyzer) -> None:
        """Bonferroni: p_corrected = p * n, capped at 1.0."""
        p_values = [0.01, 0.04, 0.06]
        corrected = default_analyzer.apply_correction(p_values, method="bonferroni")

        assert len(corrected) == 3
        assert corrected[0] == pytest.approx(0.03)  # 0.01 * 3
        assert corrected[1] == pytest.approx(0.12)  # 0.04 * 3
        assert corrected[2] == pytest.approx(0.18)  # 0.06 * 3

    def test_bonferroni_caps_at_one(
        self, default_analyzer: StatisticalAnalyzer
    ) -> None:
        """Bonferroni-corrected values should not exceed 1.0."""
        p_values = [0.5, 0.8]
        corrected = default_analyzer.apply_correction(p_values, method="bonferroni")

        assert all(p <= 1.0 for p in corrected)
        assert corrected[0] == 1.0  # 0.5 * 2 = 1.0
        assert corrected[1] == 1.0  # 0.8 * 2 = 1.6 -> capped at 1.0

    def test_holm_correction(self, default_analyzer: StatisticalAnalyzer) -> None:
        """Holm step-down should be less conservative than Bonferroni."""
        p_values = [0.01, 0.04, 0.06]
        bonf = default_analyzer.apply_correction(p_values, method="bonferroni")
        holm = default_analyzer.apply_correction(p_values, method="holm")

        # Holm is uniformly <= Bonferroni
        for h, b in zip(holm, bonf, strict=False):
            assert h <= b + 1e-10

    def test_holm_monotonicity(self, default_analyzer: StatisticalAnalyzer) -> None:
        """Holm-corrected values should be monotone non-decreasing when sorted."""
        p_values = [0.001, 0.01, 0.03, 0.05, 0.1]
        corrected = default_analyzer.apply_correction(p_values, method="holm")

        sorted_indices = np.argsort(p_values)
        sorted_corrected = [corrected[i] for i in sorted_indices]
        for i in range(len(sorted_corrected) - 1):
            assert sorted_corrected[i] <= sorted_corrected[i + 1] + 1e-10

    def test_fdr_correction(self, default_analyzer: StatisticalAnalyzer) -> None:
        """FDR (Benjamini-Hochberg) should be less conservative than Bonferroni."""
        p_values = [0.01, 0.04, 0.06]
        bonf = default_analyzer.apply_correction(p_values, method="bonferroni")
        fdr = default_analyzer.apply_correction(p_values, method="fdr")

        for f, b in zip(fdr, bonf, strict=False):
            assert f <= b + 1e-10

    def test_fdr_caps_at_one(self, default_analyzer: StatisticalAnalyzer) -> None:
        """FDR-corrected values should not exceed 1.0."""
        p_values = [0.5, 0.8, 0.9]
        corrected = default_analyzer.apply_correction(p_values, method="fdr")
        assert all(p <= 1.0 for p in corrected)

    def test_none_correction(self, default_analyzer: StatisticalAnalyzer) -> None:
        """'none' correction should return p-values unchanged."""
        p_values = [0.01, 0.05, 0.1]
        corrected = default_analyzer.apply_correction(p_values, method="none")
        assert corrected == p_values

    def test_unknown_correction_raises(
        self, default_analyzer: StatisticalAnalyzer
    ) -> None:
        """Unknown correction method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown correction"):
            default_analyzer.apply_correction([0.05], method="sidak")

    def test_single_p_value(self, default_analyzer: StatisticalAnalyzer) -> None:
        """Single p-value should be corrected correctly."""
        corrected = default_analyzer.apply_correction([0.03], method="bonferroni")
        assert corrected == [pytest.approx(0.03)]  # n=1, so p*1 = p


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_ci_contains_true_diff(
        self,
        bootstrap_analyzer: StatisticalAnalyzer,
        different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """CI should contain the true mean difference for large effects."""
        np.random.seed(SEED)
        baseline, treatment = different_samples
        result = bootstrap_analyzer.compare_runs(baseline, treatment)

        assert result.confidence_interval is not None
        lower, upper = result.confidence_interval
        # True difference is ~3.0
        assert lower < 3.5
        assert upper > 2.5

    def test_ci_width_decreases_with_sample_size(
        self,
        bootstrap_analyzer: StatisticalAnalyzer,
    ) -> None:
        """Larger samples should yield narrower CIs."""
        np.random.seed(SEED)
        small_baseline = np.random.normal(0, 1, 10)
        small_treatment = np.random.normal(1, 1, 10)
        result_small = bootstrap_analyzer.compare_runs(small_baseline, small_treatment)

        np.random.seed(SEED)
        large_baseline = np.random.normal(0, 1, 200)
        large_treatment = np.random.normal(1, 1, 200)
        result_large = bootstrap_analyzer.compare_runs(large_baseline, large_treatment)

        assert result_small.confidence_interval is not None
        assert result_large.confidence_interval is not None

        width_small = result_small.confidence_interval[1] - result_small.confidence_interval[0]
        width_large = result_large.confidence_interval[1] - result_large.confidence_interval[0]

        assert width_large < width_small


# ---------------------------------------------------------------------------
# create_analyzer factory
# ---------------------------------------------------------------------------


class TestCreateAnalyzer:
    """Tests for create_analyzer factory function."""

    def test_default(self) -> None:
        analyzer = create_analyzer()
        assert analyzer.test_config.test_type == "bootstrap"
        assert analyzer.test_config.alpha == 0.05

    @pytest.mark.parametrize("test_type", ["t_test", "mann_whitney", "bootstrap", "permutation"])
    def test_various_types(self, test_type: str) -> None:
        analyzer = create_analyzer(test_type=test_type)
        assert analyzer.test_config.test_type == test_type

    def test_custom_alpha(self) -> None:
        analyzer = create_analyzer(alpha=0.01)
        assert analyzer.test_config.alpha == 0.01

    def test_extra_kwargs(self) -> None:
        analyzer = create_analyzer(
            test_type="bootstrap",
            n_bootstrap=5000,
            confidence_level=0.99,
        )
        assert analyzer.test_config.n_bootstrap == 5000
        assert analyzer.test_config.confidence_level == 0.99


# ---------------------------------------------------------------------------
# compare_runs result structure
# ---------------------------------------------------------------------------


class TestCompareRunsResult:
    """Tests verifying ComparisonResult field values."""

    def test_accepts_lists(self, t_test_analyzer: StatisticalAnalyzer) -> None:
        """compare_runs should accept plain Python lists."""
        result = t_test_analyzer.compare_runs([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert isinstance(result, ComparisonResult)

    def test_accepts_arrays(self, t_test_analyzer: StatisticalAnalyzer) -> None:
        """compare_runs should accept numpy arrays."""
        result = t_test_analyzer.compare_runs(
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        )
        assert isinstance(result, ComparisonResult)

    def test_statistics_computed(
        self,
        t_test_analyzer: StatisticalAnalyzer,
        different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """All summary statistics should be populated."""
        baseline, treatment = different_samples
        result = t_test_analyzer.compare_runs(baseline, treatment)

        assert result.mean_baseline == pytest.approx(np.mean(baseline))
        assert result.mean_treatment == pytest.approx(np.mean(treatment))
        assert result.std_baseline == pytest.approx(np.std(baseline, ddof=1))
        assert result.std_treatment == pytest.approx(np.std(treatment, ddof=1))
