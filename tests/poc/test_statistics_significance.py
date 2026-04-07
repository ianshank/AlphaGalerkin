"""Tests for statistical significance testing utilities.

Validates:
    - SignificanceTest Pydantic configuration and constraints
    - ComparisonResult and EffectSizeResult dataclasses
    - StatisticalAnalyzer.compare_runs with all test types
    - StatisticalAnalyzer.effect_size calculations
    - Multiple comparison correction methods
    - Bootstrap and normal confidence intervals
    - create_analyzer factory function
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from src.poc.statistics.significance import (
    ComparisonResult,
    EffectSizeResult,
    SignificanceTest,
    StatisticalAnalyzer,
    create_analyzer,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG_SEED = 42
N_SAMPLES = 30
N_BOOTSTRAP_FAST = 1000  # minimum allowed by SignificanceTest; keep tests reasonably fast


@pytest.fixture()
def rng() -> np.random.Generator:
    """Return a seeded random generator for reproducible data."""
    return np.random.default_rng(RNG_SEED)


@pytest.fixture()
def identical_samples(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two samples drawn from the same distribution (baseline == treatment)."""
    baseline = rng.normal(0.0, 1.0, N_SAMPLES)
    treatment = rng.normal(0.0, 1.0, N_SAMPLES)
    return baseline, treatment


@pytest.fixture()
def clearly_different_samples(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Samples with a large, obvious mean separation (d ~ 5)."""
    baseline = rng.normal(0.0, 0.5, N_SAMPLES)
    treatment = rng.normal(5.0, 0.5, N_SAMPLES)
    return baseline, treatment


@pytest.fixture()
def slightly_different_samples(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Samples with a small mean difference (d ~ 0.3)."""
    baseline = rng.normal(0.0, 1.0, N_SAMPLES)
    treatment = rng.normal(0.3, 1.0, N_SAMPLES)
    return baseline, treatment


@pytest.fixture()
def fast_bootstrap_config() -> SignificanceTest:
    """SignificanceTest config with small n_bootstrap for speed."""
    return SignificanceTest(n_bootstrap=N_BOOTSTRAP_FAST)


@pytest.fixture()
def analyzer_fast(fast_bootstrap_config: SignificanceTest) -> StatisticalAnalyzer:
    """StatisticalAnalyzer using the fast bootstrap config."""
    return StatisticalAnalyzer(test_config=fast_bootstrap_config)


# ---------------------------------------------------------------------------
# 1. TestSignificanceTestConfig
# ---------------------------------------------------------------------------


class TestSignificanceTestConfig:
    """Tests for SignificanceTest Pydantic configuration model."""

    def test_default_values(self) -> None:
        """Default config should use bootstrap, alpha=0.05, and bonferroni."""
        config = SignificanceTest()
        assert config.test_type == "bootstrap"
        assert config.alpha == pytest.approx(0.05)
        assert config.correction == "bonferroni"
        assert config.alternative == "two-sided"
        assert config.n_bootstrap == 10000
        assert config.confidence_level == pytest.approx(0.95)

    def test_alpha_must_be_positive(self) -> None:
        """alpha <= 0 is invalid."""
        with pytest.raises(ValidationError):
            SignificanceTest(alpha=0.0)

    def test_alpha_must_be_less_than_one(self) -> None:
        """alpha >= 1 is invalid."""
        with pytest.raises(ValidationError):
            SignificanceTest(alpha=1.0)

    def test_alpha_valid_range(self) -> None:
        """Arbitrary valid alpha values in (0, 1) should be accepted."""
        for alpha in (0.001, 0.01, 0.05, 0.1, 0.5, 0.99):
            config = SignificanceTest(alpha=alpha)
            assert config.alpha == pytest.approx(alpha)

    def test_test_type_options(self) -> None:
        """All four test types should be valid literals."""
        for test_type in ("t_test", "mann_whitney", "bootstrap", "permutation"):
            config = SignificanceTest(test_type=test_type)
            assert config.test_type == test_type

    def test_invalid_test_type_rejected(self) -> None:
        """Unknown test type should raise a ValidationError."""
        with pytest.raises(ValidationError):
            SignificanceTest(test_type="chi_square")  # type: ignore[arg-type]

    def test_correction_options(self) -> None:
        """All four correction methods should be valid literals."""
        for correction in ("none", "bonferroni", "holm", "fdr"):
            config = SignificanceTest(correction=correction)
            assert config.correction == correction

    def test_alternative_options(self) -> None:
        """All three alternative hypotheses should be valid."""
        for alt in ("two-sided", "less", "greater"):
            config = SignificanceTest(alternative=alt)
            assert config.alternative == alt

    def test_n_bootstrap_minimum(self) -> None:
        """n_bootstrap must be >= 1000."""
        with pytest.raises(ValidationError):
            SignificanceTest(n_bootstrap=999)

    def test_n_bootstrap_valid(self) -> None:
        """n_bootstrap at the boundary (1000) and above should be accepted."""
        config = SignificanceTest(n_bootstrap=1000)
        assert config.n_bootstrap == 1000

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should be rejected due to model_config extra='forbid'."""
        with pytest.raises(ValidationError):
            SignificanceTest(unknown_field=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 2. TestComparisonResult
# ---------------------------------------------------------------------------


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_instantiation_with_required_fields(self) -> None:
        """ComparisonResult can be constructed with all required fields."""
        result = ComparisonResult(
            test_type="t_test",
            statistic=2.5,
            p_value=0.01,
            p_value_corrected=None,
            is_significant=True,
            confidence_interval=(-5.0, -0.5),
            n_baseline=20,
            n_treatment=20,
            mean_baseline=0.0,
            mean_treatment=2.0,
            std_baseline=1.0,
            std_treatment=1.0,
            mean_difference=2.0,
        )
        assert result.test_type == "t_test"
        assert result.p_value == pytest.approx(0.01)
        assert result.is_significant is True

    def test_is_significant_reflects_p_value(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """is_significant should match whether p_value < alpha."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer(
            test_config=SignificanceTest(test_type="t_test", alpha=0.05)
        )
        result = analyzer.compare_runs(baseline, treatment)
        assert result.is_significant == (result.p_value < 0.05)

    def test_confidence_interval_optional(self) -> None:
        """confidence_interval field may be None."""
        result = ComparisonResult(
            test_type="t_test",
            statistic=1.0,
            p_value=0.3,
            p_value_corrected=None,
            is_significant=False,
            confidence_interval=None,
            n_baseline=10,
            n_treatment=10,
            mean_baseline=0.0,
            mean_treatment=0.1,
            std_baseline=1.0,
            std_treatment=1.0,
            mean_difference=0.1,
        )
        assert result.confidence_interval is None

    def test_effect_size_optional_default_none(self) -> None:
        """effect_size defaults to None when not provided."""
        result = ComparisonResult(
            test_type="bootstrap",
            statistic=0.5,
            p_value=0.2,
            p_value_corrected=None,
            is_significant=False,
            confidence_interval=(-0.1, 1.1),
            n_baseline=15,
            n_treatment=15,
            mean_baseline=0.0,
            mean_treatment=0.5,
            std_baseline=1.0,
            std_treatment=1.0,
            mean_difference=0.5,
        )
        assert result.effect_size is None


# ---------------------------------------------------------------------------
# 3. TestStatisticalAnalyzerCompareRuns
# ---------------------------------------------------------------------------


class TestStatisticalAnalyzerCompareRuns:
    """Tests for StatisticalAnalyzer.compare_runs across all test types."""

    @pytest.mark.parametrize("test_type", ["t_test", "mann_whitney", "bootstrap", "permutation"])
    def test_all_test_types_return_valid_result(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
        test_type: str,
    ) -> None:
        """Every test type should return a ComparisonResult with p_value in [0, 1]."""
        baseline, treatment = clearly_different_samples
        config = SignificanceTest(test_type=test_type, n_bootstrap=N_BOOTSTRAP_FAST)
        analyzer = StatisticalAnalyzer(test_config=config)
        result = analyzer.compare_runs(baseline, treatment)

        assert isinstance(result, ComparisonResult)
        assert 0.0 <= result.p_value <= 1.0
        assert result.test_type == test_type

    def test_clearly_different_is_significant(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
        fast_bootstrap_config: SignificanceTest,
    ) -> None:
        """Clearly separated distributions should yield is_significant=True."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer(test_config=fast_bootstrap_config)
        result = analyzer.compare_runs(baseline, treatment)
        assert result.is_significant

    def test_identical_not_significant(
        self,
        identical_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Samples from the same distribution should not be significant most of the time."""
        baseline, treatment = identical_samples
        config = SignificanceTest(test_type="t_test", alpha=0.05)
        analyzer = StatisticalAnalyzer(test_config=config)
        result = analyzer.compare_runs(baseline, treatment)
        # With a seeded RNG the p_value should be well above 0.05.
        assert result.p_value > 0.05

    def test_sample_counts_are_correct(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
        fast_bootstrap_config: SignificanceTest,
    ) -> None:
        """n_baseline and n_treatment should reflect actual input lengths."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer(test_config=fast_bootstrap_config)
        result = analyzer.compare_runs(baseline, treatment)
        assert result.n_baseline == len(baseline)
        assert result.n_treatment == len(treatment)

    def test_mean_difference_sign(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
        fast_bootstrap_config: SignificanceTest,
    ) -> None:
        """mean_difference should equal mean_treatment - mean_baseline."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer(test_config=fast_bootstrap_config)
        result = analyzer.compare_runs(baseline, treatment)
        expected = np.mean(treatment) - np.mean(baseline)
        assert result.mean_difference == pytest.approx(expected, abs=1e-6)

    def test_effect_size_included_in_result(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
        fast_bootstrap_config: SignificanceTest,
    ) -> None:
        """compare_runs should populate effect_size with Cohen's d."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer(test_config=fast_bootstrap_config)
        result = analyzer.compare_runs(baseline, treatment)
        assert result.effect_size is not None

    def test_confidence_interval_is_tuple(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
        fast_bootstrap_config: SignificanceTest,
    ) -> None:
        """compare_runs should always return a non-None confidence_interval."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer(test_config=fast_bootstrap_config)
        result = analyzer.compare_runs(baseline, treatment)
        assert result.confidence_interval is not None
        lower, upper = result.confidence_interval
        assert lower < upper

    def test_per_call_test_override(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """An explicit test argument should override the analyzer's default config."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer(test_config=SignificanceTest(test_type="t_test"))
        override = SignificanceTest(test_type="mann_whitney", n_bootstrap=N_BOOTSTRAP_FAST)
        result = analyzer.compare_runs(baseline, treatment, test=override)
        assert result.test_type == "mann_whitney"

    def test_accepts_list_inputs(
        self,
        fast_bootstrap_config: SignificanceTest,
    ) -> None:
        """compare_runs should work with plain Python lists, not just ndarrays."""
        baseline = [0.0, 0.1, -0.1, 0.05, -0.05] * 6
        treatment = [5.0, 4.9, 5.1, 5.05, 4.95] * 6
        analyzer = StatisticalAnalyzer(test_config=fast_bootstrap_config)
        result = analyzer.compare_runs(baseline, treatment)
        assert isinstance(result, ComparisonResult)
        assert result.is_significant


# ---------------------------------------------------------------------------
# 4. TestEffectSize
# ---------------------------------------------------------------------------


class TestEffectSize:
    """Tests for StatisticalAnalyzer.effect_size."""

    def test_effect_size_returns_correct_type(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """effect_size should return an EffectSizeResult instance."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer()
        result = analyzer.effect_size(baseline, treatment)
        assert isinstance(result, EffectSizeResult)

    def test_cohens_d_near_zero_for_identical(
        self,
        identical_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Cohen's d should be close to 0 when samples come from the same distribution."""
        baseline, treatment = identical_samples
        analyzer = StatisticalAnalyzer()
        result = analyzer.effect_size(baseline, treatment)
        assert abs(result.cohens_d) < 1.0  # both drawn from N(0,1), d << large

    def test_large_cohens_d_for_clearly_different(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Clearly separated distributions should yield a large Cohen's d (>0.5)."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer()
        result = analyzer.effect_size(baseline, treatment)
        assert abs(result.cohens_d) > 0.5

    def test_hedges_g_present(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """EffectSizeResult should have a finite Hedges' g value."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer()
        result = analyzer.effect_size(baseline, treatment)
        assert np.isfinite(result.hedges_g)

    def test_cliff_delta_present_and_bounded(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Cliff's delta should be in [-1, 1]."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer()
        result = analyzer.effect_size(baseline, treatment)
        assert result.cliff_delta is not None
        assert -1.0 <= result.cliff_delta <= 1.0

    def test_interpretation_for_large_effect(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Interpretation should be 'large' for a very clearly different distribution."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer()
        result = analyzer.effect_size(baseline, treatment)
        assert result.interpretation == "large"

    def test_interpretation_for_negligible_effect(self, rng: np.random.Generator) -> None:
        """Interpretation should be 'negligible' when |d| < 0.2."""
        baseline = rng.normal(0.0, 1.0, 50)
        treatment = baseline + rng.normal(0.0, 0.01, 50)  # tiny perturbation
        analyzer = StatisticalAnalyzer()
        result = analyzer.effect_size(baseline, treatment)
        assert result.interpretation == "negligible"

    def test_cohens_d_sign_reflects_direction(self, rng: np.random.Generator) -> None:
        """Cohen's d should be positive when treatment > baseline and vice-versa."""
        baseline = rng.normal(0.0, 0.5, 40)
        treatment_higher = rng.normal(3.0, 0.5, 40)
        treatment_lower = rng.normal(-3.0, 0.5, 40)

        analyzer = StatisticalAnalyzer()
        assert analyzer.effect_size(baseline, treatment_higher).cohens_d > 0
        assert analyzer.effect_size(baseline, treatment_lower).cohens_d < 0


# ---------------------------------------------------------------------------
# 5. TestMultipleComparisonCorrection
# ---------------------------------------------------------------------------


class TestMultipleComparisonCorrection:
    """Tests for StatisticalAnalyzer.apply_correction."""

    @pytest.fixture()
    def raw_p_values(self) -> list[float]:
        """A set of raw p-values for correction tests."""
        return [0.01, 0.04, 0.20, 0.50, 0.90]

    @pytest.fixture()
    def small_p_values(self) -> list[float]:
        """P-values that would overflow past 1.0 under naive Bonferroni."""
        return [0.5, 0.6, 0.7, 0.8]

    def test_none_correction_returns_original(
        self, raw_p_values: list[float]
    ) -> None:
        """'none' correction must return the original p-values unchanged."""
        analyzer = StatisticalAnalyzer()
        corrected = analyzer.apply_correction(raw_p_values, method="none")
        assert corrected == raw_p_values

    def test_bonferroni_multiplies_by_n(
        self, raw_p_values: list[float]
    ) -> None:
        """Bonferroni correction should multiply each p by n, capped at 1.0."""
        analyzer = StatisticalAnalyzer()
        n = len(raw_p_values)
        corrected = analyzer.apply_correction(raw_p_values, method="bonferroni")
        for original, corr in zip(raw_p_values, corrected):
            expected = min(1.0, original * n)
            assert corr == pytest.approx(expected)

    def test_bonferroni_never_exceeds_one(
        self, small_p_values: list[float]
    ) -> None:
        """Bonferroni corrected values must never exceed 1.0."""
        analyzer = StatisticalAnalyzer()
        corrected = analyzer.apply_correction(small_p_values, method="bonferroni")
        assert all(p <= 1.0 for p in corrected)

    def test_holm_returns_valid_p_values(
        self, raw_p_values: list[float]
    ) -> None:
        """Holm-Bonferroni correction should return values in [0, 1]."""
        analyzer = StatisticalAnalyzer()
        corrected = analyzer.apply_correction(raw_p_values, method="holm")
        assert len(corrected) == len(raw_p_values)
        assert all(0.0 <= p <= 1.0 for p in corrected)

    def test_holm_monotonicity(self, raw_p_values: list[float]) -> None:
        """Holm corrected values, when sorted by original rank, must be non-decreasing."""
        analyzer = StatisticalAnalyzer()
        corrected = analyzer.apply_correction(raw_p_values, method="holm")
        sorted_corrected = [corrected[i] for i in np.argsort(raw_p_values)]
        for prev, curr in zip(sorted_corrected, sorted_corrected[1:]):
            assert curr >= prev - 1e-12

    def test_fdr_returns_valid_p_values(
        self, raw_p_values: list[float]
    ) -> None:
        """FDR (Benjamini-Hochberg) correction should return values in [0, 1]."""
        analyzer = StatisticalAnalyzer()
        corrected = analyzer.apply_correction(raw_p_values, method="fdr")
        assert len(corrected) == len(raw_p_values)
        assert all(0.0 <= p <= 1.0 for p in corrected)

    def test_unknown_method_raises(self) -> None:
        """An unknown correction method should raise a ValueError."""
        analyzer = StatisticalAnalyzer()
        with pytest.raises(ValueError, match="Unknown correction method"):
            analyzer.apply_correction([0.05], method="sidak")

    def test_single_p_value_bonferroni(self) -> None:
        """Bonferroni on a single p-value should return the same p-value."""
        analyzer = StatisticalAnalyzer()
        corrected = analyzer.apply_correction([0.03], method="bonferroni")
        assert corrected == pytest.approx([0.03])


# ---------------------------------------------------------------------------
# 6. TestBootstrapCI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Tests for StatisticalAnalyzer bootstrap confidence intervals."""

    def test_bootstrap_ci_lower_less_than_upper(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
        fast_bootstrap_config: SignificanceTest,
    ) -> None:
        """Bootstrap CI lower bound must be strictly less than upper bound."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer(test_config=fast_bootstrap_config)
        lower, upper = analyzer._bootstrap_ci(baseline, treatment, fast_bootstrap_config)
        assert lower < upper

    def test_bootstrap_ci_excludes_zero_for_clearly_different(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
        fast_bootstrap_config: SignificanceTest,
    ) -> None:
        """95% CI should exclude zero when distributions are clearly different."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer(test_config=fast_bootstrap_config)
        lower, upper = analyzer._bootstrap_ci(baseline, treatment, fast_bootstrap_config)
        # treatment mean is ~5 higher than baseline, so CI should be entirely positive
        assert lower > 0.0

    def test_normal_ci_lower_less_than_upper(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Normal-approximation CI lower bound must be strictly less than upper bound."""
        baseline, treatment = clearly_different_samples
        config = SignificanceTest(test_type="t_test", n_bootstrap=N_BOOTSTRAP_FAST)
        analyzer = StatisticalAnalyzer(test_config=config)
        lower, upper = analyzer._normal_ci(baseline, treatment, config)
        assert lower < upper

    def test_compare_runs_bootstrap_has_ci(
        self,
        clearly_different_samples: tuple[np.ndarray, np.ndarray],
        fast_bootstrap_config: SignificanceTest,
    ) -> None:
        """compare_runs with bootstrap test_type should include a CI in the result."""
        baseline, treatment = clearly_different_samples
        analyzer = StatisticalAnalyzer(test_config=fast_bootstrap_config)
        result = analyzer.compare_runs(baseline, treatment)
        assert result.confidence_interval is not None
        assert result.confidence_interval[0] < result.confidence_interval[1]


# ---------------------------------------------------------------------------
# 7. TestCreateAnalyzer
# ---------------------------------------------------------------------------


class TestCreateAnalyzer:
    """Tests for the create_analyzer factory function."""

    def test_factory_returns_statistical_analyzer(self) -> None:
        """create_analyzer should return a StatisticalAnalyzer instance."""
        analyzer = create_analyzer(
            test_type="t_test",
            alpha=0.05,
            n_bootstrap=1000,
        )
        assert isinstance(analyzer, StatisticalAnalyzer)

    def test_factory_passes_test_type(self) -> None:
        """The test_type argument must be forwarded to the embedded config."""
        analyzer = create_analyzer(test_type="mann_whitney", n_bootstrap=1000)
        assert analyzer.test_config.test_type == "mann_whitney"

    def test_factory_passes_alpha(self) -> None:
        """The alpha argument must be forwarded to the embedded config."""
        analyzer = create_analyzer(alpha=0.01, n_bootstrap=1000)
        assert analyzer.test_config.alpha == pytest.approx(0.01)

    def test_factory_passes_kwargs(self) -> None:
        """Extra kwargs such as correction and alternative must be forwarded."""
        analyzer = create_analyzer(
            correction="fdr",
            alternative="greater",
            n_bootstrap=1000,
        )
        assert analyzer.test_config.correction == "fdr"
        assert analyzer.test_config.alternative == "greater"

    def test_factory_default_config_is_valid(self) -> None:
        """create_analyzer with no extra args should produce a valid analyzer."""
        # n_bootstrap defaults to 10_000 which is valid
        analyzer = create_analyzer()
        assert isinstance(analyzer.test_config, SignificanceTest)
