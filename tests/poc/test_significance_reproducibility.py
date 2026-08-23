"""Reproducibility of the resampling-based significance tests.

Every confidence interval and bootstrap p-value this module has ever produced
was drawn from NumPy's *global, unseeded* stream, at four separate sites
(``_bootstrap_test``, ``_permutation_test``, and two draws in ``_bootstrap_ci``).
So two runs over identical inputs returned different intervals -- in the one
module whose entire job is rigour, inside a project whose governance position is
that every number traces to a committed artifact.

The fix is a typed ``random_seed`` field plus an injectable resampler, with the
unseeded path left byte-identical to the historical behaviour so nothing that
seeds globally today changes. These tests pin all three properties: that a seed
makes a result reproducible, that it does so *without* touching global state,
and that the seed is genuinely consumed rather than accepted and ignored.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
from pydantic import ValidationError

from src.poc.statistics.significance import (
    MAX_PERMUTATIONS,
    SignificanceTest,
    StatisticalAnalyzer,
    resolve_resampler,
)

# Small n_bootstrap keeps these fast; the field's own floor is 1000.
_N: Final[int] = 1000
_BASELINE: Final[list[float]] = [0.10, 0.12, 0.09, 0.14, 0.11, 0.13, 0.08, 0.15]
_TREATMENT: Final[list[float]] = [0.20, 0.24, 0.19, 0.26, 0.22, 0.21, 0.25, 0.18]

# Deliberately overlapping, so a permutation p-value lands strictly between 0
# and 1 and the RNG draw can actually show through in the result.
_OVERLAPPING_A: Final[list[float]] = [0.10, 0.18, 0.12, 0.22, 0.14, 0.20, 0.11, 0.19]
_OVERLAPPING_B: Final[list[float]] = [0.13, 0.21, 0.15, 0.24, 0.12, 0.23, 0.16, 0.17]


def _test(**overrides: object) -> SignificanceTest:
    return SignificanceTest(n_bootstrap=_N, **overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the defect itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("test_type", ["bootstrap", "permutation"])
def test_a_seed_makes_the_result_reproducible(test_type: str) -> None:
    """The property the module lacked. Same inputs, same seed, same numbers."""
    config = _test(test_type=test_type, random_seed=20260823)
    first = StatisticalAnalyzer().compare_runs(_BASELINE, _TREATMENT, config)
    second = StatisticalAnalyzer().compare_runs(_BASELINE, _TREATMENT, config)

    assert first.p_value == second.p_value
    assert first.confidence_interval == second.confidence_interval
    assert first.statistic == second.statistic


@pytest.mark.parametrize("test_type", ["bootstrap", "permutation"])
def test_a_seeded_run_is_independent_of_global_numpy_state(test_type: str) -> None:
    """Reproducible *and* isolated.

    A seeded run that still drew from the global stream would be reproducible
    only when nothing else in the process had drawn first -- which is a property
    no library can rely on.
    """
    config = _test(test_type=test_type, random_seed=7)

    np.random.seed(1)
    first = StatisticalAnalyzer().compare_runs(_BASELINE, _TREATMENT, config)
    np.random.seed(999)
    for _ in range(37):  # perturb the global stream by an arbitrary amount
        np.random.random()
    second = StatisticalAnalyzer().compare_runs(_BASELINE, _TREATMENT, config)

    assert first.p_value == second.p_value
    assert first.confidence_interval == second.confidence_interval


def test_a_seeded_run_does_not_consume_the_global_stream() -> None:
    """The other direction: this module must not perturb *its caller's* RNG."""
    config = _test(random_seed=42)

    np.random.seed(4321)
    untouched = np.random.random()

    np.random.seed(4321)
    StatisticalAnalyzer().compare_runs(_BASELINE, _TREATMENT, config)
    after = np.random.random()

    assert untouched == after


@pytest.mark.parametrize("test_type", ["bootstrap", "permutation"])
def test_different_seeds_give_different_draws(test_type: str) -> None:
    """A seed accepted and then ignored must not look like success.

    That is what a `random_seed` field quietly falling through to the global
    stream would look like, and it would pass every test above.

    Uses *overlapping* groups on purpose. With the well-separated fixtures the
    other tests use, no permutation is ever as extreme as the observed
    difference, so the permutation p-value is 0.0 for every seed and this test
    would pass whether or not the seed was consumed -- a false negative that a
    first draft of it actually produced.
    """
    analyzer = StatisticalAnalyzer()
    a = analyzer.compare_runs(
        _OVERLAPPING_A, _OVERLAPPING_B, _test(test_type=test_type, random_seed=1)
    )
    b = analyzer.compare_runs(
        _OVERLAPPING_A, _OVERLAPPING_B, _test(test_type=test_type, random_seed=2)
    )

    differs = (a.p_value != b.p_value) or (a.confidence_interval != b.confidence_interval)
    assert differs, "two different seeds produced identical draws -- the seed is not being used"


# --------------------------------------------------------------------------
# backwards compatibility of the unseeded default
# --------------------------------------------------------------------------


def test_the_unseeded_default_still_honours_a_globally_seeded_caller() -> None:
    """The historical contract.

    A caller who seeds NumPy today gets reproducible results today, and must
    keep getting them.
    """
    config = _test(test_type="bootstrap")
    assert config.random_seed is None

    np.random.seed(11)
    first = StatisticalAnalyzer().compare_runs(_BASELINE, _TREATMENT, config)
    np.random.seed(11)
    second = StatisticalAnalyzer().compare_runs(_BASELINE, _TREATMENT, config)

    assert first.p_value == second.p_value
    assert first.confidence_interval == second.confidence_interval


def test_the_unseeded_default_does_draw_from_the_global_stream() -> None:
    """States the cost of the default honestly, and pins it.

    Without a seed this module *is* non-reproducible; the field exists precisely
    because of that. If this ever starts passing with two equal results, the
    default has silently changed and the back-compat claim above is void.
    """
    config = _test(test_type="bootstrap")
    np.random.seed(3)
    first = StatisticalAnalyzer().compare_runs(_BASELINE, _TREATMENT, config)
    # No re-seed: the second call continues the same stream.
    second = StatisticalAnalyzer().compare_runs(_BASELINE, _TREATMENT, config)
    assert first.confidence_interval != second.confidence_interval


def test_a_config_dict_without_the_new_field_still_parses() -> None:
    """Old serialised configs must keep loading.

    The model is `extra="forbid"`, so a *new* field is the safe direction and a
    renamed one is not.
    """
    config = SignificanceTest.model_validate(
        {"test_type": "bootstrap", "alpha": 0.05, "n_bootstrap": 1000}
    )
    assert config.random_seed is None


def test_an_unknown_field_is_still_rejected() -> None:
    with pytest.raises(ValidationError):
        SignificanceTest.model_validate({"randon_seed": 5})


def test_a_negative_seed_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SignificanceTest(random_seed=-1)


def test_zero_is_a_valid_seed_and_is_not_confused_with_unset() -> None:
    """Zero is a legal seed, not an absent one.

    `if test.random_seed:` would treat it as unset.
    """
    config = _test(random_seed=0)
    assert config.random_seed == 0
    first = StatisticalAnalyzer().compare_runs(_BASELINE, _TREATMENT, config)
    second = StatisticalAnalyzer().compare_runs(_BASELINE, _TREATMENT, config)
    assert first.confidence_interval == second.confidence_interval


# --------------------------------------------------------------------------
# resolve_resampler precedence
# --------------------------------------------------------------------------


def test_resolver_falls_back_to_the_global_module_when_unseeded() -> None:
    assert resolve_resampler(_test()) is np.random


def test_resolver_returns_a_generator_when_seeded() -> None:
    resampler = resolve_resampler(_test(random_seed=5))
    assert isinstance(resampler, np.random.Generator)


def test_resolver_prefers_an_explicit_override_over_the_configured_seed() -> None:
    override = np.random.default_rng(999)
    assert resolve_resampler(_test(random_seed=5), override) is override


def test_resolver_prefers_an_explicit_override_over_the_global_fallback() -> None:
    override = np.random.default_rng(999)
    assert resolve_resampler(_test(), override) is override


def test_an_injected_resampler_reaches_every_draw_site() -> None:
    """Constructor injection must beat the config, not merely coexist with it.

    It is the seam a caller threading its own seeded stream needs.
    """
    seeded = StatisticalAnalyzer(resampler=np.random.default_rng(2026))
    first = seeded.compare_runs(_BASELINE, _TREATMENT, _test(random_seed=1))

    seeded_again = StatisticalAnalyzer(resampler=np.random.default_rng(2026))
    second = seeded_again.compare_runs(_BASELINE, _TREATMENT, _test(random_seed=1))

    assert first.confidence_interval == second.confidence_interval


def test_every_draw_site_is_routed_through_the_resolver() -> None:
    """The plan named one site; there were four.

    A counting spy proves each of the three resampling code paths actually
    draws from the injected RNG, rather than one of them silently keeping the
    global stream -- which is how three of these survived the first audit.
    """

    class CountingRng:
        def __init__(self) -> None:
            self.inner = np.random.default_rng(0)
            self.shuffles = 0
            self.choices = 0

        def shuffle(self, x: np.ndarray) -> None:
            self.shuffles += 1
            self.inner.shuffle(x)

        def choice(self, a: np.ndarray, size: int, replace: bool) -> np.ndarray:
            self.choices += 1
            return self.inner.choice(a, size=size, replace=replace)

    spy = CountingRng()
    analyzer = StatisticalAnalyzer(resampler=spy)  # type: ignore[arg-type]

    analyzer.compare_runs(_BASELINE, _TREATMENT, _test(test_type="bootstrap"))
    # `_bootstrap_test` shuffles once per resample; the CI it computes draws
    # twice per resample (once per arm). Exactly 2*_N, not 4*_N: the CI is
    # computed once now, where it used to be computed and thrown away first.
    assert spy.shuffles == _N, "the bootstrap hypothesis test bypassed the injected RNG"
    assert spy.choices == 2 * _N, (
        f"expected exactly {2 * _N} CI draws, got {spy.choices} -- "
        "either the CI bypassed the injected RNG, or it is being computed twice again"
    )

    before = spy.shuffles
    analyzer.compare_runs(_BASELINE, _TREATMENT, _test(test_type="permutation"))
    assert spy.shuffles > before, "the permutation test bypassed the injected RNG"


def test_the_permutation_cap_is_a_usable_bound() -> None:
    """Two properties, neither of which is "the constant equals 10,000".

    Its exact value is a tunable and is deliberately not pinned -- a test that
    re-states a constant it imports guards nothing. What *is* a bug rather than
    a tuning choice: a cap below the config's own `n_bootstrap` floor would
    silently cap every legal configuration below what it asked for, and a cap
    large enough to make the capping test below run for minutes has stopped
    being a bound at all.
    """
    floor = SignificanceTest.model_json_schema()["properties"]["n_bootstrap"]["minimum"]
    assert floor <= MAX_PERMUTATIONS, (
        f"cap {MAX_PERMUTATIONS} sits below the n_bootstrap floor {floor}: "
        "every legal config would be silently truncated"
    )
    assert MAX_PERMUTATIONS <= 100_000, (
        f"cap {MAX_PERMUTATIONS} is large enough that the capping test itself "
        "becomes a long-running job; a bound that expensive to verify is not a bound"
    )


def test_permutation_count_is_capped_by_the_named_constant() -> None:
    """The cap is what a caller asking for more actually gets.

    `min(n_bootstrap, 10000)` was a bare literal; it is now a named constant.
    """
    spy_shuffles = 0

    class CountingRng:
        def __init__(self) -> None:
            self.inner = np.random.default_rng(0)

        def shuffle(self, x: np.ndarray) -> None:
            nonlocal spy_shuffles
            spy_shuffles += 1
            self.inner.shuffle(x)

        def choice(self, a: np.ndarray, size: int, replace: bool) -> np.ndarray:
            return self.inner.choice(a, size=size, replace=replace)

    analyzer = StatisticalAnalyzer(resampler=CountingRng())  # type: ignore[arg-type]
    analyzer.compare_runs(
        _BASELINE,
        _TREATMENT,
        SignificanceTest(test_type="permutation", n_bootstrap=MAX_PERMUTATIONS + 5000),
    )
    assert spy_shuffles == MAX_PERMUTATIONS
