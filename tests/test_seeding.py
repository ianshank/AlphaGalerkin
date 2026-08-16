"""Tests for ``src/seeding.py`` — the shared reproducibility-seeding primitives.

``derive_seeds`` is the single definition of the ``base + i * stride`` seed
arithmetic that five call sites depend on (``LLMPriorAblationConfig``,
``ScalingLawConfig``, ``NoyronBasisConfig``, ``ResearchLoopConfig`` and
``src/research/seed_sweep.resolved_seeds``). A silent change to it would
re-derive every per-cell RNG stream and invalidate the medians committed to
``config/baselines/*.json``, so it is pinned here explicitly rather than only
indirectly through its callers.

Validates:
    - Boundary arities: ``n_seeds`` 0 / 1 / large.
    - Exact stride arithmetic, including negative and zero base seeds.
    - Degenerate ``stride=0`` (currently accepted — see the docstring on
      :func:`test_zero_stride_collapses_to_repeated_base_seed`).
    - Determinism: same arguments produce an equal, independent list.
    - Hypothesis: the closed form and pairwise distinctness for non-zero
      strides.
    - Call-site agreement: the five production callers all yield the closed
      form for their own stride.
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from src.seeding import derive_seeds, set_global_seeds


class TestDeriveSeedsArity:
    """Boundary behaviour of the ``n_seeds`` argument."""

    def test_zero_seeds_returns_empty_list(self) -> None:
        """``n_seeds=0`` yields no seeds (not ``[base]``)."""
        assert derive_seeds(42, 0, 1009) == []

    def test_one_seed_returns_base_only(self) -> None:
        """``n_seeds=1`` yields exactly the base seed, stride unused."""
        assert derive_seeds(42, 1, 1009) == [42]
        assert derive_seeds(42, 1, 7919) == [42]

    def test_normal_stride_arithmetic(self) -> None:
        """Successive seeds advance by exactly ``stride``."""
        assert derive_seeds(42, 4, 1009) == [42, 1051, 2060, 3069]

    def test_large_n_is_exact_and_correct_length(self) -> None:
        """A large sweep stays on the closed form (no overflow / drift)."""
        seeds = derive_seeds(0, 1000, 7919)
        assert len(seeds) == 1000
        assert seeds[0] == 0
        assert seeds[-1] == 999 * 7919
        assert len(set(seeds)) == 1000

    def test_negative_n_seeds_returns_empty_list(self) -> None:
        """A negative count degrades to an empty list.

        ``range(n)`` is empty for ``n < 0``, so no exception is raised. Every
        production caller constrains ``n_seeds`` with a Pydantic ``ge=1``
        field, so this is unreachable from config — pinned as documented
        behaviour rather than endorsed API.
        """
        assert derive_seeds(42, -1, 1009) == []


class TestDeriveSeedsBaseAndStride:
    """Base-seed and stride edge cases."""

    def test_zero_base_seed(self) -> None:
        """A zero base seed is a legal starting point."""
        assert derive_seeds(0, 3, 1009) == [0, 1009, 2018]

    def test_negative_base_seed_is_not_clamped(self) -> None:
        """Negative base seeds pass through unchanged.

        ``derive_seeds`` performs arithmetic only; it does not clamp into
        numpy's valid ``[0, 2**32 - 1]`` seed range. Callers that feed the
        result to ``np.random.seed`` are responsible for the clamp (the
        contract ``set_global_seeds`` documents).
        """
        assert derive_seeds(-5, 3, 10) == [-5, 5, 15]

    def test_negative_stride_walks_backwards(self) -> None:
        """A negative stride is accepted and produces a descending sweep."""
        assert derive_seeds(100, 3, -10) == [100, 90, 80]

    def test_zero_stride_collapses_to_repeated_base_seed(self) -> None:
        """``stride=0`` yields ``n_seeds`` *identical* seeds — not rejected.

        This is a genuinely degenerate sweep: every "independent" repeat runs
        the same RNG stream, so a median-over-seeds statistic would report
        zero variance from what is really a single sample. ``derive_seeds``
        performs no validation, so nothing raises. No production caller passes
        0 (every stride is a hard-coded prime), which is why this is pinned as
        current behaviour rather than a ``pytest.raises``; adding a
        ``stride != 0`` guard would be a strict improvement.
        """
        assert derive_seeds(42, 4, 0) == [42, 42, 42, 42]
        assert len(set(derive_seeds(42, 4, 0))) == 1


class TestDeriveSeedsDeterminism:
    """The function must be pure and repeatable."""

    def test_same_arguments_give_equal_lists(self) -> None:
        """Two calls with identical arguments compare equal."""
        assert derive_seeds(7, 5, 1009) == derive_seeds(7, 5, 1009)

    def test_returns_a_fresh_list_each_call(self) -> None:
        """Mutating a returned list cannot corrupt a later call."""
        first = derive_seeds(7, 3, 1009)
        first.append(999_999)
        assert derive_seeds(7, 3, 1009) == [7, 1016, 2025]

    def test_unaffected_by_global_rng_state(self) -> None:
        """Derivation is arithmetic, not sampled — global RNG is irrelevant."""
        set_global_seeds(1)
        a = derive_seeds(11, 4, 1009)
        set_global_seeds(999)
        b = derive_seeds(11, 4, 1009)
        assert a == b


class TestDeriveSeedsProperties:
    """Hypothesis invariants over the derivation."""

    @given(
        base=st.integers(min_value=-10_000, max_value=10_000),
        n=st.integers(min_value=0, max_value=64),
        stride=st.integers(min_value=-10_000, max_value=10_000),
    )
    def test_matches_closed_form(self, base: int, n: int, stride: int) -> None:
        """``derive_seeds(b, n, s) == [b + i*s for i in range(n)]``."""
        assert derive_seeds(base, n, stride) == [base + i * stride for i in range(n)]

    @given(
        base=st.integers(min_value=-10_000, max_value=10_000),
        n=st.integers(min_value=0, max_value=64),
        stride=st.integers(min_value=1, max_value=10_000),
    )
    def test_nonzero_stride_gives_distinct_seeds(self, base: int, n: int, stride: int) -> None:
        """Every derived seed is unique whenever ``stride != 0``."""
        seeds = derive_seeds(base, n, stride)
        assert len(set(seeds)) == len(seeds) == n

    @given(
        base=st.integers(min_value=-10_000, max_value=10_000),
        n=st.integers(min_value=1, max_value=64),
        stride=st.integers(min_value=-10_000, max_value=10_000),
    )
    def test_prefix_stability(self, base: int, n: int, stride: int) -> None:
        """Growing the sweep only appends; it never renumbers earlier seeds.

        This is what lets a committed 3-seed baseline stay comparable to a
        later 5-seed rerun.
        """
        assert derive_seeds(base, n, stride)[:-1] == derive_seeds(base, n - 1, stride)


class TestProductionCallSitesAgree:
    """Every caller of ``derive_seeds`` still produces the closed form."""

    def test_seed_sweep_resolved_seeds(self) -> None:
        """``src/research/seed_sweep.py`` uses stride 7919."""
        from src.research.seed_sweep import SEED_PRIME_STRIDE, resolved_seeds

        assert SEED_PRIME_STRIDE == 7919
        assert resolved_seeds(3, 3) == [3 + i * 7919 for i in range(3)]

    def test_llm_prior_config_resolved_seeds(self) -> None:
        """``LLMPriorAblationConfig`` derives from its own prime stride."""
        from src.poc.scenarios.llm_prior_config import (
            _SEED_PRIME_STRIDE,
            LLMPriorAblationConfig,
        )

        config = LLMPriorAblationConfig(seed=13, n_seeds=3)
        assert config.resolved_seeds() == [13 + i * _SEED_PRIME_STRIDE for i in range(3)]

    def test_scaling_law_config_resolved_seeds(self) -> None:
        """``ScalingLawConfig`` derives from its own prime stride."""
        from src.poc.scenarios.scaling_law_config import (
            _SEED_PRIME_STRIDE,
            ScalingLawConfig,
        )

        config = ScalingLawConfig(seed=13, n_seeds=3)
        assert config.resolved_seeds() == [13 + i * _SEED_PRIME_STRIDE for i in range(3)]

    def test_noyron_basis_config_resolved_seeds(self) -> None:
        """``NoyronBasisConfig`` derives from its own prime stride."""
        from src.poc.scenarios.noyron_basis_config import (
            _SEED_PRIME_STRIDE,
            NoyronBasisConfig,
        )

        config = NoyronBasisConfig(seed=13, n_seeds=3)
        assert config.resolved_seeds() == [13 + i * _SEED_PRIME_STRIDE for i in range(3)]

    def test_research_loop_config_resolved_seeds(self) -> None:
        """``ResearchLoopConfig`` derives from its own prime stride."""
        from src.agents.config import (
            _SEED_PRIME_STRIDE,
            ResearchLoopConfig,
            ResearchProblemSpec,
        )

        config = ResearchLoopConfig(
            name="seed-check",
            problems=[ResearchProblemSpec(name="p0", pde="poisson")],
            seed=13,
            n_seeds=3,
        )
        assert config.resolved_seeds() == [13 + i * _SEED_PRIME_STRIDE for i in range(3)]

    def test_explicit_seeds_bypass_derivation(self) -> None:
        """An explicit ``seeds`` list wins over the derived sweep (deduped)."""
        from src.poc.scenarios.scaling_law_config import ScalingLawConfig

        config = ScalingLawConfig(seed=13, n_seeds=3, seeds=[5, 5, 9])
        assert config.resolved_seeds() == [5, 9]


class TestSetGlobalSeeds:
    """``set_global_seeds`` seeds both RNGs."""

    def test_numpy_stream_is_reproducible(self) -> None:
        """Re-seeding replays the same numpy draws."""
        set_global_seeds(1234)
        first = np.random.rand(4).tolist()
        set_global_seeds(1234)
        assert np.random.rand(4).tolist() == first

    def test_torch_stream_is_reproducible(self) -> None:
        """Re-seeding replays the same torch draws."""
        set_global_seeds(1234)
        first = torch.rand(4)
        set_global_seeds(1234)
        assert torch.equal(torch.rand(4), first)

    def test_distinct_seeds_give_distinct_streams(self) -> None:
        """Two different seeds do not collide on the first draws."""
        set_global_seeds(1)
        a = (np.random.rand(4).tolist(), torch.rand(4))
        set_global_seeds(2)
        b = (np.random.rand(4).tolist(), torch.rand(4))
        assert a[0] != b[0]
        assert not torch.equal(a[1], b[1])

    def test_derived_seeds_produce_distinct_streams(self) -> None:
        """The whole point: strided seeds decorrelate the RNG streams."""
        draws = []
        for seed in derive_seeds(42, 3, 1009):
            set_global_seeds(seed)
            draws.append(np.random.rand(3).tolist())
        assert len({tuple(d) for d in draws}) == 3
