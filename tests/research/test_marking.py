"""Parity tests for the shared Dörfler marking primitive.

``src.research.marking.dorfler_mark`` unifies two historically divergent
implementations: ``DorflerAMRSolver._dorfler_mark`` (squared bulk quantity,
``src.research.baselines``) and ``ScikitFEMPoissonSolver._dorfler_mark`` (linear
bulk quantity, ``src.research.fem_baseline``). These tests pin both variants
against independent, frozen reference re-derivations of the original formulas
(so a future edit to ``dorfler_mark`` that silently drifts from either legacy
behaviour fails here), and confirm both legacy classes now delegate to it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from src.research.baselines import DorflerAMRSolver
from src.research.fem_baseline import FEMConfig, ScikitFEMPoissonSolver
from src.research.marking import dorfler_mark

_INDICATOR_ARRAYS = st.lists(
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=25,
).map(np.array)
_THETAS = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)


def _reference_squared(indicators: npt.NDArray[np.float64], theta: float) -> npt.NDArray[np.bool_]:
    """Byte-for-byte reproduction of the pre-delegation ``DorflerAMRSolver._dorfler_mark``."""
    total = np.sum(indicators**2)
    threshold = theta * total
    sorted_idx = np.argsort(indicators)[::-1]
    cumsum = np.cumsum(indicators[sorted_idx] ** 2)
    n_mark = int(np.searchsorted(cumsum, threshold)) + 1
    marked = np.zeros(len(indicators), dtype=bool)
    marked[sorted_idx[:n_mark]] = True
    return marked


def _reference_linear(indicators: npt.NDArray[np.float64], theta: float) -> npt.NDArray[np.bool_]:
    """Byte-for-byte reproduction of the pre-delegation ``ScikitFEMPoissonSolver._dorfler_mark``."""
    total = float(np.sum(indicators))
    if total <= 0.0:
        return np.zeros_like(indicators, dtype=bool)
    threshold = theta * total
    order = np.argsort(indicators)[::-1]
    cumulative = np.cumsum(indicators[order])
    cutoff = int(np.searchsorted(cumulative, threshold) + 1)
    marked = np.zeros_like(indicators, dtype=bool)
    marked[order[:cutoff]] = True
    return marked


class TestDorflerMarkSquaredVariant:
    """``variant="squared"`` must match the frozen ``DorflerAMRSolver`` formula."""

    @given(indicators=_INDICATOR_ARRAYS, theta=_THETAS)
    @example(indicators=np.zeros(5), theta=0.3)
    def test_matches_reference_formula(
        self, indicators: npt.NDArray[np.float64], theta: float
    ) -> None:
        expected = _reference_squared(indicators, theta)
        actual = dorfler_mark(indicators, theta, variant="squared")
        np.testing.assert_array_equal(actual, expected)

    def test_all_zeros_marks_exactly_one_element(self) -> None:
        """AC4: an all-zero indicator array still marks >=1 element under "squared"."""
        marked = dorfler_mark(np.zeros(8), theta=0.3, variant="squared")
        assert marked.sum() == 1

    def test_delegate_matches_shared_function(self) -> None:
        solver = DorflerAMRSolver(marking_fraction=0.4)
        indicators = np.array([0.1, 0.9, 0.3, 0.05, 1.2])
        expected = dorfler_mark(indicators, 0.4, variant="squared")
        np.testing.assert_array_equal(solver._dorfler_mark(indicators), expected)


class TestDorflerMarkLinearVariant:
    """``variant="linear"`` must match the frozen ``ScikitFEMPoissonSolver`` formula."""

    @given(indicators=_INDICATOR_ARRAYS, theta=_THETAS)
    @example(indicators=np.zeros(5), theta=0.3)
    def test_matches_reference_formula(
        self, indicators: npt.NDArray[np.float64], theta: float
    ) -> None:
        expected = _reference_linear(indicators, theta)
        actual = dorfler_mark(indicators, theta, variant="linear")
        np.testing.assert_array_equal(actual, expected)

    def test_all_zeros_marks_nothing(self) -> None:
        """AC4: an all-zero indicator array marks nothing under "linear"."""
        marked = dorfler_mark(np.zeros(8), theta=0.3, variant="linear")
        assert not marked.any()

    def test_delegate_matches_shared_function(self) -> None:
        solver = ScikitFEMPoissonSolver(FEMConfig(marking_fraction=0.4))
        indicators = np.array([0.1, 0.9, 0.3, 0.05, 1.2])
        expected = dorfler_mark(indicators, 0.4, variant="linear")
        np.testing.assert_array_equal(solver._dorfler_mark(indicators), expected)


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="Unknown marking variant"):
        dorfler_mark(np.array([1.0, 2.0]), 0.5, variant="cubic")  # type: ignore[arg-type]
