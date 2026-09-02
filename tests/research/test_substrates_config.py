"""Validation tests for ``SubstrateConfig``.

The shared data contract every ``RefinementSubstrate`` implementation
(``TensorGridSubstrate``, ``SkfemTriSubstrate``) constructs against. See
``specs/refinement_substrate.spec.md``'s Data Contract table for the
field-by-field source of truth this file pins.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.research.substrates.config import (
    AREA_FLOOR,
    RATE_FIT_MIN_POINTS,
    RATIO_FLOOR,
    SubstrateConfig,
)


class TestSubstrateConfigDefaults:
    def test_defaults_match_spec(self) -> None:
        config = SubstrateConfig(name="test")
        assert config.kind == "skfem_tri"
        assert config.element_type == "P1"
        assert config.initial_refinements == 2
        assert config.initial_side == 4
        assert config.marking_variant == "squared"
        assert config.error_metric == "quadrature"
        assert config.enforce_immutable_meshes is True
        assert config.solve_cache_max_entries == 4096


class TestSubstrateConfigValidation:
    @pytest.mark.parametrize("kind", ["tensor_grid", "skfem_tri"])
    def test_accepts_valid_kind(self, kind: str) -> None:
        assert SubstrateConfig(name="test", kind=kind).kind == kind

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", kind="quadtree")  # type: ignore[arg-type]

    @pytest.mark.parametrize("element_type", ["P1", "P2", "P3"])
    def test_accepts_valid_element_type(self, element_type: str) -> None:
        assert SubstrateConfig(name="test", element_type=element_type).element_type == (
            element_type
        )

    def test_rejects_unknown_element_type(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", element_type="P4")  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [-1, 9])
    def test_initial_refinements_bounds(self, value: int) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", initial_refinements=value)

    def test_initial_refinements_boundary_values_accepted(self) -> None:
        assert SubstrateConfig(name="test", initial_refinements=0).initial_refinements == 0
        assert SubstrateConfig(name="test", initial_refinements=8).initial_refinements == 8

    @pytest.mark.parametrize("value", [3, 5, 7])
    def test_rejects_odd_initial_side(self, value: int) -> None:
        with pytest.raises(ValidationError, match="must be even"):
            SubstrateConfig(name="test", initial_side=value)

    @pytest.mark.parametrize("value", [2, 4, 64])
    def test_accepts_even_initial_side_within_bounds(self, value: int) -> None:
        assert SubstrateConfig(name="test", initial_side=value).initial_side == value

    @pytest.mark.parametrize("value", [1, 65])
    def test_initial_side_out_of_bounds_rejected(self, value: int) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", initial_side=value)

    @pytest.mark.parametrize("variant", ["squared", "linear"])
    def test_accepts_valid_marking_variant(self, variant: str) -> None:
        assert SubstrateConfig(name="test", marking_variant=variant).marking_variant == variant

    def test_rejects_unknown_marking_variant(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", marking_variant="cubic")  # type: ignore[arg-type]

    @pytest.mark.parametrize("metric", ["quadrature", "nodal_rms"])
    def test_accepts_valid_error_metric(self, metric: str) -> None:
        assert SubstrateConfig(name="test", error_metric=metric).error_metric == metric

    def test_rejects_unknown_error_metric(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", error_metric="max_norm")  # type: ignore[arg-type]

    def test_solve_cache_max_entries_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", solve_cache_max_entries=0)
        assert SubstrateConfig(name="test", solve_cache_max_entries=1).solve_cache_max_entries == 1

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", not_a_real_field=1)  # type: ignore[call-arg]


def test_named_constants_match_spec() -> None:
    """Pins the three constants against ``specs/refinement_substrate.spec.md``'s table.

    Be clear about what this is and is not. It guards **doc/code drift** — the
    spec names these values, so silently changing one here should fail. It is
    *not* a behavioural test, and on its own it would be close to a tautology
    (asserting a constant defined as ``1e-15`` equals ``1e-15``). The
    behavioural coverage lives in ``tests/research/test_substrates_sweep.py``,
    which drives all three through their real consumers in
    ``src/research/substrates/sweep.py``: ``RATE_FIT_MIN_POINTS`` as the
    fit-refusal boundary, ``AREA_FLOOR`` as the degenerate-unit threshold, and
    ``RATIO_FLOOR`` as the log-interpolation guard. Until Slice D added that
    module the three constants had **no consumer at all**, and this test was
    the only thing referencing them.
    """
    assert RATIO_FLOOR == 1e-15
    assert AREA_FLOOR == 1e-30
    assert RATE_FIT_MIN_POINTS == 3
