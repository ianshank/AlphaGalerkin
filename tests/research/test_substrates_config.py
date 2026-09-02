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
        """``initial_side`` is tensor-grid-scoped, so this must name that kind.

        It reads as boilerplate but is not: on the default
        ``kind="skfem_tri"`` this construction is now *rejected*, because a
        tensor-grid knob on an skfem config is a silent no-op. Two of these
        parametrisations went red the moment the scope validator landed --
        which is the validator doing its job on real calling code, not just
        on a hypothetical.
        """
        config = SubstrateConfig(name="test", kind="tensor_grid", initial_side=value)
        assert config.initial_side == value

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

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
    def test_rejects_marking_fraction_outside_the_unit_interval(self, value: float) -> None:
        with pytest.raises(ValidationError):
            SubstrateConfig(name="test", marking_fraction=value)

    @pytest.mark.parametrize("value", [0.001, 0.3, 1.0])
    def test_accepts_marking_fraction_in_range(self, value: float) -> None:
        assert SubstrateConfig(name="test", marking_fraction=value).marking_fraction == value


class TestKindScopedFieldValidation:
    """A knob the chosen substrate never reads must be rejected, not ignored.

    ``SubstrateConfig(kind="tensor_grid", element_type="P2")`` previously
    constructed cleanly, validated cleanly, and did nothing — the config
    equivalent of a dead abstraction, and worse than a magic number because it
    looks like a working knob.
    """

    @pytest.mark.parametrize(
        ("kind", "field", "value"),
        [
            ("tensor_grid", "element_type", "P2"),
            ("tensor_grid", "initial_refinements", 5),
            ("skfem_tri", "initial_side", 8),
        ],
    )
    def test_rejects_field_scoped_to_the_other_kind(
        self, kind: str, field: str, value: object
    ) -> None:
        with pytest.raises(ValidationError, match="ignores"):
            SubstrateConfig(name="test", kind=kind, **{field: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kind", "field", "value"),
        [
            ("skfem_tri", "element_type", "P2"),
            ("skfem_tri", "initial_refinements", 5),
            ("tensor_grid", "initial_side", 8),
        ],
    )
    def test_accepts_field_scoped_to_its_own_kind(
        self, kind: str, field: str, value: object
    ) -> None:
        config = SubstrateConfig(name="test", kind=kind, **{field: value})  # type: ignore[arg-type]
        assert getattr(config, field) == value

    @pytest.mark.parametrize(
        ("kind", "field"),
        [
            ("tensor_grid", "element_type"),
            ("tensor_grid", "initial_refinements"),
            ("skfem_tri", "initial_side"),
        ],
    )
    def test_explicitly_setting_the_default_is_allowed(self, kind: str, field: str) -> None:
        """Only a value set *away from* its default is rejected.

        Restating the default is harmless and, more importantly, means no
        existing construction breaks just because it was explicit.
        """
        default = SubstrateConfig.model_fields[field].default
        config = SubstrateConfig(name="test", kind=kind, **{field: default})  # type: ignore[arg-type]
        assert getattr(config, field) == default

    @pytest.mark.parametrize(
        "field",
        ["marking_variant", "error_metric", "enforce_immutable_meshes", "solve_cache_max_entries"],
    )
    @pytest.mark.parametrize("kind", ["tensor_grid", "skfem_tri"])
    def test_shared_fields_are_accepted_by_both_kinds(self, kind: str, field: str) -> None:
        """Guards the validator's *scope*, not just its trigger.

        A rule that rejected everything would pass the two tests above and be
        useless. These four fields are read by both substrates, so neither kind
        may reject them — which is what makes ``_KIND_SCOPED_FIELDS`` a scope
        rather than a blanket.
        """
        non_default = {
            "marking_variant": "linear",
            "error_metric": "nodal_rms",
            "enforce_immutable_meshes": False,
            "solve_cache_max_entries": 7,
        }[field]
        config = SubstrateConfig(name="test", kind=kind, **{field: non_default})  # type: ignore[arg-type]
        assert getattr(config, field) == non_default


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
