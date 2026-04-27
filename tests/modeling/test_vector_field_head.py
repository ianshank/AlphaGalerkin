"""Tests for the registered VectorFieldHead (Track B).

Covers:

* Registry round-trip via ``HeadRegistry().get("vector_field")`` and
  the ``make_vector_field_head`` factory.
* Output shape per registered field name: ``[B, n, components_per_field]``.
* Backwards-compatible scalar mode (``n_fields=1``) returns a single-key
  dict whose tensor matches the legacy
  :class:`src.modeling.model.DenseHead` shape (``[B, n, 1]``).
* Constructor validation: rejects mismatched ``len(field_names)``,
  duplicate names, non-positive shapes.
* End-to-end integration with :class:`src.modeling.model.ModelOutput`
  via the ``with_vector_fields`` helper.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.modeling.heads import (
    HeadBase,
    HeadRegistry,
    VectorFieldHead,
    make_vector_field_head,
)
from src.modeling.model import ModelOutput

# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestVectorFieldHeadConstruction:
    def test_default_n_fields_1_uses_scalar_name(self) -> None:
        head = VectorFieldHead(d_model=4)
        assert head.n_fields == 1
        assert head.field_names == ("scalar",)

    def test_default_n_fields_gt_1_uses_indexed_names(self) -> None:
        head = VectorFieldHead(d_model=4, n_fields=3)
        assert head.field_names == ("field_0", "field_1", "field_2")

    def test_explicit_field_names_honored(self) -> None:
        head = VectorFieldHead(d_model=4, n_fields=3, field_names=("u", "v", "p"))
        assert head.field_names == ("u", "v", "p")

    def test_field_names_length_must_match_n_fields(self) -> None:
        with pytest.raises(ValueError, match="len\\(field_names\\)"):
            VectorFieldHead(d_model=4, n_fields=2, field_names=("a", "b", "c"))

    def test_duplicate_field_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            VectorFieldHead(d_model=4, n_fields=2, field_names=("u", "u"))

    def test_non_positive_d_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            VectorFieldHead(d_model=0)

    def test_non_positive_n_fields_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_fields"):
            VectorFieldHead(d_model=4, n_fields=0)

    def test_non_positive_components_per_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="components_per_field"):
            VectorFieldHead(d_model=4, n_fields=1, components_per_field=0)


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------


class TestRegistryRoundTrip:
    def test_registry_resolves_by_name(self) -> None:
        cls = HeadRegistry().get("vector_field")
        assert cls is VectorFieldHead

    def test_make_vector_field_head_factory(self) -> None:
        head = make_vector_field_head(d_model=4, n_fields=2, field_names=("a", "b"))
        assert isinstance(head, VectorFieldHead)
        assert head.field_names == ("a", "b")

    def test_make_factory_via_explicit_head_name(self) -> None:
        head = make_vector_field_head(d_model=4, head_name="vector_field")
        assert isinstance(head, HeadBase)


# ---------------------------------------------------------------------------
# Forward pass — shapes and contracts
# ---------------------------------------------------------------------------


class TestForwardShapes:
    def test_scalar_mode_returns_single_key_dict(self) -> None:
        head = VectorFieldHead(d_model=4, n_fields=1)
        x = torch.randn(2, 8, 4)
        out = head(x)
        assert set(out.keys()) == {"scalar"}
        assert out["scalar"].shape == (2, 8, 1)

    def test_three_field_mode_navier_stokes_shape(self) -> None:
        head = VectorFieldHead(d_model=8, n_fields=3, field_names=("u", "v", "p"))
        x = torch.randn(4, 16, 8)
        out = head(x)
        assert set(out.keys()) == {"u", "v", "p"}
        for name in ("u", "v", "p"):
            assert out[name].shape == (4, 16, 1)

    def test_components_per_field_widens_output(self) -> None:
        head = VectorFieldHead(
            d_model=8, n_fields=2, field_names=("velocity", "stress"), components_per_field=3
        )
        x = torch.randn(2, 5, 8)
        out = head(x)
        assert out["velocity"].shape == (2, 5, 3)
        assert out["stress"].shape == (2, 5, 3)

    def test_input_dim_validation(self) -> None:
        head = VectorFieldHead(d_model=4)
        with pytest.raises(ValueError, match=r"\[B, n, d_model\]"):
            head(torch.randn(8, 4))  # missing batch dim

    def test_gradient_flows_into_input(self) -> None:
        head = VectorFieldHead(d_model=4, n_fields=2, field_names=("a", "b"))
        x = torch.randn(1, 4, 4, requires_grad=True)
        out = head(x)
        loss = out["a"].sum() + out["b"].sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# ModelOutput integration
# ---------------------------------------------------------------------------


class TestModelOutputIntegration:
    def test_attaches_to_model_output_via_with_vector_fields(self) -> None:
        head = VectorFieldHead(d_model=4, n_fields=2, field_names=("u", "v"))
        x = torch.randn(2, 8, 4)
        scalar = ModelOutput(
            policy_logits=torch.zeros(2, 9),
            value=torch.zeros(2, 1),
        )
        enriched = scalar.with_vector_fields(head(x))
        assert enriched.has_vector_fields()
        assert enriched.vector_fields is not None
        assert set(enriched.vector_fields.keys()) == {"u", "v"}

    def test_scalar_only_model_output_remains_unchanged(self) -> None:
        # The default ModelOutput shape is preserved (no breaking change).
        scalar = ModelOutput(
            policy_logits=torch.zeros(1, 9),
            value=torch.zeros(1, 1),
        )
        assert scalar.vector_fields is None
        assert not scalar.has_vector_fields()


# ---------------------------------------------------------------------------
# Operator metadata (Track B exposes n_fields / field_names on operators)
# ---------------------------------------------------------------------------


class TestOperatorMultiFieldMetadata:
    def test_default_pde_operator_is_scalar(self) -> None:
        from src.pde.config import PDEConfig, PDEType
        from src.pde.operators import HeatOperator

        cfg = PDEConfig(name="t", pde_type=PDEType.HEAT)
        op = HeatOperator(cfg)
        assert op.n_fields == 1
        assert op.field_names == ("scalar",)

    def test_navier_stokes_advertises_three_fields(self) -> None:
        from src.pde.config import PDEConfig, PDEType
        from src.pde.operators import NavierStokesOperator

        cfg = PDEConfig(name="t", pde_type=PDEType.NAVIER_STOKES)
        op = NavierStokesOperator(cfg)
        assert op.n_fields == 3
        assert op.field_names == ("u", "v", "p")

    def test_navier_stokes_metadata_is_class_level(self) -> None:
        # Track B contract: tools that read n_fields without instantiating
        # the operator (e.g. config validators) MUST find these on the
        # class itself, not just on instances.
        from src.pde.operators import NavierStokesOperator

        assert NavierStokesOperator.n_fields == 3
        assert NavierStokesOperator.field_names == ("u", "v", "p")


# ---------------------------------------------------------------------------
# Dynamic head selection wiring (smoke)
# ---------------------------------------------------------------------------


class TestDynamicSelection:
    def test_head_chosen_by_string_via_factory(self) -> None:
        head = make_vector_field_head(d_model=4, head_name="vector_field")
        assert isinstance(head, nn.Module)
        out = head(torch.randn(1, 3, 4))
        assert "scalar" in out
