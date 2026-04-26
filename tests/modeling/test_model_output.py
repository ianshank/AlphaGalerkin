"""Tests for the :class:`ModelOutput` frozen-dataclass migration.

Covers:

* Backwards-compatibility with the previous :class:`NamedTuple` API
  (positional construction, iteration, indexing).
* Vector-field extension (``with_vector_fields``, ``has_vector_fields``).
* Frozen semantics (mutation must raise).
* Optional-field defaults (``lbb_constant``, ``vector_fields``,
  ``field_metadata`` all None by default).

These tests are intentionally independent of model construction —
they only exercise the data-carrier contract.  Existing model-level
tests in ``tests/modeling/test_model.py`` cover the upstream
constructors.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from src.modeling.model import ModelOutput


def _make_basic_output(lbb: bool = False) -> ModelOutput:
    return ModelOutput(
        policy_logits=torch.zeros(2, 5),
        value=torch.zeros(2, 1),
        lbb_constant=torch.tensor([0.3, 0.4]) if lbb else None,
    )


class TestModelOutputBackwardsCompat:
    """Exercise the legacy NamedTuple-style call surface."""

    def test_positional_construction_with_lbb(self) -> None:
        policy = torch.zeros(1, 5)
        value = torch.zeros(1, 1)
        lbb = torch.tensor([0.5])
        out = ModelOutput(policy, value, lbb)
        assert out.policy_logits is policy
        assert out.value is value
        assert out.lbb_constant is lbb

    def test_positional_construction_without_lbb(self) -> None:
        out = ModelOutput(torch.zeros(1, 5), torch.zeros(1, 1))
        assert out.lbb_constant is None
        assert out.vector_fields is None
        assert out.field_metadata is None

    def test_keyword_construction(self) -> None:
        out = _make_basic_output(lbb=True)
        assert out.lbb_constant is not None

    def test_iteration_yields_three_fields(self) -> None:
        out = _make_basic_output(lbb=True)
        items = list(out)
        assert len(items) == 3
        assert items[0] is out.policy_logits
        assert items[1] is out.value
        assert items[2] is out.lbb_constant

    def test_indexing_returns_three_fields(self) -> None:
        out = _make_basic_output(lbb=True)
        assert out[0] is out.policy_logits
        assert out[1] is out.value
        assert out[2] is out.lbb_constant


class TestModelOutputFrozen:
    """Frozen dataclass semantics."""

    def test_mutation_raises(self) -> None:
        out = _make_basic_output()
        with pytest.raises(FrozenInstanceError):
            out.policy_logits = torch.zeros(1, 1)  # type: ignore[misc]

    def test_lbb_mutation_raises(self) -> None:
        out = _make_basic_output()
        with pytest.raises(FrozenInstanceError):
            out.lbb_constant = torch.tensor([1.0])  # type: ignore[misc]


class TestModelOutputVectorFields:
    """Multi-physics extension surface."""

    def test_default_no_vector_fields(self) -> None:
        out = _make_basic_output()
        assert out.has_vector_fields() is False
        assert out.vector_fields is None

    def test_with_vector_fields_returns_new_instance(self) -> None:
        out = _make_basic_output()
        velocity = torch.zeros(2, 2, 4, 4)
        pressure = torch.zeros(2, 1, 4, 4)
        new = out.with_vector_fields({"velocity": velocity, "pressure": pressure})
        # Original untouched
        assert out.has_vector_fields() is False
        # New populated
        assert new.has_vector_fields() is True
        assert new.vector_fields is not None
        assert set(new.vector_fields.keys()) == {"velocity", "pressure"}
        # Scalar fields preserved
        assert new.policy_logits is out.policy_logits
        assert new.value is out.value
        assert new.lbb_constant is out.lbb_constant

    def test_with_vector_fields_metadata(self) -> None:
        out = _make_basic_output()
        meta = {"units": "SI", "convention": "y-up"}
        new = out.with_vector_fields(
            {"velocity": torch.zeros(2, 2, 4, 4)},
            field_metadata=meta,
        )
        assert new.field_metadata == meta
        # Defensive copy: mutating original metadata must not affect output
        meta["units"] = "imperial"
        assert new.field_metadata is not None
        assert new.field_metadata["units"] == "SI"

    def test_with_vector_fields_defensive_copy(self) -> None:
        """Vector-field dict must be defensively copied."""
        out = _make_basic_output()
        original = {"velocity": torch.zeros(2, 2, 4, 4)}
        new = out.with_vector_fields(original)
        assert new.vector_fields is not None
        original["velocity"] = torch.ones(2, 2, 4, 4)
        # New instance still references the original tensor, not the
        # post-mutation replacement.
        assert torch.equal(new.vector_fields["velocity"], torch.zeros(2, 2, 4, 4))
