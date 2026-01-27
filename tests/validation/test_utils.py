"""Tests for validation utilities."""

from __future__ import annotations

import pytest

from src.validation.utils import deep_merge, deep_merge_inplace


class TestDeepMerge:
    """Tests for deep_merge function."""

    def test_basic_merge(self) -> None:
        """Test basic key-value merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        """Test nested dictionary merge."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}
        result = deep_merge(base, override)
        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_deeply_nested_merge(self) -> None:
        """Test deeply nested dictionary merge."""
        base = {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}
        override = {"l1": {"l2": {"l3": {"b": 3}}}}
        result = deep_merge(base, override)
        assert result == {"l1": {"l2": {"l3": {"a": 1, "b": 3}}}}

    def test_override_non_dict_with_non_dict(self) -> None:
        """Test overriding non-dict value."""
        base = {"a": 1}
        override = {"a": "string"}
        result = deep_merge(base, override)
        assert result == {"a": "string"}

    def test_override_dict_with_non_dict(self) -> None:
        """Test overriding dict with non-dict."""
        base = {"a": {"b": 1}}
        override = {"a": "replaced"}
        result = deep_merge(base, override)
        assert result == {"a": "replaced"}

    def test_override_non_dict_with_dict(self) -> None:
        """Test overriding non-dict with dict."""
        base = {"a": 1}
        override = {"a": {"b": 2}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": 2}}

    def test_original_not_modified(self) -> None:
        """Test that original dictionaries are not modified."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        original_base = {"a": {"b": 1}}
        original_override = {"a": {"c": 2}}

        deep_merge(base, override)

        assert base == original_base
        assert override == original_override

    def test_empty_override(self) -> None:
        """Test merge with empty override."""
        base = {"a": 1, "b": {"c": 2}}
        result = deep_merge(base, {})
        assert result == base

    def test_empty_base(self) -> None:
        """Test merge with empty base."""
        override = {"a": 1, "b": {"c": 2}}
        result = deep_merge({}, override)
        assert result == override

    def test_both_empty(self) -> None:
        """Test merge with both empty."""
        result = deep_merge({}, {})
        assert result == {}

    def test_list_values_replaced(self) -> None:
        """Test that list values are replaced, not concatenated."""
        base = {"list": [1, 2, 3]}
        override = {"list": [4, 5]}
        result = deep_merge(base, override)
        assert result == {"list": [4, 5]}

    def test_none_value_override(self) -> None:
        """Test that None values override correctly."""
        base = {"a": 1}
        override = {"a": None}
        result = deep_merge(base, override)
        assert result == {"a": None}

    def test_pydantic_like_config_merge(self) -> None:
        """Test merge scenario similar to Pydantic config usage."""
        base = {
            "seed": 42,
            "gpu_training": {"d_model": 256, "n_heads": 8, "batch_size": 64},
            "tolerance": {"level": "standard", "rtol": None},
            "run_gpu_training": True,
        }
        override = {
            "gpu_training": {"d_model": 128},  # Only override d_model
            "seed": 123,  # Override seed
        }
        result = deep_merge(base, override)

        # d_model should be overridden, n_heads and batch_size preserved
        assert result["gpu_training"]["d_model"] == 128
        assert result["gpu_training"]["n_heads"] == 8
        assert result["gpu_training"]["batch_size"] == 64
        assert result["seed"] == 123
        assert result["run_gpu_training"] is True


class TestDeepMergeInplace:
    """Tests for deep_merge_inplace function."""

    def test_modifies_base_inplace(self) -> None:
        """Test that base is modified in-place."""
        base = {"a": 1, "nested": {"b": 2}}
        deep_merge_inplace(base, {"nested": {"c": 3}})

        assert base == {"a": 1, "nested": {"b": 2, "c": 3}}

    def test_nested_inplace_merge(self) -> None:
        """Test nested in-place merge."""
        base = {"outer": {"inner": {"a": 1}}}
        deep_merge_inplace(base, {"outer": {"inner": {"b": 2}}})

        assert base == {"outer": {"inner": {"a": 1, "b": 2}}}
