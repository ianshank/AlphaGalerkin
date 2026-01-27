"""Utility functions for the validation framework.

This module provides common utility functions used across
the validation framework.
"""

from __future__ import annotations

from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    Performs a deep merge where nested dictionaries are merged recursively
    instead of being completely overwritten. Non-dict values are replaced.

    Args:
        base: Base dictionary to merge into.
        override: Dictionary with values to override/add.

    Returns:
        New dictionary with merged values. Original dicts are not modified.

    Example:
        >>> base = {"gpu_training": {"d_model": 256, "n_heads": 8}}
        >>> override = {"gpu_training": {"d_model": 128}}
        >>> deep_merge(base, override)
        {"gpu_training": {"d_model": 128, "n_heads": 8}}

        >>> base = {"a": 1, "nested": {"b": 2, "c": 3}}
        >>> override = {"nested": {"c": 30, "d": 4}}
        >>> deep_merge(base, override)
        {"a": 1, "nested": {"b": 2, "c": 30, "d": 4}}
    """
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override the value (handles lists, scalars, None, etc.)
            result[key] = value

    return result


def deep_merge_inplace(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge override into base in-place.

    Unlike deep_merge(), this modifies the base dictionary directly.
    Use when you want to avoid creating a new dictionary.

    Args:
        base: Base dictionary to merge into (modified in-place).
        override: Dictionary with values to override/add.

    Example:
        >>> base = {"a": 1, "nested": {"b": 2}}
        >>> deep_merge_inplace(base, {"nested": {"c": 3}})
        >>> base
        {"a": 1, "nested": {"b": 2, "c": 3}}
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            # Recursively merge nested dicts
            deep_merge_inplace(base[key], value)
        else:
            # Override the value
            base[key] = value
