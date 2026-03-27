"""Tests for JAX PRNG KeyManager.

Covers initialization, next/split/reset/current methods, and error handling.
JAX is mocked to avoid requiring it at test time.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_jax_random():
    """Create a mock jax.random module."""
    mock_random = MagicMock()
    mock_random.PRNGKey.side_effect = lambda s: np.array([s, 0])

    def _split(key, num=2):
        return [np.array([i, int(key[0])]) for i in range(num)]

    mock_random.split.side_effect = _split
    return mock_random


@pytest.fixture
def key_manager(mock_jax_random):
    """Create a KeyManager with mocked JAX."""
    with patch("src.backend.rng._get_jax_random", return_value=mock_jax_random):
        from src.backend.rng import KeyManager

        km = KeyManager(seed=42)
        yield km, mock_jax_random


class TestKeyManager:
    """Test KeyManager with mocked JAX."""

    def test_init(self, key_manager) -> None:
        km, mock_random = key_manager
        mock_random.PRNGKey.assert_called_once_with(42)

    def test_next_returns_subkey(self, key_manager) -> None:
        km, mock_random = key_manager
        subkey = km.next()
        assert subkey is not None
        mock_random.split.assert_called_once()

    def test_next_advances_state(self, key_manager) -> None:
        km, mock_random = key_manager
        km.next()
        km.next()
        assert mock_random.split.call_count == 2

    def test_split_returns_list(self, mock_jax_random) -> None:
        with patch(
            "src.backend.rng._get_jax_random",
            return_value=mock_jax_random,
        ):
            from src.backend.rng import KeyManager

            km = KeyManager(seed=0)
            keys = km.split(num=4)
            assert isinstance(keys, list)
            assert len(keys) == 4

    def test_split_invalid_num(self, key_manager) -> None:
        km, _ = key_manager
        with pytest.raises(ValueError, match="num must be >= 1"):
            km.split(num=0)

    def test_split_num_one(self, mock_jax_random) -> None:
        with patch(
            "src.backend.rng._get_jax_random",
            return_value=mock_jax_random,
        ):
            from src.backend.rng import KeyManager

            km = KeyManager(seed=0)
            keys = km.split(num=1)
            assert len(keys) == 1

    def test_reset(self, key_manager) -> None:
        km, mock_random = key_manager
        km.next()
        km.reset(seed=99)
        assert mock_random.PRNGKey.call_count == 2
        mock_random.PRNGKey.assert_called_with(99)

    def test_current_property(self, key_manager) -> None:
        km, _ = key_manager
        current = km.current
        assert current is not None

    def test_current_does_not_advance(self, key_manager) -> None:
        km, mock_random = key_manager
        _ = km.current
        _ = km.current
        mock_random.split.assert_not_called()


class TestGetJaxRandom:
    """Test the _get_jax_random helper."""

    def test_import_error_message(self) -> None:
        """When JAX is not installed, a helpful message is raised."""
        import sys

        with patch.dict(sys.modules, {"jax": None, "jax.random": None}):
            # Need to reimport the module to trigger fresh import
            import importlib

            import src.backend.rng as rng_mod

            importlib.reload(rng_mod)

            with pytest.raises(ImportError, match="JAX is required"):
                rng_mod._get_jax_random()
