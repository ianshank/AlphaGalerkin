"""Tests for seeding utilities (utils/seeding.py)."""
from __future__ import annotations

import random

import numpy as np
import torch

from src.alphagalerkin.utils.seeding import get_rng, seed_everything


class TestSeedEverything:
    """seed_everything sets deterministic state."""

    def test_python_random_deterministic(self) -> None:
        seed_everything(seed=123)
        a = random.random()
        seed_everything(seed=123)
        b = random.random()

        assert a == b

    def test_numpy_deterministic(self) -> None:
        seed_everything(seed=456)
        a = np.random.rand(5)
        seed_everything(seed=456)
        b = np.random.rand(5)

        np.testing.assert_array_equal(a, b)

    def test_torch_deterministic(self) -> None:
        seed_everything(seed=789)
        a = torch.rand(5)
        seed_everything(seed=789)
        b = torch.rand(5)

        assert torch.equal(a, b)

    def test_different_seeds_produce_different_values(self) -> None:
        seed_everything(seed=1)
        a = torch.rand(10)
        seed_everything(seed=2)
        b = torch.rand(10)

        assert not torch.equal(a, b)

    def test_default_seed(self) -> None:
        # Should not raise with default argument.
        seed_everything()
        val = torch.rand(1)
        assert val.shape == (1,)


class TestGetRng:
    """get_rng returns a numpy Generator."""

    def test_returns_generator(self) -> None:
        rng = get_rng(seed=42)

        assert isinstance(rng, np.random.Generator)

    def test_seeded_deterministic(self) -> None:
        a = get_rng(seed=99).random(5)
        b = get_rng(seed=99).random(5)

        np.testing.assert_array_equal(a, b)

    def test_none_seed_returns_generator(self) -> None:
        rng = get_rng(seed=None)

        assert isinstance(rng, np.random.Generator)
        # Should be able to generate numbers without error.
        val = rng.random()
        assert 0.0 <= val < 1.0

    def test_different_seeds_differ(self) -> None:
        a = get_rng(seed=10).random(20)
        b = get_rng(seed=11).random(20)

        assert not np.array_equal(a, b)
