"""Tests for collocation point allocation strategies."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.collocation import (
    AdaptiveAllocator,
    CollocationRegistry,
    ErrorGuidedAllocator,
    ImportanceWeightedAllocator,
    UniformAllocator,
    create_collocation_allocator,
)
from src.agents.config import CollocationConfig, CollocationStrategy


class TestUniformAllocator:
    """Tests for UniformAllocator."""

    def test_correct_shape(self, sample_collocation_config: CollocationConfig) -> None:
        allocator = UniformAllocator(sample_collocation_config)
        points = allocator.allocate(
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            dim=2,
            seed=42,
        )
        assert points.shape == (100, 2)
        assert points.dtype == np.float32

    def test_within_bounds(self, sample_collocation_config: CollocationConfig) -> None:
        allocator = UniformAllocator(sample_collocation_config)
        points = allocator.allocate(
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            dim=2,
            seed=42,
        )
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)

    def test_custom_bounds(self, sample_collocation_config: CollocationConfig) -> None:
        allocator = UniformAllocator(sample_collocation_config)
        points = allocator.allocate(
            domain_min=[-1.0, -2.0],
            domain_max=[3.0, 4.0],
            dim=2,
            seed=42,
        )
        assert np.all(points[:, 0] >= -1.0)
        assert np.all(points[:, 0] <= 3.0)
        assert np.all(points[:, 1] >= -2.0)
        assert np.all(points[:, 1] <= 4.0)

    def test_deterministic_with_seed(self, sample_collocation_config: CollocationConfig) -> None:
        allocator = UniformAllocator(sample_collocation_config)
        p1 = allocator.allocate([0.0], [1.0], dim=1, seed=42)
        p2 = allocator.allocate([0.0], [1.0], dim=1, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_1d(self, sample_collocation_config: CollocationConfig) -> None:
        allocator = UniformAllocator(sample_collocation_config)
        points = allocator.allocate([0.0], [1.0], dim=1, seed=42)
        assert points.shape == (100, 1)

    def test_3d(self, sample_collocation_config: CollocationConfig) -> None:
        allocator = UniformAllocator(sample_collocation_config)
        points = allocator.allocate([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], dim=3, seed=42)
        assert points.shape == (100, 3)


class TestAdaptiveAllocator:
    """Tests for AdaptiveAllocator."""

    def test_fallback_to_uniform(self, sample_collocation_config: CollocationConfig) -> None:
        config = sample_collocation_config.with_overrides(strategy=CollocationStrategy.ADAPTIVE)
        allocator = AdaptiveAllocator(config)
        points = allocator.allocate([0.0, 0.0], [1.0, 1.0], dim=2, seed=42)
        assert points.shape == (100, 2)

    def test_concentrates_near_high_residual(self) -> None:
        config = CollocationConfig(
            name="adaptive",
            strategy=CollocationStrategy.ADAPTIVE,
            n_points=200,
            adaptation_rate=0.8,
        )
        allocator = AdaptiveAllocator(config)

        coords = np.random.default_rng(42).random((50, 2)).astype(np.float32)
        residuals = np.zeros(50, dtype=np.float32)
        residuals[0] = 100.0  # One very high residual at coords[0]

        points = allocator.allocate(
            [0.0, 0.0],
            [1.0, 1.0],
            dim=2,
            residuals=residuals,
            coords=coords,
            seed=42,
        )
        assert points.shape == (200, 2)

        # Points should cluster near coords[0]
        distances = np.linalg.norm(points - coords[0], axis=1)
        near_point = np.sum(distances < 0.2)
        assert near_point > 10  # Significantly more than uniform

    def test_zero_residual_fallback(self) -> None:
        config = CollocationConfig(name="adaptive", n_points=50)
        allocator = AdaptiveAllocator(config)
        coords = np.random.default_rng(42).random((20, 2)).astype(np.float32)
        residuals = np.zeros(20, dtype=np.float32)
        points = allocator.allocate(
            [0.0, 0.0],
            [1.0, 1.0],
            dim=2,
            residuals=residuals,
            coords=coords,
            seed=42,
        )
        assert points.shape == (50, 2)

    def test_within_bounds(self) -> None:
        config = CollocationConfig(name="adaptive", n_points=100, adaptation_rate=0.8)
        allocator = AdaptiveAllocator(config)
        coords = np.random.default_rng(42).random((30, 2)).astype(np.float32)
        residuals = np.random.default_rng(42).random(30).astype(np.float32)
        points = allocator.allocate(
            [0.0, 0.0],
            [1.0, 1.0],
            dim=2,
            residuals=residuals,
            coords=coords,
            seed=42,
        )
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)


class TestImportanceWeightedAllocator:
    """Tests for ImportanceWeightedAllocator."""

    def test_correct_shape(self) -> None:
        config = CollocationConfig(name="iw", n_points=50)
        allocator = ImportanceWeightedAllocator(config)
        coords = np.random.default_rng(42).random((30, 2)).astype(np.float32)
        residuals = np.random.default_rng(42).random(30).astype(np.float32)
        points = allocator.allocate(
            [0.0, 0.0],
            [1.0, 1.0],
            dim=2,
            residuals=residuals,
            coords=coords,
            seed=42,
        )
        assert points.shape == (50, 2)

    def test_fallback_without_residuals(self) -> None:
        config = CollocationConfig(name="iw", n_points=50)
        allocator = ImportanceWeightedAllocator(config)
        points = allocator.allocate([0.0, 0.0], [1.0, 1.0], dim=2, seed=42)
        assert points.shape == (50, 2)


class TestErrorGuidedAllocator:
    """Tests for ErrorGuidedAllocator."""

    def test_correct_shape(self) -> None:
        config = CollocationConfig(name="eg", n_points=50, adaptation_rate=0.5)
        allocator = ErrorGuidedAllocator(config)
        coords = np.random.default_rng(42).random((30, 2)).astype(np.float32)
        residuals = np.random.default_rng(42).random(30).astype(np.float32)
        points = allocator.allocate(
            [0.0, 0.0],
            [1.0, 1.0],
            dim=2,
            residuals=residuals,
            coords=coords,
            seed=42,
        )
        assert points.shape[0] == 50
        assert points.shape[1] == 2

    def test_within_bounds(self) -> None:
        config = CollocationConfig(name="eg", n_points=100, adaptation_rate=0.5)
        allocator = ErrorGuidedAllocator(config)
        coords = np.random.default_rng(42).random((50, 2)).astype(np.float32)
        residuals = np.random.default_rng(42).random(50).astype(np.float32)
        points = allocator.allocate(
            [0.0, 0.0],
            [1.0, 1.0],
            dim=2,
            residuals=residuals,
            coords=coords,
            seed=42,
        )
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)

    def test_fallback_without_residuals(self) -> None:
        config = CollocationConfig(name="eg", n_points=30)
        allocator = ErrorGuidedAllocator(config)
        points = allocator.allocate([0.0], [1.0], dim=1, seed=42)
        assert points.shape == (30, 1)


class TestImportanceWeightedZeroResiduals:
    """Tests for importance-weighted allocator with zero residuals."""

    def test_zero_residuals_falls_back_to_uniform(self) -> None:
        """All-zero residuals should fall back to uniform allocation."""
        config = CollocationConfig(name="iw_zero", n_points=50)
        allocator = ImportanceWeightedAllocator(config)
        coords = np.random.default_rng(42).random((30, 2)).astype(np.float32)
        residuals = np.zeros(30, dtype=np.float32)
        points = allocator.allocate(
            [0.0, 0.0],
            [1.0, 1.0],
            dim=2,
            residuals=residuals,
            coords=coords,
            seed=42,
        )
        assert points.shape == (50, 2)
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)


class TestErrorGuidedEdgeCases:
    """Edge case tests for ErrorGuidedAllocator."""

    def test_empty_high_error_region(self) -> None:
        """When all residuals are below threshold, fall back to uniform."""
        config = CollocationConfig(
            name="eg_edge",
            n_points=30,
            adaptation_rate=0.01,
        )
        allocator = ErrorGuidedAllocator(config)
        coords = np.random.default_rng(42).random((50, 2)).astype(np.float32)
        # Very uniform residuals so threshold leaves nothing
        residuals = np.full(50, 0.001, dtype=np.float32)
        points = allocator.allocate(
            [0.0, 0.0],
            [1.0, 1.0],
            dim=2,
            residuals=residuals,
            coords=coords,
            seed=42,
        )
        assert points.shape[0] == 30

    def test_perturbation_stays_in_bounds(self) -> None:
        """Verify perturbed points are clamped within domain bounds."""
        config = CollocationConfig(
            name="eg_bounds",
            n_points=100,
            adaptation_rate=0.9,
        )
        allocator = ErrorGuidedAllocator(config)
        # Put high-error coordinates near domain boundary
        coords = np.ones((20, 2), dtype=np.float32) * 0.99
        residuals = np.ones(20, dtype=np.float32) * 10.0
        points = allocator.allocate(
            [0.0, 0.0],
            [1.0, 1.0],
            dim=2,
            residuals=residuals,
            coords=coords,
            seed=42,
        )
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)


class TestReallocate:
    """Tests for reallocate convenience method."""

    def test_reallocate_calls_allocate(self) -> None:
        config = CollocationConfig(name="realloc", n_points=50, seed=42)
        allocator = UniformAllocator(config)
        coords = np.random.default_rng(42).random((20, 2)).astype(np.float32)
        residuals = np.random.default_rng(42).random(20).astype(np.float32)
        points = allocator.reallocate(
            coords, residuals, domain_min=[0.0, 0.0], domain_max=[1.0, 1.0]
        )
        assert points.shape == (50, 2)


class TestCollocationFactory:
    """Tests for create_collocation_allocator factory."""

    @pytest.mark.parametrize("strategy", list(CollocationStrategy))
    def test_all_strategies(self, strategy: CollocationStrategy) -> None:
        config = CollocationConfig(name="factory", strategy=strategy, n_points=20)
        allocator = create_collocation_allocator(config)
        assert allocator is not None
        points = allocator.allocate([0.0, 0.0], [1.0, 1.0], dim=2, seed=42)
        assert points.shape[0] == 20

    def test_registry_populated(self) -> None:
        items = CollocationRegistry().list_items()
        assert "uniform" in items
        assert "adaptive" in items
        assert "importance_weighted" in items
        assert "error_guided" in items
