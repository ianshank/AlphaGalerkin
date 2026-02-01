"""Tests for physics zero-shot transfer demo.

Tests cover:
- Sample generation
- Transfer evaluation
- Visualization outputs
- Result dataclass
"""

from __future__ import annotations

import numpy as np
import pytest

from src.demos.config import PhysicsDemoConfig
from src.demos.physics_demo import PhysicsDemo, TransferResult


class TestTransferResult:
    """Tests for TransferResult dataclass."""

    def test_creation(self) -> None:
        """Test TransferResult creation."""
        result = TransferResult(
            grid_size=9,
            ground_truth=np.zeros((9, 9), dtype=np.float32),
            prediction=np.zeros((9, 9), dtype=np.float32),
            mse=0.001,
            mae=0.02,
            inference_time_ms=5.0,
        )

        assert result.grid_size == 9
        assert result.mse == 0.001
        assert result.mae == 0.02
        assert result.inference_time_ms == 5.0

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        gt = np.ones((9, 9), dtype=np.float32)
        pred = np.ones((9, 9), dtype=np.float32) * 0.9

        result = TransferResult(
            grid_size=9,
            ground_truth=gt,
            prediction=pred,
            mse=0.01,
            mae=0.1,
            inference_time_ms=10.0,
        )

        d = result.to_dict()
        assert d["grid_size"] == 9
        assert d["mse"] == 0.01
        assert d["mae"] == 0.1
        assert "ground_truth" in d
        assert "prediction" in d


class TestPhysicsDemo:
    """Tests for PhysicsDemo class."""

    @pytest.fixture
    def demo(self) -> PhysicsDemo:
        """Create a PhysicsDemo instance without model."""
        return PhysicsDemo()

    @pytest.fixture
    def demo_custom_config(self) -> PhysicsDemo:
        """Create a PhysicsDemo with custom config."""
        config = PhysicsDemoConfig(
            train_grid_size=7,
            eval_grid_sizes=[7, 9, 13],
            n_charges=3,
        )
        return PhysicsDemo(config)

    def test_initialization(self, demo: PhysicsDemo) -> None:
        """Test demo initialization."""
        assert demo.model is None
        assert demo.device == "cpu"
        assert demo.config.train_grid_size == 9
        assert demo.field_viz is not None
        assert demo.chart_viz is not None

    def test_custom_config(self, demo_custom_config: PhysicsDemo) -> None:
        """Test demo with custom configuration."""
        assert demo_custom_config.config.train_grid_size == 7
        assert demo_custom_config.config.eval_grid_sizes == [7, 9, 13]
        assert demo_custom_config.config.n_charges == 3

    def test_solver_lazy_init(self, demo: PhysicsDemo) -> None:
        """Test solver is lazily initialized."""
        assert demo._solver is None
        solver = demo.solver
        assert solver is not None
        assert demo._solver is not None

    def test_generate_sample(self, demo: PhysicsDemo) -> None:
        """Test sample generation."""
        grid_size = 9
        charges, potential, coords = demo.generate_sample(grid_size, seed=42)

        # Check shapes
        assert charges.shape == (grid_size * grid_size,)
        assert potential.shape == (grid_size * grid_size,)
        assert coords.shape == (grid_size * grid_size, 2)

        # Check coords are in [0, 1]
        assert coords.min() >= 0.0
        assert coords.max() <= 1.0

        # Check data types
        assert charges.dtype == np.float32
        assert potential.dtype == np.float32
        assert coords.dtype == np.float32

    def test_generate_sample_reproducible(self, demo: PhysicsDemo) -> None:
        """Test sample generation is reproducible with same seed."""
        c1, p1, _ = demo.generate_sample(9, seed=42)
        c2, p2, _ = demo.generate_sample(9, seed=42)

        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(p1, p2)

    def test_generate_sample_different_seeds(self, demo: PhysicsDemo) -> None:
        """Test different seeds produce different samples."""
        c1, _, _ = demo.generate_sample(9, seed=42)
        c2, _, _ = demo.generate_sample(9, seed=123)

        assert not np.allclose(c1, c2)

    def test_generate_sample_different_sizes(self, demo: PhysicsDemo) -> None:
        """Test sample generation for different grid sizes."""
        for size in [5, 9, 13, 19]:
            charges, potential, coords = demo.generate_sample(size, seed=42)
            n_points = size * size

            assert charges.shape == (n_points,)
            assert potential.shape == (n_points,)
            assert coords.shape == (n_points, 2)

    def test_predict_without_model(self, demo: PhysicsDemo) -> None:
        """Test predict returns zeros without model."""
        coords = np.random.rand(81, 2).astype(np.float32)
        charges = np.random.randn(81).astype(np.float32)

        prediction, time_ms = demo.predict(coords, charges)

        assert prediction.shape == charges.shape
        np.testing.assert_array_equal(prediction, np.zeros_like(charges))
        assert time_ms == 0.0

    def test_evaluate_transfer_without_model(self, demo: PhysicsDemo) -> None:
        """Test transfer evaluation without model."""
        result = demo.evaluate_transfer(grid_size=9, seed=42)

        assert isinstance(result, TransferResult)
        assert result.grid_size == 9
        assert result.ground_truth.shape == (9, 9)
        assert result.prediction.shape == (9, 9)
        # Without model, prediction is zeros, so MSE equals variance of ground truth
        assert result.mse > 0

    def test_run_transfer_evaluation(self, demo: PhysicsDemo) -> None:
        """Test full transfer evaluation across sizes."""
        results = demo.run_transfer_evaluation(seed=42)

        assert len(results) == 3  # Default: [9, 13, 19]
        assert 9 in results
        assert 13 in results
        assert 19 in results

        for size, result in results.items():
            assert result.grid_size == size
            assert result.ground_truth.shape == (size, size)

    def test_visualize_sample(self, demo: PhysicsDemo) -> None:
        """Test sample visualization."""
        charges_img, potential_img, info = demo.visualize_sample(
            grid_size=9,
            seed=42,
        )

        assert charges_img.ndim == 3
        assert charges_img.shape[2] == 3
        assert potential_img.ndim == 3
        assert "Grid Size: 9" in info

    def test_visualize_transfer(self, demo: PhysicsDemo) -> None:
        """Test transfer visualization."""
        comparison_img, mse_chart_img, results_text = demo.visualize_transfer(seed=42)

        assert comparison_img.ndim == 3
        assert mse_chart_img.ndim == 3
        assert "Zero-Shot Transfer Results" in results_text

    def test_demonstrate_resolution_independence(self, demo: PhysicsDemo) -> None:
        """Test resolution independence demonstration."""
        comparison_img, explanation = demo.demonstrate_resolution_independence(
            small_size=9,
            large_size=19,
            seed=42,
        )

        assert comparison_img.ndim == 3
        assert "Resolution Independence" in explanation
        assert "9×9" in explanation
        assert "19×19" in explanation


class TestPhysicsDemoCustomConfig:
    """Tests for PhysicsDemo with various configurations."""

    def test_different_n_charges(self) -> None:
        """Test with different numbers of point charges."""
        for n_charges in [1, 5, 10]:
            config = PhysicsDemoConfig(n_charges=n_charges)
            demo = PhysicsDemo(config)

            charges, _, _ = demo.generate_sample(9, seed=42)
            # Charges should be generated (actual distribution depends on solver)
            assert charges.shape == (81,)

    def test_continuous_charges(self) -> None:
        """Test with continuous charge field (n_charges=None)."""
        config = PhysicsDemoConfig(n_charges=None)
        demo = PhysicsDemo(config)

        # This would need to be supported by the config - skip if not
        # For now, just verify the demo can be created
        assert demo.config.n_charges is None

    def test_custom_eval_sizes(self) -> None:
        """Test with custom evaluation sizes."""
        config = PhysicsDemoConfig(
            eval_grid_sizes=[7, 11, 15],
            max_grid_size=32,
        )
        demo = PhysicsDemo(config)

        results = demo.run_transfer_evaluation(seed=42)
        assert len(results) == 3
        assert 7 in results
        assert 11 in results
        assert 15 in results

    def test_mse_threshold_in_output(self) -> None:
        """Test MSE threshold appears in output."""
        config = PhysicsDemoConfig(mse_threshold=0.01)
        demo = PhysicsDemo(config)

        _, _, results_text = demo.visualize_transfer(seed=42)
        assert "Threshold: 0.01" in results_text


class TestPhysicsDemoEdgeCases:
    """Edge case tests for PhysicsDemo."""

    def test_minimum_grid_size(self) -> None:
        """Test with minimum grid size."""
        config = PhysicsDemoConfig(
            train_grid_size=5,
            eval_grid_sizes=[5],
        )
        demo = PhysicsDemo(config)

        charges, potential, coords = demo.generate_sample(5, seed=42)
        assert charges.shape == (25,)

    def test_large_grid_size(self) -> None:
        """Test with larger grid size."""
        config = PhysicsDemoConfig(
            max_grid_size=32,
            eval_grid_sizes=[32],
        )
        demo = PhysicsDemo(config)

        charges, potential, coords = demo.generate_sample(32, seed=42)
        assert charges.shape == (1024,)

    def test_seed_propagation(self) -> None:
        """Test seed is properly propagated to solvers."""
        demo = PhysicsDemo()

        # Same seed should give same results
        r1 = demo.evaluate_transfer(9, seed=123)
        r2 = demo.evaluate_transfer(9, seed=123)

        np.testing.assert_array_almost_equal(r1.ground_truth, r2.ground_truth)
