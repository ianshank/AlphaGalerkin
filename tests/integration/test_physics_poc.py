"""Integration tests for Physics PoC (Proof of Concept).

Tests the complete physics pipeline:
1. Poisson solver correctness
2. PhysicsOperator training
3. Zero-shot transfer validation (train on 9x9 → evaluate on 19x19)

Success criterion from CLAUDE.md: MSE < 0.05 on 19x19 without retraining.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.experiments.physics_model import GalerkinBlock, PhysicsLoss, PhysicsOperator
from src.physics.poisson import (
    PoissonDataset,
    PoissonSample,
    PoissonSolver,
    generate_influence_field,
    generate_random_charges,
)


class TestPoissonSolver:
    """Tests for Poisson equation solver correctness."""

    def test_zero_charge_zero_potential(self) -> None:
        """Zero charge density should give zero potential."""
        solver = PoissonSolver()
        charges = np.zeros((9, 9), dtype=np.float32)
        potential = solver.solve(charges)

        assert np.allclose(potential, 0.0, atol=1e-10)

    def test_constant_charge_smooth_potential(self) -> None:
        """Constant charge should give smooth, non-zero potential."""
        solver = PoissonSolver()
        charges = np.ones((9, 9), dtype=np.float32)
        potential = solver.solve(charges)

        # Potential should be non-zero
        assert np.abs(potential).max() > 0.0

        # For Poisson equation ∇²φ = ρ with ρ > 0 and zero Dirichlet BC,
        # the potential is negative everywhere. The magnitude (abs value)
        # should be largest at the center and smaller at the corners.
        center_magnitude = np.abs(potential[4, 4])
        corner_magnitudes = [
            np.abs(potential[0, 0]),
            np.abs(potential[0, 8]),
            np.abs(potential[8, 0]),
            np.abs(potential[8, 8]),
        ]
        assert center_magnitude > max(corner_magnitudes)

    def test_symmetry_preserved(self) -> None:
        """Symmetric charge distribution should give symmetric potential."""
        solver = PoissonSolver()

        # Create symmetric charge (point charge in center)
        charges = np.zeros((9, 9), dtype=np.float32)
        charges[4, 4] = 1.0

        potential = solver.solve(charges)

        # Check 4-fold symmetry
        assert np.allclose(potential, potential[::-1, :], atol=1e-6)
        assert np.allclose(potential, potential[:, ::-1], atol=1e-6)

    def test_spectral_vs_iterative_agreement(self) -> None:
        """Spectral and iterative methods should agree (approximately)."""
        charges = generate_random_charges(grid_size=7, n_charges=3, seed=42)

        solver_spectral = PoissonSolver(use_spectral=True)
        solver_iterative = PoissonSolver(use_spectral=False)

        potential_spectral = solver_spectral.solve(charges)
        potential_iterative = solver_iterative.solve(charges)

        # Should agree within tolerance (iterative is less accurate)
        correlation = np.corrcoef(potential_spectral.flatten(), potential_iterative.flatten())[0, 1]
        assert correlation > 0.95

    def test_dst2d_inverse_relation(self) -> None:
        """DST transform should be invertible."""
        solver = PoissonSolver()
        x = np.random.randn(8, 8).astype(np.float32)

        # DST-I is self-inverse with normalization factor
        transformed = solver._dst2d(x)
        recovered = solver._idst2d(transformed)

        assert np.allclose(x, recovered, atol=1e-5)

    def test_different_grid_sizes(self) -> None:
        """Solver should work for various grid sizes."""
        solver = PoissonSolver()

        for grid_size in [5, 9, 13, 17, 25]:
            charges = np.random.randn(grid_size, grid_size).astype(np.float32)
            potential = solver.solve(charges)

            assert potential.shape == (grid_size, grid_size)
            assert np.isfinite(potential).all()

    def test_flattened_input_output(self) -> None:
        """Solver should handle flattened inputs."""
        solver = PoissonSolver()

        charges_2d = np.random.randn(9, 9).astype(np.float32)
        charges_1d = charges_2d.flatten()

        potential_from_2d = solver.solve(charges_2d)
        potential_from_1d = solver.solve(charges_1d)

        assert np.allclose(potential_from_2d.flatten(), potential_from_1d)


class TestPoissonDataGeneration:
    """Tests for Poisson data generation utilities."""

    def test_generate_influence_field_structure(self) -> None:
        """Generated sample should have correct structure."""
        sample = generate_influence_field(grid_size=9, n_charges=5, seed=42)

        assert isinstance(sample, PoissonSample)
        assert sample.coords.shape == (81, 2)  # 9x9 grid
        assert sample.charges.shape == (81,)
        assert sample.potential.shape == (81,)
        assert sample.grid_size == 9

    def test_generate_reproducible_with_seed(self) -> None:
        """Same seed should produce identical samples."""
        sample1 = generate_influence_field(grid_size=9, n_charges=5, seed=123)
        sample2 = generate_influence_field(grid_size=9, n_charges=5, seed=123)

        assert np.allclose(sample1.coords, sample2.coords)
        assert np.allclose(sample1.charges, sample2.charges)
        assert np.allclose(sample1.potential, sample2.potential)

    def test_dataset_length(self) -> None:
        """Dataset should have correct length."""
        dataset = PoissonDataset(grid_size=9, n_samples=100, seed=42)
        assert len(dataset) == 100

    def test_dataset_indexing(self) -> None:
        """Dataset indexing should return valid samples."""
        dataset = PoissonDataset(grid_size=9, n_samples=10, seed=42)

        sample = dataset[0]
        assert isinstance(sample, PoissonSample)

        # Different indices should give different samples
        sample0 = dataset[0]
        sample1 = dataset[1]
        assert not np.allclose(sample0.charges, sample1.charges)


class TestPhysicsOperator:
    """Tests for PhysicsOperator neural network."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get computation device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward_shape(self, device: torch.device) -> None:
        """Forward pass should produce correct output shape."""
        model = PhysicsOperator(d_model=64, n_layers=2).to(device)

        batch_size, n_points = 4, 81
        coords = torch.rand(batch_size, n_points, 2, device=device)
        charges = torch.randn(batch_size, n_points, device=device)

        output = model(coords, charges)

        assert output.shape == (batch_size, n_points)

    def test_forward_finite(self, device: torch.device) -> None:
        """Forward pass should produce finite values."""
        model = PhysicsOperator(d_model=64, n_layers=2).to(device)

        coords = torch.rand(2, 49, 2, device=device)
        charges = torch.randn(2, 49, device=device)

        output = model(coords, charges)

        assert output.isfinite().all()

    def test_gradient_flow(self, device: torch.device) -> None:
        """Gradients should flow through the model."""
        model = PhysicsOperator(d_model=64, n_layers=2).to(device)

        coords = torch.rand(2, 49, 2, device=device)
        charges = torch.randn(2, 49, device=device, requires_grad=True)

        output = model(coords, charges)
        loss = output.sum()
        loss.backward()

        assert charges.grad is not None
        assert charges.grad.isfinite().all()

    def test_resolution_independence(self, device: torch.device) -> None:
        """Model should work for different resolutions without reinitialization."""
        model = PhysicsOperator(d_model=64, n_layers=2).to(device)
        model.eval()

        # Test multiple resolutions
        for n_points in [25, 49, 81, 169, 361]:
            coords = torch.rand(1, n_points, 2, device=device)
            charges = torch.randn(1, n_points, device=device)

            with torch.no_grad():
                output = model(coords, charges)

            assert output.shape == (1, n_points)
            assert output.isfinite().all()

    def test_deterministic_eval_mode(self, device: torch.device) -> None:
        """Model should be deterministic in eval mode."""
        model = PhysicsOperator(d_model=64, n_layers=2, dropout=0.1).to(device)
        model.eval()

        coords = torch.rand(2, 49, 2, device=device)
        charges = torch.randn(2, 49, device=device)

        with torch.no_grad():
            output1 = model(coords, charges)
            output2 = model(coords, charges)

        assert torch.allclose(output1, output2)


class TestGalerkinBlock:
    """Tests for Galerkin attention block."""

    def test_shape_preservation(self) -> None:
        """Galerkin block should preserve shape."""
        block = GalerkinBlock(d_model=64, n_heads=4)

        x = torch.randn(2, 81, 64)
        output = block(x)

        assert output.shape == x.shape

    def test_residual_connection(self) -> None:
        """Output should be related to input (residual connection)."""
        block = GalerkinBlock(d_model=64, n_heads=4)

        x = torch.randn(2, 81, 64)
        output = block(x)

        # With residual, output shouldn't be zero when input is non-zero
        assert output.abs().mean() > 0


class TestPhysicsLoss:
    """Tests for physics loss function."""

    def test_mse_computation(self) -> None:
        """Loss should compute MSE correctly."""
        loss_fn = PhysicsLoss()

        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.1, 2.2, 2.8]])

        loss = loss_fn(pred, target)

        expected = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(loss, expected)

    def test_loss_positive(self) -> None:
        """Loss should be non-negative."""
        loss_fn = PhysicsLoss()

        pred = torch.randn(4, 81)
        target = torch.randn(4, 81)

        loss = loss_fn(pred, target)

        assert loss >= 0


class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get computation device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_single_training_step(self, device: torch.device) -> None:
        """Model should be able to perform a training step."""
        model = PhysicsOperator(d_model=64, n_layers=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = PhysicsLoss()

        # Generate a batch
        dataset = PoissonDataset(grid_size=9, n_samples=4, seed=42)
        samples = [dataset[i] for i in range(4)]

        coords = torch.tensor(np.stack([s.coords for s in samples]), device=device)
        charges = torch.tensor(np.stack([s.charges for s in samples]), device=device)
        targets = torch.tensor(np.stack([s.potential for s in samples]), device=device)

        # Training step
        model.train()
        optimizer.zero_grad()
        predictions = model(coords, charges)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        assert loss.isfinite()

    def test_loss_decreases(self, device: torch.device) -> None:
        """Loss should decrease over multiple training steps."""
        torch.manual_seed(42)
        np.random.seed(42)

        model = PhysicsOperator(d_model=64, n_layers=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = PhysicsLoss()

        dataset = PoissonDataset(grid_size=9, n_samples=8, seed=42)
        samples = [dataset[i] for i in range(8)]

        coords = torch.tensor(np.stack([s.coords for s in samples]), device=device)
        charges = torch.tensor(np.stack([s.charges for s in samples]), device=device)
        targets = torch.tensor(np.stack([s.potential for s in samples]), device=device)

        losses = []
        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            predictions = model(coords, charges)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease
        assert losses[-1] < losses[0]

    @pytest.mark.slow
    def test_zero_shot_transfer_smoke(self, device: torch.device) -> None:
        """Smoke test for zero-shot transfer (train 9x9 → eval 19x19).

        This is a reduced test; full verification is in verify_transfer.py.
        """
        torch.manual_seed(42)
        np.random.seed(42)

        # Small model for fast test
        model = PhysicsOperator(d_model=32, n_layers=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = PhysicsLoss()

        # Training on 9x9
        train_dataset = PoissonDataset(grid_size=9, n_samples=20, seed=42)

        model.train()
        for _epoch in range(10):
            for i in range(len(train_dataset)):
                sample = train_dataset[i]
                coords = torch.tensor(sample.coords[None], device=device)
                charges = torch.tensor(sample.charges[None], device=device)
                targets = torch.tensor(sample.potential[None], device=device)

                optimizer.zero_grad()
                pred = model(coords, charges)
                loss = loss_fn(pred, targets)
                loss.backward()
                optimizer.step()

        # Evaluate on 19x19 (zero-shot)
        model.eval()
        eval_dataset = PoissonDataset(grid_size=19, n_samples=5, seed=100)

        mse_values = []
        with torch.no_grad():
            for i in range(len(eval_dataset)):
                sample = eval_dataset[i]
                coords = torch.tensor(sample.coords[None], device=device)
                charges = torch.tensor(sample.charges[None], device=device)
                targets = torch.tensor(sample.potential[None], device=device)

                pred = model(coords, charges)
                mse = torch.nn.functional.mse_loss(pred, targets).item()
                mse_values.append(mse)

        avg_mse = np.mean(mse_values)

        # Smoke test: just verify it runs and produces finite values
        # Full accuracy tests are in verify_transfer.py
        assert np.isfinite(avg_mse)
        assert avg_mse < 100.0  # Very loose bound for smoke test


class TestEdgeCases:
    """Edge case tests for numerical stability."""

    def test_very_small_charges(self) -> None:
        """Model should handle very small charge magnitudes."""
        solver = PoissonSolver()
        charges = np.random.randn(9, 9).astype(np.float32) * 1e-8
        potential = solver.solve(charges)

        assert np.isfinite(potential).all()

    def test_very_large_charges(self) -> None:
        """Model should handle large charge magnitudes."""
        solver = PoissonSolver()
        charges = np.random.randn(9, 9).astype(np.float32) * 1e6
        potential = solver.solve(charges)

        assert np.isfinite(potential).all()

    def test_sparse_charges(self) -> None:
        """Model should handle sparse charge distributions."""
        solver = PoissonSolver()
        charges = np.zeros((13, 13), dtype=np.float32)
        charges[6, 6] = 1.0  # Single point charge

        potential = solver.solve(charges)

        assert np.isfinite(potential).all()
        # For positive point charge with zero BC, potential is negative,
        # with maximum magnitude (most negative) near the charge location.
        # Check that potential at charge location has significant magnitude.
        assert np.abs(potential[6, 6]) > 0
