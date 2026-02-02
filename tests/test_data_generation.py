"""Tests for physics data generation infrastructure.

Tests:
- PhysicsDataset functionality
- Reproducibility with fixed seeds
- Train/val/test split disjointness
- Normalization correctness
"""

import pytest
import torch

from src.data.physics_dataset import PhysicsDataset
from src.physics.darcy import DarcyFlowSolver
from src.physics.heat import HeatSolver
from src.physics.poisson import PoissonSolver


class TestPhysicsDataset:
    """Test suite for PhysicsDataset."""

    @pytest.fixture
    def darcy_solver(self) -> DarcyFlowSolver:
        return DarcyFlowSolver(resolution=16)

    def test_dataset_length(self, darcy_solver: DarcyFlowSolver) -> None:
        """Test dataset returns correct length."""
        dataset = PhysicsDataset(darcy_solver, n_samples=50, seed=42)
        assert len(dataset) == 50

    def test_dataset_item_structure(self, darcy_solver: DarcyFlowSolver) -> None:
        """Test dataset returns correct tensor structure."""
        dataset = PhysicsDataset(darcy_solver, n_samples=10, seed=42)
        sample = dataset[0]

        assert "input" in sample
        assert "output" in sample
        assert "coords" in sample
        assert "grid_size" in sample

        assert isinstance(sample["input"], torch.Tensor)
        assert isinstance(sample["output"], torch.Tensor)
        assert sample["input"].dtype == torch.float32

    def test_reproducibility_with_seed(self, darcy_solver: DarcyFlowSolver) -> None:
        """Test same seed produces same data."""
        ds1 = PhysicsDataset(darcy_solver, n_samples=10, seed=42)
        ds2 = PhysicsDataset(darcy_solver, n_samples=10, seed=42)

        for i in range(10):
            assert torch.allclose(ds1[i]["input"], ds2[i]["input"])
            assert torch.allclose(ds1[i]["output"], ds2[i]["output"])

    def test_different_seeds_produce_different_data(self, darcy_solver: DarcyFlowSolver) -> None:
        """Test different seeds produce different data."""
        ds1 = PhysicsDataset(darcy_solver, n_samples=10, seed=42)
        ds2 = PhysicsDataset(darcy_solver, n_samples=10, seed=123)

        # At least first sample should differ
        assert not torch.allclose(ds1[0]["input"], ds2[0]["input"])

    def test_normalization_applied(self, darcy_solver: DarcyFlowSolver) -> None:
        """Test normalization when enabled."""
        ds_norm = PhysicsDataset(darcy_solver, n_samples=100, seed=42, normalize=True)
        ds_raw = PhysicsDataset(darcy_solver, n_samples=100, seed=42, normalize=False)

        # Normalized data should have different values
        assert not torch.allclose(ds_norm[0]["input"], ds_raw[0]["input"])

        # Check stats are computed
        stats = ds_norm.get_stats()
        assert "input_mean" in stats
        assert "input_std" in stats

    def test_normalization_roughly_centered(self, darcy_solver: DarcyFlowSolver) -> None:
        """Test normalized data is roughly zero-mean unit-variance."""
        dataset = PhysicsDataset(darcy_solver, n_samples=100, seed=42, normalize=True)

        all_inputs = torch.stack([dataset[i]["input"] for i in range(len(dataset))])

        # Should be approximately zero-mean
        assert abs(all_inputs.mean().item()) < 0.5
        # Should be approximately unit-variance (relaxed)
        assert 0.5 < all_inputs.std().item() < 2.0


class TestDatasetSplits:
    """Test train/val/test split functionality."""

    def test_splits_have_correct_sizes(self) -> None:
        """Test splits have requested sizes."""
        solver = DarcyFlowSolver(resolution=16)

        train_ds, val_ds, test_ds = PhysicsDataset.create_splits(
            solver,
            n_train=100,
            n_val=20,
            n_test=10,
            seed=42,
        )

        assert len(train_ds) == 100
        assert len(val_ds) == 20
        assert len(test_ds) == 10

    def test_splits_are_disjoint(self) -> None:
        """Test splits use disjoint seed ranges."""
        solver = DarcyFlowSolver(resolution=16)

        train_ds, val_ds, test_ds = PhysicsDataset.create_splits(
            solver,
            n_train=50,
            n_val=10,
            n_test=10,
            seed=42,
        )

        # Check first sample of each split is different
        train_sample = train_ds[0]["input"]
        val_sample = val_ds[0]["input"]
        test_sample = test_ds[0]["input"]

        assert not torch.allclose(train_sample, val_sample)
        assert not torch.allclose(train_sample, test_sample)
        assert not torch.allclose(val_sample, test_sample)

    def test_splits_reproducible(self) -> None:
        """Test splits are reproducible with same seed."""
        solver = DarcyFlowSolver(resolution=16)

        train1, val1, test1 = PhysicsDataset.create_splits(
            solver, n_train=20, n_val=5, n_test=5, seed=42
        )
        train2, val2, test2 = PhysicsDataset.create_splits(
            solver, n_train=20, n_val=5, n_test=5, seed=42
        )

        assert torch.allclose(train1[0]["input"], train2[0]["input"])
        assert torch.allclose(val1[0]["input"], val2[0]["input"])
        assert torch.allclose(test1[0]["input"], test2[0]["input"])


class TestMultipleSolvers:
    """Test PhysicsDataset works with different solvers."""

    @pytest.mark.parametrize(
        "SolverClass,kwargs",
        [
            (DarcyFlowSolver, {"resolution": 16}),
            (HeatSolver, {"resolution": 16}),
            (PoissonSolver, {"resolution": 16}),
        ],
    )
    def test_dataset_works_with_solver(self, SolverClass, kwargs) -> None:
        """Test dataset works with various solvers."""
        solver = SolverClass(**kwargs)
        dataset = PhysicsDataset(solver, n_samples=10, seed=42)

        sample = dataset[0]
        assert sample["input"].shape[0] == 16 * 16
        assert sample["output"].shape[0] == 16 * 16


class TestDataLoader:
    """Test integration with PyTorch DataLoader."""

    def test_dataloader_iteration(self) -> None:
        """Test dataset works with DataLoader."""
        solver = DarcyFlowSolver(resolution=16)
        dataset = PhysicsDataset(solver, n_samples=32, seed=42)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
        )

        batch = next(iter(loader))
        assert batch["input"].shape == (8, 16 * 16)
        assert batch["output"].shape == (8, 16 * 16)

    def test_full_epoch_iteration(self) -> None:
        """Test iterating through full epoch."""
        solver = DarcyFlowSolver(resolution=16)
        dataset = PhysicsDataset(solver, n_samples=64, seed=42)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
        )

        total_samples = 0
        for batch in loader:
            total_samples += batch["input"].shape[0]

        assert total_samples == 64
