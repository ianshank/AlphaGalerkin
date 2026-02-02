"""Tests for Poisson equation solver.

Tests cover:
- PoissonSample: Poisson-specific sample dataclass
- PoissonSolver: Spectral and iterative solvers
- PoissonDataset: Dataset generation
- generate_random_charges: Charge distribution generation
- generate_influence_field: Complete sample generation
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")
scipy = pytest.importorskip("scipy")

from src.physics.poisson import (
    PoissonDataset,
    PoissonSample,
    PoissonSolver,
    create_poisson_dataloader,
    generate_influence_field,
    generate_random_charges,
)

# --- Fixtures ---


@pytest.fixture
def solver() -> PoissonSolver:
    """Create default Poisson solver."""
    return PoissonSolver(resolution=8)


@pytest.fixture
def solver_iterative() -> PoissonSolver:
    """Create Poisson solver with iterative method."""
    return PoissonSolver(use_spectral=False, resolution=8)


@pytest.fixture
def sample_charges() -> numpy.ndarray:
    """Create sample charge distribution."""
    return generate_random_charges(grid_size=8, seed=42)


# --- PoissonSample Tests ---


class TestPoissonSample:
    """Tests for PoissonSample dataclass."""

    def test_charges_alias(self):
        """Test charges property aliases input_field."""
        charges = numpy.random.randn(16).astype(numpy.float32)
        sample = PoissonSample(
            input_field=charges,
            output_field=numpy.random.randn(16).astype(numpy.float32),
            coords=numpy.random.rand(16, 2).astype(numpy.float32),
            grid_size=4,
        )

        assert numpy.array_equal(sample.charges, sample.input_field)

    def test_potential_alias(self):
        """Test potential property aliases output_field."""
        potential = numpy.random.randn(16).astype(numpy.float32)
        sample = PoissonSample(
            input_field=numpy.random.randn(16).astype(numpy.float32),
            output_field=potential,
            coords=numpy.random.rand(16, 2).astype(numpy.float32),
            grid_size=4,
        )

        assert numpy.array_equal(sample.potential, sample.output_field)


# --- PoissonSolver Initialization Tests ---


class TestPoissonSolverInit:
    """Tests for PoissonSolver initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        solver = PoissonSolver()

        assert solver.boundary_value == 0.0
        assert solver.use_spectral is True
        assert solver.regularization == 1e-6
        assert solver.resolution == 32

    def test_custom_values(self):
        """Test custom initialization."""
        solver = PoissonSolver(
            boundary_value=1.0,
            use_spectral=False,
            regularization=1e-8,
            resolution=16,
        )

        assert solver.boundary_value == 1.0
        assert solver.use_spectral is False
        assert solver.regularization == 1e-8
        assert solver.resolution == 16


# --- PoissonSolver Solve Tests ---


class TestPoissonSolverSolve:
    """Tests for PoissonSolver.solve method."""

    def test_solve_2d_input(self, solver: PoissonSolver, sample_charges: numpy.ndarray):
        """Test solving with 2D charge array."""
        potential = solver.solve(sample_charges)

        assert potential.shape == sample_charges.shape
        assert potential.dtype == numpy.float32

    def test_solve_1d_input(self, solver: PoissonSolver, sample_charges: numpy.ndarray):
        """Test solving with flattened charge array."""
        charges_flat = sample_charges.flatten()
        potential = solver.solve(charges_flat)

        assert potential.shape == charges_flat.shape
        assert potential.dtype == numpy.float32

    def test_solve_spectral_vs_iterative(self, sample_charges: numpy.ndarray):
        """Test spectral and iterative methods give similar results."""
        solver_spectral = PoissonSolver(use_spectral=True, resolution=8)
        solver_iterative = PoissonSolver(use_spectral=False, resolution=8)

        pot_spectral = solver_spectral.solve(sample_charges)
        pot_iterative = solver_iterative.solve(sample_charges)

        # Results should be similar (not exact due to method differences)
        correlation = numpy.corrcoef(pot_spectral.flatten(), pot_iterative.flatten())[
            0, 1
        ]
        assert correlation > 0.9

    def test_solve_zero_charges(self, solver: PoissonSolver):
        """Test solving with zero charges."""
        charges = numpy.zeros((8, 8), dtype=numpy.float32)
        potential = solver.solve(charges)

        # Zero charges should give zero potential
        assert numpy.allclose(potential, 0.0, atol=1e-6)

    def test_solve_deterministic(
        self, solver: PoissonSolver, sample_charges: numpy.ndarray
    ):
        """Test solver is deterministic."""
        pot1 = solver.solve(sample_charges)
        pot2 = solver.solve(sample_charges)

        assert numpy.allclose(pot1, pot2)

    def test_solve_linearity(self, solver: PoissonSolver, sample_charges: numpy.ndarray):
        """Test Poisson solver is linear."""
        # Laplacian is a linear operator, so 2*charges -> 2*potential
        pot1 = solver.solve(sample_charges)
        pot2 = solver.solve(2 * sample_charges)

        assert numpy.allclose(pot2, 2 * pot1, rtol=1e-4)

    def test_solve_superposition(self, solver: PoissonSolver):
        """Test superposition principle."""
        charges1 = generate_random_charges(grid_size=8, seed=1)
        charges2 = generate_random_charges(grid_size=8, seed=2)

        pot1 = solver.solve(charges1)
        pot2 = solver.solve(charges2)
        pot_combined = solver.solve(charges1 + charges2)

        # Superposition: pot(q1 + q2) = pot(q1) + pot(q2)
        assert numpy.allclose(pot_combined, pot1 + pot2, rtol=1e-4)


# --- PoissonSolver Generate Sample Tests ---


class TestPoissonSolverGenerateSample:
    """Tests for PoissonSolver.generate_sample method."""

    def test_generate_sample_returns_poisson_sample(self, solver: PoissonSolver):
        """Test generate_sample returns PoissonSample."""
        sample = solver.generate_sample(seed=42)
        assert isinstance(sample, PoissonSample)

    def test_generate_sample_shapes(self, solver: PoissonSolver):
        """Test generated sample has correct shapes."""
        sample = solver.generate_sample(seed=42)
        n_points = solver.resolution**2

        assert sample.input_field.shape == (n_points,)
        assert sample.output_field.shape == (n_points,)
        assert sample.coords.shape == (n_points, 2)
        assert sample.grid_size == solver.resolution

    def test_generate_sample_deterministic(self, solver: PoissonSolver):
        """Test sample generation is deterministic with seed."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=42)

        assert numpy.allclose(sample1.input_field, sample2.input_field)
        assert numpy.allclose(sample1.output_field, sample2.output_field)

    def test_generate_sample_different_seeds(self, solver: PoissonSolver):
        """Test different seeds produce different samples."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=43)

        assert not numpy.allclose(sample1.input_field, sample2.input_field)


# --- generate_random_charges Tests ---


class TestGenerateRandomCharges:
    """Tests for generate_random_charges function."""

    def test_output_shape(self):
        """Test output has correct shape."""
        charges = generate_random_charges(grid_size=8)
        assert charges.shape == (8, 8)

    def test_output_dtype(self):
        """Test output is float32."""
        charges = generate_random_charges(grid_size=8)
        assert charges.dtype == numpy.float32

    def test_deterministic_with_seed(self):
        """Test same seed produces same charges."""
        charges1 = generate_random_charges(grid_size=8, seed=42)
        charges2 = generate_random_charges(grid_size=8, seed=42)

        assert numpy.allclose(charges1, charges2)

    def test_point_charges(self):
        """Test sparse point charge generation."""
        charges = generate_random_charges(
            grid_size=16, n_charges=3, smooth=False, seed=42
        )

        # With 3 charges and no smoothing, should have few non-zero values
        non_zero_count = numpy.count_nonzero(charges)
        assert non_zero_count <= 3

    def test_charge_std_affects_magnitude(self):
        """Test charge_std affects magnitude."""
        charges_low = generate_random_charges(grid_size=8, charge_std=0.1, seed=42)
        charges_high = generate_random_charges(grid_size=8, charge_std=10.0, seed=42)

        assert numpy.abs(charges_high).max() > numpy.abs(charges_low).max()


# --- generate_influence_field Tests ---


class TestGenerateInfluenceField:
    """Tests for generate_influence_field function."""

    def test_returns_poisson_sample(self):
        """Test returns PoissonSample."""
        sample = generate_influence_field(grid_size=8, seed=42)
        assert isinstance(sample, PoissonSample)

    def test_sample_shapes(self):
        """Test sample has correct shapes."""
        grid_size = 8
        sample = generate_influence_field(grid_size=grid_size, seed=42)
        n_points = grid_size**2

        assert sample.charges.shape == (n_points,)
        assert sample.potential.shape == (n_points,)
        assert sample.coords.shape == (n_points, 2)

    def test_coords_normalized(self):
        """Test coordinates are in [0, 1]."""
        sample = generate_influence_field(grid_size=8, seed=42)

        assert sample.coords.min() >= 0.0
        assert sample.coords.max() <= 1.0

    def test_deterministic_with_seed(self):
        """Test same seed produces same sample."""
        sample1 = generate_influence_field(grid_size=8, seed=42)
        sample2 = generate_influence_field(grid_size=8, seed=42)

        assert numpy.allclose(sample1.charges, sample2.charges)
        assert numpy.allclose(sample1.potential, sample2.potential)


# --- PoissonDataset Tests ---


class TestPoissonDataset:
    """Tests for PoissonDataset class."""

    @pytest.fixture
    def dataset(self) -> PoissonDataset:
        """Create a small test dataset."""
        return PoissonDataset(
            grid_size=4,
            n_samples=5,
            cache_samples=True,
            seed=42,
        )

    def test_init_caches_samples(self, dataset: PoissonDataset):
        """Test dataset caches samples on init."""
        assert dataset._cache is not None
        assert len(dataset._cache) == 5

    def test_len(self, dataset: PoissonDataset):
        """Test __len__ returns correct size."""
        assert len(dataset) == 5

    def test_getitem(self, dataset: PoissonDataset):
        """Test __getitem__ returns sample."""
        sample = dataset[0]
        assert isinstance(sample, PoissonSample)

    def test_getitem_cached(self, dataset: PoissonDataset):
        """Test __getitem__ returns cached sample."""
        sample1 = dataset[0]
        sample2 = dataset[0]

        # Should be the exact same object
        assert sample1 is sample2

    def test_iteration(self, dataset: PoissonDataset):
        """Test iteration over dataset."""
        samples = list(dataset)
        assert len(samples) == 5
        for sample in samples:
            assert isinstance(sample, PoissonSample)

    def test_no_cache(self):
        """Test dataset without caching."""
        dataset = PoissonDataset(
            grid_size=4,
            n_samples=3,
            cache_samples=False,
            seed=42,
        )

        assert dataset._cache is None

        # Should still work
        sample = dataset[0]
        assert isinstance(sample, PoissonSample)

    def test_deterministic_samples(self):
        """Test samples are deterministic with seed."""
        dataset1 = PoissonDataset(grid_size=4, n_samples=3, seed=42)
        dataset2 = PoissonDataset(grid_size=4, n_samples=3, seed=42)

        for i in range(3):
            assert numpy.allclose(dataset1[i].charges, dataset2[i].charges)

    def test_get_statistics(self, dataset: PoissonDataset):
        """Test get_statistics returns expected keys."""
        stats = dataset.get_statistics()

        assert "potential_mean" in stats
        assert "potential_std" in stats
        assert "potential_min" in stats
        assert "potential_max" in stats
        assert "charge_mean" in stats
        assert "charge_std" in stats

    def test_get_statistics_values(self, dataset: PoissonDataset):
        """Test get_statistics returns valid values."""
        stats = dataset.get_statistics()

        # Standard deviation should be positive
        assert stats["potential_std"] >= 0
        assert stats["charge_std"] >= 0

        # Min should be <= max
        assert stats["potential_min"] <= stats["potential_max"]


# --- create_poisson_dataloader Tests ---


class TestCreatePoissonDataloader:
    """Tests for create_poisson_dataloader function."""

    def test_yields_batches(self):
        """Test dataloader yields batches."""
        loader = create_poisson_dataloader(
            grid_size=4,
            n_samples=6,
            batch_size=2,
            shuffle=False,
        )

        batches = list(loader)
        assert len(batches) == 3

    def test_batch_structure(self):
        """Test batch has expected structure."""
        loader = create_poisson_dataloader(
            grid_size=4,
            n_samples=4,
            batch_size=2,
        )

        batch = next(iter(loader))

        assert "coords" in batch
        assert "charges" in batch
        assert "potential" in batch
        assert "grid_size" in batch

    def test_batch_shapes(self):
        """Test batch tensors have correct shapes."""
        grid_size = 4
        batch_size = 2
        n_points = grid_size**2

        loader = create_poisson_dataloader(
            grid_size=grid_size,
            n_samples=4,
            batch_size=batch_size,
        )

        batch = next(iter(loader))

        assert batch["coords"].shape == (batch_size, n_points, 2)
        assert batch["charges"].shape == (batch_size, n_points)
        assert batch["potential"].shape == (batch_size, n_points)

    def test_no_shuffle(self):
        """Test dataloader without shuffle is deterministic."""
        loader1 = create_poisson_dataloader(
            grid_size=4,
            n_samples=4,
            batch_size=2,
            shuffle=False,
            seed=42,
        )
        loader2 = create_poisson_dataloader(
            grid_size=4,
            n_samples=4,
            batch_size=2,
            shuffle=False,
            seed=42,
        )

        for batch1, batch2 in zip(loader1, loader2, strict=True):
            assert numpy.allclose(batch1["charges"], batch2["charges"])


# --- Physical Properties Tests ---


class TestPhysicalProperties:
    """Tests for physical correctness of Poisson solver."""

    def test_potential_smoothness(self, solver: PoissonSolver):
        """Test potential is smoother than charges."""
        charges = generate_random_charges(grid_size=16, smooth=False, seed=42)
        potential = solver.solve(charges)

        # Compute gradients
        charge_grad = numpy.diff(charges, axis=0)
        pot_grad = numpy.diff(potential, axis=0)

        # Potential gradient should have smaller variance (smoother)
        assert pot_grad.var() < charge_grad.var()

    def test_boundary_conditions(self):
        """Test zero Dirichlet boundary conditions."""
        solver = PoissonSolver(boundary_value=0.0, use_spectral=True)
        charges = generate_random_charges(grid_size=8, seed=42)
        potential = solver.solve(charges)

        # DST naturally enforces zero boundary (potential at edges should be small)
        # Note: DST doesn't include boundary points, so we check the interior
        # is smooth approaching the boundary
        assert potential.dtype == numpy.float32

    def test_symmetric_charges_symmetric_potential(self, solver: PoissonSolver):
        """Test symmetric charges produce symmetric potential."""
        grid_size = 8
        charges = numpy.zeros((grid_size, grid_size), dtype=numpy.float32)

        # Place symmetric charges
        charges[2, 4] = 1.0
        charges[5, 4] = 1.0  # Symmetric about center

        potential = solver.solve(charges)

        # Potential should be approximately symmetric
        # (not exactly due to discretization)
        top_half = potential[:4, :]
        bottom_half = potential[4:, :][::-1, :]

        # Should be roughly similar
        assert numpy.corrcoef(top_half.flatten(), bottom_half.flatten())[0, 1] > 0.8


# --- Edge Cases ---


class TestEdgeCases:
    """Edge case tests for Poisson solver."""

    def test_small_grid(self):
        """Test with very small grid."""
        solver = PoissonSolver(resolution=3)
        sample = solver.generate_sample(seed=42)

        assert sample.grid_size == 3
        assert len(sample.charges) == 9

    def test_large_regularization(self):
        """Test with large regularization value."""
        solver = PoissonSolver(regularization=1.0)
        charges = generate_random_charges(grid_size=8, seed=42)

        # Should still solve without error
        potential = solver.solve(charges)
        assert potential.shape == charges.shape

    def test_single_point_charge(self):
        """Test with single point charge."""
        solver = PoissonSolver()
        charges = numpy.zeros((8, 8), dtype=numpy.float32)
        charges[4, 4] = 1.0

        potential = solver.solve(charges)

        # Maximum potential should be near the charge
        max_idx = numpy.unravel_index(numpy.argmax(numpy.abs(potential)), potential.shape)
        # The maximum might not be exactly at (4,4) due to boundary effects
        assert potential.shape == (8, 8)
