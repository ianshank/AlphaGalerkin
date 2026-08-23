"""Tests for src.data.physics_dataset.PhysicsDataset.

Tests cover:
- Eager caching (``cache=True``, the default) vs on-demand generation
  (``cache=False``), including the invariant that a populated cache is
  never re-queried on repeated indexing.
- Normalization statistics: computation, the ``_compute_stats`` no-op
  guard when no cache exists, and the ``get_stats()`` error branch.
- Boundary sizes: an empty (``n_samples=0``) dataset, a single-sample
  dataset, and the ``_generate_cache`` progress-logging boundary at
  every 100th sample.
- dtype handling: ``input``/``output`` fields are always cast to
  float32 regardless of the solver's native dtype, while ``coords``
  passes through uncast.
- ``create_splits()``: requested sizes, disjoint seed derivation, and
  ``**kwargs`` forwarding to every split.
- Structured logging events emitted during initialization, cache
  generation, and split creation.

A dependency-free ``FakeConstantSolver`` stands in for the real
(scipy-backed) physics solvers (``DarcyFlowSolver`` etc.) so these tests
stay fast, deterministic, and exercise ``PhysicsDataset`` in isolation.
Solver correctness itself is covered elsewhere (``tests/physics/``,
``tests/test_data_generation.py``).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.data.physics_dataset import PhysicsDataset
from src.physics.solver import DiffEqSolver, PhysicsSample

# --- Constants (named, not inlined magic numbers) ---

RESOLUTION = 4  # 4x4 grid keeps the fake solver's fields tiny and fast
BASE_SEED = 1000
SMALL_N_SAMPLES = 5
SINGLE_SAMPLE = 1
EMPTY_N_SAMPLES = 0
# Mirrors the literal `(i + 1) % 100 == 0` check in
# PhysicsDataset._generate_cache -- must be reached to exercise that branch.
PROGRESS_LOG_BOUNDARY = 100
SPLIT_N_TRAIN = 4
SPLIT_N_VAL = 2
SPLIT_N_TEST = 2
SPLIT_BASE_SEED = 10
# Mirrors the epsilon PhysicsDataset._compute_stats / __getitem__ add to the
# denominator of the normalization formula -- referenced here (not
# reinvented) so the expected-value math matches the production contract.
NORMALIZE_EPSILON = 1e-8


class FakeConstantSolver(DiffEqSolver[NDArray[np.float32], NDArray[np.float32]]):
    """Deterministic, dependency-free ``DiffEqSolver`` double.

    ``generate_sample(seed=s)`` returns a field filled with the value
    ``s``, so tests can assert exact per-sample identity instead of fuzzy
    not-quite-equal checks. ``field_dtype``/``coords_dtype`` let a test
    independently control the dtype PhysicsDataset receives, to probe its
    casting behaviour.
    """

    def __init__(
        self,
        resolution: int = RESOLUTION,
        field_dtype: type = np.float32,
        coords_dtype: type | None = None,
    ) -> None:
        super().__init__(resolution=resolution)
        self._field_dtype = field_dtype
        # None => reuse the base-class helper, which always returns float32.
        self._coords_dtype = coords_dtype

    def solve(self, input_field: NDArray[np.float32]) -> NDArray[np.float32]:
        return input_field * 2.0

    def generate_sample(self, seed: int | None = None) -> PhysicsSample:
        value = float(seed if seed is not None else 0.0)
        n = self.resolution
        input_field: NDArray[np.float32] = np.full((n * n,), value, dtype=self._field_dtype)
        output_field = self.solve(input_field)
        if self._coords_dtype is None:
            coords = self._get_grid_coords(n)
        else:
            coords = np.zeros((n * n, 2), dtype=self._coords_dtype)
        return PhysicsSample(
            input_field=input_field,
            output_field=output_field,
            coords=coords,
            grid_size=n,
        )


@pytest.fixture
def solver() -> FakeConstantSolver:
    """Fast, deterministic solver double for dataset-level tests."""
    return FakeConstantSolver()


# --- Eager caching vs on-demand generation ---


class TestCachingModes:
    """``cache=True`` eagerly materializes; ``cache=False`` defers to the solver."""

    def test_cache_true_populates_internal_cache(self, solver: FakeConstantSolver) -> None:
        dataset = PhysicsDataset(solver, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED)
        assert dataset._cache is not None
        assert len(dataset._cache) == SMALL_N_SAMPLES

    def test_cache_false_leaves_internal_cache_none(self, solver: FakeConstantSolver) -> None:
        dataset = PhysicsDataset(solver, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED, cache=False)
        assert dataset._cache is None

    def test_cache_false_also_skips_stats_even_when_normalize_true(
        self, solver: FakeConstantSolver
    ) -> None:
        """``normalize=True`` only takes effect nested under ``cache=True``.

        ``__init__`` only calls ``_compute_stats()`` inside the
        ``if cache:`` block, so requesting normalization with caching
        disabled silently never computes statistics.
        """
        dataset = PhysicsDataset(
            solver, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED, cache=False, normalize=True
        )
        assert dataset._stats is None
        with pytest.raises(ValueError, match="Statistics not computed"):
            dataset.get_stats()

    def test_getitem_without_cache_queries_solver_on_demand(
        self, solver: FakeConstantSolver
    ) -> None:
        dataset = PhysicsDataset(
            solver, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED, cache=False, normalize=False
        )
        idx = 2
        sample = dataset[idx]
        assert sample["input"][0].item() == pytest.approx(BASE_SEED + idx)

    def test_getitem_with_cache_never_calls_solver_again(self, solver: FakeConstantSolver) -> None:
        """Repeated indexing must reuse the eager cache, not regenerate samples."""
        dataset = PhysicsDataset(solver, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED)
        with patch.object(solver, "generate_sample", wraps=solver.generate_sample) as spy:
            for i in range(SMALL_N_SAMPLES):
                _ = dataset[i]
            spy.assert_not_called()


# --- _compute_stats() guard ---


class TestComputeStatsGuard:
    """``_compute_stats()`` defensively no-ops when no cache exists yet."""

    def test_compute_stats_is_noop_without_cache(self, solver: FakeConstantSolver) -> None:
        dataset = PhysicsDataset(solver, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED, cache=False)
        assert dataset._stats is None
        # get_stats()'s own error message names this as the recovery path;
        # exercise it directly since __init__ can never reach it once
        # _generate_cache has run (it always leaves a list, never None).
        dataset._compute_stats()
        assert dataset._stats is None


# --- get_stats() ---


class TestGetStats:
    def test_get_stats_returns_expected_keys_when_normalized(
        self, solver: FakeConstantSolver
    ) -> None:
        dataset = PhysicsDataset(solver, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED, normalize=True)
        stats = dataset.get_stats()
        assert set(stats) == {"input_mean", "input_std", "output_mean", "output_std"}
        assert all(isinstance(v, float) for v in stats.values())

    def test_get_stats_raises_when_normalize_false(self, solver: FakeConstantSolver) -> None:
        dataset = PhysicsDataset(solver, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED, normalize=False)
        with pytest.raises(ValueError, match="Statistics not computed"):
            dataset.get_stats()

    def test_getitem_applies_normalization_when_enabled(self, solver: FakeConstantSolver) -> None:
        """Verify the normalization formula end-to-end.

        Compares against an independently derived reference mean/std
        (not against ``dataset.get_stats()``, to avoid a tautological
        check).
        """
        dataset = PhysicsDataset(solver, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED, normalize=True)
        # FakeConstantSolver.generate_sample(seed=s) fills every element
        # with `s`, so sample i's raw value is exactly BASE_SEED + i.
        per_sample_values = np.array(
            [BASE_SEED + i for i in range(SMALL_N_SAMPLES)], dtype=np.float64
        )
        expected_mean = float(np.mean(per_sample_values))
        expected_std = float(np.std(per_sample_values))
        expected0 = (float(BASE_SEED) - expected_mean) / (expected_std + NORMALIZE_EPSILON)

        sample0 = dataset[0]
        assert sample0["input"][0].item() == pytest.approx(expected0, rel=1e-5)


# --- Boundary sizes ---


class TestBoundarySizes:
    def test_empty_dataset_length_is_zero(self, solver: FakeConstantSolver) -> None:
        dataset = PhysicsDataset(solver, n_samples=EMPTY_N_SAMPLES, seed=BASE_SEED, normalize=False)
        assert len(dataset) == EMPTY_N_SAMPLES
        assert dataset._cache == []

    def test_empty_dataset_with_default_normalize_raises(self, solver: FakeConstantSolver) -> None:
        """Document current behaviour for an empty dataset.

        With the default ``normalize=True``, an empty dataset cannot
        compute statistics because ``np.stack([])`` requires at least one
        array. Callers must pass ``normalize=False`` explicitly for
        ``n_samples=0``.
        """
        with pytest.raises(ValueError, match="need at least one array to stack"):
            PhysicsDataset(solver, n_samples=EMPTY_N_SAMPLES, seed=BASE_SEED)

    def test_single_sample_dataset(self, solver: FakeConstantSolver) -> None:
        dataset = PhysicsDataset(solver, n_samples=SINGLE_SAMPLE, seed=BASE_SEED)
        assert len(dataset) == SINGLE_SAMPLE
        assert dataset[0]["input"].shape[0] == RESOLUTION * RESOLUTION

    def test_progress_log_fires_at_100_sample_boundary(self, solver: FakeConstantSolver) -> None:
        with patch("src.data.physics_dataset.logger") as mock_logger:
            PhysicsDataset(solver, n_samples=PROGRESS_LOG_BOUNDARY, seed=BASE_SEED, normalize=False)
        mock_logger.debug.assert_any_call(
            "generation_progress",
            completed=PROGRESS_LOG_BOUNDARY,
            total=PROGRESS_LOG_BOUNDARY,
        )

    def test_no_progress_log_below_100_samples(self, solver: FakeConstantSolver) -> None:
        with patch("src.data.physics_dataset.logger") as mock_logger:
            PhysicsDataset(solver, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED, normalize=False)
        for call in mock_logger.debug.call_args_list:
            assert call.args[0] != "generation_progress"


# --- dtype handling ---


class TestDtypeCasting:
    def test_input_and_output_are_always_cast_to_float32(self) -> None:
        solver64 = FakeConstantSolver(field_dtype=np.float64)
        dataset = PhysicsDataset(
            solver64, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED, normalize=False
        )
        sample = dataset[0]
        assert sample["input"].dtype == torch.float32
        assert sample["output"].dtype == torch.float32

    def test_coords_dtype_passes_through_uncast(self) -> None:
        """Confirm coords bypass the float32 cast applied to input/output.

        Unlike ``input``/``output``, ``__getitem__`` never casts
        ``coords`` -- a solver returning float64 coordinates leaks a
        float64 tensor straight through to the caller.
        """
        solver64_coords = FakeConstantSolver(coords_dtype=np.float64)
        dataset = PhysicsDataset(
            solver64_coords, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED, normalize=False
        )
        sample = dataset[0]
        assert sample["coords"].dtype == torch.float64


# --- create_splits() ---


class TestCreateSplits:
    def test_split_sizes_match_request(self, solver: FakeConstantSolver) -> None:
        train_ds, val_ds, test_ds = PhysicsDataset.create_splits(
            solver,
            n_train=SPLIT_N_TRAIN,
            n_val=SPLIT_N_VAL,
            n_test=SPLIT_N_TEST,
            seed=SPLIT_BASE_SEED,
            normalize=False,
        )
        assert len(train_ds) == SPLIT_N_TRAIN
        assert len(val_ds) == SPLIT_N_VAL
        assert len(test_ds) == SPLIT_N_TEST

    def test_split_seeds_are_disjoint_ranges(self, solver: FakeConstantSolver) -> None:
        train_ds, val_ds, test_ds = PhysicsDataset.create_splits(
            solver,
            n_train=SPLIT_N_TRAIN,
            n_val=SPLIT_N_VAL,
            n_test=SPLIT_N_TEST,
            seed=SPLIT_BASE_SEED,
            normalize=False,
        )
        assert train_ds.seed == SPLIT_BASE_SEED
        assert val_ds.seed == SPLIT_BASE_SEED + SPLIT_N_TRAIN
        assert test_ds.seed == SPLIT_BASE_SEED + SPLIT_N_TRAIN + SPLIT_N_VAL
        # The deterministic fake solver (value == seed) lets us assert the
        # *data* is disjoint too, not just the recorded .seed attribute.
        assert train_ds[0]["input"][0].item() == pytest.approx(SPLIT_BASE_SEED)
        assert val_ds[0]["input"][0].item() == pytest.approx(SPLIT_BASE_SEED + SPLIT_N_TRAIN)
        assert test_ds[0]["input"][0].item() == pytest.approx(
            SPLIT_BASE_SEED + SPLIT_N_TRAIN + SPLIT_N_VAL
        )

    def test_kwargs_forwarded_to_each_split(self, solver: FakeConstantSolver) -> None:
        train_ds, val_ds, test_ds = PhysicsDataset.create_splits(
            solver,
            n_train=SPLIT_N_TRAIN,
            n_val=SPLIT_N_VAL,
            n_test=SPLIT_N_TEST,
            seed=SPLIT_BASE_SEED,
            normalize=False,
        )
        assert train_ds.normalize is False
        assert val_ds.normalize is False
        assert test_ds.normalize is False

    def test_create_splits_logs_summary_event(self, solver: FakeConstantSolver) -> None:
        with patch("src.data.physics_dataset.logger") as mock_logger:
            PhysicsDataset.create_splits(
                solver,
                n_train=SPLIT_N_TRAIN,
                n_val=SPLIT_N_VAL,
                n_test=SPLIT_N_TEST,
                seed=SPLIT_BASE_SEED,
                normalize=False,
            )
        mock_logger.info.assert_any_call(
            "dataset_splits_created",
            train=SPLIT_N_TRAIN,
            val=SPLIT_N_VAL,
            test=SPLIT_N_TEST,
        )


# --- Structured logging (initialization) ---


class TestInitializationLogging:
    def test_init_logs_physics_dataset_initialized_event(self, solver: FakeConstantSolver) -> None:
        with patch("src.data.physics_dataset.logger") as mock_logger:
            PhysicsDataset(solver, n_samples=SMALL_N_SAMPLES, seed=BASE_SEED, normalize=False)
        mock_logger.info.assert_any_call(
            "physics_dataset_initialized",
            solver="FakeConstantSolver",
            n_samples=SMALL_N_SAMPLES,
            resolution=RESOLUTION,
            seed=BASE_SEED,
            normalize=False,
        )
