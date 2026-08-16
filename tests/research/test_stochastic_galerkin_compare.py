"""Tests for the two-arm Fokker-Planck/OU comparison harness (AC8).

Micro budgets throughout (grid 12–16, 1–3 epochs) — the harness contract is
mechanism, not headline numbers: shared-eval-set fairness, metric shape,
artifact emission, and the stochastic arm's near-exactness on this benchmark.
"""

from __future__ import annotations

import csv

import pytest
import torch

from src.pde.stochastic.analytic import ou_covariance, ou_mean
from src.research.stochastic_galerkin_compare import (
    MultiSeedStochasticComparison,
    StochasticCompareParams,
    analytic_final_moments,
    build_shared_eval_set,
    density_fields,
    export_csv,
    export_plot,
    grid_coords,
    normalized_grid_coords,
    run_multiseed_comparison,
    run_stochastic_arm,
    run_stochastic_galerkin_comparison,
    sample_initial_conditions,
)

F64 = torch.float64


def _micro_params(**overrides) -> StochasticCompareParams:
    defaults = {
        "grid_n": 12,
        "n_train_samples": 6,
        "n_eval_samples": 3,
        "n_epochs": 1,
        "d_model": 16,
        "n_fourier_features": 8,
        "batch_size": 4,
    }
    defaults.update(overrides)
    return StochasticCompareParams(**defaults)


class TestParamsValidation:
    def test_defaults_valid(self):
        StochasticCompareParams()

    def test_grid_too_small_rejected(self):
        with pytest.raises(ValueError, match="grid_n"):
            StochasticCompareParams(grid_n=2)

    def test_p0_range_rejected(self):
        with pytest.raises(ValueError, match="p0_min"):
            StochasticCompareParams(p0_min=0.4, p0_max=0.2)
        with pytest.raises(ValueError, match="p0_min"):
            StochasticCompareParams(p0_min=0.0)

    def test_dt_exceeding_horizon_rejected(self):
        with pytest.raises(ValueError, match="strang_dt"):
            StochasticCompareParams(strang_dt=2.0, t_end=1.0)

    def test_non_2d_rejected(self):
        with pytest.raises(ValueError, match="2D"):
            StochasticCompareParams(drift_bias=(0.1,))

    def test_bad_matrix_shape_rejected(self):
        with pytest.raises(ValueError, match="drift_matrix"):
            StochasticCompareParams(drift_matrix=((-1.0,),))


class TestSharedBenchmarkData:
    def test_grid_shapes_and_normalization(self):
        params = _micro_params()
        coords = grid_coords(params)
        assert coords.shape == (144, 2)
        normalized = normalized_grid_coords(params)
        assert float(normalized.min()) >= 0.0
        assert float(normalized.max()) <= 1.0

    def test_initial_conditions_seeded(self):
        params = _micro_params()
        m1, c1 = sample_initial_conditions(params, 5, 7)
        m2, c2 = sample_initial_conditions(params, 5, 7)
        torch.testing.assert_close(m1, m2)
        torch.testing.assert_close(c1, c2)
        m3, _ = sample_initial_conditions(params, 5, 8)
        assert not torch.allclose(m1, m3)

    def test_density_fields_normalize_on_grid(self):
        params = _micro_params(grid_n=48)
        means, covs = sample_initial_conditions(params, 3, 0)
        fields = density_fields(params, means, covs)
        dx = 2.0 * params.domain_half_width / (params.grid_n - 1)
        integrals = fields.sum(dim=1) * dx * dx
        assert bool(((integrals - 1.0).abs() < 0.05).all())

    def test_analytic_final_moments_match_direct_call(self):
        params = _micro_params()
        means, covs = sample_initial_conditions(params, 2, 1)
        final_means, final_covs = analytic_final_moments(params, means, covs)
        a = torch.tensor([list(r) for r in params.drift_matrix], dtype=F64)
        b = torch.tensor(list(params.drift_bias), dtype=F64)
        g = torch.tensor([list(r) for r in params.diffusion], dtype=F64)
        torch.testing.assert_close(final_means[0], ou_mean(a, b, means[0], params.t_end))
        torch.testing.assert_close(final_covs[0], ou_covariance(a, g @ g.T, covs[0], params.t_end))

    def test_shared_eval_set_ignores_training_seed(self):
        """AC8 fairness invariant: the eval set depends on eval_seed_base only."""
        a = build_shared_eval_set(_micro_params(seed=1))
        b = build_shared_eval_set(_micro_params(seed=999))
        for left, right in zip(a, b, strict=True):
            torch.testing.assert_close(left, right)

    def test_different_eval_seed_base_changes_eval_set(self):
        a = build_shared_eval_set(_micro_params())
        b = build_shared_eval_set(_micro_params(eval_seed_base=1234))
        assert not torch.allclose(a[0], b[0])


class TestArms:
    def test_stochastic_arm_near_exact(self):
        params = _micro_params()
        eval_means, eval_covs, _inputs, targets = build_shared_eval_set(params)
        result = run_stochastic_arm(params, eval_means, eval_covs, targets)
        assert result.density_mse < 1e-6
        assert result.n_params == 0
        assert result.wall_clock_s > 0.0

    def test_comparison_metrics_shape(self):
        result = run_stochastic_galerkin_comparison(_micro_params())
        metrics = result.metrics
        for key in (
            "stochastic_density_mse",
            "deterministic_density_mse",
            "stochastic_vs_deterministic_mse_ratio",
            "stochastic_wall_clock_s",
            "deterministic_wall_clock_s",
            "deterministic_n_params",
        ):
            assert key in metrics
            assert torch.isfinite(torch.tensor(metrics[key]))
        assert metrics["deterministic_n_params"] > 0
        assert result.mse_ratio == pytest.approx(
            metrics["stochastic_density_mse"] / metrics["deterministic_density_mse"]
        )


class TestMultiSeed:
    @pytest.fixture(scope="class")
    def comparison(self) -> MultiSeedStochasticComparison:
        return run_multiseed_comparison(_micro_params(), seeds=[11, 12])

    def test_per_seed_results(self, comparison):
        assert len(comparison.results) == 2
        assert comparison.representative in comparison.results

    def test_stochastic_arm_seed_independent(self, comparison):
        mses = {r.stochastic.density_mse for r in comparison.results}
        assert len(mses) == 1

    def test_median_metrics(self, comparison):
        metrics = comparison.metrics
        assert metrics["n_seeds"] == 2.0
        assert "deterministic_density_mse_median" in metrics

    def test_default_seed_fallback(self):
        single = run_multiseed_comparison(_micro_params())
        assert len(single.results) == 1
        assert single.results[0].train_seed == _micro_params().seed

    def test_export_csv(self, comparison, tmp_path):
        path = export_csv(comparison, tmp_path / "compare.csv")
        assert path.exists()
        with path.open() as handle:
            rows = list(csv.reader(handle))
        assert rows[0][0] == "row"
        assert len(rows) == 1 + 2 + 1  # header + 2 seeds + median
        assert rows[-1][0] == "median"

    def test_export_plot(self, comparison, tmp_path):
        path = export_plot(comparison, tmp_path / "compare.png")
        assert path is not None
        assert path.exists()
        assert path.stat().st_size > 0
