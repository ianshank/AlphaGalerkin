"""Comparison harness: deterministic Galerkin-attention vs stochastic Galerkin projection.

Shared benchmark (AC8): the 2D Fokker-Planck equation for an OU process — the
density is exactly ``N(m(t), P(t))``, so ground truth is free and identical
for both arms. Task: initial Gaussian density (varying ``m0, P0`` per sample)
→ density field ``p(·, T)`` on a shared grid.

- **Deterministic arm** (existing path, untouched): a small ``PhysicsOperator``
  trained supervised on (initial-density field → analytic target field) pairs.
- **Stochastic arm** (new path): ``GalerkinMomentProjection`` +
  ``StrangSplitStep`` (K=1, no jump) propagates the moments and renders the
  density — no training, no data.

Fairness invariant: the eval initial conditions are a function of
``eval_seed_base`` only, so every arm and every training seed scores on the
identical held-out set. The two paths share only benchmark data structures —
no code merging (change-doc constraint).

Honesty rule (spec Thresholds): only the stochastic arm's absolute density
MSE is gated by the scenario; the deterministic arm's MSE and the ratio are
recorded ungated — on this benchmark the stochastic path is near-exact by
construction, so gating a ratio would be a self-serving benchmark.

Spec: specs/stochastic_galerkin_nke.spec.md (AC8, change-doc task 1.6).
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
import torch
from torch import Tensor, nn
from torch.optim import AdamW

from src.experiments.physics_model import PhysicsOperator
from src.pde.stochastic.analytic import (
    gaussian_density_on_grid,
    ou_covariance,
    ou_mean,
)
from src.pde.stochastic.config import StochasticGeneratorConfig, StrangSplittingConfig
from src.pde.stochastic.gaussian_mixture import GaussianMixtureState
from src.pde.stochastic.generator import KolmogorovGenerator
from src.pde.stochastic.projection import GalerkinMomentProjection
from src.pde.stochastic.strang import StrangSplitStep

logger = structlog.get_logger(__name__)

F64 = torch.float64

DEFAULT_EVAL_SEED_BASE = 9973
"""Eval-set seed base (prime; mirrors EVAL_SEED_STRIDE conventions)."""


@dataclass(frozen=True)
class StochasticCompareParams:
    """Configuration of the two-arm Fokker-Planck/OU density benchmark."""

    # Shared benchmark
    grid_n: int = 32
    domain_half_width: float = 2.0
    drift_matrix: tuple[tuple[float, ...], ...] = ((-1.0, 0.3), (0.0, -0.8))
    drift_bias: tuple[float, ...] = (0.1, -0.2)
    diffusion: tuple[tuple[float, ...], ...] = ((0.4, 0.0), (0.0, 0.3))
    t_end: float = 1.0
    strang_dt: float = 0.1
    n_train_samples: int = 64
    n_eval_samples: int = 16
    m0_half_range: float = 0.5
    p0_min: float = 0.1
    p0_max: float = 0.3
    # Deterministic arm (PhysicsOperator) budget
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 2
    n_fourier_features: int = 16
    fourier_scale: float = 5.0
    use_fnet: bool = False
    dropout: float = 0.0
    n_epochs: int = 40
    learning_rate: float = 1e-3
    batch_size: int = 8
    # Seeds
    seed: int = 42
    eval_seed_base: int = DEFAULT_EVAL_SEED_BASE
    # Device ('cpu', 'cuda', 'cuda:N') — both arms honor it; resolve 'auto' via
    # src.poc.device.resolve_device upstream (the scenario does).
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.grid_n < 4:
            msg = f"grid_n must be >= 4; got {self.grid_n}"
            raise ValueError(msg)
        if not (0.0 < self.p0_min <= self.p0_max):
            msg = f"require 0 < p0_min <= p0_max; got {self.p0_min}, {self.p0_max}"
            raise ValueError(msg)
        if self.strang_dt > self.t_end:
            msg = f"strang_dt ({self.strang_dt}) must not exceed t_end ({self.t_end})"
            raise ValueError(msg)
        dim = len(self.drift_bias)
        if dim != 2:
            msg = "the shared benchmark is 2D (drift_bias must have length 2)"
            raise ValueError(msg)
        if len(self.drift_matrix) != dim or any(len(r) != dim for r in self.drift_matrix):
            msg = "drift_matrix must be 2x2"
            raise ValueError(msg)
        if len(self.diffusion) != dim:
            msg = "diffusion must have 2 rows"
            raise ValueError(msg)
        if self.n_eval_samples < 1 or self.n_train_samples < 1:
            msg = "sample counts must be positive"
            raise ValueError(msg)


@dataclass
class StochasticArmResult:
    """One arm's evaluation on the shared held-out set."""

    name: str
    density_mse: float
    wall_clock_s: float
    n_params: int


@dataclass
class StochasticCompareResult:
    """Two-arm outcome on the shared benchmark."""

    params: StochasticCompareParams
    train_seed: int
    deterministic: StochasticArmResult
    stochastic: StochasticArmResult

    @property
    def mse_ratio(self) -> float:
        """Stochastic / deterministic density MSE (recorded, never gated)."""
        if self.deterministic.density_mse <= 0.0:
            return float("inf")
        return self.stochastic.density_mse / self.deterministic.density_mse

    @property
    def metrics(self) -> dict[str, float]:
        """Flat metric dict (scenario/baseline-harness consumable)."""
        return {
            "stochastic_density_mse": self.stochastic.density_mse,
            "deterministic_density_mse": self.deterministic.density_mse,
            "stochastic_vs_deterministic_mse_ratio": self.mse_ratio,
            "stochastic_wall_clock_s": self.stochastic.wall_clock_s,
            "deterministic_wall_clock_s": self.deterministic.wall_clock_s,
            "deterministic_n_params": float(self.deterministic.n_params),
        }


# --------------------------------------------------------------------------- #
# Shared benchmark data                                                        #
# --------------------------------------------------------------------------- #


def _matrix(rows: tuple[tuple[float, ...], ...]) -> Tensor:
    return torch.tensor([list(r) for r in rows], dtype=F64)


def grid_coords(params: StochasticCompareParams) -> Tensor:
    """Physical grid coordinates (N, 2) over the square domain, float64."""
    axis = torch.linspace(
        -params.domain_half_width, params.domain_half_width, params.grid_n, dtype=F64
    )
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def normalized_grid_coords(params: StochasticCompareParams) -> Tensor:
    """Grid coordinates normalized to [0, 1] (the PhysicsOperator convention)."""
    coords = grid_coords(params)
    return (coords + params.domain_half_width) / (2.0 * params.domain_half_width)


def sample_initial_conditions(
    params: StochasticCompareParams, n: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Seeded initial conditions: means (n, 2) and diagonal covariances (n, 2, 2)."""
    gen = torch.Generator().manual_seed(seed)
    means = (torch.rand(n, 2, dtype=F64, generator=gen) * 2 - 1) * params.m0_half_range
    variances = (
        torch.rand(n, 2, dtype=F64, generator=gen) * (params.p0_max - params.p0_min) + params.p0_min
    )
    covs = torch.diag_embed(variances)
    return means, covs


def density_fields(params: StochasticCompareParams, means: Tensor, covs: Tensor) -> Tensor:
    """Gaussian density fields (n, N) at the grid for each (mean, cov) pair."""
    coords = grid_coords(params)
    return torch.stack(
        [gaussian_density_on_grid(means[i], covs[i], coords) for i in range(means.shape[0])]
    )


def analytic_final_moments(
    params: StochasticCompareParams, means: Tensor, covs: Tensor
) -> tuple[Tensor, Tensor]:
    """Exact OU moments at t_end for each initial condition."""
    a = _matrix(params.drift_matrix)
    b = torch.tensor(list(params.drift_bias), dtype=F64)
    g = _matrix(params.diffusion)
    q = g @ g.T
    final_means = torch.stack(
        [ou_mean(a, b, means[i], params.t_end) for i in range(means.shape[0])]
    )
    final_covs = torch.stack(
        [ou_covariance(a, q, covs[i], params.t_end) for i in range(means.shape[0])]
    )
    return final_means, final_covs


def build_shared_eval_set(
    params: StochasticCompareParams,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Held-out ICs + analytic targets, a function of ``eval_seed_base`` ONLY.

    Returns (eval_means, eval_covs, eval_inputs (n, N), eval_targets (n, N)).
    """
    means, covs = sample_initial_conditions(params, params.n_eval_samples, params.eval_seed_base)
    inputs = density_fields(params, means, covs)
    final_means, final_covs = analytic_final_moments(params, means, covs)
    targets = density_fields(params, final_means, final_covs)
    return means, covs, inputs, targets


# --------------------------------------------------------------------------- #
# Arms                                                                         #
# --------------------------------------------------------------------------- #


def build_stochastic_propagator(params: StochasticCompareParams) -> StrangSplitStep:
    """The stochastic arm's propagator (K=1, no jump, exact flows)."""
    cfg = StochasticGeneratorConfig(
        dim=2,
        drift_matrix=[list(r) for r in params.drift_matrix],
        drift_bias=list(params.drift_bias),
        diffusion=[list(r) for r in params.diffusion],
    )
    generator = KolmogorovGenerator(cfg)
    projection = GalerkinMomentProjection(
        generator, StrangSplittingConfig(dt=params.strang_dt, t_end=params.t_end)
    )
    return StrangSplitStep(projection)


def run_stochastic_arm(
    params: StochasticCompareParams,
    eval_means: Tensor,
    eval_covs: Tensor,
    eval_targets: Tensor,
) -> StochasticArmResult:
    """Propagate every eval IC through the Strang composition and score."""
    device = torch.device(params.device)
    stepper = build_stochastic_propagator(params)
    coords = grid_coords(params).to(device)
    t0 = time.perf_counter()
    predictions = []
    for i in range(eval_means.shape[0]):
        state = GaussianMixtureState(
            weights=torch.ones(1, dtype=F64, device=device),
            means=eval_means[i : i + 1].to(device),
            covariances=eval_covs[i : i + 1].to(device),
        )
        final, _t = stepper.propagate(state)[-1]
        predictions.append(final.density_on_grid(coords))
    stacked = torch.stack(predictions)
    wall = time.perf_counter() - t0
    mse = float(torch.mean((stacked - eval_targets.to(device)) ** 2))
    logger.info("stochastic_arm_done", density_mse=mse, wall_clock_s=wall)
    return StochasticArmResult(
        name="stochastic_galerkin", density_mse=mse, wall_clock_s=wall, n_params=0
    )


def build_operator(params: StochasticCompareParams, device: torch.device) -> PhysicsOperator:
    """The deterministic arm's Galerkin-attention operator (existing path)."""
    return PhysicsOperator(
        d_model=params.d_model,
        n_heads=params.n_heads,
        n_layers=params.n_layers,
        n_fourier_features=params.n_fourier_features,
        fourier_scale=params.fourier_scale,
        use_fnet=params.use_fnet,
        dropout=params.dropout,
    ).to(device)


def run_deterministic_arm(
    params: StochasticCompareParams,
    train_seed: int,
    eval_inputs: Tensor,
    eval_targets: Tensor,
) -> StochasticArmResult:
    """Train the operator on seeded ICs and score on the SHARED eval set."""
    device = torch.device(params.device)
    torch.manual_seed(train_seed)
    model = build_operator(params, device)
    n_params = sum(p.numel() for p in model.parameters())

    train_means, train_covs = sample_initial_conditions(params, params.n_train_samples, train_seed)
    train_inputs = density_fields(params, train_means, train_covs).to(
        dtype=torch.float32, device=device
    )
    final_means, final_covs = analytic_final_moments(params, train_means, train_covs)
    train_targets = density_fields(params, final_means, final_covs).to(
        dtype=torch.float32, device=device
    )
    coords_row = normalized_grid_coords(params).to(dtype=torch.float32, device=device)

    optimizer = AdamW(model.parameters(), lr=params.learning_rate)
    model.train()
    t0 = time.perf_counter()
    n = train_inputs.shape[0]
    order_gen = torch.Generator().manual_seed(train_seed)
    for _epoch in range(params.n_epochs):
        order = torch.randperm(n, generator=order_gen).to(device)
        for start in range(0, n, params.batch_size):
            idx = order[start : start + params.batch_size]
            batch_inputs = train_inputs[idx]
            batch_targets = train_targets[idx]
            coords = coords_row.unsqueeze(0).expand(batch_inputs.shape[0], -1, -1)
            optimizer.zero_grad()
            predictions = model(coords, batch_inputs)
            loss = nn.functional.mse_loss(predictions, batch_targets)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_inputs32 = eval_inputs.to(dtype=torch.float32, device=device)
        coords = coords_row.unsqueeze(0).expand(eval_inputs32.shape[0], -1, -1)
        predictions = model(coords, eval_inputs32)
        mse = float(torch.mean((predictions.to(F64) - eval_targets.to(device)) ** 2))
    wall = time.perf_counter() - t0
    logger.info("deterministic_arm_done", density_mse=mse, wall_clock_s=wall, n_params=n_params)
    return StochasticArmResult(
        name="deterministic_galerkin_attention",
        density_mse=mse,
        wall_clock_s=wall,
        n_params=n_params,
    )


# --------------------------------------------------------------------------- #
# Comparison drivers + artifacts                                               #
# --------------------------------------------------------------------------- #


def run_stochastic_galerkin_comparison(
    params: StochasticCompareParams, train_seed: int | None = None
) -> StochasticCompareResult:
    """Run both arms on the shared benchmark for one training seed."""
    seed = params.seed if train_seed is None else train_seed
    eval_means, eval_covs, eval_inputs, eval_targets = build_shared_eval_set(params)
    stochastic = run_stochastic_arm(params, eval_means, eval_covs, eval_targets)
    deterministic = run_deterministic_arm(params, seed, eval_inputs, eval_targets)
    return StochasticCompareResult(
        params=params,
        train_seed=seed,
        deterministic=deterministic,
        stochastic=stochastic,
    )


@dataclass
class MultiSeedStochasticComparison:
    """Per-seed results plus median-over-seeds summary metrics."""

    results: list[StochasticCompareResult]

    @property
    def representative(self) -> StochasticCompareResult:
        """The median-deterministic-MSE run (plots/CSV row)."""
        ordered = sorted(self.results, key=lambda r: r.deterministic.density_mse)
        return ordered[len(ordered) // 2]

    @property
    def metrics(self) -> dict[str, float]:
        """Median-over-seeds metrics (the stochastic arm is seed-independent)."""
        det = sorted(r.deterministic.density_mse for r in self.results)
        median_det = det[len(det) // 2]
        rep = self.representative
        return {
            **rep.metrics,
            "deterministic_density_mse_median": median_det,
            "n_seeds": float(len(self.results)),
        }


def run_multiseed_comparison(
    params: StochasticCompareParams, seeds: list[int] | None = None
) -> MultiSeedStochasticComparison:
    """Run the comparison across training seeds (stochastic arm is reused)."""
    resolved = seeds if seeds else [params.seed]
    results = [run_stochastic_galerkin_comparison(params, seed) for seed in resolved]
    return MultiSeedStochasticComparison(results=results)


def export_csv(comparison: MultiSeedStochasticComparison, output_path: str | Path) -> Path:
    """Write per-seed rows + a median summary row."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "train_seed",
                "stochastic_density_mse",
                "deterministic_density_mse",
                "mse_ratio",
                "stochastic_wall_clock_s",
                "deterministic_wall_clock_s",
                "deterministic_n_params",
            ]
        )
        for result in comparison.results:
            writer.writerow(
                [
                    "seed",
                    result.train_seed,
                    f"{result.stochastic.density_mse:.8e}",
                    f"{result.deterministic.density_mse:.8e}",
                    f"{result.mse_ratio:.6e}",
                    f"{result.stochastic.wall_clock_s:.4f}",
                    f"{result.deterministic.wall_clock_s:.4f}",
                    result.deterministic.n_params,
                ]
            )
        summary = comparison.metrics
        writer.writerow(
            [
                "median",
                "-",
                f"{summary['stochastic_density_mse']:.8e}",
                f"{summary['deterministic_density_mse_median']:.8e}",
                f"{summary['stochastic_vs_deterministic_mse_ratio']:.6e}",
                f"{summary['stochastic_wall_clock_s']:.4f}",
                f"{summary['deterministic_wall_clock_s']:.4f}",
                int(summary["deterministic_n_params"]),
            ]
        )
    logger.info("comparison_csv_written", path=str(path))
    return path


def export_plot(comparison: MultiSeedStochasticComparison, output_path: str | Path) -> Path | None:
    """Density heatmaps (target / stochastic / deterministic) for one eval IC."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover — matplotlib is a hard dep in practice
        logger.warning("matplotlib_unavailable_skipping_plot")
        return None

    result = comparison.representative
    params = result.params
    eval_means, eval_covs, eval_inputs, eval_targets = build_shared_eval_set(params)
    n = params.grid_n
    coords = grid_coords(params)

    stepper = build_stochastic_propagator(params)
    state = GaussianMixtureState(
        weights=torch.ones(1, dtype=F64),
        means=eval_means[0:1],
        covariances=eval_covs[0:1],
    )
    final, _t = stepper.propagate(state)[-1]
    stochastic_field = final.density_on_grid(coords).reshape(n, n)
    target_field = eval_targets[0].reshape(n, n)
    initial_field = eval_inputs[0].reshape(n, n)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, field, title in [
        (axes[0], initial_field, "initial density p(x, 0)"),
        (axes[1], target_field, f"analytic p(x, T={params.t_end})"),
        (axes[2], stochastic_field, "stochastic Galerkin prediction"),
    ]:
        im = axis.imshow(field.numpy(), origin="lower", cmap="viridis")
        axis.set_title(title, fontsize=10)
        fig.colorbar(im, ax=axis, fraction=0.046)
    metrics = comparison.metrics
    fig.suptitle(
        "Fokker-Planck/OU shared benchmark — "
        f"stochastic MSE {metrics['stochastic_density_mse']:.2e}, "
        f"deterministic MSE {metrics['deterministic_density_mse_median']:.2e} "
        "(ratio recorded, not gated)",
        fontsize=10,
    )
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("comparison_plot_written", path=str(path))
    return path
