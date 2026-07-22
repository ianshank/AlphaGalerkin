"""Honest zero-shot-transfer head-to-head: AlphaGalerkin operator vs a retrained CNN.

This module runs the thesis-critical controlled experiment behind the
``transfer_baseline_compare`` PoC scenario. It replaces the fabricated
"MSE 0.000209 / 240x better than threshold" self-comparison (a number no code ever
computed) with a *falsifiable* claim measured against an honest baseline:

    The resolution-independent :class:`~src.experiments.physics_model.PhysicsOperator`,
    trained ONLY at ``train_resolution`` and applied zero-shot at ``target_resolution``,
    matches or beats a discrete CNN **retrained at** ``target_resolution``.

Both arms train on their own :class:`~src.physics.poisson.PoissonDataset` and score MSE
on a single **shared held-out** dataset at the target resolution (eval seed a function
of resolution only), over the identical ``PoissonSample.potential`` targets — the only
difference is the model and the training resolution.

Reported honestly, mirroring ``specs/lshape_amr_compare.spec.md`` (AC3/AC4):

* ``transfer_mse_ratio_<t>x<t>`` — operator-zero-shot / CNN-retrained MSE, the gated
  headline, taken as the **median across seeds** (a single training run is
  high-variance).
* ``transfer_mse_ratio_<t>x<t>_matched_compute`` — the same ratio when the CNN is given
  a training budget matched to the operator's actual cost (recorded, not gated).
* ``mse_cnn_zeroshot_<t>x<t>`` — a CNN trained at ``train_resolution`` then evaluated at
  ``target_resolution``: the mechanism check proving the discrete baseline *cannot*
  transfer and must be retrained.

If the operator LOSES (ratio >= 1) the harness still reports it faithfully — that is the
benchmark's purpose. The gate is calibrated from a measured run, not assumed.

Design note: the operator training loop mirrors
:meth:`src.poc.scenarios.transfer.TransferScenario._train_model` rather than sharing
code with it. ``transfer.py`` sits on a documented regression surface; duplicating a
~30-line loop here keeps that scenario byte-for-byte untouched (the low-risk option the
plan permitted) at the cost of a small, deliberate repetition.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import structlog
import torch
from torch import Tensor, nn
from torch.optim import AdamW

from src.experiments.cnn_baseline import DiscreteCNNBaseline, count_parameters, match_cnn_channels
from src.experiments.physics_model import PhysicsOperator
from src.physics.poisson import PoissonDataset, PoissonSample

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Named numerical constants (no magic numbers)                                 #
# --------------------------------------------------------------------------- #

# Floor for any MSE denominator so the ratio stays finite even if a retrained CNN
# reaches (near) zero error on a trivial target.
TRANSFER_RATIO_FLOOR: float = 1e-15

# Prime stride decorrelating per-seed RNG streams in the multi-seed sweep (mirrors
# lshape_amr_compare / scaling_law).
SEED_PRIME_STRIDE: int = 7919


# --------------------------------------------------------------------------- #
# Data contracts                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TransferComparisonParams:
    """All tunables for one operator-vs-CNN transfer comparison run.

    Every field is explicit and defaulted so the harness is reusable and testable
    without the PoC scenario layer. The PoC config
    (:class:`~src.poc.scenarios.transfer_baseline_compare_config.TransferBaselineCompareConfig`)
    supplies these from validated Pydantic fields.
    """

    seed: int = 42
    device: str = "cpu"
    # Resolutions
    train_resolution: int = 9
    target_resolution: int = 19
    secondary_resolutions: tuple[int, ...] = (9, 13)
    # Data
    n_train_samples: int = 5000
    n_eval_samples: int = 500
    n_charges: int = 5
    charge_std: float = 1.0
    # Training (shared by both arms → matched training budget)
    batch_size: int = 32
    n_epochs: int = 100
    learning_rate: float = 1e-3
    eval_seed_base: int = 50000
    # Operator architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    n_fourier_features: int = 64
    fourier_scale: float = 10.0
    use_fnet: bool = True
    dropout: float = 0.1
    # CNN baseline architecture
    cnn_n_layers: int = 6
    cnn_kernel_size: int = 3
    cnn_channels: int | None = None
    cnn_use_batchnorm: bool = True
    cnn_dropout: float = 0.0
    cnn_param_match_tolerance: float = 0.15
    # Matched-compute variant: "grad_steps" (deterministic; equals the primary CNN
    # arm since both arms share n_epochs/n_train_samples/batch_size) or "wall_clock"
    # (the CNN retrains for the seconds the operator's training consumed → a genuinely
    # distinct end-to-end number, used in the full/nightly run).
    matched_budget_mode: str = "grad_steps"
    n_seeds: int = 5

    def __post_init__(self) -> None:
        """Validate invariants that direct harness callers must also honour."""
        if self.train_resolution < 3:
            raise ValueError(f"train_resolution must be >= 3, got {self.train_resolution}")
        if self.target_resolution < 5:
            raise ValueError(f"target_resolution must be >= 5, got {self.target_resolution}")
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")
        valid_modes = {"grad_steps", "wall_clock"}
        if self.matched_budget_mode not in valid_modes:
            raise ValueError(
                f"matched_budget_mode must be one of {sorted(valid_modes)}, "
                f"got {self.matched_budget_mode!r}"
            )

    @property
    def all_resolutions(self) -> list[int]:
        """Distinct resolutions the operator is evaluated at (target + secondaries)."""
        seen: dict[int, None] = {}
        for res in (*self.secondary_resolutions, self.target_resolution):
            seen[int(res)] = None
        return sorted(seen)


@dataclass
class TransferArmResult:
    """One arm's outcome: its target-resolution MSE plus provenance."""

    arm: str
    mse_at_target: float
    n_params: int
    train_seconds: float
    train_steps: int
    mse_by_resolution: dict[int, float] = field(default_factory=dict)


@dataclass
class TransferComparisonResult:
    """Outcome of one seed's comparison: all arms + the headline ratios."""

    operator: TransferArmResult
    cnn_retrained: TransferArmResult
    cnn_zeroshot: TransferArmResult
    cnn_matched_compute: TransferArmResult
    transfer_mse_ratio: float
    transfer_mse_ratio_matched_compute: float
    param_count_ratio: float
    target_resolution: int
    seed: int

    def metrics(self) -> dict[str, float]:
        """Flat, resolution-suffixed metric dict for the scenario/baseline harness."""
        t = self.target_resolution
        out: dict[str, float] = {
            f"transfer_mse_ratio_{t}x{t}": self.transfer_mse_ratio,
            f"transfer_mse_ratio_{t}x{t}_matched_compute": (
                self.transfer_mse_ratio_matched_compute
            ),
            f"mse_alphagalerkin_zeroshot_{t}x{t}": self.operator.mse_at_target,
            f"mse_cnn_retrained_{t}x{t}": self.cnn_retrained.mse_at_target,
            f"mse_cnn_zeroshot_{t}x{t}": self.cnn_zeroshot.mse_at_target,
            f"mse_cnn_matched_compute_{t}x{t}": self.cnn_matched_compute.mse_at_target,
            "param_count_ratio": self.param_count_ratio,
            "alphagalerkin_n_params": float(self.operator.n_params),
            "cnn_n_params": float(self.cnn_retrained.n_params),
            "operator_train_seconds": self.operator.train_seconds,
            "cnn_train_seconds": self.cnn_retrained.train_seconds,
        }
        # Operator zero-shot curve (mirrors the historical transfer scenario keys).
        for res, mse in self.operator.mse_by_resolution.items():
            out[f"mse_{res}x{res}"] = mse
        return out


# --------------------------------------------------------------------------- #
# Model builders                                                              #
# --------------------------------------------------------------------------- #


def build_operator(params: TransferComparisonParams, device: torch.device) -> PhysicsOperator:
    """Instantiate the resolution-independent operator on ``device``."""
    return PhysicsOperator(
        d_model=params.d_model,
        n_heads=params.n_heads,
        n_layers=params.n_layers,
        n_fourier_features=params.n_fourier_features,
        fourier_scale=params.fourier_scale,
        use_fnet=params.use_fnet,
        dropout=params.dropout,
    ).to(device)


def resolve_cnn_channels(params: TransferComparisonParams, operator_n_params: int) -> int:
    """Resolve the CNN channel width (explicit, or matched to the operator's params)."""
    if params.cnn_channels is not None:
        return params.cnn_channels
    return match_cnn_channels(
        operator_n_params,
        n_layers=params.cnn_n_layers,
        kernel_size=params.cnn_kernel_size,
        use_batchnorm=params.cnn_use_batchnorm,
        tolerance=params.cnn_param_match_tolerance,
    )


def build_cnn(
    params: TransferComparisonParams, channels: int, device: torch.device
) -> DiscreteCNNBaseline:
    """Instantiate the discrete CNN baseline on ``device``."""
    return DiscreteCNNBaseline(
        n_layers=params.cnn_n_layers,
        channels=channels,
        kernel_size=params.cnn_kernel_size,
        use_batchnorm=params.cnn_use_batchnorm,
        dropout=params.cnn_dropout,
    ).to(device)


# --------------------------------------------------------------------------- #
# Data / training / evaluation primitives                                     #
# --------------------------------------------------------------------------- #


def _stack_batch(
    samples: list[PoissonSample], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Stack a list of PoissonSamples into (coords, charges, targets) tensors."""
    coords = torch.tensor(np.stack([s.coords for s in samples]), dtype=torch.float32, device=device)
    charges = torch.tensor(
        np.stack([s.charges for s in samples]), dtype=torch.float32, device=device
    )
    targets = torch.tensor(
        np.stack([s.potential for s in samples]), dtype=torch.float32, device=device
    )
    return coords, charges, targets


def _make_dataset(
    params: TransferComparisonParams, resolution: int, seed: int, n_samples: int
) -> PoissonDataset:
    """Build a cached PoissonDataset at ``resolution`` with the given ``seed``."""
    return PoissonDataset(
        grid_size=resolution,
        n_samples=n_samples,
        n_charges=params.n_charges,
        charge_std=params.charge_std,
        seed=seed,
    )


def _train_model(
    model: nn.Module,
    dataset: PoissonDataset,
    params: TransferComparisonParams,
    device: torch.device,
    *,
    forward: str,
    max_steps: int | None = None,
    max_seconds: float | None = None,
) -> tuple[float, int]:
    """Train ``model`` on ``dataset`` with plain MSE. Returns (seconds, steps).

    Args:
        model: Either a PhysicsOperator (``forward="operator"``) or a
            DiscreteCNNBaseline (``forward="cnn"``).
        dataset: Training dataset (its own resolution/seed).
        params: Comparison params (epochs, lr, batch size).
        device: Torch device.
        forward: ``"operator"`` calls ``model(coords, charges)``; ``"cnn"`` calls
            ``model(charges)``.
        max_steps: Optional hard cap on gradient steps (matched-grad-steps budget).
        max_seconds: Optional wall-clock cap (matched-wall-clock budget); checked
            between steps.

    Returns:
        ``(train_seconds, train_steps)``.

    """
    optimizer = AdamW(model.parameters(), lr=params.learning_rate)
    model.train()
    steps = 0
    t0 = time.perf_counter()
    for _epoch in range(params.n_epochs):
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        for start in range(0, len(indices), params.batch_size):
            batch_idx = indices[start : start + params.batch_size]
            samples = [dataset[int(i)] for i in batch_idx]
            coords, charges, targets = _stack_batch(samples, device)
            optimizer.zero_grad()
            predictions = model(coords, charges) if forward == "operator" else model(charges)
            loss = torch.nn.functional.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return time.perf_counter() - t0, steps
        if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
            break
    return time.perf_counter() - t0, steps


@torch.no_grad()
def evaluate_on_grid(
    model: nn.Module,
    resolution: int,
    params: TransferComparisonParams,
    device: torch.device,
    *,
    forward: str,
) -> float:
    """Compute MSE of ``model`` on the SHARED held-out set at ``resolution``.

    The eval dataset seed is ``eval_seed_base + resolution`` — a function of
    resolution only — so every arm and every training seed scores on the identical
    held-out data (fairness invariant AC2).
    """
    dataset = _make_dataset(
        params, resolution, params.eval_seed_base + resolution, params.n_eval_samples
    )
    model.eval()
    sq_error_sum = 0.0
    count = 0
    for start in range(0, len(dataset), params.batch_size):
        samples = [dataset[i] for i in range(start, min(start + params.batch_size, len(dataset)))]
        coords, charges, targets = _stack_batch(samples, device)
        predictions = model(coords, charges) if forward == "operator" else model(charges)
        sq_error_sum += float(torch.sum((predictions - targets) ** 2).item())
        count += targets.numel()
    return sq_error_sum / max(count, 1)


# --------------------------------------------------------------------------- #
# Single-seed comparison                                                       #
# --------------------------------------------------------------------------- #


def _seed_everything(seed: int) -> None:
    """Seed torch + numpy for a reproducible training run."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_transfer_comparison(params: TransferComparisonParams) -> TransferComparisonResult:
    """Train all arms for one seed and assemble the :class:`TransferComparisonResult`."""
    device = torch.device(params.device)
    t = params.target_resolution

    logger.info(
        "transfer_comparison_start",
        seed=params.seed,
        train_resolution=params.train_resolution,
        target_resolution=t,
        matched_budget_mode=params.matched_budget_mode,
    )

    # --- Operator arm: train at train_resolution, evaluate the full curve. ---
    _seed_everything(params.seed)
    operator_model = build_operator(params, device)
    op_dataset = _make_dataset(params, params.train_resolution, params.seed, params.n_train_samples)
    op_seconds, op_steps = _train_model(
        operator_model, op_dataset, params, device, forward="operator"
    )
    op_mse_by_resolution = {
        res: evaluate_on_grid(operator_model, res, params, device, forward="operator")
        for res in params.all_resolutions
    }
    op_n_params = count_parameters(operator_model)
    operator = TransferArmResult(
        arm="operator",
        mse_at_target=op_mse_by_resolution[t],
        n_params=op_n_params,
        train_seconds=op_seconds,
        train_steps=op_steps,
        mse_by_resolution=op_mse_by_resolution,
    )

    # CNN width matched to the operator's parameter count (secondary sanity metric).
    channels = resolve_cnn_channels(params, op_n_params)

    # --- CNN retrained at the target resolution (matched training budget). ---
    _seed_everything(params.seed)
    cnn_model = build_cnn(params, channels, device)
    cnn_dataset = _make_dataset(params, t, params.seed, params.n_train_samples)
    cnn_seconds, cnn_steps = _train_model(cnn_model, cnn_dataset, params, device, forward="cnn")
    cnn_mse = evaluate_on_grid(cnn_model, t, params, device, forward="cnn")
    cnn_n_params = count_parameters(cnn_model)
    cnn_retrained = TransferArmResult(
        arm="cnn_retrained",
        mse_at_target=cnn_mse,
        n_params=cnn_n_params,
        train_seconds=cnn_seconds,
        train_steps=cnn_steps,
        mse_by_resolution={t: cnn_mse},
    )

    # --- CNN zero-shot (train at train_resolution, eval at target): AC1 mechanism. ---
    _seed_everything(params.seed)
    cnn_zs_model = build_cnn(params, channels, device)
    cnn_zs_dataset = _make_dataset(
        params, params.train_resolution, params.seed, params.n_train_samples
    )
    zs_seconds, zs_steps = _train_model(cnn_zs_model, cnn_zs_dataset, params, device, forward="cnn")
    cnn_zs_mse = evaluate_on_grid(cnn_zs_model, t, params, device, forward="cnn")
    cnn_zeroshot = TransferArmResult(
        arm="cnn_zeroshot",
        mse_at_target=cnn_zs_mse,
        n_params=count_parameters(cnn_zs_model),
        train_seconds=zs_seconds,
        train_steps=zs_steps,
        mse_by_resolution={t: cnn_zs_mse},
    )

    # --- CNN matched-compute. In grad_steps mode this is exactly the primary CNN
    #     arm (both share n_epochs/n_train_samples/batch_size → identical steps), so
    #     we alias it rather than retrain. In wall_clock mode the CNN retrains for the
    #     seconds the operator consumed → a genuinely distinct end-to-end number. ---
    if params.matched_budget_mode == "grad_steps":
        cnn_matched_compute = replace(cnn_retrained, arm="cnn_matched_compute")
    else:
        _seed_everything(params.seed)
        mc_model = build_cnn(params, channels, device)
        mc_dataset = _make_dataset(params, t, params.seed, params.n_train_samples)
        mc_seconds, mc_steps = _train_model(
            mc_model, mc_dataset, params, device, forward="cnn", max_seconds=op_seconds
        )
        mc_mse = evaluate_on_grid(mc_model, t, params, device, forward="cnn")
        cnn_matched_compute = TransferArmResult(
            arm="cnn_matched_compute",
            mse_at_target=mc_mse,
            n_params=count_parameters(mc_model),
            train_seconds=mc_seconds,
            train_steps=mc_steps,
            mse_by_resolution={t: mc_mse},
        )

    ratio = operator.mse_at_target / max(cnn_retrained.mse_at_target, TRANSFER_RATIO_FLOOR)
    ratio_mc = operator.mse_at_target / max(cnn_matched_compute.mse_at_target, TRANSFER_RATIO_FLOOR)
    param_ratio = cnn_n_params / max(op_n_params, 1)

    logger.info(
        "transfer_comparison_done",
        seed=params.seed,
        transfer_mse_ratio=ratio,
        mse_operator_zeroshot=operator.mse_at_target,
        mse_cnn_retrained=cnn_retrained.mse_at_target,
        mse_cnn_zeroshot=cnn_zeroshot.mse_at_target,
    )
    return TransferComparisonResult(
        operator=operator,
        cnn_retrained=cnn_retrained,
        cnn_zeroshot=cnn_zeroshot,
        cnn_matched_compute=cnn_matched_compute,
        transfer_mse_ratio=ratio,
        transfer_mse_ratio_matched_compute=ratio_mc,
        param_count_ratio=param_ratio,
        target_resolution=t,
        seed=params.seed,
    )


# --------------------------------------------------------------------------- #
# Multi-seed aggregation                                                       #
# --------------------------------------------------------------------------- #


def resolved_seeds(base_seed: int, n_seeds: int) -> list[int]:
    """Deterministic, decorrelated per-seed RNG seeds for the sweep."""
    return [base_seed + i * SEED_PRIME_STRIDE for i in range(n_seeds)]


@dataclass
class MultiSeedTransferComparison:
    """Aggregate of ``n_seeds`` single-seed comparisons.

    A single training run is high-variance, so the headline ratio is the **median**
    across seeds, with the full per-seed spread recorded for honesty. Mirrors the
    lshape_amr_compare / scaling_law median-over-seeds convention.
    """

    per_seed: list[TransferComparisonResult]
    seeds: list[int]

    @property
    def ratios(self) -> list[float]:
        """Per-seed transfer MSE ratios (operator zero-shot / CNN retrained)."""
        return [r.transfer_mse_ratio for r in self.per_seed]

    @property
    def representative(self) -> TransferComparisonResult:
        """The per-seed result whose ratio is the median — used for the artifact."""
        ratios = self.ratios
        order = sorted(range(len(ratios)), key=lambda i: ratios[i])
        return self.per_seed[order[len(order) // 2]]

    def metrics(self) -> dict[str, float]:
        """Headline (median) metrics plus per-seed spread and win fraction.

        The gated key ``transfer_mse_ratio_<t>x<t>`` is the **median** across seeds;
        absolutes come from the representative (median) seed.
        """
        t = self.representative.target_resolution
        ratios = np.array(self.ratios, dtype=np.float64)
        ratios_mc = np.array(
            [r.transfer_mse_ratio_matched_compute for r in self.per_seed], dtype=np.float64
        )
        out = dict(self.representative.metrics())
        out.update(
            {
                f"transfer_mse_ratio_{t}x{t}": float(np.median(ratios)),
                f"transfer_mse_ratio_{t}x{t}_matched_compute": float(np.median(ratios_mc)),
                "transfer_ratio_seed_min": float(np.min(ratios)),
                "transfer_ratio_seed_max": float(np.max(ratios)),
                "transfer_ratio_seed_std": float(np.std(ratios)),
                "alphagalerkin_win_fraction": float(np.mean(ratios < 1.0)),
                "n_seeds": float(len(self.per_seed)),
            }
        )
        return out


def run_multiseed_transfer_comparison(
    params: TransferComparisonParams,
) -> MultiSeedTransferComparison:
    """Run ``params.n_seeds`` comparisons and aggregate the median headline.

    Seeds are derived from ``params.seed`` via :func:`resolved_seeds` so the run is
    fully reproducible. Both arms are stochastic (unlike lshape's deterministic Dörfler
    arm), so every seed re-runs a full :func:`run_transfer_comparison`.
    """
    seeds = resolved_seeds(params.seed, params.n_seeds)
    per_seed = [run_transfer_comparison(replace(params, seed=s)) for s in seeds]
    result = MultiSeedTransferComparison(per_seed=per_seed, seeds=seeds)
    logger.info(
        "transfer_multiseed_done",
        n_seeds=len(seeds),
        median_ratio=float(np.median(result.ratios)),
        win_fraction=float(np.mean(np.array(result.ratios) < 1.0)),
    )
    return result


# --------------------------------------------------------------------------- #
# Artifact writers                                                            #
# --------------------------------------------------------------------------- #


def export_csv(comparison: MultiSeedTransferComparison, output_path: str | Path) -> Path:
    """Write one row per (seed, arm, resolution) to CSV.

    Columns: ``problem, seed, arm, resolution, mse, n_params, train_seconds,
    train_steps``. The headline ratios are recomputable from these raw rows, so the
    CSV is the reproducible record.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "problem",
                "seed",
                "arm",
                "resolution",
                "mse",
                "n_params",
                "train_seconds",
                "train_steps",
            ]
        )
        for result in comparison.per_seed:
            arms = (
                result.operator,
                result.cnn_retrained,
                result.cnn_zeroshot,
                result.cnn_matched_compute,
            )
            for arm in arms:
                for res, mse in arm.mse_by_resolution.items():
                    writer.writerow(
                        [
                            "poisson_transfer",
                            result.seed,
                            arm.arm,
                            res,
                            f"{mse:.8e}",
                            arm.n_params,
                            f"{arm.train_seconds:.6f}",
                            arm.train_steps,
                        ]
                    )
    logger.info("transfer_csv_export", path=str(path), n_seeds=len(comparison.per_seed))
    return path


def export_plot(comparison: MultiSeedTransferComparison, output_path: str | Path) -> Path | None:
    """Render the honest comparison PNG (bar chart + operator zero-shot curve).

    Matplotlib is imported lazily (``Agg`` backend) so importing this module never
    requires it. Returns ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - environment guard
        logger.warning("transfer_plot_skipped", reason="matplotlib unavailable")
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rep = comparison.representative
    metrics = comparison.metrics()
    t = rep.target_resolution

    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(11, 4.5))

    labels = ["AG\nzero-shot", "CNN\nretrained", "CNN\nzero-shot"]
    values = [
        rep.operator.mse_at_target,
        rep.cnn_retrained.mse_at_target,
        rep.cnn_zeroshot.mse_at_target,
    ]
    colours = ["#2ca02c", "#1f77b4", "#d62728"]
    ax_bar.bar(labels, values, color=colours)
    ax_bar.set_yscale("log")
    ax_bar.set_ylabel(f"MSE @ {t}x{t}")
    median_ratio = metrics[f"transfer_mse_ratio_{t}x{t}"]
    ax_bar.set_title(
        f"Transfer MSE ratio (AG zero-shot / CNN retrained)\nmedian = {median_ratio:.3f} "
        f"over {int(metrics['n_seeds'])} seeds"
    )

    curve_res = sorted(rep.operator.mse_by_resolution)
    curve_mse = [rep.operator.mse_by_resolution[r] for r in curve_res]
    ax_curve.plot(curve_res, curve_mse, "o-", color="#2ca02c", label="AG operator (zero-shot)")
    ax_curve.scatter(
        [t], [rep.cnn_retrained.mse_at_target], color="#1f77b4", zorder=3, label="CNN retrained"
    )
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("grid size (N x N)")
    ax_curve.set_ylabel("MSE")
    ax_curve.set_title("Operator zero-shot MSE vs grid size")
    ax_curve.set_xticks(curve_res)
    ax_curve.legend()
    ax_curve.grid(True, which="both", alpha=0.3)

    fig.suptitle("Honest zero-shot transfer: AlphaGalerkin operator vs retrained CNN")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("transfer_plot_export", path=str(path))
    return path
