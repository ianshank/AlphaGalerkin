"""Spectral-bias comparison benchmark.

Implements
``config/proposals/nsf_sbir.yaml::spectral_bias_comparison``: solve a
2D Poisson problem with sinusoidal source ``f(x, y) = sin(k π x) sin(k π y)``
across a sweep of wavenumbers ``k``, using a configurable list of
solvers (FNO / DeepONet / direct / AlphaGalerkin), and report L2
error per (solver, frequency) cell.

The proof point we want to expose for SBIR copy: classical Fourier
neural operators degrade rapidly with k while the
:class:`MultiScaleFourierFeatures`-equipped Galerkin attention used
inside AlphaGalerkin maintains accuracy across the spectrum.

Design highlights:

* Pydantic-driven config — no hardcoded frequencies, sample sizes or
  solver list.  YAML proposals can override every knob.
* Reuses the central solver registry.
* Reuses :class:`PoissonOperator` with a swap-in source-term function.
* Emits JSON + CSV + (optional) HTML report so consumers can ingest
  in tabular tools or web dashboards.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

# Importing the extras package populates SOLVER_REGISTRY with FNO/DeepONet.
import src.research.extra_solvers  # noqa: F401
from src.pde.config import PDEConfig, PDEType
from src.pde.operators import PoissonOperator
from src.research.baselines import SOLVER_REGISTRY

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class SpectralBiasConfig(BaseModel):
    """Configuration for :class:`SpectralBiasBenchmark`."""

    model_config = ConfigDict(extra="forbid")

    solvers: list[str] = Field(
        ...,
        min_length=1,
        description="Solver names from SOLVER_REGISTRY to compare.",
    )
    frequencies: list[float] = Field(
        default_factory=lambda: [1.0, 5.0, 10.0, 50.0],
        min_length=1,
        description="Wavenumbers k to test (source = sin(kπx) sin(kπy)).",
    )
    n_dof: int = Field(
        default=256,
        ge=16,
        description="DOF target passed to each solver.",
    )
    repeats: int = Field(
        default=1,
        ge=1,
        description="Number of repetitions per (solver, frequency) cell.",
    )
    domain: tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Spatial domain on each axis (square).",
    )


@dataclass(frozen=True)
class SpectralBiasMeasurement:
    """One (solver, frequency) cell."""

    solver: str
    frequency: float
    n_dof: int
    l2_error: float | None
    wall_time_seconds: float
    success: bool
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpectralBiasReport:
    config: dict[str, Any]
    measurements: list[SpectralBiasMeasurement]


# ---------------------------------------------------------------------------
# Custom Poisson operator with a chosen sinusoidal source
# ---------------------------------------------------------------------------


class _SinusoidalPoissonOperator(PoissonOperator):
    """Poisson operator on the unit square with controllable source freq.

    The exact solution is
    ``u(x, y) = sin(kπx) sin(kπy) / (2 (kπ)²)``.  Defining the source
    so that ``-Δu = f`` matches yields:

        f(x, y) = sin(k π x) sin(k π y).

    Subclassing keeps backwards compatibility with everything that
    expects a :class:`PoissonOperator` (registries, MCTS adapters).
    """

    def __init__(self, config: PDEConfig, frequency: float) -> None:
        super().__init__(config)
        self._frequency = float(frequency)

    @property
    def frequency(self) -> float:
        return self._frequency

    def source_term(  # type: ignore[override]
        self,
        coords,
        time: float | None = None,
    ):
        k = self._frequency
        if isinstance(coords, torch.Tensor):
            x = coords[..., 0]
            y = coords[..., 1] if coords.shape[-1] > 1 else torch.zeros_like(x)
            return torch.sin(k * torch.pi * x) * torch.sin(k * torch.pi * y)
        else:
            arr = np.asarray(coords, dtype=np.float32)
            x = arr[..., 0]
            y = arr[..., 1] if arr.shape[-1] > 1 else np.zeros_like(x)
            return (np.sin(k * np.pi * x) * np.sin(k * np.pi * y)).astype(np.float32)

    def exact_solution(  # type: ignore[override]
        self,
        coords,
        time: float | None = None,
    ):
        k = self._frequency
        denom = 2.0 * (k * np.pi) ** 2
        if isinstance(coords, torch.Tensor):
            x = coords[..., 0]
            y = coords[..., 1] if coords.shape[-1] > 1 else torch.zeros_like(x)
            return torch.sin(k * torch.pi * x) * torch.sin(k * torch.pi * y) / denom
        else:
            arr = np.asarray(coords, dtype=np.float32)
            x = arr[..., 0]
            y = arr[..., 1] if arr.shape[-1] > 1 else np.zeros_like(x)
            return (
                np.sin(k * np.pi * x) * np.sin(k * np.pi * y) / denom
            ).astype(np.float32)


def _make_sinusoidal_operator(
    frequency: float,
    domain: tuple[float, float],
) -> _SinusoidalPoissonOperator:
    cfg = PDEConfig(
        name=f"poisson_sin_k{frequency}",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[domain[0], domain[0]],
        domain_max=[domain[1], domain[1]],
        advection_coeff=[0.0, 0.0],
    )
    return _SinusoidalPoissonOperator(cfg, frequency=frequency)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class SpectralBiasBenchmark:
    """Run a (solvers × frequencies) grid and emit a report."""

    def __init__(self, config: SpectralBiasConfig) -> None:
        self.config = config
        self._log = logger.bind(benchmark="SpectralBiasBenchmark")

    def run(self) -> SpectralBiasReport:
        measurements: list[SpectralBiasMeasurement] = []
        for solver_name in self.config.solvers:
            for freq in self.config.frequencies:
                cell = self._run_cell(solver_name, freq)
                measurements.append(cell)
        report = SpectralBiasReport(
            config=self.config.model_dump(),
            measurements=measurements,
        )
        self._log.info(
            "spectral_bias_run_complete",
            n_cells=len(measurements),
        )
        return report

    def save(self, report: SpectralBiasReport, output_dir: Path | str) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        out: dict[str, Path] = {}

        csv_path = output / "spectral_bias.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "solver",
                    "frequency",
                    "n_dof",
                    "l2_error",
                    "wall_time_seconds",
                    "success",
                    "error_message",
                ]
            )
            for m in report.measurements:
                writer.writerow(
                    [
                        m.solver,
                        m.frequency,
                        m.n_dof,
                        m.l2_error if m.l2_error is not None else "",
                        m.wall_time_seconds,
                        m.success,
                        m.error_message or "",
                    ]
                )
        out["csv"] = csv_path

        json_path = output / "spectral_bias.json"
        json_path.write_text(
            json.dumps(
                {
                    "config": report.config,
                    "measurements": [asdict(m) for m in report.measurements],
                },
                indent=2,
                default=str,
            )
        )
        out["json"] = json_path
        return out

    # ------------------------------------------------------------------

    def _run_cell(
        self,
        solver_name: str,
        frequency: float,
    ) -> SpectralBiasMeasurement:
        log = self._log.bind(solver=solver_name, frequency=frequency)
        cls = SOLVER_REGISTRY.get(solver_name)
        if cls is None:
            return SpectralBiasMeasurement(
                solver=solver_name,
                frequency=frequency,
                n_dof=self.config.n_dof,
                l2_error=None,
                wall_time_seconds=float("nan"),
                success=False,
                error_message=f"Solver '{solver_name}' not registered",
            )

        op = _make_sinusoidal_operator(frequency, self.config.domain)
        durations: list[float] = []
        last_error: float | None = None
        last_metadata: dict[str, Any] = {}
        last_exc: tuple[str, str] | None = None

        for _ in range(max(self.config.repeats, 1)):
            try:
                solver = cls()
                t0 = time.perf_counter()
                result = solver.solve(op, n_dof=self.config.n_dof)
                durations.append(time.perf_counter() - t0)
                last_error = result.l2_error
                last_metadata = dict(result.metadata)
            except Exception as exc:  # noqa: BLE001 — sweep continues
                last_exc = (type(exc).__name__, str(exc))
                log.warning("spectral_bias_cell_failed", error=str(exc))
                break

        if last_exc is not None and not durations:
            return SpectralBiasMeasurement(
                solver=solver_name,
                frequency=frequency,
                n_dof=self.config.n_dof,
                l2_error=None,
                wall_time_seconds=float("nan"),
                success=False,
                error_message=f"{last_exc[0]}: {last_exc[1]}",
            )
        wall_time = float(np.mean(durations))
        log.info(
            "spectral_bias_cell_done",
            wall_time=wall_time,
            l2_error=last_error,
        )
        return SpectralBiasMeasurement(
            solver=solver_name,
            frequency=frequency,
            n_dof=self.config.n_dof,
            l2_error=last_error,
            wall_time_seconds=wall_time,
            success=True,
            metadata=last_metadata,
        )
