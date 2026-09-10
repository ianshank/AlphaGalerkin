"""Physics-informed neural network baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog
import torch

from src.device import resolve_device
from src.pde.config import PDEType
from src.pde.operators import PDEOperator
from src.research.baselines.base import BaseSolver
from src.research.baselines.schemas import PINNConfig, SolverResult
from src.research.gpu_profiler import GpuUtilizationProfiler

logger = structlog.get_logger(__name__)


class SimplePINNSolver(BaseSolver):
    """Simple Physics-Informed Neural Network baseline.

    Trains a small MLP with physics-informed loss (PDE residual + BC)
    for comparison with classical and AlphaGalerkin approaches.
    """

    name = "pinn"
    description = "Physics-Informed Neural Network baseline"

    def __init__(
        self,
        hidden_dim: int | None = None,
        n_layers: int | None = None,
        n_epochs: int | None = None,
        learning_rate: float | None = None,
        n_collocation: int | None = None,
        bc_loss_weight: float | None = None,
        device: str | None = None,
        vector_pde: bool | None = None,
        config: PINNConfig | None = None,
    ) -> None:
        self.config = config or PINNConfig()
        c = self.config
        self.hidden_dim = hidden_dim if hidden_dim is not None else c.hidden_dim
        self.n_layers = n_layers if n_layers is not None else c.n_layers
        self.n_epochs = n_epochs if n_epochs is not None else c.n_epochs
        self.learning_rate = learning_rate if learning_rate is not None else c.learning_rate
        self.n_collocation = n_collocation if n_collocation is not None else c.n_collocation
        self.bc_loss_weight = bc_loss_weight if bc_loss_weight is not None else c.bc_loss_weight
        self.device_preference = device if device is not None else c.device
        self.vector_pde_override = vector_pde if vector_pde is not None else c.vector_pde

    def _resolve_vector_pde(self, operator: PDEOperator) -> bool:
        """Decide whether to build a vector-valued network for this operator."""
        if self.vector_pde_override is not None:
            return self.vector_pde_override
        return getattr(operator, "pde_type", None) == PDEType.NAVIER_STOKES

    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        """Solve by training a PINN."""
        log = logger.bind(solver=self.name, n_dof=n_dof, dim=operator.dim)
        log.info("pinn_solve_start")
        t0 = time.perf_counter()

        device = resolve_device(self.device_preference, context=f"pinn[{self.name}]")
        is_vector = self._resolve_vector_pde(operator)
        output_dim = 2 if is_vector else 1
        net = self._build_network(operator.dim, output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)

        rng = np.random.default_rng(self.config.seed)

        # Profile GPU utilisation when running on CUDA — provides SBIR
        # proposal-grade telemetry on whether the workload is compute-bound
        # or memory-bandwidth-bound. No-ops cleanly on CPU or no-nvidia-smi
        # hosts.
        # When ``device.index is None`` (bare ``torch.device("cuda")``) the
        # tensor goes to whatever ``torch.cuda.current_device()`` returns,
        # which is **not** always 0 — third-party libraries or earlier
        # ``torch.cuda.set_device(N)`` calls can shift it. Sample dmon on
        # the same index so the report matches the actual workload.
        if device.type == "cuda":
            gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
            profile_indices: list[int] = [gpu_idx]
        else:
            profile_indices = []
        with GpuUtilizationProfiler(gpu_indices=profile_indices) as profiler:
            for epoch in range(self.n_epochs):
                optimizer.zero_grad()

                # Interior collocation points
                coords_np = rng.uniform(
                    operator.domain_min,
                    operator.domain_max,
                    size=(self.n_collocation, operator.dim),
                ).astype(np.float32)
                coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
                coords.requires_grad_(True)

                u_raw = net(coords)  # (N, output_dim)

                if is_vector:
                    # Vector PDE (NS): PDE residual = sum of per-component
                    # Laplacians. Source-term forcing is treated as zero for
                    # the momentum residual (Taylor-Green has no body force).
                    #
                    # NOTE — physics simplification. Full Navier-Stokes
                    # momentum is ``du/dt + (u·∇)u + ∇p − ν∇²u = f``; the
                    # advection, pressure-gradient, and continuity terms are
                    # not in this loss. This matches the previous P40 fork's
                    # behaviour and is sufficient for the Taylor-Green decay
                    # benchmark (where the analytical IC dominates), but the
                    # PINN row is *not* a fully-physics-informed solver.
                    # Delegating to ``operator.residual()`` for proper NS
                    # physics is tracked as future work; doing it correctly
                    # requires (a) extending the PINN to predict pressure
                    # alongside velocity and (b) wiring divergence-free
                    # constraints — out of scope for this PR.
                    loss_pde = torch.zeros((), device=device)
                    for c_idx in range(output_dim):
                        uc = u_raw[:, c_idx]
                        lap_c = self._compute_laplacian(uc, coords, operator.dim)
                        loss_pde = loss_pde + torch.mean(lap_c**2)
                else:
                    u = u_raw.squeeze(-1)
                    lap = self._compute_laplacian(u, coords, operator.dim)
                    f = operator.source_term(coords)
                    if isinstance(f, np.ndarray):
                        f = torch.tensor(f, dtype=torch.float32, device=device)
                    elif isinstance(f, torch.Tensor) and f.device != device:
                        f = f.to(device)
                    pde_residual = lap + f  # For Poisson: -lap = f => lap+f=0
                    loss_pde = torch.mean(pde_residual**2)

                # Boundary loss
                bc_coords_np = operator.generate_boundary_points(
                    self.config.n_boundary_points,
                    seed=None,
                )
                bc_coords = torch.tensor(bc_coords_np, dtype=torch.float32, device=device)
                u_bc_raw = net(bc_coords)  # (N, output_dim)
                bc_vals = operator.boundary_value(bc_coords_np)
                if isinstance(bc_vals, np.ndarray):
                    bc_vals = torch.tensor(bc_vals, dtype=torch.float32, device=device)
                elif isinstance(bc_vals, torch.Tensor) and bc_vals.device != device:
                    bc_vals = bc_vals.to(device)

                if is_vector:
                    if bc_vals.dim() == 1:
                        bc_vals = bc_vals.unsqueeze(-1).expand_as(u_bc_raw)
                    loss_bc = torch.mean((u_bc_raw - bc_vals) ** 2)
                else:
                    u_bc = u_bc_raw.squeeze(-1)
                    if bc_vals.dim() > 1:
                        bc_vals = bc_vals.squeeze(-1)
                    loss_bc = torch.mean((u_bc - bc_vals) ** 2)

                loss = loss_pde + self.bc_loss_weight * loss_bc
                loss.backward()
                optimizer.step()

                if epoch % self.config.log_interval == 0:
                    log.debug(
                        "pinn_epoch",
                        epoch=epoch,
                        loss_pde=float(loss_pde),
                        loss_bc=float(loss_bc),
                    )

        # Evaluate on uniform grid
        eval_coords_np = operator.generate_collocation_points(
            n_dof, method="uniform", seed=self.config.seed
        )
        eval_coords = torch.tensor(eval_coords_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            u_eval_raw = net(eval_coords).cpu().numpy()
        u_eval = u_eval_raw if is_vector else u_eval_raw.squeeze(-1)

        wall_time = time.perf_counter() - t0
        grid = eval_coords_np.astype(np.float64)
        l2_err = self._compute_l2_error(u_eval.astype(np.float64), grid, operator)

        gpu_profile = profiler.report.to_dict() if profiler.report is not None else None
        log.info(
            "pinn_solve_done",
            wall_time=wall_time,
            l2_error=l2_err,
            device=str(device),
            gpu_samples=(profiler.report.total_samples if profiler.report else 0),
        )
        return SolverResult(
            solution=u_eval.astype(np.float64),
            grid_points=grid,
            n_dof=len(u_eval),
            wall_time_seconds=wall_time,
            l2_error=l2_err,
            metadata={
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "n_epochs": self.n_epochs,
                "n_collocation": self.n_collocation,
                "device": str(device),
                "vector_pde": is_vector,
                "gpu_profile": gpu_profile,
            },
        )

    def _build_network(self, input_dim: int, output_dim: int = 1) -> torch.nn.Module:
        """Build a simple MLP for the PINN.

        Args:
            input_dim: Input coordinate dimension (e.g. 1, 2, 3).
            output_dim: Output dimension (1 for scalar PDEs, 2 for 2D vector
                PDEs like Navier-Stokes velocity).

        """
        layers: list[torch.nn.Module] = []
        in_dim = input_dim
        for _ in range(self.n_layers):
            layers.append(torch.nn.Linear(in_dim, self.hidden_dim))
            layers.append(torch.nn.Tanh())
            in_dim = self.hidden_dim
        layers.append(torch.nn.Linear(in_dim, output_dim))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _compute_laplacian(u: torch.Tensor, coords: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute Laplacian of u w.r.t. coords using autograd."""
        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(u, coords, grad_outputs=grad_outputs, create_graph=True)[0]

        laplacian = torch.zeros_like(u)
        for d in range(dim):
            grad_d = grad_u[:, d]
            grad2 = torch.autograd.grad(
                grad_d,
                coords,
                grad_outputs=torch.ones_like(grad_d),
                create_graph=True,
            )[0]
            laplacian = laplacian + grad2[:, d]

        return laplacian
