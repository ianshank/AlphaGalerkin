"""Noyron HX zero-shot transfer scenario (Leap 71 integration).

Trains a 3D PINN-style :class:`PhysicsOperator` at low collocation-point
density on an SDF-bounded helical heat exchanger, then evaluates zero-shot
at higher density. Reports MSE on the steady-state temperature field
against either an analytical harmonic reference (CI-friendly) or the
in-repo voxel-FDM solver (headline run).

GPU is the preferred device. Setting ``config.device='cuda'`` (the default)
fails loud if CUDA is unavailable so we never silently fall back to a
20-minute CPU run.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from torch import Tensor, nn
from torch.optim import Adam

from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.config_noyron import NoyronHXScenarioConfig
from src.poc.device import resolve_device as _resolve_device
from src.poc.logging import ScenarioLogger
from src.poc.registry import BaseScenario, scenario

logger = structlog.get_logger(__name__)


@scenario("noyron_hx")
class NoyronHXScenario(BaseScenario):
    """Zero-shot 3D heat-equation transfer on Leap 71's helical HX."""

    config_class = NoyronHXScenarioConfig

    def __init__(
        self,
        config: NoyronHXScenarioConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        # Type-check refinement: tell mypy the config attribute is the
        # narrower NoyronHXScenarioConfig, not the BaseScenarioConfig
        # declared on the base class.
        self.config: NoyronHXScenarioConfig

        self._model: nn.Module | None = None
        self._device: torch.device | None = None
        self._output_dir: Path | None = None
        self._scenario_logger: ScenarioLogger | None = None
        self._operator: Any = None  # HelicalHeatOperator (lazy import)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self) -> None:
        from src.pde.config import PDEConfig, PDEType
        from src.pde.geometry import GeometryConfig, GeometryType
        from src.pde.operators_picogk import HelicalHeatOperator

        self._device = _resolve_device(
            self.config.device, context=f"NoyronHXScenario({self.name})"
        )
        self._output_dir = Path("outputs/poc/noyron_hx")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._scenario_logger = ScenarioLogger(
            scenario_name=self.name,
            config_hash=self.config.compute_hash(),
        )

        bbox_min = [
            -(self.config.helix_R_major + self.config.helix_r_minor),
            -(self.config.helix_R_major + self.config.helix_r_minor),
            0.0,
        ]
        bbox_max = [
            self.config.helix_R_major + self.config.helix_r_minor,
            self.config.helix_R_major + self.config.helix_r_minor,
            self.config.helix_pitch * self.config.helix_n_turns,
        ]

        pde_config = PDEConfig(
            name="noyron_hx_pde",
            pde_type=PDEType.HEAT,
            domain_dim=3,
            domain_min=bbox_min,
            domain_max=bbox_max,
            advection_coeff=[0.0, 0.0, 0.0],
            diffusion_coeff=self.config.diffusivity,
            geometry=GeometryConfig(
                geometry_type=GeometryType.PICOGK,
                sdf_kind=("picogk" if self.config.use_picogk else "analytical_helix"),
                picogk_voxel_path=self.config.picogk_voxel_path,
                helix_R_major=self.config.helix_R_major,
                helix_r_minor=self.config.helix_r_minor,
                helix_pitch=self.config.helix_pitch,
                helix_n_turns=self.config.helix_n_turns,
            ),
        )

        self._operator = HelicalHeatOperator(
            pde_config, diffusivity=self.config.diffusivity
        )

        self._scenario_logger.info(
            "setup_complete",
            device=str(self._device),
            output_dir=str(self._output_dir),
            sdf_kind=pde_config.geometry.sdf_kind,
        )

    def teardown(self) -> None:
        self._model = None
        self._operator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self) -> ScenarioResult:
        from src.experiments.physics_model import PhysicsOperator

        assert self._device is not None
        assert self._scenario_logger is not None
        assert self._operator is not None

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        model = PhysicsOperator(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            n_fourier_features=self.config.n_fourier_features,
            fourier_scale=self.config.fourier_scale,
            use_fnet=self.config.use_fnet,
            input_dim=3,
        ).to(self._device)
        self._model = model

        # ----- training -----
        self._scenario_logger.info(
            "training_start",
            n_pts=self.config.n_train_pts,
            n_boundary_pts=self.config.n_train_boundary_pts,
            n_epochs=self.config.n_epochs,
            ref_solver_kind=self.config.ref_solver_kind,
        )
        with self._scenario_logger.timed("training"):
            train_loss = self._train(model)
        self.record_metric("train_loss_final", float(train_loss))
        self._scenario_logger.metric("train_loss_final", float(train_loss))

        # ----- evaluation at training point density -----
        with self._scenario_logger.timed("eval_low_density"):
            mse_low = self._evaluate(model, self.config.n_train_pts, seed_offset=1)
        self.record_metric("mse_low", float(mse_low))
        self._scenario_logger.metric(
            "mse_low", float(mse_low), n_pts=self.config.n_train_pts
        )

        # ----- evaluation at zero-shot higher density -----
        with self._scenario_logger.timed("eval_high_density"):
            mse_high = self._evaluate(model, self.config.n_eval_pts, seed_offset=2)
        self.record_metric("mse_high", float(mse_high))
        self._scenario_logger.metric(
            "mse_high", float(mse_high), n_pts=self.config.n_eval_pts
        )

        transfer_ratio = float(mse_high / max(mse_low, 1e-12))
        self.record_metric("transfer_ratio", transfer_ratio)
        self._scenario_logger.metric("transfer_ratio", transfer_ratio)

        # ----- pass/fail -----
        threshold_results = {
            "mse_low": mse_low < self.config.mse_threshold_low,
            "mse_high": mse_high < self.config.mse_threshold_high,
            "transfer_ratio": transfer_ratio < self.config.transfer_ratio_threshold,
        }
        all_passed = all(threshold_results.values())
        status = ScenarioStatus.PASSED if all_passed else ScenarioStatus.FAILED

        # ----- artifact -----
        if self._output_dir is not None:
            ckpt_path = self._output_dir / f"model_{self.config.compute_hash()}.pt"
            torch.save(model.state_dict(), ckpt_path)
            self.record_artifact("model", str(ckpt_path))

        end_time = datetime.now()
        assert self._start_time is not None
        duration = (end_time - self._start_time).total_seconds()

        return ScenarioResult.model_validate(
            {
                "scenario_name": self.name,
                "config_hash": self.config.compute_hash(),
                "status": status,
                "passed": all_passed,
                "metrics": dict(self._metrics),
                "threshold_results": threshold_results,
                "artifacts": {k: str(v) for k, v in self._artifacts.items()},
                "start_time": self._start_time,
                "end_time": end_time,
                "duration_seconds": duration,
                "device": str(self._device),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "ref_solver_kind": self.config.ref_solver_kind,
                "transfer_ratio": transfer_ratio,
            }
        )

    # ------------------------------------------------------------------
    # Reference solutions
    # ------------------------------------------------------------------

    def _harmonic_reference(self, coords: Tensor) -> Tensor:
        """Closed-form temperature field used by analytical_harmonic mode.

        ``u(x, y, z) = sin(k x) + sin(k y) + sin(k z)`` for the configured
        wave number ``k``. The Laplacian is ``-k^2 (sum of sins)``, so
        ``_harmonic_source`` returns ``kappa * k^2 * (sum of sins)`` to
        make ``-kappa Laplacian u = f`` consistent.
        """
        k = self.config.harmonic_wave_number
        return (
            torch.sin(k * coords[:, 0])
            + torch.sin(k * coords[:, 1])
            + torch.sin(k * coords[:, 2])
        )

    def _harmonic_source(self, coords: Tensor) -> Tensor:
        """Source matching the harmonic reference under -kappa Laplacian u = f."""
        k = self.config.harmonic_wave_number
        return self.config.diffusivity * (k**2) * (
            torch.sin(k * coords[:, 0])
            + torch.sin(k * coords[:, 1])
            + torch.sin(k * coords[:, 2])
        )

    def _voxel_fdm_reference(self) -> tuple[Tensor, Tensor]:
        """Solve once on a 3D voxel grid via the in-repo FDM reference.

        Returns ``(coords, u)`` for the interior voxels: ``coords`` of
        shape ``(N, 3)`` and ``u`` of shape ``(N,)``.
        """
        from src.physics.voxel_fdm import solve_steady_heat_voxel, voxelize_sdf

        sdf = self._operator.geometry.sdf_evaluator

        def sdf_fn(pts: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                values = sdf.sdf(torch.from_numpy(pts.astype(np.float32)))
            return values.cpu().numpy()

        (mins, maxs) = self._operator.geometry.bounding_box()
        bbox_min: tuple[float, float, float] = (
            float(mins[0]),
            float(mins[1]),
            float(mins[2]),
        )
        bbox_max: tuple[float, float, float] = (
            float(maxs[0]),
            float(maxs[1]),
            float(maxs[2]),
        )
        interior_mask, voxel_coords = voxelize_sdf(
            sdf_fn=sdf_fn,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            resolution=self.config.voxel_fdm_resolution,
        )

        def boundary_value_fn(pts: np.ndarray) -> np.ndarray:
            return np.asarray(
                self._operator.boundary_value(pts), dtype=np.float32
            )

        u = solve_steady_heat_voxel(
            interior_mask=interior_mask,
            voxel_coords=voxel_coords,
            boundary_value_fn=boundary_value_fn,
            diffusivity=self.config.diffusivity,
            n_iterations=self.config.voxel_fdm_iterations,
            tolerance=self.config.voxel_fdm_tolerance,
        )

        flat_coords = voxel_coords.reshape(-1, 3)
        flat_u = u.reshape(-1)
        flat_mask = interior_mask.reshape(-1)
        return (
            torch.from_numpy(flat_coords[flat_mask]),
            torch.from_numpy(flat_u[flat_mask]),
        )

    # ------------------------------------------------------------------
    # Train / evaluate
    # ------------------------------------------------------------------

    def _normalize(self, coords: Tensor) -> Tensor:
        """Map world-space helix coords into ``[0, 1]^3`` for the encoder.

        ``PhysicsOperator``'s Fourier features expect normalized inputs;
        we use the geometry bounding box for the affine transform.
        """
        (mins, maxs) = self._operator.geometry.bounding_box()
        mins_t = torch.tensor(
            mins, dtype=coords.dtype, device=coords.device
        ).view(1, 3)
        maxs_t = torch.tensor(
            maxs, dtype=coords.dtype, device=coords.device
        ).view(1, 3)
        return (coords - mins_t) / (maxs_t - mins_t).clamp_min(1e-9)

    def _sample_training_batch(
        self, n_pts: int, n_boundary_pts: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Draw a single PINN training batch.

        Returns ``(interior_coords, interior_target, source, boundary_coords,
        boundary_target)``, all on the configured device.
        """
        assert self._device is not None
        assert self._operator is not None

        interior = self._operator.geometry.sample_interior(n_pts).to(self._device)
        boundary = self._operator.geometry.sample_boundary(n_boundary_pts).to(
            self._device
        )

        if self.config.ref_solver_kind == "analytical_harmonic":
            interior_target = self._harmonic_reference(interior)
            source = self._harmonic_source(interior)
            boundary_target = self._harmonic_reference(boundary)
        else:
            # voxel_fdm reference is precomputed once; here we still need
            # synthetic supervision for training. Use the harmonic field
            # for training and reserve voxel_fdm for the held-out eval.
            interior_target = self._harmonic_reference(interior)
            source = self._harmonic_source(interior)
            boundary_target = self._harmonic_reference(boundary)

        return interior, interior_target, source, boundary, boundary_target

    def _train(self, model: nn.Module) -> float:
        assert self._device is not None
        assert self._scenario_logger is not None
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        last_loss = float("inf")

        for epoch in range(self.config.n_epochs):
            (
                interior,
                interior_target,
                source,
                boundary,
                boundary_target,
            ) = self._sample_training_batch(
                self.config.n_train_pts, self.config.n_train_boundary_pts
            )

            coords_norm = self._normalize(interior)
            boundary_norm = self._normalize(boundary)

            optimizer.zero_grad()
            interior_pred = model(
                coords_norm.unsqueeze(0), source.unsqueeze(0)
            ).squeeze(0)
            boundary_pred = model(
                boundary_norm.unsqueeze(0),
                torch.zeros_like(boundary[:, 0]).unsqueeze(0),
            ).squeeze(0)

            interior_loss = loss_fn(interior_pred, interior_target)
            boundary_loss = loss_fn(boundary_pred, boundary_target)
            loss = interior_loss + boundary_loss
            loss.backward()
            optimizer.step()

            last_loss = float(loss.item())
            if (epoch + 1) % max(1, self.config.n_epochs // 10) == 0:
                self._scenario_logger.progress(
                    epoch + 1, self.config.n_epochs, operation="training"
                )
                self._scenario_logger.metric(
                    "train_loss",
                    last_loss,
                    epoch=epoch + 1,
                )
        return last_loss

    def _evaluate(
        self,
        model: nn.Module,
        n_pts: int,
        seed_offset: int,
    ) -> float:
        assert self._device is not None
        assert self._operator is not None

        torch.manual_seed(self.config.seed + seed_offset * 9973)

        model.eval()
        with torch.no_grad():
            if self.config.ref_solver_kind == "voxel_fdm":
                # Use the FDM reference; subsample to n_pts to keep eval
                # cost tractable.
                ref_coords, ref_u = self._voxel_fdm_reference()
                idx = torch.randperm(ref_coords.shape[0])[:n_pts]
                coords = ref_coords[idx].to(self._device)
                target = ref_u[idx].to(self._device)
                source = self._harmonic_source(coords)
            else:
                coords = self._operator.geometry.sample_interior(n_pts).to(
                    self._device
                )
                target = self._harmonic_reference(coords)
                source = self._harmonic_source(coords)

            coords_norm = self._normalize(coords)
            pred = model(
                coords_norm.unsqueeze(0), source.unsqueeze(0)
            ).squeeze(0)
            mse = float(((pred - target) ** 2).mean().item())
        model.train()
        return mse
