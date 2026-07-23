"""Deterministic-vs-stochastic Galerkin comparison scenario (NKE layer, AC8).

Wraps ``src.research.stochastic_galerkin_compare`` in the PoC lifecycle: both
arms score on the shared 2D Fokker-Planck/OU benchmark; only the stochastic
arm's absolute density MSE is gated (see the config module's honesty rule).
The two solver paths stay separate — this scenario shares only benchmark data
structures between them (change-doc constraint).

Spec: specs/stochastic_galerkin_nke.spec.md (AC8).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.device import resolve_device
from src.poc.logging import ScenarioLogger
from src.poc.registry import BaseScenario, scenario
from src.poc.scenarios.stochastic_galerkin_compare_config import (
    SCENARIO_NAME,
    StochasticGalerkinCompareConfig,
)

if TYPE_CHECKING:
    from src.research.stochastic_galerkin_compare import (
        MultiSeedStochasticComparison,
        StochasticCompareParams,
    )


@scenario(SCENARIO_NAME)
class StochasticGalerkinCompareScenario(BaseScenario):
    """Galerkin attention vs stochastic Galerkin projection on shared FP/OU data."""

    config_class = StochasticGalerkinCompareConfig

    def __init__(
        self, config: StochasticGalerkinCompareConfig | None = None, **kwargs: Any
    ) -> None:
        super().__init__(config, **kwargs)
        self.config: StochasticGalerkinCompareConfig  # type narrowing
        self._device: torch.device | None = None
        self._scenario_logger: ScenarioLogger | None = None

    def setup(self) -> None:
        """Resolve the device, build the logger, and install the threshold."""
        self._device = resolve_device(self.config.device, context=SCENARIO_NAME)
        self._scenario_logger = ScenarioLogger(
            scenario_name=self.name,
            run_id=self.config.compute_hash(),
            device=str(self._device),
        )
        if not self.config.thresholds:
            self.config.thresholds = self.config.get_default_thresholds()
        self._scenario_logger.info(
            "setup_complete",
            grid_n=self.config.grid_n,
            n_seeds=self.config.n_seeds,
            seed=self.config.seed,
        )

    def teardown(self) -> None:
        """Release GPU memory (no-op on CPU)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def execute(self) -> ScenarioResult:
        """Run both arms, record metrics, and write the CSV/PNG artifacts."""
        assert self._scenario_logger is not None
        # Lazy import keeps the config module light for load_config_from_dict.
        from src.research.stochastic_galerkin_compare import (
            export_csv,
            export_plot,
            run_multiseed_comparison,
        )

        params = self._build_params()
        comparison = run_multiseed_comparison(params, seeds=self.config.resolved_seeds())
        self._record_metrics(comparison)
        self._write_artifacts(comparison, export_csv, export_plot)
        return self._create_result(status=ScenarioStatus.RUNNING)

    def _build_params(self) -> StochasticCompareParams:
        """Assemble harness params from the validated config.

        The device is the scenario's resolved one (``resolve_device`` handles
        'auto' GPU-preferred / 'cuda' fail-loud / 'cpu'), so both arms are
        GPU/CPU agnostic.
        """
        from src.research.stochastic_galerkin_compare import StochasticCompareParams

        assert self._device is not None
        cfg = self.config
        return StochasticCompareParams(
            device=str(self._device),
            grid_n=cfg.grid_n,
            domain_half_width=cfg.domain_half_width,
            drift_matrix=tuple(tuple(r) for r in cfg.drift_matrix),
            drift_bias=tuple(cfg.drift_bias),
            diffusion=tuple(tuple(r) for r in cfg.diffusion),
            t_end=cfg.t_end,
            strang_dt=cfg.strang_dt,
            n_train_samples=cfg.n_train_samples,
            n_eval_samples=cfg.n_eval_samples,
            m0_half_range=cfg.m0_half_range,
            p0_min=cfg.p0_min,
            p0_max=cfg.p0_max,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            n_fourier_features=cfg.n_fourier_features,
            fourier_scale=cfg.fourier_scale,
            use_fnet=cfg.use_fnet,
            dropout=cfg.dropout,
            n_epochs=cfg.n_epochs,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            eval_seed_base=cfg.eval_seed_base,
        )

    def _record_metrics(self, comparison: MultiSeedStochasticComparison) -> None:
        """Record the gated MSE plus the ungated comparison diagnostics."""
        assert self._scenario_logger is not None
        for name, value in comparison.metrics.items():
            self.record_metric(name, value)
            self._scenario_logger.metric(name, value)
        self._scenario_logger.info(
            "comparison_recorded",
            gated_metric="stochastic_density_mse",
            stochastic_density_mse=comparison.metrics["stochastic_density_mse"],
        )

    def _write_artifacts(
        self,
        comparison: MultiSeedStochasticComparison,
        export_csv: Any,
        export_plot: Any,
    ) -> None:
        """Write and register the CSV/PNG artifacts."""
        assert self._scenario_logger is not None
        base_str = str(Path(self.config.output_dir) / self.config.artifact_basename)
        csv_path = export_csv(comparison, Path(f"{base_str}.csv"))
        self.record_artifact("csv", str(csv_path))
        png_path = export_plot(comparison, Path(f"{base_str}.png"))
        if png_path is not None:
            self.record_artifact("png", str(png_path))
        else:
            self._scenario_logger.warning("artifact_png_skipped", reason="matplotlib unavailable")
