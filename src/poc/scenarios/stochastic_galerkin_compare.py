"""Deterministic-vs-stochastic Galerkin comparison scenario (NKE layer, AC8).

Wraps ``src.research.stochastic_galerkin_compare`` in the PoC lifecycle: both
arms score on the shared 2D Fokker-Planck/OU benchmark; only the stochastic
arm's absolute density MSE is gated (see the config module's honesty rule).
The two solver paths stay separate — this scenario shares only benchmark data
structures between them (change-doc constraint).

Spec: specs/stochastic_galerkin_nke.spec.md (AC8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.registry import scenario
from src.poc.scenarios._compare_common import CompareScenarioBase
from src.poc.scenarios.stochastic_galerkin_compare_config import (
    SCENARIO_NAME,
    StochasticGalerkinCompareConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.research.stochastic_galerkin_compare import (
        StochasticCompareParams,
    )


@scenario(SCENARIO_NAME)
class StochasticGalerkinCompareScenario(CompareScenarioBase):
    """Galerkin attention vs stochastic Galerkin projection on shared FP/OU data."""

    config_class = StochasticGalerkinCompareConfig

    def __init__(
        self, config: StochasticGalerkinCompareConfig | None = None, **kwargs: Any
    ) -> None:
        super().__init__(config, **kwargs)
        self.config: StochasticGalerkinCompareConfig  # type narrowing

    def _setup_log_fields(self) -> dict[str, Any]:
        return {
            "grid_n": self.config.grid_n,
            "n_seeds": self.config.n_seeds,
            "seed": self.config.seed,
        }

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
        self._write_csv_png_artifacts(comparison, export_csv, export_plot)
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

    def _log_metrics_recorded(
        self,
        comparison: object,
        metrics: Mapping[str, float],
    ) -> None:
        del comparison
        assert self._scenario_logger is not None
        self._scenario_logger.info(
            "comparison_recorded",
            gated_metric="stochastic_density_mse",
            stochastic_density_mse=metrics["stochastic_density_mse"],
        )
