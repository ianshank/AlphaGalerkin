"""Honest zero-shot transfer comparison scenario: operator vs retrained CNN.

Replaces the fabricated "MSE 0.000209 / 240x better than threshold" self-comparison
with a falsifiable head-to-head (``src.research.transfer_baseline_compare``): the
resolution-independent :class:`~src.experiments.physics_model.PhysicsOperator`, trained
only at ``train_resolution`` and applied zero-shot at ``target_resolution``, versus a
discrete CNN retrained at ``target_resolution``.

Reports (see ``specs/transfer_baseline_compare.spec.md``):

* ``transfer_mse_ratio_<t>x<t>`` — the **primary, falsifiable gate** (median over seeds):
  is the operator's zero-shot MSE below a retrained CNN's? (``< threshold`` passes.)
* ``transfer_mse_ratio_<t>x<t>_matched_compute`` — recorded secondary (CNN given a
  training budget matched to the operator's cost).
* ``mse_cnn_zeroshot_<t>x<t>`` — the mechanism check: a CNN trained at ``train_resolution``
  cannot transfer and must be retrained.

Like ``lshape_amr_compare`` there is no arm gating — both arms are always available on CPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.registry import scenario
from src.poc.scenarios._compare_common import CompareScenarioBase
from src.poc.scenarios.transfer_baseline_compare_config import (
    SCENARIO_NAME,
    TransferBaselineCompareConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.research.transfer_baseline_compare import (
        TransferComparisonParams,
    )


@scenario(SCENARIO_NAME)
class TransferBaselineCompareScenario(CompareScenarioBase):
    """AlphaGalerkin operator zero-shot vs a retrained discrete CNN."""

    config_class = TransferBaselineCompareConfig

    def __init__(self, config: TransferBaselineCompareConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.config: TransferBaselineCompareConfig  # type narrowing

    def _setup_log_fields(self) -> dict[str, Any]:
        return {
            "train_resolution": self.config.train_resolution,
            "target_resolution": self.config.target_resolution,
            "n_seeds": self.config.n_seeds,
            "seed": self.config.seed,
        }

    def execute(self) -> ScenarioResult:
        """Sweep seeds, record the median headline, and write the artifacts."""
        assert self._scenario_logger is not None
        assert self._device is not None
        # Imported lazily so the config module (and load_config_from_dict) stays
        # importable without torch-heavy deps eagerly loaded at registration time.
        from src.research.transfer_baseline_compare import (
            export_csv,
            export_plot,
            run_multiseed_transfer_comparison,
        )

        params = self._build_params(str(self._device))
        multiseed = run_multiseed_transfer_comparison(params)
        self._record_metrics(multiseed)
        self._write_csv_png_artifacts(multiseed, export_csv, export_plot)

        return self._create_result(status=ScenarioStatus.RUNNING)

    def _build_params(self, device: str) -> TransferComparisonParams:
        """Assemble the harness params from the validated config."""
        from src.research.transfer_baseline_compare import TransferComparisonParams

        cfg = self.config
        return TransferComparisonParams(
            seed=cfg.seed,
            device=device,
            train_resolution=cfg.train_resolution,
            target_resolution=cfg.target_resolution,
            secondary_resolutions=tuple(cfg.secondary_resolutions),
            n_train_samples=cfg.n_train_samples,
            n_eval_samples=cfg.n_eval_samples,
            n_charges=cfg.n_charges,
            charge_std=cfg.charge_std,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            learning_rate=cfg.learning_rate,
            eval_seed_base=cfg.eval_seed_base,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            n_fourier_features=cfg.n_fourier_features,
            fourier_scale=cfg.fourier_scale,
            use_fnet=cfg.use_fnet,
            dropout=cfg.dropout,
            cnn_n_layers=cfg.cnn_n_layers,
            cnn_kernel_size=cfg.cnn_kernel_size,
            cnn_channels=cfg.cnn_channels,
            cnn_use_batchnorm=cfg.cnn_use_batchnorm,
            cnn_dropout=cfg.cnn_dropout,
            cnn_param_match_tolerance=cfg.cnn_param_match_tolerance,
            matched_budget_mode=cfg.matched_budget_mode,
            n_seeds=cfg.n_seeds,
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
            gated_metric=self.config.target_metric_name,
            transfer_mse_ratio=metrics[self.config.target_metric_name],
            alphagalerkin_win_fraction=metrics["alphagalerkin_win_fraction"],
            n_seeds=metrics["n_seeds"],
        )
