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

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.device import resolve_device
from src.poc.logging import ScenarioLogger
from src.poc.registry import BaseScenario, scenario
from src.poc.scenarios.transfer_baseline_compare_config import (
    SCENARIO_NAME,
    TransferBaselineCompareConfig,
)

if TYPE_CHECKING:
    from src.research.transfer_baseline_compare import (
        MultiSeedTransferComparison,
        TransferComparisonParams,
    )


@scenario(SCENARIO_NAME)
class TransferBaselineCompareScenario(BaseScenario):
    """AlphaGalerkin operator zero-shot vs a retrained discrete CNN."""

    config_class = TransferBaselineCompareConfig

    def __init__(self, config: TransferBaselineCompareConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.config: TransferBaselineCompareConfig  # type narrowing
        self._device: torch.device | None = None
        self._scenario_logger: ScenarioLogger | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

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
            train_resolution=self.config.train_resolution,
            target_resolution=self.config.target_resolution,
            n_seeds=self.config.n_seeds,
            seed=self.config.seed,
        )

    def teardown(self) -> None:
        """Release GPU memory (no-op on CPU)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        self._write_artifacts(multiseed, export_csv, export_plot)

        return self._create_result(status=ScenarioStatus.RUNNING)

    # ------------------------------------------------------------------ #
    # Construction helpers                                                #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Recording                                                           #
    # ------------------------------------------------------------------ #

    def _record_metrics(self, multiseed: MultiSeedTransferComparison) -> None:
        """Record the median headline + per-seed spread metrics."""
        assert self._scenario_logger is not None
        metrics = multiseed.metrics()
        for name, value in metrics.items():
            self.record_metric(name, value)
            self._scenario_logger.metric(name, value)
        self._scenario_logger.info(
            "comparison_recorded",
            gated_metric=self.config.target_metric_name,
            transfer_mse_ratio=metrics[self.config.target_metric_name],
            alphagalerkin_win_fraction=metrics["alphagalerkin_win_fraction"],
            n_seeds=metrics["n_seeds"],
        )

    def _write_artifacts(
        self,
        multiseed: MultiSeedTransferComparison,
        export_csv: Any,
        export_plot: Any,
    ) -> None:
        """Write and register the committed CSV/PNG artifacts (all seeds in the CSV)."""
        assert self._scenario_logger is not None
        # Append extensions by string concatenation (not Path.with_suffix, which would
        # truncate an internal dot in a custom artifact_basename).
        base_str = str(Path(self.config.output_dir) / self.config.artifact_basename)
        csv_path = export_csv(multiseed, Path(f"{base_str}.csv"))
        self.record_artifact("csv", str(csv_path))
        png_path = export_plot(multiseed, Path(f"{base_str}.png"))
        if png_path is not None:
            self.record_artifact("png", str(png_path))
        else:
            self._scenario_logger.warning("artifact_png_skipped", reason="matplotlib unavailable")
