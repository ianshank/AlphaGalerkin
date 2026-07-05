"""L-shaped Poisson AMR comparison scenario: MCTS refinement vs Dörfler.

The thesis-critical PoC. On the standard L-shaped Poisson benchmark it runs an
MCTS refinement policy and classical Dörfler bulk marking through the **same**
masked finite-difference solver, residual error estimator, geometry and
active-DOF accounting (``src.research.lshape_amr_compare``), then reports two
honest comparisons:

* ``l2_error_ratio_at_matched_dof`` — the **primary, falsifiable gate**: is the
  MCTS refinement *policy* better than Dörfler at matched DOF? (``< 1`` passes.)
* ``error_per_dof_ratio_mcts_over_dorfler`` — recorded as a transparent
  secondary metric (end-to-end matched wall-clock; expected ``> 1`` for an
  untrained MCTS because search costs ``n_simulations`` real solves).

Unlike ``noyron_basis`` there is no arm gating — the ``dorfler`` and ``mcts``
arms are always available on CPU. See ``specs/lshape_amr_compare.spec.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.device import resolve_device
from src.poc.logging import ScenarioLogger
from src.poc.registry import BaseScenario, scenario
from src.poc.scenarios.lshape_amr_compare_config import (
    SCENARIO_NAME,
    LShapeAMRCompareConfig,
)

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator
    from src.research.lshape_amr_compare import (
        ComparisonParams,
        ComparisonResult,
        MultiSeedComparison,
    )


@scenario(SCENARIO_NAME)
class LShapeAMRCompareScenario(BaseScenario):
    """MCTS refinement vs Dörfler marking on the L-shaped Poisson benchmark."""

    config_class = LShapeAMRCompareConfig

    def __init__(self, config: LShapeAMRCompareConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.config: LShapeAMRCompareConfig  # type narrowing
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
            scale=self.config.scale,
            max_dof=self.config.max_dof,
            n_simulations=self.config.n_simulations,
            seed=self.config.seed,
        )

    def teardown(self) -> None:
        """Release GPU memory (no-op on CPU)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def execute(self) -> ScenarioResult:
        """Sweep seeds, record the median headline, and write the artifacts."""
        assert self._scenario_logger is not None
        # Imported lazily so the config module (and load_config_from_dict) stays
        # importable without scipy / the MCTS engine present.
        from src.research.lshape_amr_compare import (
            export_csv,
            export_plot,
            run_multiseed_comparison,
        )

        operator = self._build_operator()
        game_config = self._build_game_config(operator)
        params = self._build_params()

        multiseed = run_multiseed_comparison(operator, game_config, params)
        self._record_metrics(multiseed)
        # The committed artifact uses the median (representative) seed's run.
        self._write_artifacts(multiseed.representative, export_csv, export_plot)

        return self._create_result(status=ScenarioStatus.RUNNING)

    # ------------------------------------------------------------------ #
    # Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def _build_operator(self) -> PDEOperator:
        """Build the L-shaped Poisson operator at the configured scale."""
        from src.pde.config import PDEConfig, PDEType
        from src.pde.geometry import GeometryConfig, GeometryType
        from src.pde.operators import LShapedPoissonOperator

        pde_config = PDEConfig(
            name="lshape_amr_compare",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-self.config.scale, -self.config.scale],
            domain_max=[self.config.scale, self.config.scale],
            advection_coeff=[0.0, 0.0],
            geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=self.config.scale),
        )
        return LShapedPoissonOperator(pde_config)

    def _build_game_config(self, operator: PDEOperator) -> Any:
        """Build the PDEGameConfig backing the MCTS arm's game."""
        from src.pde.config import PDEConfig, PDEGameConfig, PDEType
        from src.pde.geometry import GeometryConfig, GeometryType

        pde_config = PDEConfig(
            name="lshape_amr_compare",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-self.config.scale, -self.config.scale],
            domain_max=[self.config.scale, self.config.scale],
            advection_coeff=[0.0, 0.0],
            geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=self.config.scale),
        )
        return PDEGameConfig(
            name="lshape_amr_game",
            pde_config=pde_config,
            game_mode="mesh_refinement",
            max_dof=self.config.max_dof,
            max_steps=self.config.max_steps,
            error_tolerance=self.config.error_tolerance,
        )

    def _build_params(self) -> ComparisonParams:
        """Assemble the harness params from the validated config."""
        from src.research.lshape_amr_compare import ComparisonParams

        return ComparisonParams(
            seed=self.config.seed,
            scale=self.config.scale,
            initial_side=self.config.initial_side,
            max_dof=self.config.max_dof,
            max_steps=self.config.max_steps,
            marking_fraction=self.config.marking_fraction,
            max_refinements=self.config.max_refinements,
            error_tolerance=self.config.error_tolerance,
            n_candidate_elements=self.config.n_candidate_elements,
            n_simulations=self.config.n_simulations,
            value_scale=self.config.value_scale,
            c_puct=self.config.c_puct,
            add_noise=self.config.add_noise,
            n_seeds=self.config.n_seeds,
        )

    # ------------------------------------------------------------------ #
    # Recording                                                           #
    # ------------------------------------------------------------------ #

    def _record_metrics(self, multiseed: MultiSeedComparison) -> None:
        """Record the median headline + per-seed spread metrics."""
        assert self._scenario_logger is not None
        metrics = multiseed.metrics()
        for name, value in metrics.items():
            self.record_metric(name, value)
            self._scenario_logger.metric(name, value)
        self._scenario_logger.info(
            "comparison_recorded",
            l2_error_ratio_at_matched_dof=metrics["l2_error_ratio_at_matched_dof"],
            error_per_dof_ratio_mcts_over_dorfler=metrics["error_per_dof_ratio_mcts_over_dorfler"],
            mcts_win_fraction=metrics["mcts_win_fraction"],
            n_seeds=metrics["n_seeds"],
        )

    def _write_artifacts(
        self,
        result: ComparisonResult,
        export_csv: Any,
        export_plot: Any,
    ) -> None:
        """Write and register the committed CSV/PNG artifacts."""
        assert self._scenario_logger is not None
        base = Path(self.config.output_dir) / self.config.artifact_basename
        csv_path = export_csv(result, base.with_suffix(".csv"))
        self.record_artifact("csv", str(csv_path))
        png_path = export_plot(result, base.with_suffix(".png"))
        if png_path is not None:
            self.record_artifact("png", str(png_path))
        else:
            self._scenario_logger.warning("artifact_png_skipped", reason="matplotlib unavailable")
