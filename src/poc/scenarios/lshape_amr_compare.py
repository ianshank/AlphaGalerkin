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

from typing import TYPE_CHECKING, Any

from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.registry import scenario
from src.poc.scenarios._compare_common import CompareScenarioBase
from src.poc.scenarios.lshape_amr_compare_config import (
    SCENARIO_NAME,
    LShapeAMRCompareConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.pde.operators import PDEOperator
    from src.research.lshape_amr_compare import (
        ComparisonParams,
    )


@scenario(SCENARIO_NAME)
class LShapeAMRCompareScenario(CompareScenarioBase):
    """MCTS refinement vs Dörfler marking on the L-shaped Poisson benchmark."""

    config_class = LShapeAMRCompareConfig

    def __init__(self, config: LShapeAMRCompareConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.config: LShapeAMRCompareConfig  # type narrowing

    def _setup_log_fields(self) -> dict[str, Any]:
        return {
            "scale": self.config.scale,
            "max_dof": self.config.max_dof,
            "n_simulations": self.config.n_simulations,
            "seed": self.config.seed,
        }

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
        self._write_csv_png_artifacts(multiseed.representative, export_csv, export_plot)

        return self._create_result(status=ScenarioStatus.RUNNING)

    def _build_pde_config(self) -> Any:
        """Build the shared L-shaped Poisson PDEConfig.

        Single source of truth for the geometry/domain used by *both* the
        operator and the MCTS game config, so the two arms cannot silently
        drift apart and break the apples-to-apples comparison invariant.
        """
        from src.pde.config import PDEConfig, PDEType
        from src.pde.geometry import GeometryConfig, GeometryType

        return PDEConfig(
            name="lshape_amr_compare",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-self.config.scale, -self.config.scale],
            domain_max=[self.config.scale, self.config.scale],
            advection_coeff=[0.0, 0.0],
            geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=self.config.scale),
        )

    def _build_operator(self) -> PDEOperator:
        """Build the L-shaped Poisson operator at the configured scale."""
        from src.pde.operators import LShapedPoissonOperator

        return LShapedPoissonOperator(self._build_pde_config())

    def _build_game_config(self, operator: PDEOperator) -> Any:
        """Build the PDEGameConfig backing the MCTS arm's game."""
        from src.pde.config import PDEGameConfig

        return PDEGameConfig(
            name="lshape_amr_game",
            pde_config=self._build_pde_config(),
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
            search_mode=self.config.search_mode,
            n_seeds=self.config.n_seeds,
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
            l2_error_ratio_at_matched_dof=metrics["l2_error_ratio_at_matched_dof"],
            error_per_dof_ratio_mcts_over_dorfler=metrics["error_per_dof_ratio_mcts_over_dorfler"],
            mcts_win_fraction=metrics["mcts_win_fraction"],
            n_seeds=metrics["n_seeds"],
        )
