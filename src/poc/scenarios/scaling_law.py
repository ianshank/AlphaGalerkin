"""MCTS-budget scaling-law scenario.

Sweeps the MCTS-simulation budget (per-decision search compute) and fits a
log-log scaling curve of final residual against budget for each evaluator arm.
This is the concrete "bitter lesson / scaling curve" experiment: does more
search compute predictably reduce the solver residual?

The inner MCTS rollout, operator/game construction, and arm evaluator factory
are shared with the other centaur scenarios via
:mod:`src.poc.scenarios._centaur_common`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.integrations.lm_studio.client import LMStudioClient
from src.integrations.lm_studio.preflight import check_lm_studio_server
from src.integrations.lm_studio.schema import LMStudioError
from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.device import resolve_device
from src.poc.logging import ScenarioLogger
from src.poc.registry import BaseScenario, scenario
from src.poc.scenarios._centaur_common import (
    build_arm_evaluator,
    build_basis_game,
    build_pde_operator,
    enumerate_basis_descriptions,
    run_basis_selection_cell,
)
from src.poc.scenarios.scaling_law_config import (
    SCALING_SCENARIO_NAME,
    ScalingLawConfig,
)
from src.poc.statistics.significance import SignificanceTest, StatisticalAnalyzer

if TYPE_CHECKING:
    from src.pde.operators import PDEOperator

# Residuals are clamped to this floor before the log-log fit so a cell that
# happens to hit the (tiny) target residual cannot produce log(0) = -inf.
_RESIDUAL_LOG_FLOOR = 1e-12

# Primary-arm metric aliases that the headline thresholds read.
_PRIMARY_SLOPE_METRIC = "residual_scaling_exponent"
_PRIMARY_R2_METRIC = "residual_fit_r2"


def _median(samples: list[float]) -> float:
    if not samples:
        return float("nan")
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def fit_log_log(budgets: list[int], residual_medians: list[float]) -> tuple[float, float]:
    """Fit ``log(residual) = slope * log(budget) + intercept``.

    Args:
        budgets: Simulation budgets (the x-axis, all >= 1).
        residual_medians: Median residual per budget (clamped to a positive
            floor before taking the log).

    Returns:
        ``(slope, r_squared)``. A negative slope means more compute lowers
        the residual. Returns ``(0.0, 0.0)`` when fewer than two finite
        points are available or the x-variance is degenerate.

    """
    pairs = [
        (b, r)
        for b, r in zip(budgets, residual_medians, strict=True)
        if b >= 1 and np.isfinite(r)
    ]
    if len(pairs) < 2:
        return 0.0, 0.0

    log_x = np.log(np.asarray([b for b, _ in pairs], dtype=np.float64))
    residual_arr = np.asarray([r for _, r in pairs], dtype=np.float64)
    log_y = np.log(np.maximum(residual_arr, _RESIDUAL_LOG_FLOOR))

    if float(np.var(log_x)) == 0.0:
        return 0.0, 0.0

    slope, intercept = np.polyfit(log_x, log_y, 1)
    predicted = slope * log_x + intercept
    ss_res = float(np.sum((log_y - predicted) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return float(slope), float(r_squared)


@scenario(SCALING_SCENARIO_NAME)
class ScalingLawScenario(BaseScenario):
    """Fit a residual-vs-MCTS-budget scaling curve per evaluator arm."""

    config_class = ScalingLawConfig

    def __init__(self, config: ScalingLawConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.config: ScalingLawConfig  # type narrowing
        self._device: torch.device | None = None
        self._scenario_logger: ScenarioLogger | None = None
        self._lm_client: LMStudioClient | None = None
        self._trained_model: Any | None = None
        self._active_arms: list[str] = []

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        """Resolve device, gate arms, install primary-arm thresholds."""
        self._device = resolve_device(self.config.device, context=SCALING_SCENARIO_NAME)
        self._scenario_logger = ScenarioLogger(
            scenario_name=self.name,
            run_id=self.config.compute_hash(),
            device=str(self._device),
        )
        if not self.config.thresholds:
            self.config.thresholds = self.config.get_default_thresholds()

        self._active_arms = self._gate_arms()

        # The headline thresholds describe the *primary* arm's scaling fit.
        # If the primary arm gated off, drop them so absent metrics don't
        # auto-FAIL the run (BaseScenario can't represent SKIPPED).
        if self.config.primary_arm not in self._active_arms:
            self._drop_primary_thresholds()

        self._scenario_logger.info(
            "setup_complete",
            pde=self.config.pde,
            active_arms=self._active_arms,
            budgets=self.config.simulation_budgets,
            n_seeds=self.config.n_seeds,
        )

    def teardown(self) -> None:
        if self._lm_client is not None:
            self._lm_client.close()
            self._lm_client = None
        self._trained_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def execute(self) -> ScenarioResult:
        """Run the sweep, fit per-arm scaling curves, record metrics."""
        assert self._scenario_logger is not None

        if not self._active_arms:
            return self._create_result(status=ScenarioStatus.SKIPPED)

        seeds = self.config.resolved_seeds()
        budgets = self.config.simulation_budgets

        operator = build_pde_operator(self.config.pde)
        basis_descriptions = enumerate_basis_descriptions(self._build_game(operator))

        # residual_medians[arm][budget] = median final residual over seeds.
        residual_medians: dict[str, dict[int, float]] = {}
        # raw residuals at the largest budget, for the arm-vs-arm comparison.
        largest_budget = budgets[-1]
        residuals_at_largest: dict[str, list[float]] = {}

        for arm in self._active_arms:
            per_budget_median: dict[int, float] = {}
            for budget in budgets:
                cell_residuals: list[float] = []
                for seed in seeds:
                    cell_logger = self._scenario_logger.bind(arm=arm, budget=budget, seed=seed)
                    residual = self._run_cell(
                        arm=arm,
                        operator=operator,
                        basis_descriptions=basis_descriptions,
                        budget=budget,
                        seed=seed,
                        cell_logger=cell_logger,
                    )
                    cell_residuals.append(residual)
                per_budget_median[budget] = _median(cell_residuals)
                if budget == largest_budget:
                    residuals_at_largest[arm] = cell_residuals
                self._scenario_logger.metric(
                    "budget_residual_median",
                    per_budget_median[budget],
                    arm=arm,
                    budget=budget,
                )
            residual_medians[arm] = per_budget_median

        self._record_scaling_metrics(budgets, residual_medians)
        self._record_arm_comparison(residuals_at_largest)
        self._maybe_render_report(budgets, residual_medians)
        return self._create_result(status=ScenarioStatus.RUNNING)

    # ------------------------------------------------------------------ #
    # Arm gating                                                          #
    # ------------------------------------------------------------------ #

    def _gate_arms(self) -> list[str]:
        """Return the runnable arms after preflight / checkpoint gating."""
        assert self._scenario_logger is not None
        active: list[str] = []
        for arm in self.config.arms:
            if arm == "random":
                active.append("random")
            elif arm == "trained":
                if self._gate_trained_arm():
                    active.append("trained")
            elif arm == "llm":
                if self._gate_llm_arm():
                    active.append("llm")
        return active

    def _gate_trained_arm(self) -> bool:
        assert self._scenario_logger is not None
        checkpoint = self.config.trained_checkpoint_path
        if checkpoint is None:
            self._scenario_logger.warning(
                "trained_arm_skipped_no_checkpoint",
                reason="trained_checkpoint_path is None",
            )
            return False
        try:
            self._trained_model = self._load_trained_model(checkpoint)
        except (FileNotFoundError, RuntimeError) as exc:
            self._scenario_logger.warning(
                "trained_arm_load_failed",
                checkpoint=str(checkpoint),
                error=str(exc),
            )
            return False
        return True

    def _gate_llm_arm(self) -> bool:
        assert self._scenario_logger is not None
        if not self.config.lm_studio.enabled:
            self._scenario_logger.warning(
                "llm_arm_disabled_by_config",
                reason="lm_studio.enabled is False",
            )
            return False
        try:
            report = check_lm_studio_server(self.config.lm_studio)
        except (LMStudioError, OSError, RuntimeError, ValueError, AttributeError) as exc:
            self._scenario_logger.warning(
                "llm_preflight_raised",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return False
        if not report.passed:
            self._scenario_logger.warning(
                "llm_preflight_failed",
                failure_reason=report.failure_reason,
                available_models=report.available_models,
            )
            return False
        client_config = self.config.lm_studio.model_copy(update={"preflight_on_construct": False})
        try:
            self._lm_client = LMStudioClient(client_config, scenario_logger=self._scenario_logger)
        except (LMStudioError, OSError, RuntimeError, ValueError, ImportError) as exc:
            self._scenario_logger.warning(
                "llm_client_construction_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return False
        return True

    def _load_trained_model(self, checkpoint: Path) -> Any:
        from src.training.checkpoint import create_model_from_checkpoint

        assert self._device is not None
        model, _saved_config = create_model_from_checkpoint(
            checkpoint,
            device=str(self._device),
            strict=False,
        )
        return model

    def _drop_primary_thresholds(self) -> None:
        primary_keys = {_PRIMARY_SLOPE_METRIC, _PRIMARY_R2_METRIC}
        self.config.thresholds = [
            t for t in self.config.thresholds if t.name not in primary_keys
        ]

    # ------------------------------------------------------------------ #
    # Per-cell run                                                        #
    # ------------------------------------------------------------------ #

    def _build_game(self, operator: PDEOperator) -> Any:
        return build_basis_game(
            self.config.pde,
            operator,
            max_basis_functions=self.config.max_basis_functions,
            n_candidate_bases=self.config.n_candidate_bases,
            target_residual=self.config.target_residual,
        )

    def _run_cell(
        self,
        *,
        arm: str,
        operator: PDEOperator,
        basis_descriptions: list[str],
        budget: int,
        seed: int,
        cell_logger: ScenarioLogger,
    ) -> float:
        """Run one (arm, budget, seed) cell — return the final residual."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        game = self._build_game(operator)
        evaluator = build_arm_evaluator(
            arm,
            game=game,
            pde_name=self.config.pde,
            basis_descriptions=basis_descriptions,
            seed=seed,
            lm_client=self._lm_client,
            trained_model=self._trained_model,
            device=self._device,
            scenario_logger=cell_logger,
        )
        outcome = run_basis_selection_cell(
            game=game,
            evaluator=evaluator,
            target_residual=self.config.target_residual,
            max_rollouts=self.config.max_rollouts_for_budget(budget),
            n_simulations=budget,
            scenario_logger=cell_logger,
        )
        cell_logger.info(
            "cell_complete",
            budget=budget,
            rollouts_used=outcome.rollouts_used,
            final_residual=outcome.final_residual,
        )
        return outcome.final_residual

    # ------------------------------------------------------------------ #
    # Metrics                                                             #
    # ------------------------------------------------------------------ #

    def _record_scaling_metrics(
        self,
        budgets: list[int],
        residual_medians: dict[str, dict[int, float]],
    ) -> None:
        assert self._scenario_logger is not None
        for arm, per_budget in residual_medians.items():
            ordered = [per_budget[b] for b in budgets]
            for b in budgets:
                self.record_metric(f"{arm}_residual_median_b{b}", per_budget[b])
            slope, r2 = fit_log_log(budgets, ordered)
            self.record_metric(f"{arm}_residual_scaling_exponent", slope)
            self.record_metric(f"{arm}_residual_fit_r2", r2)
            self._scenario_logger.metric(
                "arm_scaling_fit",
                slope,
                arm=arm,
                r_squared=r2,
            )
            if arm == self.config.primary_arm:
                # Aliases the headline thresholds read.
                self.record_metric(_PRIMARY_SLOPE_METRIC, slope)
                self.record_metric(_PRIMARY_R2_METRIC, r2)

    def _record_arm_comparison(self, residuals_at_largest: dict[str, list[float]]) -> None:
        """Compare the two leading arms at the largest budget (no threshold)."""
        assert self._scenario_logger is not None
        if len(self._active_arms) < 2:
            return
        arm_a, arm_b = self._active_arms[0], self._active_arms[1]
        samples_a = residuals_at_largest.get(arm_a, [])
        samples_b = residuals_at_largest.get(arm_b, [])
        if len(samples_a) < 2 or len(samples_b) < 2:
            return
        analyzer = StatisticalAnalyzer()
        test = SignificanceTest(
            test_type=self.config.significance_test_type,
            alpha=self.config.significance_alpha,
            n_bootstrap=self.config.n_bootstrap,
        )
        result = analyzer.compare_runs(samples_a, samples_b, test=test)
        self.record_metric("arm_comparison_p", float(result.p_value))
        self.record_metric(
            "arm_comparison_significant",
            1.0 if result.is_significant else 0.0,
        )
        if result.effect_size is not None:
            self.record_metric("arm_comparison_effect_size", float(result.effect_size))
        self._scenario_logger.metric(
            "arm_comparison",
            float(result.p_value),
            arm_a=arm_a,
            arm_b=arm_b,
            significant=result.is_significant,
        )

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #

    def _maybe_render_report(
        self,
        budgets: list[int],
        residual_medians: dict[str, dict[int, float]],
    ) -> None:
        try:
            from matplotlib.figure import Figure

            from src.poc.visualization.config import VisualizationConfig
            from src.poc.visualization.reports import HTMLReportGenerator, ReportSection
        except ImportError:
            return

        fig = Figure(figsize=(6.0, 4.0))
        ax = fig.add_subplot(111)
        plotted = False
        for arm, per_budget in residual_medians.items():
            ys = [per_budget[b] for b in budgets]
            finite = [(b, y) for b, y in zip(budgets, ys, strict=True) if np.isfinite(y) and y > 0]
            if len(finite) < 2:
                continue
            ax.plot(
                [b for b, _ in finite],
                [y for _, y in finite],
                marker="o",
                label=arm,
            )
            plotted = True
        if not plotted:
            return
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MCTS simulation budget")
        ax.set_ylabel("Median final residual")
        ax.set_title(f"Residual scaling law ({self.config.pde})")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()

        vis_config = VisualizationConfig(name=self.name)
        generator = HTMLReportGenerator()
        sections = [
            ReportSection(
                title="MCTS-Budget Scaling Law",
                content=(
                    f"PDE: {self.config.pde}; arms: {', '.join(self._active_arms)}; "
                    f"budgets: {budgets}; seeds: {self.config.n_seeds}."
                ),
                figures=[fig],
            ),
        ]
        html = generator.generate_report(
            title="MCTS-Budget Scaling Law",
            sections=sections,
            config=vis_config,
        )
        output_dir = Path("outputs/poc/scaling_law")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"report_{self.config.compute_hash()}.html"
        report_path.write_text(html, encoding="utf-8")
        self.record_artifact("html_report", str(report_path))


__all__ = ["ScalingLawScenario", "fit_log_log"]
