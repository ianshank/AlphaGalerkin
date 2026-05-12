"""LLM-prior MCTS basis-selection ablation scenario.

Compares three MCTS evaluators on an in-distribution PDE (Poisson) and an
out-of-distribution PDE (Burgers):

    * RandomEvaluator — uniform-prior baseline.
    * FNetEvaluator — domain-trained neural evaluator (skipped when no
      checkpoint is configured).
    * LMStudioEvaluator — generalist LLM (Qwen-14B via LM Studio) acting
      as a policy prior with no PDE-specific training.

Headline metrics (see :class:`LLMPriorAblationConfig.get_default_thresholds`):

    * ``id_rollout_reduction_pct`` — median-rollout reduction of LLM vs
      random on the ID PDE.
    * ``ood_llm_residual`` — median final residual for the LLM arm on
      the OOD PDE.
    * ``ood_trained_residual`` — median final residual for the trained
      arm on the OOD PDE (the failure threshold that proves the
      generalisation gap).
    * ``llm_call_p95_latency_ms`` — p95 LLM-call latency across all LLM
      calls in the scenario.

The scenario is **GPU-only**. ``setup()`` calls
``src.poc.device.resolve_device(config.device, context=...)`` which raises
``RuntimeError`` if CUDA is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy import stats

from src.integrations.lm_studio.client import LMStudioClient
from src.integrations.lm_studio.evaluator import LMStudioEvaluator
from src.integrations.lm_studio.preflight import check_lm_studio_server
from src.mcts.evaluator import RandomEvaluator
from src.mcts.search import MCTS
from src.pde.config import (
    BasisSelectionConfig,
    PDEConfig,
    PDEGameConfig,
    PDEType,
)
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.mcts_adapter import PDEGameAdapter
from src.pde.registry import get_pde_operator
from src.poc.config import (
    ScenarioResult,
    ScenarioStatus,
)
from src.poc.device import resolve_device
from src.poc.logging import ScenarioLogger
from src.poc.registry import BaseScenario, scenario
from src.poc.scenarios.llm_prior_config import (
    SCENARIO_NAME,
    LLMPriorAblationConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.mcts.evaluator import Evaluator
    from src.pde.operators import PDEOperator


_PDE_TYPE_MAP: dict[str, PDEType] = {
    "poisson": PDEType.POISSON,
    "burgers": PDEType.BURGERS,
    "heat": PDEType.HEAT,
    "advection_diffusion": PDEType.ADVECTION_DIFFUSION,
    "navier_stokes": PDEType.NAVIER_STOKES,
    "poisson_lshaped": PDEType.POISSON,
}
"""Mapping from PDE registry name to PDEType enum value."""

_NON_CONVERGED_RESIDUAL = float("nan")
"""Placeholder for cells where MCTS failed to even start (e.g. terminal game)."""

_LATENCY_P95 = 95.0
"""Percentile used for the headline latency threshold."""


def _median(samples: list[float]) -> float:
    """Median that returns NaN for empty input (instead of raising)."""
    if not samples:
        return float("nan")
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _percentile(samples: list[float], q: float) -> float:
    """Percentile that returns NaN for empty input (instead of raising)."""
    if not samples:
        return float("nan")
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q))


@scenario(SCENARIO_NAME)
class LLMPriorAblationScenario(BaseScenario):
    """Ablation comparing random / trained / LLM evaluators on ID + OOD PDEs."""

    config_class = LLMPriorAblationConfig

    def __init__(
        self,
        config: LLMPriorAblationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the scenario."""
        super().__init__(config, **kwargs)
        self.config: LLMPriorAblationConfig  # type narrowing
        self._device: torch.device | None = None
        self._scenario_logger: ScenarioLogger | None = None
        self._lm_client: LMStudioClient | None = None
        self._trained_model: Any | None = None
        self._llm_arm_enabled: bool = self.config.run_llm_arm
        self._trained_arm_enabled: bool = self.config.run_trained_arm
        self._random_arm_enabled: bool = self.config.run_random_arm
        self._llm_latencies_ms: list[float] = []

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        """Resolve device, gate arms, install thresholds, prepare logger."""
        self._device = resolve_device(self.config.device, context=SCENARIO_NAME)
        self._scenario_logger = ScenarioLogger(
            scenario_name=self.name,
            run_id=self.config.compute_hash(),
            device=str(self._device),
        )
        # Threshold wiring: BaseScenario._evaluate_thresholds reads
        # self.config.thresholds verbatim; it never auto-calls
        # get_default_thresholds(). We install defaults here and drop
        # arm-specific thresholds when an arm is gated off so absent
        # metrics don't FAIL the run.
        if not self.config.thresholds:
            self.config.thresholds = self.config.get_default_thresholds()

        self._gate_llm_arm()
        self._gate_trained_arm()

        self._scenario_logger.info(
            "setup_complete",
            random_arm_enabled=self._random_arm_enabled,
            trained_arm_enabled=self._trained_arm_enabled,
            llm_arm_enabled=self._llm_arm_enabled,
            id_pde=self.config.id_pde,
            ood_pde=self.config.ood_pde,
            n_seeds=self.config.n_seeds,
        )

    def teardown(self) -> None:
        """Release GPU memory and close the LM Studio client."""
        if self._lm_client is not None:
            self._lm_client.close()
            self._lm_client = None
        self._trained_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def execute(self) -> ScenarioResult:
        """Run the full ablation grid and return a ``ScenarioResult``."""
        assert self._scenario_logger is not None

        active_arms = self._active_arms()
        if not active_arms:
            return self._create_result(status=ScenarioStatus.SKIPPED)

        seeds = self.config.resolved_seeds()
        rollouts: dict[tuple[str, str], list[int]] = {}
        residuals: dict[tuple[str, str], list[float]] = {}

        for pde_name in (self.config.id_pde, self.config.ood_pde):
            operator = self._build_pde_operator(pde_name)
            basis_descriptions = self._enumerate_basis_descriptions(pde_name)
            for arm in active_arms:
                cell_rollouts: list[int] = []
                cell_residuals: list[float] = []
                for seed in seeds:
                    cell_logger = self._scenario_logger.bind(arm=arm, pde=pde_name, seed=seed)
                    rollout_count, final_residual = self._run_cell(
                        arm=arm,
                        pde_name=pde_name,
                        operator=operator,
                        basis_descriptions=basis_descriptions,
                        seed=seed,
                        cell_logger=cell_logger,
                    )
                    cell_rollouts.append(rollout_count)
                    cell_residuals.append(final_residual)
                    cell_logger.metric(
                        "cell_rollouts",
                        float(rollout_count),
                        arm=arm,
                        pde=pde_name,
                        seed=seed,
                    )
                    cell_logger.metric(
                        "cell_final_residual",
                        final_residual,
                        arm=arm,
                        pde=pde_name,
                        seed=seed,
                    )
                rollouts[(arm, pde_name)] = cell_rollouts
                residuals[(arm, pde_name)] = cell_residuals

        self._record_aggregates(rollouts, residuals)
        self._maybe_render_report(rollouts, residuals)
        return self._create_result(status=ScenarioStatus.RUNNING)

    # ------------------------------------------------------------------ #
    # Arm gating                                                          #
    # ------------------------------------------------------------------ #

    def _gate_llm_arm(self) -> None:
        """Preflight the LM Studio server and disable the arm on failure."""
        assert self._scenario_logger is not None
        if not self._llm_arm_enabled:
            self._drop_llm_thresholds()
            return
        if not self.config.lm_studio.enabled:
            self._scenario_logger.warning(
                "llm_arm_disabled_by_config",
                reason="lm_studio.enabled is False",
            )
            self._llm_arm_enabled = False
            self._drop_llm_thresholds()
            return
        try:
            report = check_lm_studio_server(self.config.lm_studio)
        except Exception as exc:
            self._scenario_logger.warning(
                "llm_preflight_raised",
                error=str(exc),
            )
            self._llm_arm_enabled = False
            self._drop_llm_thresholds()
            return
        if not report.passed:
            self._scenario_logger.warning(
                "llm_preflight_failed",
                failure_reason=report.failure_reason,
                available_models=report.available_models,
            )
            self._llm_arm_enabled = False
            self._drop_llm_thresholds()
            return
        # Preflight passed once; the client can skip its own preflight.
        client_config = self.config.lm_studio.model_copy(update={"preflight_on_construct": False})
        try:
            self._lm_client = LMStudioClient(
                client_config,
                scenario_logger=self._scenario_logger,
            )
        except Exception as exc:
            self._scenario_logger.warning(
                "llm_client_construction_failed",
                error=str(exc),
            )
            self._llm_arm_enabled = False
            self._drop_llm_thresholds()

    def _gate_trained_arm(self) -> None:
        """Lazy-load the trained checkpoint or disable the arm."""
        assert self._scenario_logger is not None
        if not self._trained_arm_enabled:
            self._drop_trained_thresholds()
            return
        checkpoint = self.config.trained_checkpoint_path
        if checkpoint is None:
            self._scenario_logger.warning(
                "trained_arm_skipped_no_checkpoint",
                reason="trained_checkpoint_path is None",
            )
            self._trained_arm_enabled = False
            self._drop_trained_thresholds()
            return
        try:
            self._trained_model = self._load_trained_model(checkpoint)
        except (FileNotFoundError, RuntimeError) as exc:
            self._scenario_logger.warning(
                "trained_arm_load_failed",
                checkpoint=str(checkpoint),
                error=str(exc),
            )
            self._trained_arm_enabled = False
            self._drop_trained_thresholds()

    def _load_trained_model(self, checkpoint: Path) -> Any:
        # Local import: src.training.checkpoint pulls heavy modeling code
        # that is irrelevant when the trained arm is skipped.
        from src.training.checkpoint import create_model_from_checkpoint

        assert self._device is not None
        # Peer-review fix: create_model_from_checkpoint returns
        # (model, config_dict). Mirror src/alphagalerkin/solver.py:593-597.
        model, _saved_config = create_model_from_checkpoint(
            checkpoint,
            device=str(self._device),
            strict=False,
        )
        return model

    def _drop_llm_thresholds(self) -> None:
        """Remove LLM-bound thresholds — BaseScenario can't represent SKIPPED."""
        llm_keys = {
            "id_rollout_reduction_pct",
            "ood_llm_residual",
            "llm_call_p95_latency_ms",
        }
        self.config.thresholds = [t for t in self.config.thresholds if t.name not in llm_keys]

    def _drop_trained_thresholds(self) -> None:
        trained_keys = {"ood_trained_residual"}
        self.config.thresholds = [t for t in self.config.thresholds if t.name not in trained_keys]

    # ------------------------------------------------------------------ #
    # Per-cell run                                                        #
    # ------------------------------------------------------------------ #

    def _run_cell(
        self,
        *,
        arm: str,
        pde_name: str,
        operator: PDEOperator,
        basis_descriptions: list[str],
        seed: int,
        cell_logger: ScenarioLogger,
    ) -> tuple[int, float]:
        """Run one (arm, pde, seed) cell — return (rollouts_used, final_residual)."""
        # Per-cell seeding; MCTS.__init__ has no seed kwarg so we set the
        # global RNGs before constructing the search.
        np.random.seed(seed)
        torch.manual_seed(seed)

        game = self._build_game(pde_name, operator)
        adapter = PDEGameAdapter(game)
        evaluator = self._build_evaluator(
            arm=arm,
            pde_name=pde_name,
            game=game,
            basis_descriptions=basis_descriptions,
            seed=seed,
            cell_logger=cell_logger,
        )

        target = self.config.target_residual
        max_rollouts = self.config.max_rollouts
        sims_per_step = self.config.n_mcts_simulations
        rollouts_used = 0

        if adapter.current_error <= target:
            return rollouts_used, float(adapter.current_error)

        mcts = MCTS(evaluator=evaluator, n_simulations=sims_per_step)

        while (
            not adapter.is_terminal()
            and adapter.current_error > target
            and rollouts_used + sims_per_step <= max_rollouts
        ):
            action = mcts.get_action(adapter, temperature=0.0, add_noise=False)
            if action < 0:
                break
            adapter.apply_action(action)
            rollouts_used += sims_per_step
            # Reuse the subtree rooted at the chosen action so we don't
            # discard search work between macro-steps. `_get_or_create_root`
            # never resets on game-state divergence, so the explicit
            # `advance` is what keeps the tree aligned with the adapter.
            mcts.advance(action)

        final_residual = float(adapter.current_error)
        if arm == "llm" and isinstance(evaluator, LMStudioEvaluator):
            self._llm_latencies_ms.extend(evaluator.latencies_ms)
        cell_logger.info(
            "cell_complete",
            rollouts_used=rollouts_used,
            final_residual=final_residual,
        )
        return rollouts_used, final_residual

    # ------------------------------------------------------------------ #
    # Construction helpers                                                #
    # ------------------------------------------------------------------ #

    def _build_pde_operator(self, pde_name: str) -> PDEOperator:
        if pde_name not in _PDE_TYPE_MAP:
            raise ValueError(
                f"PDE {pde_name!r} has no PDEType mapping; known: {sorted(_PDE_TYPE_MAP)}"
            )
        pde_type = _PDE_TYPE_MAP[pde_name]
        pde_config = PDEConfig(name=pde_name, pde_type=pde_type)
        operator_cls = get_pde_operator(pde_name)
        return operator_cls(pde_config)

    def _build_game(self, pde_name: str, operator: PDEOperator) -> BasisSelectionGame:
        pde_config = operator.config
        basis_config = BasisSelectionConfig(
            name=f"{pde_name}_basis",
            max_basis_functions=self.config.max_basis_functions,
            n_candidate_bases=self.config.n_candidate_bases,
        )
        game_config = PDEGameConfig(
            name=f"{pde_name}_game",
            pde_config=pde_config,
            game_mode="basis_selection",
            basis_config=basis_config,
            error_tolerance=self.config.target_residual,
        )
        return BasisSelectionGame(operator, game_config)

    def _enumerate_basis_descriptions(self, pde_name: str) -> list[str]:
        """Build a one-shot game just to read its basis library descriptions.

        The library is deterministic given config, so this is cheap and
        avoids leaking the inner game past the scenario boundary.
        """
        operator = self._build_pde_operator(pde_name)
        game = self._build_game(pde_name, operator)
        return [game.action_to_string(i) for i in range(game.action_space_size)]

    def _build_evaluator(
        self,
        *,
        arm: str,
        pde_name: str,
        game: BasisSelectionGame,
        basis_descriptions: list[str],
        seed: int,
        cell_logger: ScenarioLogger,
    ) -> Evaluator:
        if arm == "random":
            return RandomEvaluator(n_actions=game.action_space_size)
        if arm == "trained":
            return self._build_trained_evaluator()
        if arm == "llm":
            return self._build_llm_evaluator(
                game=game,
                pde_name=pde_name,
                basis_descriptions=basis_descriptions,
                seed=seed,
                cell_logger=cell_logger,
            )
        raise ValueError(f"unknown arm {arm!r}")

    def _build_trained_evaluator(self) -> Evaluator:
        # Local import keeps the FNet/AlphaGalerkinModel surface out of
        # the cold path when the trained arm is gated off.
        from src.mcts.evaluator import FNetEvaluator

        if self._trained_model is None:
            raise RuntimeError(
                "trained arm requested but model not loaded; "
                "trained-arm gating should have caught this"
            )
        assert self._device is not None
        return FNetEvaluator(model=self._trained_model, device=self._device)

    def _build_llm_evaluator(
        self,
        *,
        game: BasisSelectionGame,
        pde_name: str,
        basis_descriptions: list[str],
        seed: int,
        cell_logger: ScenarioLogger,
    ) -> Evaluator:
        if self._lm_client is None:
            raise RuntimeError(
                "LLM arm requested but client not built; LLM-arm gating should have caught this"
            )
        return LMStudioEvaluator(
            self._lm_client,
            action_space_size=game.action_space_size,
            pde_family=pde_name,
            basis_descriptions=basis_descriptions,
            seed=seed,
            scenario_logger=cell_logger,
        )

    def _active_arms(self) -> list[str]:
        arms: list[str] = []
        if self._random_arm_enabled:
            arms.append("random")
        if self._trained_arm_enabled:
            arms.append("trained")
        if self._llm_arm_enabled:
            arms.append("llm")
        return arms

    # ------------------------------------------------------------------ #
    # Aggregation                                                         #
    # ------------------------------------------------------------------ #

    def _record_aggregates(
        self,
        rollouts: dict[tuple[str, str], list[int]],
        residuals: dict[tuple[str, str], list[float]],
    ) -> None:
        """Record headline metrics derived from per-cell raw data."""
        assert self._scenario_logger is not None
        id_pde = self.config.id_pde
        ood_pde = self.config.ood_pde

        # ID rollout-reduction
        if self._llm_arm_enabled and self._random_arm_enabled:
            random_rollouts = rollouts.get(("random", id_pde), [])
            llm_rollouts = rollouts.get(("llm", id_pde), [])
            self._record_id_metrics(random_rollouts, llm_rollouts)

        # OOD LLM residual
        if self._llm_arm_enabled:
            llm_ood = residuals.get(("llm", ood_pde), [])
            llm_med = _median(llm_ood)
            self.record_metric("ood_llm_residual", llm_med)
            self._scenario_logger.metric("ood_llm_residual", llm_med, pde=ood_pde)

        # OOD trained residual
        if self._trained_arm_enabled:
            trained_ood = residuals.get(("trained", ood_pde), [])
            trained_med = _median(trained_ood)
            self.record_metric("ood_trained_residual", trained_med)
            self._scenario_logger.metric("ood_trained_residual", trained_med, pde=ood_pde)

        # LLM latency p95
        if self._llm_arm_enabled:
            p95 = _percentile(self._llm_latencies_ms, _LATENCY_P95)
            self.record_metric("llm_call_p95_latency_ms", p95)
            self._scenario_logger.metric(
                "llm_call_p95_latency_ms",
                p95,
                n_samples=len(self._llm_latencies_ms),
            )

    def _record_id_metrics(
        self,
        random_rollouts: list[int],
        llm_rollouts: list[int],
    ) -> None:
        assert self._scenario_logger is not None
        random_med = _median([float(x) for x in random_rollouts])
        llm_med = _median([float(x) for x in llm_rollouts])
        if random_med > 0:
            reduction_pct = 100.0 * (1.0 - llm_med / random_med)
        else:
            reduction_pct = 0.0
        self.record_metric("id_rollout_reduction_pct", reduction_pct)
        self._scenario_logger.metric(
            "id_rollout_reduction_pct",
            reduction_pct,
            random_median=random_med,
            llm_median=llm_med,
        )

        # Mann-Whitney with scipy; ``alternative="less"`` tests whether
        # the LLM uses *fewer* rollouts than random.
        if len(random_rollouts) >= 2 and len(llm_rollouts) >= 2:
            result = stats.mannwhitneyu(
                np.asarray(llm_rollouts, dtype=np.float64),
                np.asarray(random_rollouts, dtype=np.float64),
                alternative="less",
            )
            self.record_metric("id_mannwhitney_p", float(result.pvalue))
            self._scenario_logger.metric(
                "id_mannwhitney_p",
                float(result.pvalue),
                statistic=float(result.statistic),
                alpha=self.config.significance_alpha,
            )

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #

    def _maybe_render_report(
        self,
        rollouts: dict[tuple[str, str], list[int]],
        residuals: dict[tuple[str, str], list[float]],
    ) -> None:
        """Generate an HTML artifact summarising the ablation."""
        try:
            from matplotlib.figure import Figure

            from src.poc.visualization.config import VisualizationConfig
            from src.poc.visualization.reports import (
                HTMLReportGenerator,
                ReportSection,
            )
        except ImportError:
            # Matplotlib is a dev dep; skip the report if unavailable.
            return

        figures: list[Figure] = []
        rollouts_fig = self._plot_rollouts(rollouts, Figure)
        if rollouts_fig is not None:
            figures.append(rollouts_fig)
        residuals_fig = self._plot_ood_residuals(residuals, Figure)
        if residuals_fig is not None:
            figures.append(residuals_fig)

        if not figures:
            return

        vis_config = VisualizationConfig(name=self.name)
        generator = HTMLReportGenerator()
        sections = [
            ReportSection(
                title="LLM-Prior MCTS Basis-Selection Ablation",
                content=(
                    f"Active arms: {', '.join(self._active_arms())}. "
                    f"ID PDE: {self.config.id_pde}; "
                    f"OOD PDE: {self.config.ood_pde}; "
                    f"seeds: {self.config.n_seeds}; "
                    f"simulations per macro-step: {self.config.n_mcts_simulations}."
                ),
                figures=figures,
            ),
        ]
        html = generator.generate_report(
            title="LLM-Prior MCTS Ablation",
            sections=sections,
            config=vis_config,
        )
        from pathlib import Path

        output_dir = Path("outputs/poc/llm_prior_ablation")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"report_{self.config.compute_hash()}.html"
        report_path.write_text(html, encoding="utf-8")
        self.record_artifact("html_report", str(report_path))

    def _plot_rollouts(
        self,
        rollouts: dict[tuple[str, str], list[int]],
        figure_cls: Any,
    ) -> Any:
        """Box plot of rollouts-to-target on the ID PDE."""
        id_pde = self.config.id_pde
        data: list[list[int]] = []
        labels: list[str] = []
        for arm in self._active_arms():
            samples = rollouts.get((arm, id_pde), [])
            if samples:
                data.append(samples)
                labels.append(arm)
        if not data:
            return None
        fig = figure_cls(figsize=(6.0, 4.0))
        ax = fig.add_subplot(111)
        # Keep `labels=` (not `tick_labels=`) for compatibility with the
        # project's matplotlib>=3.7 lower bound; `tick_labels` only landed
        # in matplotlib 3.9. The 3.9+ deprecation warning is acceptable.
        ax.boxplot(data, labels=labels)
        ax.set_ylabel("Rollouts to target residual")
        ax.set_title(f"Rollouts to target on ID PDE ({id_pde})")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig

    def _plot_ood_residuals(
        self,
        residuals: dict[tuple[str, str], list[float]],
        figure_cls: Any,
    ) -> Any:
        """Box plot of final residuals on the OOD PDE."""
        ood_pde = self.config.ood_pde
        data: list[list[float]] = []
        labels: list[str] = []
        for arm in self._active_arms():
            samples = [x for x in residuals.get((arm, ood_pde), []) if np.isfinite(x)]
            if samples:
                data.append(samples)
                labels.append(arm)
        if not data:
            return None
        fig = figure_cls(figsize=(6.0, 4.0))
        ax = fig.add_subplot(111)
        # Keep `labels=` (not `tick_labels=`) for compatibility with the
        # project's matplotlib>=3.7 lower bound; `tick_labels` only landed
        # in matplotlib 3.9. The 3.9+ deprecation warning is acceptable.
        ax.boxplot(data, labels=labels)
        ax.set_ylabel("Final residual")
        ax.set_yscale("log")
        ax.set_title(f"Final residual on OOD PDE ({ood_pde})")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig


__all__ = ["LLMPriorAblationScenario"]
