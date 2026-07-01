"""Noyron basis-selection scenario (Leap 71 v2.2).

Drives MCTS-guided Galerkin basis selection on an SDF-defined helical operator
and records the **first documented MCTS-on-Noyron result**: how much the search
reduces the Galerkin error estimate on a real Leap 71 geometry.

Design (see ``specs/noyron_basis.spec.md``):

* The helical operators carry their geometry on ``PDEConfig.geometry`` and are
  **not** in ``PDE_TYPE_MAP``, so the scenario builds its operator/game through
  the ``pde_basis_helical`` construction path (``_create_helical_pde_config`` +
  :func:`build_basis_game`) rather than ``_centaur_common.build_pde_operator``.
* It reuses only the geometry-agnostic primitives — :func:`build_arm_evaluator`
  and :func:`run_basis_selection_cell` — mirroring ``llm_prior_ablation``.
* Arm gating is graceful: a missing checkpoint / failing LLM preflight disables
  the arm **and** removes its thresholds so absent metrics do not auto-FAIL.

Headline metrics (see :meth:`NoyronBasisConfig.get_default_thresholds`):

* ``error_reduction_pct`` — primary-arm median ``100*(init-final)/init``.
* ``final_residual`` — primary-arm median final residual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.register_games import _create_helical_pde_config
from src.pde.registry import PDEOperatorRegistry
from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.device import resolve_device
from src.poc.logging import ScenarioLogger
from src.poc.registry import BaseScenario, scenario
from src.poc.scenarios._centaur_common import (
    build_arm_evaluator,
    build_basis_game,
    enumerate_basis_descriptions,
    median_of,
    run_basis_selection_cell,
)
from src.poc.scenarios.noyron_basis_config import SCENARIO_NAME, NoyronBasisConfig

if TYPE_CHECKING:
    from src.integrations.lm_studio.client import LMStudioClient
    from src.mcts.evaluator import Evaluator
    from src.modeling.model import AlphaGalerkinModel
    from src.pde.operators import PDEOperator

_ERROR_REDUCTION_KEY = "error_reduction_pct"
_FINAL_RESIDUAL_KEY = "final_residual"
_REDUCTION_FLOOR = 1e-12
"""Denominator floor guarding the reduction ratio against a zero initial error."""
_EXTENT_FLOOR = 1e-9
"""Per-axis extent floor guarding the manufactured normalisation against zero-width axes."""


def make_manufactured_operator(operator: PDEOperator, wavenumber: int) -> PDEOperator:
    """Return a copy of ``operator`` with a manufactured product-of-sines target.

    The helical operators are homogeneous (zero source, no steady exact
    solution), so a basis-selection game built on them starts at zero error and
    is degenerate. This wraps the operator so ``exact_solution`` returns a
    smooth ``∏_d sin(k·π·x_norm_d)`` field over the operator's bounding box —
    a valid Galerkin target that vanishes on the box boundary — giving the game
    a real field to approximate. The wrapper subclasses the concrete operator so
    ``source_term`` / ``residual`` / collocation sampling are unchanged.

    Reusable for any operator, not just helical ones.

    Args:
        operator: The operator to augment (its ``config`` is reused verbatim).
        wavenumber: Integer wavenumber ``k`` of the manufactured target.

    Returns:
        A new operator instance whose ``exact_solution`` is the manufactured field.

    """
    domain_min = np.asarray(operator.config.domain_min, dtype=np.float64)
    domain_max = np.asarray(operator.config.domain_max, dtype=np.float64)
    extent = np.maximum(domain_max - domain_min, _EXTENT_FLOOR)
    n_dims = extent.shape[0]
    base_cls = type(operator)

    def _manufactured_field(coords: NDArray[np.float64]) -> NDArray[np.float32]:
        normalized = (coords[:, :n_dims] - domain_min) / extent
        field = np.prod(np.sin(wavenumber * np.pi * normalized), axis=-1)
        return field.astype(np.float32)

    class _ManufacturedOperator(base_cls):  # type: ignore[valid-type,misc]
        """Concrete operator with a manufactured exact solution."""

        def exact_solution(
            self,
            coords: NDArray[np.float32] | torch.Tensor,
            time: float | None = None,
        ) -> NDArray[np.float32] | torch.Tensor:
            if isinstance(coords, torch.Tensor):
                arr = coords.detach().cpu().numpy().astype(np.float64)
                field = _manufactured_field(arr)
                return torch.from_numpy(field).to(coords.device)
            return _manufactured_field(np.asarray(coords, dtype=np.float64))

    return _ManufacturedOperator(operator.config)


@scenario(SCENARIO_NAME)
class NoyronBasisScenario(BaseScenario):
    """MCTS basis selection on a Leap 71 helical operator."""

    config_class = NoyronBasisConfig

    def __init__(self, config: NoyronBasisConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.config: NoyronBasisConfig  # type narrowing
        self._device: torch.device | None = None
        self._scenario_logger: ScenarioLogger | None = None
        self._lm_client: LMStudioClient | None = None
        self._trained_model: AlphaGalerkinModel | None = None
        self._arm_enabled: dict[str, bool] = dict.fromkeys(self.config.arms, True)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        """Resolve device, gate arms, and install thresholds."""
        self._device = resolve_device(self.config.device, context=SCENARIO_NAME)
        self._scenario_logger = ScenarioLogger(
            scenario_name=self.name,
            run_id=self.config.compute_hash(),
            device=str(self._device),
        )
        # BaseScenario._evaluate_thresholds reads self.config.thresholds
        # verbatim; install defaults, then drop thresholds for disabled arms.
        if not self.config.thresholds:
            self.config.thresholds = self.config.get_default_thresholds()

        self._gate_trained_arm()
        self._gate_llm_arm()
        self._maybe_drop_primary_thresholds()

        self._scenario_logger.info(
            "setup_complete",
            operator_name=self.config.operator_name,
            active_arms=self._active_arms(),
            primary_arm=self.config.primary_arm,
            n_seeds=self.config.n_seeds,
        )

    def teardown(self) -> None:
        """Release the LLM client / GPU memory."""
        if self._lm_client is not None:
            self._lm_client.close()
            self._lm_client = None
        self._trained_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def execute(self) -> ScenarioResult:
        """Run every (arm, seed) cell and record the primary-arm aggregates."""
        assert self._scenario_logger is not None
        active_arms = self._active_arms()
        if not active_arms:
            self._scenario_logger.warning("no_active_arms", reason="all arms gated off")
            return self._create_result(status=ScenarioStatus.SKIPPED)

        operator = self._build_operator()
        basis_descriptions = enumerate_basis_descriptions(self._build_game(operator))
        seeds = self.config.resolved_seeds()

        reductions: dict[str, list[float]] = {}
        finals: dict[str, list[float]] = {}
        for arm in active_arms:
            arm_reductions: list[float] = []
            arm_finals: list[float] = []
            for seed in seeds:
                cell_logger = self._scenario_logger.bind(arm=arm, seed=seed)
                reduction_pct, final_residual = self._run_cell(
                    arm=arm,
                    operator=operator,
                    basis_descriptions=basis_descriptions,
                    seed=seed,
                    cell_logger=cell_logger,
                )
                arm_reductions.append(reduction_pct)
                arm_finals.append(final_residual)
            reductions[arm] = arm_reductions
            finals[arm] = arm_finals
            self._scenario_logger.metric(
                "arm_median_reduction_pct", median_of(arm_reductions), arm=arm
            )
            self._scenario_logger.metric(
                "arm_median_final_residual", median_of(arm_finals), arm=arm
            )

        self._record_primary_aggregates(reductions, finals)
        return self._create_result(status=ScenarioStatus.RUNNING)

    # ------------------------------------------------------------------ #
    # Arm gating                                                          #
    # ------------------------------------------------------------------ #

    def _gate_trained_arm(self) -> None:
        if "trained" not in self._arm_enabled:
            return
        assert self._scenario_logger is not None
        checkpoint = self.config.trained_checkpoint_path
        if checkpoint is None:
            self._scenario_logger.warning(
                "trained_arm_skipped_no_checkpoint",
                reason="trained_checkpoint_path is None",
            )
            self._disable_arm("trained")
            return
        try:
            self._trained_model = self._load_trained_model(checkpoint)
        except (FileNotFoundError, RuntimeError) as exc:
            self._scenario_logger.warning(
                "trained_arm_load_failed", checkpoint=str(checkpoint), error=str(exc)
            )
            self._disable_arm("trained")

    def _gate_llm_arm(self) -> None:
        if "llm" not in self._arm_enabled:
            return
        assert self._scenario_logger is not None
        lm_config = self.config.lm_studio
        if lm_config is None or not lm_config.enabled:
            self._scenario_logger.warning(
                "llm_arm_disabled",
                reason="lm_studio missing or disabled",
            )
            self._disable_arm("llm")
            return
        # Local imports keep the openai SDK surface out of the cold path.
        from src.integrations.lm_studio.client import LMStudioClient
        from src.integrations.lm_studio.preflight import check_lm_studio_server
        from src.integrations.lm_studio.schema import LMStudioError

        try:
            report = check_lm_studio_server(lm_config)
        except (LMStudioError, OSError, RuntimeError, ValueError, AttributeError) as exc:
            self._scenario_logger.warning(
                "llm_preflight_raised", error=str(exc), error_type=type(exc).__name__
            )
            self._disable_arm("llm")
            return
        if not report.passed:
            self._scenario_logger.warning(
                "llm_preflight_failed", failure_reason=report.failure_reason
            )
            self._disable_arm("llm")
            return
        client_config = lm_config.model_copy(update={"preflight_on_construct": False})
        try:
            self._lm_client = LMStudioClient(client_config, scenario_logger=self._scenario_logger)
        except (LMStudioError, OSError, RuntimeError, ValueError, ImportError) as exc:
            self._scenario_logger.warning(
                "llm_client_construction_failed", error=str(exc), error_type=type(exc).__name__
            )
            self._disable_arm("llm")

    def _disable_arm(self, arm: str) -> None:
        self._arm_enabled[arm] = False

    def _maybe_drop_primary_thresholds(self) -> None:
        """Drop the headline thresholds if the primary arm was gated off.

        The thresholds are computed from the primary (first) arm; if that arm
        is disabled there is no metric to evaluate, so the thresholds would
        auto-FAIL. Dropping them yields SKIPPED-like semantics (mirrors the
        threshold-drop pattern in ``llm_prior_ablation``).
        """
        assert self._scenario_logger is not None
        if not self._arm_enabled.get(self.config.primary_arm, False):
            self._scenario_logger.warning(
                "primary_arm_disabled",
                primary_arm=self.config.primary_arm,
                reason="dropping headline thresholds to avoid spurious FAIL",
            )
            self.config.thresholds = [
                t
                for t in self.config.thresholds
                if t.name not in (_ERROR_REDUCTION_KEY, _FINAL_RESIDUAL_KEY)
            ]

    def _active_arms(self) -> list[str]:
        return [arm for arm in self.config.arms if self._arm_enabled.get(arm, False)]

    # ------------------------------------------------------------------ #
    # Per-cell run                                                        #
    # ------------------------------------------------------------------ #

    def _run_cell(
        self,
        *,
        arm: str,
        operator: PDEOperator,
        basis_descriptions: list[str],
        seed: int,
        cell_logger: ScenarioLogger,
    ) -> tuple[float, float]:
        """Run one (arm, seed) cell -> (error_reduction_pct, final_residual)."""
        # Seed the global RNGs before constructing the search (MCTS.__init__
        # has no seed kwarg) — preserves per-seed reproducibility.
        np.random.seed(seed)
        torch.manual_seed(seed)

        game = self._build_game(operator)
        initial_error = float(game.get_initial_state().error_estimate)

        evaluator = self._build_evaluator(
            arm=arm,
            game=game,
            basis_descriptions=basis_descriptions,
            seed=seed,
            cell_logger=cell_logger,
        )
        outcome = run_basis_selection_cell(
            game=game,
            evaluator=evaluator,
            target_residual=self.config.target_residual,
            max_rollouts=self.config.max_rollouts_for_cell(),
            n_simulations=self.config.n_simulations,
            scenario_logger=cell_logger,
        )
        reduction_pct = self._reduction_pct(initial_error, outcome.final_residual)
        cell_logger.info(
            "cell_complete",
            initial_error=initial_error,
            final_residual=outcome.final_residual,
            reduction_pct=reduction_pct,
            rollouts_used=outcome.rollouts_used,
        )
        return reduction_pct, outcome.final_residual

    @staticmethod
    def _reduction_pct(initial_error: float, final_residual: float) -> float:
        """Percentage error reduction, floored denominator to avoid div-by-zero."""
        denom = max(abs(initial_error), _REDUCTION_FLOOR)
        return 100.0 * (initial_error - final_residual) / denom

    # ------------------------------------------------------------------ #
    # Construction helpers (helical path + reused primitives)             #
    # ------------------------------------------------------------------ #

    def _build_operator(self) -> PDEOperator:
        """Instantiate the helical operator with the configured geometry.

        When ``config.manufactured`` is set (the default), the operator is
        wrapped with a manufactured product-of-sines target so the basis game
        is non-degenerate (the raw helical operators are homogeneous).
        """
        pde_config = _create_helical_pde_config(
            self.config.operator_name,
            helix_R_major=self.config.helix_r_major,
            helix_r_minor=self.config.helix_r_minor,
            helix_pitch=self.config.helix_pitch,
            helix_n_turns=self.config.helix_n_turns,
        )
        operator_cls = PDEOperatorRegistry().get_or_raise(self.config.operator_name)
        operator = operator_cls(pde_config)
        if self.config.manufactured:
            operator = make_manufactured_operator(operator, self.config.manufactured_wavenumber)
        return operator

    def _build_game(self, operator: PDEOperator) -> BasisSelectionGame:
        return build_basis_game(
            self.config.operator_name,
            operator,
            max_basis_functions=self.config.max_basis_functions,
            n_candidate_bases=self.config.n_candidate_bases,
            target_residual=self.config.target_residual,
        )

    def _build_evaluator(
        self,
        *,
        arm: str,
        game: BasisSelectionGame,
        basis_descriptions: list[str],
        seed: int,
        cell_logger: ScenarioLogger,
    ) -> Evaluator:
        return build_arm_evaluator(
            arm,
            game=game,
            pde_name=self.config.operator_name,
            basis_descriptions=basis_descriptions,
            seed=seed,
            lm_client=self._lm_client,
            trained_model=self._trained_model,
            device=self._device,
            scenario_logger=cell_logger,
        )

    def _load_trained_model(self, checkpoint: str) -> AlphaGalerkinModel:
        from src.training.checkpoint import create_model_from_checkpoint

        assert self._device is not None
        model, _saved_config = create_model_from_checkpoint(
            checkpoint, device=str(self._device), strict=False
        )
        return model

    # ------------------------------------------------------------------ #
    # Aggregation                                                         #
    # ------------------------------------------------------------------ #

    def _record_primary_aggregates(
        self, reductions: dict[str, list[float]], finals: dict[str, list[float]]
    ) -> None:
        """Record the headline metrics from the primary arm."""
        assert self._scenario_logger is not None
        primary = self.config.primary_arm
        if primary not in reductions:
            # Primary arm gated off after threshold drop — nothing to record.
            return
        reduction_med = median_of(reductions[primary])
        final_med = median_of(finals[primary])
        self.record_metric(_ERROR_REDUCTION_KEY, reduction_med)
        self.record_metric(_FINAL_RESIDUAL_KEY, final_med)
        self._scenario_logger.metric(_ERROR_REDUCTION_KEY, reduction_med, arm=primary)
        self._scenario_logger.metric(_FINAL_RESIDUAL_KEY, final_med, arm=primary)
