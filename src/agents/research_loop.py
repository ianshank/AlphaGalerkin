"""Centaur research-loop harness — sweep MCTS+evaluator across a manifest.

This extends the agent-physics subsystem with Brown's "billions of Einsteins"
loop: a cloneable solver driven across a config-defined manifest of independent
problems, aggregating a per-problem *discovery ledger* of which evaluator arm
reached the lowest residual.

The orchestrator follows the same ``BaseExecutable`` pattern as
:class:`~src.agents.orchestrator.AgentOrchestrator` and reuses the shared MCTS
basis-selection primitives in :mod:`src.poc.scenarios._centaur_common`, so the
inner rollout, operator/game construction, and arm evaluator factory are not
re-implemented here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.integrations.lm_studio.client import LMStudioClient
from src.integrations.lm_studio.preflight import check_lm_studio_server
from src.integrations.lm_studio.schema import LMStudioError
from src.poc.device import resolve_device
from src.poc.scenarios._centaur_common import (
    CellOutcome,
    build_arm_evaluator,
    build_basis_game,
    build_pde_operator,
    enumerate_basis_descriptions,
    run_basis_selection_cell,
)
from src.templates.base import BaseExecutable, ExecutionResult, ExecutionStatus
from src.templates.logging import create_logger_class

if TYPE_CHECKING:
    from src.agents.config import ResearchLoopConfig, ResearchProblemSpec
    from src.pde.operators import PDEOperator

ResearchLoopLogger = create_logger_class("ResearchLoop")

# Result of solving one problem: arm -> list of per-seed cell outcomes.
ProblemResults = dict[str, list[CellOutcome]]


def _median(samples: list[float]) -> float:
    if not samples:
        return float("nan")
    return float(np.median(np.asarray(samples, dtype=np.float64)))


class ResearchLoopOrchestrator(BaseExecutable["ResearchLoopConfig"]):
    """Drive MCTS+evaluator across a manifest and build a discovery ledger.

    Args:
        config: Research-loop configuration (manifest + arm/budget knobs).
        run_id: Optional run identifier.

    """

    _executable_name = "research_loop"
    _logger_class = ResearchLoopLogger

    def __init__(self, config: ResearchLoopConfig, run_id: str | None = None) -> None:
        super().__init__(config, run_id)
        self._device: torch.device | None = None
        self._lm_client: LMStudioClient | None = None
        self._trained_model: Any | None = None
        self._available_arms: set[str] = set()

    # ------------------------------------------------------------------ #
    # Entry point                                                         #
    # ------------------------------------------------------------------ #

    def execute(self) -> ExecutionResult:
        """Run the full research loop and return an ``ExecutionResult``."""
        try:
            self._device = resolve_device(self.config.device, context=self._executable_name)
            self._available_arms = self._gate_arms()

            if not self._available_arms:
                self.logger.warning("no_arms_available", reason="all arms gated off")
                return self._create_result(status=ExecutionStatus.SKIPPED)

            seeds = self.config.resolved_seeds()
            self.logger.info(
                "research_loop_started",
                n_problems=len(self.config.problems),
                available_arms=sorted(self._available_arms),
                n_seeds=len(seeds),
                parallel=self.config.parallel,
            )

            if self.config.parallel:
                results = self._run_parallel(seeds)
            else:
                results = self._run_sequential(seeds)

            metrics, metadata = self._aggregate(results)
            status = self._status_from_metrics(metrics)
            return self._create_result(status=status, metrics=metrics, metadata=metadata)
        except Exception as exc:  # noqa: BLE001 - surfaced as a FAILED result
            self.logger.exception("research_loop_failed", error=str(exc))
            return self._create_result(status=ExecutionStatus.FAILED, error=str(exc))
        finally:
            self._teardown()

    def _teardown(self) -> None:
        if self._lm_client is not None:
            self._lm_client.close()
            self._lm_client = None
        self._trained_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Arm gating                                                          #
    # ------------------------------------------------------------------ #

    def _gate_arms(self) -> set[str]:
        """Return the runnable arms across the whole manifest."""
        requested: set[str] = set(self.config.default_arms)
        for problem in self.config.problems:
            requested.update(self.config.arms_for(problem))

        available: set[str] = set()
        if "random" in requested:
            available.add("random")
        if "trained" in requested and self._gate_trained_arm():
            available.add("trained")
        if "llm" in requested and self._gate_llm_arm():
            available.add("llm")
        return available

    def _gate_trained_arm(self) -> bool:
        checkpoint = self.config.trained_checkpoint_path
        if checkpoint is None:
            self.logger.warning(
                "trained_arm_skipped_no_checkpoint",
                reason="trained_checkpoint_path is None",
            )
            return False
        try:
            self._trained_model = self._load_trained_model(checkpoint)
        except (FileNotFoundError, RuntimeError) as exc:
            self.logger.warning(
                "trained_arm_load_failed",
                checkpoint=str(checkpoint),
                error=str(exc),
            )
            return False
        return True

    def _gate_llm_arm(self) -> bool:
        if not self.config.lm_studio.enabled:
            self.logger.warning("llm_arm_disabled_by_config", reason="lm_studio.enabled is False")
            return False
        try:
            report = check_lm_studio_server(self.config.lm_studio)
        except (LMStudioError, OSError, RuntimeError, ValueError, AttributeError) as exc:
            self.logger.warning(
                "llm_preflight_raised", error=str(exc), error_type=type(exc).__name__
            )
            return False
        if not report.passed:
            self.logger.warning(
                "llm_preflight_failed",
                failure_reason=report.failure_reason,
                available_models=report.available_models,
            )
            return False
        client_config = self.config.lm_studio.model_copy(update={"preflight_on_construct": False})
        try:
            self._lm_client = LMStudioClient(client_config)
        except (LMStudioError, OSError, RuntimeError, ValueError, ImportError) as exc:
            self.logger.warning(
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
            checkpoint, device=str(self._device), strict=False
        )
        return model

    def _effective_arms(self, problem: ResearchProblemSpec) -> list[str]:
        """Per-problem arms intersected with the globally available arms."""
        return [a for a in self.config.arms_for(problem) if a in self._available_arms]

    # ------------------------------------------------------------------ #
    # Solving                                                             #
    # ------------------------------------------------------------------ #

    def _run_sequential(self, seeds: list[int]) -> dict[str, ProblemResults]:
        return {p.name: self._solve_problem(p, seeds) for p in self.config.problems}

    def _run_parallel(self, seeds: list[int]) -> dict[str, ProblemResults]:
        """One worker thread per problem (problems are independent)."""
        problems = self.config.problems
        with ThreadPoolExecutor(max_workers=len(problems)) as pool:
            solved = list(pool.map(lambda p: self._solve_problem(p, seeds), problems))
        # Preserve manifest order regardless of completion order.
        return {p.name: r for p, r in zip(problems, solved, strict=True)}

    def _solve_problem(self, problem: ResearchProblemSpec, seeds: list[int]) -> ProblemResults:
        operator = build_pde_operator(problem.pde)
        descriptions = enumerate_basis_descriptions(self._build_game(problem, operator))
        results: ProblemResults = {}
        for arm in self._effective_arms(problem):
            outcomes: list[CellOutcome] = []
            for seed in seeds:
                outcomes.append(self._solve_cell(problem, operator, descriptions, arm, seed))
            results[arm] = outcomes
        return results

    def _build_game(self, problem: ResearchProblemSpec, operator: PDEOperator) -> Any:
        return build_basis_game(
            problem.pde,
            operator,
            max_basis_functions=self.config.max_basis_functions,
            n_candidate_bases=self.config.n_candidate_bases,
            target_residual=self.config.target_residual,
        )

    def _solve_cell(
        self,
        problem: ResearchProblemSpec,
        operator: PDEOperator,
        descriptions: list[str],
        arm: str,
        seed: int,
    ) -> CellOutcome:
        """Solve a single (problem, arm, seed) cell. Override point for tests."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        game = self._build_game(problem, operator)
        evaluator = build_arm_evaluator(
            arm,
            game=game,
            pde_name=problem.pde,
            basis_descriptions=descriptions,
            seed=seed,
            lm_client=self._lm_client,
            trained_model=self._trained_model,
            device=self._device,
        )
        return run_basis_selection_cell(
            game=game,
            evaluator=evaluator,
            target_residual=self.config.target_residual,
            max_rollouts=self.config.max_rollouts,
            n_simulations=self.config.n_mcts_simulations,
        )

    # ------------------------------------------------------------------ #
    # Aggregation                                                         #
    # ------------------------------------------------------------------ #

    def _aggregate(
        self, results: dict[str, ProblemResults]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Build the discovery ledger, headline metrics, and metadata."""
        metrics: dict[str, float] = {}
        ledger: dict[str, Any] = {}
        arm_wins: dict[str, int] = {}
        n_solved = 0

        for problem_name, arm_results in results.items():
            per_arm: dict[str, dict[str, float]] = {}
            best_arm: str | None = None
            best_residual = float("inf")
            best_rollouts = float("inf")
            for arm, outcomes in arm_results.items():
                residual = _median([o.final_residual for o in outcomes])
                rollouts = _median([float(o.rollouts_used) for o in outcomes])
                per_arm[arm] = {"median_residual": residual, "median_rollouts": rollouts}
                metrics[f"{problem_name}_{arm}_median_residual"] = residual
                metrics[f"{problem_name}_{arm}_median_rollouts"] = rollouts
                # Lower residual wins; ties broken by fewer rollouts.
                if np.isfinite(residual) and (
                    residual < best_residual
                    or (residual == best_residual and rollouts < best_rollouts)
                ):
                    best_arm = arm
                    best_residual = residual
                    best_rollouts = rollouts

            solved = best_arm is not None and best_residual <= self.config.target_residual
            if solved:
                n_solved += 1
            if best_arm is not None:
                arm_wins[best_arm] = arm_wins.get(best_arm, 0) + 1

            ledger[problem_name] = {
                "best_arm": best_arm,
                "best_residual": best_residual if np.isfinite(best_residual) else None,
                "best_rollouts": best_rollouts if np.isfinite(best_rollouts) else None,
                "solved": solved,
                "per_arm": per_arm,
            }
            metrics[f"{problem_name}_solved"] = 1.0 if solved else 0.0

        n_problems = len(results)
        metrics["n_problems"] = float(n_problems)
        metrics["n_problems_solved"] = float(n_solved)
        metrics["solved_fraction"] = (n_solved / n_problems) if n_problems else 0.0
        for arm, wins in arm_wins.items():
            metrics[f"arm_wins_{arm}"] = float(wins)

        metadata: dict[str, Any] = {
            "discovery_ledger": ledger,
            "available_arms": sorted(self._available_arms),
            "device": str(self._device),
        }
        self.logger.info(
            "research_loop_aggregated",
            n_problems=n_problems,
            n_solved=n_solved,
            arm_wins=arm_wins,
        )
        return metrics, metadata

    def _status_from_metrics(self, metrics: dict[str, float]) -> ExecutionStatus:
        solved_fraction = metrics.get("solved_fraction", 0.0)
        if solved_fraction >= self.config.min_solved_fraction:
            return ExecutionStatus.COMPLETED
        return ExecutionStatus.FAILED


__all__ = ["ResearchLoopOrchestrator"]
