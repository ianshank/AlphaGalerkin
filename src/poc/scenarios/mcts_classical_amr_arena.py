"""MCTS vs classical AMR arena scenario (element-local substrate)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.registry import scenario
from src.poc.scenarios._compare_common import CompareScenarioBase
from src.poc.scenarios.mcts_classical_amr_arena_config import (
    SCENARIO_NAME,
    MCTSClassicalAMRArenaConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.research.mcts_classical_amr_arena import MultiSeedArena
    from src.research.run_manifest import GitProvenance


@scenario(SCENARIO_NAME)
class MCTSClassicalAMRArenaScenario(CompareScenarioBase):
    """Registered PoC scenario for the pre-registered AMR arena."""

    config_class = MCTSClassicalAMRArenaConfig

    def __init__(
        self,
        config: MCTSClassicalAMRArenaConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.config: MCTSClassicalAMRArenaConfig

    def _setup_log_fields(self) -> dict[str, Any]:
        return {
            "kind": self.config.substrate.kind,
            "theta": self.config.marking_fraction,
            "n_simulations": self.config.n_simulations,
            "require_adequacy": self.config.require_adequacy_precondition,
        }

    def execute(self) -> ScenarioResult:
        """Run classical + MCTS arms and persist CSV / sidecar / optional PNG."""
        assert self._scenario_logger is not None
        from src.research.mcts_classical_amr_arena import (
            export_csv,
            export_plot,
            run_comparison,
            write_arena_manifest,
        )
        from src.research.run_manifest import collect_git_provenance

        # Snapshot before any artifact write so untracked CSV/PNG cannot
        # flip git.dirty on an otherwise clean source tree.
        git = collect_git_provenance()
        arena = run_comparison(self.config)
        self._record_metrics(arena)
        self._write_artifacts(arena, export_csv, export_plot, write_arena_manifest, git=git)
        return self._create_result(status=ScenarioStatus.RUNNING)

    def _log_metrics_recorded(
        self,
        comparison: object,
        metrics: Mapping[str, float],
    ) -> None:
        assert self._scenario_logger is not None
        arena = comparison
        dof_convention = getattr(arena, "dof_convention", None)
        self._scenario_logger.info(
            "arena_recorded",
            l2_error_ratio_at_matched_dof=metrics["l2_error_ratio_at_matched_dof"],
            mcts_win_fraction=metrics["mcts_win_fraction"],
            n_seeds=metrics["n_seeds"],
            dof_convention=dof_convention,
        )

    def _write_artifacts(
        self,
        arena: MultiSeedArena,
        export_csv: Any,
        export_plot: Any,
        write_arena_manifest: Any,
        *,
        git: GitProvenance,
    ) -> None:
        """Write CSV, optional PNG, and the run sidecar.

        Arena keeps ``write_arena_manifest`` + ``.run.json``; the CSV/PNG
        helper used by the other compare families must not swallow this.
        """
        base_str = self._artifact_base_str()
        csv_path = Path(f"{base_str}.csv")
        png_path = Path(f"{base_str}.png")
        export_csv(arena, csv_path)
        self.record_artifact("csv", str(csv_path))
        plotted = export_plot(arena, png_path)
        if plotted is not None:
            self.record_artifact("png", str(plotted))
        sidecar = write_arena_manifest(
            arena,
            self.config,
            csv_path,
            plotted,
            proposal_grade=False,
            git=git,
        )
        self.record_artifact("run_json", str(sidecar))


__all__ = ["MCTSClassicalAMRArenaScenario"]
