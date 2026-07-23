"""``ScenarioResultSink`` — bridge a harness ``RunResult`` into the PoC gate.

The single authoritative cross-run regression gate in AlphaGalerkin is
``src/poc/baselines`` (driven by ``python -m src.poc.cli record-baseline`` /
``diff``). Rather than add a second gate, this sink writes a
``ScenarioResult``-shaped JSON document into the PoC results layout
(``{output_dir}/results/{run_id}/<file>.json``) that ``_load_run_result_dicts``
and ``observed_from_result_dicts`` already parse — so a harness run is gated by
the *existing* machinery with no duplicated logic.

Each harness ``ScoreAggregate`` becomes two metrics: ``<score>_mean`` (lower-better
by default; residual) and ``<score>_pass_rate`` (higher-better — wire ``_pass_rate``
into ``DEFAULT_HIGHER_BETTER_SUFFIXES``). The harness's own ``GateConfig`` remains
an independent in-run sanity check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from eval_harness.core.interfaces import ResultSink

if TYPE_CHECKING:
    from eval_harness.core.types import RunResult

logger = structlog.get_logger(__name__)

DEFAULT_RESULTS_DIR: str = "outputs/poc"
"""Default PoC output root; the sink writes under ``{dir}/results/{run_id}/``."""

DEFAULT_FILENAME: str = "eval_harness.json"
"""Default basename for the emitted ScenarioResult-shaped document."""


class ScenarioResultSink(ResultSink):  # eval_harness ships no stubs; mypy 'misc' off for src.*
    """Persist a ``RunResult`` as a PoC ``ScenarioResult`` JSON for the baseline gate."""

    def __init__(
        self,
        output_dir: str = DEFAULT_RESULTS_DIR,
        scenario_name: str | None = None,
        filename: str = DEFAULT_FILENAME,
    ) -> None:
        """Construct the sink.

        Args:
            output_dir: PoC output root; the document is written under
                ``{output_dir}/results/{run_id}/``.
            scenario_name: Scenario label recorded in the document; defaults to
                ``run.config_name`` when ``None``.
            filename: Basename of the emitted JSON file.

        """
        self.output_dir = Path(output_dir)
        self.scenario_name = scenario_name
        self.filename = filename

    def emit(self, run: RunResult) -> None:
        """Write the run's aggregate scores as a ScenarioResult-shaped JSON doc."""
        scenario = self.scenario_name or run.config_name
        metrics: dict[str, float] = {}
        for name, agg in run.aggregate.items():
            metrics[f"{name}_mean"] = float(agg.mean)
            if agg.pass_rate is not None:
                metrics[f"{name}_pass_rate"] = float(agg.pass_rate)
        payload = {
            "scenario_name": scenario,
            "run_id": run.run_id,
            "n_items": len(run.items),
            "metrics": metrics,
        }
        run_dir = self.output_dir / "results" / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / self.filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info(
            "eval_harness_result_emitted",
            path=str(path),
            scenario=scenario,
            n_metrics=len(metrics),
        )


__all__ = ["DEFAULT_FILENAME", "DEFAULT_RESULTS_DIR", "ScenarioResultSink"]
