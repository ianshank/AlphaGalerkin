"""Gate 1 look-ahead-vs-greedy scenario: one testbed's pre-registered go/no-go.

The harness is imported at module level, unlike the older ``*_compare``
scenarios' lazy imports (``_compare_common`` AD7). Those exist to break a
poc<->research import cycle, which needs a *runtime* ``research -> poc`` edge to
close; the Gate 1 harness has none (its poc imports are ``TYPE_CHECKING``-only),
and ``tests/research/test_lookahead_vs_greedy.py`` pins that importing it loads
no ``src.poc`` module.

Outcomes: GO -> ``PASSED``, NO-GO -> ``FAILED`` (both completed runs, with
artifacts); an adequacy abort -> ``SKIPPED`` with ``adequacy_aborted = 1`` and
no artifacts. The verdict is the config's pre-registered thresholds, evaluated
by the harness and by this scenario on the same objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.registry import scenario
from src.poc.scenarios._compare_common import CompareScenarioBase
from src.poc.scenarios.lookahead_vs_greedy_config import (
    SCENARIO_NAME,
    LookaheadVsGreedyConfig,
)
from src.research.lookahead_vs_greedy import (
    ABORTED,
    ADEQUACY_ABORTED_METRIC,
    AdequacyAbortedError,
    LookaheadVsGreedyResult,
    adequacy_metrics,
    run_lookahead_vs_greedy,
)
from src.research.lookahead_vs_greedy_artifacts import write_artifacts
from src.research.run_manifest import RunRecorder

if TYPE_CHECKING:
    from collections.abc import Mapping


@scenario(SCENARIO_NAME)
class LookaheadVsGreedyScenario(CompareScenarioBase):
    """Registered PoC scenario for the Gate 1 go/no-go on one testbed."""

    config_class = LookaheadVsGreedyConfig

    def __init__(
        self,
        config: LookaheadVsGreedyConfig | None = None,
        *,
        proposal_grade: bool = False,
        **kwargs: Any,
    ) -> None:
        """Construct the scenario.

        Args:
            config: The testbed's config (defaults to T1).
            proposal_grade: Refuse to start on a tree that is not provably
                clean, and re-verify the sidecar from disk (``RunRecorder``).
                A run-mode flag, not a config field: it must not move the
                config hash.
            **kwargs: Config overrides (``BaseScenario`` semantics).

        """
        super().__init__(config, **kwargs)
        self.config: LookaheadVsGreedyConfig
        self._proposal_grade = proposal_grade

    def _setup_log_fields(self) -> dict[str, Any]:
        return {
            "testbed": self.config.testbed_label(),
            "kind": self.config.substrate.kind,
            "operator_name": self.config.operator_name,
            "max_steps": self.config.max_steps,
            "n_simulations": self.config.n_simulations,
            "top_k_actions": self.config.top_k_actions,
            "robust_n_seeds": self.config.robust_n_seeds,
            "require_adequacy": self.config.require_adequacy_precondition,
            "proposal_grade": self._proposal_grade,
        }

    def execute(self) -> ScenarioResult:
        """Snapshot git, run the testbed, record metrics, write the artifacts."""
        assert self._scenario_logger is not None
        # Before anything is computed or written: a probe taken after the
        # artifacts exist would report them as a dirty tree.
        recorder = RunRecorder.start(
            config_hash=self.config.compute_hash(), proposal_grade=self._proposal_grade
        )
        try:
            result = run_lookahead_vs_greedy(self.config)
        except AdequacyAbortedError as abort:
            return self._aborted(abort)
        self._record_metrics(result)
        for name, path in write_artifacts(result, self.config, recorder).items():
            self.record_artifact(name, str(path))
        return self._create_result(status=ScenarioStatus.RUNNING)

    def _aborted(self, abort: AdequacyAbortedError) -> ScenarioResult:
        """Record an adequacy abort: no comparison ran, so no artifact and no verdict."""
        assert self._scenario_logger is not None
        self.record_metric(ADEQUACY_ABORTED_METRIC, ABORTED)
        for name, value in adequacy_metrics(abort.separation).items():
            self.record_metric(name, value)
        self._scenario_logger.info(
            "lookahead_testbed_aborted",
            testbed=self.config.testbed_label(),
            violations=abort.violations,
        )
        return self._create_result(status=ScenarioStatus.SKIPPED)

    def _log_metrics_recorded(
        self,
        comparison: object,
        metrics: Mapping[str, float],
    ) -> None:
        assert self._scenario_logger is not None
        if not isinstance(comparison, LookaheadVsGreedyResult):
            raise TypeError(f"expected LookaheadVsGreedyResult, got {type(comparison).__name__}")
        self._scenario_logger.info(
            "lookahead_recorded",
            testbed=comparison.testbed,
            verdict=comparison.verdict.label,
            best_classical=comparison.primary.reading.best_classical,
            **{item.metric: item.value for item in comparison.verdict.criteria},
        )


__all__ = ["LookaheadVsGreedyScenario"]
