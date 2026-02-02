"""Scenario runner for executing PoC scenarios.

This module provides the main execution engine for running scenarios
either individually or in batch.

Features:
    - Configuration-driven execution
    - Parallel execution support
    - Retry logic with exponential backoff
    - Progress reporting
    - Result aggregation
"""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import yaml

from src.poc.config import (
    BaseScenarioConfig,
    ScenarioResult,
    ScenarioStatus,
    load_config_from_dict,
)
from src.poc.registry import BaseScenario, ScenarioRegistry
from src.poc.results import ResultCollector

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


class ScenarioRunner:
    """Orchestrates scenario execution.

    Handles:
        - Loading scenarios from config files or registry
        - Sequential and parallel execution
        - Retry logic
        - Result collection and reporting
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs/poc",
        max_workers: int = 1,
        retry_delay_base: float = 1.0,
        fail_fast: bool = False,
    ) -> None:
        """Initialize the runner.

        Args:
            output_dir: Directory for artifacts and results.
            max_workers: Max parallel executions (1 = sequential).
            retry_delay_base: Base delay for exponential backoff retries.
            fail_fast: Stop on first failure.

        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers
        self.retry_delay_base = retry_delay_base
        self.fail_fast = fail_fast

        self.registry = ScenarioRegistry()
        self.collector = ResultCollector(self.output_dir)

        self._logger = structlog.get_logger(__name__)

    def load_config(self, config_path: str | Path) -> list[BaseScenarioConfig]:
        """Load scenario configurations from YAML file.

        Args:
            config_path: Path to YAML config file.

        Returns:
            List of scenario configs.

        """
        config_path = Path(config_path)
        self._logger.info("loading_config", path=str(config_path))

        with open(config_path) as f:
            data = yaml.safe_load(f)

        configs: list[BaseScenarioConfig] = []

        # Support both single scenario and list of scenarios
        scenarios_data = data.get("scenarios", [data])
        if not isinstance(scenarios_data, list):
            scenarios_data = [scenarios_data]

        for scenario_data in scenarios_data:
            config = load_config_from_dict(scenario_data)
            configs.append(config)
            self._logger.debug(
                "config_loaded",
                scenario=config.name,
                tier=config.tier.value,
            )

        return configs

    def run(
        self,
        scenario_name: str,
        config: BaseScenarioConfig | None = None,
        **kwargs: Any,
    ) -> ScenarioResult:
        """Run a single scenario.

        Args:
            scenario_name: Registered scenario name.
            config: Optional config. If None, uses defaults.
            **kwargs: Config overrides.

        Returns:
            ScenarioResult.

        Raises:
            ValueError: If scenario not found.

        """
        scenario_cls = self.registry.get(scenario_name)
        if scenario_cls is None:
            raise ValueError(
                f"Scenario '{scenario_name}' not found. Available: {self.registry.list_scenarios()}"
            )

        self._logger.info(
            "running_scenario",
            scenario=scenario_name,
        )

        # Create scenario instance
        scenario = scenario_cls(config=config, **kwargs)

        # Handle retries
        result = self._run_with_retry(scenario)

        # Collect result
        self.collector.collect(result)

        return result

    def run_all(
        self,
        filter_tier: str | None = None,
        filter_names: Sequence[str] | None = None,
    ) -> list[ScenarioResult]:
        """Run all registered scenarios.

        Args:
            filter_tier: Only run scenarios of this tier.
            filter_names: Only run scenarios with these names.

        Returns:
            List of results.

        """
        scenarios = self.registry.get_all()

        # Apply filters
        if filter_names:
            scenarios = {k: v for k, v in scenarios.items() if k in filter_names}

        self._logger.info(
            "running_all_scenarios",
            count=len(scenarios),
            filter_tier=filter_tier,
        )

        results: list[ScenarioResult] = []

        if self.max_workers > 1:
            results = self._run_parallel(scenarios, filter_tier)
        else:
            results = self._run_sequential(scenarios, filter_tier)

        # Generate summary
        self._print_summary(results)

        return results

    def run_from_config(
        self,
        config_path: str | Path,
    ) -> list[ScenarioResult]:
        """Run scenarios defined in a config file.

        Args:
            config_path: Path to YAML config file.

        Returns:
            List of results.

        """
        configs = self.load_config(config_path)

        results: list[ScenarioResult] = []
        for config in configs:
            if not config.enabled:
                self._logger.info("scenario_skipped", scenario=config.name)
                continue

            try:
                result = self.run(config.name, config=config)
                results.append(result)

                if self.fail_fast and not result.passed:
                    self._logger.warning(
                        "fail_fast_triggered",
                        scenario=config.name,
                    )
                    break

            except ValueError as e:
                self._logger.error("scenario_not_found", error=str(e))

        self._print_summary(results)
        return results

    def _run_with_retry(
        self,
        scenario_or_cls: BaseScenario | type[BaseScenario],
        config: BaseScenarioConfig | None = None,
    ) -> ScenarioResult:
        """Run scenario with retry logic.

        Creates a fresh scenario instance for each retry attempt to avoid
        state contamination from previous failed attempts.

        Args:
            scenario_or_cls: Scenario instance or class.
            config: Optional config (used when passing class).

        Returns:
            Final result (last attempt).

        """
        # Get scenario class and config for fresh instantiation
        if isinstance(scenario_or_cls, type):
            scenario_cls = scenario_or_cls
            if config is None:
                # Create temporary instance to get default config
                temp = scenario_cls()
                config = temp.config
        else:
            scenario_cls = type(scenario_or_cls)
            config = scenario_or_cls.config

        max_attempts = config.retry_count + 1
        result: ScenarioResult | None = None

        for attempt in range(1, max_attempts + 1):
            # Create fresh scenario instance for each attempt
            scenario = scenario_cls(config=config)

            self._logger.debug(
                "attempt_starting",
                scenario=scenario.name,
                attempt=attempt,
                max_attempts=max_attempts,
            )

            result = scenario.run()

            if result.status in (ScenarioStatus.PASSED, ScenarioStatus.SKIPPED):
                return result

            if attempt < max_attempts:
                delay = self.retry_delay_base * (2 ** (attempt - 1))
                self._logger.info(
                    "retrying_scenario",
                    scenario=scenario.name,
                    delay=delay,
                    attempt=attempt,
                )
                time.sleep(delay)

        # result is guaranteed to be set after loop
        assert result is not None
        return result

    def _run_sequential(
        self,
        scenarios: dict[str, type[BaseScenario]],
        filter_tier: str | None,
    ) -> list[ScenarioResult]:
        """Run scenarios sequentially.

        Args:
            scenarios: Dict of scenario name -> class.
            filter_tier: Optional tier filter.

        Returns:
            List of results.

        """
        results: list[ScenarioResult] = []

        for _name, scenario_cls in scenarios.items():
            # Create temporary instance to check config (filters)
            temp_scenario = scenario_cls()
            config = temp_scenario.config

            # Apply tier filter
            if filter_tier and config.tier.value != filter_tier:
                continue

            if not config.enabled:
                continue

            # Run with fresh instance via retry logic
            result = self._run_with_retry(scenario_cls, config=config)
            results.append(result)
            self.collector.collect(result)

            if self.fail_fast and not result.passed:
                break

        return results

    def _run_parallel(
        self,
        scenarios: dict[str, type[BaseScenario]],
        filter_tier: str | None,
    ) -> list[ScenarioResult]:
        """Run scenarios in parallel.

        Args:
            scenarios: Dict of scenario name -> class.
            filter_tier: Optional tier filter.

        Returns:
            List of results.

        """
        from datetime import datetime

        results: list[ScenarioResult] = []

        # Filter scenarios and collect configs
        filtered: list[tuple[str, type[BaseScenario], BaseScenarioConfig]] = []
        for name, scenario_cls in scenarios.items():
            temp_scenario = scenario_cls()
            config = temp_scenario.config
            if filter_tier and config.tier.value != filter_tier:
                continue
            if not config.enabled:
                continue
            filtered.append((name, scenario_cls, config))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_info: dict[
                concurrent.futures.Future[ScenarioResult],
                tuple[str, BaseScenarioConfig],
            ] = {
                executor.submit(self._run_scenario_task, cls, cfg): (name, cfg)
                for name, cls, cfg in filtered
            }

            for future in concurrent.futures.as_completed(future_to_info):
                name, config = future_to_info[future]
                try:
                    # Use config timeout if available
                    timeout = config.timeout_seconds
                    result = future.result(timeout=timeout)
                    results.append(result)
                    self.collector.collect(result)
                except concurrent.futures.TimeoutError:
                    self._logger.error(
                        "scenario_timeout",
                        scenario=name,
                        timeout=config.timeout_seconds,
                    )
                    # Create timeout result
                    timeout_result = ScenarioResult(
                        scenario_name=name,
                        config_hash=config.compute_hash(),
                        status=ScenarioStatus.ERROR,
                        passed=False,
                        metrics={},
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        duration_seconds=config.timeout_seconds,
                        error_message=f"Scenario timed out after {config.timeout_seconds}s",
                    )
                    results.append(timeout_result)
                    self.collector.collect(timeout_result)
                except Exception as e:
                    self._logger.error(
                        "parallel_execution_error",
                        scenario=name,
                        error=str(e),
                        exc_info=True,
                    )

        return results

    def _run_scenario_task(
        self,
        scenario_cls: type[BaseScenario],
        config: BaseScenarioConfig,
    ) -> ScenarioResult:
        """Task for parallel execution.

        Args:
            scenario_cls: Scenario class to instantiate and run.
            config: Scenario configuration.

        Returns:
            ScenarioResult.

        """
        return self._run_with_retry(scenario_cls, config=config)

    def _print_summary(self, results: list[ScenarioResult]) -> None:
        """Print execution summary.

        Args:
            results: List of results.

        """
        if not results:
            print("\nNo scenarios executed.")
            return

        print("\n" + "=" * 60)
        print("POC SCENARIO EXECUTION SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        total_time = sum(r.duration_seconds for r in results)

        print(f"\nResults: {passed} passed, {failed} failed")
        print(f"Total time: {total_time:.2f}s")
        print()

        for result in results:
            print(result.summary())
            print()

        print("=" * 60)
        if failed == 0:
            print("ALL SCENARIOS PASSED")
        else:
            print(f"FAILED: {failed} scenario(s)")
        print("=" * 60)

        # Save summary
        self.collector.save_summary(results)


def create_runner(
    output_dir: str | Path = "outputs/poc",
    **kwargs: Any,
) -> ScenarioRunner:
    """Factory function to create a configured runner.

    Args:
        output_dir: Output directory.
        **kwargs: Additional runner configuration.

    Returns:
        Configured ScenarioRunner.

    """
    return ScenarioRunner(output_dir=output_dir, **kwargs)
