"""Validation runner for orchestrating validation scenarios.

Provides sequential and parallel execution of validation scenarios.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.validation.config import (
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
)
from src.validation.logging import ValidationLogger, configure_validation_logging
from src.validation.scenarios.gpu_training import GPUTrainingValidator
from src.validation.scenarios.merge_readiness import MergeReadinessChecker
from src.validation.scenarios.tolerance_fixer import ToleranceTestFixer
from src.validation.scenarios.transfer import TransferValidator

logger = structlog.get_logger(__name__)


class ValidationRunner:
    """Orchestrates validation scenario execution.

    Manages the full validation lifecycle:
    1. Configuration loading
    2. Scenario instantiation
    3. Sequential or parallel execution
    4. Result collection and persistence
    """

    def __init__(
        self,
        config: ValidationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize validation runner.

        Args:
            config: Validation configuration.
            **kwargs: Override config fields.
        """
        if config is None:
            config = ValidationConfig(**kwargs)
        elif kwargs:
            from src.validation.utils import deep_merge

            config_dict = config.model_dump()
            # Use deep_merge to properly merge nested configs
            config_dict = deep_merge(config_dict, kwargs)
            config = ValidationConfig(**config_dict)

        self.config = config
        self._results: dict[str, ValidationResult] = {}
        self._logger = ValidationLogger(
            "validation_runner",
            config_hash=config.compute_hash(),
        )

        # Configure logging
        configure_validation_logging(
            level=config.log_level,
            json_output=False,
        )

        # Create output directory
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> dict[str, ValidationResult]:
        """Run all configured validations.

        Returns:
            Dictionary mapping validation names to results.
        """
        self._logger.info(
            "validation_run_starting",
            run_tolerance_fix=self.config.run_tolerance_fix,
            run_gpu_training=self.config.run_gpu_training,
            run_transfer_validation=self.config.run_transfer_validation,
            run_merge_readiness=self.config.run_merge_readiness,
        )

        start_time = datetime.now()
        self._results = {}

        # Build list of validations to run
        validations: list[tuple[str, Any, Any]] = []

        if self.config.run_tolerance_fix:
            validations.append((
                "tolerance_fix",
                ToleranceTestFixer,
                self.config.tolerance,
            ))

        if self.config.run_gpu_training:
            validations.append((
                "gpu_training",
                GPUTrainingValidator,
                self.config.gpu_training,
            ))

        if self.config.run_transfer_validation:
            validations.append((
                "transfer_validation",
                TransferValidator,
                self.config.transfer_validation,
            ))

        if self.config.run_merge_readiness:
            validations.append((
                "merge_readiness",
                MergeReadinessChecker,
                self.config.merge_readiness,
            ))

        # Run validations
        if self.config.parallel and len(validations) > 1:
            self._run_parallel(validations)
        else:
            self._run_sequential(validations)

        # Calculate summary
        duration = (datetime.now() - start_time).total_seconds()
        passed = sum(1 for r in self._results.values() if r.passed)
        failed = sum(1 for r in self._results.values() if not r.passed)

        self._logger.info(
            "validation_run_complete",
            duration_seconds=duration,
            total=len(self._results),
            passed=passed,
            failed=failed,
        )

        # Save results
        if self.config.save_results:
            self._save_results()

        return self._results

    def _run_sequential(
        self,
        validations: list[tuple[str, Any, Any]],
    ) -> None:
        """Run validations sequentially.

        Args:
            validations: List of (name, validator_class, config) tuples.
        """
        for name, validator_cls, config in validations:
            self._logger.info("starting_validation", name=name)

            try:
                validator = validator_cls(config=config)
                result = validator.run()
                self._results[name] = result

                if not result.passed and self.config.stop_on_failure:
                    self._logger.warning(
                        "stopping_on_failure",
                        name=name,
                    )
                    break

            except Exception as e:
                self._logger.error(
                    "validation_exception",
                    name=name,
                    error=str(e),
                )
                self._results[name] = self._create_error_result(name, e)

                if self.config.stop_on_failure:
                    break

    def _run_parallel(
        self,
        validations: list[tuple[str, Any, Any]],
    ) -> None:
        """Run validations in parallel.

        Args:
            validations: List of (name, validator_class, config) tuples.
        """
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}

            for name, validator_cls, config in validations:
                future = executor.submit(self._run_single, name, validator_cls, config)
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    self._results[name] = result

                    if not result.passed and self.config.stop_on_failure:
                        self._logger.warning(
                            "stopping_on_failure",
                            name=name,
                        )
                        executor.shutdown(wait=False)
                        break

                except Exception as e:
                    self._logger.error(
                        "validation_exception",
                        name=name,
                        error=str(e),
                    )
                    self._results[name] = self._create_error_result(name, e)

    def _run_single(
        self,
        name: str,
        validator_cls: Any,
        config: Any,
    ) -> ValidationResult:
        """Run a single validation.

        Args:
            name: Validation name.
            validator_cls: Validator class.
            config: Validation configuration.

        Returns:
            ValidationResult.
        """
        self._logger.info("starting_validation", name=name)
        validator = validator_cls(config=config)
        return validator.run()

    def _create_error_result(
        self,
        name: str,
        error: Exception,
    ) -> ValidationResult:
        """Create an error result.

        Args:
            name: Validation name.
            error: Exception that occurred.

        Returns:
            ValidationResult with error status.
        """
        import sys
        import traceback

        now = datetime.now()
        return ValidationResult(
            validation_name=name,
            config_hash="",
            status=ValidationStatus.ERROR,
            passed=False,
            start_time=now,
            end_time=now,
            duration_seconds=0,
            error_message=str(error),
            error_traceback=traceback.format_exc(),
            python_version=sys.version,
        )

    def _save_results(self) -> None:
        """Save validation results to JSON."""
        results_file = self._output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "config_hash": self.config.compute_hash(),
            "results": {
                name: {
                    "status": result.status.value,
                    "passed": result.passed,
                    "duration_seconds": result.duration_seconds,
                    "metrics": result.metrics,
                    "error_message": result.error_message,
                }
                for name, result in self._results.items()
            },
            "summary": {
                "total": len(self._results),
                "passed": sum(1 for r in self._results.values() if r.passed),
                "failed": sum(1 for r in self._results.values() if not r.passed),
            },
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        self._logger.info("results_saved", path=str(results_file))

    def get_summary(self) -> str:
        """Get a human-readable summary of results.

        Returns:
            Summary string.
        """
        if not self._results:
            return "No validations have been run."

        lines = [
            "=" * 60,
            "VALIDATION SUMMARY",
            "=" * 60,
            "",
        ]

        for name, result in self._results.items():
            lines.append(result.summary())
            lines.append("")

        # Overall status
        passed = sum(1 for r in self._results.values() if r.passed)
        total = len(self._results)

        lines.extend([
            "=" * 60,
            f"OVERALL: {passed}/{total} validations passed",
            "=" * 60,
        ])

        return "\n".join(lines)


def run_validation(
    config_path: str | None = None,
    **overrides: Any,
) -> dict[str, ValidationResult]:
    """Convenience function to run validation.

    Args:
        config_path: Optional path to config file.
        **overrides: Config overrides.

    Returns:
        Dictionary of validation results.
    """
    # Load config from file if provided
    if config_path:
        import yaml

        from src.validation.utils import deep_merge

        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        # Use deep_merge to properly merge nested Pydantic model configs
        merged_config = deep_merge(config_data, overrides)
        config = ValidationConfig(**merged_config)
    else:
        config = ValidationConfig(**overrides)

    runner = ValidationRunner(config=config)
    return runner.run_all()
