"""Base classes for validation scenarios.

Provides common functionality for all validation scenarios.
"""

from __future__ import annotations

import sys
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel

from src.validation.config import ValidationResult, ValidationStatus
from src.validation.logging import ValidationLogger, create_validation_logger

# Type variable for config classes
C = TypeVar("C", bound=BaseModel)


class BaseValidator(ABC):
    """Abstract base class for validation scenarios.

    Provides lifecycle management and result collection.
    """

    # Override in subclasses
    name: str = "base"
    config_class: type[BaseModel] = BaseModel

    def __init__(
        self,
        config: BaseModel | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize validator.

        Args:
            config: Validation configuration.
            **kwargs: Override config fields.
        """
        if config is None:
            config = self.config_class(**kwargs)
        elif kwargs:
            from src.validation.utils import deep_merge

            config_dict = config.model_dump()
            # Use deep_merge to properly merge nested configs
            config_dict = deep_merge(config_dict, kwargs)
            config = self.config_class(**config_dict)

        self.config = config
        self._start_time: datetime | None = None
        self._metrics: dict[str, float] = {}
        self._details: dict[str, Any] = {}
        self._artifacts: dict[str, str] = {}
        self._logger = create_validation_logger(self.name, config=config)

    def setup(self) -> None:
        """Pre-validation setup.

        Override in subclasses for initialization.
        """
        pass

    @abstractmethod
    def validate(self) -> ValidationResult:
        """Execute validation logic.

        Must be implemented by subclasses.

        Returns:
            ValidationResult with status and metrics.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """Post-validation cleanup.

        Override in subclasses for cleanup.
        """
        pass

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value.

        Args:
            name: Metric name.
            value: Metric value.
        """
        self._metrics[name] = value
        self._logger.metric(name, value)

    def record_detail(self, key: str, value: Any) -> None:
        """Record additional detail.

        Args:
            key: Detail key.
            value: Detail value.
        """
        self._details[key] = value

    def record_artifact(self, name: str, path: str) -> None:
        """Record an artifact path.

        Args:
            name: Artifact name.
            path: Path to artifact.
        """
        self._artifacts[name] = path
        self._logger.debug("artifact_recorded", artifact=name, path=path)

    def run(self) -> ValidationResult:
        """Execute the full validation lifecycle.

        Returns:
            ValidationResult capturing the outcome.
        """
        self._start_time = datetime.now()
        self._metrics = {}
        self._details = {}
        self._artifacts = {}

        self._logger.info(
            "validation_starting",
            config_hash=self.config.compute_hash() if hasattr(self.config, "compute_hash") else "",
        )

        try:
            self.setup()
            result = self.validate()

            self._logger.info(
                "validation_completed",
                status=result.status.value,
                passed=result.passed,
                duration=result.duration_seconds,
            )

            return result

        except Exception as e:
            end_time = datetime.now()
            assert self._start_time is not None
            duration = (end_time - self._start_time).total_seconds()

            error_result = ValidationResult(
                validation_name=self.name,
                config_hash=self.config.compute_hash() if hasattr(self.config, "compute_hash") else "",
                status=ValidationStatus.ERROR,
                passed=False,
                metrics=self._metrics,
                details=self._details,
                artifacts={k: str(v) for k, v in self._artifacts.items()},
                start_time=self._start_time,
                end_time=end_time,
                duration_seconds=duration,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
                python_version=sys.version,
            )

            self._logger.error(
                "validation_error",
                error=str(e),
                duration=duration,
            )

            return error_result

        finally:
            try:
                self.teardown()
            except Exception as teardown_error:
                self._logger.warning(
                    "teardown_error",
                    error=str(teardown_error),
                )

    def _create_result(
        self,
        status: ValidationStatus,
        passed: bool,
        **extra: Any,
    ) -> ValidationResult:
        """Create a ValidationResult from current state.

        Args:
            status: Validation status.
            passed: Whether validation passed.
            **extra: Additional fields.

        Returns:
            ValidationResult instance.
        """
        end_time = datetime.now()
        assert self._start_time is not None
        duration = (end_time - self._start_time).total_seconds()

        # Get device and GPU info
        device = "cpu"
        gpu_info = None
        try:
            import torch

            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
                gpu_info = torch.cuda.get_device_name()
        except ImportError:
            pass

        return ValidationResult(
            validation_name=self.name,
            config_hash=self.config.compute_hash() if hasattr(self.config, "compute_hash") else "",
            status=status,
            passed=passed,
            metrics=self._metrics,
            details={**self._details, **extra},
            artifacts={k: str(v) for k, v in self._artifacts.items()},
            start_time=self._start_time,
            end_time=end_time,
            duration_seconds=duration,
            device=device,
            python_version=sys.version,
            gpu_info=gpu_info,
        )
