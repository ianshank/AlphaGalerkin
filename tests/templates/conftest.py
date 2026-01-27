"""Pytest configuration and fixtures for template tests."""

from __future__ import annotations

import pytest
from pydantic import Field

from src.templates.config import BaseModuleConfig
from src.templates.registry import BaseRegistry, create_registry
from src.templates.base import BaseExecutable, ExecutionResult, ExecutionStatus


# ============================================================================
# Sample Configuration for Testing
# ============================================================================


class SampleConfig(BaseModuleConfig):
    """Sample configuration for testing."""

    param_int: int = Field(default=100, ge=1, le=1000, description="Integer param")
    param_float: float = Field(default=0.5, gt=0.0, lt=1.0, description="Float param")
    param_list: list[int] = Field(
        default_factory=lambda: [1, 2, 3],
        description="List param",
    )


# ============================================================================
# Sample Base Class for Registry Testing
# ============================================================================


class SampleBase:
    """Sample base class for registry testing."""

    def process(self, data: str) -> str:
        """Process data."""
        raise NotImplementedError


# ============================================================================
# Sample Executable for Testing
# ============================================================================


class SampleExecutable(BaseExecutable[SampleConfig]):
    """Sample executable for testing."""

    _executable_name = "sample"

    def execute(self) -> ExecutionResult:
        """Execute sample logic."""
        # Simulate work
        total = sum(self.config.param_list) * self.config.param_int
        return self._create_result(
            status=ExecutionStatus.COMPLETED,
            metrics={"total": float(total)},
        )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_config() -> SampleConfig:
    """Create a sample configuration."""
    return SampleConfig(name="test_config")


@pytest.fixture
def sample_config_custom() -> SampleConfig:
    """Create a sample configuration with custom values."""
    return SampleConfig(
        name="custom_config",
        description="A custom test configuration",
        param_int=50,
        param_float=0.25,
        param_list=[5, 10, 15],
        seed=123,
    )


@pytest.fixture
def sample_registry():
    """Create and return a fresh sample registry.

    Automatically cleans up after the test.
    """
    SampleRegistry, register_sample = create_registry("Sample", SampleBase)

    yield SampleRegistry, register_sample

    # Cleanup
    SampleRegistry().clear()


@pytest.fixture
def sample_executable(sample_config: SampleConfig) -> SampleExecutable:
    """Create a sample executable."""
    return SampleExecutable(sample_config)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton state before each test.

    This ensures test isolation for registries.
    """
    # Store original instances
    original_instances: dict[type, object] = {}

    yield

    # Reset any registries that were created during the test
    # This is handled by individual test fixtures


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for test artifacts."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir
