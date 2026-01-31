"""Tests for the scenario registry.

Validates:
    - Scenario registration via decorator
    - Scenario discovery and instantiation
    - Base scenario lifecycle
"""

from __future__ import annotations


import pytest

from src.poc.config import (
    BaseScenarioConfig,
    ScenarioResult,
    ScenarioStatus,
)
from src.poc.registry import BaseScenario, ScenarioRegistry, scenario


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Clean registry before each test."""
    ScenarioRegistry().clear()


class TestScenarioRegistry:
    """Tests for ScenarioRegistry."""

    def test_singleton(self) -> None:
        """Test registry is a singleton."""
        reg1 = ScenarioRegistry()
        reg2 = ScenarioRegistry()
        assert reg1 is reg2

    def test_register_scenario(self) -> None:
        """Test scenario registration."""
        registry = ScenarioRegistry()

        class TestScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        registry.register("test_scenario", TestScenario)

        assert "test_scenario" in registry.list_scenarios()
        assert registry.get("test_scenario") is TestScenario

    def test_duplicate_registration_fails(self) -> None:
        """Test that duplicate names raise errors."""
        registry = ScenarioRegistry()

        class TestScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        registry.register("duplicate", TestScenario)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("duplicate", TestScenario)

    def test_get_nonexistent(self) -> None:
        """Test getting non-existent scenario returns None."""
        registry = ScenarioRegistry()
        assert registry.get("nonexistent") is None

    def test_clear_registry(self) -> None:
        """Test clearing the registry."""
        registry = ScenarioRegistry()

        class TestScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        registry.register("test", TestScenario)
        assert len(registry.list_scenarios()) > 0

        registry.clear()
        assert len(registry.list_scenarios()) == 0


class TestScenarioDecorator:
    """Tests for @scenario decorator."""

    def test_decorator_registers(self) -> None:
        """Test that decorator registers the scenario."""
        registry = ScenarioRegistry()

        @scenario("decorated_test")
        class DecoratedScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        assert registry.get("decorated_test") is DecoratedScenario

    def test_decorator_sets_name(self) -> None:
        """Test that decorator sets _scenario_name."""

        @scenario("named_scenario")
        class NamedScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        assert NamedScenario._scenario_name == "named_scenario"


class TestBaseScenario:
    """Tests for BaseScenario base class."""

    def test_default_config(self) -> None:
        """Test scenario with default config."""

        class SimpleScenario(BaseScenario):
            config_class = BaseScenarioConfig

            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        scenario_instance = SimpleScenario(name="simple", description="Simple test")
        assert scenario_instance.config.name == "simple"

    def test_config_override(self) -> None:
        """Test config override via kwargs."""

        class ConfigurableScenario(BaseScenario):
            config_class = BaseScenarioConfig

            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        scenario_instance = ConfigurableScenario(
            name="config_test",
            description="Test",
            seed=123,
            timeout_seconds=60,
        )

        assert scenario_instance.config.seed == 123
        assert scenario_instance.config.timeout_seconds == 60

    def test_record_metric(self) -> None:
        """Test metric recording."""

        class MetricScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                self.record_metric("test_metric", 0.123)
                return self._create_result(ScenarioStatus.PASSED)

        scenario_instance = MetricScenario(name="metric", description="Test")
        result = scenario_instance.run()

        assert "test_metric" in result.metrics
        assert result.metrics["test_metric"] == 0.123

    def test_record_artifact(self) -> None:
        """Test artifact recording."""

        class ArtifactScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                self.record_artifact("model", "/path/to/model.pt")
                return self._create_result(ScenarioStatus.PASSED)

        scenario_instance = ArtifactScenario(name="artifact", description="Test")
        result = scenario_instance.run()

        assert "model" in result.artifacts
        assert result.artifacts["model"] == "/path/to/model.pt"

    def test_lifecycle_called(self) -> None:
        """Test that setup and teardown are called."""
        call_order: list[str] = []

        class LifecycleScenario(BaseScenario):
            def setup(self) -> None:
                call_order.append("setup")

            def execute(self) -> ScenarioResult:
                call_order.append("execute")
                return self._create_result(ScenarioStatus.PASSED)

            def teardown(self) -> None:
                call_order.append("teardown")

        scenario_instance = LifecycleScenario(name="lifecycle", description="Test")
        scenario_instance.run()

        assert call_order == ["setup", "execute", "teardown"]

    def test_teardown_called_on_error(self) -> None:
        """Test that teardown is called even on error."""
        teardown_called = False

        class ErrorScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                raise ValueError("Test error")

            def teardown(self) -> None:
                nonlocal teardown_called
                teardown_called = True

        scenario_instance = ErrorScenario(name="error", description="Test")
        result = scenario_instance.run()

        assert teardown_called
        assert result.status == ScenarioStatus.ERROR
        assert "Test error" in (result.error_message or "")

    def test_threshold_evaluation(self) -> None:
        """Test threshold evaluation."""
        from src.poc.config import MetricThreshold

        class ThresholdScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                self.record_metric("accuracy", 0.95)
                self.record_metric("loss", 0.05)
                return self._create_result(ScenarioStatus.RUNNING)

        config = BaseScenarioConfig(
            name="threshold",
            description="Test",
            thresholds=[
                MetricThreshold(name="accuracy", operator=">", value=0.9),
                MetricThreshold(name="loss", operator="<", value=0.1),
            ],
        )

        scenario_instance = ThresholdScenario(config=config)
        result = scenario_instance.run()

        assert result.threshold_results["accuracy"] is True
        assert result.threshold_results["loss"] is True
        assert result.passed is True

    def test_threshold_failure(self) -> None:
        """Test that threshold failure sets passed=False."""
        from src.poc.config import MetricThreshold

        class FailingScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                self.record_metric("accuracy", 0.5)  # Below threshold
                return self._create_result(ScenarioStatus.RUNNING)

        config = BaseScenarioConfig(
            name="failing",
            description="Test",
            thresholds=[
                MetricThreshold(name="accuracy", operator=">", value=0.9),
            ],
        )

        scenario_instance = FailingScenario(config=config)
        result = scenario_instance.run()

        assert result.threshold_results["accuracy"] is False
        assert result.passed is False
        assert result.status == ScenarioStatus.FAILED

    def test_duration_recorded(self) -> None:
        """Test that duration is recorded."""
        import time

        class SlowScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                time.sleep(0.1)
                return self._create_result(ScenarioStatus.PASSED)

        scenario_instance = SlowScenario(name="slow", description="Test")
        result = scenario_instance.run()

        assert result.duration_seconds >= 0.1

    def test_environment_captured(self) -> None:
        """Test that environment info is captured."""

        class EnvScenario(BaseScenario):
            def execute(self) -> ScenarioResult:
                return self._create_result(ScenarioStatus.PASSED)

        scenario_instance = EnvScenario(name="env", description="Test")
        result = scenario_instance.run()

        assert result.python_version != ""
        assert result.torch_version != ""
        assert result.device in ("cpu", "cuda")
