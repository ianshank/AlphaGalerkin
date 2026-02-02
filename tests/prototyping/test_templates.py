"""Tests for experiment templates."""

from __future__ import annotations

from typing import Any

from src.prototyping.templates import (
    AblationTemplate,
    BenchmarkTemplate,
    ExperimentTemplate,
    TemplateRegistry,
    TemplateResult,
    TransferTemplate,
    create_template,
    register_template,
)


class TestTemplateResult:
    """Tests for TemplateResult dataclass."""

    def test_initialization(self) -> None:
        """Test result initialization."""
        result = TemplateResult(
            template_id="tmpl123",
            template_name="test",
        )

        assert result.template_id == "tmpl123"
        assert result.template_name == "test"
        assert result.model is None
        assert result.train_result is None

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = TemplateResult(
            template_id="tmpl123",
            template_name="test",
            duration_seconds=10.5,
        )

        data = result.to_dict()

        assert data["template_id"] == "tmpl123"
        assert data["duration_seconds"] == 10.5

    def test_summary(self) -> None:
        """Test summary generation."""
        result = TemplateResult(
            template_id="tmpl123",
            template_name="test",
            duration_seconds=10.5,
        )

        summary = result.summary()

        assert "test" in summary
        assert "tmpl123" in summary
        assert "10.5" in summary


class TestTransferTemplate:
    """Tests for TransferTemplate."""

    def test_initialization(self) -> None:
        """Test template initialization."""
        template = TransferTemplate()

        assert template.name == "transfer"
        assert "zero-shot" in template.description.lower()

    def test_run_basic(self) -> None:
        """Test basic template run."""
        template = TransferTemplate()

        result = template.run(
            source_size=9,
            target_sizes=[9],
            n_train_samples=10,
            n_eval_samples=5,
            seed=42,
        )

        assert result is not None
        assert result.template_name == "transfer"
        assert result.model is not None
        assert result.train_result is not None
        assert len(result.eval_results) == 1

    def test_run_multiple_targets(self) -> None:
        """Test with multiple target sizes."""
        template = TransferTemplate()

        result = template.run(
            source_size=9,
            target_sizes=[9, 13],
            n_train_samples=10,
            n_eval_samples=5,
        )

        assert len(result.eval_results) == 2

    def test_run_with_custom_functions(self) -> None:
        """Test with custom functions."""
        template = TransferTemplate()

        def train_fn(m: Any, batch: Any) -> float:
            return 0.05

        def predict_fn(m: Any, inp: Any) -> list[float]:
            return [0.0] * len(inp)

        result = template.run(
            source_size=9,
            target_sizes=[9],
            n_train_samples=10,
            n_eval_samples=5,
            train_fn=train_fn,
            predict_fn=predict_fn,
        )

        assert result is not None


class TestAblationTemplate:
    """Tests for AblationTemplate."""

    def test_initialization(self) -> None:
        """Test template initialization."""
        template = AblationTemplate()

        assert template.name == "ablation"

    def test_run_basic(self) -> None:
        """Test basic ablation run."""
        template = AblationTemplate()

        result = template.run(
            param_name="d_model",
            param_values=[32, 64],
            n_samples=20,
        )

        assert result is not None
        assert result.template_name == "ablation"
        assert len(result.eval_results) == 2

    def test_run_different_param(self) -> None:
        """Test ablating different parameter."""
        template = AblationTemplate()

        result = template.run(
            param_name="n_layers",
            param_values=[1, 2],
            n_samples=20,
        )

        assert len(result.eval_results) == 2


class TestBenchmarkTemplate:
    """Tests for BenchmarkTemplate."""

    def test_initialization(self) -> None:
        """Test template initialization."""
        template = BenchmarkTemplate()

        assert template.name == "benchmark"

    def test_run_basic(self) -> None:
        """Test basic benchmark run."""
        template = BenchmarkTemplate()

        result = template.run(
            board_sizes=[9],
            batch_sizes=[1],
            n_iterations=10,
        )

        assert result is not None
        assert result.template_name == "benchmark"
        assert "timings" in result.metadata
        assert "throughput" in result.metadata

    def test_run_multiple_sizes(self) -> None:
        """Test with multiple board sizes."""
        template = BenchmarkTemplate()

        result = template.run(
            board_sizes=[9, 13],
            batch_sizes=[1, 8],
            n_iterations=5,
        )

        assert 9 in result.metadata["timings"]
        assert 13 in result.metadata["timings"]


class TestTemplateRegistry:
    """Tests for TemplateRegistry."""

    def test_singleton(self) -> None:
        """Test singleton pattern."""
        registry1 = TemplateRegistry()
        registry2 = TemplateRegistry()

        assert registry1 is registry2

    def test_list_templates(self, template_registry: TemplateRegistry) -> None:
        """Test listing templates."""
        templates = template_registry.list_templates()

        assert "transfer" in templates
        assert "ablation" in templates
        assert "benchmark" in templates

    def test_get_template(self, template_registry: TemplateRegistry) -> None:
        """Test getting template by name."""
        template = template_registry.get("transfer")

        assert template is not None
        assert template.name == "transfer"

    def test_get_unknown(self, template_registry: TemplateRegistry) -> None:
        """Test getting unknown template."""
        template = template_registry.get("nonexistent")
        assert template is None

    def test_register(self, template_registry: TemplateRegistry) -> None:
        """Test registering template."""

        class CustomTemplate(ExperimentTemplate):
            def __init__(self) -> None:
                super().__init__("custom", "Custom template")

            def run(self, **kwargs: Any) -> TemplateResult:
                return TemplateResult(
                    template_id="custom123",
                    template_name=self.name,
                )

        template_registry.register("custom", CustomTemplate)

        assert "custom" in template_registry.list_templates()

    def test_get_info(self, template_registry: TemplateRegistry) -> None:
        """Test getting template info."""
        info = template_registry.get_info("transfer")

        assert info is not None
        assert info["name"] == "transfer"
        assert "description" in info


class TestCreateTemplate:
    """Tests for create_template factory."""

    def test_create_transfer(self) -> None:
        """Test creating transfer template."""
        template = create_template("transfer")

        assert template is not None
        assert template.name == "transfer"

    def test_create_ablation(self) -> None:
        """Test creating ablation template."""
        template = create_template("ablation")

        assert template is not None
        assert template.name == "ablation"

    def test_create_unknown(self) -> None:
        """Test creating unknown template."""
        template = create_template("nonexistent")
        assert template is None


class TestRegisterTemplateDecorator:
    """Tests for register_template decorator."""

    def test_decorator(self) -> None:
        """Test decorator registration."""

        @register_template("decorated")
        class DecoratedTemplate(ExperimentTemplate):
            def __init__(self) -> None:
                super().__init__("decorated", "Decorated template")

            def run(self, **kwargs: Any) -> TemplateResult:
                return TemplateResult(
                    template_id="dec123",
                    template_name=self.name,
                )

        registry = TemplateRegistry()
        assert "decorated" in registry.list_templates()

        template = registry.get("decorated")
        assert template is not None
