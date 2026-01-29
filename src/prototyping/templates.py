"""Experiment templates for rapid prototyping.

Provides pre-built experiment templates for common
scenarios in AlphaGalerkin research.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from src.prototyping.builder import ModelBuilder, PrototypeModel
from src.prototyping.config import (
    PresetType,
    PrototypeConfig,
    create_prototype_config,
    create_quick_train_config,
)
from src.prototyping.data import DataGenerator
from src.prototyping.evaluator import EvalResult, QuickEvaluator
from src.prototyping.trainer import QuickTrainer, TrainResult

logger = structlog.get_logger(__name__)


@dataclass
class TemplateResult:
    """Result from running an experiment template.

    Attributes:
        template_id: Unique identifier.
        template_name: Template name.
        model: Built model.
        train_result: Training result.
        eval_results: Evaluation results.
        duration_seconds: Total duration.
        metadata: Additional metadata.

    """

    template_id: str
    template_name: str
    model: PrototypeModel | None = None
    train_result: TrainResult | None = None
    eval_results: list[EvalResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "model_id": self.model.model_id if self.model else None,
            "train_result": self.train_result.to_dict() if self.train_result else None,
            "eval_results": [r.to_dict() for r in self.eval_results],
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate result summary."""
        lines = [
            f"Template Result: {self.template_name}",
            f"ID: {self.template_id}",
            f"Duration: {self.duration_seconds:.2f}s",
        ]

        if self.model:
            lines.append(f"Model: {self.model.model_id}")

        if self.train_result:
            lines.append(f"Training: {self.train_result.n_epochs} epochs")
            lines.append(f"  Final Loss: {self.train_result.final_loss:.6f}")

        if self.eval_results:
            lines.append(f"Evaluations: {len(self.eval_results)}")
            for result in self.eval_results:
                for name, metric in result.metrics.items():
                    lines.append(f"  {name}: {metric.value:.6f}")

        return "\n".join(lines)


class ExperimentTemplate:
    """Base class for experiment templates.

    Templates provide pre-configured experiment workflows
    for common research patterns.
    """

    def __init__(
        self,
        name: str = "base",
        description: str = "",
    ) -> None:
        """Initialize template.

        Args:
            name: Template name.
            description: Template description.

        """
        self.name = name
        self.description = description
        self._logger = logger.bind(template=name)

    def run(self, **kwargs: Any) -> TemplateResult:
        """Run the experiment template.

        Args:
            **kwargs: Template-specific arguments.

        Returns:
            Template result.

        """
        raise NotImplementedError("Subclasses must implement run()")


class TransferTemplate(ExperimentTemplate):
    """Template for zero-shot transfer experiments.

    Tests model's ability to generalize across board sizes.
    """

    def __init__(self) -> None:
        super().__init__(
            name="transfer",
            description="Zero-shot transfer across board sizes",
        )

    def run(
        self,
        source_size: int = 9,
        target_sizes: list[int] | None = None,
        n_train_samples: int = 1000,
        n_eval_samples: int = 500,
        preset: PresetType = PresetType.TRANSFER,
        model_factory: Callable[[PrototypeConfig], Any] | None = None,
        train_fn: Callable[[Any, Any], float] | None = None,
        predict_fn: Callable[[Any, Any], Any] | None = None,
        **kwargs: Any,
    ) -> TemplateResult:
        """Run transfer experiment.

        Args:
            source_size: Training board size.
            target_sizes: Evaluation board sizes.
            n_train_samples: Training samples.
            n_eval_samples: Evaluation samples.
            preset: Model preset.
            model_factory: Custom model factory.
            train_fn: Custom training function.
            predict_fn: Custom prediction function.
            **kwargs: Additional arguments.

        Returns:
            Experiment result.

        """
        import time
        start_time = time.time()

        target_sizes = target_sizes or [9, 13, 19]

        self._logger.info(
            "starting_transfer",
            source=source_size,
            targets=target_sizes,
        )

        # Build model
        builder = ModelBuilder(model_factory=model_factory)
        model = builder.build_from_preset(
            preset=preset,
            name=f"transfer_{source_size}",
            board_sizes=[source_size],
        )

        # Generate training data
        generator = DataGenerator(seed=kwargs.get("seed", 42))
        train_data = generator.generate(
            "board",
            n_samples=n_train_samples,
            board_size=source_size,
        )

        # Default functions
        if train_fn is None:
            def train_fn(m: Any, batch: Any) -> float:
                # Mock training
                return 0.1 * (1 - len(batch) / n_train_samples)

        if predict_fn is None:
            def predict_fn(m: Any, inp: Any) -> Any:
                # Mock prediction
                return [0.0] * len(inp)

        # Train
        trainer = QuickTrainer(
            config=create_quick_train_config(preset=preset),
        )
        train_result = trainer.train(
            model=model,
            train_fn=train_fn,
            data_iterator=lambda: train_data.batch(32),
        )

        # Evaluate on each target size
        evaluator = QuickEvaluator()
        eval_results = []

        for target in target_sizes:
            eval_data = generator.generate(
                "board",
                n_samples=n_eval_samples,
                board_size=target,
            )

            result = evaluator.evaluate(
                model=model,
                predict_fn=predict_fn,
                data=list(eval_data),
            )
            result.metadata["target_size"] = target
            eval_results.append(result)

            self._logger.info(
                "evaluated_target",
                target=target,
                mse=result.get_metric("mse"),
            )

        duration = time.time() - start_time

        return TemplateResult(
            template_id=str(uuid.uuid4())[:8],
            template_name=self.name,
            model=model,
            train_result=train_result,
            eval_results=eval_results,
            duration_seconds=duration,
            metadata={
                "source_size": source_size,
                "target_sizes": target_sizes,
            },
        )


class AblationTemplate(ExperimentTemplate):
    """Template for ablation studies.

    Systematically varies model components to measure impact.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ablation",
            description="Systematic component ablation study",
        )

    def run(
        self,
        param_name: str = "d_model",
        param_values: list[Any] | None = None,
        n_samples: int = 500,
        preset: PresetType = PresetType.SMALL,
        model_factory: Callable[[PrototypeConfig], Any] | None = None,
        train_fn: Callable[[Any, Any], float] | None = None,
        predict_fn: Callable[[Any, Any], Any] | None = None,
        **kwargs: Any,
    ) -> TemplateResult:
        """Run ablation study.

        Args:
            param_name: Parameter to vary.
            param_values: Values to try.
            n_samples: Samples per run.
            preset: Base model preset.
            model_factory: Custom model factory.
            train_fn: Custom training function.
            predict_fn: Custom prediction function.
            **kwargs: Additional arguments.

        Returns:
            Experiment result.

        """
        import time
        start_time = time.time()

        param_values = param_values or [32, 64, 128]

        self._logger.info(
            "starting_ablation",
            param=param_name,
            values=param_values,
        )

        # Generate data
        generator = DataGenerator(seed=kwargs.get("seed", 42))
        data = generator.generate("linear", n_samples=n_samples)
        train_data, eval_data = data.split(train_ratio=0.8)

        # Default functions
        if train_fn is None:
            def train_fn(m: Any, batch: Any) -> float:
                return 0.1

        if predict_fn is None:
            def predict_fn(m: Any, inp: Any) -> Any:
                return [0.0]

        # Build and train models
        builder = ModelBuilder(model_factory=model_factory)
        trainer = QuickTrainer(
            config=create_quick_train_config(preset=PresetType.MINIMAL),
        )
        evaluator = QuickEvaluator()

        eval_results = []

        for value in param_values:
            model = builder.build(
                config=create_prototype_config(
                    name=f"ablation_{param_name}_{value}",
                    preset=preset,
                    **{param_name: value},
                ),
            )

            # Train
            trainer.train(
                model=model,
                train_fn=train_fn,
                data_iterator=lambda: train_data.batch(32),
            )

            # Evaluate
            result = evaluator.evaluate(
                model=model,
                predict_fn=predict_fn,
                data=list(eval_data),
            )
            result.metadata["param_name"] = param_name
            result.metadata["param_value"] = value
            eval_results.append(result)

            self._logger.info(
                "ablation_complete",
                param=param_name,
                value=value,
                mse=result.get_metric("mse"),
            )

        duration = time.time() - start_time

        return TemplateResult(
            template_id=str(uuid.uuid4())[:8],
            template_name=self.name,
            eval_results=eval_results,
            duration_seconds=duration,
            metadata={
                "param_name": param_name,
                "param_values": param_values,
            },
        )


class BenchmarkTemplate(ExperimentTemplate):
    """Template for performance benchmarking.

    Measures inference speed and throughput.
    """

    def __init__(self) -> None:
        super().__init__(
            name="benchmark",
            description="Performance benchmarking",
        )

    def run(
        self,
        board_sizes: list[int] | None = None,
        batch_sizes: list[int] | None = None,
        n_iterations: int = 100,
        preset: PresetType = PresetType.BENCHMARK,
        model_factory: Callable[[PrototypeConfig], Any] | None = None,
        **kwargs: Any,
    ) -> TemplateResult:
        """Run benchmark.

        Args:
            board_sizes: Board sizes to benchmark.
            batch_sizes: Batch sizes to benchmark.
            n_iterations: Iterations per benchmark.
            preset: Model preset.
            model_factory: Custom model factory.
            **kwargs: Additional arguments.

        Returns:
            Experiment result.

        """
        import time
        start_time = time.time()

        board_sizes = board_sizes or [9, 13, 19]
        batch_sizes = batch_sizes or [1, 32, 64]

        self._logger.info(
            "starting_benchmark",
            board_sizes=board_sizes,
            batch_sizes=batch_sizes,
        )

        # Build model
        builder = ModelBuilder(model_factory=model_factory)
        model = builder.build_from_preset(
            preset=preset,
            name="benchmark",
            board_sizes=board_sizes,
        )

        # Benchmark results
        results: dict[str, Any] = {
            "board_sizes": board_sizes,
            "batch_sizes": batch_sizes,
            "timings": {},
            "throughput": {},
        }

        generator = DataGenerator(seed=42)

        for board_size in board_sizes:
            results["timings"][board_size] = {}
            results["throughput"][board_size] = {}

            for batch_size in batch_sizes:
                # Generate batch
                data = generator.generate(
                    "board",
                    n_samples=batch_size,
                    board_size=board_size,
                )

                # Benchmark
                times = []
                for _ in range(n_iterations):
                    batch_start = time.time()
                    # Mock inference
                    for inp, _ in data:
                        _ = [0.0] * len(inp)
                    batch_time = time.time() - batch_start
                    times.append(batch_time)

                avg_time = sum(times) / len(times) if times else 0.0
                throughput = batch_size / avg_time if avg_time > 0 else float("inf")

                results["timings"][board_size][batch_size] = avg_time * 1000  # ms
                results["throughput"][board_size][batch_size] = throughput

                self._logger.info(
                    "benchmark_result",
                    board_size=board_size,
                    batch_size=batch_size,
                    time_ms=avg_time * 1000,
                    throughput=throughput,
                )

        duration = time.time() - start_time

        return TemplateResult(
            template_id=str(uuid.uuid4())[:8],
            template_name=self.name,
            model=model,
            duration_seconds=duration,
            metadata=results,
        )


class TemplateRegistry:
    """Registry for experiment templates.

    Provides discovery and instantiation of templates.
    """

    _instance: TemplateRegistry | None = None
    _templates: dict[str, type[ExperimentTemplate]] = {}

    def __new__(cls) -> TemplateRegistry:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Register built-in templates
            cls._templates = {
                "transfer": TransferTemplate,
                "ablation": AblationTemplate,
                "benchmark": BenchmarkTemplate,
            }
        return cls._instance

    def register(
        self,
        name: str,
        template_cls: type[ExperimentTemplate],
    ) -> None:
        """Register a template.

        Args:
            name: Template name.
            template_cls: Template class.

        """
        self._templates[name] = template_cls
        logger.info("registered_template", name=name)

    def get(self, name: str) -> ExperimentTemplate | None:
        """Get a template by name.

        Args:
            name: Template name.

        Returns:
            Template instance or None.

        """
        template_cls = self._templates.get(name)
        if template_cls:
            return template_cls()
        return None

    def list_templates(self) -> list[str]:
        """List all registered templates."""
        return list(self._templates.keys())

    def get_info(self, name: str) -> dict[str, str] | None:
        """Get template information.

        Args:
            name: Template name.

        Returns:
            Template info or None.

        """
        template = self.get(name)
        if template:
            return {
                "name": template.name,
                "description": template.description,
            }
        return None


def create_template(name: str) -> ExperimentTemplate | None:
    """Create a template by name.

    Args:
        name: Template name.

    Returns:
        Template instance or None.

    """
    return TemplateRegistry().get(name)


def register_template(name: str) -> Callable[[type[ExperimentTemplate]], type[ExperimentTemplate]]:
    """Decorator to register a template.

    Args:
        name: Template name.

    Returns:
        Decorator function.

    """
    def decorator(cls: type[ExperimentTemplate]) -> type[ExperimentTemplate]:
        TemplateRegistry().register(name, cls)
        return cls
    return decorator
