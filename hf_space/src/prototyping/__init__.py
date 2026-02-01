"""Fast prototyping tools for AlphaGalerkin.

This module provides utilities for rapid experimentation
and quick iteration on ideas.
"""

from __future__ import annotations

from src.prototyping.builder import (
    ModelBuilder,
    PrototypeModel,
    create_model_builder,
)
from src.prototyping.config import (
    PresetType,
    PrototypeConfig,
    QuickEvalConfig,
    QuickTrainConfig,
    create_prototype_config,
    create_quick_train_config,
)
from src.prototyping.data import (
    DataGenerator,
    SyntheticData,
    create_data_generator,
)
from src.prototyping.evaluator import (
    EvalResult,
    QuickEvaluator,
    create_quick_evaluator,
)
from src.prototyping.templates import (
    ExperimentTemplate,
    TemplateRegistry,
    create_template,
)
from src.prototyping.trainer import (
    QuickTrainer,
    TrainResult,
    create_quick_trainer,
)
from src.prototyping.visualizer import (
    PlotType,
    Visualizer,
    create_visualizer,
)

__all__ = [
    # Config
    "PrototypeConfig",
    "QuickTrainConfig",
    "QuickEvalConfig",
    "PresetType",
    "create_prototype_config",
    "create_quick_train_config",
    # Builder
    "ModelBuilder",
    "PrototypeModel",
    "create_model_builder",
    # Trainer
    "QuickTrainer",
    "TrainResult",
    "create_quick_trainer",
    # Evaluator
    "QuickEvaluator",
    "EvalResult",
    "create_quick_evaluator",
    # Data
    "DataGenerator",
    "SyntheticData",
    "create_data_generator",
    # Visualizer
    "Visualizer",
    "PlotType",
    "create_visualizer",
    # Templates
    "ExperimentTemplate",
    "TemplateRegistry",
    "create_template",
]
