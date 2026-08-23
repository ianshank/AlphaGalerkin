"""Configuration module for AlphaGalerkin."""

from config.board import (
    GTP_LETTERS,
    KOMI_BY_SIZE,
    STAR_POINTS_BY_SIZE,
    BoardRenderConfig,
    BoardSize,
    CoordinateLabelConfig,
    SpaceConfig,
    get_column_letter,
    get_default_space_config,
)
from config.schemas import (
    AlphaGalerkinConfig,
    DomainConfig,
    MCTSConfig,
    OperatorConfig,
    TrainingConfig,
)

__all__ = [
    # Existing schemas
    "AlphaGalerkinConfig",
    "DomainConfig",
    "MCTSConfig",
    "OperatorConfig",
    "TrainingConfig",
    # Board configuration
    "BoardRenderConfig",
    "BoardSize",
    "CoordinateLabelConfig",
    "GTP_LETTERS",
    "KOMI_BY_SIZE",
    "SpaceConfig",
    "STAR_POINTS_BY_SIZE",
    "get_column_letter",
    "get_default_space_config",
]
