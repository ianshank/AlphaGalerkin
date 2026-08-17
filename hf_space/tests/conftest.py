"""Pytest fixtures for HuggingFace Space tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure imports work from test directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.board import (
    BoardRenderConfig,
    CoordinateLabelConfig,
    SpaceConfig,
    get_default_space_config,
)
from src.game_manager import GameManager
from src.rendering.board_renderer import BoardRenderer

from src.tools.gtp import SimpleGoGame


@pytest.fixture
def space_config() -> SpaceConfig:
    """Get default space configuration."""
    return get_default_space_config()


@pytest.fixture
def render_config() -> BoardRenderConfig:
    """Get default render configuration."""
    return BoardRenderConfig()


@pytest.fixture
def label_config() -> CoordinateLabelConfig:
    """Get default coordinate label configuration."""
    return CoordinateLabelConfig()


@pytest.fixture
def renderer(render_config: BoardRenderConfig) -> BoardRenderer:
    """Create board renderer."""
    return BoardRenderer(render_config)


@pytest.fixture
def game_manager(space_config: SpaceConfig) -> GameManager:
    """Create game manager without evaluator."""
    return GameManager(config=space_config, evaluator=None)


@pytest.fixture(params=[9, 13, 19])
def board_size(request: pytest.FixtureRequest) -> int:
    """Parametrized fixture for all supported board sizes."""
    return request.param


@pytest.fixture
def game_9x9() -> SimpleGoGame:
    """Create 9x9 game."""
    return SimpleGoGame(9)


@pytest.fixture
def game_13x13() -> SimpleGoGame:
    """Create 13x13 game."""
    return SimpleGoGame(13)


@pytest.fixture
def game_19x19() -> SimpleGoGame:
    """Create 19x19 game."""
    return SimpleGoGame(19)
