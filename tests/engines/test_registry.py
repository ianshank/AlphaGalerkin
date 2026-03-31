"""Tests for engine protocol registry.

Tests engine registration, lookup, and factory creation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.engines.config import UCIConfig
from src.engines.registry import EngineRegistry, create_engine


class TestEngineRegistry:
    """Tests for the engine registry."""

    def test_uci_is_registered(self) -> None:
        """UCI engine should be registered on module import."""
        registry = EngineRegistry()
        assert registry.is_registered("uci")

    def test_get_uci_engine_class(self) -> None:
        """Should return the UCIEngine class."""
        from src.engines.uci import UCIEngine

        registry = EngineRegistry()
        engine_cls = registry.get_or_raise("uci")
        assert engine_cls is UCIEngine

    def test_get_unregistered_raises(self) -> None:
        """Looking up unregistered protocol should raise."""
        registry = EngineRegistry()
        with pytest.raises(KeyError):
            registry.get_or_raise("nonexistent_protocol")


class TestCreateEngine:
    """Tests for the create_engine factory function."""

    def test_create_uci_engine(self) -> None:
        """Factory should create a UCIEngine from UCIConfig."""
        from src.engines.uci import UCIEngine

        config = UCIConfig(
            name="test",
            engine_path=Path("/fake/stockfish"),
            depth_limit=10,
        )
        engine = create_engine(config)
        assert isinstance(engine, UCIEngine)

    def test_create_engine_passes_config(self) -> None:
        """Created engine should have the config."""
        config = UCIConfig(
            name="test_sf",
            engine_path=Path("/fake/stockfish"),
            depth_limit=20,
            hash_mb=128,
        )
        engine = create_engine(config)
        assert engine.config.depth_limit == 20
        assert engine.config.hash_mb == 128
