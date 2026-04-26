"""E2E tests for configuration validation.

Tests that all configuration files load correctly and Pydantic
validation works as expected.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.e2e
def test_train_yaml_exists(config_dir: Path) -> None:
    """Verify train.yaml exists."""
    config_file = config_dir / "train.yaml"
    assert config_file.exists(), f"Config file not found: {config_file}"


@pytest.mark.e2e
def test_train_fast_yaml_exists(config_dir: Path) -> None:
    """Verify train_fast.yaml exists."""
    config_file = config_dir / "train_fast.yaml"
    assert config_file.exists(), f"Config file not found: {config_file}"


@pytest.mark.e2e
def test_train_yaml_loads() -> None:
    """Verify train.yaml loads without errors."""
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")

    config_path = Path(__file__).parents[2] / "config" / "train.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config is not None
    assert isinstance(config, dict)


@pytest.mark.e2e
def test_train_fast_yaml_loads() -> None:
    """Verify train_fast.yaml loads without errors."""
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")

    config_path = Path(__file__).parents[2] / "config" / "train_fast.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config is not None
    assert isinstance(config, dict)


@pytest.mark.e2e
def test_tournament_config_validation() -> None:
    """Verify tournament config Pydantic validation works."""
    try:
        from pydantic import ValidationError

        from src.tournament.config import TournamentConfig

        # Valid config should work
        config = TournamentConfig(name="test_tournament")
        assert config.name == "test_tournament"

        # Invalid config should raise
        with pytest.raises(ValidationError):
            TournamentConfig(name="")  # Empty name should fail

    except ImportError as e:
        pytest.skip(f"Required module not available: {e}")


@pytest.mark.e2e
def test_match_config_validation() -> None:
    """Verify match config Pydantic validation works."""
    try:
        from pydantic import ValidationError

        from src.tournament.config import MatchConfig

        # Valid config should work
        config = MatchConfig()
        assert config.board_size == 19  # default

        # Invalid board size should fail
        with pytest.raises(ValidationError):
            MatchConfig(board_size=100)  # Too large

    except ImportError as e:
        pytest.skip(f"Required module not available: {e}")


@pytest.mark.e2e
def test_rating_config_validation() -> None:
    """Verify rating config Pydantic validation works."""
    try:
        from pydantic import ValidationError

        from src.tournament.config import RatingConfig

        # Valid config should work
        config = RatingConfig()
        assert config.initial_rating == 1500.0  # default

        # Invalid k_factor should fail
        with pytest.raises(ValidationError):
            RatingConfig(k_factor=-1)  # Negative not allowed

    except ImportError as e:
        pytest.skip(f"Required module not available: {e}")


@pytest.mark.e2e
def test_config_hash_consistency() -> None:
    """Verify config hash is deterministic."""
    try:
        from src.tournament.config import TournamentConfig

        config1 = TournamentConfig(name="test", seed=42)
        config2 = TournamentConfig(name="test", seed=42)

        # Same config should produce same hash
        hash1 = hash(config1.model_dump_json())
        hash2 = hash(config2.model_dump_json())
        assert hash1 == hash2

        # Different config should produce different hash
        config3 = TournamentConfig(name="different", seed=42)
        hash3 = hash(config3.model_dump_json())
        assert hash1 != hash3

    except ImportError as e:
        pytest.skip(f"Required module not available: {e}")


@pytest.mark.e2e
def test_poc_config_loads() -> None:
    """Verify PoC scenario configs load."""
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")

    config_path = Path(__file__).parents[2] / "config" / "scenarios" / "poc_quick.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config is not None
