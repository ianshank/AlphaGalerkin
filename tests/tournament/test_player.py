"""Tests for player management."""

from __future__ import annotations

import pytest

from src.tournament.player import Player, PlayerRegistry, PlayerStats


class TestPlayerStats:
    """Tests for PlayerStats dataclass."""

    def test_default_values(self) -> None:
        """Test default statistics values."""
        stats = PlayerStats()
        assert stats.games_played == 0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.draws == 0
        assert stats.win_rate == 0.0
        assert stats.score == 0.0

    def test_win_rate_calculation(self) -> None:
        """Test win rate calculation."""
        stats = PlayerStats(games_played=10, wins=7, losses=2, draws=1)
        assert stats.win_rate == 0.7

    def test_draw_rate_calculation(self) -> None:
        """Test draw rate calculation."""
        stats = PlayerStats(games_played=10, wins=5, losses=3, draws=2)
        assert stats.draw_rate == 0.2

    def test_score_calculation(self) -> None:
        """Test score calculation."""
        stats = PlayerStats(wins=5, draws=2)
        assert stats.score == 6.0  # 5 + 0.5*2

    def test_record_result_win(self) -> None:
        """Test recording a win."""
        stats = PlayerStats()
        stats.record_result(won=True, drawn=False, as_black=True)

        assert stats.games_played == 1
        assert stats.wins == 1
        assert stats.losses == 0
        assert stats.games_as_black == 1
        assert stats.wins_as_black == 1
        assert stats.total_points == 1.0

    def test_record_result_loss(self) -> None:
        """Test recording a loss."""
        stats = PlayerStats()
        stats.record_result(won=False, drawn=False, as_black=False)

        assert stats.games_played == 1
        assert stats.wins == 0
        assert stats.losses == 1
        assert stats.games_as_white == 1
        assert stats.total_points == 0.0

    def test_record_result_draw(self) -> None:
        """Test recording a draw."""
        stats = PlayerStats()
        stats.record_result(won=False, drawn=True, as_black=True)

        assert stats.games_played == 1
        assert stats.draws == 1
        assert stats.total_points == 0.5

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        stats = PlayerStats(games_played=10, wins=5, losses=3, draws=2)
        data = stats.to_dict()

        assert data["games_played"] == 10
        assert data["wins"] == 5
        assert data["losses"] == 3
        assert data["draws"] == 2
        assert data["win_rate"] == 0.5
        assert data["score"] == 6.0


class TestPlayer:
    """Tests for Player dataclass."""

    def test_initialization(self, sample_player: Player) -> None:
        """Test player initialization."""
        assert sample_player.name == "TestPlayer"
        assert sample_player.player_id == "test123"
        assert sample_player.rating == 1500.0
        assert sample_player.is_ai is True

    def test_default_player_id(self) -> None:
        """Test automatic player ID generation."""
        player = Player(name="AutoID")
        assert player.player_id is not None
        assert len(player.player_id) == 8

    def test_games_played_property(self, sample_player: Player) -> None:
        """Test games_played property."""
        assert sample_player.games_played == 0
        sample_player.stats.games_played = 5
        assert sample_player.games_played == 5

    def test_win_rate_property(self, sample_player: Player) -> None:
        """Test win_rate property."""
        assert sample_player.win_rate == 0.0
        sample_player.stats = PlayerStats(games_played=10, wins=7)
        assert sample_player.win_rate == 0.7

    def test_record_result(self, sample_player: Player) -> None:
        """Test recording a game result."""
        sample_player.record_result(
            won=True,
            drawn=False,
            as_black=True,
            rating_change=15.0,
        )

        assert sample_player.stats.wins == 1
        assert sample_player.rating == 1515.0

    def test_to_dict(self, sample_player: Player) -> None:
        """Test serialization to dict."""
        data = sample_player.to_dict()

        assert data["name"] == "TestPlayer"
        assert data["player_id"] == "test123"
        assert data["rating"] == 1500.0
        assert data["is_ai"] is True
        assert "stats" in data

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "name": "LoadedPlayer",
            "player_id": "loaded123",
            "rating": 1800.0,
            "is_ai": False,
            "stats": {
                "games_played": 50,
                "wins": 30,
                "losses": 15,
                "draws": 5,
            },
        }

        player = Player.from_dict(data)

        assert player.name == "LoadedPlayer"
        assert player.player_id == "loaded123"
        assert player.rating == 1800.0
        assert player.is_ai is False
        assert player.stats.games_played == 50

    def test_hash_and_equality(self) -> None:
        """Test hash and equality."""
        p1 = Player(name="Same", player_id="same123")
        p2 = Player(name="Different Name", player_id="same123")
        p3 = Player(name="Same", player_id="different")

        assert hash(p1) == hash(p2)
        assert p1 == p2
        assert p1 != p3


class TestPlayerRegistry:
    """Tests for PlayerRegistry."""

    def test_initialization(self, player_registry: PlayerRegistry) -> None:
        """Test registry initialization."""
        assert len(player_registry) == 0

    def test_register_player(
        self, player_registry: PlayerRegistry, sample_player: Player
    ) -> None:
        """Test player registration."""
        player_registry.register(sample_player)
        assert len(player_registry) == 1
        assert sample_player.player_id in player_registry

    def test_register_duplicate(
        self, player_registry: PlayerRegistry, sample_player: Player
    ) -> None:
        """Test registering duplicate player."""
        player_registry.register(sample_player)
        player_registry.register(sample_player)  # Should not add twice
        assert len(player_registry) == 1

    def test_create_player(self, player_registry: PlayerRegistry) -> None:
        """Test creating and registering a player."""
        player = player_registry.create_player(
            name="New Player",
            rating=1600.0,
            is_ai=True,
            model_path="/path/to/model",
            version="1.0",
        )

        assert player.name == "New Player"
        assert player.rating == 1600.0
        assert player.metadata["version"] == "1.0"
        assert len(player_registry) == 1

    def test_get_by_id(
        self, populated_registry: PlayerRegistry
    ) -> None:
        """Test getting player by ID."""
        player = populated_registry.get("p1")
        assert player is not None
        assert player.name == "Player1"

        assert populated_registry.get("nonexistent") is None

    def test_get_by_name(
        self, populated_registry: PlayerRegistry
    ) -> None:
        """Test getting player by name."""
        player = populated_registry.get_by_name("Player1")
        assert player is not None
        assert player.player_id == "p1"

        # Case insensitive
        player = populated_registry.get_by_name("PLAYER1")
        assert player is not None

        assert populated_registry.get_by_name("nonexistent") is None

    def test_remove_player(
        self, populated_registry: PlayerRegistry
    ) -> None:
        """Test removing a player."""
        assert len(populated_registry) == 4

        result = populated_registry.remove("p1")
        assert result is True
        assert len(populated_registry) == 3
        assert populated_registry.get("p1") is None

        result = populated_registry.remove("nonexistent")
        assert result is False

    def test_list_players(
        self, populated_registry: PlayerRegistry
    ) -> None:
        """Test listing all players."""
        players = populated_registry.list_players()
        assert len(players) == 4

    def test_get_ranked_by_rating(
        self, populated_registry: PlayerRegistry
    ) -> None:
        """Test getting ranked players by rating."""
        ranked = populated_registry.get_ranked(by="rating")
        assert len(ranked) == 4
        assert ranked[0].name == "Player1"  # Highest rating
        assert ranked[-1].name == "Player3"  # Lowest rating

    def test_get_ranked_by_wins(
        self, populated_registry: PlayerRegistry
    ) -> None:
        """Test getting ranked players by wins."""
        # Add some wins
        populated_registry.get("p2").stats.wins = 10
        populated_registry.get("p3").stats.wins = 5

        ranked = populated_registry.get_ranked(by="wins")
        assert ranked[0].player_id == "p2"

    def test_get_ranked_by_score(
        self, populated_registry: PlayerRegistry
    ) -> None:
        """Test getting ranked players by score."""
        # Add some scores
        populated_registry.get("p1").stats.wins = 3
        populated_registry.get("p2").stats.wins = 2
        populated_registry.get("p2").stats.draws = 2

        ranked = populated_registry.get_ranked(by="score")
        assert ranked[0].player_id == "p1"  # 3.0
        assert ranked[1].player_id == "p2"  # 3.0 (ties possible)

    def test_iter_players(
        self, populated_registry: PlayerRegistry
    ) -> None:
        """Test iterating through players."""
        player_ids = [p.player_id for p in populated_registry.iter_players()]
        assert len(player_ids) == 4
        assert "p1" in player_ids

    def test_contains(
        self, populated_registry: PlayerRegistry
    ) -> None:
        """Test contains check."""
        assert "p1" in populated_registry
        assert "nonexistent" not in populated_registry

    def test_to_dict(
        self, populated_registry: PlayerRegistry
    ) -> None:
        """Test serialization to dict."""
        data = populated_registry.to_dict()
        assert "players" in data
        assert len(data["players"]) == 4

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "players": [
                {"name": "P1", "player_id": "id1", "rating": 1500.0},
                {"name": "P2", "player_id": "id2", "rating": 1600.0},
            ]
        }

        registry = PlayerRegistry.from_dict(data)
        assert len(registry) == 2
        assert registry.get("id1") is not None
