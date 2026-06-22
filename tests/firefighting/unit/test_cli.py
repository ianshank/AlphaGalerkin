"""Tests for the firefighting CLI benchmark dispatch."""

from __future__ import annotations

import pytest

import src.firefighting.cli as cli
from src.firefighting.cli import main


class TestMainDispatch:
    def test_list(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("sys.argv", ["fire", "--list"])
        main()
        out = capsys.readouterr().out
        assert "grass_fire_50x50" in out
        assert "transfer_50_to_500" in out

    def test_grass_fire_dispatch(self, monkeypatch) -> None:
        called = {}
        monkeypatch.setattr("sys.argv", ["fire", "--benchmark", "grass_fire_50x50"])
        monkeypatch.setattr(cli, "run_grass_fire_benchmark", lambda: called.setdefault("ok", True))
        main()
        assert called["ok"] is True

    def test_transfer_dispatch(self, monkeypatch) -> None:
        called = {}
        monkeypatch.setattr("sys.argv", ["fire", "--benchmark", "transfer_50_to_500"])
        monkeypatch.setattr(cli, "run_transfer_benchmark", lambda: called.setdefault("ok", True))
        main()
        assert called["ok"] is True

    def test_fds_comparison_raises(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.argv", ["fire", "--benchmark", "fds_comparison"])
        with pytest.raises(NotImplementedError, match="FDS reference"):
            main()

    def test_profile_dispatch(self, monkeypatch) -> None:
        called = {}
        monkeypatch.setattr("sys.argv", ["fire", "--profile", "--max-memory", "2048"])
        monkeypatch.setattr(cli, "run_edge_profile", lambda mb: called.setdefault("mb", mb))
        main()
        assert called["mb"] == 2048

    def test_no_args_prints_help(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("sys.argv", ["fire"])
        main()
        assert "usage" in capsys.readouterr().out.lower()
