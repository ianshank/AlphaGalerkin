"""Tests for the firefighting CLI benchmark dispatch."""

from __future__ import annotations

import pytest

import src.firefighting.cli as cli
from src.firefighting.cli import main, run_edge_profile, run_grass_fire_benchmark


class TestBenchmarkFunctions:
    def test_grass_fire_small_grid(self) -> None:
        # Parameterized (no hardcoded 50x50): small/fast grid exercises the
        # real solver path and returns True.
        assert run_grass_fire_benchmark(n=16, horizon_s=15.0) is True

    def test_edge_profile_runs(self) -> None:
        # Cheap (no solver); n_cycles parameterized.
        run_edge_profile(max_memory_mb=2048, n_cycles=3)


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

    def test_list_dispatch_only_runs_list(self, monkeypatch, capsys) -> None:
        # --list takes precedence and returns before any benchmark dispatch.
        monkeypatch.setattr("sys.argv", ["fire", "--list", "--benchmark", "fds_comparison"])
        main()
        assert "fds_comparison" in capsys.readouterr().out

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
