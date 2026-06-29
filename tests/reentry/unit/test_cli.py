"""Tests for the reentry CLI benchmark dispatch."""

from __future__ import annotations

import pytest

import src.reentry.cli as cli
from src.reentry.cli import (
    _SHOCK_TUBE_ICS,
    _UNIMPLEMENTED_BENCHMARKS,
    main,
    run_shock_tube_benchmark,
    run_sod_benchmark,
)


class TestShockTubeBenchmarks:
    @pytest.mark.parametrize("benchmark", sorted(_SHOCK_TUBE_ICS))
    def test_each_shock_tube_ic_runs(self, benchmark: str) -> None:
        # Small grid / short time keeps this fast; just assert it completes.
        assert run_shock_tube_benchmark(benchmark, n_cells=32, t_final=0.05) is True

    def test_sod_alias(self) -> None:
        assert run_sod_benchmark() is True


class TestUnimplementedBenchmarks:
    @pytest.mark.parametrize("benchmark", sorted(_UNIMPLEMENTED_BENCHMARKS))
    def test_documented_as_unimplemented(self, benchmark: str) -> None:
        # Greenfield benchmarks are honestly catalogued (not silent placeholders).
        assert benchmark in _UNIMPLEMENTED_BENCHMARKS
        assert _UNIMPLEMENTED_BENCHMARKS[benchmark]  # non-empty reason


class TestMainDispatch:
    def test_list(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("sys.argv", ["reentry", "--list"])
        main()
        out = capsys.readouterr().out
        assert "sod_shock_tube" in out

    def test_audit_dispatch(self, monkeypatch) -> None:
        called = {}
        monkeypatch.setattr("sys.argv", ["reentry", "--audit-conservation"])
        monkeypatch.setattr(cli, "run_audit_conservation", lambda: called.setdefault("ok", True))
        main()
        assert called["ok"] is True

    def test_shock_tube_dispatch(self, monkeypatch) -> None:
        seen = {}
        monkeypatch.setattr("sys.argv", ["reentry", "--benchmark", "sod_shock_tube"])
        monkeypatch.setattr(cli, "run_shock_tube_benchmark", lambda b: seen.setdefault("b", b))
        main()
        assert seen["b"] == "sod_shock_tube"

    def test_unimplemented_dispatch_raises(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.argv", ["reentry", "--benchmark", "mach6_cylinder"])
        with pytest.raises(NotImplementedError, match="mach6_cylinder"):
            main()

    def test_no_args_prints_help(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("sys.argv", ["reentry"])
        main()
        assert "usage" in capsys.readouterr().out.lower()
