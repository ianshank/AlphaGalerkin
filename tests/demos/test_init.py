"""Tests for demos package __init__.py lazy imports."""

from __future__ import annotations

import pytest

import src.demos as demos_pkg


class TestDemosLazyImport:
    """Test lazy import mechanism in demos __init__."""

    def test_physics_demo_lazy(self) -> None:
        cls = demos_pkg.PhysicsDemo
        assert cls is not None

    def test_transfer_result_lazy(self) -> None:
        cls = demos_pkg.TransferResult
        assert cls is not None

    def test_benchmark_demo_lazy(self) -> None:
        cls = demos_pkg.BenchmarkDemo
        assert cls is not None

    def test_benchmark_result_lazy(self) -> None:
        cls = demos_pkg.BenchmarkResult
        assert cls is not None

    def test_benchmark_suite_lazy(self) -> None:
        cls = demos_pkg.BenchmarkSuite
        assert cls is not None

    def test_architecture_demo_lazy(self) -> None:
        cls = demos_pkg.ArchitectureDemo
        assert cls is not None

    def test_unknown_attr_raises(self) -> None:
        with pytest.raises(AttributeError, match="has no attribute"):
            demos_pkg.NonExistentClass  # noqa: B018

    def test_config_always_available(self) -> None:
        assert demos_pkg.DemoConfig is not None
        assert demos_pkg.ColorScheme is not None

    def test_visualization_always_available(self) -> None:
        assert demos_pkg.PlotResult is not None
        assert demos_pkg.figure_to_image is not None
