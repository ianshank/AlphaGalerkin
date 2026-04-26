"""Direct tests for :func:`make_optional_dependency_stub`.

The helper is consumed by neural-operator solvers when ``torch`` is
unavailable (a configuration we cannot exercise in CI without making
torch optional).  These tests instead drive it directly so that the
generated stub class behaves as documented.
"""

from __future__ import annotations

import pytest

from src.research.baselines import BaseSolver
from src.research.extra_solvers._optional import make_optional_dependency_stub


class TestMakeOptionalDependencyStub:
    def test_stub_subclasses_basesolver(self) -> None:
        stub_cls = make_optional_dependency_stub(
            name="dummy_solver",
            description="Stub for tests",
            dependency="zzz",
            install_hint="pip install zzz",
        )
        assert issubclass(stub_cls, BaseSolver)

    def test_stub_construction_succeeds(self) -> None:
        """Construction must never raise — registry inspection should work."""
        stub_cls = make_optional_dependency_stub(
            name="dummy_solver",
            description="Stub",
            dependency="zzz",
            install_hint="pip install zzz",
        )
        instance = stub_cls("ignored", arbitrary="kw")
        # Construction args are recorded for debugging but do not raise.
        assert instance._args == ("ignored",)
        assert instance._kwargs == {"arbitrary": "kw"}

    def test_solve_raises_with_install_hint(self) -> None:
        stub_cls = make_optional_dependency_stub(
            name="multigrid_dummy",
            description="MG stub",
            dependency="my_pkg",
            install_hint="pip install my_pkg --pre",
        )
        with pytest.raises(ImportError) as excinfo:
            stub_cls().solve(operator=None, n_dof=4)
        msg = str(excinfo.value)
        assert "multigrid_dummy" in msg
        assert "my_pkg" in msg
        assert "pip install my_pkg --pre" in msg

    def test_class_name_is_descriptive(self) -> None:
        stub_cls = make_optional_dependency_stub(
            name="snake_case_name",
            description="Stub",
            dependency="pkg",
            install_hint="pip install pkg",
        )
        # Name strips underscores for a tidy class identifier.
        assert "Missing" in stub_cls.__name__
        assert "SnakeCaseName" in stub_cls.__name__

    def test_description_includes_dep_marker(self) -> None:
        stub_cls = make_optional_dependency_stub(
            name="x",
            description="Linear solver",
            dependency="foolib",
            install_hint="pip install foolib",
        )
        assert "foolib" in stub_cls.description
        assert "Linear solver" in stub_cls.description
