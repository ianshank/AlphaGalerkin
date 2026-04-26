"""Helpers for optional-dependency baseline solvers.

The pattern: each new baseline depends on a third-party library
(``pyamg``, ``neuraloperator``, ...).  We always want to register the
solver name so that ``get_solver(name)`` produces a clear, actionable
error when the dep is missing — rather than mysteriously failing
elsewhere.

:func:`make_optional_dependency_stub` builds a :class:`BaseSolver`
subclass that raises :class:`ImportError` (with install hint) the
moment :meth:`solve` is called.  Every solver module registers the
real class when its dep imports successfully and the stub otherwise.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.research.baselines import BaseSolver, SolverResult

logger = structlog.get_logger(__name__)


def make_optional_dependency_stub(
    name: str,
    description: str,
    dependency: str,
    install_hint: str,
) -> type[BaseSolver]:
    """Build a :class:`BaseSolver` stub class for a missing dependency.

    Args:
        name: Solver name (used in registration and error messages).
        description: Short human-readable description.
        dependency: Name of the missing Python package.
        install_hint: Pip install hint shown in the error (e.g.
            ``"pip install pyamg"``).

    Returns:
        A :class:`BaseSolver` subclass whose ``solve`` raises
        :class:`ImportError` with the install hint.

    """
    _name = name
    _description = description
    _dependency = dependency
    _hint = install_hint

    class _MissingDependencySolver(BaseSolver):
        name = _name
        description = f"{_description} (stub: missing optional dep '{_dependency}')"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Construction must succeed so users can inspect the
            # registry; the failure surfaces only on solve().
            self._args = args
            self._kwargs = kwargs

        def solve(self, *args: Any, **kwargs: Any) -> SolverResult:
            raise ImportError(
                f"Solver '{_name}' requires the optional package "
                f"'{_dependency}'.  Install it with: {_hint}"
            )

    _MissingDependencySolver.__name__ = f"_Missing{name.title().replace('_', '')}Solver"
    return _MissingDependencySolver
