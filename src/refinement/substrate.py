"""Domain-free contract for element-local (or grid-local) refinement substrates.

``RefinementSubstrate`` is the interface a concrete mesh/grid representation
implements so that marking, refinement, and error estimation can be driven
generically — by a ``RefinementGame``, by the adequacy gate that compares two
substrates' rate of convergence, or by a future MCTS-vs-classical arena —
without that caller needing to know whether the underlying representation is
a tensor-product grid (``src.research.substrates.tensor_grid``) or a
``scikit-fem`` triangular mesh (``src.research.substrates.skfem_tri``).

Per ``openspec/changes/element-local-substrate/design.md``, this module
imports **numpy only**: no scipy, no torch, no skfem. ``src/pde/games/__init__.py``
documents a real SIGSEGV caused by rippling the torch import graph into
unrelated coverage gates under the C tracer — keeping this layer import-light
is what makes it safe to depend on from anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

TMesh = TypeVar("TMesh")


@dataclass(frozen=True)
class SubstrateSolveResult:
    """Outcome of solving on one mesh/grid state.

    Attributes:
        values: Solution field at the mesh's degrees of freedom.
        indicators: Per-unit (element/cell) error indicators, flat, with
            ``len(indicators) == n_units(mesh)``.
        l2_error: Mesh-independent (quadrature) L2 error against the exact
            solution — the primary, comparison-safe error metric (see
            design.md's "Two error metrics, not one").
        n_dof: The declared comparison axis (the DOF count two substrates or
            two refinement policies are matched against).
        n_dof_free: Unknowns actually solved for (may be fewer than ``n_dof``
            once Dirichlet boundary DOFs are eliminated).
        extra: Auxiliary metrics that must not replace ``l2_error`` — e.g.
            ``l2_error_nodal_rms``, the biased metric ``l2_error`` exists to
            avoid.

    """

    values: NDArray[np.float64]
    indicators: NDArray[np.float64]
    l2_error: float
    n_dof: int
    n_dof_free: int
    extra: Mapping[str, float]

    def __post_init__(self) -> None:
        """Enforce AC5's invariant: free unknowns cannot exceed the declared DOF count."""
        if self.n_dof < 0:
            raise ValueError(f"n_dof must be >= 0, got {self.n_dof}")
        if self.n_dof_free < 0:
            raise ValueError(f"n_dof_free must be >= 0, got {self.n_dof_free}")
        if self.n_dof_free > self.n_dof:
            raise ValueError(
                f"n_dof_free ({self.n_dof_free}) must be <= n_dof ({self.n_dof}) -- AC5"
            )


@runtime_checkable
class RefinementSubstrate(Protocol[TMesh]):
    """A mesh/grid representation with solve, mark, and refine primitives.

    Every member here is meant to have a real caller — ``scripts.audit_abstractions``
    fails the build on a ``Protocol`` member with no reader (the F1 defect
    class), so this Protocol is meant to grow only alongside the concrete
    substrate and caller that need the new member. Seven of the eight members
    (``initial_mesh``, ``solve``, ``mark``, ``refine``, ``n_units``,
    ``refinable_mask``, ``describe``) are read by the sweep driver in
    ``src/research/substrates/sweep.py`` and are audited live since Slice D —
    with the honest caveat that the driver is entered only from the
    adequacy-gate test (``tests/research/test_amr_arena_interpretability.py``),
    so its ``src/`` call sites are one indirection from test-only. The one
    remaining intentional, disclosed exception is ``fingerprint``: its only
    consumer, the fingerprint-keyed solve cache, lands with
    element-local-substrate Slice E (task 7.1), so it alone is exempted via
    the audit's ``_STAGED_FOR_UPCOMING_TASK`` allowlist rather than genuinely
    read yet. That exemption is self-expiring, not documentary:
    ``tests/scripts/test_audit_abstractions.py`` fails once a staged member
    gains a reader, and this docstring is pinned to the allowlist by the
    same file, so neither can silently outlive the other.

    ``@runtime_checkable`` is what lets ``src.templates.registry.create_registry``'s
    ``issubclass(cls, RefinementSubstrate)`` structural check work when
    registering a concrete substrate — Protocols are structural, so a concrete
    substrate need not (and should not) inherit from this class explicitly.
    """

    def initial_mesh(self) -> TMesh:
        """Return the coarsest mesh/grid this substrate starts refinement from."""
        ...

    def solve(self, mesh: TMesh) -> SubstrateSolveResult:
        """Solve the substrate's PDE on ``mesh`` and return the result."""
        ...

    def mark(self, indicators: NDArray[np.float64], theta: float) -> NDArray[np.bool_]:
        """Select which units to refine next, given their error indicators."""
        ...

    def refine(self, mesh: TMesh, marked: NDArray[np.bool_]) -> TMesh:
        """Return a new, more refined mesh/grid; must not mutate ``mesh``."""
        ...

    def n_units(self, mesh: TMesh) -> int:
        """Number of markable units (elements/cells) in ``mesh``."""
        ...

    def refinable_mask(self, mesh: TMesh) -> NDArray[np.bool_]:
        """Which units in ``mesh`` are eligible for refinement."""
        ...

    def fingerprint(self, mesh: TMesh) -> bytes:
        """A stable, hashable identity for ``mesh`` (for memoising solves)."""
        ...

    def describe(self) -> dict[str, str | int | float]:
        """Human/log-readable metadata about this substrate's configuration."""
        ...
