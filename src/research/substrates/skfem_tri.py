"""Element-local ``RefinementSubstrate`` over a ``scikit-fem`` triangular mesh.

Unlike ``TensorGridSubstrate`` (Slice B), which reproduces the legacy
tensor-product-grid substrate's full-grid-line refinement defect on purpose
(it is the back-compat control), ``SkfemTriSubstrate`` refines only the
marked triangles -- the element-local substrate the charter's own adequacy
claim needs to be measurable at all (`specs/refinement_substrate.spec.md`).

Reuses ``src.research.fem_baseline``'s module-level primitives
(``build_lshaped_initial_mesh``, ``assemble_and_solve``,
``quadrature_l2_error``, ``zz_indicator``) rather than re-implementing mesh
construction/assembly/estimation a second time -- the same solver code the
classical ``ScikitFEMLShapedSolver`` baseline uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.refinement.substrate import SubstrateSolveResult
from src.research.fem_baseline import (
    _make_element,
    _require_skfem,
    assemble_and_solve,
    build_lshaped_initial_mesh,
    quadrature_l2_error,
    zz_indicator,
)
from src.research.marking import dorfler_mark
from src.research.substrates.config import SubstrateConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.pde.operators import PDEOperator


@dataclass(frozen=True)
class SkfemTriMesh:
    """Wraps an opaque ``skfem.MeshTri`` so ``TMesh`` has a stable type."""

    mesh: Any


class SkfemTriSubstrate:
    """``RefinementSubstrate`` over a ``scikit-fem`` triangular mesh (element-local).

    Does not inherit from ``RefinementSubstrate`` -- Protocols are structural.
    """

    def __init__(
        self,
        operator: PDEOperator,
        config: SubstrateConfig | None = None,
    ) -> None:
        self._operator = operator
        self._config = config or SubstrateConfig(name="skfem_tri_substrate", kind="skfem_tri")
        self._skfem = _require_skfem()

    def initial_mesh(self) -> SkfemTriMesh:
        """The coarse L-shaped mesh (three unit squares), per AC8."""
        mesh = build_lshaped_initial_mesh(
            self._operator,
            self._skfem,
            initial_mesh_refinements=self._config.initial_refinements,
        )
        return SkfemTriMesh(mesh=self._maybe_freeze(mesh))

    def solve(self, mesh: SkfemTriMesh) -> SubstrateSolveResult:
        element = _make_element(self._config.element_type, self._skfem)
        basis = self._skfem.Basis(mesh.mesh, element)
        n_dof = basis.N
        n_dof_free = n_dof - len(self._dirichlet_dof_indices(basis))

        u, _coords, nodal_rms = assemble_and_solve(
            mesh.mesh, element, self._operator, self._skfem, self._nodal_rms_l2
        )
        quad_l2 = quadrature_l2_error(mesh.mesh, element, u, self._operator, self._skfem)
        indicators = zz_indicator(mesh.mesh, u)

        l2_error = quad_l2 if self._config.error_metric == "quadrature" else (nodal_rms or 0.0)
        extra = {"l2_error_nodal_rms": float(nodal_rms or 0.0), "l2_error_quadrature": quad_l2}

        return SubstrateSolveResult(
            values=u,
            indicators=indicators,
            l2_error=l2_error,
            n_dof=n_dof,
            n_dof_free=n_dof_free,
            extra=extra,
        )

    @staticmethod
    def _dirichlet_dof_indices(basis: Any) -> NDArray[np.int64]:
        """Mirror ``assemble_and_solve``'s own Dirichlet-dof extraction."""
        dirichlet_dofs = basis.get_dofs()
        if hasattr(dirichlet_dofs, "flatten"):
            return np.asarray(dirichlet_dofs.flatten())
        if hasattr(dirichlet_dofs, "nodal"):
            return np.asarray(dirichlet_dofs.nodal["u"])
        return np.asarray(dirichlet_dofs)

    def _nodal_rms_l2(
        self,
        solution: NDArray[np.float64],
        coords: NDArray[np.float64],
        operator: PDEOperator,
    ) -> float | None:
        """The legacy nodal-RMS metric (``BaseSolver._compute_l2_error``'s formula).

        Injected into ``assemble_and_solve`` so this substrate never imports
        ``src.research.baselines`` (avoiding a needless coupling); the
        formula itself is reproduced verbatim.
        """
        exact = operator.exact_solution(coords.astype(np.float32))
        if isinstance(exact, torch.Tensor):
            exact = exact.detach().cpu().numpy()
        exact = np.asarray(exact, dtype=np.float64)
        diff = solution.flatten() - exact.flatten()
        n = len(diff)
        return float(np.sqrt(np.sum(diff**2) / n)) if n > 0 else None

    def mark(self, indicators: NDArray[np.float64], theta: float) -> NDArray[np.bool_]:
        return dorfler_mark(indicators, theta, variant=self._config.marking_variant)

    def refine(self, mesh: SkfemTriMesh, marked: NDArray[np.bool_]) -> SkfemTriMesh:
        """Element-local refinement: only the marked triangles are split."""
        refined = mesh.mesh.refined(np.where(marked)[0])
        return SkfemTriMesh(mesh=self._maybe_freeze(refined))

    def n_units(self, mesh: SkfemTriMesh) -> int:
        return int(mesh.mesh.t.shape[1])

    def refinable_mask(self, mesh: SkfemTriMesh) -> NDArray[np.bool_]:
        """Every triangle is refinable: the L-shape mesh has no masked-out elements."""
        return np.ones(self.n_units(mesh), dtype=bool)

    def fingerprint(self, mesh: SkfemTriMesh) -> bytes:
        return mesh.mesh.p.tobytes() + b"|" + mesh.mesh.t.tobytes()

    def describe(self) -> dict[str, str | int | float]:
        return {
            "kind": "skfem_tri",
            "dof_convention": "fem_basis_dofs",
            "element_type": self._config.element_type,
        }

    def _maybe_freeze(self, mesh: Any) -> Any:
        """Clear numpy write flags on mesh arrays (AC3).

        ``skfem``'s ``refined()`` does not mutate its input, but
        ``mesh.p.flags.writeable`` is ``True`` by default -- immutability is
        a property of the refinement *API*, not of the array, unless enforced.
        """
        if self._config.enforce_immutable_meshes:
            mesh.p.flags.writeable = False
            mesh.t.flags.writeable = False
        return mesh
