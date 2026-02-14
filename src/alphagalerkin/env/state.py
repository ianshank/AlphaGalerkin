"""Discretization state with mesh + basis assignments.

:class:`DiscretizationState` is the "board" for PDE discretization
games.  It bundles:

* the :class:`~alphagalerkin.env.mesh_graph.MeshGraph` topology,
* per-element :class:`~alphagalerkin.core.types.BasisSpec` assignments,
* optional solution and residual arrays, and
* step counter and arbitrary metadata.

State transitions are functional: :meth:`apply_action` returns a
**new** state object; the original is not mutated.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
import torch

from src.alphagalerkin.core.types import ActionType, BasisSpec, ElementID
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.mesh_graph import MeshGraph

logger = structlog.get_logger("state")


@dataclass
class DiscretizationState:
    """The 'board state' for PDE discretization games.

    Attributes
    ----------
    mesh:
        Current mesh topology.
    basis_assignments:
        Per-element basis specification (polynomial order, family,
        enrichment functions).
    solution:
        Most recent approximate solution vector (may be ``None``
        before the first solve).
    residual:
        Most recent residual vector (may be ``None``).
    step:
        Number of actions applied since the episode started.
    metadata:
        Arbitrary key-value store for solver / scenario data.

    """

    mesh: MeshGraph
    basis_assignments: dict[ElementID, BasisSpec]
    solution: np.ndarray | None = None
    residual: np.ndarray | None = None
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- derived quantities ------------------------------------------

    @property
    def dof_count(self) -> int:
        """Total degrees of freedom (approximate CG sharing).

        For each element the local DOF count depends on the element
        shape and polynomial order *p*:

        * Quad: ``(p+1)^2``
        * Triangle: ``(p+1)(p+2)/2``

        A 30 % reduction factor approximates the node sharing that
        occurs in a conforming continuous-Galerkin assembly.  (Real
        implementations use assembly maps; this heuristic suffices
        for the RL reward signal.)
        """
        cg_sharing_factor = 0.7
        total = 0
        for eid, basis in self.basis_assignments.items():
            p = basis.polynomial_order
            elem = self.mesh.get_element(eid)
            n_verts = len(elem.vertices)
            if n_verts == 4:  # quad
                total += (p + 1) ** 2
            else:  # triangle
                total += (p + 1) * (p + 2) // 2
        return max(1, int(total * cg_sharing_factor))

    # -- construction ------------------------------------------------

    @classmethod
    def from_mesh(
        cls,
        mesh: MeshGraph,
        initial_polynomial_order: int = 1,
        basis_family: str = "lagrange",
    ) -> DiscretizationState:
        """Create an initial state from a mesh with uniform basis.

        Every element receives the same polynomial order and family.
        """
        basis = {
            eid: BasisSpec(
                polynomial_order=initial_polynomial_order,
                basis_family=basis_family,
            )
            for eid in mesh.element_ids
        }
        return cls(mesh=mesh, basis_assignments=basis)

    # -- state transitions -------------------------------------------

    def apply_action(self, action: Action) -> DiscretizationState:
        """Apply *action* and return a **new** state.

        The caller's state object is never mutated.  After any
        topology-changing action the cached ``solution`` and
        ``residual`` are invalidated (set to ``None``).
        """
        new_state = self.clone()
        new_state.step += 1

        if action.action_type == ActionType.NO_OP:
            return new_state

        eid = action.element_id

        if action.action_type == ActionType.H_REFINE:
            old_basis = new_state.basis_assignments.pop(
                eid,
                BasisSpec(polynomial_order=1),
            )
            new_ids = new_state.mesh.h_refine(eid)
            for nid in new_ids:
                new_state.basis_assignments[nid] = BasisSpec(
                    polynomial_order=old_basis.polynomial_order,
                    basis_family=old_basis.basis_family,
                )

        elif action.action_type == ActionType.P_REFINE:
            if eid in new_state.basis_assignments:
                old = new_state.basis_assignments[eid]
                new_state.basis_assignments[eid] = BasisSpec(
                    polynomial_order=old.polynomial_order + 1,
                    basis_family=old.basis_family,
                    enrichment_functions=list(
                        old.enrichment_functions
                    ),
                )

        elif action.action_type == ActionType.P_COARSEN:
            if eid in new_state.basis_assignments:
                old = new_state.basis_assignments[eid]
                new_state.basis_assignments[eid] = BasisSpec(
                    polynomial_order=max(
                        1, old.polynomial_order - 1
                    ),
                    basis_family=old.basis_family,
                    enrichment_functions=list(
                        old.enrichment_functions
                    ),
                )

        elif action.action_type == ActionType.H_COARSEN:
            # H-coarsening requires merging siblings back into
            # a parent -- complex bookkeeping.  Logged and
            # treated as no-op in this version.
            logger.debug(
                "state.h_coarsen", element=str(eid)
            )

        elif action.action_type == ActionType.SWAP_BASIS:
            new_family = action.params.get(
                "basis_family", "lagrange"
            )
            if eid in new_state.basis_assignments:
                old = new_state.basis_assignments[eid]
                new_state.basis_assignments[eid] = BasisSpec(
                    polynomial_order=old.polynomial_order,
                    basis_family=new_family,
                )

        elif action.action_type == ActionType.ADD_ENRICHMENT:
            enrichment_id = action.params.get(
                "enrichment_id", "default"
            )
            if eid in new_state.basis_assignments:
                old = new_state.basis_assignments[eid]
                existing = list(old.enrichment_functions)
                if enrichment_id not in existing:
                    existing.append(enrichment_id)
                new_state.basis_assignments[eid] = BasisSpec(
                    polynomial_order=old.polynomial_order,
                    basis_family=old.basis_family,
                    enrichment_functions=existing,
                )

        # Invalidate stale solver data
        new_state.solution = None
        new_state.residual = None

        return new_state

    # -- cloning -----------------------------------------------------

    def clone(self) -> DiscretizationState:
        """Return a deep copy of this state."""
        return DiscretizationState(
            mesh=self.mesh.clone(),
            basis_assignments={
                k: BasisSpec(
                    polynomial_order=v.polynomial_order,
                    basis_family=v.basis_family,
                    enrichment_functions=list(
                        v.enrichment_functions
                    ),
                )
                for k, v in self.basis_assignments.items()
            },
            solution=(
                self.solution.copy()
                if self.solution is not None
                else None
            ),
            residual=(
                self.residual.copy()
                if self.residual is not None
                else None
            ),
            step=self.step,
            metadata=copy.deepcopy(self.metadata),
        )

    # -- validation --------------------------------------------------

    def validate(self) -> bool:
        """Check state invariants.

        Returns ``True`` iff:

        1. Every active element has a basis assignment.
        2. No orphan basis entries (keys without a mesh element).
        3. ``dof_count > 0``.
        """
        for eid in self.mesh.element_ids:
            if eid not in self.basis_assignments:
                return False
        for eid in self.basis_assignments:
            if eid not in self.mesh.element_ids:
                return False
        if self.dof_count <= 0:
            return False
        return True

    # -- neural-network interface ------------------------------------

    def to_feature_tensor(self) -> torch.Tensor:
        """Convert state to a per-element feature tensor.

        Each row contains eight features:

        ====  =======================
        col   meaning
        ====  =======================
        0     polynomial order
        1     element diameter
        2     refinement level
        3     centroid x
        4     centroid y
        5     neighbour count
        6     residual estimate (0)
        7     solution norm    (0)
        ====  =======================

        Shape: ``(num_elements, 8)``, dtype ``float32``.
        """
        n_features = 8
        features: list[list[float]] = []
        for eid in self.mesh.element_ids:
            elem = self.mesh.get_element(eid)
            basis = self.basis_assignments[eid]
            feat = [
                float(basis.polynomial_order),
                elem.size,
                float(elem.level),
                float(elem.centroid[0]),
                float(elem.centroid[1]),
                float(len(elem.neighbors)),
                0.0,  # residual placeholder
                0.0,  # solution norm placeholder
            ]
            assert len(feat) == n_features
            features.append(feat)
        return torch.tensor(features, dtype=torch.float32)
