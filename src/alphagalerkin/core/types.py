"""Core type definitions for AlphaGalerkin PDE discretization framework.

This module defines all shared types, enums, and type aliases used
throughout the codebase. All enums inherit from (str, Enum) for
JSON/YAML serialization compatibility with Pydantic v2.

Type Aliases:
    ElementID: Unique identifier for mesh elements.
    PolicyDistribution: Mapping from action names to probabilities.
    RewardWeights: Mapping from reward component names to weights.

Enums:
    ActionType: Available discretization actions in the MCTS tree.
    PDEType: Classification of PDE by mathematical character.
    BackupStrategy: Value backup strategies for MCTS tree.
    GNNArchitecture: Graph neural network architecture choices.
    SelectionPolicy: MCTS child-selection policies.
    TemperatureScheduleType: Temperature annealing schedules.
    NormalizationType: Normalization layer variants.
    PoolingType: Graph pooling strategies.
    Formulation: Galerkin formulation variants (CG/DG).

Dataclasses:
    BasisSpec: Specification for a single basis function on an element.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NewType

# ---------------------------------------------------------------------------
# Opaque identifiers
# ---------------------------------------------------------------------------

ElementID = NewType("ElementID", str)
"""Unique identifier for a mesh element (e.g., ``"elem_003"``)."""


# ---------------------------------------------------------------------------
# Enums -- all (str, Enum) for Pydantic v2 serialization
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Available discretization actions in the MCTS action space.

    Each action modifies a single element or the global approximation
    space. ``NO_OP`` is always valid and leaves the state unchanged.
    """

    H_REFINE = "h_refine"
    """Split the element into children (reduce *h*)."""

    H_COARSEN = "h_coarsen"
    """Merge sibling elements back into parent (increase *h*)."""

    P_REFINE = "p_refine"
    """Increase polynomial order on the element."""

    P_COARSEN = "p_coarsen"
    """Decrease polynomial order on the element."""

    SWAP_BASIS = "swap_basis"
    """Replace the basis family (e.g. Lagrange -> Legendre)."""

    ADD_ENRICHMENT = "add_enrichment"
    """Add an enrichment function (e.g. singular, discontinuous)."""

    NO_OP = "no_op"
    """Do nothing -- always a valid action."""

    REFINE_ALL_BOUNDARY = "refine_all_boundary"
    """H-refine every boundary element (fewer than max neighbours)."""

    COARSEN_ALL_INTERIOR = "coarsen_all_interior"
    """P-coarsen every interior element (those with max neighbours)."""

    UNIFORM_P_REFINE = "uniform_p_refine"
    """Increment polynomial order on ALL elements."""


class PDEType(str, Enum):
    """Classification of a PDE by its mathematical character.

    Used by the physics module to select appropriate weak-form
    assembly routines, stability checks, and solver strategies.
    """

    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"
    HYPERBOLIC = "hyperbolic"
    MIXED = "mixed"


class BackupStrategy(str, Enum):
    """Value backup strategy used during MCTS tree propagation.

    Attributes:
        MEAN:  Average child values (standard AlphaZero).
        MAX:   Use the maximum child value (optimistic).
        MIXED: Weighted combination of MEAN and MAX.

    """

    MEAN = "mean"
    MAX = "max"
    MIXED = "mixed"


class GNNArchitecture(str, Enum):
    """Graph neural network architecture for mesh encoding.

    The GNN processes the mesh as a graph where nodes are elements
    and edges represent adjacency / face-sharing.
    """

    GAT = "gat"
    GCN = "gcn"
    GRAPHSAGE = "graphsage"
    CUSTOM = "custom"
    GALERKIN = "galerkin"
    FNET = "fnet"
    GRAPH_MP = "graph_mp"


class SelectionPolicy(str, Enum):
    """MCTS child-selection policy at interior tree nodes."""

    PUCT = "puct"
    """Predictor + UCT (AlphaZero default)."""

    UCB1 = "ucb1"
    """Upper Confidence Bound 1 (classic MCTS)."""

    RAVE = "rave"
    """Rapid Action Value Estimation."""


class TemperatureScheduleType(str, Enum):
    """Temperature annealing schedule for action sampling."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    CONSTANT = "constant"


class NormalizationType(str, Enum):
    """Normalization layer variant for the GNN / policy-value heads."""

    BATCH = "batch"
    LAYER = "layer"
    NONE = "none"


class PoolingType(str, Enum):
    """Graph-level pooling strategy for the value head."""

    MEAN = "mean"
    ATTENTION = "attention"
    MAX = "max"


class Formulation(str, Enum):
    """Galerkin formulation variant.

    CG (Continuous Galerkin) enforces inter-element continuity.
    DG (Discontinuous Galerkin) allows jumps across faces and uses
    numerical fluxes for coupling.
    """

    CONTINUOUS = "continuous"
    DISCONTINUOUS = "discontinuous"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BasisSpec:
    """Specification for the polynomial / enrichment basis on one element.

    Attributes:
        polynomial_order: Polynomial degree *p* (>= 0).
        basis_family: Family name, e.g. ``"lagrange"``, ``"legendre"``.
        enrichment_functions: Additional non-polynomial enrichment
            identifiers (e.g. ``["tip_singular", "heaviside"]``).

    """

    polynomial_order: int
    basis_family: str = "lagrange"
    enrichment_functions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PolicyDistribution = dict[str, float]
"""Mapping from action identifier to probability."""

RewardWeights = dict[str, float]
"""Mapping from reward component name to scalar weight."""
