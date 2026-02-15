"""Discretization environment for AlphaGalerkin.

This package implements the "game board" for PDE discretization:
mesh topology, element state, discretization actions, reward
composition, and a Gym-like step/reset API.

Public API
----------
MeshGraph, Element
    Mesh topology and element data structures.
Action
    Immutable discretization action on a single element.
DiscretizationState
    The full "board state": mesh + basis assignments.
DiscretizationEnvironment, StepResult
    Gym-like environment wrapping state transitions and rewards.
RewardComposer
    Configurable multi-component reward aggregation.
check_all_invariants
    Validates state consistency (DOF, coverage, orphans).
"""
from __future__ import annotations

from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.env.environment import (
    DiscretizationEnvironment,
    StepResult,
)
from src.alphagalerkin.env.invariants import check_all_invariants
from src.alphagalerkin.env.mesh_graph import Element, MeshGraph
from src.alphagalerkin.env.rewards import RewardComposer
from src.alphagalerkin.env.state import DiscretizationState

__all__ = [
    # Mesh topology
    "MeshGraph",
    "Element",
    # Actions
    "Action",
    # State
    "DiscretizationState",
    # Environment
    "DiscretizationEnvironment",
    "StepResult",
    # Rewards
    "RewardComposer",
    # Invariants
    "check_all_invariants",
]
