"""Typed exception hierarchy for AlphaGalerkin.

Every exception carries structured context so that callers can
programmatically inspect *what* went wrong without parsing messages.

Hierarchy
---------
AlphaGalerkinError
 +-- ConfigError
 |    +-- ConfigValidationError
 +-- MCTSError
 |    +-- TreeDepthExceededError
 |    +-- NoValidActionsError
 +-- DiscretizationError          (environment-level, avoids shadowing builtins)
 |    +-- InvariantViolationError
 |    +-- DOFBudgetExceededError
 |    +-- MeshIntegrityError
 +-- PhysicsError
 |    +-- SolverDivergenceError
 |    +-- StabilityViolationError
 +-- TrainingError
      +-- CheckpointError
           +-- CheckpointVersionMismatchError
"""

from __future__ import annotations

from typing import Any

# -----------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------


class AlphaGalerkinError(Exception):
    """Root exception for the AlphaGalerkin framework.

    All framework-specific exceptions inherit from this class so that
    callers can write a single ``except AlphaGalerkinError`` guard.

    Parameters
    ----------
    message:
        Human-readable description.
    context:
        Arbitrary structured data attached to the exception for
        programmatic inspection (e.g. config dicts, element IDs).

    """

    def __init__(
        self,
        message: str = "",
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.context: dict[str, Any] = context or {}
        super().__init__(message)

    def __repr__(self) -> str:
        cls = type(self).__name__
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{cls}({self.args[0]!r}, context={{{ctx}}})"
        return f"{cls}({self.args[0]!r})"


# -----------------------------------------------------------------------
# Configuration errors
# -----------------------------------------------------------------------


class ConfigError(AlphaGalerkinError):
    """Raised when configuration loading or parsing fails.

    Parameters
    ----------
    message:
        What went wrong (e.g. missing file, bad YAML).
    path:
        Filesystem path to the offending config file, if applicable.

    """

    def __init__(
        self,
        message: str = "",
        *,
        path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if path is not None:
            ctx["path"] = path
        super().__init__(message, context=ctx)
        self.path: str | None = path


class ConfigValidationError(ConfigError):
    """Raised when a configuration value fails Pydantic validation.

    Parameters
    ----------
    message:
        Description of the validation failure.
    field:
        Dotted path to the offending field (e.g. ``"mcts.c_puct"``).
    value:
        The rejected value.

    """

    def __init__(
        self,
        message: str = "",
        *,
        field: str | None = None,
        value: Any = None,
        path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if field is not None:
            ctx["field"] = field
        if value is not None:
            ctx["value"] = value
        super().__init__(message, path=path, context=ctx)
        self.field: str | None = field
        self.value: Any = value


# -----------------------------------------------------------------------
# MCTS errors
# -----------------------------------------------------------------------


class MCTSError(AlphaGalerkinError):
    """Base exception for Monte Carlo Tree Search failures."""


class TreeDepthExceededError(MCTSError):
    """Raised when the MCTS tree exceeds its maximum allowed depth.

    Parameters
    ----------
    message:
        Description.
    depth:
        Current tree depth when the error was raised.
    max_depth:
        Configured maximum depth.

    """

    def __init__(
        self,
        message: str = "",
        *,
        depth: int | None = None,
        max_depth: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if depth is not None:
            ctx["depth"] = depth
        if max_depth is not None:
            ctx["max_depth"] = max_depth
        super().__init__(message, context=ctx)
        self.depth: int | None = depth
        self.max_depth: int | None = max_depth


class NoValidActionsError(MCTSError):
    """Raised when no legal actions are available in a non-terminal state.

    Parameters
    ----------
    message:
        Description.
    state_id:
        Identifier of the problematic state, if available.

    """

    def __init__(
        self,
        message: str = "",
        *,
        state_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if state_id is not None:
            ctx["state_id"] = state_id
        super().__init__(message, context=ctx)
        self.state_id: str | None = state_id


# -----------------------------------------------------------------------
# Discretization / environment errors
# -----------------------------------------------------------------------


class DiscretizationError(AlphaGalerkinError):
    """Base for environment-level discretization failures.

    Named ``DiscretizationError`` (not ``EnvironmentError``) to avoid
    shadowing the built-in ``EnvironmentError``.
    """


class InvariantViolationError(DiscretizationError):
    """Raised when a mathematical invariant is violated.

    Examples: negative Jacobian, non-positive-definite stiffness
    matrix, broken partition of unity.

    Parameters
    ----------
    message:
        Description of the violated invariant.
    invariant:
        Name of the invariant (e.g. ``"positive_jacobian"``).

    """

    def __init__(
        self,
        message: str = "",
        *,
        invariant: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if invariant is not None:
            ctx["invariant"] = invariant
        super().__init__(message, context=ctx)
        self.invariant: str | None = invariant


class DOFBudgetExceededError(DiscretizationError):
    """Raised when DOF budget is exhausted.

    Parameters
    ----------
    message:
        Description.
    current_dof:
        Number of DOFs at the time of the violation.
    max_dof:
        Configured DOF budget.

    """

    def __init__(
        self,
        message: str = "",
        *,
        current_dof: int | None = None,
        max_dof: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if current_dof is not None:
            ctx["current_dof"] = current_dof
        if max_dof is not None:
            ctx["max_dof"] = max_dof
        super().__init__(message, context=ctx)
        self.current_dof: int | None = current_dof
        self.max_dof: int | None = max_dof


class MeshIntegrityError(DiscretizationError):
    """Raised when the mesh data structure is inconsistent.

    Examples: dangling half-edges, non-conforming interfaces where
    conformity is required, orphaned nodes.

    Parameters
    ----------
    message:
        Description.
    element_id:
        Identifier of the offending element, if applicable.

    """

    def __init__(
        self,
        message: str = "",
        *,
        element_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if element_id is not None:
            ctx["element_id"] = element_id
        super().__init__(message, context=ctx)
        self.element_id: str | None = element_id


# -----------------------------------------------------------------------
# Physics / solver errors
# -----------------------------------------------------------------------


class PhysicsError(AlphaGalerkinError):
    """Base exception for physics module failures."""


class SolverDivergenceError(PhysicsError):
    """Raised when the PDE solver fails to converge.

    Parameters
    ----------
    message:
        Description.
    iteration:
        Iteration at which divergence was detected.
    residual:
        Residual norm at the point of failure.

    """

    def __init__(
        self,
        message: str = "",
        *,
        iteration: int | None = None,
        residual: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if iteration is not None:
            ctx["iteration"] = iteration
        if residual is not None:
            ctx["residual"] = residual
        super().__init__(message, context=ctx)
        self.iteration: int | None = iteration
        self.residual: float | None = residual


class StabilityViolationError(PhysicsError):
    """Raised when a stability condition (e.g. LBB / inf-sup) is violated.

    Parameters
    ----------
    message:
        Description.
    condition:
        Name of the violated stability condition.
    value:
        The computed stability indicator.
    threshold:
        The minimum acceptable value.

    """

    def __init__(
        self,
        message: str = "",
        *,
        condition: str | None = None,
        value: float | None = None,
        threshold: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if condition is not None:
            ctx["condition"] = condition
        if value is not None:
            ctx["value"] = value
        if threshold is not None:
            ctx["threshold"] = threshold
        super().__init__(message, context=ctx)
        self.condition: str | None = condition
        self.value: float | None = value
        self.threshold: float | None = threshold


# -----------------------------------------------------------------------
# Training errors
# -----------------------------------------------------------------------


class TrainingError(AlphaGalerkinError):
    """Base exception for training pipeline failures."""


class CheckpointError(TrainingError):
    """Raised when checkpoint save / load fails.

    Parameters
    ----------
    message:
        Description.
    checkpoint_path:
        Path to the checkpoint file.

    """

    def __init__(
        self,
        message: str = "",
        *,
        checkpoint_path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if checkpoint_path is not None:
            ctx["checkpoint_path"] = checkpoint_path
        super().__init__(message, context=ctx)
        self.checkpoint_path: str | None = checkpoint_path


class CheckpointVersionMismatchError(CheckpointError):
    """Raised when a checkpoint was written by an incompatible version.

    Parameters
    ----------
    message:
        Description.
    expected_version:
        The version string the loader expects.
    found_version:
        The version string found in the checkpoint.

    """

    def __init__(
        self,
        message: str = "",
        *,
        expected_version: str | None = None,
        found_version: str | None = None,
        checkpoint_path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        if expected_version is not None:
            ctx["expected_version"] = expected_version
        if found_version is not None:
            ctx["found_version"] = found_version
        super().__init__(
            message,
            checkpoint_path=checkpoint_path,
            context=ctx,
        )
        self.expected_version: str | None = expected_version
        self.found_version: str | None = found_version
