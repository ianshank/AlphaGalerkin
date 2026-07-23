"""Typed exceptions for the stochastic Galerkin operator-splitting layer.

Spec: specs/stochastic_galerkin_nke.spec.md (AC2).
"""

from __future__ import annotations


class StochasticConfigurationError(ValueError):
    """A stochastic-layer component was configured inconsistently.

    Raised for contract violations that Pydantic field bounds cannot express,
    e.g. cross-object shape mismatches or a trainer with nothing trainable.
    """


class JumpModelMissingError(StochasticConfigurationError):
    """A generator has a jump term but no jump-semigroup model was supplied.

    The change doc requires this to be a hard configuration error: the jump
    component must never be silently ignored (AC2).
    """
