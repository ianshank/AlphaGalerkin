"""Reward shaping helpers for PDE games.

Currently exposes the proposal-form log reward used by the DOE Genesis
submission narrative::

    R(s, a, s') = -alpha * log L2(s') - beta * log C(s')

Each concrete ``PDEGame`` selects between the legacy linear reward and
this log form via ``PDEGameConfig.reward_form``. The log form is
mathematically equivalent to the linear form up to an affine
reparameterization on each episode (see
``docs/doe_genesis/mdp_specification.md § 2.4``), but exact parity is
needed to reproduce reviewer-run experiments against the written
proposal.
"""

from __future__ import annotations

import math


def log_reward(
    *,
    error: float,
    cost: float,
    alpha: float,
    beta: float,
    epsilon: float,
) -> float:
    """Return the log-form PDE reward.

    Computes ``-alpha * log(max(error, epsilon)) - beta * log(max(cost, epsilon))``.
    The ``epsilon`` floor keeps the logarithm finite when the state
    happens to land at zero error or zero DOF (edge cases reachable on
    trivial initial states).

    Args:
        error: New-state L2 (or surrogate) error estimate.
        cost: New-state cumulative cost measure; typically ``state.dof``.
        alpha: Coefficient on the ``-log(error)`` term.
        beta: Coefficient on the ``-log(cost)`` term.
        epsilon: Floor applied to ``error`` and ``cost`` before taking
            the logarithm. Must be strictly positive.

    Returns:
        Scalar reward value.

    Raises:
        ValueError: If ``epsilon`` is not strictly positive.

    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    safe_error = error if error > epsilon else epsilon
    safe_cost = cost if cost > epsilon else epsilon
    return -alpha * math.log(safe_error) - beta * math.log(safe_cost)
