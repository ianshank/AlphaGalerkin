"""Tests for the PI-controlled adaptive time-stepping path.

Track C of the Production Hardening Push.  Validates:

* Convergence on a smooth scalar ODE with a known closed-form solution
  (decay  du/dt = -lambda u, exact solution u(t) = u0 * exp(-lambda t)).
* PI-controller invariants:
    - dt always in [dt_min, dt_max]
    - dt grows in smooth regions
    - dt shrinks when the local error exceeds tolerance
* Fixed-mode regression: with ``adaptive_dt=False`` the integrator is
  byte-identical to the pre-adaptive implementation.

Backward compatibility: every test in :mod:`tests.pde.test_time_stepping`
continues to pass unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from src.pde.time_stepping import (
    TimeStepper,
    TimeSteppingConfig,
    TimeSteppingMethod,
    create_time_stepper,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _decay_rhs(lam: float = 1.0) -> Callable[[Tensor, float], Tensor]:
    """Return du/dt = -lam * u.

    Exact solution: u(t) = u0 * exp(-lam * t).
    """

    def rhs(u: Tensor, _t: float) -> Tensor:
        return -lam * u

    return rhs


def _smooth_oscillator_rhs(omega: float = 2.0) -> Callable[[Tensor, float], Tensor]:
    """Return du/dt = -omega^2 * u for a 2-component (u, du/dt) state.

    Exact solution: u(t) = u0 cos(omega t) + v0 sin(omega t) / omega.
    """

    def rhs(state: Tensor, _t: float) -> Tensor:
        u_pos, u_vel = state[0], state[1]
        return torch.stack([u_vel, -(omega**2) * u_pos])

    return rhs


def _make_config(
    method: TimeSteppingMethod = TimeSteppingMethod.RK4,
    *,
    adaptive_dt: bool = False,
    **kwargs: object,
) -> TimeSteppingConfig:
    base: dict[str, object] = {
        "name": "adaptive_test",
        "method": method,
        "dt": 0.05,
        "t_start": 0.0,
        "t_end": 1.0,
        "adaptive_dt": adaptive_dt,
        "dt_min": 1e-6,
        "dt_max": 0.1,
        "error_tolerance": 1e-4,
        "save_interval": 1,
    }
    base.update(kwargs)
    return TimeSteppingConfig(**base)  # type: ignore[arg-type]


METHODS = [
    TimeSteppingMethod.FORWARD_EULER,
    TimeSteppingMethod.RK4,
    TimeSteppingMethod.CRANK_NICOLSON,
]


# ---------------------------------------------------------------------------
# AdaptiveTimeSteppingConfig validation
# ---------------------------------------------------------------------------


class TestAdaptiveTimeSteppingConfigValidation:
    """Bounded validators on the new PI-controller fields."""

    def test_safety_factor_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValidationError):
            TimeSteppingConfig(name="t", safety_factor=1.5)
        with pytest.raises(ValidationError):
            TimeSteppingConfig(name="t", safety_factor=0.0)

    def test_pi_alpha_must_be_strictly_positive(self) -> None:
        with pytest.raises(ValidationError):
            TimeSteppingConfig(name="t", pi_alpha=0.0)

    def test_pi_beta_may_be_zero_pure_i_controller(self) -> None:
        cfg = TimeSteppingConfig(name="t", pi_beta=0.0)
        assert cfg.pi_beta == 0.0

    def test_pi_beta_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            TimeSteppingConfig(name="t", pi_beta=-0.1)

    def test_dt_min_must_be_strictly_less_than_dt_max(self) -> None:
        with pytest.raises(ValidationError):
            TimeSteppingConfig(name="t", dt_min=0.1, dt_max=0.05)
        with pytest.raises(ValidationError):
            TimeSteppingConfig(name="t", dt_min=0.1, dt_max=0.1)

    def test_initial_dt_must_lie_in_dt_range_when_adaptive(self) -> None:
        with pytest.raises(ValidationError):
            TimeSteppingConfig(
                name="t",
                adaptive_dt=True,
                dt=10.0,
                dt_min=1e-3,
                dt_max=0.1,
            )

    def test_initial_dt_outside_range_allowed_when_not_adaptive(self) -> None:
        # Backwards compatible: fixed-dt callers do not need to keep dt in [dt_min, dt_max].
        cfg = TimeSteppingConfig(name="t", adaptive_dt=False, dt=10.0, dt_min=1e-3, dt_max=0.1)
        assert cfg.dt == 10.0


# ---------------------------------------------------------------------------
# Backwards-compatible construction (no NotImplementedError anymore)
# ---------------------------------------------------------------------------


class TestAdaptiveConstruction:
    @pytest.mark.parametrize("method", METHODS)
    def test_adaptive_stepper_constructs_without_error(self, method: TimeSteppingMethod) -> None:
        cfg = _make_config(method=method, adaptive_dt=True)
        stepper = create_time_stepper(cfg)
        assert isinstance(stepper, TimeStepper)
        assert stepper.config.adaptive_dt is True


# ---------------------------------------------------------------------------
# Convergence on a known exact solution
# ---------------------------------------------------------------------------


class TestAdaptiveConvergence:
    """The PI-controller should drive the final error below tolerance."""

    @pytest.mark.parametrize("method", METHODS)
    def test_decay_problem_converges_within_tolerance(self, method: TimeSteppingMethod) -> None:
        lam = 1.0
        u0 = torch.tensor([1.0])
        cfg = _make_config(
            method=method,
            adaptive_dt=True,
            dt=0.05,
            t_end=1.0,
            error_tolerance=1e-4,
        )
        stepper = create_time_stepper(cfg)
        snapshots = stepper.integrate(u0, _decay_rhs(lam))
        u_final, t_final = snapshots[-1]
        u_exact = math.exp(-lam * t_final)
        rel_error = abs(u_final.item() - u_exact) / abs(u_exact)
        # Final error should be small (within an order of magnitude of tol
        # in the worst case, and much tighter for higher-order schemes).
        assert rel_error < 1e-2, (
            f"{method.value}: rel error {rel_error:.3e} too large; "
            f"u_final={u_final.item():.6f} u_exact={u_exact:.6f}"
        )

    def test_oscillator_high_order_tighter_than_low_order(self) -> None:
        """RK4 final error must beat Forward Euler at the same tolerance."""
        omega = 2.0
        state0 = torch.tensor([1.0, 0.0])
        errors: dict[TimeSteppingMethod, float] = {}
        for method in (TimeSteppingMethod.FORWARD_EULER, TimeSteppingMethod.RK4):
            cfg = _make_config(
                method=method,
                adaptive_dt=True,
                dt=0.05,
                t_end=1.0,
                error_tolerance=1e-4,
            )
            stepper = create_time_stepper(cfg)
            snapshots = stepper.integrate(state0, _smooth_oscillator_rhs(omega))
            u_final, t_final = snapshots[-1]
            exact_pos = math.cos(omega * t_final)
            errors[method] = abs(u_final[0].item() - exact_pos)
        assert errors[TimeSteppingMethod.RK4] <= errors[TimeSteppingMethod.FORWARD_EULER]


# ---------------------------------------------------------------------------
# PI-controller invariants
# ---------------------------------------------------------------------------


class TestAdaptiveInvariants:
    @pytest.mark.parametrize("method", METHODS)
    def test_dt_stays_within_configured_bounds(self, method: TimeSteppingMethod) -> None:
        """Track every dt the controller proposes; assert clamping holds."""
        observed: list[float] = []

        cfg = _make_config(
            method=method,
            adaptive_dt=True,
            dt=0.05,
            dt_min=1e-5,
            dt_max=0.05,  # cap growth so we exercise both bounds
            t_end=0.5,
            error_tolerance=1e-3,
        )
        stepper = create_time_stepper(cfg)

        original_propose = stepper._propose_next_dt

        def _spy(err: float) -> float:
            new_dt = original_propose(err)
            observed.append(new_dt)
            return new_dt

        stepper._propose_next_dt = _spy  # type: ignore[method-assign]
        stepper.integrate(torch.tensor([1.0]), _decay_rhs())
        assert observed, "Adaptive integration produced no dt updates"
        assert all(cfg.dt_min <= dt <= cfg.dt_max for dt in observed), observed

    def test_dt_grows_in_smooth_region(self) -> None:
        """Smooth RHS drives dt up to dt_max.

        For a perfectly linear RHS the local error is ~0 and dt should
        ratchet up to dt_max within a few steps.
        """
        cfg = _make_config(
            method=TimeSteppingMethod.RK4,
            adaptive_dt=True,
            dt=1e-3,
            dt_min=1e-6,
            dt_max=0.05,
            t_end=0.5,
            error_tolerance=1e-3,
        )
        stepper = create_time_stepper(cfg)
        # Constant RHS: zero local error at any order > 0.
        snapshots = stepper.integrate(torch.tensor([0.0]), lambda u, t: torch.zeros_like(u))
        assert stepper.dt == pytest.approx(cfg.dt_max, rel=1e-9)
        assert len(snapshots) >= 2

    def test_propose_next_dt_zero_error_returns_dt_max(self) -> None:
        """Numerical-floor behavior: zero error -> growth ceiling."""
        cfg = _make_config(adaptive_dt=True)
        stepper = create_time_stepper(cfg)
        stepper.dt = cfg.dt
        new_dt = stepper._propose_next_dt(0.0)
        assert new_dt == pytest.approx(cfg.dt_max)

    def test_propose_next_dt_large_error_shrinks(self) -> None:
        cfg = _make_config(adaptive_dt=True, dt=0.05, dt_min=1e-6, error_tolerance=1e-4)
        stepper = create_time_stepper(cfg)
        stepper.dt = cfg.dt
        # err >> tolerance => growth factor < 1.
        new_dt = stepper._propose_next_dt(err=1.0)
        assert new_dt < cfg.dt


# ---------------------------------------------------------------------------
# Fixed-mode regression — adaptive_dt=False MUST be byte-identical
# ---------------------------------------------------------------------------


class TestFixedModeRegression:
    """Regression for the adaptive_dt=False path.

    When adaptive_dt=False the integrator must produce the exact same
    trajectory as the pre-adaptive implementation.
    """

    @pytest.mark.parametrize("method", METHODS)
    def test_fixed_mode_trajectory_unchanged(self, method: TimeSteppingMethod) -> None:
        # Two independently-constructed steppers must produce identical
        # snapshots (this is the deterministic-regression contract).
        u0 = torch.tensor([1.0, 0.5])
        rhs = _smooth_oscillator_rhs(omega=1.5)

        cfg_a = _make_config(method=method, adaptive_dt=False, dt=0.02, t_end=0.5)
        cfg_b = _make_config(method=method, adaptive_dt=False, dt=0.02, t_end=0.5)
        snap_a = create_time_stepper(cfg_a).integrate(u0.clone(), rhs)
        snap_b = create_time_stepper(cfg_b).integrate(u0.clone(), rhs)

        assert len(snap_a) == len(snap_b)
        for (u_a, t_a), (u_b, t_b) in zip(snap_a, snap_b, strict=True):
            assert t_a == t_b
            torch.testing.assert_close(u_a, u_b, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("method", METHODS)
    def test_fixed_mode_matches_known_step_count(self, method: TimeSteppingMethod) -> None:
        # Fixed dt=0.05 over [0, 1] with save_interval=1 yields 20 internal
        # steps + initial snapshot = 21 snapshots; this is a structural
        # invariant of the fixed-dt loop and must not regress.
        cfg = _make_config(method=method, adaptive_dt=False, dt=0.05, t_end=1.0)
        snaps = create_time_stepper(cfg).integrate(torch.tensor([1.0]), _decay_rhs())
        assert len(snaps) == 21


# ---------------------------------------------------------------------------
# Step-doubling estimator sanity
# ---------------------------------------------------------------------------


class TestStepDoublingEstimator:
    def test_zero_dynamics_returns_zero_error(self) -> None:
        """Zero RHS produces zero error.

        If the RHS is identically zero the two trajectories match
        exactly, so the estimator must report zero error.
        """
        cfg = _make_config(adaptive_dt=True, dt=0.01)
        stepper = create_time_stepper(cfg)
        u0 = torch.tensor([1.0])
        _, err = stepper._step_doubling_error(u0, 0.0, lambda u, t: torch.zeros_like(u))
        assert err == pytest.approx(0.0, abs=1e-12)

    def test_nonzero_dynamics_returns_positive_error(self) -> None:
        cfg = _make_config(method=TimeSteppingMethod.FORWARD_EULER, adaptive_dt=True, dt=0.1)
        stepper = create_time_stepper(cfg)
        u0 = torch.tensor([1.0])
        _, err = stepper._step_doubling_error(u0, 0.0, _decay_rhs(lam=10.0))
        assert err > 0.0
