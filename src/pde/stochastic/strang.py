"""Symmetric Strang composition of the projected A, D, and J flows.

One step over ``dt``:

    Φ_dt ≈ Φ^A_{dt/2} ∘ Φ^D_{dt/2} ∘ Φ^J_{dt} ∘ Φ^D_{dt/2} ∘ Φ^A_{dt/2}

The symmetric composition has local error O(dt³) and global error O(dt²)
(AC4). Without a jump term the middle flow is skipped and the composition
degenerates to ``A(dt/2) ∘ D(dt) ∘ A(dt/2)`` — still symmetric.

Spec: specs/stochastic_galerkin_nke.spec.md (AC3, AC4).
"""

from __future__ import annotations

from src.pde.stochastic.errors import JumpModelMissingError
from src.pde.stochastic.gaussian_mixture import GaussianMixtureState
from src.pde.stochastic.generator import JumpSemigroup
from src.pde.stochastic.projection import GalerkinMomentProjection

_TIME_ATOL = 1e-12
"""Absolute slack when comparing propagation time against the horizon."""


class StrangSplitStep:
    """Strang-splitting propagator over a ``GalerkinMomentProjection``."""

    def __init__(
        self,
        projection: GalerkinMomentProjection,
        jump_step: JumpSemigroup | None = None,
    ) -> None:
        """Compose the propagator; defense-in-depth re-check of the AC2 contract.

        Args:
            projection: The projected A+D flows.
            jump_step: Jump-semigroup model; defaults to the generator's own
                ``jump_semigroup`` when unset.

        Raises:
            JumpModelMissingError: The generator has a jump term but no jump
                model is available from either source (defense in depth —
                ``KolmogorovGenerator`` already enforces this at construction).

        """
        generator = projection.generator
        resolved = jump_step if jump_step is not None else generator.jump_semigroup
        if generator.has_jump and resolved is None:
            msg = (
                "Strang composition over a jump generator requires a jump-semigroup "
                "model (MDNJumpSemigroup or AnalyticCompoundPoissonMoments); the jump "
                "component is never silently ignored"
            )
            raise JumpModelMissingError(msg)
        self.projection = projection
        self.jump_step = resolved if generator.has_jump else None

    def step(self, state: GaussianMixtureState, dt: float) -> GaussianMixtureState:
        """One symmetric Strang step over ``dt``."""
        half = 0.5 * dt
        s = self.projection.advection_flow(state, half)
        s = self.projection.diffusion_flow(s, half)
        if self.jump_step is not None:
            s = self.jump_step.apply(s, dt)
        s = self.projection.diffusion_flow(s, half)
        return self.projection.advection_flow(s, half)

    def propagate(self, state: GaussianMixtureState) -> list[tuple[GaussianMixtureState, float]]:
        """Propagate from t=0 to ``config.t_end`` in steps of ``config.dt``.

        The final step is shortened to land exactly on the horizon (matching
        ``TimeStepper.integrate`` semantics). Returns (state, time) pairs
        including the initial state.
        """
        dt = self.projection.config.dt
        t_end = self.projection.config.t_end
        out: list[tuple[GaussianMixtureState, float]] = [(state, 0.0)]
        current = state
        t = 0.0
        while t < t_end - _TIME_ATOL:
            step_dt = min(dt, t_end - t)
            current = self.step(current, step_dt)
            t += step_dt
            out.append((current, t))
        return out
