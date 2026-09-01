"""Burgers equation operator: u_t + u.grad(u) = nu*grad^2(u)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from scipy.special import ive
from torch import Tensor

from src.pde.config import PDEConfig, PDEType
from src.pde.operators.base import PDEOperator, PDEResidual

logger = structlog.get_logger(__name__)


COLE_HOPF_N_TERMS: int = 50
"""Minimum number of Fourier-Bessel terms retained by
``BurgersOperator.exact_solution``'s Cole-Hopf series (shared verbatim by the
torch and numpy branches).

This is a floor, not the actual count: the coefficient envelope
``I_n(R)/I_0(R) ~ exp(-n^2/(2R))`` with ``R = 1/(2*pi*nu)`` widens as the
viscosity shrinks, so :func:`_cole_hopf_coefficients` adds terms as needed.
50 terms already truncate below float64 round-off for ``nu >= 0.01``."""

COLE_HOPF_MAX_TERMS: int = 4096
"""Hard cap on the adaptive Cole-Hopf term count.

Bounds both work and memory: the series is evaluated as an
``(n_points, n_terms)`` float64 array. The cap only binds for ``nu < 1e-6``,
three orders of magnitude below :data:`COLE_HOPF_MIN_RESOLVED_VISCOSITY`,
where the series has already lost all significance to cancellation."""

COLE_HOPF_TERM_TOLERANCE: float = 1e-17
"""Relative coefficient magnitude at which the Cole-Hopf series is truncated.

Terms are retained while ``I_n(R)/I_0(R) ~ exp(-n^2/(2R)) >= tolerance``,
i.e. up to ``n = sqrt(2*R*ln(1/tolerance))``. Chosen just below float64
machine epsilon so that truncation is never the dominant error term."""

COLE_HOPF_CLAMP_EPS: float = 1e-14
"""Strict-positivity floor applied to the Cole-Hopf denominator ``phi`` before
forming ``u = -2*nu*phi_x/phi`` in ``BurgersOperator.exact_solution``.

The exponentially-scaled coefficients satisfy ``c_0 + sum |c_n| == 1``
exactly (``e^-R * (I_0(R) + 2*sum I_n(R)) == 1``), so ``phi`` is bounded above
by 1 and this *absolute* floor is simultaneously a *relative* floor at ~45 ULP
of the series' own l1 norm -- it engages only where ``phi`` has been consumed
by cancellation and carries no significant digits.

Analytically ``phi > 0`` everywhere (it is ``exp`` of a real number smoothed by
the heat kernel); the floor exists purely to keep the *computed* value positive
once ``nu < COLE_HOPF_MIN_RESOLVED_VISCOSITY`` pushes ``phi`` into float64
round-off (measured raw minimum -4.9e-16 at ``nu=0.001``). Lowering it does not
buy accuracy, it only lets meaningless round-off reach the denominator:
measured over a 401-point grid at ``nu=0.001, t=0``, a floor of 1e-30 restores
the historical ~4e13 blow-up while 1e-14 keeps ``max|u| = 0.59``. Raising it
would clip physically meaningful values (``min phi = 1.5e-14`` at ``nu=0.01``).
Guarded by ``tests/pde/test_operators.py::TestBurgersColeHopf::
test_denominator_is_strictly_positive_across_viscosities``."""

COLE_HOPF_MIN_RESOLVED_VISCOSITY: float = 0.01
"""Smallest viscosity at which the float64 Cole-Hopf series resolves ``u``
across the *whole* domain.

``phi`` spans ``[exp(-2R), 1]`` with ``R = 1/(2*pi*nu)``, while the round-off of
the summation is ~machine epsilon of its unit l1 norm. Significance is
therefore lost wherever ``phi`` falls under that floor, which first happens at
``x = 0`` (where ``phi = exp(-2R)``) once ``exp(-2R) ~ 1e-16``, i.e.
``nu ~ 0.009``.

**The degraded region is not a neighbourhood of the origin -- it is a
left-hand interval whose width grows rapidly as the viscosity falls, and it
covers most of the domain well before ``nu`` reaches 1e-3.** Since
``phi(x, 0) = exp(-R*(1 + cos(pi*x)))`` (exponentially scaled), the value
drops below the :data:`COLE_HOPF_CLAMP_EPS` floor for every ``x`` left of::

    x_c(nu) = arccos(2*pi*nu*ln(1/COLE_HOPF_CLAMP_EPS) - 1) / pi

(and ``x_c = 0``, i.e. no degraded region, once the argument exceeds 1 at
``nu >= 1/(pi*ln(1/COLE_HOPF_CLAMP_EPS)) = 0.00987`` -- which is what sets
this constant, rounded up to 0.01). Left of ``x_c`` the clamp pins ``u`` to
~0 while the true solution is O(1), so the *absolute* error there is as large
as the solution itself; right of it the series is accurate to float64.

Measured ``max |u(x,0) + sin(pi*x)|`` on a 401-point grid (float64 internals,
before the float32 cast), with the region satisfying ``err > 1e-3``::

    nu      degraded region     fraction of domain   max err   x_c(nu)
    1.0     none                             0.0%    6e-16      0
    0.1     none                             0.0%    3e-15      0
    0.01    none                             0.0%    9.6e-4     0
    0.009   x <= 0.23                       20.2%    0.33       0.19
    0.005   x <= 0.52                       50.4%    0.98       0.50
    0.001   x <= 0.80                       79.6%    1.0        0.79

(``nu = 0.01`` sits exactly on the boundary: on a finer 4001-point grid its
max error is 1.2e-3, confined to ``x <= 0.10``.)

``t = 0`` is the worst case: diffusion lifts ``phi``'s minimum, so the
degraded interval shrinks with time (at ``nu = 0.005`` it spans ``x <~ 0.5``
at ``t = 0`` and has vanished by ``t = 1``, checked against a dps=300 mpmath
reference).

This is intrinsic to the Fourier-Bessel representation rather than to this
implementation -- ``ive`` itself is only accurate to float64 relative
precision, so no summation scheme can recover the lost digits. Below this
threshold :func:`_cole_hopf_coefficients` emits a ``cole_hopf_underresolved``
warning (once per distinct viscosity, via the cache)."""

COLE_HOPF_COEFFICIENT_CACHE_SIZE: int = 32
"""Number of distinct viscosities for which :func:`_cole_hopf_coefficients`
retains its (time-independent) Fourier-Bessel coefficients. Operators are
typically constructed with a handful of viscosities, so a small cache removes
the ``ive`` evaluation from every ``exact_solution`` call."""


@lru_cache(maxsize=COLE_HOPF_COEFFICIENT_CACHE_SIZE)
def _cole_hopf_coefficients(
    viscosity: float,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Fourier-Bessel coefficients of the Cole-Hopf potential for 1D Burgers.

    The Cole-Hopf transformation ``u = -2*nu*phi_x/phi`` maps
    ``u_t + u*u_x = nu*u_xx`` to the heat equation ``phi_t = nu*phi_xx``. For the
    standard benchmark initial condition ``u(x, 0) = -sin(pi*x)`` on ``[0, 1]``::

        phi(x, 0) = exp(-1/(2*nu) * int_0^x u(s, 0) ds)
                  = exp((1 - cos(pi*x)) / (2*pi*nu))
                  = e^R * exp(-R*cos(pi*x)),      R = 1 / (2*pi*nu)

    and the modified-Bessel generating function
    ``exp(z*cos(theta)) = I_0(z) + 2*sum_n I_n(z)*cos(n*theta)`` evaluated at
    ``z = -R`` (using ``I_n(-R) = (-1)^n * I_n(R)``) gives::

        phi(x, 0) / e^R = I_0(R) + 2*sum_n (-1)^n * I_n(R) * cos(n*pi*x)

    The Dirichlet condition ``u(0, t) = u(1, t) = 0`` is the Neumann condition
    ``phi_x = 0`` on the potential, under which every cosine mode decays as
    ``exp(-n^2*pi^2*nu*t)``. The coefficients are therefore time-independent,
    which is why they can be cached here and reused for every ``t``.

    The alternating ``(-1)^n`` is load-bearing and was verified numerically:
    dropping it produces the Cole-Hopf image of ``u(x, 0) = +sin(pi*x)``, and
    setting every coefficient to 1 -- the historical defect -- produces the
    image of a Dirac comb (a valid solution to the wrong problem, whose
    truncated denominator is negative over half the domain).

    ``scipy.special.ive`` (exponentially scaled, ``ive(n, R) = I_n(R)*e^-R``) is
    used rather than ``iv`` because ``iv`` overflows for small viscosity
    (``iv(0, R) = 4.2e+67`` already at ``nu = 0.001``). The common ``e^-R``
    factor cancels between ``phi_x`` and ``phi``, so ``u`` is unchanged, and the
    scaled coefficients gain the exact normalisation ``c_0 + sum |c_n| == 1``.

    Args:
        viscosity: Kinematic viscosity ``nu``; must be strictly positive.

    Returns:
        ``(c0, n, c)`` with ``c0 = ive(0, R)``, ``n = [1, ..., n_terms]`` and
        ``c[k] = 2*(-1)^(k+1)*ive(k+1, R)``, all float64. The arrays are cached
        and shared; callers must treat them as read-only.

    Raises:
        ValueError: If ``viscosity`` is not strictly positive.

    """
    if viscosity <= 0.0:
        raise ValueError(
            f"Cole-Hopf requires a strictly positive viscosity, got {viscosity}",
        )

    r = 1.0 / (2.0 * np.pi * viscosity)
    n_needed = int(np.ceil(np.sqrt(2.0 * r * np.log(1.0 / COLE_HOPF_TERM_TOLERANCE)))) + 1
    n_terms = min(max(n_needed, COLE_HOPF_N_TERMS), COLE_HOPF_MAX_TERMS)

    if viscosity < COLE_HOPF_MIN_RESOLVED_VISCOSITY:
        logger.warning(
            "cole_hopf_underresolved",
            viscosity=viscosity,
            min_resolved_viscosity=COLE_HOPF_MIN_RESOLVED_VISCOSITY,
            n_terms=n_terms,
            reason=(
                "phi spans [exp(-2R), 1] and loses float64 significance over "
                "the whole interval x < arccos(2*pi*nu*ln(1/clamp_eps) - 1)/pi "
                "(~50% of the domain at nu=0.005, ~80% at nu=0.001), not just "
                "near x=0"
            ),
        )

    n = np.arange(1, n_terms + 1, dtype=np.float64)
    coefficients: NDArray[np.float64] = 2.0 * ((-1.0) ** n) * ive(n, r)
    return float(ive(0.0, r)), n, coefficients


class BurgersOperator(PDEOperator):
    """Burgers equation operator: u_t + u·∇u = ν∇²u.

    The Burgers equation is a fundamental nonlinear PDE that:
    - Models fluid dynamics and shock formation
    - Serves as a simplified Navier-Stokes equation
    - Exhibits both advection and diffusion

    This implementation supports:
    - Time-dependent and steady-state cases
    - Variable viscosity
    - 1D and 2D domains

    The time-dependent case is pinned to the standard Basdevant / Cole-Hopf
    benchmark, and :meth:`initial_condition`, :meth:`boundary_value` and
    :meth:`exact_solution` all describe that one problem::

        u_t + u*u_x = nu*u_xx    on x in [0, 1]
        u(x, 0) = -sin(pi*x)
        u(0, t) = u(1, t) = 0    (homogeneous Dirichlet)

    ``exact_solution`` is the closed-form Cole-Hopf series for exactly this
    initial/boundary data, so ``exact_solution(coords, time=0.0)`` reproduces
    ``initial_condition(coords)`` to machine precision.
    """

    name = "burgers"
    description = "Burgers equation: u_t + u·∇u = ν∇²u"
    pde_type = PDEType.BURGERS
    is_time_dependent = True
    is_linear = False
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        viscosity: float | None = None,
    ) -> None:
        """Initialize Burgers operator.

        Args:
            config: PDE configuration.
            viscosity: Kinematic viscosity (overrides config if provided).

        """
        super().__init__(config)
        self.viscosity = viscosity if viscosity is not None else config.diffusion_coeff
        # PDEConfig.is_time_dependent defaults to False, and unconditionally
        # assigning it here silently downgraded every caller that never
        # mentioned the flag (e.g. _centaur_common.build_pde_operator) from
        # the class-level `is_time_dependent = True` default to False, which
        # made exact_solution() return None unconditionally (the flat-reward
        # defect tracked as P0-1 in docs/CODE_HYGIENE_AUDIT.md). Only honour
        # config.is_time_dependent when the caller actually set it — either
        # explicitly at construction (``PDEConfig(..., is_time_dependent=...)``)
        # or via a later mutation — so an explicit False still disables the
        # Cole-Hopf solution (see test_steady_returns_none) while an unset
        # config keeps Burgers' true default of being time-dependent.
        if "is_time_dependent" in config.model_fields_set:
            self.is_time_dependent = config.is_time_dependent

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
        time: float | None = None,
    ) -> PDEResidual:
        """Compute the *steady-state* Burgers residual: R = u·∇u - ν∇²u.

        The time derivative ``u_t`` is deliberately absent (see the "Steady
        state" comment below): this operator evaluates the residual of a single
        spatial configuration and is never handed a time history, so it cannot
        form ``u_t``. Consequently the usual "residual vanishes on the exact
        solution" manufactured-solution check does **not** apply to Burgers --
        :meth:`exact_solution` solves the *time-dependent* equation, whose
        steady residual is non-zero by construction. Correctness of
        :meth:`exact_solution` is pinned instead by its agreement with
        :meth:`initial_condition` at ``t = 0``.

        Args:
            u: Solution values at ``coords``.
            coords: Collocation points (N, dim); requires grad for autodiff.
            compute_derivatives: Whether to return the derivative dictionary.
            time: Unused; accepted for interface compatibility.

        Returns:
            The steady-state residual and its norms.

        """
        derivatives = self.compute_derivatives(u, coords)

        laplacian = derivatives.get("laplacian", torch.zeros_like(u))

        # Nonlinear advection term: u · ∇u
        #
        # ``u.squeeze()`` on the accumulator as well as the operand, matching
        # ``AdvectionDiffusionOperator`` (which already does this) and the
        # ``(N,)``/``(N, 1)`` contract in :meth:`PDEOperator.residual`'s
        # docstring. Accumulating into ``zeros_like(u)`` instead broadcast a
        # ``(N, 1)`` input against the ``(N,)`` derivative and produced an
        # ``(N, N)`` residual -- silently, because ``zeros_like(u)`` makes every
        # row identical, so ``l2_norm`` and ``max_norm`` still came out right.
        # What was wrong was ``PDEResidual.values``: N=500 yielded 250 000
        # entries where callers document and reshape ``(N,)``
        # (``basis_selection.py`` assigns it to ``PDEState.residuals`` and then
        # reshapes to a square grid), plus an O(N^2) allocation.
        advection = torch.zeros_like(u.squeeze())
        for d in range(self.dim):
            du_dx = derivatives.get(f"u_x{d}", torch.zeros_like(u.squeeze()))
            advection = advection + u.squeeze() * du_dx

        # Steady state: u·∇u = ν∇²u
        residual_values = advection - self.viscosity * laplacian

        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())

        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives=derivatives if compute_derivatives else {},
        )

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Burgers equation has no explicit source term."""
        if isinstance(coords, Tensor):
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        return np.zeros(coords.shape[0], dtype=np.float32)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute boundary values: homogeneous Dirichlet ``u(0, t) = u(1, t) = 0``.

        This is the boundary data of the Cole-Hopf benchmark solved by
        :meth:`exact_solution` (whose ``-sin(pi*x)`` initial condition already
        vanishes at both endpoints, and whose Neumann potential condition
        ``phi_x = 0`` keeps it vanishing for all ``t``).

        Previously this returned a ``0.5*(1 - tanh(w*(x - x0)))`` shock profile,
        i.e. ``u(0, t) = 1``, ``u(1, t) = 0`` -- a *third* problem, inconsistent
        with both the initial condition and the exact solution.
        """
        if isinstance(coords, Tensor):
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        return np.zeros(coords.shape[0], dtype=np.float32)

    def initial_condition(
        self,
        coords: NDArray[np.float32] | Tensor,
    ) -> NDArray[np.float32] | Tensor:
        """Compute initial condition ``u(x, 0) = -sin(pi*x)``.

        This is the standard Basdevant / Cole-Hopf benchmark profile, and it is
        the initial datum that :meth:`exact_solution` propagates: the two agree
        to machine precision at ``t = 0``.

        Previously this returned ``sin(2*pi*x)``, which is neither what
        :meth:`exact_solution` propagates nor compatible with the homogeneous
        Dirichlet data of :meth:`boundary_value` at ``x = 0.5``.
        """
        if isinstance(coords, Tensor):
            x = coords[:, 0]
            return -torch.sin(np.pi * x)
        else:
            x = coords[:, 0]
            return (-np.sin(np.pi * x)).astype(np.float32)

    def cole_hopf_potential(
        self,
        coords: NDArray[np.float32] | NDArray[np.float64],
        time: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the Cole-Hopf potential ``phi`` and its ``x``-derivative.

        This is the numerator/denominator pair behind :meth:`exact_solution`,
        exposed (in full float64, *before* the :data:`COLE_HOPF_CLAMP_EPS`
        positivity floor) so that the denominator's sign and magnitude can be
        inspected directly -- the historical defect was a *negative* denominator
        that the floor silently converted into a finite but meaningless ``u``.

        Args:
            coords: Points (N, dim); only the first column (``x``) is used.
            time: Time at which to evaluate (default 0).

        Returns:
            ``(phi, dphi)``, both float64 of shape (N,). ``phi`` is scaled by
            ``e^-R`` relative to the true potential, which cancels in
            ``u = -2*nu*dphi/phi``.

        """
        c0, n, coefficients = _cole_hopf_coefficients(self.viscosity)
        t = time if time is not None else 0.0
        pi = float(np.pi)

        amplitude = coefficients * np.exp(-((n * pi) ** 2) * self.viscosity * t)
        angle = np.outer(np.asarray(coords[:, 0], dtype=np.float64), n) * pi
        phi: NDArray[np.float64] = c0 + np.sum(np.cos(angle) * amplitude, axis=-1)
        dphi: NDArray[np.float64] = -np.sum(np.sin(angle) * (amplitude * n * pi), axis=-1)
        return phi, dphi

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor | None:
        """Cole-Hopf exact solution for the 1D viscous Burgers equation.

        The Cole-Hopf transformation ``u = -2*nu*(phi_x/phi)`` converts the
        nonlinear Burgers equation into the linear heat equation. For the
        benchmark data ``u(x, 0) = -sin(pi*x)`` on ``[0, 1]`` with
        ``u(0, t) = u(1, t) = 0`` (see :meth:`initial_condition` and
        :meth:`boundary_value`), the exact solution is::

            u(x, t) = -2*nu * sum_n  c_n * (-n*pi) * exp(-n^2*pi^2*nu*t) * sin(n*pi*x)
                             / ( c_0 + sum_n c_n * exp(-n^2*pi^2*nu*t) * cos(n*pi*x) )

            c_0 = ive(0, R),  c_n = 2*(-1)^n*ive(n, R),  R = 1/(2*pi*nu)

        with ``ive`` the exponentially-scaled modified Bessel function of the
        first kind. The coefficients are derived and cached in
        :func:`_cole_hopf_coefficients`; every one of them was previously
        hardcoded to 1, which is the Cole-Hopf image of a Dirac comb rather than
        of a sinusoid.

        The series is assembled in float64 (the denominator legitimately reaches
        ``1.5e-14`` at ``nu = 0.01``, which would underflow float32) and cast
        back to the input dtype on return.

        For ``nu < COLE_HOPF_MIN_RESOLVED_VISCOSITY`` accuracy is lost over the
        whole left-hand interval ``x < arccos(2*pi*nu*ln(1/eps) - 1)/pi``, not
        merely in a neighbourhood of ``x = 0``: that is ~20% of the domain at
        ``nu = 0.009``, ~50% at ``nu = 0.005`` and ~80% at ``nu = 0.001``, and
        inside it ``u`` is pinned to ~0 while the true solution is O(1). Only
        the far field can be trusted there; see
        :data:`COLE_HOPF_MIN_RESOLVED_VISCOSITY` for the measured table.

        Args:
            coords: Points (N, dim) with spatial coordinates.
            time: Time at which to evaluate (default 0).

        Returns:
            Exact solution values (N,) in the input's dtype/device, or None for
            a steady-state (non-time-dependent) configuration, which has no
            Cole-Hopf solution.

        """
        if not self.is_time_dependent:
            return None

        t = time if time is not None else 0.0
        nu = self.viscosity

        if isinstance(coords, Tensor):
            c0, n_np, coefficients_np = _cole_hopf_coefficients(nu)
            pi = float(np.pi)
            # torch.tensor (not as_tensor) copies, so the cached read-only
            # numpy arrays are never aliased into a mutable tensor.
            n = torch.tensor(n_np, dtype=torch.float64, device=coords.device)
            coefficients = torch.tensor(
                coefficients_np,
                dtype=torch.float64,
                device=coords.device,
            )
            # float64 assembly: gradients still flow through coords, since
            # .to(float64) and the final .to(coords.dtype) are differentiable.
            x = coords[:, 0].to(torch.float64)
            amplitude = coefficients * torch.exp(-((n * pi) ** 2) * nu * t)
            angle = x.unsqueeze(-1) * n.unsqueeze(0) * pi
            phi = c0 + (torch.cos(angle) * amplitude).sum(dim=-1)
            dphi = -(torch.sin(angle) * (amplitude * n * pi)).sum(dim=-1)
            u = -2.0 * nu * dphi / phi.clamp(min=COLE_HOPF_CLAMP_EPS)
            return u.to(coords.dtype)

        phi_np, dphi_np = self.cole_hopf_potential(coords, time=t)
        u_np = -2.0 * nu * dphi_np / np.maximum(phi_np, COLE_HOPF_CLAMP_EPS)
        return u_np.astype(np.float32)

    def convergence_rate(self, h_values: list[float], errors: list[float]) -> float:
        """Compute convergence rate from h-refinement study.

        Given errors at different mesh sizes h, fits log(error) = p*log(h) + C
        to estimate the convergence order p.

        Args:
            h_values: Mesh sizes (decreasing).
            errors: Corresponding L2 errors.

        Returns:
            Estimated convergence rate p.

        """
        log_h = np.log(np.array(h_values))
        log_e = np.log(np.array(errors))
        # Linear regression: log(e) = p * log(h) + C
        coeffs = np.polyfit(log_h, log_e, 1)
        return float(coeffs[0])
