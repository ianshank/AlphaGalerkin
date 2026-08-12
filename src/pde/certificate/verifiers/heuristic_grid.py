r"""Dense-grid residual verifier — reference implementation (WS1).

This verifier evaluates the residual on a Cartesian grid over the domain
and returns the sample max as an *upper bound*. It carries
``rigor='heuristic'`` and ``domain_coverage='grid_sampled'``, so the
structural guard in
:class:`~src.pde.certificate.types.CertifiedResidualBound` refuses any
attempt to promote it to ``rigor='rigorous'`` (spec §4 AC3).

Why include it?
    * CI smoke: gives every scenario a working Track B path on a base
      install with no optional extras.
    * Regression oracle: the rigorous WS2 verifiers must produce a bound
      that is *at least as tight* as the heuristic sample max on the same
      residual, up to the stability-constant multiplier from
      :class:`~src.pde.certificate.stability.StabilityConstantRegistry`.
    * Debug / demo path: users can experiment with the certificate
      pipeline without installing ``[certificate-rigorous]``.

No hardcoded values — every knob comes from
:attr:`~src.pde.certificate.config.CertificateConfig.heuristic_grid_resolution`
or the :class:`~src.pde.certificate.types.DomainSpec`.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

from src.pde.certificate.interface import ResidualVerifier
from src.pde.certificate.registry import capture_hardware_meta, register_verifier
from src.pde.certificate.types import (
    CertificationBudget,
    CertifiedModel,
    CertifiedResidualBound,
    DomainSpec,
    VerifierBackend,
)

logger = structlog.get_logger(__name__)

# Fallback resolution when the caller has not set ``DomainSpec.grid_resolution``.
# Deliberately conservative so a smoke test on a laptop finishes fast; users
# should override via :attr:`CertificateConfig.heuristic_grid_resolution`.
_DEFAULT_GRID_RESOLUTION: int = 16


@register_verifier("heuristic_grid")
class HeuristicGridResidualVerifier:
    """Sample-max residual verifier — always available, heuristic tier."""

    #: Registry key. Matches :data:`VerifierBackend`.
    backend_name: VerifierBackend = "heuristic_grid"

    def __init__(self, *, default_grid_resolution: int = _DEFAULT_GRID_RESOLUTION) -> None:
        if default_grid_resolution <= 0:
            raise ValueError(
                f"default_grid_resolution must be positive, got {default_grid_resolution!r}"
            )
        self._default_grid_resolution = default_grid_resolution

    # ------------------------------------------------------------------
    # Grid construction — pure numpy, no framework imports.
    # ------------------------------------------------------------------

    def _build_grid(self, domain: DomainSpec) -> NDArray[np.float64]:
        """Construct a Cartesian grid inside ``domain.bounds``.

        Raises:
            ValueError: if ``domain.kind`` is not ``'rectangular'``.
                SDF coverage is a WS2 concern (Newton-projected boundary
                sampling in :mod:`src.pde.geometry_picogk`).

        """
        if domain.kind != "rectangular":
            raise ValueError(
                f"HeuristicGridResidualVerifier only supports rectangular "
                f"domains; got kind={domain.kind!r}. SDF support lands in WS2."
            )
        resolution = domain.grid_resolution or self._default_grid_resolution
        axes = [np.linspace(lo, hi, resolution, dtype=np.float64) for lo, hi in domain.bounds]
        # Cartesian product → shape (resolution**d, d). Vectorised via
        # itertools.product then a single np.asarray call — no per-axis
        # meshgrid inflation.
        grid = np.asarray(list(itertools.product(*axes)), dtype=np.float64)
        return grid

    # ------------------------------------------------------------------
    # Model call — supports the three ``CertifiedModel.backend`` values.
    # ------------------------------------------------------------------

    def _evaluate_model(
        self,
        model: CertifiedModel,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate ``model.model_fn`` at ``points`` and return a numpy array.

        For ``backend='numpy'`` the call is direct. For ``'torch'`` and
        ``'jax'`` we do a *just-in-time* framework import so the module
        remains ML-free at load time (spec §3, Gate 5 sibling).
        """
        fn = model.model_fn
        if model.backend == "numpy":
            result: Any = fn(points)
            return np.asarray(result, dtype=np.float64)

        if model.backend == "torch":  # pragma: no cover — env-dependent
            import torch  # type: ignore[import-not-found, unused-ignore]

            with torch.no_grad():
                tensor = torch.as_tensor(points, dtype=torch.float64)
                out = fn(tensor)
            return np.asarray(out.detach().cpu().numpy(), dtype=np.float64)

        if model.backend == "jax":  # pragma: no cover — env-dependent
            import jax.numpy as jnp  # type: ignore[import-not-found, unused-ignore]

            out = fn(jnp.asarray(points))
            return np.asarray(out, dtype=np.float64)

        raise ValueError(f"unsupported model.backend={model.backend!r}")

    # ------------------------------------------------------------------
    # Public: certify
    # ------------------------------------------------------------------

    def certify(
        self,
        *,
        model: CertifiedModel,
        domain: DomainSpec,
        budget: CertificationBudget,
    ) -> CertifiedResidualBound:
        """Return a heuristic-tier bound (sample max) with cost accounting."""
        # Local Dtype/Device — we always run this verifier in float64 on
        # whatever the caller declared; the hardware_meta reflects that.
        dtype = "float64"
        # Try to infer a device string from the model, else fall back to cpu.
        device: str = "cpu"

        t0 = time.perf_counter()
        try:
            grid = self._build_grid(domain)
            values = self._evaluate_model(model, grid)
            # Support scalar-per-point (residual r(x)) or vector fields — flatten.
            residual = np.abs(values).reshape(-1)
            upper = float(np.max(residual)) if residual.size else 0.0
        except Exception as exc:  # pragma: no cover — narrow catch documented
            cert_wall = time.perf_counter() - t0
            hw = capture_hardware_meta(device=device, dtype=dtype)
            logger.warning(
                "certificate.verifier_end",
                backend=self.backend_name,
                rigor="failed",
                failure_reason=type(exc).__name__,
                cert_wall_s=cert_wall,
            )
            return CertifiedResidualBound(
                upper_bound=0.0,
                rigor="failed",
                backend=self.backend_name,
                domain_coverage="grid_sampled",
                compile_wall_s=0.0,
                cert_wall_s=cert_wall,
                steady_state_wall_s=cert_wall,
                hardware_meta=hw,
                failure_reason=f"{type(exc).__name__}: {exc}",
                notes="",
            )

        cert_wall = time.perf_counter() - t0

        # Budget accounting — heuristic tier can either "fail" or emit a
        # note-tagged bound depending on ``allow_heuristic_fallback``.
        if cert_wall > budget.max_wall_s:
            if not budget.allow_heuristic_fallback:
                hw = capture_hardware_meta(device=device, dtype=dtype)
                logger.warning(
                    "certificate.budget_overrun",
                    backend=self.backend_name,
                    cert_wall_s=cert_wall,
                    max_wall_s=budget.max_wall_s,
                )
                return CertifiedResidualBound(
                    upper_bound=0.0,
                    rigor="failed",
                    backend=self.backend_name,
                    domain_coverage="grid_sampled",
                    compile_wall_s=0.0,
                    cert_wall_s=cert_wall,
                    steady_state_wall_s=cert_wall,
                    hardware_meta=hw,
                    failure_reason=(
                        f"budget_exceeded: cert_wall_s={cert_wall:.4f} "
                        f"exceeded max_wall_s={budget.max_wall_s}"
                    ),
                    notes="",
                )
            # allow_heuristic_fallback: emit the heuristic bound anyway,
            # tagged in notes.
            notes = "budget_exceeded"
        else:
            notes = ""

        hw = capture_hardware_meta(device=device, dtype=dtype)
        logger.info(
            "certificate.verifier_end",
            backend=self.backend_name,
            rigor="heuristic",
            upper_bound=upper,
            grid_points=int(grid.shape[0]),
            cert_wall_s=cert_wall,
        )
        return CertifiedResidualBound(
            upper_bound=upper,
            rigor="heuristic",
            backend=self.backend_name,
            domain_coverage="grid_sampled",
            compile_wall_s=0.0,
            cert_wall_s=cert_wall,
            steady_state_wall_s=cert_wall,
            hardware_meta=hw,
            failure_reason=None,
            notes=notes,
        )


__all__ = ["HeuristicGridResidualVerifier"]


# Static compile-time contract check — asserts the class shape-matches the
# Protocol without needing a runtime ``isinstance`` call. If a future edit
# drops ``certify`` or ``backend_name`` this line breaks the type checker.
_: type[ResidualVerifier] = HeuristicGridResidualVerifier  # type: ignore[type-abstract, unused-ignore]
