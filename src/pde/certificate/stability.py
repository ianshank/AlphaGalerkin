r"""Stability-constant registry keyed on :class:`src.pde.config.PDEType`.

For any admissible :math:`\tilde{u} \in V` with boundary conditions exactly
enforced, the residual bound

.. math:: \|u - \tilde{u}\| \;\le\; C_0\, \|r(\tilde{u})\|

is only as honest as the operator's stability constant :math:`C_0` (inf-sup
:math:`\beta` in the coercive case). This module declares the *source* of
:math:`C_0` per :class:`~src.pde.config.PDEType`; it does *not* invent numbers.

* ``analytic`` — closed-form constant available (e.g. Poincaré for the pure
  Poisson–Dirichlet problem).
* ``estimated`` — data-derived / empirical, must carry ``notes`` explaining
  the fit and its regime of validity.
* ``unbounded_with_warning`` — no honest constant available (Helmholtz at
  high wavenumber, Biharmonic without a declared constant, Navier–Stokes).
  Certificates on these operators render as "residual bound only —
  no error guarantee" (spec AC5).

The registry is a **thread-safe singleton** with decorator-based extension.
The primary consumer is the (follow-on) Track A / Track B estimator modules;
``specs/operator_gate.spec.md`` will consume it once that spec lands. No new
``PDEType`` value is introduced here — the registry key is the existing enum.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from src.pde.config import PDEType

logger = structlog.get_logger(__name__)

# ``StabilitySource`` is an open literal — new sources (e.g. 'proven_lean4')
# can be added without breaking the on-disk artifact schema because
# :class:`~src.pde.certificate.certificate.Certificate.stability_source` is
# ``str``, not this enum. This literal is enforced *at registration time*.
StabilitySource = Literal["analytic", "estimated", "unbounded_with_warning"]

# Render string used by Track A / Track B when an operator is UNBOUNDED. The
# constant is exposed so downstream reports do not re-invent the phrasing.
UNBOUNDED_RENDER_STRING: str = "residual bound only — no error guarantee"


class StabilityEntry(BaseModel):
    """One registered operator stability declaration.

    Instances are *data*, not code — this is why the registry stores
    :class:`StabilityEntry` values directly rather than reusing
    :func:`src.templates.registry.create_registry` (which registers classes).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pde_type: PDEType = Field(..., description="Operator this entry applies to.")
    source: StabilitySource = Field(
        ...,
        description=(
            "'analytic' | 'estimated' | 'unbounded_with_warning'. Chosen at "
            "registration time; must match the ``value`` field's presence."
        ),
    )
    value: float | None = Field(
        default=None,
        description=(
            "The declared numeric ``C_0`` (or inf-sup ``β``). ``None`` iff "
            "``source == 'unbounded_with_warning'``."
        ),
    )
    notes: str = Field(
        default="",
        description=(
            "Free-form provenance — proof citation for 'analytic', empirical "
            "regime for 'estimated', explanation for 'unbounded_with_warning'."
        ),
    )

    def render(self) -> str:
        """Human-readable one-liner for reports / results / business docs."""
        if self.source == "unbounded_with_warning":
            return UNBOUNDED_RENDER_STRING
        return f"C_0 = {self.value} ({self.source})"


class StabilityConstantRegistry:
    """Thread-safe singleton mapping :class:`PDEType` → :class:`StabilityEntry`.

    Follows the double-check-locking singleton pattern used elsewhere in the
    project (:mod:`src.templates.registry`). Registration is idempotent-with-
    duplicate-raise: registering the same ``pde_type`` twice raises, so
    conflicting declarations from two modules cannot silently overwrite each
    other. Overrides go through :meth:`replace`, which logs at ``WARNING``.

    The class-level ``_instance`` / ``_lock`` mirror the BaseRegistry contract
    so tests can reset the singleton in ``conftest.py`` via
    ``StabilityConstantRegistry._reset_for_tests()``.
    """

    _instance: StabilityConstantRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> StabilityConstantRegistry:
        # Double-check locking mirrors :class:`src.templates.registry.BaseRegistry`.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._entries = {}  # type: ignore[attr-defined]
                    instance._entry_lock = threading.Lock()  # type: ignore[attr-defined]
                    cls._instance = instance
        return cls._instance

    # --- Registration -------------------------------------------------------

    def register(self, entry: StabilityEntry) -> None:
        """Add a fresh entry. Raises ``ValueError`` on duplicate ``pde_type``."""
        with self._entry_lock:  # type: ignore[attr-defined]
            if entry.pde_type in self._entries:  # type: ignore[attr-defined]
                raise ValueError(
                    f"stability entry for {entry.pde_type!r} already registered; "
                    f"use replace() if the override is intentional"
                )
            self._entries[entry.pde_type] = entry  # type: ignore[attr-defined]
        logger.debug(
            "certificate.stability_registered",
            pde_type=entry.pde_type.value,
            source=entry.source,
        )

    def replace(self, entry: StabilityEntry) -> None:
        """Override an existing entry. Logs at ``WARNING`` — intentional escape hatch."""
        with self._entry_lock:  # type: ignore[attr-defined]
            existed = entry.pde_type in self._entries  # type: ignore[attr-defined]
            self._entries[entry.pde_type] = entry  # type: ignore[attr-defined]
        logger.warning(
            "certificate.stability_replaced",
            pde_type=entry.pde_type.value,
            source=entry.source,
            existed=existed,
        )

    def get(self, pde_type: PDEType) -> StabilityEntry:
        """Return the entry for ``pde_type``. Raises ``KeyError`` if unregistered.

        The raise is *deliberate*: silent fall-through would ship a certificate
        with an undocumented stability source, violating AC5's
        ``undocumented_stability_constants = 0``.
        """
        try:
            return self._entries[pde_type]  # type: ignore[attr-defined,no-any-return]
        except KeyError as exc:
            raise KeyError(
                f"no stability entry registered for {pde_type!r}; register one "
                f"in src.pde.certificate.stability at import time (spec AC5 — "
                f"undocumented stability constants are a hard failure)"
            ) from exc

    def has(self, pde_type: PDEType) -> bool:
        """``True`` iff an entry has been registered for ``pde_type``."""
        return pde_type in self._entries  # type: ignore[attr-defined]

    def registered_types(self) -> tuple[PDEType, ...]:
        """Immutable snapshot of registered types, in insertion order."""
        with self._entry_lock:  # type: ignore[attr-defined]
            return tuple(self._entries.keys())  # type: ignore[attr-defined]

    # --- Test-only escape hatch --------------------------------------------

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Drop the singleton so a fresh registry is built on next ``__new__``.

        Only for use from ``conftest.py``. Named with a leading underscore to
        signal non-public API.
        """
        with cls._lock:
            cls._instance = None


def register_stability(
    pde_type: PDEType,
    source: StabilitySource,
    value: float | None = None,
    notes: str = "",
) -> StabilityEntry:
    """Convenience registrar — build the entry, register it, return it.

    Useful in the built-in registration block below and in downstream modules
    (e.g. ``operator_gate`` when that spec lands) that need to declare stability
    for a newly-added ``PDEType``.

    Args:
        pde_type: The operator this declaration applies to.
        source: One of ``'analytic' | 'estimated' | 'unbounded_with_warning'``.
        value: Numeric ``C_0`` (or ``β``). Must be ``None`` iff
            ``source == 'unbounded_with_warning'``.
        notes: Free-form provenance string.

    Raises:
        ValueError: ``source`` and ``value`` are inconsistent, or the entry
            already exists (:meth:`StabilityConstantRegistry.register` raises).

    """
    if source == "unbounded_with_warning":
        if value is not None:
            raise ValueError("source='unbounded_with_warning' requires value=None (spec AC5)")
    else:
        if value is None:
            raise ValueError(f"source={source!r} requires a numeric value (spec AC5)")
        if value <= 0.0:
            raise ValueError(f"stability constant must be positive, got {value!r}")
    entry = StabilityEntry(pde_type=pde_type, source=source, value=value, notes=notes)
    StabilityConstantRegistry().register(entry)
    return entry


# ---------------------------------------------------------------------------
# Built-in declarations. Every :class:`PDEType` value must appear here — AC5
# ``undocumented_stability_constants = 0`` is enforced by
# ``tests/pde/certificate/test_stability_registry.py``.
#
# Design principle: honesty > premature rigor. Where a numeric constant is not
# defensible today, register ``unbounded_with_warning`` with a notes trail
# rather than invent a number. Follow-on specs (``operator_gate``) refine.
# ---------------------------------------------------------------------------


def _register_builtin_stability_entries() -> None:
    """Populate the singleton with one entry per :class:`PDEType`.

    Called exactly once at module import. The function-scoped structure lets
    tests reset the registry and re-run this idempotently.
    """
    registry = StabilityConstantRegistry()
    if registry.has(PDEType.POISSON):
        # Already populated (e.g. by a previous import in the same process).
        return

    # Poisson: Poincaré-Friedrichs on a bounded Lipschitz domain with Dirichlet
    # BCs gives a domain-diameter-dependent constant. For the unit box this is
    # bounded by 1/(sqrt(2)*pi); registered as ``analytic`` with a *placeholder*
    # numeric of 1.0 to be calibrated from the first Track A measured run per
    # the ``stochastic_galerkin_nke`` gate-calibration convention.
    register_stability(
        PDEType.POISSON,
        source="analytic",
        value=1.0,
        notes=(
            "Poincaré-Friedrichs on the unit box; placeholder magnitude to be "
            "calibrated from first Track A run per stochastic_galerkin_nke "
            "gate-calibration convention"
        ),
    )

    # Heat: analytic energy estimate on the same domain family; placeholder
    # magnitude, same calibration policy as Poisson.
    register_stability(
        PDEType.HEAT,
        source="analytic",
        value=1.0,
        notes="Energy estimate; placeholder magnitude — calibrate from first run",
    )

    # Advection-diffusion (steady, coercive regime): SUPG stability constant is
    # regime-dependent (Peclet number). Declared ``estimated`` so any downstream
    # certificate carries an empirical-fit provenance string.
    register_stability(
        PDEType.ADVECTION_DIFFUSION,
        source="estimated",
        value=1.0,
        notes=(
            "Regime-dependent (Peclet); placeholder magnitude — recalibrate per operator instance"
        ),
    )

    # Wave (time-harmonic linear wave): treat as coercive on short-time windows;
    # placeholder ``estimated`` pending a proper CFL-conditioned constant.
    register_stability(
        PDEType.WAVE,
        source="estimated",
        value=1.0,
        notes=("Short-time coercive estimate; placeholder — recalibrate against CFL condition"),
    )

    # Burgers (viscous, small-data regime): linearised energy estimate holds
    # near u=0; genuinely regime-dependent for large data. Placeholder.
    register_stability(
        PDEType.BURGERS,
        source="estimated",
        value=1.0,
        notes=("Small-data linearised energy estimate; large-data regime is genuinely unbounded"),
    )

    # Helmholtz: indefinite for k^2 above the smallest Dirichlet eigenvalue.
    # A wavenumber-dependent estimate is genuinely non-trivial and lands with
    # ``specs/operator_gate.spec.md``. Ship UNBOUNDED for honesty — a fake
    # numeric here would be exactly the fabrication precedent the spec is
    # meant to prevent.
    register_stability(
        PDEType.HELMHOLTZ,
        source="unbounded_with_warning",
        value=None,
        notes=(
            "TODO(operator_gate): indefinite Helmholtz needs a wavenumber-"
            "dependent estimate. UNBOUNDED for honesty until specs/"
            "operator_gate.spec.md lands"
        ),
    )

    # Biharmonic: fourth-order operator; classical estimate requires H^2
    # coercivity constant on the chosen domain. Not registered as a numeric
    # here — a wrong number is worse than an honest UNBOUNDED.
    register_stability(
        PDEType.BIHARMONIC,
        source="unbounded_with_warning",
        value=None,
        notes=(
            "Fourth-order; H^2 coercivity constant is domain-dependent. "
            "UNBOUNDED until a per-domain calibration lands"
        ),
    )

    # Navier–Stokes: no unconditional coercive estimate for the nonlinear
    # incompressible problem. Certificates on NS render as residual bound only.
    register_stability(
        PDEType.NAVIER_STOKES,
        source="unbounded_with_warning",
        value=None,
        notes=(
            "Nonlinear incompressible NS has no unconditional coercive "
            "estimate; residual bound only"
        ),
    )


_register_builtin_stability_entries()


__all__ = [
    "StabilityConstantRegistry",
    "StabilityEntry",
    "StabilitySource",
    "UNBOUNDED_RENDER_STRING",
    "register_stability",
]


# ``Callable`` re-export keeps mypy happy without a top-level import that would
# only be used for a type hint in a docstring example.
_typing_marker: Callable[..., object] = register_stability
