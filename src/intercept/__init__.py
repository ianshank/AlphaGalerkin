"""AlphaGalerkin Missile Defense & Drone Interception module.

Extends AlphaGalerkin's MCTS + Galerkin neural operator architecture
to real-time threat trajectory prediction and interceptor guidance.

Key components:
- 6-DOF rigid body dynamics with quaternion integration
- Coordinate frame transforms (NED, ENU, ECEF, body-fixed)
- ISA standard atmosphere model
- Aerodynamic force models (tabular, Galerkin neural operator)
- EKF track state estimation with confidence envelopes
- MCTS-based threat maneuver prediction
- Proportional navigation and MCTS-guided interceptor guidance
- InterceptGame: engagement as MCTS game via GameInterface protocol
- Swarm assignment: Hungarian, auction, greedy solvers with triage
- Multi-sensor fusion: radar, EO, IR with staleness tracking

Submodules are imported lazily via :pep:`562` ``__getattr__`` so that
``from src.intercept import AeroModel`` works without paying the full
import cost (gradio/torch/scipy) for unrelated callers.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Map ``__all__`` entries to ``(submodule, attribute)`` for lazy resolution.
# Keep this in sync with the ``__all__`` tuple below — the test
# ``tests/intercept/test_public_surface.py`` validates that every entry
# resolves at runtime.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Aerodynamics
    "AeroModel": ("aero", "AeroModel"),
    "SimpleAeroModel": ("aero", "SimpleAeroModel"),
    "TabularAeroModel": ("aero", "TabularAeroModel"),
    # Assignment
    "AssignmentSolver": ("assignment", "AssignmentSolver"),
    "HungarianAssigner": ("assignment", "HungarianAssigner"),
    # Atmosphere
    "ISAAtmosphere": ("atmosphere", "ISAAtmosphere"),
    "WindModel": ("atmosphere", "WindModel"),
    # Config
    "AtmosphereConfig": ("config", "AtmosphereConfig"),
    "EngagementConfig": ("config", "EngagementConfig"),
    "GuidanceConfig": ("config", "GuidanceConfig"),
    "InterceptorConfig": ("config", "InterceptorConfig"),
    "ThreatConfig": ("config", "ThreatConfig"),
    # Dynamics
    "RigidBody6DOF": ("dynamics", "RigidBody6DOF"),
    "RigidBodyState": ("dynamics", "RigidBodyState"),
    # Frames
    "FrameTransform": ("frames", "FrameTransform"),
    "QuaternionOps": ("frames", "QuaternionOps"),
    # Guidance
    "GuidanceCommand": ("guidance", "GuidanceCommand"),
    "ProportionalNavigation": ("guidance", "ProportionalNavigation"),
    # Game
    "InterceptGameAdapter": ("intercept_game", "InterceptGameAdapter"),
    # Sensors
    "RadarSensor": ("sensors", "RadarSensor"),
    "SensorFusion": ("sensors", "SensorFusion"),
    # Tracking
    "ExtendedKalmanFilter": ("tracking", "ExtendedKalmanFilter"),
    "TrackState": ("tracking", "TrackState"),
}

__all__ = sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> Any:
    """Lazily resolve attributes listed in :data:`_LAZY_ATTRS`.

    Raises :class:`AttributeError` for unknown names so that
    ``hasattr`` and ``from src.intercept import X`` behave correctly.
    """
    try:
        submodule_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'src.intercept' has no attribute {name!r}") from exc
    module = importlib.import_module(f".{submodule_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    """Expose lazily-loaded names to ``dir()`` and tab completion."""
    return sorted(set(globals()).union(_LAZY_ATTRS))


if TYPE_CHECKING:  # pragma: no cover - import-time hint only
    # These imports help static type-checkers (mypy / pylance) resolve the
    # public surface that is otherwise computed dynamically via __getattr__.
    # The names are listed in ``__all__`` (built from ``_LAZY_ATTRS``) so the
    # F401 unused-import warnings are spurious here.
    from .aero import AeroModel, SimpleAeroModel, TabularAeroModel  # noqa: F401
    from .assignment import AssignmentSolver, HungarianAssigner  # noqa: F401
    from .atmosphere import ISAAtmosphere, WindModel  # noqa: F401
    from .config import (  # noqa: F401
        AtmosphereConfig,
        EngagementConfig,
        GuidanceConfig,
        InterceptorConfig,
        ThreatConfig,
    )
    from .dynamics import RigidBody6DOF, RigidBodyState  # noqa: F401
    from .frames import FrameTransform, QuaternionOps  # noqa: F401
    from .guidance import GuidanceCommand, ProportionalNavigation  # noqa: F401
    from .intercept_game import InterceptGameAdapter  # noqa: F401
    from .sensors import RadarSensor, SensorFusion  # noqa: F401
    from .tracking import ExtendedKalmanFilter, TrackState  # noqa: F401
