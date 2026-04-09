"""AlphaGalerkin Missile Defense & Drone Interception module.

Extends AlphaGalerkin's MCTS + Galerkin neural operator architecture
to real-time threat trajectory prediction and interceptor guidance.

Key components:
- 6-DOF rigid body dynamics with quaternion integration
- Coordinate frame transforms (NED, ENU, ECEF, body-fixed)
- ISA standard atmosphere model
- Aerodynamic force models (tabular, Galerkin neural operator)
- EKF/UKF track state estimation
- MCTS-based threat maneuver prediction
- Proportional navigation and MCTS-guided interceptor guidance
- InterceptGame: engagement as MCTS game via GameInterface protocol
"""

from __future__ import annotations

__all__ = [
    "AeroModel",
    "AtmosphereConfig",
    "EngagementConfig",
    "FrameTransform",
    "GuidanceConfig",
    "ISAAtmosphere",
    "InterceptorConfig",
    "QuaternionOps",
    "RigidBody6DOF",
    "RigidBodyState",
    "SimpleAeroModel",
    "TabularAeroModel",
    "ThreatConfig",
    "WindModel",
]
