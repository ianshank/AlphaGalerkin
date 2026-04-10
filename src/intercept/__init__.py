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
"""

from __future__ import annotations

__all__ = [
    # Aerodynamics
    "AeroModel",
    "SimpleAeroModel",
    "TabularAeroModel",
    # Assignment
    "AssignmentSolver",
    "HungarianAssigner",
    # Atmosphere
    "ISAAtmosphere",
    "WindModel",
    # Config
    "AtmosphereConfig",
    "EngagementConfig",
    "GuidanceConfig",
    "InterceptorConfig",
    "ThreatConfig",
    # Dynamics
    "RigidBody6DOF",
    "RigidBodyState",
    # Frames
    "FrameTransform",
    "QuaternionOps",
    # Guidance
    "GuidanceCommand",
    "ProportionalNavigation",
    # Game
    "InterceptGameAdapter",
    # Sensors
    "RadarSensor",
    "SensorFusion",
    # Tracking
    "ExtendedKalmanFilter",
    "TrackState",
]
