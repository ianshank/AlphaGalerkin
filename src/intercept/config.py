"""Pydantic configuration schemas for the intercept module.

All configs inherit from BaseModuleConfig for type-safe validation,
deterministic hashing, and YAML serialization. Zero hardcoded values.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from src.templates.config import BaseModuleConfig

# ---------------------------------------------------------------------------
# Physical constants (single source of truth)
# ---------------------------------------------------------------------------

STANDARD_GRAVITY_MS2: float = 9.80665
"""Standard gravitational acceleration in m/s^2 (ISO 80000-3)."""

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ThreatType(str, Enum):
    """Classification of threat vehicle types."""

    BALLISTIC = "ballistic"
    CRUISE = "cruise"
    DRONE = "drone"
    HYPERSONIC = "hypersonic"


class InterceptorType(str, Enum):
    """Classification of interceptor vehicle types."""

    MISSILE = "missile"
    ROTOR_DRONE = "rotor_drone"


class GuidanceLawType(str, Enum):
    """Available guidance law implementations."""

    PN = "pn"
    APN = "apn"
    ZEM_ZEV = "zem_zev"
    MCTS_GUIDED = "mcts_guided"


class SensorType(str, Enum):
    """Sensor modalities."""

    RADAR = "radar"
    EO = "eo"
    IR = "ir"
    FUSED = "fused"


class AssignmentAlgorithm(str, Enum):
    """Swarm assignment solver algorithms."""

    HUNGARIAN = "hungarian"
    AUCTION = "auction"
    MCTS = "mcts"


class WindProfileType(str, Enum):
    """Wind profile models."""

    CONSTANT = "constant"
    LOGARITHMIC = "logarithmic"
    POWER_LAW = "power_law"


class GravityModel(str, Enum):
    """Gravity model fidelity levels."""

    CONSTANT = "constant"
    WGS84 = "wgs84"


class IntegrationMethod(str, Enum):
    """Numerical integration methods for 6-DOF dynamics."""

    EULER = "euler"
    RK4 = "rk4"


class EngagementPhase(str, Enum):
    """Phases of an intercept engagement."""

    LAUNCH = "launch"
    MIDCOURSE = "midcourse"
    TERMINAL = "terminal"
    POST_INTERCEPT = "post_intercept"


# ---------------------------------------------------------------------------
# Configuration classes
# ---------------------------------------------------------------------------


class ThreatConfig(BaseModuleConfig):
    """Configuration for a threat vehicle."""

    threat_type: ThreatType = Field(
        default=ThreatType.BALLISTIC,
        description="Threat vehicle classification.",
    )
    mass_kg: float = Field(
        default=200.0,
        gt=0.0,
        description="Threat mass in kilograms.",
    )
    reference_area_m2: float = Field(
        default=0.5,
        gt=0.0,
        description="Aerodynamic reference area in m^2.",
    )
    reference_length_m: float = Field(
        default=1.0,
        gt=0.0,
        description="Aerodynamic reference length in m.",
    )
    cd_0: float = Field(
        default=0.3,
        ge=0.0,
        description="Zero-lift drag coefficient.",
    )
    cl_alpha: float = Field(
        default=2.0,
        ge=0.0,
        description="Lift curve slope (per radian).",
    )
    max_g: float = Field(
        default=5.0,
        gt=0.0,
        description="Maximum structural g-load.",
    )
    max_aoa_rad: float = Field(
        default=0.5,
        gt=0.0,
        description="Maximum angle of attack in radians.",
    )
    has_motor: bool = Field(
        default=False,
        description="Whether threat has active propulsion.",
    )
    motor_thrust_n: float = Field(
        default=0.0,
        ge=0.0,
        description="Motor thrust in Newtons.",
    )
    motor_burn_time_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Motor burn duration in seconds.",
    )


class InterceptorConfig(BaseModuleConfig):
    """Configuration for an interceptor vehicle."""

    interceptor_type: InterceptorType = Field(
        default=InterceptorType.MISSILE,
        description="Interceptor vehicle classification.",
    )
    mass_kg: float = Field(
        default=50.0,
        gt=0.0,
        description="Interceptor mass in kilograms.",
    )
    reference_area_m2: float = Field(
        default=0.02,
        gt=0.0,
        description="Aerodynamic reference area in m^2.",
    )
    reference_length_m: float = Field(
        default=0.5,
        gt=0.0,
        description="Aerodynamic reference length in m.",
    )
    cd_0: float = Field(
        default=0.2,
        ge=0.0,
        description="Zero-lift drag coefficient.",
    )
    cl_alpha: float = Field(
        default=3.0,
        ge=0.0,
        description="Lift curve slope (per radian).",
    )
    max_g: float = Field(
        default=30.0,
        gt=0.0,
        description="Maximum structural g-load.",
    )
    max_speed_ms: float = Field(
        default=800.0,
        gt=0.0,
        description="Maximum speed in m/s.",
    )
    max_turn_rate_rads: float = Field(
        default=3.0,
        gt=0.0,
        description="Maximum turn rate in rad/s.",
    )
    motor_thrust_n: float = Field(
        default=5000.0,
        ge=0.0,
        description="Motor thrust in Newtons.",
    )
    motor_burn_time_s: float = Field(
        default=5.0,
        ge=0.0,
        description="Motor burn duration in seconds.",
    )
    fuel_mass_kg: float = Field(
        default=10.0,
        ge=0.0,
        description="Initial fuel mass in kilograms.",
    )
    seeker_fov_rad: float = Field(
        default=0.5,
        gt=0.0,
        description="Seeker field of view half-angle in radians.",
    )
    kill_radius_m: float = Field(
        default=2.0,
        gt=0.0,
        description="Kill radius in meters.",
    )
    specific_impulse_s: float = Field(
        default=250.0,
        gt=0.0,
        description="Motor specific impulse in seconds.",
    )
    fin_rate_limit_rads: float = Field(
        default=10.0,
        gt=0.0,
        description="Fin deflection rate limit in rad/s.",
    )
    fin_max_deflection_rad: float = Field(
        default=0.4,
        gt=0.0,
        description="Maximum fin deflection in radians.",
    )


class AtmosphereConfig(BaseModuleConfig):
    """Configuration for atmospheric model."""

    wind_profile: WindProfileType = Field(
        default=WindProfileType.CONSTANT,
        description="Wind profile model type.",
    )
    wind_speed_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Reference wind speed in m/s.",
    )
    wind_direction_rad: float = Field(
        default=0.0,
        description="Wind direction in radians (from North, clockwise).",
    )
    wind_reference_altitude_m: float = Field(
        default=10.0,
        gt=0.0,
        description="Reference altitude for wind profile in m.",
    )
    temperature_offset_k: float = Field(
        default=0.0,
        description="Temperature offset from ISA in Kelvin.",
    )
    log_wind_roughness_m: float = Field(
        default=0.03,
        gt=0.0,
        description="Surface roughness length for logarithmic wind profile (m).",
    )
    power_law_exponent: float = Field(
        default=0.143,
        gt=0.0,
        lt=1.0,
        description="Exponent for power-law wind profile.",
    )


class SensorConfig(BaseModuleConfig):
    """Configuration for sensor model."""

    sensor_type: SensorType = Field(
        default=SensorType.RADAR,
        description="Sensor modality.",
    )
    range_noise_m: float = Field(
        default=10.0,
        ge=0.0,
        description="Range measurement noise std dev in meters.",
    )
    azimuth_noise_rad: float = Field(
        default=0.01,
        ge=0.0,
        description="Azimuth measurement noise std dev in radians.",
    )
    elevation_noise_rad: float = Field(
        default=0.01,
        ge=0.0,
        description="Elevation measurement noise std dev in radians.",
    )
    update_rate_hz: float = Field(
        default=10.0,
        gt=0.0,
        description="Sensor update rate in Hz.",
    )
    max_range_m: float = Field(
        default=50000.0,
        gt=0.0,
        description="Maximum detection range in meters.",
    )
    fov_rad: float = Field(
        default=1.0,
        gt=0.0,
        description="Field of view half-angle in radians.",
    )
    range_uncertainty_fraction: float = Field(
        default=0.3,
        gt=0.0,
        lt=1.0,
        description="Fractional range uncertainty for bearing-only sensors.",
    )


class GuidanceConfig(BaseModuleConfig):
    """Configuration for guidance law."""

    law_type: GuidanceLawType = Field(
        default=GuidanceLawType.PN,
        description="Guidance law implementation.",
    )
    navigation_constant: float = Field(
        default=3.0,
        gt=0.0,
        le=10.0,
        description="Navigation constant N' for proportional navigation.",
    )
    terminal_range_m: float = Field(
        default=500.0,
        gt=0.0,
        description="Range threshold for terminal phase switch in meters.",
    )
    terminal_gain_multiplier: float = Field(
        default=2.0,
        gt=0.0,
        description="Gain multiplier during terminal phase.",
    )
    breakoff_miss_m: float = Field(
        default=50.0,
        gt=0.0,
        description="Predicted miss distance above which to break off.",
    )
    breakoff_tgo_s: float = Field(
        default=2.0,
        gt=0.0,
        description="Time-to-go threshold for break-off decision.",
    )
    max_acceleration_g: float = Field(
        default=30.0,
        gt=0.0,
        description="Maximum commanded acceleration in g's.",
    )


class MCTSInterceptConfig(BaseModuleConfig):
    """MCTS configuration for intercept guidance search."""

    n_simulations: int = Field(
        default=100,
        ge=1,
        description="Number of MCTS simulations per guidance step.",
    )
    c_puct: float = Field(
        default=1.5,
        gt=0.0,
        description="PUCT exploration constant.",
    )
    action_grid_size: int = Field(
        default=7,
        ge=3,
        description="Grid size per axis for acceleration discretization (total = grid^3 + 1).",
    )
    time_horizon_s: float = Field(
        default=5.0,
        gt=0.0,
        description="MCTS rollout time horizon in seconds.",
    )
    rollout_dt_s: float = Field(
        default=0.1,
        gt=0.0,
        description="Time step for MCTS rollout simulation.",
    )
    time_budget_ms: float = Field(
        default=15.0,
        gt=0.0,
        description="Maximum wall-clock time for MCTS search in ms.",
    )
    dirichlet_alpha: float = Field(
        default=0.25,
        gt=0.0,
        description="Dirichlet noise alpha for exploration.",
    )
    divergence_velocity_ms: float = Field(
        default=-100.0,
        lt=0.0,
        description="Closing velocity threshold for divergence detection (m/s, negative).",
    )
    divergence_min_steps: int = Field(
        default=5,
        ge=1,
        description="Minimum steps before divergence detection activates.",
    )


class AssignmentConfig(BaseModuleConfig):
    """Configuration for swarm assignment solver."""

    algorithm: AssignmentAlgorithm = Field(
        default=AssignmentAlgorithm.HUNGARIAN,
        description="Assignment solver algorithm.",
    )
    max_threats: int = Field(
        default=50,
        ge=1,
        description="Maximum number of threats to handle.",
    )
    max_interceptors: int = Field(
        default=20,
        ge=1,
        description="Maximum number of interceptors.",
    )
    time_budget_ms: float = Field(
        default=200.0,
        gt=0.0,
        description="Maximum time for assignment computation in ms.",
    )
    reassignment_interval_s: float = Field(
        default=1.0,
        gt=0.0,
        description="Minimum interval between reassignments in seconds.",
    )


class EdgeConfig(BaseModuleConfig):
    """Configuration for edge deployment."""

    target_device: Literal["cpu", "cuda", "tensorrt"] = Field(
        default="cpu",
        description="Target inference device.",
    )
    onnx_opset: int = Field(
        default=17,
        ge=11,
        le=20,
        description="ONNX opset version for export.",
    )
    quantization_bits: Literal[32, 16, 8] = Field(
        default=32,
        description="Quantization precision in bits.",
    )
    max_guidance_latency_ms: float = Field(
        default=20.0,
        gt=0.0,
        description="Maximum guidance loop latency in ms.",
    )
    max_prediction_latency_ms: float = Field(
        default=50.0,
        gt=0.0,
        description="Maximum prediction loop latency in ms.",
    )
    max_memory_mb: float = Field(
        default=2300.0,
        gt=0.0,
        description="Maximum memory budget in MB.",
    )


class DynamicsConfig(BaseModuleConfig):
    """Configuration for 6-DOF dynamics integration."""

    integration_method: IntegrationMethod = Field(
        default=IntegrationMethod.RK4,
        description="Numerical integration method.",
    )
    gravity_model: GravityModel = Field(
        default=GravityModel.CONSTANT,
        description="Gravity model fidelity.",
    )
    g0: float = Field(
        default=9.80665,
        ge=0.0,
        description="Standard gravitational acceleration in m/s^2.",
    )
    dt: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Integration time step in seconds.",
    )


class EngagementConfig(BaseModuleConfig):
    """Top-level engagement scenario configuration.

    Combines all sub-configs for a complete engagement simulation.
    """

    threat: ThreatConfig = Field(
        default_factory=lambda: ThreatConfig(name="threat"),
        description="Threat vehicle configuration.",
    )
    interceptor: InterceptorConfig = Field(
        default_factory=lambda: InterceptorConfig(name="interceptor"),
        description="Interceptor vehicle configuration.",
    )
    atmosphere: AtmosphereConfig = Field(
        default_factory=lambda: AtmosphereConfig(name="atmosphere"),
        description="Atmospheric model configuration.",
    )
    sensor: SensorConfig = Field(
        default_factory=lambda: SensorConfig(name="sensor"),
        description="Sensor model configuration.",
    )
    guidance: GuidanceConfig = Field(
        default_factory=lambda: GuidanceConfig(name="guidance"),
        description="Guidance law configuration.",
    )
    mcts: MCTSInterceptConfig = Field(
        default_factory=lambda: MCTSInterceptConfig(name="mcts"),
        description="MCTS search configuration.",
    )
    dynamics: DynamicsConfig = Field(
        default_factory=lambda: DynamicsConfig(name="dynamics"),
        description="Dynamics integration configuration.",
    )
    max_time_s: float = Field(
        default=60.0,
        gt=0.0,
        description="Maximum engagement duration in seconds.",
    )
    dt_s: float = Field(
        default=0.02,
        gt=0.0,
        le=1.0,
        description="Simulation time step in seconds.",
    )

    @model_validator(mode="after")
    def validate_engagement(self) -> EngagementConfig:
        """Cross-validate engagement parameters."""
        if self.dt_s > self.max_time_s:
            raise ValueError(f"dt_s ({self.dt_s}) must be <= max_time_s ({self.max_time_s})")
        return self
