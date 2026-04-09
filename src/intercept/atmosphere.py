"""International Standard Atmosphere (ISA) model and wind profiles.

Provides density, pressure, temperature, and speed of sound as functions
of altitude. All operations are vectorized PyTorch tensor operations
for batched computation and GPU acceleration.

Reference: ICAO Standard Atmosphere (ISO 2533:1975).
"""

from __future__ import annotations

import math

import structlog
import torch
from torch import Tensor

from src.intercept.config import AtmosphereConfig, WindProfileType

logger = structlog.get_logger(__name__)

# ISA constants
ISA_T0 = 288.15  # Sea-level temperature (K)
ISA_P0 = 101325.0  # Sea-level pressure (Pa)
ISA_RHO0 = 1.225  # Sea-level density (kg/m^3)
ISA_LAPSE_RATE = -0.0065  # Temperature lapse rate (K/m) in troposphere
ISA_TROPOPAUSE_ALT = 11000.0  # Tropopause altitude (m)
ISA_TROPOPAUSE_T = 216.65  # Tropopause temperature (K)
ISA_R = 287.05287  # Specific gas constant for dry air (J/(kg·K))
ISA_GAMMA = 1.4  # Ratio of specific heats for air
ISA_G0 = 9.80665  # Standard gravity (m/s^2)


class ISAAtmosphere:
    """International Standard Atmosphere model.

    Computes atmospheric properties as functions of geometric altitude.
    Supports troposphere (0-11 km) and lower stratosphere (11-20 km).
    """

    def __init__(
        self,
        config: AtmosphereConfig | None = None,
    ) -> None:
        self.config = config or AtmosphereConfig(name="default_atmosphere")
        self._temp_offset = self.config.temperature_offset_k

    def temperature(self, altitude_m: Tensor) -> Tensor:
        """Compute temperature at given altitude(s).

        Args:
            altitude_m: Geometric altitude in meters (...).

        Returns:
            Temperature in Kelvin (...).

        """
        alt = torch.clamp(altitude_m, min=0.0)
        tropo_t = ISA_T0 + ISA_LAPSE_RATE * alt + self._temp_offset
        strato_t = torch.tensor(
            ISA_TROPOPAUSE_T + self._temp_offset,
            device=alt.device,
            dtype=alt.dtype,
        )
        return torch.where(
            alt <= ISA_TROPOPAUSE_ALT,
            tropo_t,
            strato_t.expand_as(tropo_t),
        )

    def pressure(self, altitude_m: Tensor) -> Tensor:
        """Compute pressure at given altitude(s).

        Args:
            altitude_m: Geometric altitude in meters (...).

        Returns:
            Pressure in Pascals (...).

        """
        alt = torch.clamp(altitude_m, min=0.0)
        temp = self.temperature(alt)

        # Troposphere: P = P0 * (T/T0)^(g0 / (R * lapse))
        exponent = ISA_G0 / (ISA_R * (-ISA_LAPSE_RATE))
        t0_adj = ISA_T0 + self._temp_offset
        tropo_p = ISA_P0 * (temp / t0_adj) ** exponent

        # Stratosphere: P = P_tropo * exp(-g0 * (h - h_tropo) / (R * T_tropo))
        tropo_alt = torch.tensor(ISA_TROPOPAUSE_ALT, device=alt.device, dtype=alt.dtype)
        tropo_temp = ISA_TROPOPAUSE_T + self._temp_offset
        p_tropo = ISA_P0 * (tropo_temp / t0_adj) ** exponent
        strato_p = p_tropo * torch.exp(-ISA_G0 * (alt - tropo_alt) / (ISA_R * tropo_temp))

        return torch.where(alt <= ISA_TROPOPAUSE_ALT, tropo_p, strato_p)

    def density(self, altitude_m: Tensor) -> Tensor:
        """Compute air density at given altitude(s).

        Uses ideal gas law: rho = P / (R * T).

        Args:
            altitude_m: Geometric altitude in meters (...).

        Returns:
            Density in kg/m^3 (...).

        """
        return self.pressure(altitude_m) / (ISA_R * self.temperature(altitude_m))

    def speed_of_sound(self, altitude_m: Tensor) -> Tensor:
        """Compute speed of sound at given altitude(s).

        a = sqrt(gamma * R * T).

        Args:
            altitude_m: Geometric altitude in meters (...).

        Returns:
            Speed of sound in m/s (...).

        """
        return torch.sqrt(ISA_GAMMA * ISA_R * self.temperature(altitude_m))

    def mach_number(self, speed_ms: Tensor, altitude_m: Tensor) -> Tensor:
        """Compute Mach number.

        Args:
            speed_ms: Speed in m/s (...).
            altitude_m: Altitude in meters (...).

        Returns:
            Mach number (...).

        """
        return speed_ms / self.speed_of_sound(altitude_m)

    def dynamic_pressure(self, speed_ms: Tensor, altitude_m: Tensor) -> Tensor:
        """Compute dynamic pressure q = 0.5 * rho * V^2.

        Args:
            speed_ms: Speed in m/s (...).
            altitude_m: Altitude in meters (...).

        Returns:
            Dynamic pressure in Pa (...).

        """
        return 0.5 * self.density(altitude_m) * speed_ms**2


class WindModel:
    """Configurable wind profile model.

    Supports constant, logarithmic, and power-law wind profiles.
    All outputs are in NED frame.
    """

    def __init__(self, config: AtmosphereConfig | None = None) -> None:
        self.config = config or AtmosphereConfig(name="default_wind")

    def get_wind(
        self,
        altitude_m: Tensor,
        time_s: Tensor | None = None,
    ) -> Tensor:
        """Compute wind vector in NED frame at given altitude(s).

        Args:
            altitude_m: Altitude in meters (...).
            time_s: Time in seconds (unused for steady wind).

        Returns:
            Wind vector in NED [north, east, down] in m/s (..., 3).

        """
        base_speed = self.config.wind_speed_ms
        direction = self.config.wind_direction_rad

        if base_speed == 0.0:
            return torch.zeros(
                *altitude_m.shape, 3, device=altitude_m.device, dtype=altitude_m.dtype
            )

        # Compute speed profile
        if self.config.wind_profile == WindProfileType.CONSTANT:
            speed = torch.full_like(altitude_m, base_speed)
        elif self.config.wind_profile == WindProfileType.LOGARITHMIC:
            z0 = 0.03  # roughness length (m) for open terrain
            ref_alt = self.config.wind_reference_altitude_m
            alt_clamped = torch.clamp(altitude_m, min=z0 + 0.1)
            speed = base_speed * (torch.log(alt_clamped / z0) / math.log(ref_alt / z0))
        else:  # POWER_LAW
            ref_alt = self.config.wind_reference_altitude_m
            alpha = 0.143  # power law exponent for open terrain
            alt_clamped = torch.clamp(altitude_m, min=0.1)
            speed = base_speed * (alt_clamped / ref_alt) ** alpha

        # Convert speed + direction to NED components
        # Direction is "from" angle clockwise from North
        wind_north = -speed * math.cos(direction)
        wind_east = -speed * math.sin(direction)
        wind_down = torch.zeros_like(speed)

        return torch.stack([wind_north, wind_east, wind_down], dim=-1)
