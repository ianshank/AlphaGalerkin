"""Shared fixtures for the PDE test suite.

Concentrates the helical-tube SDF parameters used by the Leap 71 / PicoGK
integration tests so they live in exactly one place. Individual test
modules can override any of these via `pytest.fixture(...)` overrides if
they need a different geometry regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class HelixParams:
    """Standard helical-tube geometry used by the SDF / domain / operator tests.

    Values mirror the Leap 71 downloadable HX scale (centimetre-class) but
    use a slightly larger r/R ratio than production defaults so the Newton
    projector has a forgiving curvature regime in unit tests.
    """

    R_major: float = 0.05
    r_minor: float = 0.012
    pitch: float = 0.02
    n_turns: int = 3

    @property
    def z_max(self) -> float:
        return self.pitch * self.n_turns

    @property
    def bbox_min(self) -> tuple[float, float, float]:
        return (-(self.R_major + self.r_minor), -(self.R_major + self.r_minor), 0.0)

    @property
    def bbox_max(self) -> tuple[float, float, float]:
        return (self.R_major + self.r_minor, self.R_major + self.r_minor, self.z_max)


@pytest.fixture
def helix_params() -> HelixParams:
    """Default helical-tube parameters used across the PDE test suite."""
    return HelixParams()
