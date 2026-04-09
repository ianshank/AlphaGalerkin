"""Tests for aerodynamic models."""

from __future__ import annotations

import pytest
import torch

from src.intercept.aero import (
    AeroModelRegistry,
    SimpleAeroModel,
    TabularAeroModel,
)
from src.intercept.atmosphere import ISAAtmosphere
from src.intercept.config import ThreatConfig
from src.intercept.dynamics import create_initial_state


class TestSimpleAeroModel:
    def setup_method(self) -> None:
        self.model = SimpleAeroModel(cd=0.3, reference_area=0.5, mass=100.0)
        self.atmo = ISAAtmosphere()

    def test_zero_velocity_zero_drag(self) -> None:
        state = create_initial_state(
            position=[0.0, 0.0, -1000.0],
            velocity=[0.0, 0.0, 0.0],
        )
        force, torque = self.model.compute_forces(state, self.atmo)
        assert torch.allclose(force, torch.zeros(3, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(torque, torch.zeros(3, dtype=torch.float64), atol=1e-10)

    def test_drag_opposes_velocity(self) -> None:
        state = create_initial_state(
            position=[0.0, 0.0, -1000.0],
            velocity=[100.0, 0.0, 0.0],  # moving North
        )
        force, torque = self.model.compute_forces(state, self.atmo)
        # Drag should be negative North (opposing velocity)
        assert force[0].item() < 0.0
        assert torch.allclose(force[1], torch.tensor(0.0, dtype=torch.float64), atol=1e-10)

    def test_drag_magnitude(self) -> None:
        """F_drag = 0.5 * rho * V^2 * Cd * Sref."""
        state = create_initial_state(
            position=[0.0, 0.0, -1000.0],
            velocity=[100.0, 0.0, 0.0],
        )
        force, _ = self.model.compute_forces(state, self.atmo)
        rho = self.atmo.density(state.altitude).item()
        expected = 0.5 * rho * 100.0**2 * 0.3 * 0.5
        assert abs(force[0].item()) == pytest.approx(expected, rel=0.01)

    def test_drag_increases_with_speed(self) -> None:
        state_slow = create_initial_state(position=[0.0, 0.0, -1000.0], velocity=[50.0, 0.0, 0.0])
        state_fast = create_initial_state(position=[0.0, 0.0, -1000.0], velocity=[200.0, 0.0, 0.0])
        f_slow, _ = self.model.compute_forces(state_slow, self.atmo)
        f_fast, _ = self.model.compute_forces(state_fast, self.atmo)
        assert torch.norm(f_fast) > torch.norm(f_slow)

    def test_drag_decreases_with_altitude(self) -> None:
        state_low = create_initial_state(position=[0.0, 0.0, -100.0], velocity=[100.0, 0.0, 0.0])
        state_high = create_initial_state(position=[0.0, 0.0, -10000.0], velocity=[100.0, 0.0, 0.0])
        f_low, _ = self.model.compute_forces(state_low, self.atmo)
        f_high, _ = self.model.compute_forces(state_high, self.atmo)
        assert torch.norm(f_low) > torch.norm(f_high)

    def test_max_g_load(self) -> None:
        model = SimpleAeroModel(max_g=10.0)
        state = create_initial_state()
        assert model.max_g_load(state).item() == 10.0

    def test_from_config(self) -> None:
        config = ThreatConfig(name="test", cd_0=0.5, reference_area_m2=1.0)
        model = SimpleAeroModel.from_config(config)
        assert model.cd == 0.5
        assert model.reference_area == 1.0


class TestTabularAeroModel:
    def setup_method(self) -> None:
        self.model = TabularAeroModel(cd_0=0.3, cl_alpha=2.0, reference_area=0.5, mass=100.0)
        self.atmo = ISAAtmosphere()

    def test_zero_velocity(self) -> None:
        state = create_initial_state(
            position=[0.0, 0.0, -1000.0],
            velocity=[0.0, 0.0, 0.0],
        )
        force, torque = self.model.compute_forces(state, self.atmo)
        # Should be small/zero with zero velocity
        assert torch.norm(force).item() < 1e-6

    def test_drag_present(self) -> None:
        state = create_initial_state(
            position=[0.0, 0.0, -1000.0],
            velocity=[100.0, 0.0, 0.0],
        )
        force, _ = self.model.compute_forces(state, self.atmo)
        # Should have negative North component (drag)
        assert force[0].item() < 0.0

    def test_from_config(self) -> None:
        config = ThreatConfig(name="test", cd_0=0.4, cl_alpha=3.0)
        model = TabularAeroModel.from_config(config)
        assert model.cd_0 == 0.4
        assert model.cl_alpha == 3.0

    def test_with_lookup_tables(self) -> None:
        alpha_table = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        mach_table = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        cd_table = torch.tensor(
            [[0.1, 0.15, 0.3], [0.05, 0.08, 0.2], [0.1, 0.15, 0.3]],
            dtype=torch.float64,
        )
        cl_table = torch.tensor(
            [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.float64,
        )
        model = TabularAeroModel(
            alpha_table=alpha_table,
            mach_table=mach_table,
            cd_table=cd_table,
            cl_table=cl_table,
            reference_area=0.5,
        )
        state = create_initial_state(
            position=[0.0, 0.0, -1000.0],
            velocity=[100.0, 0.0, 0.0],
        )
        force, _ = model.compute_forces(state, self.atmo)
        assert force.shape == (3,)


class TestAeroModelRegistry:
    def test_simple_registered(self) -> None:
        model_cls = AeroModelRegistry().get("simple")
        assert model_cls is SimpleAeroModel

    def test_tabular_registered(self) -> None:
        model_cls = AeroModelRegistry().get("tabular")
        assert model_cls is TabularAeroModel

    def test_list_models(self) -> None:
        models = AeroModelRegistry().list_items()
        assert "simple" in models
        assert "tabular" in models
