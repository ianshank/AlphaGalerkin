"""Coverage boost tests: two-temperature, state, viscous, NS, fire2."""

from __future__ import annotations

import numpy as np
import pytest

from src.reentry.config.gas import GasConfig
from src.reentry.config.mesh import ReentryMeshConfig
from src.reentry.config.solver import FluxScheme, ReentrySolverConfig
from src.reentry.gas.species import SpeciesData, get_species_data


def _gas_config() -> GasConfig:
    return GasConfig(
        name="test",
        species=["N2", "O2"],
        molecular_weights={"N2": 0.0280134, "O2": 0.0319988},
        formation_enthalpies={"N2": 0.0, "O2": 0.0},
        theta_v={"N2": 3395.0, "O2": 2239.0},
    )


class TestConservativeState:
    def test_basic_properties(self) -> None:
        from src.reentry.solver.state import ConservativeState

        s = ConservativeState(
            density=np.array([1.225]),
            momentum_x=np.array([1.225 * 100.0]),
            momentum_y=np.array([0.0]),
            total_energy=np.array([1.225 * 300000.0]),
        )
        assert s.n_cells == 1
        np.testing.assert_allclose(s.velocity_x(), [100.0])
        np.testing.assert_allclose(s.velocity_y(), [0.0])

    def test_with_species(self) -> None:
        from src.reentry.solver.state import ConservativeState

        s = ConservativeState(
            density=np.array([1.0, 2.0]),
            momentum_x=np.array([100.0, 200.0]),
            momentum_y=np.array([0.0, 0.0]),
            total_energy=np.array([1e6, 2e6]),
            species_density=np.array([[0.76, 0.24], [1.52, 0.48]]),
        )
        assert s.n_cells == 2
        assert s.species_density.shape == (2, 2)

    def test_derived_quantities(self) -> None:
        from src.reentry.solver.state import ConservativeState

        rho, u, v = 1.225, 100.0, 50.0
        e_total = 101325.0 / (0.4 * rho) + 0.5 * (u**2 + v**2)
        s = ConservativeState(
            density=np.array([rho]),
            momentum_x=np.array([rho * u]),
            momentum_y=np.array([rho * v]),
            total_energy=np.array([rho * e_total]),
        )
        ke = s.kinetic_energy()
        assert ke[0] > 0
        e_int = s.specific_internal_energy()
        assert e_int[0] > 0
        e_spec = s.specific_total_energy()
        np.testing.assert_allclose(e_spec, [e_total], rtol=1e-10)

    def test_to_array_roundtrip(self) -> None:
        from src.reentry.solver.state import ConservativeState

        s = ConservativeState(
            density=np.array([1.0]),
            momentum_x=np.array([100.0]),
            momentum_y=np.array([50.0]),
            total_energy=np.array([250000.0]),
        )
        arr = s.to_array()
        assert arr.shape[1] >= 4
        s2 = ConservativeState.from_array(arr, n_species=0)
        np.testing.assert_allclose(s2.density, s.density)

    def test_mass_fractions(self) -> None:
        from src.reentry.solver.state import ConservativeState

        s = ConservativeState(
            density=np.array([1.0]),
            momentum_x=np.array([100.0]),
            momentum_y=np.array([0.0]),
            total_energy=np.array([250000.0]),
            species_density=np.array([[0.76]]),
        )
        y = s.mass_fractions(n_species=2)
        # Should have shape (1, 2)
        assert y.shape == (1, 2)
        # Sum should be 1.0
        np.testing.assert_allclose(y.sum(axis=1), [1.0], rtol=1e-10)


class TestTwoTemperatureModel:
    def test_compute_temperatures(self) -> None:
        from src.reentry.gas.two_temperature import TwoTemperatureModel

        config = _gas_config()
        model = TwoTemperatureModel(config)
        n = 5
        density = np.ones(n) * 0.5
        e_total = np.ones(n) * 300000.0
        e_vib = np.ones(n) * 50000.0
        y = np.column_stack([np.full(n, 0.76), np.full(n, 0.24)])

        state = model.compute_temperatures(density, e_total, e_vib, y)
        assert np.all(state.t_tr > 0)
        assert np.all(state.t_ve > 0)

    def test_energy_exchange(self) -> None:
        from src.reentry.gas.two_temperature import TwoTemperatureModel

        config = _gas_config()
        model = TwoTemperatureModel(config)
        n = 3
        density = np.ones(n)
        t_tr = np.ones(n) * 5000.0
        t_ve = np.ones(n) * 2000.0
        y = np.column_stack([np.full(n, 0.76), np.full(n, 0.24)])

        q_tv = model.energy_exchange_rate(density, t_tr, t_ve, y)
        assert np.all(q_tv > 0)

    def test_vibrational_energy(self) -> None:
        from src.reentry.gas.two_temperature import TwoTemperatureModel

        config = _gas_config()
        model = TwoTemperatureModel(config)
        n = 3
        t_ve = np.array([1000.0, 3000.0, 5000.0])
        y = np.column_stack([np.full(n, 0.76), np.full(n, 0.24)])
        e = model.mixture_vibrational_energy(t_ve, y)
        assert e[2] > e[1] > e[0]


class TestSpeciesDataExtended:
    def test_cv_vib_array(self) -> None:
        config = _gas_config()
        data = get_species_data(config)
        t = np.array([1000.0, 3000.0, 5000.0])
        cv = data["N2"].cv_vib(t)
        assert cv.shape == (3,)
        assert np.all(cv > 0)

    def test_e_vib(self) -> None:
        config = _gas_config()
        data = get_species_data(config)
        e = data["N2"].e_vib(5000.0)
        assert e > 0

    def test_monatomic_no_vib(self) -> None:
        sp = SpeciesData("N", 0.0140067, 0.0, 0.0, 1)
        assert sp.cv_vib(3000.0) == 0.0
        assert sp.e_vib(3000.0) == 0.0
        assert sp.cv_rot == 0.0
        assert sp.cv_trans > 0


class TestViscousFlux:
    def test_uniform_flow(self) -> None:
        from src.reentry.flux.viscous import ViscousFlux

        vf = ViscousFlux(gamma=1.4, prandtl=0.72, r_specific=287.058)
        ny, nx = 10, 10
        rhs = vf.compute(
            np.ones((ny, nx)) * 1.225,
            np.ones((ny, nx)) * 100.0,
            np.zeros((ny, nx)),
            np.ones((ny, nx)) * 101325.0,
            np.ones((ny, nx)) * 1.8e-5,
            np.ones((ny, nx)) * 0.01,
            np.ones((ny, nx)) * 0.01,
        )
        assert rhs.shape == (ny, nx, 4)
        np.testing.assert_allclose(rhs, 0.0, atol=1e-6)


class TestNavierStokes:
    def test_ns_solver_runs(self) -> None:
        from src.reentry.config.freestream import FreestreamConfig
        from src.reentry.mesh.structured import StructuredMesh2D
        from src.reentry.solver.boundary import BoundaryFace, FreestreamBC
        from src.reentry.solver.navier_stokes import NavierStokes2DSolver

        mesh_cfg = ReentryMeshConfig(
            name="t",
            nx=10,
            ny=8,
            wall_clustering=False,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=0.5,
        )
        solver_cfg = ReentrySolverConfig(
            name="t",
            flux_scheme=FluxScheme.HLLC,
            cfl=0.2,
            max_iterations=10,
            enable_viscous=True,
        )
        mesh = StructuredMesh2D(mesh_cfg, n_ghost=2)
        solver = NavierStokes2DSolver(solver_cfg, mesh, gamma=1.4)

        fs = FreestreamConfig(
            name="t",
            mach=3.0,
            velocity_m_s=1000.0,
            density_kg_m3=0.5,
            temperature_K=250.0,
        )
        for face in BoundaryFace:
            solver.set_bc(face, FreestreamBC(fs))

        q0 = solver.initialize_uniform(0.5, 1000.0, 0.0, 50000.0)
        result = solver.solve(q0, t_final=0.0001)
        assert not np.any(np.isnan(result.density))


class TestFire2:
    def test_trajectory(self) -> None:
        from src.reentry.geometry.fire2 import fire2_trajectory

        traj = fire2_trajectory()
        assert traj.vehicle_name == "FIRE_II"
        assert len(traj.points) >= 5

    def test_geometry(self) -> None:
        from src.reentry.geometry.fire2 import SphereConeGeometry

        geom = SphereConeGeometry(nose_radius=0.9347, cone_half_angle_deg=33.0)
        assert geom.nose_radius == pytest.approx(0.9347)
        x, r = geom.surface_points(n_points=50)
        assert len(x) == 50
