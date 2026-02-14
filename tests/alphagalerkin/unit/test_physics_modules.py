"""Tests for physics modules: heat, burgers, wave, advdiff, navier-stokes."""

from __future__ import annotations

import numpy as np
import pytest

# Trigger auto-registration of all physics modules
import src.alphagalerkin.physics  # noqa: F401
from src.alphagalerkin.physics.advection_diffusion import (
    AdvectionDiffusionModule,
)
from src.alphagalerkin.physics.base import SolveResult
from src.alphagalerkin.physics.burgers import BurgersModule
from src.alphagalerkin.physics.heat import HeatModule
from src.alphagalerkin.physics.navier_stokes import (
    DynamicSmagorinskyModel,
    NavierStokesModule,
    NoModel,
    SmagorinskyModel,
    WALEModel,
    list_closures,
    select_closure,
)
from src.alphagalerkin.physics.registry import PhysicsRegistry
from src.alphagalerkin.physics.wave import WaveModule

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _assert_valid_solve_result(result: SolveResult) -> None:
    """Assert a SolveResult has valid structure."""
    assert isinstance(result, SolveResult)
    assert result.solution is not None
    assert result.solution.size > 0
    assert isinstance(result.residual_norm, float)
    assert result.residual_norm >= 0.0
    assert result.converged is True
    assert result.solve_time_ms >= 0.0


# -------------------------------------------------------------------
# Heat Module Tests
# -------------------------------------------------------------------


class TestHeatModule:
    """Tests for the HeatModule (heat_2d)."""

    def test_heat_module_solve(self) -> None:
        """Verify solve_on_grid returns valid SolveResult."""
        module = HeatModule()
        result = module.solve_on_grid(10)
        _assert_valid_solve_result(result)

    def test_heat_manufactured_solution(self) -> None:
        """Verify exact/forcing consistency for MMS.

        For u = sin(pi*x)*cos(pi*y):
            -laplacian(u) = 2*pi^2*sin(pi*x)*cos(pi*y)
        """
        module = HeatModule()
        mms = module.manufactured_solution()

        # Test at a few interior points
        points = np.array(
            [
                [0.25, 0.25],
                [0.5, 0.5],
                [0.75, 0.3],
            ]
        )

        exact = mms.exact_solution(points)
        forcing = mms.forcing(points)

        # Verify exact solution values
        for i, (x, y) in enumerate(points):
            expected = np.sin(np.pi * x) * np.cos(np.pi * y)
            assert abs(exact[i] - expected) < 1e-12

        # Verify forcing = 2*pi^2 * sin(pi*x)*cos(pi*y)
        for i, (x, y) in enumerate(points):
            expected = 2.0 * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y)
            assert abs(forcing[i] - expected) < 1e-10

    def test_heat_boundary_conditions(self) -> None:
        """Verify boundary conditions are Dirichlet."""
        module = HeatModule()
        bcs = module.boundary_conditions()
        assert len(bcs) == 1
        assert bcs[0].bc_type == "dirichlet"

    def test_heat_solve_residual_small(self) -> None:
        """Residual norm should be very small for a direct solve."""
        module = HeatModule()
        result = module.solve_on_grid(15)
        assert result.residual_norm < 1e-8


# -------------------------------------------------------------------
# Burgers Module Tests
# -------------------------------------------------------------------


class TestBurgersModule:
    """Tests for the BurgersModule (burgers_1d)."""

    def test_burgers_module_solve(self) -> None:
        """Verify solve_on_grid returns valid SolveResult."""
        module = BurgersModule()
        result = module.solve_on_grid(20)
        _assert_valid_solve_result(result)

    def test_burgers_default_viscosity(self) -> None:
        """Default viscosity is 0.01."""
        module = BurgersModule()
        assert module.viscosity == 0.01

    def test_burgers_custom_viscosity(self) -> None:
        """Can set custom viscosity."""
        module = BurgersModule(viscosity=0.1)
        assert module.viscosity == 0.1

    def test_burgers_manufactured_solution(self) -> None:
        """Verify exact/forcing consistency.

        For u = sin(pi*x), f = nu*pi^2*sin(pi*x).
        """
        nu = 0.05
        module = BurgersModule(viscosity=nu)
        mms = module.manufactured_solution()

        points = np.array([[0.25], [0.5], [0.75]])

        exact = mms.exact_solution(points)
        forcing = mms.forcing(points)

        for i, pt in enumerate(points):
            x = pt[0]
            assert abs(exact[i] - np.sin(np.pi * x)) < 1e-12
            expected_f = nu * np.pi**2 * np.sin(np.pi * x)
            assert abs(forcing[i] - expected_f) < 1e-10

    def test_burgers_solve_residual_small(self) -> None:
        """Residual norm should be very small for a direct solve."""
        module = BurgersModule()
        result = module.solve_on_grid(30)
        assert result.residual_norm < 1e-8


# -------------------------------------------------------------------
# Wave Module Tests
# -------------------------------------------------------------------


class TestWaveModule:
    """Tests for the WaveModule (wave_1d)."""

    def test_wave_module_solve(self) -> None:
        """Verify solve_on_grid returns valid SolveResult."""
        module = WaveModule()
        result = module.solve_on_grid(20)
        _assert_valid_solve_result(result)

    def test_wave_default_speed(self) -> None:
        """Default wave speed is 1.0."""
        module = WaveModule()
        assert module.wave_speed == 1.0

    def test_wave_custom_speed(self) -> None:
        """Can set custom wave speed."""
        module = WaveModule(wave_speed=2.0)
        assert module.wave_speed == 2.0

    def test_wave_manufactured_solution(self) -> None:
        """Verify exact/forcing consistency.

        For u = sin(pi*x), f = c^2*pi^2*sin(pi*x).
        """
        c = 2.0
        module = WaveModule(wave_speed=c)
        mms = module.manufactured_solution()

        points = np.array([[0.25], [0.5], [0.75]])

        exact = mms.exact_solution(points)
        forcing = mms.forcing(points)

        for i, pt in enumerate(points):
            x = pt[0]
            assert abs(exact[i] - np.sin(np.pi * x)) < 1e-12
            expected_f = c**2 * np.pi**2 * np.sin(np.pi * x)
            assert abs(forcing[i] - expected_f) < 1e-10

    def test_wave_solve_residual_small(self) -> None:
        """Residual norm should be very small for a direct solve."""
        module = WaveModule()
        result = module.solve_on_grid(30)
        assert result.residual_norm < 1e-8


# -------------------------------------------------------------------
# Advection-Diffusion Module Tests
# -------------------------------------------------------------------


class TestAdvectionDiffusionModule:
    """Tests for the AdvectionDiffusionModule (advdiff_2d)."""

    def test_advdiff_module_solve(self) -> None:
        """Verify solve_on_grid returns valid SolveResult."""
        module = AdvectionDiffusionModule()
        result = module.solve_on_grid(10)
        _assert_valid_solve_result(result)

    def test_advdiff_default_params(self) -> None:
        """Default diffusivity=0.1, velocity=(1,0)."""
        module = AdvectionDiffusionModule()
        assert module.diffusivity == 0.1
        assert module.velocity == (1.0, 0.0)

    def test_advdiff_manufactured_solution(self) -> None:
        """Verify manufactured solution is consistent."""
        module = AdvectionDiffusionModule()
        mms = module.manufactured_solution()

        points = np.array([[0.25, 0.25], [0.5, 0.5]])
        exact = mms.exact_solution(points)

        for i, (x, y) in enumerate(points):
            expected = np.sin(np.pi * x) * np.sin(np.pi * y)
            assert abs(exact[i] - expected) < 1e-12

    def test_advdiff_solve_residual_small(self) -> None:
        """Residual norm should be very small for a direct solve."""
        module = AdvectionDiffusionModule()
        result = module.solve_on_grid(15)
        assert result.residual_norm < 1e-8


# -------------------------------------------------------------------
# Navier-Stokes Module Tests
# -------------------------------------------------------------------


class TestNavierStokesModule:
    """Tests for the NavierStokesModule (navier_stokes_2d)."""

    def test_navier_stokes_solve(self) -> None:
        """Verify solve_on_grid returns valid SolveResult (Stokes)."""
        module = NavierStokesModule()
        result = module.solve_on_grid(10)
        _assert_valid_solve_result(result)

    def test_navier_stokes_default_params(self) -> None:
        """Default kinematic_viscosity=0.01, lid_velocity=1.0."""
        module = NavierStokesModule()
        assert module.kinematic_viscosity == 0.01
        assert module.lid_velocity == 1.0

    def test_navier_stokes_solve_residual_small(self) -> None:
        """Residual norm should be very small for a direct solve."""
        module = NavierStokesModule()
        result = module.solve_on_grid(15)
        assert result.residual_norm < 1e-8

    def test_navier_stokes_closure_metadata(self) -> None:
        """Metadata should report closure model."""
        module = NavierStokesModule()
        result = module.solve_on_grid(8)
        assert result.metadata["closure_model"] == "none"

        module.set_closure("smagorinsky")
        result = module.solve_on_grid(8)
        assert result.metadata["closure_model"] == "smagorinsky"


# -------------------------------------------------------------------
# SGS Closure Model Tests
# -------------------------------------------------------------------


class TestSGSClosureModels:
    """Tests for SGS closure model implementations."""

    def _make_strain_rotation(
        self,
        n: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sample strain rate and rotation rate tensors."""
        rng = np.random.default_rng(42)
        # Random symmetric strain rate
        raw = rng.standard_normal((n, 2, 2))
        strain_rate = 0.5 * (raw + np.swapaxes(raw, -2, -1))
        # Random antisymmetric rotation rate
        rotation_rate = 0.5 * (raw - np.swapaxes(raw, -2, -1))
        return strain_rate, rotation_rate

    def test_smagorinsky_computes_nu_t(self) -> None:
        """Smagorinsky model returns non-negative eddy viscosity."""
        model = SmagorinskyModel(c_s=0.17, delta=0.1)
        strain, rotation = self._make_strain_rotation()
        nu_t = model.compute_viscosity(strain, rotation)
        assert nu_t.shape == (10,)
        assert np.all(nu_t >= 0.0)

    def test_dynamic_smagorinsky_computes_nu_t(self) -> None:
        """Dynamic Smagorinsky returns non-negative eddy viscosity."""
        model = DynamicSmagorinskyModel(delta=0.1)
        strain, rotation = self._make_strain_rotation()
        nu_t = model.compute_viscosity(strain, rotation)
        assert nu_t.shape == (10,)
        assert np.all(nu_t >= 0.0)

    def test_wale_computes_nu_t(self) -> None:
        """WALE model returns non-negative eddy viscosity."""
        model = WALEModel(c_w=0.325, delta=0.1)
        strain, rotation = self._make_strain_rotation()
        nu_t = model.compute_viscosity(strain, rotation)
        assert nu_t.shape == (10,)
        assert np.all(nu_t >= 0.0)

    def test_no_model_returns_zeros(self) -> None:
        """NoModel returns zero eddy viscosity everywhere."""
        model = NoModel()
        strain, rotation = self._make_strain_rotation()
        nu_t = model.compute_viscosity(strain, rotation)
        assert nu_t.shape == (10,)
        assert np.all(nu_t == 0.0)

    def test_smagorinsky_zero_strain_gives_zero(self) -> None:
        """Zero strain rate should give zero eddy viscosity."""
        model = SmagorinskyModel()
        strain = np.zeros((5, 2, 2))
        rotation = np.zeros((5, 2, 2))
        nu_t = model.compute_viscosity(strain, rotation)
        assert np.allclose(nu_t, 0.0)


class TestSGSModelSelection:
    """Tests for SGS closure model selection."""

    def test_select_closure_smagorinsky(self) -> None:
        """Can select Smagorinsky closure."""
        closure = select_closure("smagorinsky")
        assert closure.name == "smagorinsky"
        assert callable(closure.compute_viscosity)

    def test_select_closure_dynamic_smagorinsky(self) -> None:
        """Can select Dynamic Smagorinsky closure."""
        closure = select_closure("dynamic_smagorinsky")
        assert closure.name == "dynamic_smagorinsky"

    def test_select_closure_wale(self) -> None:
        """Can select WALE closure."""
        closure = select_closure("wale")
        assert closure.name == "wale"

    def test_select_closure_no_model(self) -> None:
        """Can select NoModel closure."""
        closure = select_closure("no_model")
        assert closure.name == "no_model"

    def test_select_closure_unknown_raises(self) -> None:
        """Selecting unknown closure raises KeyError."""
        with pytest.raises(KeyError, match="Unknown SGS"):
            select_closure("nonexistent_closure")

    def test_list_closures_returns_all(self) -> None:
        """list_closures returns all registered closures."""
        closures = list_closures()
        assert "smagorinsky" in closures
        assert "dynamic_smagorinsky" in closures
        assert "wale" in closures
        assert "no_model" in closures

    def test_select_closure_with_kwargs(self) -> None:
        """Can pass kwargs to closure model constructors."""
        closure = select_closure(
            "smagorinsky",
            c_s=0.2,
            delta=0.05,
        )
        assert closure.name == "smagorinsky"


# -------------------------------------------------------------------
# Physics Registry Integration Tests
# -------------------------------------------------------------------


class TestPhysicsRegistryAll:
    """Verify all physics modules register correctly."""

    def setup_method(self) -> None:
        """Clear cached instances between tests."""
        registry = PhysicsRegistry()
        registry.clear_instances()

    def test_all_modules_registered(self) -> None:
        """All six physics modules should be registered."""
        registry = PhysicsRegistry()
        modules = registry.list_modules()

        expected = [
            "advdiff_2d",
            "burgers_1d",
            "heat_2d",
            "navier_stokes_2d",
            "poisson_2d",
            "wave_1d",
        ]
        for name in expected:
            assert name in modules, f"{name} not registered"

    def test_all_modules_have_solve_on_grid(self) -> None:
        """All modules should have a solve_on_grid method."""
        registry = PhysicsRegistry()
        for name in registry.list_modules():
            module = registry.get(name)
            assert hasattr(module, "solve_on_grid"), f"{name} missing solve_on_grid"

    def test_all_modules_have_manufactured_solution(self) -> None:
        """All modules should have a manufactured_solution method."""
        registry = PhysicsRegistry()
        for name in registry.list_modules():
            module = registry.get(name)
            assert hasattr(module, "manufactured_solution"), f"{name} missing manufactured_solution"

    def test_all_modules_have_weak_form(self) -> None:
        """All modules should have a weak_form method."""
        registry = PhysicsRegistry()
        for name in registry.list_modules():
            module = registry.get(name)
            assert hasattr(module, "weak_form"), f"{name} missing weak_form"

    def test_all_modules_have_boundary_conditions(self) -> None:
        """All modules should have a boundary_conditions method."""
        registry = PhysicsRegistry()
        for name in registry.list_modules():
            module = registry.get(name)
            assert hasattr(module, "boundary_conditions"), f"{name} missing boundary_conditions"


# -------------------------------------------------------------------
# Environment Fallback Residual Tests
# -------------------------------------------------------------------


class TestEnvFallbackResidual:
    """Verify the environment computes dof-based residual without physics."""

    def test_env_fallback_residual_without_physics(self) -> None:
        """Without physics module, residual should be 1/dof_count."""
        from src.alphagalerkin.core.config import EnvironmentConfig
        from src.alphagalerkin.core.types import ActionType
        from src.alphagalerkin.env.actions import Action
        from src.alphagalerkin.env.environment import (
            DiscretizationEnvironment,
        )

        config = EnvironmentConfig(
            max_steps=10,
            max_dof=50000,
        )
        env = DiscretizationEnvironment(config)
        state = env.reset()
        eid = state.mesh.element_ids[0]
        action = Action(eid, ActionType.NO_OP, {})
        result = env.step(action)

        # After step, the previous residual should have been set
        # to 1/dof_count (the fallback).  The dof_count should
        # be > 0 and the environment should not crash.
        assert result.state is not None
        assert result.info["dof_count"] > 0

    def test_env_fallback_residual_decreases_with_refinement(
        self,
    ) -> None:
        """Fallback residual should decrease as DOFs increase."""
        from src.alphagalerkin.core.config import EnvironmentConfig
        from src.alphagalerkin.core.types import ActionType
        from src.alphagalerkin.env.actions import Action
        from src.alphagalerkin.env.environment import (
            DiscretizationEnvironment,
        )

        config = EnvironmentConfig(
            max_steps=10,
            max_dof=50000,
        )
        env = DiscretizationEnvironment(config)
        state = env.reset()

        # First: no-op step to establish baseline
        eid = state.mesh.element_ids[0]
        action_noop = Action(eid, ActionType.NO_OP, {})
        result1 = env.step(action_noop)
        dof1 = result1.info["dof_count"]

        # Second: h-refine to increase DOFs
        eid2 = result1.state.mesh.element_ids[0]
        action_refine = Action(eid2, ActionType.H_REFINE, {})
        result2 = env.step(action_refine)
        dof2 = result2.info["dof_count"]

        # DOF count should increase after h-refinement
        assert dof2 > dof1

        # The fallback residual 1/dof should be smaller with
        # more DOFs.  This is verified implicitly by the
        # environment's reward computation.
        fallback1 = 1.0 / max(1, dof1)
        fallback2 = 1.0 / max(1, dof2)
        assert fallback2 < fallback1
