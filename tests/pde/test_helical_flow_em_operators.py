"""Tests for HelicalStokesOperator and HelicalMagnetostaticsOperator.

These operators cover the v2.3 (Noyron RP coolant flow) and v3.1
(Noyron EA actuator magnetostatics) expansion items. Both follow the
same SDF-aware override pattern as ``HelicalHeatOperator``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pde.config import PDEConfig, PDEType
from src.pde.geometry import GeometryConfig, GeometryType
from src.pde.geometry_picogk import PicoGKDomain
from src.pde.operators import PDEOperator
from src.pde.operators_picogk import (
    HelicalMagnetostaticsOperator,
    HelicalStokesOperator,
)

HELIX_R_MAJOR = 0.05
HELIX_R_MINOR = 0.012
HELIX_PITCH = 0.02
HELIX_N_TURNS = 2


def _make_picogk_pde_config(pde_type: PDEType) -> PDEConfig:
    bbox_min = [-(HELIX_R_MAJOR + HELIX_R_MINOR), -(HELIX_R_MAJOR + HELIX_R_MINOR), 0.0]
    bbox_max = [
        HELIX_R_MAJOR + HELIX_R_MINOR,
        HELIX_R_MAJOR + HELIX_R_MINOR,
        HELIX_PITCH * HELIX_N_TURNS,
    ]
    return PDEConfig(
        name="helical_test",
        pde_type=pde_type,
        domain_dim=3,
        domain_min=bbox_min,
        domain_max=bbox_max,
        advection_coeff=[0.0, 0.0, 0.0],
        geometry=GeometryConfig(
            geometry_type=GeometryType.PICOGK,
            sdf_kind="analytical_helix",
            helix_R_major=HELIX_R_MAJOR,
            helix_r_minor=HELIX_R_MINOR,
            helix_pitch=HELIX_PITCH,
            helix_n_turns=HELIX_N_TURNS,
        ),
    )


# ---------------------------------------------------------------------------
# HelicalStokesOperator
# ---------------------------------------------------------------------------


class TestHelicalStokesOperator:
    @pytest.fixture
    def operator(self) -> HelicalStokesOperator:
        return HelicalStokesOperator(_make_picogk_pde_config(PDEType.NAVIER_STOKES))

    def test_inherits_pde_operator(self, operator: HelicalStokesOperator) -> None:
        assert isinstance(operator, PDEOperator)

    def test_holds_picogk_geometry(self, operator: HelicalStokesOperator) -> None:
        assert isinstance(operator.geometry, PicoGKDomain)
        assert operator.geometry.dim == 3

    def test_steady_and_linear(self, operator: HelicalStokesOperator) -> None:
        # Stokes (no convective term) is steady and linear by construction.
        assert operator.is_time_dependent is False
        assert operator.is_linear is True

    def test_rejects_non_picogk_geometry(self) -> None:
        cfg = PDEConfig(
            name="bad",
            pde_type=PDEType.NAVIER_STOKES,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
        )
        with pytest.raises(ValueError, match="PICOGK"):
            HelicalStokesOperator(cfg)

    def test_rejects_non_3d_domain(self) -> None:
        cfg = PDEConfig(
            name="bad",
            pde_type=PDEType.NAVIER_STOKES,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            geometry=GeometryConfig(geometry_type=GeometryType.PICOGK, sdf_kind="analytical_helix"),
        )
        with pytest.raises(ValueError, match="domain_dim=3"):
            HelicalStokesOperator(cfg)

    def test_rejects_non_positive_viscosity(self) -> None:
        cfg = _make_picogk_pde_config(PDEType.NAVIER_STOKES)
        with pytest.raises(ValueError, match="viscosity"):
            HelicalStokesOperator(cfg, viscosity=0.0)

    def test_collocation_points_inside_domain(self, operator: HelicalStokesOperator) -> None:
        points = operator.generate_collocation_points(64, seed=0)
        assert points.shape == (64, 3)
        assert bool(operator.geometry.contains_point(torch.from_numpy(points)).all().item())

    def test_boundary_points_on_surface(self, operator: HelicalStokesOperator) -> None:
        points = operator.generate_boundary_points(32, seed=0)
        assert points.shape == (32, 3)
        sdf = operator.geometry.sdf_evaluator.sdf(torch.from_numpy(points))
        assert bool(sdf.abs().max().item() < operator.geometry.boundary_tolerance)

    def test_no_slip_boundary_value(self, operator: HelicalStokesOperator) -> None:
        coords = torch.zeros(8, 3)
        bv = operator.boundary_value(coords)
        assert torch.allclose(bv, torch.zeros_like(bv))

    def test_zero_source_term(self, operator: HelicalStokesOperator) -> None:
        coords = torch.randn(8, 3)
        s = operator.source_term(coords)
        assert torch.allclose(s, torch.zeros_like(s))

    def test_zero_source_term_numpy_path(self, operator: HelicalStokesOperator) -> None:
        coords_np = np.zeros((4, 3), dtype=np.float32)
        s = operator.source_term(coords_np)
        assert isinstance(s, np.ndarray)
        np.testing.assert_array_equal(s, np.zeros(4, dtype=np.float32))

    def test_residual_finite_on_random_field(self, operator: HelicalStokesOperator) -> None:
        coords_np = operator.generate_collocation_points(32, seed=1)
        coords = torch.from_numpy(coords_np).requires_grad_(True)
        u = torch.sin(coords[:, 0]) + torch.sin(coords[:, 1]) + torch.sin(coords[:, 2])
        residual = operator.residual(u, coords)
        assert residual.values.shape == (32,)
        assert torch.isfinite(residual.values).all()

    def test_is_boundary_point_numpy_path(self, operator: HelicalStokesOperator) -> None:
        interior_np = operator.generate_collocation_points(8, seed=2)
        mask = operator.is_boundary_point(interior_np, tolerance=1e-6)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_

    def test_no_slip_boundary_value_numpy_path(self, operator: HelicalStokesOperator) -> None:
        coords_np = np.zeros((6, 3), dtype=np.float32)
        bv = operator.boundary_value(coords_np)
        assert isinstance(bv, np.ndarray)
        np.testing.assert_array_equal(bv, np.zeros(6, dtype=np.float32))


# ---------------------------------------------------------------------------
# HelicalMagnetostaticsOperator
# ---------------------------------------------------------------------------


class TestHelicalMagnetostaticsOperator:
    @pytest.fixture
    def operator(self) -> HelicalMagnetostaticsOperator:
        return HelicalMagnetostaticsOperator(_make_picogk_pde_config(PDEType.POISSON))

    def test_inherits_pde_operator(self, operator: HelicalMagnetostaticsOperator) -> None:
        assert isinstance(operator, PDEOperator)

    def test_holds_picogk_geometry(self, operator: HelicalMagnetostaticsOperator) -> None:
        assert isinstance(operator.geometry, PicoGKDomain)
        assert operator.geometry.dim == 3

    def test_rejects_non_picogk_geometry(self) -> None:
        cfg = PDEConfig(
            name="bad",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
        )
        with pytest.raises(ValueError, match="PICOGK"):
            HelicalMagnetostaticsOperator(cfg)

    def test_rejects_non_positive_permeability(self) -> None:
        cfg = _make_picogk_pde_config(PDEType.POISSON)
        with pytest.raises(ValueError, match="permeability"):
            HelicalMagnetostaticsOperator(cfg, permeability=0.0)

    def test_constant_current_source(self, operator: HelicalMagnetostaticsOperator) -> None:
        coords = torch.zeros(6, 3)
        s = operator.source_term(coords)
        assert torch.allclose(s, torch.full_like(s, operator.current_density))

    def test_zero_dirichlet_far_field(self, operator: HelicalMagnetostaticsOperator) -> None:
        coords = torch.zeros(4, 3)
        bv = operator.boundary_value(coords)
        assert torch.allclose(bv, torch.zeros_like(bv))

    def test_residual_finite_on_random_field(self, operator: HelicalMagnetostaticsOperator) -> None:
        coords_np = operator.generate_collocation_points(32, seed=1)
        coords = torch.from_numpy(coords_np).requires_grad_(True)
        u = torch.sin(coords[:, 0]) + torch.sin(coords[:, 1]) + torch.sin(coords[:, 2])
        residual = operator.residual(u, coords)
        assert residual.values.shape == (32,)
        assert torch.isfinite(residual.values).all()

    def test_constant_current_source_numpy_path(
        self, operator: HelicalMagnetostaticsOperator
    ) -> None:
        coords_np = np.zeros((4, 3), dtype=np.float32)
        s = operator.source_term(coords_np)
        assert isinstance(s, np.ndarray)
        np.testing.assert_allclose(s, np.full(4, operator.current_density, dtype=np.float32))

    def test_zero_dirichlet_far_field_numpy_path(
        self, operator: HelicalMagnetostaticsOperator
    ) -> None:
        coords_np = np.zeros((4, 3), dtype=np.float32)
        bv = operator.boundary_value(coords_np)
        assert isinstance(bv, np.ndarray)
        np.testing.assert_array_equal(bv, np.zeros(4, dtype=np.float32))

    def test_is_boundary_point_tensor_path(self, operator: HelicalMagnetostaticsOperator) -> None:
        interior_np = operator.generate_collocation_points(8, seed=2)
        interior = torch.from_numpy(interior_np)
        mask = operator.is_boundary_point(interior, tolerance=1e-6)
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool

    def test_is_boundary_point_numpy_path(self, operator: HelicalMagnetostaticsOperator) -> None:
        interior_np = operator.generate_collocation_points(8, seed=3)
        mask = operator.is_boundary_point(interior_np, tolerance=1e-6)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_


# ---------------------------------------------------------------------------
# Operator-registry coverage
# ---------------------------------------------------------------------------


class TestRegistryRegistration:
    """The three SDF-aware helical operators must be registered by name."""

    @pytest.mark.parametrize(
        "name",
        ["helical_heat", "helical_stokes", "helical_magnetostatics"],
    )
    def test_lookup_returns_correct_class(self, name: str) -> None:
        from src.pde.registry import PDEOperatorRegistry

        cls = PDEOperatorRegistry().get(name)
        assert cls is not None
        assert cls.__name__.startswith("Helical")
