"""Tests for HelicalHeatOperator (Leap 71 Noyron HX integration)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pde.config import PDEConfig, PDEType
from src.pde.geometry import GeometryConfig, GeometryType
from src.pde.geometry_picogk import PicoGKDomain
from src.pde.operators import HeatOperator
from src.pde.operators_picogk import HelicalHeatOperator

HELIX_R_MAJOR = 0.05
HELIX_R_MINOR = 0.012
HELIX_PITCH = 0.02
HELIX_N_TURNS = 3


def _make_pde_config(domain_dim: int = 3) -> PDEConfig:
    """Build a 3D PDE config with the analytical-helix geometry."""
    bbox_min = [-(HELIX_R_MAJOR + HELIX_R_MINOR), -(HELIX_R_MAJOR + HELIX_R_MINOR), 0.0]
    bbox_max = [
        HELIX_R_MAJOR + HELIX_R_MINOR,
        HELIX_R_MAJOR + HELIX_R_MINOR,
        HELIX_PITCH * HELIX_N_TURNS,
    ]
    return PDEConfig(
        name="helical_heat_test",
        pde_type=PDEType.HEAT,
        domain_dim=domain_dim,
        domain_min=bbox_min[:domain_dim],
        domain_max=bbox_max[:domain_dim],
        advection_coeff=[0.0] * domain_dim,
        geometry=GeometryConfig(
            geometry_type=GeometryType.PICOGK,
            sdf_kind="analytical_helix",
            helix_R_major=HELIX_R_MAJOR,
            helix_r_minor=HELIX_R_MINOR,
            helix_pitch=HELIX_PITCH,
            helix_n_turns=HELIX_N_TURNS,
        ),
    )


@pytest.fixture
def operator() -> HelicalHeatOperator:
    return HelicalHeatOperator(_make_pde_config())


class TestHelicalHeatOperatorConstruction:
    def test_inherits_heat_operator(self, operator: HelicalHeatOperator) -> None:
        assert isinstance(operator, HeatOperator)

    def test_holds_picogk_geometry(self, operator: HelicalHeatOperator) -> None:
        assert isinstance(operator.geometry, PicoGKDomain)
        assert operator.geometry.dim == 3

    def test_steady_state(self, operator: HelicalHeatOperator) -> None:
        # The v1 PoC uses the steady heat equation.
        assert operator.is_time_dependent is False

    def test_rejects_non_picogk_geometry(self) -> None:
        cfg = PDEConfig(
            name="bad",
            pde_type=PDEType.HEAT,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
        )
        with pytest.raises(ValueError, match="PICOGK"):
            HelicalHeatOperator(cfg)

    def test_rejects_2d_domain(self) -> None:
        cfg = PDEConfig(
            name="too_flat",
            pde_type=PDEType.HEAT,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            geometry=GeometryConfig(
                geometry_type=GeometryType.PICOGK,
                sdf_kind="analytical_helix",
            ),
        )
        with pytest.raises(ValueError, match="domain_dim=3"):
            HelicalHeatOperator(cfg)


class TestHelicalHeatOperatorSampling:
    def test_collocation_points_in_domain(self, operator: HelicalHeatOperator) -> None:
        points = operator.generate_collocation_points(256, seed=0)
        assert points.shape == (256, 3)
        # All collocation points must be inside the SDF (sdf <= 0).
        with torch.no_grad():
            mask = operator.geometry.contains_point(torch.from_numpy(points))
        assert bool(mask.all().item())

    def test_boundary_points_on_surface(self, operator: HelicalHeatOperator) -> None:
        points = operator.generate_boundary_points(64, seed=0)
        assert points.shape == (64, 3)
        with torch.no_grad():
            sdf = operator.geometry.sdf_evaluator.sdf(torch.from_numpy(points))
        # Newton-projected; tolerate the configured boundary tolerance.
        tol = operator.geometry.boundary_tolerance
        assert bool(sdf.abs().max().item() < tol)

    def test_sampling_deterministic_under_seed(self, operator: HelicalHeatOperator) -> None:
        a = operator.generate_collocation_points(64, seed=42)
        b = operator.generate_collocation_points(64, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_is_boundary_point(self, operator: HelicalHeatOperator) -> None:
        # A point sampled inside should not be reported as boundary.
        interior = operator.generate_collocation_points(32, seed=1)
        mask = operator.is_boundary_point(interior, tolerance=1e-6)
        assert not bool(np.asarray(mask).any())

    def test_is_boundary_point_tensor_path(self, operator: HelicalHeatOperator) -> None:
        """The Tensor branch must round-trip without numpy conversion."""
        interior_np = operator.generate_collocation_points(8, seed=3)
        interior = torch.from_numpy(interior_np)
        mask = operator.is_boundary_point(interior, tolerance=1e-6)
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool


class TestHelicalHeatOperatorPhysics:
    def test_residual_finite_on_random_u(self, operator: HelicalHeatOperator) -> None:
        coords_np = operator.generate_collocation_points(128, seed=2)
        coords = torch.from_numpy(coords_np).requires_grad_(True)
        # A simple harmonic test field; gradient must trace through coords.
        u = torch.sin(coords[:, 0]) + torch.sin(coords[:, 1]) + torch.sin(coords[:, 2])
        residual = operator.residual(u, coords)
        assert residual.values.shape == (128,)
        assert torch.isfinite(residual.values).all()
        assert np.isfinite(residual.l2_norm)

    def test_inner_dirichlet_boundary_value(self, operator: HelicalHeatOperator) -> None:
        coords = torch.zeros(8, 3)
        bv = operator.boundary_value(coords)
        # default config.boundary_value is 0.0
        assert torch.allclose(bv, torch.zeros_like(bv))

    def test_hot_cold_boundary_value(self) -> None:
        cfg = _make_pde_config()
        op = HelicalHeatOperator(cfg, boundary_mode="hot_cold", hot_value=2.5, cold_value=-1.5)
        z_max = HELIX_PITCH * HELIX_N_TURNS
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.1 * z_max],  # bottom: hot
                [0.0, 0.0, 0.9 * z_max],  # top: cold
            ]
        )
        bv = op.boundary_value(coords)
        assert float(bv[0]) == pytest.approx(2.5)
        assert float(bv[1]) == pytest.approx(-1.5)

    def test_hot_cold_boundary_value_numpy_path(self) -> None:
        cfg = _make_pde_config()
        op = HelicalHeatOperator(cfg, boundary_mode="hot_cold", hot_value=1.0, cold_value=0.0)
        z_max = HELIX_PITCH * HELIX_N_TURNS
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, z_max],
            ],
            dtype=np.float32,
        )
        bv = op.boundary_value(coords)
        assert isinstance(bv, np.ndarray)
        assert float(bv[0]) == pytest.approx(1.0)
        assert float(bv[1]) == pytest.approx(0.0)
