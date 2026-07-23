"""Tests for the out-of-distribution operators: Helmholtz and Biharmonic.

These two operators provide the "held-out generalisation" benchmark for the
LLM-prior MCTS ablation. The tests cover operator properties, the
manufactured source/boundary/exact-solution analytics, residual smallness on
the exact solution (the correctness contract), wavenumber resolution, registry
round-trip, and end-to-end construction of a ``BasisSelectionGame`` (the MCTS
entry point).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pde.config import (
    BasisSelectionConfig,
    BoundaryCondition,
    PDEConfig,
    PDEGameConfig,
    PDEType,
)
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.mcts_adapter import PDEGameAdapter
from src.pde.operators import (
    DEFAULT_HELMHOLTZ_WAVENUMBER,
    BiharmonicOperator,
    HelmholtzOperator,
    PDEResidual,
)
from src.pde.registry import get_pde_operator, list_pde_operators


def _config(pde_type: PDEType, name: str, **overrides: object) -> PDEConfig:
    return PDEConfig(
        name=name,
        pde_type=pde_type,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        boundary_condition=BoundaryCondition.DIRICHLET,
        boundary_value=0.0,
        **overrides,  # type: ignore[arg-type]
    )


def _connected_coords(points: np.ndarray) -> torch.Tensor:
    """Tensor coords with grad enabled so the manufactured solution connects."""
    return torch.tensor(points, dtype=torch.float32, requires_grad=True)


# --------------------------------------------------------------------------- #
# Helmholtz                                                                    #
# --------------------------------------------------------------------------- #


class TestHelmholtzOperator:
    """Tests for the Helmholtz operator ∇²u + k²u = f."""

    @pytest.fixture
    def operator(self) -> HelmholtzOperator:
        return HelmholtzOperator(_config(PDEType.HELMHOLTZ, "test_helmholtz"))

    def test_operator_properties(self, operator: HelmholtzOperator) -> None:
        assert operator.name == "helmholtz"
        assert operator.pde_type is PDEType.HELMHOLTZ
        assert operator.is_time_dependent is False
        assert operator.is_linear is True
        assert operator.order == 2
        assert operator.wavenumber == DEFAULT_HELMHOLTZ_WAVENUMBER

    def test_wavenumber_explicit_argument_wins(self) -> None:
        op = HelmholtzOperator(_config(PDEType.HELMHOLTZ, "k"), wavenumber=3.5)
        assert op.wavenumber == 3.5

    def test_wavenumber_falls_back_to_reaction_coeff(self) -> None:
        # reaction_coeff is interpreted as k², so k = sqrt(4) = 2.
        op = HelmholtzOperator(_config(PDEType.HELMHOLTZ, "k", reaction_coeff=4.0))
        assert op.wavenumber == pytest.approx(2.0)

    def test_wavenumber_default_when_unspecified(self) -> None:
        op = HelmholtzOperator(_config(PDEType.HELMHOLTZ, "k"))
        assert op.wavenumber == DEFAULT_HELMHOLTZ_WAVENUMBER

    def test_non_positive_wavenumber_raises(self) -> None:
        with pytest.raises(ValueError, match="wavenumber must be positive"):
            HelmholtzOperator(_config(PDEType.HELMHOLTZ, "k"), wavenumber=0.0)

    def test_source_term_matches_manufactured_analytics(self, operator: HelmholtzOperator) -> None:
        coords = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
        source = operator.source_term(coords)
        assert source.shape == (2,)
        u = np.sin(np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1])
        coefficient = operator.wavenumber**2 - 2 * (np.pi**2)
        np.testing.assert_allclose(source, coefficient * u, rtol=1e-5)

    def test_exact_solution_is_product_of_sines(self, operator: HelmholtzOperator) -> None:
        coords = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=np.float32)
        exact = operator.exact_solution(coords)
        assert exact is not None
        expected = np.sin(np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1])
        np.testing.assert_allclose(exact, expected, rtol=1e-5)

    def test_boundary_value_dirichlet(self, operator: HelmholtzOperator) -> None:
        coords = np.array([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)
        vals = operator.boundary_value(coords)
        assert vals.shape == (2,)
        np.testing.assert_allclose(vals, [0.0, 0.0])

    def test_residual_small_on_exact_solution(self, operator: HelmholtzOperator) -> None:
        points = operator.generate_collocation_points(64, method="random", seed=7)
        coords = _connected_coords(points)
        u = operator.exact_solution(coords)
        assert isinstance(u, torch.Tensor)
        residual = operator.residual(u, coords)
        assert isinstance(residual, PDEResidual)
        assert residual.l2_norm < 1e-3

    @pytest.mark.parametrize("u_shape", ["1d", "2d"])
    def test_residual_shape_stable_for_column_u(
        self, operator: HelmholtzOperator, u_shape: str
    ) -> None:
        # The k²u term must not broadcast against the (N,) laplacian/source when
        # u is a column vector (N, 1) — the residual stays (N,).
        points = operator.generate_collocation_points(32, method="random", seed=5)
        coords = _connected_coords(points)
        u = operator.exact_solution(coords)
        assert isinstance(u, torch.Tensor)
        if u_shape == "2d":
            u = u.unsqueeze(-1)
        residual = operator.residual(u, coords)
        assert residual.values.shape == (32,)
        assert residual.l2_norm < 1e-3

    @settings(max_examples=20, deadline=None)
    @given(k=st.floats(min_value=0.5, max_value=8.0))
    def test_residual_small_across_wavenumbers(self, k: float) -> None:
        op = HelmholtzOperator(_config(PDEType.HELMHOLTZ, "k_sweep"), wavenumber=k)
        points = op.generate_collocation_points(48, method="random", seed=3)
        coords = _connected_coords(points)
        u = op.exact_solution(coords)
        assert isinstance(u, torch.Tensor)
        residual = op.residual(u, coords)
        assert residual.l2_norm < 1e-2


# --------------------------------------------------------------------------- #
# Biharmonic                                                                   #
# --------------------------------------------------------------------------- #


class TestBiharmonicOperator:
    """Tests for the biharmonic operator ∇⁴u = f."""

    @pytest.fixture
    def operator(self) -> BiharmonicOperator:
        return BiharmonicOperator(_config(PDEType.BIHARMONIC, "test_biharmonic"))

    def test_operator_properties(self, operator: BiharmonicOperator) -> None:
        assert operator.name == "biharmonic"
        assert operator.pde_type is PDEType.BIHARMONIC
        assert operator.is_time_dependent is False
        assert operator.is_linear is True
        assert operator.order == 4

    def test_source_term_matches_manufactured_analytics(self, operator: BiharmonicOperator) -> None:
        coords = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
        source = operator.source_term(coords)
        assert source.shape == (2,)
        u = np.sin(np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1])
        coefficient = (2 * (np.pi**2)) ** 2
        np.testing.assert_allclose(source, coefficient * u, rtol=1e-5)

    def test_exact_solution_is_product_of_sines(self, operator: BiharmonicOperator) -> None:
        coords = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=np.float32)
        exact = operator.exact_solution(coords)
        assert exact is not None
        expected = np.sin(np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1])
        np.testing.assert_allclose(exact, expected, rtol=1e-5)

    def test_boundary_value_dirichlet(self, operator: BiharmonicOperator) -> None:
        coords = np.array([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)
        vals = operator.boundary_value(coords)
        np.testing.assert_allclose(vals, [0.0, 0.0])

    def test_residual_small_on_exact_solution(self, operator: BiharmonicOperator) -> None:
        points = operator.generate_collocation_points(64, method="random", seed=11)
        coords = _connected_coords(points)
        u = operator.exact_solution(coords)
        assert isinstance(u, torch.Tensor)
        residual = operator.residual(u, coords)
        assert isinstance(residual, PDEResidual)
        # Fourth-order autodiff in float32 — looser tolerance than 2nd order.
        assert residual.l2_norm < 1e-3
        assert "biharmonic" in residual.derivatives

    def test_residual_zeros_when_solution_disconnected(self, operator: BiharmonicOperator) -> None:
        # A solution disconnected from coords (no autograd graph) yields a
        # zero biharmonic term, so the residual reduces to -f.
        points = operator.generate_collocation_points(16, method="random", seed=1)
        coords = torch.tensor(points, dtype=torch.float32)
        u = torch.tensor(np.asarray(operator.exact_solution(points)), dtype=torch.float32)
        residual = operator.residual(u, coords)
        source = np.asarray(operator.source_term(points))
        np.testing.assert_allclose(residual.values.detach().numpy(), -source, rtol=1e-5, atol=1e-6)

    def test_residual_does_not_mutate_caller_coords(self, operator: BiharmonicOperator) -> None:
        # A solution connected to a parameter but NOT to coords must not flip the
        # caller's leaf tensor to requires_grad in place (regression for the
        # in-place ``requires_grad_`` side effect).
        points = operator.generate_collocation_points(12, method="random", seed=2)
        coords = torch.tensor(points, dtype=torch.float32)
        assert coords.requires_grad is False
        param = torch.nn.Parameter(torch.ones(()))
        u = torch.tensor(np.asarray(operator.exact_solution(points)), dtype=torch.float32) * param
        residual = operator.residual(u, coords)
        # Caller's coords is untouched; biharmonic w.r.t. coords is undefined here
        # (zero), so the residual reduces to -f.
        assert coords.requires_grad is False
        source = np.asarray(operator.source_term(points))
        np.testing.assert_allclose(residual.values.detach().numpy(), -source, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# Registry + game integration                                                 #
# --------------------------------------------------------------------------- #


class TestOODRegistryAndGame:
    """Registry round-trip and BasisSelectionGame construction."""

    @pytest.mark.parametrize(
        ("name", "pde_type", "cls"),
        [
            ("helmholtz", PDEType.HELMHOLTZ, HelmholtzOperator),
            ("biharmonic", PDEType.BIHARMONIC, BiharmonicOperator),
        ],
    )
    def test_registry_round_trip(self, name: str, pde_type: PDEType, cls: type) -> None:
        assert name in list_pde_operators()
        resolved = get_pde_operator(name)
        assert resolved is cls
        operator = resolved(_config(pde_type, name))
        assert operator.name == name

    @pytest.mark.parametrize("name", ["helmholtz", "biharmonic"])
    def test_basis_selection_game_constructs_with_finite_error(self, name: str) -> None:
        pde_type = PDEType.HELMHOLTZ if name == "helmholtz" else PDEType.BIHARMONIC
        operator = get_pde_operator(name)(_config(pde_type, name))
        game_config = PDEGameConfig(
            name=f"{name}_game",
            pde_config=operator.config,
            game_mode="basis_selection",
            basis_config=BasisSelectionConfig(
                name=f"{name}_basis",
                max_basis_functions=4,
                n_candidate_bases=6,
            ),
            error_tolerance=1e-2,
        )
        game = BasisSelectionGame(operator, game_config)
        adapter = PDEGameAdapter(game)
        assert game.action_space_size == 6
        assert np.isfinite(adapter.current_error)
        # Basis descriptions enumerate without error.
        descriptions = [game.action_to_string(i) for i in range(game.action_space_size)]
        assert len(descriptions) == 6
