"""Tests for physics-informed loss components (src.training.losses.physics)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from src.pde.operators import PDEResidual
from src.training.losses.physics import (
    BoundaryLoss,
    CombinedAlphaGalerkinPhysicsLoss,
    ConservationLoss,
    InitialConditionLoss,
    PhysicsInformedLoss,
    PhysicsLossConfig,
    PhysicsLossOutput,
    ResidualLoss,
    _get_device_from_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
N_POINTS = 10
DIM = 2


def _make_mock_operator(*, is_time_dependent: bool = False) -> MagicMock:
    """Create a mock PDE operator."""
    op = MagicMock()

    # Make is_time_dependent a plain attribute, not a Mock
    op.is_time_dependent = is_time_dependent

    # residual returns a PDEResidual
    def _residual(u, coords, compute_derivatives=True):
        n = u.shape[0] if isinstance(u, Tensor) else len(u)
        vals = torch.zeros(n)
        return PDEResidual(
            values=vals,
            l2_norm=0.0,
            max_norm=0.0,
            derivatives={},
        )

    op.residual.side_effect = _residual

    # boundary_value returns zeros matching input shape
    def _boundary_value(coords, time=None):
        n = coords.shape[0] if isinstance(coords, Tensor) else len(coords)
        return torch.zeros(n)

    op.boundary_value.side_effect = _boundary_value

    # initial_condition returns zeros
    def _initial_condition(coords):
        n = coords.shape[0] if isinstance(coords, Tensor) else len(coords)
        return torch.zeros(n)

    op.initial_condition.side_effect = _initial_condition

    # source_term
    def _source_term(coords, time=None):
        n = coords.shape[0] if isinstance(coords, Tensor) else len(coords)
        return torch.zeros(n)

    op.source_term.side_effect = _source_term

    # collocation/boundary point generation
    op.generate_collocation_points.return_value = np.random.rand(N_POINTS, DIM).astype(
        np.float32
    )
    op.generate_boundary_points.return_value = np.random.rand(N_POINTS, DIM).astype(
        np.float32
    )

    return op


# ---------------------------------------------------------------------------
# PhysicsLossConfig tests
# ---------------------------------------------------------------------------


class TestPhysicsLossConfig:
    """Test PhysicsLossConfig validation."""

    def test_default_values(self) -> None:
        """Default config has expected values."""
        cfg = PhysicsLossConfig(name="test")
        assert cfg.residual_weight == 1.0
        assert cfg.boundary_weight == 10.0
        assert cfg.initial_weight == 10.0
        assert cfg.conservation_weight == 1.0
        assert cfg.n_collocation_points == 1000
        assert cfg.n_boundary_points == 200
        assert cfg.use_adaptive_weights is True

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        cfg = PhysicsLossConfig(
            name="custom",
            residual_weight=2.0,
            boundary_weight=5.0,
            n_collocation_points=500,
        )
        assert cfg.residual_weight == 2.0
        assert cfg.boundary_weight == 5.0
        assert cfg.n_collocation_points == 500

    def test_negative_weight_rejected(self) -> None:
        """Negative weights should be rejected."""
        with pytest.raises(Exception):
            PhysicsLossConfig(name="bad", residual_weight=-1.0)

    def test_collocation_points_bounds(self) -> None:
        """Collocation points must be within bounds."""
        with pytest.raises(Exception):
            PhysicsLossConfig(name="bad", n_collocation_points=5)  # < 10

        with pytest.raises(Exception):
            PhysicsLossConfig(name="bad", n_collocation_points=200000)  # > 100000

    def test_boundary_points_bounds(self) -> None:
        """Boundary points must be within bounds."""
        with pytest.raises(Exception):
            PhysicsLossConfig(name="bad", n_boundary_points=5)  # < 10


# ---------------------------------------------------------------------------
# ResidualLoss tests
# ---------------------------------------------------------------------------


class TestResidualLoss:
    """Test ResidualLoss computation."""

    def test_zero_residual(self) -> None:
        """Zero residual produces zero loss."""
        op = _make_mock_operator()
        loss_fn = ResidualLoss(op, reduction="mean")

        u = torch.randn(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u, coords)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_nonzero_residual(self) -> None:
        """Non-zero residual produces positive loss."""
        op = _make_mock_operator()

        def _residual(u, coords, compute_derivatives=True):
            n = u.shape[0] if isinstance(u, Tensor) else len(u)
            vals = torch.ones(n)
            return PDEResidual(values=vals, l2_norm=1.0, max_norm=1.0, derivatives={})

        op.residual.side_effect = _residual

        loss_fn = ResidualLoss(op, reduction="mean")
        u = torch.randn(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u, coords)
        assert loss > 0

    def test_invalid_reduction_raises(self) -> None:
        """Invalid reduction string raises ValueError."""
        op = _make_mock_operator()
        with pytest.raises(ValueError, match="Invalid reduction"):
            ResidualLoss(op, reduction="invalid")

    def test_sum_reduction(self) -> None:
        """Sum reduction produces larger values than mean."""
        op = _make_mock_operator()

        def _residual(u, coords, compute_derivatives=True):
            n = u.shape[0] if isinstance(u, Tensor) else len(u)
            vals = torch.ones(n)
            return PDEResidual(values=vals, l2_norm=1.0, max_norm=1.0, derivatives={})

        op.residual.side_effect = _residual

        loss_mean = ResidualLoss(op, reduction="mean")
        loss_sum = ResidualLoss(op, reduction="sum")

        u = torch.randn(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        l_mean = loss_mean(u, coords)
        l_sum = loss_sum(u, coords)

        assert l_sum >= l_mean

    def test_numpy_residual_values(self) -> None:
        """Residual with numpy values works correctly."""
        op = _make_mock_operator()

        def _residual(u, coords, compute_derivatives=True):
            n = u.shape[0] if isinstance(u, Tensor) else len(u)
            vals = np.ones(n, dtype=np.float32)
            return PDEResidual(values=vals, l2_norm=1.0, max_norm=1.0, derivatives={})

        op.residual.side_effect = _residual

        loss_fn = ResidualLoss(op, reduction="mean")
        u = torch.randn(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u, coords)
        assert loss.isfinite()
        assert loss > 0


# ---------------------------------------------------------------------------
# BoundaryLoss tests
# ---------------------------------------------------------------------------


class TestBoundaryLoss:
    """Test BoundaryLoss computation."""

    def test_zero_error(self) -> None:
        """Boundary loss is zero when prediction matches target."""
        op = _make_mock_operator()
        loss_fn = BoundaryLoss(op, bc_type="dirichlet")

        # u_boundary = 0 and boundary_value returns 0
        u_boundary = torch.zeros(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u_boundary, coords)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_nonzero_error(self) -> None:
        """Boundary loss is positive when prediction differs from target."""
        op = _make_mock_operator()
        loss_fn = BoundaryLoss(op, bc_type="dirichlet")

        u_boundary = torch.ones(BATCH_SIZE, N_POINTS)  # pred = 1, target = 0
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u_boundary, coords)
        assert loss > 0

    def test_invalid_bc_type_raises(self) -> None:
        """Invalid BC type raises ValueError."""
        op = _make_mock_operator()
        with pytest.raises(ValueError, match="Invalid bc_type"):
            BoundaryLoss(op, bc_type="invalid")

    def test_invalid_reduction_raises(self) -> None:
        """Invalid reduction raises ValueError."""
        op = _make_mock_operator()
        with pytest.raises(ValueError, match="Invalid reduction"):
            BoundaryLoss(op, bc_type="dirichlet", reduction="invalid")

    def test_numpy_boundary_values(self) -> None:
        """Boundary values returned as numpy arrays work correctly."""
        op = _make_mock_operator()

        def _boundary_value(coords, time=None):
            n = coords.shape[0] if isinstance(coords, Tensor) else len(coords)
            return np.zeros(n, dtype=np.float32)

        op.boundary_value.side_effect = _boundary_value

        loss_fn = BoundaryLoss(op, bc_type="dirichlet")
        u_boundary = torch.ones(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u_boundary, coords)
        assert loss.isfinite()
        assert loss > 0


# ---------------------------------------------------------------------------
# InitialConditionLoss tests
# ---------------------------------------------------------------------------


class TestInitialConditionLoss:
    """Test InitialConditionLoss computation."""

    def test_zero_error(self) -> None:
        """IC loss is zero when prediction matches initial condition."""
        op = _make_mock_operator(is_time_dependent=True)
        loss_fn = InitialConditionLoss(op)

        u_initial = torch.zeros(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u_initial, coords)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_nonzero_error(self) -> None:
        """IC loss is positive when prediction differs from initial condition."""
        op = _make_mock_operator(is_time_dependent=True)
        loss_fn = InitialConditionLoss(op)

        u_initial = torch.ones(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u_initial, coords)
        assert loss > 0

    def test_numpy_initial_condition(self) -> None:
        """IC with numpy return values works."""
        op = _make_mock_operator(is_time_dependent=True)

        def _ic(coords):
            n = coords.shape[0] if isinstance(coords, Tensor) else len(coords)
            return np.zeros(n, dtype=np.float32)

        op.initial_condition.side_effect = _ic

        loss_fn = InitialConditionLoss(op)
        u_initial = torch.ones(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u_initial, coords)
        assert loss.isfinite()


# ---------------------------------------------------------------------------
# ConservationLoss tests
# ---------------------------------------------------------------------------


class TestConservationLoss:
    """Test ConservationLoss computation."""

    def test_zero_loss_with_initial(self) -> None:
        """Conservation loss is zero when u equals u_initial."""
        loss_fn = ConservationLoss()

        u = torch.ones(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)
        u_initial = torch.ones(BATCH_SIZE, N_POINTS)

        loss = loss_fn(u, coords, u_initial)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_nonzero_loss(self) -> None:
        """Conservation loss is positive when integral changes."""
        loss_fn = ConservationLoss()

        u = torch.ones(BATCH_SIZE, N_POINTS) * 2
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)
        u_initial = torch.ones(BATCH_SIZE, N_POINTS)

        loss = loss_fn(u, coords, u_initial)
        assert loss > 0

    def test_with_fixed_initial_integral(self) -> None:
        """Conservation loss with explicit initial integral."""
        loss_fn = ConservationLoss(initial_integral=1.0)

        u = torch.ones(BATCH_SIZE, N_POINTS) * 2  # integral = 2.0
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u, coords)
        assert loss > 0

    def test_custom_conserved_quantity(self) -> None:
        """Custom conserved quantity function is used."""

        def energy(u: Tensor, coords: Tensor) -> Tensor:
            return (u**2).sum(dim=-1) / u.shape[-1]

        loss_fn = ConservationLoss(conserved_quantity=energy, initial_integral=1.0)

        u = torch.ones(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u, coords)
        # energy = 1.0, initial = 1.0 -> loss = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_first_call_stores_initial(self) -> None:
        """First call with no initial stores current as baseline and returns 0."""
        loss_fn = ConservationLoss()

        u = torch.ones(BATCH_SIZE, N_POINTS)
        coords = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        loss = loss_fn(u, coords)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)
        assert loss_fn._stored_initial is not None


# ---------------------------------------------------------------------------
# PhysicsLossOutput tests
# ---------------------------------------------------------------------------


class TestPhysicsLossOutput:
    """Test PhysicsLossOutput dataclass."""

    def test_to_dict_full(self) -> None:
        """to_dict with all components."""
        output = PhysicsLossOutput(
            total=torch.tensor(1.0),
            residual=torch.tensor(0.5),
            boundary=torch.tensor(0.3),
            initial=torch.tensor(0.1),
            conservation=torch.tensor(0.1),
            weights={"residual": 1.0, "boundary": 10.0},
        )
        d = output.to_dict()
        assert d["total"] == pytest.approx(1.0)
        assert d["residual"] == pytest.approx(0.5)
        assert d["boundary"] == pytest.approx(0.3)
        assert d["initial"] == pytest.approx(0.1)
        assert d["conservation"] == pytest.approx(0.1)
        assert d["weight_residual"] == pytest.approx(1.0)
        assert d["weight_boundary"] == pytest.approx(10.0)

    def test_to_dict_optional_none(self) -> None:
        """to_dict excludes None optional components."""
        output = PhysicsLossOutput(
            total=torch.tensor(0.8),
            residual=torch.tensor(0.5),
            boundary=torch.tensor(0.3),
            initial=None,
            conservation=None,
            weights={"residual": 1.0},
        )
        d = output.to_dict()
        assert "initial" not in d
        assert "conservation" not in d


# ---------------------------------------------------------------------------
# _get_device_from_model tests
# ---------------------------------------------------------------------------


class TestGetDeviceFromModel:
    """Test _get_device_from_model helper."""

    def test_model_with_parameters(self) -> None:
        """Returns device of model's first parameter."""
        model = nn.Linear(10, 5)
        device = _get_device_from_model(model)
        assert device == torch.device("cpu")

    def test_model_without_parameters(self) -> None:
        """Returns CPU for model with no parameters."""
        model = nn.Module()  # No parameters
        device = _get_device_from_model(model)
        assert device == torch.device("cpu")


# ---------------------------------------------------------------------------
# PhysicsInformedLoss tests
# ---------------------------------------------------------------------------


class TestPhysicsInformedLoss:
    """Test PhysicsInformedLoss combining components."""

    def test_basic_forward(self) -> None:
        """Basic forward pass produces PhysicsLossOutput."""
        op = _make_mock_operator()
        cfg = PhysicsLossConfig(
            name="test",
            use_adaptive_weights=False,
            n_collocation_points=10,
            n_boundary_points=10,
        )
        loss_fn = PhysicsInformedLoss(op, cfg)

        model = nn.Linear(DIM, N_POINTS)
        coords_interior = torch.randn(BATCH_SIZE, N_POINTS, DIM)
        coords_boundary = torch.randn(BATCH_SIZE, N_POINTS, DIM)

        result = loss_fn(model, coords_interior=coords_interior, coords_boundary=coords_boundary)

        assert isinstance(result, PhysicsLossOutput)
        assert result.total.isfinite() if isinstance(result.total, Tensor) else True
        assert result.residual.isfinite()
        assert result.boundary.isfinite()

    def test_auto_generated_points(self) -> None:
        """Collocation points are auto-generated when not provided."""
        op = _make_mock_operator()
        cfg = PhysicsLossConfig(
            name="test",
            use_adaptive_weights=False,
            n_collocation_points=10,
            n_boundary_points=10,
        )
        loss_fn = PhysicsInformedLoss(op, cfg)

        model = nn.Linear(DIM, N_POINTS)
        result = loss_fn(model)

        assert isinstance(result, PhysicsLossOutput)
        op.generate_collocation_points.assert_called_once()
        op.generate_boundary_points.assert_called_once()

    def test_static_weights(self) -> None:
        """Static weights are used when adaptive is disabled."""
        op = _make_mock_operator()
        cfg = PhysicsLossConfig(
            name="test",
            use_adaptive_weights=False,
            residual_weight=2.0,
            boundary_weight=5.0,
        )
        loss_fn = PhysicsInformedLoss(op, cfg)
        assert loss_fn.balancer is None
        assert loss_fn._static_weights["residual"] == 2.0
        assert loss_fn._static_weights["boundary"] == 5.0

    def test_time_dependent_includes_initial_loss(self) -> None:
        """Time-dependent PDE creates initial condition loss."""
        op = _make_mock_operator(is_time_dependent=True)
        cfg = PhysicsLossConfig(
            name="test",
            use_adaptive_weights=False,
            n_collocation_points=10,
            n_boundary_points=10,
        )
        loss_fn = PhysicsInformedLoss(op, cfg)
        assert loss_fn.initial_loss is not None

    def test_steady_state_no_initial_loss(self) -> None:
        """Steady-state PDE does not create initial condition loss."""
        op = _make_mock_operator(is_time_dependent=False)
        cfg = PhysicsLossConfig(
            name="test",
            use_adaptive_weights=False,
        )
        loss_fn = PhysicsInformedLoss(op, cfg)
        assert loss_fn.initial_loss is None


# ---------------------------------------------------------------------------
# CombinedAlphaGalerkinPhysicsLoss tests
# ---------------------------------------------------------------------------


class TestCombinedAlphaGalerkinPhysicsLoss:
    """Test CombinedAlphaGalerkinPhysicsLoss."""

    def test_without_physics(self) -> None:
        """Works without a PDE operator (physics disabled)."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(pde_operator=None)

        policy_logits = torch.randn(BATCH_SIZE, 25)
        value = torch.randn(BATCH_SIZE, 1)
        target_policy = torch.softmax(torch.randn(BATCH_SIZE, 25), dim=-1)
        target_value = torch.randn(BATCH_SIZE, 1)

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert "total" in result
        assert "policy" in result
        assert "value" in result
        assert "lbb" in result
        assert isinstance(result["total"], Tensor)
        assert result["total"].isfinite()

    def test_custom_weights(self) -> None:
        """Custom weights are stored correctly."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(
            policy_weight=2.0,
            value_weight=0.5,
            lbb_weight=0.1,
            physics_weight=0.2,
        )
        assert loss_fn.policy_weight == 2.0
        assert loss_fn.value_weight == 0.5
        assert loss_fn.lbb_weight == 0.1
        assert loss_fn.physics_weight == 0.2

    def test_output_has_physics_zero_without_operator(self) -> None:
        """Physics loss is zero tensor when no operator provided."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(pde_operator=None)

        policy_logits = torch.randn(BATCH_SIZE, 25)
        value = torch.randn(BATCH_SIZE, 1)
        target_policy = torch.softmax(torch.randn(BATCH_SIZE, 25), dim=-1)
        target_value = torch.randn(BATCH_SIZE, 1)

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert result["physics"].item() == pytest.approx(0.0)

    def test_output_has_weights(self) -> None:
        """Output includes balancing weights."""
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(pde_operator=None)

        policy_logits = torch.randn(BATCH_SIZE, 25)
        value = torch.randn(BATCH_SIZE, 1)
        target_policy = torch.softmax(torch.randn(BATCH_SIZE, 25), dim=-1)
        target_value = torch.randn(BATCH_SIZE, 1)

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert "weights" in result
