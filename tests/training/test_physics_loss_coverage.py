"""Coverage tests for physics-informed loss components.

Tests cover:
- ResidualLoss: PDE residual computation with various reductions
- BoundaryLoss: Boundary condition enforcement (Dirichlet, Neumann, Robin)
- InitialConditionLoss: Initial condition enforcement
- ConservationLoss: Conservation law satisfaction
- PhysicsInformedLoss: Combined physics loss with adaptive balancing
- PhysicsLossOutput: Output dataclass serialization
- PhysicsLossConfig: Configuration validation
- CombinedAlphaGalerkinPhysicsLoss: Integration with policy/value losses
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import PoissonOperator
from src.training.physics_loss import (
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

SEED = 42
BATCH_SIZE = 2
N_POINTS = 16
DIM = 2


@pytest.fixture
def pde_config() -> PDEConfig:
    """Create a minimal PDE config for testing."""
    return PDEConfig(
        name="test_poisson",
        pde_type=PDEType.POISSON,
        domain_dim=DIM,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
    )


@pytest.fixture
def poisson_operator(pde_config: PDEConfig) -> PoissonOperator:
    """Create a Poisson operator for testing."""
    return PoissonOperator(pde_config)


@pytest.fixture
def sample_coords() -> Tensor:
    """Create sample collocation point coordinates."""
    torch.manual_seed(SEED)
    return torch.rand(BATCH_SIZE, N_POINTS, DIM, requires_grad=True)


@pytest.fixture
def sample_u() -> Tensor:
    """Create sample solution values."""
    torch.manual_seed(SEED)
    return torch.rand(BATCH_SIZE, N_POINTS)


class TestGetDeviceFromModel:
    """Tests for _get_device_from_model helper."""

    def test_model_with_parameters(self) -> None:
        model = nn.Linear(4, 4)
        device = _get_device_from_model(model)
        assert device == torch.device("cpu")

    def test_model_without_parameters(self) -> None:
        model = nn.Module()  # No parameters
        device = _get_device_from_model(model)
        assert device == torch.device("cpu")


class TestPhysicsLossConfig:
    """Tests for PhysicsLossConfig validation."""

    def test_default_config(self) -> None:
        config = PhysicsLossConfig(name="test")
        assert config.residual_weight == 1.0
        assert config.boundary_weight == 10.0
        assert config.initial_weight == 10.0
        assert config.conservation_weight == 1.0
        assert config.n_collocation_points == 1000
        assert config.n_boundary_points == 200
        assert config.use_adaptive_weights is True

    def test_custom_config(self) -> None:
        config = PhysicsLossConfig(
            name="custom",
            residual_weight=2.0,
            boundary_weight=5.0,
            n_collocation_points=500,
            sampling_method="random",
        )
        assert config.residual_weight == 2.0
        assert config.boundary_weight == 5.0
        assert config.n_collocation_points == 500
        assert config.sampling_method == "random"


class TestPhysicsLossOutput:
    """Tests for PhysicsLossOutput dataclass."""

    def test_to_dict_all_fields(self) -> None:
        output = PhysicsLossOutput(
            total=torch.tensor(1.5),
            residual=torch.tensor(0.5),
            boundary=torch.tensor(0.3),
            initial=torch.tensor(0.2),
            conservation=torch.tensor(0.1),
            weights={"residual": 1.0, "boundary": 10.0},
        )
        d = output.to_dict()
        assert d["total"] == pytest.approx(1.5)
        assert d["residual"] == pytest.approx(0.5)
        assert d["boundary"] == pytest.approx(0.3)
        assert d["initial"] == pytest.approx(0.2)
        assert d["conservation"] == pytest.approx(0.1)
        assert "weight_residual" in d
        assert "weight_boundary" in d

    def test_to_dict_none_fields(self) -> None:
        output = PhysicsLossOutput(
            total=torch.tensor(1.0),
            residual=torch.tensor(0.5),
            boundary=torch.tensor(0.5),
            initial=None,
            conservation=None,
            weights={"residual": 1.0},
        )
        d = output.to_dict()
        assert "initial" not in d
        assert "conservation" not in d


class TestResidualLoss:
    """Tests for ResidualLoss module."""

    def test_initialization(self, poisson_operator: PoissonOperator) -> None:
        loss = ResidualLoss(poisson_operator, reduction="mean")
        assert loss.reduction == "mean"

    def test_invalid_reduction(self, poisson_operator: PoissonOperator) -> None:
        with pytest.raises(ValueError, match="Invalid reduction"):
            ResidualLoss(poisson_operator, reduction="invalid")

    def test_forward_mean(self, poisson_operator: PoissonOperator) -> None:
        torch.manual_seed(SEED)
        loss_fn = ResidualLoss(poisson_operator, reduction="mean")
        coords = torch.rand(1, N_POINTS, DIM, requires_grad=True)
        u = torch.sin(coords[..., 0]) * torch.sin(coords[..., 1])
        result = loss_fn(u, coords)
        assert result.shape == ()
        assert result.item() >= 0.0

    def test_forward_sum(self, poisson_operator: PoissonOperator) -> None:
        torch.manual_seed(SEED)
        loss_fn = ResidualLoss(poisson_operator, reduction="sum")
        coords = torch.rand(1, N_POINTS, DIM, requires_grad=True)
        u = torch.sin(coords[..., 0]) * torch.sin(coords[..., 1])
        result = loss_fn(u, coords)
        assert result.item() >= 0.0

    def test_forward_none(self, poisson_operator: PoissonOperator) -> None:
        torch.manual_seed(SEED)
        loss_fn = ResidualLoss(poisson_operator, reduction="none")
        coords = torch.rand(1, N_POINTS, DIM, requires_grad=True)
        u = torch.sin(coords[..., 0]) * torch.sin(coords[..., 1])
        result = loss_fn(u, coords)
        # With "none" reduction, result should not be a scalar
        assert result.numel() >= 1

    def test_forward_batched(self, poisson_operator: PoissonOperator) -> None:
        torch.manual_seed(SEED)
        loss_fn = ResidualLoss(poisson_operator, reduction="mean")
        coords = torch.rand(BATCH_SIZE, N_POINTS, DIM, requires_grad=True)
        u = torch.sin(coords[..., 0]) * torch.sin(coords[..., 1])
        result = loss_fn(u, coords)
        assert result.shape == ()


class TestBoundaryLoss:
    """Tests for BoundaryLoss module."""

    def test_initialization_dirichlet(self, poisson_operator: PoissonOperator) -> None:
        loss = BoundaryLoss(poisson_operator, bc_type="dirichlet")
        assert loss.bc_type == "dirichlet"

    def test_invalid_bc_type(self, poisson_operator: PoissonOperator) -> None:
        with pytest.raises(ValueError, match="Invalid bc_type"):
            BoundaryLoss(poisson_operator, bc_type="periodic")

    def test_invalid_reduction(self, poisson_operator: PoissonOperator) -> None:
        with pytest.raises(ValueError, match="Invalid reduction"):
            BoundaryLoss(poisson_operator, reduction="invalid")

    def test_forward_dirichlet(self, poisson_operator: PoissonOperator) -> None:
        torch.manual_seed(SEED)
        loss_fn = BoundaryLoss(poisson_operator, bc_type="dirichlet", reduction="mean")
        n_boundary = 10
        u_boundary = torch.rand(1, n_boundary)
        coords_boundary = torch.rand(1, n_boundary, DIM)
        result = loss_fn(u_boundary, coords_boundary)
        assert result.shape == ()
        assert result.item() >= 0.0

    def test_forward_sum_reduction(self, poisson_operator: PoissonOperator) -> None:
        torch.manual_seed(SEED)
        loss_fn = BoundaryLoss(poisson_operator, bc_type="dirichlet", reduction="sum")
        u_boundary = torch.rand(1, 10)
        coords_boundary = torch.rand(1, 10, DIM)
        result = loss_fn(u_boundary, coords_boundary)
        assert result.item() >= 0.0

    def test_forward_none_reduction(self, poisson_operator: PoissonOperator) -> None:
        torch.manual_seed(SEED)
        loss_fn = BoundaryLoss(poisson_operator, bc_type="dirichlet", reduction="none")
        u_boundary = torch.rand(1, 10)
        coords_boundary = torch.rand(1, 10, DIM)
        result = loss_fn(u_boundary, coords_boundary)
        assert result.numel() >= 1

    def test_forward_batched(self, poisson_operator: PoissonOperator) -> None:
        torch.manual_seed(SEED)
        loss_fn = BoundaryLoss(poisson_operator, bc_type="dirichlet", reduction="mean")
        u_boundary = torch.rand(BATCH_SIZE, 10)
        coords_boundary = torch.rand(BATCH_SIZE, 10, DIM)
        result = loss_fn(u_boundary, coords_boundary)
        assert result.shape == ()


class TestInitialConditionLoss:
    """Tests for InitialConditionLoss module."""

    def test_initialization(self, poisson_operator: PoissonOperator) -> None:
        loss = InitialConditionLoss(poisson_operator)
        assert loss.reduction == "mean"

    def test_forward_mean(self, poisson_operator: PoissonOperator) -> None:
        torch.manual_seed(SEED)
        loss_fn = InitialConditionLoss(poisson_operator, reduction="mean")
        u_initial = torch.rand(1, N_POINTS)
        coords_initial = torch.rand(1, N_POINTS, DIM)
        result = loss_fn(u_initial, coords_initial)
        assert result.shape == ()
        assert result.item() >= 0.0

    def test_forward_sum(self, poisson_operator: PoissonOperator) -> None:
        torch.manual_seed(SEED)
        loss_fn = InitialConditionLoss(poisson_operator, reduction="sum")
        u_initial = torch.rand(1, N_POINTS)
        coords_initial = torch.rand(1, N_POINTS, DIM)
        result = loss_fn(u_initial, coords_initial)
        assert result.item() >= 0.0

    def test_forward_batched(self, poisson_operator: PoissonOperator) -> None:
        torch.manual_seed(SEED)
        loss_fn = InitialConditionLoss(poisson_operator, reduction="mean")
        u_initial = torch.rand(BATCH_SIZE, N_POINTS)
        coords_initial = torch.rand(BATCH_SIZE, N_POINTS, DIM)
        result = loss_fn(u_initial, coords_initial)
        assert result.shape == ()


class TestConservationLoss:
    """Tests for ConservationLoss module."""

    def test_default_mass_conservation(self) -> None:
        torch.manual_seed(SEED)
        loss_fn = ConservationLoss()
        u = torch.rand(BATCH_SIZE, N_POINTS)
        coords = torch.rand(BATCH_SIZE, N_POINTS, DIM)
        # First call stores initial integral
        result1 = loss_fn(u, coords)
        assert result1.item() == 0.0  # First call returns 0

    def test_conservation_with_initial_integral(self) -> None:
        torch.manual_seed(SEED)
        loss_fn = ConservationLoss(initial_integral=0.5)
        u = torch.rand(BATCH_SIZE, N_POINTS)
        coords = torch.rand(BATCH_SIZE, N_POINTS, DIM)
        result = loss_fn(u, coords)
        assert result.shape == ()
        assert result.item() >= 0.0

    def test_conservation_with_u_initial(self) -> None:
        torch.manual_seed(SEED)
        loss_fn = ConservationLoss()
        u = torch.rand(BATCH_SIZE, N_POINTS)
        u_initial = torch.rand(BATCH_SIZE, N_POINTS)
        coords = torch.rand(BATCH_SIZE, N_POINTS, DIM)
        result = loss_fn(u, coords, u_initial=u_initial)
        assert result.shape == ()

    def test_conservation_with_custom_quantity(self) -> None:
        def custom_quantity(u: Tensor, coords: Tensor) -> Tensor:
            return (u**2).sum(dim=-1) / u.shape[-1]

        torch.manual_seed(SEED)
        loss_fn = ConservationLoss(conserved_quantity=custom_quantity, initial_integral=1.0)
        u = torch.ones(BATCH_SIZE, N_POINTS)
        coords = torch.rand(BATCH_SIZE, N_POINTS, DIM)
        result = loss_fn(u, coords)
        assert result.shape == ()

    def test_conservation_stored_initial(self) -> None:
        torch.manual_seed(SEED)
        loss_fn = ConservationLoss()
        u = torch.ones(BATCH_SIZE, N_POINTS)
        coords = torch.rand(BATCH_SIZE, N_POINTS, DIM)
        # First call stores initial
        loss_fn(u, coords)
        assert loss_fn._stored_initial is not None
        # Second call uses stored initial
        result = loss_fn(u * 2, coords)
        assert result.item() > 0.0


class TestPhysicsInformedLoss:
    """Tests for combined PhysicsInformedLoss."""

    @pytest.fixture
    def physics_loss(self, poisson_operator: PoissonOperator) -> PhysicsInformedLoss:
        config = PhysicsLossConfig(
            name="test",
            n_collocation_points=20,
            n_boundary_points=10,
            use_adaptive_weights=False,
        )
        return PhysicsInformedLoss(poisson_operator, config)

    @pytest.fixture
    def physics_loss_adaptive(self, poisson_operator: PoissonOperator) -> PhysicsInformedLoss:
        config = PhysicsLossConfig(
            name="test_adaptive",
            n_collocation_points=20,
            n_boundary_points=10,
            use_adaptive_weights=True,
        )
        return PhysicsInformedLoss(poisson_operator, config)

    def test_initialization_static(self, physics_loss: PhysicsInformedLoss) -> None:
        assert physics_loss.balancer is None
        assert physics_loss.residual_loss is not None
        assert physics_loss.boundary_loss is not None
        assert physics_loss.initial_loss is None  # Poisson is not time-dependent

    def test_initialization_adaptive(self, physics_loss_adaptive: PhysicsInformedLoss) -> None:
        assert physics_loss_adaptive.balancer is not None

    def test_forward_with_model(self, physics_loss: PhysicsInformedLoss) -> None:
        torch.manual_seed(SEED)
        model = nn.Linear(DIM, 1)

        # Create coords
        coords_interior = torch.rand(1, 20, DIM)
        coords_boundary = torch.rand(1, 10, DIM)

        result = physics_loss(
            model=model,
            coords_interior=coords_interior,
            coords_boundary=coords_boundary,
        )
        assert isinstance(result, PhysicsLossOutput)
        assert result.total.shape == ()
        assert result.residual.shape == ()
        assert result.boundary.shape == ()

    def test_forward_auto_generate_points(self, physics_loss: PhysicsInformedLoss) -> None:
        torch.manual_seed(SEED)
        model = nn.Linear(DIM, 1)
        # Auto-generate collocation points
        result = physics_loss(model=model)
        assert isinstance(result, PhysicsLossOutput)

    def test_forward_partial_coords(self, physics_loss: PhysicsInformedLoss) -> None:
        torch.manual_seed(SEED)
        model = nn.Linear(DIM, 1)
        coords_interior = torch.rand(1, 20, DIM)
        # Only interior coords provided, boundary auto-generated
        result = physics_loss(model=model, coords_interior=coords_interior)
        assert isinstance(result, PhysicsLossOutput)

    def test_forward_adaptive_balancer(self, physics_loss_adaptive: PhysicsInformedLoss) -> None:
        torch.manual_seed(SEED)
        model = nn.Linear(DIM, 1)
        result = physics_loss_adaptive(model=model)
        assert isinstance(result, PhysicsLossOutput)
        assert len(result.weights) > 0


class TestCombinedAlphaGalerkinPhysicsLoss:
    """Tests for CombinedAlphaGalerkinPhysicsLoss."""

    def test_initialization_without_physics(self) -> None:
        loss = CombinedAlphaGalerkinPhysicsLoss(pde_operator=None)
        assert loss.physics_loss is None
        assert loss.policy_weight == 1.0
        assert loss.value_weight == 1.0

    def test_initialization_with_physics(self, poisson_operator: PoissonOperator) -> None:
        loss = CombinedAlphaGalerkinPhysicsLoss(
            pde_operator=poisson_operator,
            physics_weight=0.5,
        )
        assert loss.physics_loss is not None
        assert loss.physics_weight == 0.5

    def test_forward_without_physics(self) -> None:
        torch.manual_seed(SEED)
        loss_fn = CombinedAlphaGalerkinPhysicsLoss(pde_operator=None)

        batch = 2
        n_actions = 10
        policy_logits = torch.randn(batch, n_actions)
        value = torch.randn(batch, 1)
        target_policy = torch.softmax(torch.randn(batch, n_actions), dim=-1)
        target_value = torch.randn(batch, 1)
        lbb_constant = torch.tensor(0.1)

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            lbb_constant=lbb_constant,
        )

        assert "total" in result
        assert "policy" in result
        assert "value" in result
        assert "lbb" in result
        assert "physics" in result
        assert result["physics"].item() == 0.0
