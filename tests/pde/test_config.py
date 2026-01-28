"""Tests for PDE configuration classes."""

import pytest
from pydantic import ValidationError

from src.pde.config import (
    ActionSpace,
    BasisSelectionConfig,
    BoundaryCondition,
    MeshRefinementConfig,
    PDEConfig,
    PDEGameConfig,
    PDEType,
    RefinementStrategy,
)


class TestPDEConfig:
    """Tests for PDEConfig."""

    def test_create_default_poisson_config(self) -> None:
        """Test creating a default Poisson config."""
        config = PDEConfig(
            name="test_poisson",
            pde_type=PDEType.POISSON,
        )
        assert config.pde_type == PDEType.POISSON
        assert config.domain_dim == 2
        assert config.diffusion_coeff == 1.0
        assert config.boundary_condition == BoundaryCondition.DIRICHLET

    def test_create_burgers_config(self) -> None:
        """Test creating Burgers equation config."""
        config = PDEConfig(
            name="test_burgers",
            pde_type=PDEType.BURGERS,
            diffusion_coeff=0.01,
            is_time_dependent=True,
        )
        assert config.pde_type == PDEType.BURGERS
        assert config.diffusion_coeff == 0.01
        assert config.is_time_dependent is True

    def test_domain_validation(self) -> None:
        """Test domain dimension validation."""
        # Correct dimensions
        config = PDEConfig(
            name="test",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            advection_coeff=[0.0, 0.0],
        )
        assert len(config.domain_min) == 2
        assert len(config.domain_max) == 2

    def test_invalid_domain_raises(self) -> None:
        """Test that invalid domain raises error."""
        with pytest.raises(ValidationError):
            PDEConfig(
                name="test",
                pde_type=PDEType.POISSON,
                domain_dim=2,
                domain_min=[0.0],  # Wrong length
                domain_max=[1.0, 1.0],
            )

    def test_domain_min_max_validation(self) -> None:
        """Test that min < max is enforced."""
        with pytest.raises(ValidationError):
            PDEConfig(
                name="test",
                pde_type=PDEType.POISSON,
                domain_min=[1.0, 0.0],  # min > max for x
                domain_max=[0.0, 1.0],
            )

    def test_time_validation(self) -> None:
        """Test time range validation."""
        with pytest.raises(ValidationError):
            PDEConfig(
                name="test",
                pde_type=PDEType.HEAT,
                is_time_dependent=True,
                time_start=1.0,
                time_end=0.5,  # start > end
            )

    def test_config_hash_deterministic(self) -> None:
        """Test that config hash is deterministic."""
        config1 = PDEConfig(name="test", pde_type=PDEType.POISSON)
        config2 = PDEConfig(name="test", pde_type=PDEType.POISSON)
        assert config1.compute_hash() == config2.compute_hash()

    def test_config_hash_changes_with_params(self) -> None:
        """Test that config hash changes with parameters."""
        config1 = PDEConfig(name="test", pde_type=PDEType.POISSON)
        config2 = PDEConfig(name="test", pde_type=PDEType.BURGERS)
        assert config1.compute_hash() != config2.compute_hash()


class TestBasisSelectionConfig:
    """Tests for BasisSelectionConfig."""

    def test_create_default_config(self) -> None:
        """Test creating default basis selection config."""
        config = BasisSelectionConfig(name="test_basis")
        assert config.basis_type == "fourier"
        assert config.max_basis_functions == 100
        assert config.n_candidate_bases == 32

    def test_fourier_config(self) -> None:
        """Test Fourier-specific configuration."""
        config = BasisSelectionConfig(
            name="test_fourier",
            basis_type="fourier",
            max_frequency=100,
            include_dc_component=True,
        )
        assert config.max_frequency == 100
        assert config.include_dc_component is True

    def test_rbf_config(self) -> None:
        """Test RBF-specific configuration."""
        config = BasisSelectionConfig(
            name="test_rbf",
            basis_type="rbf",
            rbf_kernel="gaussian",
        )
        assert config.rbf_kernel == "gaussian"

    def test_initial_vs_max_validation(self) -> None:
        """Test that initial <= max basis functions."""
        with pytest.raises(ValidationError):
            BasisSelectionConfig(
                name="test",
                initial_basis_count=200,
                max_basis_functions=100,
            )

    def test_scale_range_validation(self) -> None:
        """Test that scale range is valid."""
        with pytest.raises(ValidationError):
            BasisSelectionConfig(
                name="test",
                basis_scale_range=(10.0, 1.0),  # low > high
            )


class TestMeshRefinementConfig:
    """Tests for MeshRefinementConfig."""

    def test_create_default_config(self) -> None:
        """Test creating default mesh refinement config."""
        config = MeshRefinementConfig(name="test_mesh")
        assert config.initial_resolution == 8
        assert config.refinement_strategy == RefinementStrategy.H_REFINEMENT

    def test_h_refinement_config(self) -> None:
        """Test h-refinement configuration."""
        config = MeshRefinementConfig(
            name="test_h",
            refinement_strategy=RefinementStrategy.H_REFINEMENT,
            max_refinement_level=8,
        )
        assert config.refinement_strategy == RefinementStrategy.H_REFINEMENT
        assert config.max_refinement_level == 8

    def test_p_refinement_config(self) -> None:
        """Test p-refinement configuration."""
        config = MeshRefinementConfig(
            name="test_p",
            refinement_strategy=RefinementStrategy.P_REFINEMENT,
            max_polynomial_degree=15,
        )
        assert config.max_polynomial_degree == 15

    def test_hp_refinement_config(self) -> None:
        """Test hp-refinement configuration."""
        config = MeshRefinementConfig(
            name="test_hp",
            refinement_strategy=RefinementStrategy.HP_REFINEMENT,
        )
        assert config.refinement_strategy == RefinementStrategy.HP_REFINEMENT

    def test_resolution_validation(self) -> None:
        """Test that initial <= max resolution."""
        with pytest.raises(ValidationError):
            MeshRefinementConfig(
                name="test",
                initial_resolution=128,
                max_resolution=64,
            )

    def test_polynomial_degree_validation(self) -> None:
        """Test that initial <= max polynomial degree."""
        with pytest.raises(ValidationError):
            MeshRefinementConfig(
                name="test",
                initial_polynomial_degree=15,
                max_polynomial_degree=10,
            )


class TestPDEGameConfig:
    """Tests for PDEGameConfig."""

    def test_create_basis_selection_game(self) -> None:
        """Test creating basis selection game config."""
        pde = PDEConfig(name="poisson", pde_type=PDEType.POISSON)
        config = PDEGameConfig(
            name="test_game",
            pde_config=pde,
            game_mode="basis_selection",
        )
        assert config.game_mode == "basis_selection"
        assert config.basis_config is not None  # Auto-created

    def test_create_mesh_refinement_game(self) -> None:
        """Test creating mesh refinement game config."""
        pde = PDEConfig(name="poisson", pde_type=PDEType.POISSON)
        config = PDEGameConfig(
            name="test_game",
            pde_config=pde,
            game_mode="mesh_refinement",
        )
        assert config.game_mode == "mesh_refinement"
        assert config.mesh_config is not None  # Auto-created

    def test_explicit_sub_config(self) -> None:
        """Test providing explicit sub-configuration."""
        pde = PDEConfig(name="poisson", pde_type=PDEType.POISSON)
        basis = BasisSelectionConfig(
            name="custom_basis",
            max_basis_functions=50,
        )
        config = PDEGameConfig(
            name="test_game",
            pde_config=pde,
            game_mode="basis_selection",
            basis_config=basis,
        )
        assert config.basis_config.max_basis_functions == 50

    def test_game_parameters(self) -> None:
        """Test game-level parameters."""
        pde = PDEConfig(name="poisson", pde_type=PDEType.POISSON)
        config = PDEGameConfig(
            name="test_game",
            pde_config=pde,
            max_dof=5000,
            max_steps=50,
            error_tolerance=1e-5,
            cost_per_dof=0.1,
        )
        assert config.max_dof == 5000
        assert config.max_steps == 50
        assert config.error_tolerance == 1e-5
        assert config.cost_per_dof == 0.1

    def test_success_metrics(self) -> None:
        """Test success metric configuration."""
        pde = PDEConfig(name="poisson", pde_type=PDEType.POISSON)
        config = PDEGameConfig(
            name="test_game",
            pde_config=pde,
        )
        assert len(config.success_metrics) >= 1
        assert config.success_metrics[0].name == "final_error"


class TestEnums:
    """Tests for configuration enums."""

    def test_pde_type_values(self) -> None:
        """Test PDEType enum values."""
        assert PDEType.POISSON.value == "poisson"
        assert PDEType.BURGERS.value == "burgers"
        assert PDEType.HEAT.value == "heat"

    def test_boundary_condition_values(self) -> None:
        """Test BoundaryCondition enum values."""
        assert BoundaryCondition.DIRICHLET.value == "dirichlet"
        assert BoundaryCondition.NEUMANN.value == "neumann"
        assert BoundaryCondition.PERIODIC.value == "periodic"

    def test_refinement_strategy_values(self) -> None:
        """Test RefinementStrategy enum values."""
        assert RefinementStrategy.H_REFINEMENT.value == "h"
        assert RefinementStrategy.P_REFINEMENT.value == "p"
        assert RefinementStrategy.HP_REFINEMENT.value == "hp"

    def test_action_space_values(self) -> None:
        """Test ActionSpace enum values."""
        assert ActionSpace.DISCRETE.value == "discrete"
        assert ActionSpace.CONTINUOUS.value == "continuous"
