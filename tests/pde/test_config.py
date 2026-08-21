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

    def test_zero_measure_domain_rejected(self) -> None:
        """A degenerate (zero-measure / single-point) domain must be rejected.

        ``validate_domain`` uses ``lo >= hi`` (not strict ``>``), so
        ``domain_min == domain_max`` in any dimension -- collapsing that axis
        to a single point -- must raise, not just the strictly-inverted case
        already covered by ``test_domain_min_max_validation``.
        """
        with pytest.raises(ValidationError):
            PDEConfig(
                name="test",
                pde_type=PDEType.POISSON,
                domain_min=[0.0, 0.0],
                domain_max=[0.0, 1.0],  # x-axis collapses to a single point
            )

    def test_fully_degenerate_domain_rejected(self) -> None:
        """Every axis collapsed to a point (domain_min == domain_max) is rejected."""
        with pytest.raises(ValidationError):
            PDEConfig(
                name="test",
                pde_type=PDEType.POISSON,
                domain_min=[0.5, 0.5],
                domain_max=[0.5, 0.5],
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

    def test_hp_switchover_above_max_refinement_level_rejected(self) -> None:
        """Switchover level above the refinement cap makes p-refinement unreachable."""
        with pytest.raises(ValidationError, match="p-refinement branch"):
            MeshRefinementConfig(
                name="test",
                refinement_strategy=RefinementStrategy.HP_REFINEMENT,
                hp_switchover_level=8,
                max_refinement_level=5,
            )

    def test_hp_switchover_equal_to_max_refinement_level_rejected(self) -> None:
        """Equality is degenerate too: the p-refinement level window is empty."""
        with pytest.raises(ValidationError, match="p-refinement branch"):
            MeshRefinementConfig(
                name="test",
                refinement_strategy=RefinementStrategy.HP_REFINEMENT,
                hp_switchover_level=5,
                max_refinement_level=5,
            )

    def test_hp_switchover_one_below_max_refinement_level_accepted(self) -> None:
        """The tightest non-degenerate case keeps exactly one p-refinable level."""
        config = MeshRefinementConfig(
            name="test",
            refinement_strategy=RefinementStrategy.HP_REFINEMENT,
            hp_switchover_level=4,
            max_refinement_level=5,
        )
        assert config.hp_switchover_level == 4
        assert config.max_refinement_level == 5

    @pytest.mark.parametrize(
        "strategy",
        [RefinementStrategy.H_REFINEMENT, RefinementStrategy.P_REFINEMENT],
    )
    @pytest.mark.parametrize("max_refinement_level", [1, 2])
    def test_shallow_budget_accepted_for_non_hp_strategies(
        self,
        strategy: RefinementStrategy,
        max_refinement_level: int,
    ) -> None:
        """A shallow refinement budget is legitimate whenever hp is not in play.

        Regression test for an over-broad cross-check: the
        ``hp_switchover_level < max_refinement_level`` rule was applied
        unconditionally, so every config with ``max_refinement_level <= 2``
        (the default ``hp_switchover_level`` is 2) was rejected -- including
        pure h- and p-refinement, which never read ``hp_switchover_level``
        (``Mesh.refine_element`` dispatches on the strategy first). That made
        the advertised ``max_refinement_level >= 1`` bound unreachable for the
        *default* strategy.
        """
        config = MeshRefinementConfig(
            name="smoke",
            refinement_strategy=strategy,
            max_refinement_level=max_refinement_level,
        )

        assert config.max_refinement_level == max_refinement_level
        # The inert field keeps its default; it is simply never consulted.
        assert config.hp_switchover_level == 2

    def test_shallow_budget_accepted_on_default_strategy(self) -> None:
        """The exact call from the defect report: a fast shallow smoke config."""
        config = MeshRefinementConfig(name="smoke", max_refinement_level=2)

        assert config.refinement_strategy == RefinementStrategy.H_REFINEMENT
        assert config.max_refinement_level == 2

    @pytest.mark.parametrize(
        "strategy",
        [RefinementStrategy.H_REFINEMENT, RefinementStrategy.P_REFINEMENT],
    )
    def test_degenerate_hp_window_accepted_for_non_hp_strategies(
        self,
        strategy: RefinementStrategy,
    ) -> None:
        """The same (switchover, cap) pair that hp rejects is fine without hp.

        Pairs with ``test_hp_switchover_equal_to_max_refinement_level_rejected``
        to pin the gate from both sides: identical field values, opposite
        outcomes, decided solely by ``refinement_strategy``.
        """
        config = MeshRefinementConfig(
            name="test",
            refinement_strategy=strategy,
            hp_switchover_level=5,
            max_refinement_level=5,
        )

        assert config.hp_switchover_level == config.max_refinement_level == 5

    def test_hp_switchover_upper_bound(self) -> None:
        """hp_switchover_level carries the same standalone bound as its sibling.

        Load-bearing under the default (h) strategy: the hp cross-check is
        gated off there, so ``le=20`` is the only guard left.
        """
        with pytest.raises(ValidationError):
            MeshRefinementConfig(name="test", hp_switchover_level=21)

    def test_hp_switchover_upper_bound_under_hp_strategy(self) -> None:
        """The standalone bound also fires before the (looser) cross-check."""
        with pytest.raises(ValidationError):
            MeshRefinementConfig(
                name="test",
                refinement_strategy=RefinementStrategy.HP_REFINEMENT,
                hp_switchover_level=21,
                max_refinement_level=20,
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

    def test_winner_threshold_defaults(self) -> None:
        """Winner thresholds default to the historical 0.1 / 0.5 values."""
        pde = PDEConfig(name="poisson", pde_type=PDEType.POISSON)
        config = PDEGameConfig(name="test_game", pde_config=pde)
        assert config.winner_good_reduction_threshold == 0.1
        assert config.winner_poor_reduction_threshold == 0.5

    def test_winner_threshold_override(self) -> None:
        """Winner thresholds accept overrides in (0, 1) with good < poor."""
        pde = PDEConfig(name="poisson", pde_type=PDEType.POISSON)
        config = PDEGameConfig(
            name="test_game",
            pde_config=pde,
            winner_good_reduction_threshold=0.05,
            winner_poor_reduction_threshold=0.8,
        )
        assert config.winner_good_reduction_threshold == 0.05
        assert config.winner_poor_reduction_threshold == 0.8

    def test_winner_threshold_ordering_enforced(self) -> None:
        """Reject configurations where good >= poor at validator time."""
        pde = PDEConfig(name="poisson", pde_type=PDEType.POISSON)
        with pytest.raises(ValidationError):
            PDEGameConfig(
                name="test_game",
                pde_config=pde,
                winner_good_reduction_threshold=0.6,
                winner_poor_reduction_threshold=0.4,
            )
        with pytest.raises(ValidationError):
            PDEGameConfig(
                name="test_game",
                pde_config=pde,
                winner_good_reduction_threshold=0.3,
                winner_poor_reduction_threshold=0.3,
            )

    def test_winner_threshold_bounds(self) -> None:
        """Threshold fields enforce (0, 1) open bounds."""
        pde = PDEConfig(name="poisson", pde_type=PDEType.POISSON)
        with pytest.raises(ValidationError):
            PDEGameConfig(
                name="test_game",
                pde_config=pde,
                winner_good_reduction_threshold=0.0,
            )
        with pytest.raises(ValidationError):
            PDEGameConfig(
                name="test_game",
                pde_config=pde,
                winner_poor_reduction_threshold=1.0,
            )


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
