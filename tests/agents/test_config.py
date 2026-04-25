"""Tests for agent-physics configuration classes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.config import (
    AgentConfig,
    AgentType,
    CollocationConfig,
    CollocationStrategy,
    CouplingConfig,
    CouplingPairConfig,
    CouplingType,
    DecompositionConfig,
    DecompositionStrategy,
    MessageBusConfig,
    MessageType,
    MultiPhysicsConfig,
    OrchestratorConfig,
    SolverAgentConfig,
)
from src.pde.config import PDEConfig, PDEType


class TestAgentType:
    """Tests for AgentType enum."""

    @pytest.mark.parametrize(
        "value",
        [AgentType.SOLVER, AgentType.DECOMPOSITION, AgentType.COUPLING, AgentType.META],
    )
    def test_valid_agent_types(self, value: AgentType) -> None:
        assert isinstance(value, AgentType)
        assert value.value == value.name.lower()

    def test_from_string(self) -> None:
        assert AgentType("solver") == AgentType.SOLVER


class TestDecompositionStrategy:
    """Tests for DecompositionStrategy enum."""

    @pytest.mark.parametrize(
        "value",
        [
            DecompositionStrategy.OPERATOR_SPLITTING,
            DecompositionStrategy.DOMAIN_DECOMPOSITION,
            DecompositionStrategy.DIMENSIONAL_REDUCTION,
        ],
    )
    def test_valid_strategies(self, value: DecompositionStrategy) -> None:
        assert isinstance(value, DecompositionStrategy)


class TestCouplingType:
    """Tests for CouplingType enum."""

    @pytest.mark.parametrize(
        "value",
        [CouplingType.DIRICHLET_NEUMANN, CouplingType.ROBIN_ROBIN, CouplingType.MORTAR],
    )
    def test_valid_types(self, value: CouplingType) -> None:
        assert isinstance(value, CouplingType)


class TestCollocationStrategy:
    """Tests for CollocationStrategy enum."""

    @pytest.mark.parametrize(
        "value",
        [
            CollocationStrategy.UNIFORM,
            CollocationStrategy.ADAPTIVE,
            CollocationStrategy.IMPORTANCE_WEIGHTED,
            CollocationStrategy.ERROR_GUIDED,
        ],
    )
    def test_valid_strategies(self, value: CollocationStrategy) -> None:
        assert isinstance(value, CollocationStrategy)


class TestMessageType:
    """Tests for MessageType enum."""

    @pytest.mark.parametrize(
        "value",
        [
            MessageType.STATE_UPDATE,
            MessageType.BOUNDARY_DATA,
            MessageType.CONVERGENCE_CHECK,
            MessageType.STRATEGY_CHANGE,
            MessageType.BUDGET_UPDATE,
        ],
    )
    def test_valid_types(self, value: MessageType) -> None:
        assert isinstance(value, MessageType)


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_valid_construction(self) -> None:
        config = AgentConfig(
            name="test",
            agent_type=AgentType.SOLVER,
        )
        assert config.agent_type == AgentType.SOLVER
        assert config.max_steps == 1000
        assert config.error_tolerance == 0.01
        assert config.computational_budget == 1.0

    def test_string_agent_type(self) -> None:
        config = AgentConfig(name="test", agent_type="solver")
        assert config.agent_type == AgentType.SOLVER

    def test_invalid_max_steps(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(name="test", agent_type=AgentType.SOLVER, max_steps=0)

    def test_invalid_tolerance(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(name="test", agent_type=AgentType.SOLVER, error_tolerance=0.0)

    def test_hash_deterministic(self) -> None:
        c1 = AgentConfig(name="test", agent_type=AgentType.SOLVER)
        c2 = AgentConfig(name="test", agent_type=AgentType.SOLVER)
        assert c1.compute_hash() == c2.compute_hash()

    def test_hash_changes_with_params(self) -> None:
        c1 = AgentConfig(name="test", agent_type=AgentType.SOLVER)
        c2 = AgentConfig(name="test", agent_type=AgentType.COUPLING)
        assert c1.compute_hash() != c2.compute_hash()

    def test_with_overrides(self) -> None:
        original = AgentConfig(name="test", agent_type=AgentType.SOLVER)
        overridden = original.with_overrides(max_steps=500)
        assert overridden.max_steps == 500
        assert original.max_steps == 1000

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(name="test", agent_type=AgentType.SOLVER, unknown_field=42)


class TestSolverAgentConfig:
    """Tests for SolverAgentConfig."""

    def test_valid_construction(self) -> None:
        config = SolverAgentConfig(name="solver")
        assert config.agent_type == AgentType.SOLVER
        assert config.game_mode == "basis_selection"
        assert config.n_simulations == 200
        assert config.temperature_start == 1.0
        assert config.temperature_end == 0.1

    def test_mesh_refinement_mode(self) -> None:
        config = SolverAgentConfig(name="solver", game_mode="mesh_refinement")
        assert config.game_mode == "mesh_refinement"

    def test_invalid_game_mode(self) -> None:
        with pytest.raises(ValidationError):
            SolverAgentConfig(name="solver", game_mode="invalid")

    def test_temperature_validation(self) -> None:
        with pytest.raises(ValidationError):
            SolverAgentConfig(
                name="solver",
                temperature_start=0.1,
                temperature_end=1.0,
            )

    def test_valid_temperature_equal(self) -> None:
        config = SolverAgentConfig(
            name="solver",
            temperature_start=0.5,
            temperature_end=0.5,
        )
        assert config.temperature_start == config.temperature_end

    def test_n_simulations_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SolverAgentConfig(name="solver", n_simulations=0)

    def test_hash_deterministic(self) -> None:
        c1 = SolverAgentConfig(name="solver")
        c2 = SolverAgentConfig(name="solver")
        assert c1.compute_hash() == c2.compute_hash()


class TestDecompositionConfig:
    """Tests for DecompositionConfig."""

    def test_valid_construction(self) -> None:
        config = DecompositionConfig(name="decomp")
        assert config.agent_type == AgentType.DECOMPOSITION
        assert config.strategy == DecompositionStrategy.OPERATOR_SPLITTING

    @pytest.mark.parametrize("strategy", list(DecompositionStrategy))
    def test_all_strategies(self, strategy: DecompositionStrategy) -> None:
        config = DecompositionConfig(name="decomp", strategy=strategy)
        assert config.strategy == strategy

    def test_overlap_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DecompositionConfig(name="decomp", overlap_fraction=0.6)


class TestCouplingConfig:
    """Tests for CouplingConfig."""

    def test_valid_construction(self) -> None:
        config = CouplingConfig(name="coupling")
        assert config.agent_type == AgentType.COUPLING
        assert config.coupling_type == CouplingType.DIRICHLET_NEUMANN
        assert config.relaxation_factor == 0.5

    @pytest.mark.parametrize("coupling_type", list(CouplingType))
    def test_all_coupling_types(self, coupling_type: CouplingType) -> None:
        config = CouplingConfig(name="coupling", coupling_type=coupling_type)
        assert config.coupling_type == coupling_type

    def test_relaxation_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CouplingConfig(name="coupling", relaxation_factor=0.0)
        with pytest.raises(ValidationError):
            CouplingConfig(name="coupling", relaxation_factor=1.5)


class TestCollocationConfig:
    """Tests for CollocationConfig."""

    def test_valid_construction(self) -> None:
        config = CollocationConfig(name="colloc")
        assert config.strategy == CollocationStrategy.UNIFORM
        assert config.n_points == 1000

    def test_point_bound_validation_min_gt_n(self) -> None:
        with pytest.raises(ValidationError):
            CollocationConfig(name="colloc", min_points=2000, n_points=1000)

    def test_point_bound_validation_n_gt_max(self) -> None:
        with pytest.raises(ValidationError):
            CollocationConfig(name="colloc", n_points=2000, max_points=1000)

    @pytest.mark.parametrize("strategy", list(CollocationStrategy))
    def test_all_strategies(self, strategy: CollocationStrategy) -> None:
        config = CollocationConfig(name="colloc", strategy=strategy)
        assert config.strategy == strategy


class TestMessageBusConfig:
    """Tests for MessageBusConfig."""

    def test_valid_construction(self) -> None:
        config = MessageBusConfig(name="bus")
        assert config.buffer_size == 1000
        assert config.enable_logging is False

    def test_custom_buffer(self) -> None:
        config = MessageBusConfig(name="bus", buffer_size=50)
        assert config.buffer_size == 50


class TestMultiPhysicsConfig:
    """Tests for MultiPhysicsConfig."""

    def test_valid_construction(self, poisson_pde_config: PDEConfig) -> None:
        config = MultiPhysicsConfig(
            name="mp",
            physics=[poisson_pde_config],
        )
        assert len(config.physics) == 1

    def test_coupling_references_valid(
        self,
        poisson_pde_config: PDEConfig,
        heat_pde_config: PDEConfig,
    ) -> None:
        config = MultiPhysicsConfig(
            name="mp",
            physics=[poisson_pde_config, heat_pde_config],
            coupling_pairs=[
                CouplingPairConfig(
                    name="pair",
                    physics_a="poisson_test",
                    physics_b="heat_test",
                ),
            ],
        )
        assert len(config.coupling_pairs) == 1

    def test_coupling_references_invalid(
        self,
        poisson_pde_config: PDEConfig,
    ) -> None:
        with pytest.raises(ValidationError, match="unknown physics"):
            MultiPhysicsConfig(
                name="mp",
                physics=[poisson_pde_config],
                coupling_pairs=[
                    CouplingPairConfig(
                        name="pair",
                        physics_a="poisson_test",
                        physics_b="nonexistent",
                    ),
                ],
            )

    def test_budget_allocation_valid(
        self,
        poisson_pde_config: PDEConfig,
        heat_pde_config: PDEConfig,
    ) -> None:
        config = MultiPhysicsConfig(
            name="mp",
            physics=[poisson_pde_config, heat_pde_config],
            budget_allocation={
                "poisson_test": 0.6,
                "heat_test": 0.4,
            },
        )
        assert sum(config.budget_allocation.values()) == pytest.approx(1.0)

    def test_budget_allocation_bad_sum(
        self,
        poisson_pde_config: PDEConfig,
        heat_pde_config: PDEConfig,
    ) -> None:
        with pytest.raises(ValidationError, match="sum to 1.0"):
            MultiPhysicsConfig(
                name="mp",
                physics=[poisson_pde_config, heat_pde_config],
                budget_allocation={
                    "poisson_test": 0.6,
                    "heat_test": 0.6,
                },
            )

    def test_budget_allocation_unknown_physics(
        self,
        poisson_pde_config: PDEConfig,
    ) -> None:
        with pytest.raises(ValidationError, match="unknown physics"):
            MultiPhysicsConfig(
                name="mp",
                physics=[poisson_pde_config],
                budget_allocation={
                    "nonexistent": 1.0,
                },
            )

    def test_empty_physics_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MultiPhysicsConfig(name="mp", physics=[])


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_valid_construction(
        self,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        assert sample_orchestrator_config.parallel_solvers is False
        assert len(sample_orchestrator_config.multi_physics.physics) == 2

    def test_default_sub_configs(
        self,
        poisson_pde_config: PDEConfig,
    ) -> None:
        config = OrchestratorConfig(
            name="orch",
            multi_physics=MultiPhysicsConfig(
                name="mp",
                physics=[poisson_pde_config],
            ),
        )
        assert config.decomposition is not None
        assert config.solver_defaults is not None
        assert config.coupling is not None
        assert config.collocation is not None
        assert config.message_bus is not None

    def test_hash_deterministic(
        self,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        h1 = sample_orchestrator_config.compute_hash()
        h2 = sample_orchestrator_config.compute_hash()
        assert h1 == h2


class TestMultiPhysicsCouplingValidation:
    """Tests for MultiPhysicsConfig coupling reference validation."""

    def test_invalid_coupling_physics_a(self) -> None:
        """Coupling pair referencing unknown physics_a raises ValueError."""
        pde = PDEConfig(
            name="real_physics",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
        )
        with pytest.raises(ValidationError, match="unknown physics.*fake_a"):
            MultiPhysicsConfig(
                name="bad",
                physics=[pde],
                coupling_pairs=[
                    CouplingPairConfig(
                        name="bad_pair",
                        physics_a="fake_a",
                        physics_b="real_physics",
                    ),
                ],
            )

    def test_invalid_coupling_physics_b(self) -> None:
        """Coupling pair referencing unknown physics_b raises ValueError."""
        pde = PDEConfig(
            name="real_physics",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
        )
        with pytest.raises(ValidationError, match="unknown physics.*fake_b"):
            MultiPhysicsConfig(
                name="bad",
                physics=[pde],
                coupling_pairs=[
                    CouplingPairConfig(
                        name="bad_pair",
                        physics_a="real_physics",
                        physics_b="fake_b",
                    ),
                ],
            )

    def test_budget_allocation_invalid_physics(self) -> None:
        """Budget allocation referencing unknown physics raises ValueError."""
        pde = PDEConfig(
            name="real_physics",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
        )
        with pytest.raises(ValidationError, match="unknown physics.*fake"):
            MultiPhysicsConfig(
                name="bad",
                physics=[pde],
                budget_allocation={"fake": 1.0},
            )
