"""Tests for DecompositionAgent."""

from __future__ import annotations

import pytest

from src.agents.config import (
    DecompositionConfig,
    DecompositionStrategy,
    MultiPhysicsConfig,
)
from src.agents.decomposition import DecompositionAgent, SubproblemSpec
from src.pde.config import PDEConfig, PDEType
from src.templates.base import ExecutionStatus


class TestSubproblemSpec:
    """Tests for SubproblemSpec dataclass."""

    def test_creation(self, poisson_pde_config: PDEConfig) -> None:
        spec = SubproblemSpec(
            name="sub_0",
            pde_config=poisson_pde_config,
            subdomain_min=[0.0, 0.0],
            subdomain_max=[0.5, 1.0],
            budget_fraction=0.5,
        )
        assert spec.name == "sub_0"
        assert spec.budget_fraction == 0.5
        assert spec.coupling_neighbors == []
        assert spec.solver_config is None


class TestDecompositionAgent:
    """Tests for DecompositionAgent."""

    @pytest.fixture
    def decomp_agent(
        self,
        sample_decomposition_config: DecompositionConfig,
        sample_multi_physics_config: MultiPhysicsConfig,
    ) -> DecompositionAgent:
        return DecompositionAgent(
            config=sample_decomposition_config,
            multi_physics=sample_multi_physics_config,
        )

    def test_initial_state(self, decomp_agent: DecompositionAgent) -> None:
        assert decomp_agent.state.step == 0
        assert decomp_agent.subproblems == []

    def test_setup(self, decomp_agent: DecompositionAgent) -> None:
        decomp_agent.setup()  # Should not raise

    def test_decomp_config_property(self, decomp_agent: DecompositionAgent) -> None:
        assert decomp_agent.decomp_config.strategy == DecompositionStrategy.OPERATOR_SPLITTING

    # -- Operator Splitting --

    def test_operator_splitting(self, decomp_agent: DecompositionAgent) -> None:
        decomp_agent.setup()
        subproblems = decomp_agent.decompose()
        assert len(subproblems) == 2  # Two physics
        assert subproblems[0].name == "poisson_test"
        assert subproblems[1].name == "heat_test"

    def test_operator_splitting_budget_fractions(
        self, decomp_agent: DecompositionAgent
    ) -> None:
        decomp_agent.setup()
        subproblems = decomp_agent.decompose()
        total_budget = sum(s.budget_fraction for s in subproblems)
        assert total_budget == pytest.approx(1.0)

    def test_operator_splitting_coupling_neighbors(
        self, decomp_agent: DecompositionAgent
    ) -> None:
        decomp_agent.setup()
        subproblems = decomp_agent.decompose()
        assert "heat_test" in subproblems[0].coupling_neighbors
        assert "poisson_test" in subproblems[1].coupling_neighbors

    # -- Domain Decomposition --

    def test_domain_decomposition(
        self,
        sample_multi_physics_config: MultiPhysicsConfig,
    ) -> None:
        config = DecompositionConfig(
            name="dd",
            strategy=DecompositionStrategy.DOMAIN_DECOMPOSITION,
            max_subproblems=4,
            overlap_fraction=0.1,
        )
        agent = DecompositionAgent(config, sample_multi_physics_config)
        agent.setup()
        subproblems = agent.decompose()
        assert len(subproblems) >= 2

    def test_domain_decomposition_overlap(
        self,
        sample_multi_physics_config: MultiPhysicsConfig,
    ) -> None:
        config = DecompositionConfig(
            name="dd",
            strategy=DecompositionStrategy.DOMAIN_DECOMPOSITION,
            max_subproblems=2,
            overlap_fraction=0.2,
        )
        agent = DecompositionAgent(config, sample_multi_physics_config)
        agent.setup()
        subproblems = agent.decompose()

        # Adjacent subdomains should overlap
        if len(subproblems) >= 2:
            s0_max = subproblems[0].subdomain_max[0]
            s1_min = subproblems[1].subdomain_min[0]
            assert s0_max > s1_min  # Overlap exists

    def test_domain_decomposition_budget(
        self,
        sample_multi_physics_config: MultiPhysicsConfig,
    ) -> None:
        config = DecompositionConfig(
            name="dd",
            strategy=DecompositionStrategy.DOMAIN_DECOMPOSITION,
            max_subproblems=3,
        )
        agent = DecompositionAgent(config, sample_multi_physics_config)
        agent.setup()
        subproblems = agent.decompose()
        total = sum(s.budget_fraction for s in subproblems)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_domain_decomposition_neighbors(
        self,
        sample_multi_physics_config: MultiPhysicsConfig,
    ) -> None:
        config = DecompositionConfig(
            name="dd",
            strategy=DecompositionStrategy.DOMAIN_DECOMPOSITION,
            max_subproblems=3,
        )
        agent = DecompositionAgent(config, sample_multi_physics_config)
        agent.setup()
        subproblems = agent.decompose()
        if len(subproblems) >= 2:
            assert "subdomain_1" in subproblems[0].coupling_neighbors

    # -- Dimensional Reduction --

    def test_dimensional_reduction(self) -> None:
        thin_config = PDEConfig(
            name="thin_domain",
            pde_type=PDEType.POISSON,
            domain_dim=3,
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 0.01],  # Very thin in z
            boundary_value=0.0,
            advection_coeff=[0.0, 0.0, 0.0],
        )
        mp_config = MultiPhysicsConfig(
            name="thin_mp",
            physics=[thin_config],
        )
        decomp_config = DecompositionConfig(
            name="dr",
            strategy=DecompositionStrategy.DIMENSIONAL_REDUCTION,
        )
        agent = DecompositionAgent(decomp_config, mp_config)
        agent.setup()
        subproblems = agent.decompose()
        assert len(subproblems) == 1
        # z-dimension should be collapsed
        assert subproblems[0].pde_config.domain_dim < 3

    def test_dimensional_reduction_no_collapse(
        self,
        sample_multi_physics_config: MultiPhysicsConfig,
    ) -> None:
        config = DecompositionConfig(
            name="dr",
            strategy=DecompositionStrategy.DIMENSIONAL_REDUCTION,
        )
        agent = DecompositionAgent(config, sample_multi_physics_config)
        agent.setup()
        subproblems = agent.decompose()
        # Square domain: no dimensions should be collapsed
        for sp in subproblems:
            assert sp.pde_config.domain_dim == 2

    # -- Step and Lifecycle --

    def test_step_completes(self, decomp_agent: DecompositionAgent) -> None:
        decomp_agent.setup()
        state = decomp_agent.step()
        assert state.status == ExecutionStatus.COMPLETED
        assert state.step == 1
        assert decomp_agent.subproblems != []

    def test_reset(self, decomp_agent: DecompositionAgent) -> None:
        decomp_agent.setup()
        decomp_agent.step()
        decomp_agent.reset()
        assert decomp_agent.state.step == 0
        assert decomp_agent.subproblems == []

    def test_get_metrics(self, decomp_agent: DecompositionAgent) -> None:
        decomp_agent.setup()
        decomp_agent.decompose()
        decomp_agent._subproblems = decomp_agent.decompose()
        metrics = decomp_agent.get_metrics()
        assert "n_subproblems" in metrics
        assert "overlap_fraction" in metrics

    @pytest.mark.parametrize("strategy", list(DecompositionStrategy))
    def test_all_strategies(
        self,
        strategy: DecompositionStrategy,
        sample_multi_physics_config: MultiPhysicsConfig,
    ) -> None:
        config = DecompositionConfig(name="test", strategy=strategy)
        agent = DecompositionAgent(config, sample_multi_physics_config)
        agent.setup()
        subproblems = agent.decompose()
        assert len(subproblems) >= 1


class TestDecompositionEdgeCases:
    """Edge case tests for DecompositionAgent."""

    def test_domain_decomposition_empty_physics_internal(self) -> None:
        """Domain decomposition with empty physics returns empty list.

        Uses object.__setattr__ to bypass pydantic validate_assignment.
        """
        pde = PDEConfig(
            name="placeholder",
            pde_type=PDEType.POISSON,
        )
        mp_config = MultiPhysicsConfig(name="test", physics=[pde])
        config = DecompositionConfig(
            name="test",
            strategy=DecompositionStrategy.DOMAIN_DECOMPOSITION,
        )
        agent = DecompositionAgent(config, mp_config)
        agent.setup()
        # Bypass pydantic validation to test the guard clause
        object.__setattr__(agent._multi_physics, "physics", [])
        subproblems = agent.decompose()
        assert subproblems == []

    def test_dimensional_reduction_all_equal_extents(self) -> None:
        """When all dimensions have equal extents, all are kept."""
        pde = PDEConfig(
            name="cube",
            pde_type=PDEType.POISSON,
            domain_dim=3,
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 1.0],
            advection_coeff=[0.0, 0.0, 0.0],
        )
        mp_config = MultiPhysicsConfig(name="cube_mp", physics=[pde])
        config = DecompositionConfig(
            name="test",
            strategy=DecompositionStrategy.DIMENSIONAL_REDUCTION,
        )
        agent = DecompositionAgent(config, mp_config)
        agent.setup()
        subproblems = agent.decompose()
        assert len(subproblems) >= 1
        # All 3 dims have equal extents, so all should be kept (3D → 3D)
        for sp in subproblems:
            assert len(sp.subdomain_min) == 3

    def test_dimensional_reduction_thin_domain(self) -> None:
        """A very thin domain drops the thin dimension."""
        pde = PDEConfig(
            name="thin",
            pde_type=PDEType.POISSON,
            domain_dim=3,
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[10.0, 10.0, 0.001],
            advection_coeff=[0.0, 0.0, 0.0],
        )
        mp_config = MultiPhysicsConfig(name="thin_mp", physics=[pde])
        config = DecompositionConfig(
            name="test",
            strategy=DecompositionStrategy.DIMENSIONAL_REDUCTION,
        )
        agent = DecompositionAgent(config, mp_config)
        agent.setup()
        subproblems = agent.decompose()
        assert len(subproblems) >= 1
        # The z dimension is thin (0.001 / 10.0 = 0.0001 < 0.1), should be dropped
        for sp in subproblems:
            assert len(sp.subdomain_min) == 2
