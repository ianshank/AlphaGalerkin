"""DecompositionAgent for splitting coupled PDE systems into subproblems.

Applies decomposition strategies (operator splitting, domain decomposition,
dimensional reduction) to produce a list of subproblem specifications
that can be solved by individual SolverAgents.

Example:
    from src.agents.decomposition import DecompositionAgent, SubproblemSpec
    from src.agents.config import DecompositionConfig, DecompositionStrategy

    config = DecompositionConfig(
        name="decomp",
        strategy=DecompositionStrategy.DOMAIN_DECOMPOSITION,
    )
    agent = DecompositionAgent(config, multi_physics=mp_config)
    agent.setup()
    subproblems = agent.decompose()

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.agents.base import AgentState, BaseAgent
from src.agents.config import DecompositionStrategy
from src.templates.base import ExecutionStatus
from src.templates.logging import create_logger_class

if TYPE_CHECKING:
    from src.agents.config import DecompositionConfig, MultiPhysicsConfig, SolverAgentConfig
    from src.agents.message import MessageBus
    from src.pde.config import PDEConfig

DecompLogger = create_logger_class("DecompositionAgent")


@dataclass
class SubproblemSpec:
    """Specification for a decomposed subproblem.

    Attributes:
        name: Unique name for this subproblem.
        pde_config: PDE configuration for this subproblem.
        subdomain_min: Minimum coordinates of the subdomain.
        subdomain_max: Maximum coordinates of the subdomain.
        budget_fraction: Fraction of total budget allocated.
        solver_config: Solver configuration for this subproblem.
        coupling_neighbors: Names of neighboring subproblems for coupling.

    """

    name: str
    pde_config: PDEConfig
    subdomain_min: list[float]
    subdomain_max: list[float]
    budget_fraction: float
    solver_config: SolverAgentConfig | None = None
    coupling_neighbors: list[str] = field(default_factory=list)


class DecompositionAgent(BaseAgent):
    """Agent that decomposes coupled PDE systems into subproblems.

    This is a one-shot agent: ``decompose()`` runs the decomposition
    strategy and returns subproblem specifications. The ``step()``
    method wraps ``decompose()`` for lifecycle compatibility.

    Args:
        config: Decomposition configuration.
        multi_physics: Multi-physics problem specification.
        message_bus: Optional message bus.
        agent_id: Optional explicit ID.

    """

    def __init__(
        self,
        config: DecompositionConfig,
        multi_physics: MultiPhysicsConfig,
        message_bus: MessageBus | None = None,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(config, message_bus, agent_id)
        self._multi_physics = multi_physics
        self._subproblems: list[SubproblemSpec] = []
        self._decomp_logger = DecompLogger(
            component="decomposition",
            run_id=self._agent_id,
        )

    @property
    def decomp_config(self) -> DecompositionConfig:
        """Typed access to decomposition-specific config."""
        from src.agents.config import DecompositionConfig

        assert isinstance(self.config, DecompositionConfig)
        return self.config

    @property
    def subproblems(self) -> list[SubproblemSpec]:
        """List of decomposed subproblems (populated after decompose())."""
        return self._subproblems

    def setup(self) -> None:
        """Validate multi-physics configuration."""
        self._decomp_logger.info(
            "decomposition_setup",
            strategy=self.decomp_config.strategy.value,
            n_physics=len(self._multi_physics.physics),
            n_coupling_pairs=len(self._multi_physics.coupling_pairs),
        )

    def step(self) -> AgentState:
        """Run decomposition and mark as complete.

        Returns:
            Updated agent state.

        """
        self._subproblems = self.decompose()
        self._state.step += 1
        self._state.status = ExecutionStatus.COMPLETED
        self._state.metrics = self.get_metrics()
        return self._state

    def decompose(self) -> list[SubproblemSpec]:
        """Decompose the multi-physics problem into subproblems.

        Also updates ``self._subproblems`` so the ``subproblems``
        property is always in sync.

        Returns:
            List of subproblem specifications.

        """
        strategy = self.decomp_config.strategy

        if strategy == DecompositionStrategy.OPERATOR_SPLITTING:
            result = self._operator_splitting()
        elif strategy == DecompositionStrategy.DOMAIN_DECOMPOSITION:
            result = self._domain_decomposition()
        elif strategy == DecompositionStrategy.DIMENSIONAL_REDUCTION:
            result = self._dimensional_reduction()
        else:
            msg = f"Unknown decomposition strategy: {strategy}"
            raise ValueError(msg)

        self._subproblems = result
        return result

    def _operator_splitting(self) -> list[SubproblemSpec]:
        """Split coupled PDEs into sequential single-PDE solves.

        Each physics becomes one subproblem. Budget is allocated
        equally or according to budget_allocation from config.
        """
        physics_list = self._multi_physics.physics
        n_physics = len(physics_list)
        budget_alloc = self._multi_physics.budget_allocation

        subproblems: list[SubproblemSpec] = []
        for pde_config in physics_list:
            fraction = budget_alloc.get(pde_config.name, 1.0 / n_physics)

            neighbors = []
            for pair in self._multi_physics.coupling_pairs:
                if pair.physics_a == pde_config.name:
                    neighbors.append(pair.physics_b)
                elif pair.physics_b == pde_config.name:
                    neighbors.append(pair.physics_a)

            subproblems.append(
                SubproblemSpec(
                    name=pde_config.name,
                    pde_config=pde_config,
                    subdomain_min=list(pde_config.domain_min),
                    subdomain_max=list(pde_config.domain_max),
                    budget_fraction=fraction,
                    coupling_neighbors=neighbors,
                )
            )

        self._decomp_logger.info(
            "operator_splitting_complete",
            n_subproblems=len(subproblems),
            order=self.decomp_config.splitting_order,
        )
        return subproblems

    def _domain_decomposition(self) -> list[SubproblemSpec]:
        """Split the spatial domain into overlapping subdomains.

        Currently implements 1D strip decomposition along the first axis.
        Each subdomain shares the same PDE but covers a subset of the domain
        with configurable overlap.
        """
        if not self._multi_physics.physics:
            return []

        pde_config = self._multi_physics.physics[0]
        d_min = pde_config.domain_min
        d_max = pde_config.domain_max

        n_physics = len(self._multi_physics.physics)
        n_subdomains = (
            min(self.decomp_config.max_subproblems, n_physics)
            if n_physics > 1
            else self.decomp_config.max_subproblems
        )
        n_subdomains = max(2, n_subdomains)

        split_axis = 0
        axis_length = d_max[split_axis] - d_min[split_axis]
        strip_width = axis_length / n_subdomains
        overlap = self.decomp_config.overlap_fraction * strip_width

        subproblems: list[SubproblemSpec] = []
        for i in range(n_subdomains):
            sub_min = list(d_min)
            sub_max = list(d_max)

            sub_min[split_axis] = d_min[split_axis] + i * strip_width - (overlap if i > 0 else 0.0)
            sub_max[split_axis] = (
                d_min[split_axis]
                + (i + 1) * strip_width
                + (overlap if i < n_subdomains - 1 else 0.0)
            )

            sub_min[split_axis] = max(sub_min[split_axis], d_min[split_axis])
            sub_max[split_axis] = min(sub_max[split_axis], d_max[split_axis])

            neighbors = []
            if i > 0:
                neighbors.append(f"subdomain_{i - 1}")
            if i < n_subdomains - 1:
                neighbors.append(f"subdomain_{i + 1}")

            sub_pde = pde_config.with_overrides(
                name=f"{pde_config.name}_sub_{i}",
                domain_min=sub_min,
                domain_max=sub_max,
            )

            subproblems.append(
                SubproblemSpec(
                    name=f"subdomain_{i}",
                    pde_config=sub_pde,
                    subdomain_min=sub_min,
                    subdomain_max=sub_max,
                    budget_fraction=1.0 / n_subdomains,
                    coupling_neighbors=neighbors,
                )
            )

        self._decomp_logger.info(
            "domain_decomposition_complete",
            n_subdomains=len(subproblems),
            overlap_fraction=self.decomp_config.overlap_fraction,
        )
        return subproblems

    def _dimensional_reduction(self) -> list[SubproblemSpec]:
        """Reduce problem dimensionality for thin domains.

        For each physics, if the domain extent along any axis is
        below a threshold relative to other axes, that dimension
        is collapsed (domain_dim reduced by 1).
        """
        subproblems: list[SubproblemSpec] = []
        n_physics = len(self._multi_physics.physics)
        budget_alloc = self._multi_physics.budget_allocation

        for pde_config in self._multi_physics.physics:
            fraction = budget_alloc.get(pde_config.name, 1.0 / n_physics)
            d_min = list(pde_config.domain_min)
            d_max = list(pde_config.domain_max)
            dim = pde_config.domain_dim

            extents = [d_max[d] - d_min[d] for d in range(dim)]
            max_extent = max(extents) if extents else 1.0

            reduced_min = []
            reduced_max = []
            kept_dims: list[int] = []
            for d in range(dim):
                ratio = extents[d] / max_extent if max_extent > 0 else 1.0
                if ratio > self.decomp_config.dimensional_reduction_threshold or dim <= 1:
                    reduced_min.append(d_min[d])
                    reduced_max.append(d_max[d])
                    kept_dims.append(d)

            new_dim = len(reduced_min)
            self._decomp_logger.info(
                "dimension_analysis",
                physics=pde_config.name,
                original_dim=dim,
                kept_dims=kept_dims,
                new_dim=new_dim,
            )
            if new_dim == 0:
                reduced_min = d_min[:1]
                reduced_max = d_max[:1]
                kept_dims = [0]
                new_dim = 1

            # Adjust advection_coeff to match reduced dimensions
            adv_coeff = list(pde_config.advection_coeff)
            if len(adv_coeff) >= dim:
                reduced_adv = [adv_coeff[d] for d in kept_dims]
            elif len(adv_coeff) >= new_dim:
                reduced_adv = adv_coeff[:new_dim]
            else:
                reduced_adv = adv_coeff + [0.0] * (new_dim - len(adv_coeff))

            sub_pde = pde_config.with_overrides(
                name=f"{pde_config.name}_reduced",
                domain_dim=new_dim,
                domain_min=reduced_min,
                domain_max=reduced_max,
                advection_coeff=reduced_adv,
            )

            subproblems.append(
                SubproblemSpec(
                    name=f"{pde_config.name}_reduced",
                    pde_config=sub_pde,
                    subdomain_min=reduced_min,
                    subdomain_max=reduced_max,
                    budget_fraction=fraction,
                )
            )

        self._decomp_logger.info(
            "dimensional_reduction_complete",
            n_subproblems=len(subproblems),
        )
        return subproblems

    def reset(self) -> None:
        """Reset decomposition state."""
        self._subproblems = []
        self._state = self._create_initial_state()

    def get_metrics(self) -> dict[str, float]:
        """Return decomposition metrics."""
        return {
            "n_subproblems": float(len(self._subproblems)),
            "overlap_fraction": self.decomp_config.overlap_fraction,
        }
