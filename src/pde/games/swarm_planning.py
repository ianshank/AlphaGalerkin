"""Swarm planning game for MCTS-guided multi-agent coordination.

Models multi-agent swarm coordination as a sequential decision-making
problem solvable by MCTS. State represents agent positions/velocities,
actions are movement/formation commands, and reward reflects coverage,
safety, and communication objectives.

PDE connection: Uses potential field methods (Laplace equation) for
obstacle avoidance and flow-field-based formation control.

This bridges AlphaGalerkin's MCTS capabilities to the S500 drone
swarm project for SBIR proposals (AFWERX, DARPA Lift Challenge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray
from pydantic import Field, model_validator

from src.templates.config import BaseModuleConfig

logger = structlog.get_logger(__name__)

# --- Action definitions ---
# 7 discrete velocity changes per agent (sequential agent control)
ACTION_NAMES: list[str] = [
    "up",  # +z
    "down",  # -z
    "left",  # -y
    "right",  # +y
    "forward",  # +x
    "backward",  # -x
    "hover",  # zero velocity
]
N_ACTIONS_PER_AGENT: int = len(ACTION_NAMES)


# --- Configuration ---


class SwarmPlanningConfig(BaseModuleConfig):
    """Configuration for swarm planning game."""

    n_agents: int = Field(default=10, ge=2, le=1000, description="Number of agents in swarm")
    communication_range: float = Field(
        default=50.0, gt=0, description="Max comm range between agents"
    )
    collision_radius: float = Field(
        default=2.0, gt=0, description="Minimum safe distance between agents"
    )
    domain_size: tuple[float, float, float] = Field(
        default=(100.0, 100.0, 50.0), description="3D domain dimensions (x, y, z)"
    )
    max_velocity: float = Field(default=10.0, gt=0, description="Maximum agent velocity")
    n_obstacles: int = Field(default=5, ge=0, description="Number of obstacles")
    obstacle_radius: float = Field(default=5.0, gt=0, description="Obstacle radius")
    max_steps: int = Field(default=100, ge=1, description="Maximum game steps")
    dt: float = Field(default=1.0, gt=0, description="Time step for dynamics integration")

    # Coverage grid resolution for reward computation
    coverage_grid_res: int = Field(
        default=20, ge=5, le=200, description="Grid resolution per axis for coverage map"
    )
    coverage_sensor_range: float = Field(
        default=15.0, gt=0, description="Sensor range for coverage computation"
    )

    # Reward weights (configurable, no hardcoded values)
    coverage_weight: float = Field(default=1.0, ge=0, description="Weight for area coverage reward")
    collision_penalty: float = Field(default=10.0, ge=0, description="Penalty for collisions")
    communication_weight: float = Field(
        default=0.5, ge=0, description="Weight for comm graph connectivity"
    )
    energy_weight: float = Field(
        default=0.1, ge=0, description="Weight for energy consumption penalty"
    )
    obstacle_penalty: float = Field(
        default=10.0, ge=0, description="Penalty for obstacle collisions"
    )

    # Potential field parameters (PDE connection)
    potential_field_strength: float = Field(
        default=100.0, gt=0, description="Repulsive potential strength for obstacles"
    )
    potential_field_decay: float = Field(
        default=2.0, gt=0, description="Potential field distance decay exponent"
    )

    @model_validator(mode="after")
    def validate_swarm_config(self) -> SwarmPlanningConfig:
        """Validate swarm configuration consistency."""
        if self.collision_radius >= self.communication_range:
            raise ValueError(
                f"collision_radius ({self.collision_radius}) must be < "
                f"communication_range ({self.communication_range})"
            )
        return self


# --- State ---


@dataclass
class SwarmState:
    """State of the swarm at a given time step."""

    positions: NDArray[np.float64]  # (n_agents, 3)
    velocities: NDArray[np.float64]  # (n_agents, 3)
    obstacles: NDArray[np.float64]  # (n_obstacles, 4) - x, y, z, radius
    step: int = 0
    current_agent: int = 0  # Which agent MCTS controls this step (round-robin)
    coverage_map: NDArray[np.float64] | None = None  # Discretized coverage accumulator
    is_terminal: bool = False
    reward: float = 0.0
    cumulative_reward: float = 0.0
    history: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> SwarmState:
        """Deep copy."""
        return SwarmState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            obstacles=self.obstacles.copy(),
            step=self.step,
            current_agent=self.current_agent,
            coverage_map=self.coverage_map.copy() if self.coverage_map is not None else None,
            is_terminal=self.is_terminal,
            reward=self.reward,
            cumulative_reward=self.cumulative_reward,
            history=list(self.history),
            metadata=dict(self.metadata),
        )


# --- Game ---


class SwarmPlanningGame:
    """MCTS-guided swarm planning game.

    State: agent positions, velocities, obstacle map
    Actions: discrete velocity changes for one agent (round-robin)
    Reward: coverage + safety + communication + energy
    Terminal: max steps reached

    At each step the game controls one agent via round-robin scheduling.
    The action space is 7 discrete velocity changes for the current agent.
    """

    name = "swarm_planning"
    description = "MCTS-guided swarm coordination game"

    def __init__(self, config: SwarmPlanningConfig) -> None:
        """Initialize swarm planning game.

        Args:
            config: Swarm planning configuration.

        """
        self.config = config

        # Pre-compute velocity deltas (unit vectors scaled by max_velocity)
        v = config.max_velocity
        self._velocity_deltas = np.array(
            [
                [0.0, 0.0, v],  # up
                [0.0, 0.0, -v],  # down
                [0.0, -v, 0.0],  # left
                [0.0, v, 0.0],  # right
                [v, 0.0, 0.0],  # forward
                [-v, 0.0, 0.0],  # backward
                [0.0, 0.0, 0.0],  # hover
            ],
            dtype=np.float64,
        )

        self._rng = np.random.default_rng(config.seed)

        logger.info(
            "swarm_game_initialized",
            n_agents=config.n_agents,
            n_obstacles=config.n_obstacles,
            domain_size=config.domain_size,
            max_steps=config.max_steps,
        )

    @property
    def action_space_size(self) -> int:
        """Number of actions per step (one agent at a time)."""
        return N_ACTIONS_PER_AGENT

    def get_initial_state(self, seed: int | None = None) -> SwarmState:
        """Create initial swarm state with random positions and zero velocities.

        Args:
            seed: Optional random seed override.

        Returns:
            Initial SwarmState.

        """
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)

        domain = np.array(self.config.domain_size, dtype=np.float64)

        # Random initial positions within domain
        positions = rng.uniform(
            low=np.zeros(3),
            high=domain,
            size=(self.config.n_agents, 3),
        )

        # Zero initial velocities
        velocities = np.zeros((self.config.n_agents, 3), dtype=np.float64)

        # Generate obstacles (random within domain)
        if self.config.n_obstacles > 0:
            obs_positions = rng.uniform(
                low=np.zeros(3),
                high=domain,
                size=(self.config.n_obstacles, 3),
            )
            obs_radii = np.full(
                (self.config.n_obstacles, 1),
                self.config.obstacle_radius,
                dtype=np.float64,
            )
            obstacles = np.hstack([obs_positions, obs_radii])
        else:
            obstacles = np.empty((0, 4), dtype=np.float64)

        # Initialize coverage map
        res = self.config.coverage_grid_res
        coverage_map = np.zeros((res, res), dtype=np.float64)

        state = SwarmState(
            positions=positions,
            velocities=velocities,
            obstacles=obstacles,
            step=0,
            current_agent=0,
            coverage_map=coverage_map,
            history=[],
        )

        # Compute initial coverage
        self._update_coverage(state)

        logger.debug(
            "initial_state_created",
            n_agents=self.config.n_agents,
            n_obstacles=len(obstacles),
        )

        return state

    def get_legal_actions(self, state: SwarmState) -> list[int]:
        """Get legal actions for current agent.

        All 7 velocity changes are always legal.

        Args:
            state: Current swarm state.

        Returns:
            List of legal action indices.

        """
        if state.is_terminal:
            return []
        return list(range(N_ACTIONS_PER_AGENT))

    def apply_action(self, state: SwarmState, action: int) -> SwarmState:
        """Apply velocity change action to current agent.

        Updates the current agent's velocity, integrates positions with
        Euler step, clamps to domain, advances round-robin, and increments
        the game step when all agents have acted.

        Args:
            state: Current swarm state.
            action: Action index (0-6) for the current agent.

        Returns:
            New SwarmState after action.

        Raises:
            ValueError: If action is out of range or state is terminal.

        """
        if action < 0 or action >= N_ACTIONS_PER_AGENT:
            raise ValueError(f"Invalid action {action}, must be in [0, {N_ACTIONS_PER_AGENT})")
        if state.is_terminal:
            raise ValueError("Cannot apply action to terminal state")

        new_state = state.clone()
        agent_idx = state.current_agent

        # Set velocity for current agent
        new_state.velocities[agent_idx] = self._velocity_deltas[action]

        # Integrate position (Euler step)
        dt = self.config.dt
        new_pos = new_state.positions[agent_idx] + new_state.velocities[agent_idx] * dt

        # Clamp to domain bounds
        domain = np.array(self.config.domain_size, dtype=np.float64)
        new_pos = np.clip(new_pos, 0.0, domain)
        new_state.positions[agent_idx] = new_pos

        new_state.history.append(action)

        # Advance round-robin
        next_agent = (agent_idx + 1) % self.config.n_agents
        new_state.current_agent = next_agent

        # Increment step when full round completed
        if next_agent == 0:
            new_state.step += 1

        # Update coverage map
        self._update_coverage(new_state)

        # Compute reward
        new_state.reward = self.compute_reward(new_state)
        new_state.cumulative_reward += new_state.reward

        # Check terminal
        if new_state.step >= self.config.max_steps:
            new_state.is_terminal = True

        logger.debug(
            "action_applied",
            agent=agent_idx,
            action=ACTION_NAMES[action],
            step=new_state.step,
            current_agent=new_state.current_agent,
        )

        return new_state

    def is_terminal(self, state: SwarmState) -> bool:
        """Check if the game has ended.

        Args:
            state: Current swarm state.

        Returns:
            True if max steps reached.

        """
        return state.is_terminal or state.step >= self.config.max_steps

    def compute_reward(self, state: SwarmState) -> float:
        """Compute reward for current state.

        Combines coverage improvement, collision penalties,
        communication connectivity, and energy cost.

        Args:
            state: Current swarm state.

        Returns:
            Scalar reward value.

        """
        cfg = self.config

        # Coverage fraction
        coverage = self.compute_coverage(state)
        coverage_reward = cfg.coverage_weight * coverage

        # Collision penalty (agent-agent)
        collision_count = self._count_collisions(state)
        collision_reward = -cfg.collision_penalty * collision_count

        # Obstacle collision penalty
        obstacle_count = self._count_obstacle_collisions(state)
        obstacle_reward = -cfg.obstacle_penalty * obstacle_count

        # Communication graph connectivity
        comm_graph = self.compute_communication_graph(state)
        connectivity = self._graph_connectivity(comm_graph)
        comm_reward = cfg.communication_weight * connectivity

        # Energy cost (sum of velocity magnitudes)
        energy = float(np.sum(np.linalg.norm(state.velocities, axis=1)))
        energy_penalty = -cfg.energy_weight * energy

        total = coverage_reward + collision_reward + obstacle_reward + comm_reward + energy_penalty

        logger.debug(
            "reward_computed",
            coverage=coverage,
            collisions=collision_count,
            obstacle_hits=obstacle_count,
            connectivity=connectivity,
            energy=energy,
            total=total,
        )

        return float(total)

    # --- PDE-connected methods ---

    def compute_potential_field(self, state: SwarmState) -> NDArray[np.float64]:
        """Compute repulsive potential field from obstacles.

        Uses inverse-distance potential: phi(x) = sum_i strength / |x - c_i|^decay
        This is related to the Green's function of the Laplace equation.

        Args:
            state: Current swarm state.

        Returns:
            Potential values at each agent position, shape (n_agents,).

        """
        n_agents = len(state.positions)
        potentials = np.zeros(n_agents, dtype=np.float64)

        if len(state.obstacles) == 0:
            return potentials

        strength = self.config.potential_field_strength
        decay = self.config.potential_field_decay

        obs_centers = state.obstacles[:, :3]
        obs_radii = state.obstacles[:, 3]

        for i in range(n_agents):
            diffs = state.positions[i] - obs_centers  # (n_obs, 3)
            dists = np.linalg.norm(diffs, axis=1)  # (n_obs,)

            # Avoid division by zero; use obstacle radius as minimum distance
            effective_dists = np.maximum(dists - obs_radii, 0.1)

            potentials[i] = float(np.sum(strength / (effective_dists**decay)))

        return potentials

    def compute_communication_graph(self, state: SwarmState) -> NDArray[np.bool_]:
        """Compute adjacency matrix of the communication graph.

        Two agents are connected if their distance <= communication_range.

        Args:
            state: Current swarm state.

        Returns:
            Boolean adjacency matrix, shape (n_agents, n_agents).

        """
        n = len(state.positions)
        comm_range = self.config.communication_range

        # Pairwise distances
        diffs = state.positions[:, np.newaxis, :] - state.positions[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)

        # Adjacency (exclude self)
        adj = (dists <= comm_range) & ~np.eye(n, dtype=bool)

        return adj

    def compute_coverage(self, state: SwarmState) -> float:
        """Compute fraction of domain covered by agent sensors.

        Projects agent positions onto a 2D XY grid and marks cells
        within sensor range as covered.

        Args:
            state: Current swarm state.

        Returns:
            Coverage fraction in [0, 1].

        """
        if state.coverage_map is None:
            return 0.0

        total_cells = state.coverage_map.size
        if total_cells == 0:
            return 0.0

        covered: float = float(np.sum(state.coverage_map > 0))
        return covered / float(total_cells)

    # --- Internal helpers ---

    def _update_coverage(self, state: SwarmState) -> None:
        """Update the coverage map based on current positions.

        Marks grid cells within sensor range of any agent as covered
        (accumulated over time). Modifies state.coverage_map in-place.

        Args:
            state: Swarm state (modified in-place).

        """
        if state.coverage_map is None:
            return

        res = self.config.coverage_grid_res
        domain_x, domain_y, _ = self.config.domain_size
        sensor_range = self.config.coverage_sensor_range

        # Grid cell centers
        cx = np.linspace(0, domain_x, res, endpoint=False) + domain_x / (2 * res)
        cy = np.linspace(0, domain_y, res, endpoint=False) + domain_y / (2 * res)
        gx, gy = np.meshgrid(cx, cy, indexing="ij")  # (res, res)

        for agent_pos in state.positions:
            dx = gx - agent_pos[0]
            dy = gy - agent_pos[1]
            dist_sq = dx**2 + dy**2
            within = dist_sq <= sensor_range**2
            state.coverage_map[within] = 1.0

    def _count_collisions(self, state: SwarmState) -> int:
        """Count pairwise agent-agent collisions.

        Args:
            state: Current swarm state.

        Returns:
            Number of collision pairs.

        """
        n = len(state.positions)
        if n < 2:
            return 0

        diffs = state.positions[:, np.newaxis, :] - state.positions[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)

        # Upper triangle only (avoid double counting)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        collisions = int(np.sum((dists < self.config.collision_radius) & mask))

        return collisions

    def _count_obstacle_collisions(self, state: SwarmState) -> int:
        """Count agent-obstacle collisions.

        Args:
            state: Current swarm state.

        Returns:
            Number of agents colliding with obstacles.

        """
        if len(state.obstacles) == 0:
            return 0

        obs_centers = state.obstacles[:, :3]
        obs_radii = state.obstacles[:, 3]
        count = 0

        for pos in state.positions:
            dists = np.linalg.norm(pos - obs_centers, axis=1)
            if np.any(dists < obs_radii + self.config.collision_radius):
                count += 1

        return count

    def _graph_connectivity(self, adjacency: NDArray[np.bool_]) -> float:
        """Compute normalized algebraic connectivity of communication graph.

        Uses the ratio of connected pairs to total possible pairs as a
        fast proxy for full Fiedler value computation.

        Args:
            adjacency: Boolean adjacency matrix (n, n).

        Returns:
            Connectivity score in [0, 1].

        """
        n = adjacency.shape[0]
        if n < 2:
            return 1.0

        # Fraction of connected pairs
        total_pairs = n * (n - 1) / 2
        connected_pairs = np.sum(np.triu(adjacency, k=1))

        return float(connected_pairs / total_pairs)

    def action_to_string(self, action: int) -> str:
        """Convert action index to human-readable string.

        Args:
            action: Action index.

        Returns:
            Action name string.

        """
        if 0 <= action < N_ACTIONS_PER_AGENT:
            return ACTION_NAMES[action]
        return f"invalid_action_{action}"

    def clone(self) -> SwarmPlanningGame:
        """Create a copy of the game.

        Returns:
            New SwarmPlanningGame instance with same config.

        """
        return SwarmPlanningGame(self.config)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SwarmPlanningGame(n_agents={self.config.n_agents}, "
            f"n_obstacles={self.config.n_obstacles})"
        )
