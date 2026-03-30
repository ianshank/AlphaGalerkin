"""Agent-physics integration for multi-physics PDE solving.

This module provides an agent orchestration framework for solving
coupled PDE systems using MCTS-guided Galerkin approximation:

- Agent abstractions with lifecycle management
- Solver, decomposition, and coupling agents
- Thread-safe inter-agent communication
- Adaptive collocation point allocation
- CLI for running multi-physics solves

Key components:
    BaseAgent: Abstract base class for all agent types
    SolverAgent: Wraps PDEGame + MCTS for single-PDE solving
    DecompositionAgent: Splits coupled systems into subproblems
    CouplingAgent: Enforces interface conditions between subdomains
    MetaAgent: Orchestrates the full pipeline
    AgentOrchestrator: BaseExecutable entry point
"""

from src.agents.base import AgentState, BaseAgent
from src.agents.collocation import (
    CollocationAllocator,
    CollocationRegistry,
    create_collocation_allocator,
)
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
from src.agents.coupling import CouplingAgent
from src.agents.decomposition import DecompositionAgent, SubproblemSpec
from src.agents.message import AgentMessage, MessageBus
from src.agents.meta import MetaAgent
from src.agents.orchestrator import AgentOrchestrator
from src.agents.registry import AgentRegistry, get_agent, list_agents, register_agent
from src.agents.solver import SolverAgent

__all__ = [
    # Base
    "AgentState",
    "BaseAgent",
    # Configs
    "AgentConfig",
    "AgentType",
    "CollocationConfig",
    "CollocationStrategy",
    "CouplingConfig",
    "CouplingPairConfig",
    "CouplingType",
    "DecompositionConfig",
    "DecompositionStrategy",
    "MessageBusConfig",
    "MessageType",
    "MultiPhysicsConfig",
    "OrchestratorConfig",
    "SolverAgentConfig",
    # Agents
    "SolverAgent",
    "DecompositionAgent",
    "CouplingAgent",
    "MetaAgent",
    # Communication
    "AgentMessage",
    "MessageBus",
    # Collocation
    "CollocationAllocator",
    "CollocationRegistry",
    "create_collocation_allocator",
    # Decomposition
    "SubproblemSpec",
    # Orchestration
    "AgentOrchestrator",
    # Registry
    "AgentRegistry",
    "register_agent",
    "get_agent",
    "list_agents",
]
