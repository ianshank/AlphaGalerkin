"""Agent registration and discovery.

Uses the template registry pattern for thread-safe, decorator-based
agent type registration.

Example:
    from src.agents.registry import AgentRegistry, register_agent, get_agent

    @register_agent("my_agent")
    class MyAgent(BaseAgent):
        ...

    agent_cls = get_agent("my_agent")
    print(list_agents())

"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.templates.registry import create_registry

AgentRegistry, register_agent = create_registry("Agent", BaseAgent)  # type: ignore[type-abstract]


def get_agent(name: str) -> type[BaseAgent]:
    """Get an agent class by name.

    Args:
        name: Registered agent name.

    Returns:
        The agent class.

    Raises:
        KeyError: If the name is not registered.

    """
    return AgentRegistry().get_or_raise(name)


def list_agents() -> list[str]:
    """List all registered agent names.

    Returns:
        Sorted list of registered agent names.

    """
    return AgentRegistry().list_items()


def _register_builtin_agents() -> None:
    """Register built-in agent types.

    Called on module import to ensure core agents are always available.
    Deferred imports avoid circular dependencies.
    """
    registry = AgentRegistry()

    if not registry.is_registered("solver"):
        from src.agents.solver import SolverAgent

        register_agent("solver")(SolverAgent)

    if not registry.is_registered("decomposition"):
        from src.agents.decomposition import DecompositionAgent

        register_agent("decomposition")(DecompositionAgent)

    if not registry.is_registered("coupling"):
        from src.agents.coupling import CouplingAgent

        register_agent("coupling")(CouplingAgent)

    if not registry.is_registered("meta"):
        from src.agents.meta import MetaAgent

        register_agent("meta")(MetaAgent)
