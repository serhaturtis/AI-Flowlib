"""Component registry for agent-internal component management."""

from typing import Any, TypeVar, cast

T = TypeVar("T")


class ComponentRegistry:
    """Registry for agent components - instance-specific, not global.

    Each BaseAgent instance has its own registry to avoid conflicts
    between multiple running agents.
    """

    def __init__(self, agent_name: str = "agent"):
        """Initialize registry for a specific agent instance.

        Args:
            agent_name: Name of the owning agent for logging
        """
        self._agent_name = agent_name
        self._components: dict[str, object] = {}
        self._type_map: dict[type[Any], str] = {}

    def register(
        self, name: str, component: object, component_type: type[Any] | None = None
    ) -> None:
        """Register a component in this agent's registry.

        Args:
            name: Unique name for the component
            component: The component instance
            component_type: Optional type for typed access
        """
        self._components[name] = component
        if component_type:
            self._type_map[component_type] = name

    def get(self, name: str) -> object | None:
        """Get component by name."""
        return self._components.get(name)

    def get_typed(self, component_type: type[T]) -> T | None:
        """Get component by type with proper typing."""
        name = self._type_map.get(component_type)
        if name:
            component = self._components.get(name)
            return cast(T | None, component)
        return None

    def has(self, name: str) -> bool:
        """Check if component exists."""
        return name in self._components

    def list_components(self) -> dict[str, str]:
        """List all registered components."""
        return {name: type(comp).__name__ for name, comp in self._components.items()}


# Global component registry instance
component_registry = ComponentRegistry("default")
