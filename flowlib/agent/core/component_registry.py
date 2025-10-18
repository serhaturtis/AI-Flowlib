"""Component registry for agent-internal component management."""

from typing import Any, Dict, Optional, Type, TypeVar, cast

T = TypeVar('T')


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
        self._components: Dict[str, object] = {}
        self._type_map: Dict[Type[Any], str] = {}

    def register(self,
                 name: str,
                 component: object,
                 component_type: Optional[Type[Any]] = None) -> None:
        """Register a component in this agent's registry.
        
        Args:
            name: Unique name for the component
            component: The component instance
            component_type: Optional type for typed access
        """
        self._components[name] = component
        if component_type:
            self._type_map[component_type] = name

    def get(self, name: str) -> Optional[object]:
        """Get component by name."""
        return self._components.get(name)

    def get_typed(self, component_type: Type[T]) -> Optional[T]:
        """Get component by type with proper typing."""
        name = self._type_map.get(component_type)
        if name:
            component = self._components.get(name)
            return cast(Optional[T], component)
        return None

    def has(self, name: str) -> bool:
        """Check if component exists."""
        return name in self._components

    def list_components(self) -> Dict[str, str]:
        """List all registered components."""
        return {name: type(comp).__name__
                for name, comp in self._components.items()}


# Global component registry instance for backward compatibility
component_registry = ComponentRegistry("default")
