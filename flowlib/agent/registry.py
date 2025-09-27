"""Agent registry for tracking and accessing agent classes."""

import logging
from typing import Type, Optional, List, Dict, Union
from pydantic import BaseModel, Field, ConfigDict

from flowlib.core.registry.registry import BaseRegistry

logger = logging.getLogger(__name__)

class AgentInfo(BaseModel):
    """Information about a registered agent."""
    model_config = ConfigDict(extra="forbid")
    
    agent_class: Type = Field(..., description="The agent class")
    metadata: Dict[str, Union[str, int, bool]] = Field(default_factory=dict, description="Agent metadata")


class AgentRegistry(BaseRegistry[Type]):
    """Registry for agent classes."""

    def __init__(self) -> None:
        """Initialize the agent registry."""
        self._agents: Dict[str, AgentInfo] = {}

    def register(self, name: str, obj: Type, **metadata: Union[str, int, bool]) -> None:
        """Register an agent class.

        Args:
            name: The unique name for the agent.
            obj: The agent class (or wrapper class) to register.
            **metadata: Metadata associated with the agent.

        Raises:
            ValueError: If an agent with the same name is already registered.
        """
        if name in self._agents:
            # Allow re-registration for development/hot-reloading, but log a warning
            logger.warning(f"Agent '{name}' is already registered. Overwriting registration.")
            # To disallow overwriting, uncomment the following line:
            # raise ValueError(f"Agent '{name}' is already registered.")

        self._agents[name] = AgentInfo(
            agent_class=obj,
            metadata=metadata
        )
        class_name = getattr(obj, '__name__', str(obj))
        logger.debug(f"Registered agent: {name} (Class: {class_name})")
    
    # BaseRegistry interface implementation
    def get(self, name: str, expected_type: Optional[Type] = None) -> Type:
        """Get an agent class by name with optional type checking (BaseRegistry interface).
        
        Args:
            name: Name of the agent to retrieve
            expected_type: Optional type for type checking
            
        Returns:
            The registered agent class
            
        Raises:
            KeyError: If the agent doesn't exist
            TypeError: If the agent doesn't match the expected type
        """
        agent_class = self.get_agent_class(name)
        if agent_class is None:
            raise KeyError(f"Agent '{name}' not found in registry")
        
        if expected_type is not None and not issubclass(agent_class, expected_type):
            raise TypeError(f"Agent '{name}' is not a subclass of expected type {expected_type}")
        
        return agent_class
    
    def contains(self, name: str) -> bool:
        """Check if an agent exists in the registry (BaseRegistry interface).
        
        Args:
            name: Name to check
            
        Returns:
            True if the agent exists, False otherwise
        """
        return name in self._agents
    
    def list(self, filter_criteria: Optional[Dict[str, Union[str, int, bool]]] = None) -> List[str]:
        """List registered agents matching criteria (BaseRegistry interface).
        
        Args:
            filter_criteria: Optional criteria to filter results
            
        Returns:
            List of agent names matching the criteria
        """
        agents = self.list_agents()
        
        if filter_criteria is None:
            return agents
        
        # Apply filtering logic
        filtered_agents = []
        for agent_name in agents:
            if self._matches_criteria(agent_name, filter_criteria):
                filtered_agents.append(agent_name)
        
        return filtered_agents
    
    def _matches_criteria(self, agent_name: str, criteria: Dict[str, Union[str, int, bool]]) -> bool:
        """Check if an agent matches the given criteria.
        
        Args:
            agent_name: Name of the agent to check
            criteria: Criteria to match against
            
        Returns:
            True if the agent matches all criteria
        """
        agent_metadata = self.get_agent_metadata(agent_name) or {}
        
        for key, value in criteria.items():
            if key in agent_metadata:
                if agent_metadata[key] != value:
                    return False
            else:
                return False
        
        return True

    def get_agent_class(self, name: str) -> Optional[Type]:
        """Get the registered agent class by name.

        Args:
            name: The name of the agent.

        Returns:
            The agent class, or None if not found.
        """
        agent_info = self._agents.get(name)
        return agent_info.agent_class if agent_info else None

    def get_agent_metadata(self, name: str) -> Optional[Dict[str, Union[str, int, bool]]]:
        """Get the metadata for a registered agent.

        Args:
            name: The name of the agent.

        Returns:
            The metadata dictionary, or None if not found.
        """
        agent_info = self._agents.get(name)
        return agent_info.metadata if agent_info else None
        
    def get_agent_info(self, name: str) -> Optional[AgentInfo]:
        """Get all registered information for an agent.

        Args:
            name: The name of the agent.

        Returns:
            An AgentInfo instance, or None if not found.
        """
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """List the names of all registered agents.

        Returns:
            A list of agent names.
        """
        return sorted(list(self._agents.keys()))

    def clear(self) -> None:
        """Clear all registered agents."""
        self._agents = {}
        logger.debug("Agent registry cleared.")
    
    def remove(self, name: str) -> bool:
        """Remove a specific agent registration from the registry.
        
        Args:
            name: Name of the agent to remove
            
        Returns:
            True if the agent was found and removed, False if not found
        """
        if name in self._agents:
            del self._agents[name]
            logger.debug(f"Removed agent '{name}' from registry")
            return True
        
        return False
    
    def update(self, name: str, obj: Type, **metadata: Union[str, int, bool]) -> bool:
        """Update or replace an existing agent registration.
        
        Args:
            name: Name of the agent to update
            obj: New agent class to register
            **metadata: Additional metadata about the agent
            
        Returns:
            True if an existing agent was updated, False if this was a new registration
        """
        existing_found = self.contains(name)
        
        if existing_found:
            # Remove existing
            self.remove(name)
            
            # Re-register
            self.register(name, obj, **metadata)
            logger.debug(f"Updated existing agent '{name}' in registry")
            return True
        else:
            # New registration
            self.register(name, obj, **metadata)
            logger.debug(f"Registered new agent '{name}' in registry")
            return False

# Global instance of the agent registry
agent_registry = AgentRegistry() 