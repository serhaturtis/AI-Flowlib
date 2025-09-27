"""
Base component for the agent system.

This module defines a simplified BaseComponent class that all agent components inherit from,
providing a lightweight interface aligned with flowlib patterns.
"""

import logging
from abc import ABC
from typing import Optional
from .component_registry import ComponentRegistry


class AgentComponent(ABC):
    """Base component for the agent system.
    
    Provides:
    1. Basic lifecycle management (initialize/shutdown)
    2. Registry access for inter-component communication
    """
    
    def __init__(self, name: str):
        """Initialize component.
        
        Args:
            name: Component name
        """
        self._name = name or self.__class__.__name__.lower()
        self._initialized = False
        self._registry: Optional[ComponentRegistry] = None
        self._logger = logging.getLogger(f"{__name__}.{self._name}")
    
    def set_registry(self, registry: ComponentRegistry) -> None:
        """Set the component registry for inter-component access.
        
        Args:
            registry: The agent's component registry
        """
        self._registry = registry
        # Update logger to include agent context
        if registry:
            self._logger = logging.getLogger(
                f"{__name__}.{registry._agent_name}.{self._name}"
            )
    
    def get_component(self, name: str) -> Optional[object]:
        """Get another component from the registry.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None
        """
        if not self._registry:
            self._logger.warning(
                f"Component {self._name} has no registry access"
            )
            return None
        return self._registry.get(name)
    
    @property
    def name(self) -> str:
        """Get component name."""
        return self._name
    
    @property
    def initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized
    
    def _check_initialized(self) -> None:
        """Check if the component is initialized.
        
        Raises:
            NotInitializedError: If the component is not initialized
        """
        if not self._initialized:
            from .errors import NotInitializedError
            raise NotInitializedError(
                component_name=self._name,
                operation="component operation"
            )
    
    async def initialize(self) -> None:
        """Initialize the component."""
        if self._initialized:
            return
            
        await self._initialize_impl()
        self._initialized = True
        self._logger.debug(f"Component '{self._name}' initialized")
    
    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization.
        
        Override this method in subclasses.
        """
        pass
    
    async def shutdown(self) -> None:
        """Shutdown the component."""
        if not self._initialized:
            return
            
        await self._shutdown_impl()
        self._initialized = False
        self._logger.debug(f"Component '{self._name}' shut down")
    
    async def _shutdown_impl(self) -> None:
        """Implementation-specific shutdown.
        
        Override this method in subclasses.
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the component."""
        status = "initialized" if self._initialized else "not initialized"
        return f"{self.__class__.__name__}(name='{self._name}', status='{status}')" 