"""
Base component for the agent system.

This module defines a simplified BaseComponent class that all agent components inherit from,
providing a lightweight interface aligned with flowlib patterns.
"""

import logging
from abc import ABC
from typing import Optional

class AgentComponent(ABC):
    """Base component for the agent system.
    
    Provides:
    1. Basic lifecycle management (initialize/shutdown)
    2. Parent-child relationships
    """
    
    def __init__(self, name: str = None):
        """Initialize component.
        
        Args:
            name: Component name
        """
        self._name = name or self.__class__.__name__.lower()
        self._parent = None
        self._initialized = False
        self._logger = logging.getLogger(f"{__name__}.{self._name}")
    
    @property
    def name(self) -> str:
        """Get component name."""
        return self._name
    
    @property
    def initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized
    
    @property
    def parent(self) -> Optional['BaseComponent']:
        """Get parent component."""
        return self._parent
    
    def set_parent(self, parent: 'BaseComponent') -> None:
        """Set parent component.
        
        Args:
            parent: Parent component
        """
        self._parent = parent
        self._logger = logging.getLogger(f"{__name__}.{parent.name}.{self._name}")
    
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