"""
Core agent component interfaces.

This module defines the base protocol that all agent components must implement.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ComponentInterface(Protocol):
    """Base protocol for all agent components.
    
    Defines the lifecycle methods that all components must implement.
    """
    
    async def initialize(self) -> None:
        """Initialize the component.
        
        This method should be called before the component is used.
        It should set up any resources needed by the component.
        """
        ...
    
    async def shutdown(self) -> None:
        """Shutdown the component and release resources.
        
        This method should be called when the component is no longer needed.
        It should clean up any resources used by the component.
        """
        ...
    
    @property
    def initialized(self) -> bool:
        """Return whether the component is initialized.
        
        Returns:
            True if the component is initialized and ready to use
        """
        ...
        
    @property
    def name(self) -> str:
        """Return the component name.
        
        Returns:
            Name of the component
        """
        ... 