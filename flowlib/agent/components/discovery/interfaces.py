"""
Discovery interface definitions.

This module defines the protocols and interfaces for discovery operations.
"""

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class FlowDiscoveryInterface(Protocol):
    """Interface for flow discovery.
    
    Defines the methods for discovering and registering flows.
    """
    
    async def refresh_flows(self) -> Dict[str, Any]:
        """Refresh the flow cache.
        
        Returns:
            Dictionary of flow names to flow objects
        """
        ...
    
    def register_flow(self, flow: Any) -> None:
        """Register a flow with the discovery system.
        
        Args:
            flow: Flow to register
        """
        ...
    
    def get_flow(self, name: str) -> Optional[Any]:
        """Get a flow by name.
        
        Args:
            name: Flow name
            
        Returns:
            Flow object or None if not found
        """
        ... 