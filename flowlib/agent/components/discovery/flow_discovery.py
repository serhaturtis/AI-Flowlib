"""
Flow discovery for agent systems.

This module provides functionality for discovering agent-compatible flows
from the registered stages and converting them to a format compatible with
our planning system.
"""

import logging
from typing import Dict, Any, Optional, List, Type

from flowlib.flows.base.base import Flow
from flowlib.flows.registry.registry import flow_registry
from flowlib.flows.models.metadata import FlowMetadata
from flowlib.agent.core.errors import FlowDiscoveryError, DiscoveryError
from flowlib.agent.core.base import AgentComponent
from flowlib.agent.components.discovery.interfaces import FlowDiscoveryInterface

logger = logging.getLogger(__name__)


class FlowDiscovery(AgentComponent, FlowDiscoveryInterface):
    """Service for discovering agent-compatible flows.
    
    This class helps discover flows that have been decorated with the
    @agent_flow decorator and provides access to them.
    """
    
    def __init__(self, name: str = "flow_discovery"):
        """Initialize flow discovery.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self._flows: Dict[str, Flow] = {}
    
    async def _initialize_impl(self) -> None:
        """Initialize the flow discovery system.
        
        This automatically refreshes flows on initialization.
        """
        await self.refresh_flows()
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the flow discovery system."""
        pass
    
    async def refresh_flows(self) -> None:
        """Refresh the discovered flows.
        
        This scans the stage registry for available flows.
        
        Raises:
            FlowDiscoveryError: If discovery fails
        """
        self._logger.info("Refreshing flows from registry")
        
        try:
            # Clear existing flow references
            self._flows.clear()
            
            # Find agent-compatible flow instances directly
            flow_instances = self.discover_agent_flow_instances()
            
            # Register each discovered flow instance
            for flow_instance in flow_instances:
                try:
                    # Register the existing instance
                    self.register_flow(flow_instance)
                except Exception as e:
                    self._logger.warning(f"Failed to register flow {getattr(flow_instance, 'name', 'unknown')}: {e}")
            
            self._logger.info(f"Refreshed flows: {len(self._flows)} agent flows available")
            
        except Exception as e:
            raise FlowDiscoveryError(f"Failed to refresh flows: {str(e)}") from e
    
    def register_flow(self, flow: Any) -> None:
        """Register a flow with the discovery system.
        
        Args:
            flow: Flow to register
        """
        if not hasattr(flow, "name"):
            self._logger.warning(f"Cannot register flow without name: {flow}")
            return
            
        flow_name = flow.name
        
        # Store a reference to the flow
        self._flows[flow_name] = flow
        
        # Register with flow_registry
        if flow_registry:
            try:
                flow_registry.register_flow(flow_name, flow)
                self._logger.debug(f"Registered flow: {flow_name}")
            except Exception as e:
                self._logger.warning(f"Failed to register flow with flow_registry: {e}")
    
    def get_flow(self, name: str) -> Optional[Any]:
        """Get a flow by name.
        
        Args:
            name: Flow name
            
        Returns:
            Flow object or None if not found
        """
        return self._flows.get(name)
    
    def get_flow_registry(self):
        """Get the flow registry.
        
        Returns:
            Stage registry containing all flows
        """
        return flow_registry
    
    def get_flow_metadata(self, name: str) -> Optional[FlowMetadata]:
        """Get metadata for a flow by name.
        
        Args:
            name: Flow name
            
        Returns:
            Flow metadata or None if not found
        """
        if not flow_registry:
            return None
            
        try:
            return flow_registry.get_flow_metadata(name)
        except Exception as e:
            self._logger.warning(f"Failed to get metadata for flow {name}: {e}")
            return None
    
    def discover_agent_flows(self) -> List[Any]:
        """Discover flows that are compatible with agents.
        
        This method finds flows that have been decorated with the @agent_flow decorator.
        
        Returns:
            List of flow instances compatible with agents
            
        Raises:
            DiscoveryError: If stage registry is not available or discovery fails
        """
        self._logger.debug("Discovering agent-compatible flows")
        
        # Check if stage registry is available
        if not flow_registry:
            raise DiscoveryError(
                "Stage registry not available for flow discovery",
                operation="discover_agent_flows"
            )
        
        # Get all flow instances from the stage registry
        flow_instances = flow_registry.get_flow_instances()
        
        # Filter to only agent-compatible flows (non-infrastructure flows)
        agent_flows = []
        for flow_name, flow_instance in flow_instances.items():
            # Check if flow is not an infrastructure flow (agent-selectable)
            if hasattr(flow_instance, "__flow_metadata__"):
                is_infrastructure = flow_instance.__flow_metadata__["is_infrastructure"] if "is_infrastructure" in flow_instance.__flow_metadata__ else True
                if not is_infrastructure and flow_instance not in agent_flows:
                    agent_flows.append(flow_instance)
                self._logger.debug(f"Found agent flow: {flow_name}")
            elif hasattr(flow_instance, "is_infrastructure") and not flow_instance.is_infrastructure:
                if flow_instance not in agent_flows:
                    agent_flows.append(flow_instance)
                self._logger.debug(f"Found agent flow (via is_infrastructure attr): {flow_name}")
        
        self._logger.info(f"Discovered {len(agent_flows)} agent-compatible flows")
        return agent_flows
    
    def discover_agent_flow_instances(self) -> List[Any]:
        """Discover flow instances that are compatible with agents.
        
        This method finds flow instances that have been marked as non-infrastructure.
        
        Returns:
            List of flow instances compatible with agents
            
        Raises:
            DiscoveryError: If stage registry is not available or discovery fails
        """
        self._logger.debug("Discovering agent-compatible flow instances")
        
        # Check if stage registry is available
        if not flow_registry:
            raise DiscoveryError(
                "Stage registry not available for flow discovery",
                operation="discover_agent_flow_instances"
            )
        
        # Get all flow instances from the stage registry
        flow_instances = flow_registry.get_flow_instances()
        
        # Filter to only agent-compatible flows (non-infrastructure flows)
        agent_flow_instances = []
        for flow_name, flow_instance in flow_instances.items():
            # Check if flow is not an infrastructure flow (agent-selectable)
            if hasattr(flow_instance, "__flow_metadata__"):
                is_infrastructure = flow_instance.__flow_metadata__["is_infrastructure"] if "is_infrastructure" in flow_instance.__flow_metadata__ else True
                if not is_infrastructure:
                    agent_flow_instances.append(flow_instance)
                self._logger.debug(f"Found agent flow instance: {flow_name}")
            elif hasattr(flow_instance, "is_infrastructure") and not flow_instance.is_infrastructure:
                agent_flow_instances.append(flow_instance)
                self._logger.debug(f"Found agent flow instance (via is_infrastructure attr): {flow_name}")
        
        self._logger.info(f"Discovered {len(agent_flow_instances)} agent-compatible flow instances")
        return agent_flow_instances 