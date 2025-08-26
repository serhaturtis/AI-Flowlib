"""
Agent flow execution component.

This module handles flow registration, execution, and management
operations that were previously in BaseAgent.
"""

import logging
from typing import Any, Dict, List, Optional

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import NotInitializedError, ExecutionError
from flowlib.agent.models.config import AgentConfig
from flowlib.flows.base.base import Flow
from flowlib.flows.models.metadata import FlowMetadata
from flowlib.flows.registry.registry import flow_registry, FlowRegistry
from flowlib.flows.models.results import FlowResult
from flowlib.agent.components.discovery.flow_discovery import FlowDiscovery

logger = logging.getLogger(__name__)


class AgentFlowRunner(AgentComponent):
    """Handles agent flow operations.
    
    This component is responsible for:
    - Flow registration and management
    - Flow execution coordination
    - Flow discovery and metadata
    """
    
    def __init__(self, name: str = "flow_runner"):
        """Initialize the flow runner.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self._flows: Dict[str, Flow] = {}
        self._flow_discovery: Optional[FlowDiscovery] = None
    
    async def _initialize_impl(self) -> None:
        """Initialize the flow runner."""
        # Initialize flow discovery
        self._flow_discovery = FlowDiscovery()
        await self._flow_discovery.initialize()
        
        logger.info("Flow runner initialized")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the flow runner."""
        if self._flow_discovery:
            await self._flow_discovery.shutdown()
        logger.info("Flow runner shutdown")
    
    async def discover_flows(self) -> None:
        """Discover and register available flows."""
        if not self._flow_discovery:
            logger.warning("Flow discovery not initialized")
            return
        
        try:
            # Use the synchronous discover_agent_flows method
            discovered_flows = self._flow_discovery.discover_agent_flows()
            logger.info(f"Discovered {len(discovered_flows)} flows")
            
            for flow in discovered_flows:
                self.register_flow(flow)
        except Exception as e:
            logger.error(f"Error discovering flows: {e}")
    
    def register_flow(self, flow: Any) -> None:
        """Register a flow with the agent.
        
        Args:
            flow: Flow instance to register
        """
        if not hasattr(flow, 'name'):
            logger.warning(f"Flow {flow} does not have a name attribute, skipping registration")
            return
        
        flow_name = flow.name
        self._flows[flow_name] = flow
        logger.debug(f"Registered flow: {flow_name}")
        
        # Also register with global registry if it exists
        if flow_registry:
            try:
                flow_registry.register(flow)
            except Exception as e:
                logger.debug(f"Could not register flow {flow_name} with global registry: {e}")
    
    async def register_flow_async(self, flow: Flow) -> None:
        """Register a flow asynchronously.
        
        Args:
            flow: Flow instance to register
        """
        self.register_flow(flow)
        
        # Initialize the flow if it's a component
        if hasattr(flow, 'initialize'):
            try:
                await flow.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize flow {flow.name}: {e}")
    
    def unregister_flow(self, flow_name: str) -> None:
        """Unregister a flow from the agent.
        
        Args:
            flow_name: Name of the flow to unregister
        """
        if flow_name in self._flows:
            del self._flows[flow_name]
            logger.debug(f"Unregistered flow: {flow_name}")
        
        # Also unregister from global registry if it exists
        if flow_registry:
            try:
                flow_registry.unregister(flow_name)
            except Exception as e:
                logger.debug(f"Could not unregister flow {flow_name} from global registry: {e}")
    
    def get_flow_registry(self) -> FlowRegistry:
        """Get the flow registry.
        
        Returns:
            FlowRegistry instance
        """
        return flow_registry
    
    def get_flow_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all registered flows.
        
        Returns:
            List of flow descriptions for planning
        """
        descriptions = []
        
        for flow_name, flow in self._flows.items():
            try:
                # Try to get description from the flow
                if hasattr(flow, 'get_description'):
                    description = flow.get_description()
                elif hasattr(flow, 'description'):
                    description = flow.description
                else:
                    description = f"Flow: {flow_name}"
                
                descriptions.append({
                    "name": flow_name,
                    "description": description,
                    "type": type(flow).__name__
                })
            except Exception as e:
                logger.warning(f"Error getting description for flow {flow_name}: {e}")
                descriptions.append({
                    "name": flow_name,
                    "description": f"Flow: {flow_name} (description unavailable)",
                    "type": type(flow).__name__
                })
        
        return descriptions
    
    async def execute_flow(self,
                          flow_name: str,
                          inputs: Any,
                          **kwargs) -> FlowResult:
        """Execute a flow with given inputs.
        
        Args:
            flow_name: Name of the flow to execute
            inputs: Inputs for the flow
            **kwargs: Additional execution arguments
            
        Returns:
            FlowResult from the execution
            
        Raises:
            NotInitializedError: If the flow runner is not initialized
            ExecutionError: If the flow execution fails
        """
        if not self._initialized:
            raise NotInitializedError(
                component_name=self._name,
                operation="execute_flow"
            )
        
        # Check if flow is registered locally
        flow = self._flows.get(flow_name)
        
        if not flow:
            # Try to get from global registry
            if flow_registry:
                flow = flow_registry.get_flow(flow_name)
            
            if not flow:
                raise ExecutionError(f"Flow '{flow_name}' not found")
        
        try:
            # Flows expect Context objects, not direct inputs
            from flowlib.core.context.context import Context
            
            # Wrap inputs in Context if they aren't already
            if not isinstance(inputs, Context):
                context = Context(data=inputs)
            else:
                context = inputs
            
            # Execute the flow
            if hasattr(flow, 'execute'):
                result = await flow.execute(context)
            elif hasattr(flow, 'run_pipeline'):
                result = await flow.run_pipeline(context)
            else:
                raise ExecutionError(f"Flow '{flow_name}' does not have execute or run_pipeline method")
            
            # Log execution
            activity_stream = self.get_component("activity_stream")
            if activity_stream:
                activity_stream.execution(
                    f"Flow '{flow_name}' executed successfully",
                    inputs=str(inputs)[:100],
                    result=str(result)[:100] if result else "None"
                )
            
            return result
        except Exception as e:
            logger.error(f"Error executing flow {flow_name}: {e}")
            # Let the real exception bubble up
            raise
    
    async def list_available_flows(self) -> List[Dict[str, Any]]:
        """List all available flows.
        
        Returns:
            List of available flow information
        """
        flows = []
        
        # Add local flows
        for flow_name, flow in self._flows.items():
            flows.append({
                "name": flow_name,
                "type": type(flow).__name__,
                "source": "local",
                "description": getattr(flow, 'description', f"Flow: {flow_name}")
            })
        
        # Add flows from global registry
        if flow_registry:
            try:
                global_flows = flow_registry.get_agent_selectable_flows()
                for flow_name, flow in global_flows.items():
                    if flow_name not in self._flows:  # Don't duplicate
                        flows.append({
                            "name": flow_name,
                            "type": type(flow).__name__,
                            "source": "global_registry",
                            "description": getattr(flow, 'description', f"Flow: {flow_name}")
                        })
            except Exception as e:
                logger.warning(f"Error getting flows from global registry: {e}")
        
        return flows
    
    @property
    def flows(self) -> Dict[str, Flow]:
        """Get registered flows.
        
        Returns:
            Dictionary of registered flows
        """
        return self._flows.copy()
    
    async def validate_required_flows(self) -> None:
        """Validate that required flows are available."""
        required_flows = ["conversation", "shell-command", "message-classifier-flow"]
        missing_flows = []
        
        for flow_name in required_flows:
            if flow_name not in self._flows:
                # Check global registry
                if not flow_registry or not flow_registry.get_flow(flow_name):
                    missing_flows.append(flow_name)
        
        if missing_flows:
            logger.warning(f"Missing required flows: {missing_flows}")
        else:
            logger.info("All required flows are available")