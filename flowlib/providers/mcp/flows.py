"""MCP tool execution flows using @flow decorator pattern.

This module provides MCP tool execution following the architectural principle
that all flows must be defined using @flow decorator only.
"""

import logging
from typing import Dict, Any, Optional
from pydantic import Field

from flowlib.core.models import StrictBaseModel
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.mcp.client.provider import MCPClientProvider
from flowlib.providers.mcp.base import MCPTool

logger = logging.getLogger(__name__)


class MCPToolExecutionInput(StrictBaseModel):
    """Input for MCP tool execution."""
    
    client_name: str = Field(description="Name of the MCP client to use")
    tool_name: str = Field(description="Name of the tool to execute")
    tool_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool")


class MCPToolExecutionOutput(StrictBaseModel):
    """Output from MCP tool execution."""
    
    success: bool = Field(description="Whether the tool execution succeeded")
    result: Optional[Any] = Field(None, description="Tool execution result if successful")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    client_name: str = Field(description="Name of the MCP client that was used")
    tool_name: str = Field(description="Name of the tool that was executed")


class MCPRegistry:
    """Global registry for MCP clients and tools."""
    
    def __init__(self):
        self._clients: Dict[str, MCPClientProvider] = {}
        self._tools: Dict[str, Dict[str, MCPTool]] = {}  # client_name -> {tool_name: tool}
    
    def register_client(self, name: str, client: MCPClientProvider) -> None:
        """Register an MCP client."""
        self._clients[name] = client
        self._tools[name] = {}
        logger.debug(f"Registered MCP client: {name}")
    
    def register_tool(self, client_name: str, tool_name: str, tool: MCPTool) -> None:
        """Register a tool for a specific client."""
        if client_name not in self._tools:
            self._tools[client_name] = {}
        self._tools[client_name][tool_name] = tool
        logger.debug(f"Registered MCP tool: {client_name}.{tool_name}")
    
    def get_client(self, name: str) -> Optional[MCPClientProvider]:
        """Get an MCP client by name."""
        return self._clients.get(name)
    
    def get_tool(self, client_name: str, tool_name: str) -> Optional[MCPTool]:
        """Get a tool from a specific client."""
        client_tools = self._tools.get(client_name, {})
        return client_tools.get(tool_name)
    
    def list_tools(self, client_name: str) -> Dict[str, MCPTool]:
        """List all tools for a specific client."""
        return self._tools.get(client_name, {}).copy()
    
    def unregister_client(self, name: str) -> None:
        """Unregister an MCP client and all its tools."""
        self._clients.pop(name, None)
        self._tools.pop(name, None)
        logger.debug(f"Unregistered MCP client: {name}")


# Global MCP registry instance
mcp_registry = MCPRegistry()


@flow(name="mcp-tool-executor", description="Execute MCP tools via registered clients")
class MCPToolExecutorFlow:
    """Flow for executing MCP tools using the @flow decorator pattern."""
    
    @pipeline(input_model=MCPToolExecutionInput, output_model=MCPToolExecutionOutput)
    async def run_pipeline(self, input_data: MCPToolExecutionInput) -> MCPToolExecutionOutput:
        """Execute an MCP tool through the registered client.
        
        Args:
            input_data: Tool execution request with client name, tool name, and parameters
            
        Returns:
            Tool execution result
        """
        try:
            # Get the client from registry
            client = mcp_registry.get_client(input_data.client_name)
            if not client:
                return MCPToolExecutionOutput(
                    success=False,
                    error=f"MCP client '{input_data.client_name}' not found",
                    client_name=input_data.client_name,
                    tool_name=input_data.tool_name
                )
            
            # Get the tool from registry
            tool = mcp_registry.get_tool(input_data.client_name, input_data.tool_name)
            if not tool:
                return MCPToolExecutionOutput(
                    success=False,
                    error=f"Tool '{input_data.tool_name}' not found for client '{input_data.client_name}'",
                    client_name=input_data.client_name,
                    tool_name=input_data.tool_name
                )
            
            # Execute the tool
            result = await client.call_tool(input_data.tool_name, input_data.tool_parameters)
            
            return MCPToolExecutionOutput(
                success=True,
                result=result,
                client_name=input_data.client_name,
                tool_name=input_data.tool_name
            )
            
        except Exception as e:
            logger.error(f"Error executing MCP tool '{input_data.tool_name}' on client '{input_data.client_name}': {e}")
            return MCPToolExecutionOutput(
                success=False,
                error=str(e),
                client_name=input_data.client_name,
                tool_name=input_data.tool_name
            )