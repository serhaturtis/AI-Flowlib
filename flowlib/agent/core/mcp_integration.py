"""MCP integration for agents."""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from flowlib.agent.core.agent import AgentCore
from flowlib.providers.mcp.client.provider import MCPClientProvider, MCPClientSettings
from flowlib.providers.mcp.server.provider import MCPServerProvider, MCPServerSettings
from flowlib.providers.mcp.base import MCPTool, MCPResource, MCPTransport
from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.flows.base.base import Flow
from flowlib.flows.models.results import FlowResult
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class MCPClientConfig:
    """Configuration for an MCP client connection."""
    name: str
    server_uri: str
    transport: MCPTransport = MCPTransport.STDIO
    server_command: Optional[str] = None
    server_args: List[str] = None
    auth_token: Optional[str] = None
    auto_register_tools: bool = True


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    host: str = "localhost"
    port: int = 8080
    transport: MCPTransport = MCPTransport.SSE
    expose_flows: List[str] = None
    expose_memory: bool = True


class MCPToolFlow(Flow):
    """Wrapper flow for MCP tools."""
    
    def __init__(self, client_provider: MCPClientProvider, tool: MCPTool):
        self.client = client_provider
        self.tool = tool
        self.flow_name = f"mcp_{tool.name}"
        
    @pipeline
    async def execute_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the MCP tool."""
        try:
            result = await self.client.call_tool(self.tool.name, input_data)
            return {
                "success": True,
                "result": result,
                "tool_name": self.tool.name
            }
        except Exception as e:
            logger.error(f"Error executing MCP tool '{self.tool.name}': {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": self.tool.name
            }


class MCPEnabledAgent(AgentCore):
    """Agent with MCP integration capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mcp_clients: Dict[str, MCPClientProvider] = {}
        self._mcp_servers: Dict[str, MCPServerProvider] = {}
        self._mcp_tools: Dict[str, MCPToolFlow] = {}
        
    async def add_mcp_client(self, config: MCPClientConfig) -> MCPClientProvider:
        """Add an MCP client connection."""
        try:
            # Create client settings
            settings = MCPClientSettings(
                server_uri=config.server_uri,
                transport=config.transport,
                server_command=config.server_command,
                server_args=config.server_args or [],
                auth_token=config.auth_token
            )
            
            # Create and initialize client
            client = MCPClientProvider(name=config.name, settings=settings)
            await client.initialize()
            
            # Store client
            self._mcp_clients[config.name] = client
            
            # Auto-register tools as flows
            if config.auto_register_tools:
                await self._register_mcp_tools_as_flows(client)
            
            logger.info(f"Added MCP client '{config.name}' connected to {config.server_uri}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to add MCP client '{config.name}': {e}")
            raise
    
    async def add_mcp_server(self, config: MCPServerConfig) -> MCPServerProvider:
        """Add an MCP server to expose agent capabilities."""
        try:
            # Create server settings
            settings = MCPServerSettings(
                server_name=config.name,
                host=config.host,
                port=config.port,
                transport=config.transport,
                exposed_flows=config.expose_flows or [],
                expose_memory=config.expose_memory,
                expose_all_flows=config.expose_flows is None
            )
            
            # Create and initialize server
            server = MCPServerProvider(name=config.name, settings=settings)
            await server.initialize()
            
            # Store server
            self._mcp_servers[config.name] = server
            
            logger.info(f"Started MCP server '{config.name}' on {config.host}:{config.port}")
            return server
            
        except Exception as e:
            logger.error(f"Failed to start MCP server '{config.name}': {e}")
            raise
    
    async def _register_mcp_tools_as_flows(self, client: MCPClientProvider):
        """Register MCP tools from a client as agent flows."""
        try:
            tools = await client.list_tools()
            
            for tool_name, tool in tools.items():
                # Create wrapper flow
                tool_flow = MCPToolFlow(client, tool)
                
                # Register with agent
                self.register_flow(tool_flow, flow_name=tool_flow.flow_name)
                self._mcp_tools[tool_name] = tool_flow
                
                logger.debug(f"Registered MCP tool '{tool_name}' as flow '{tool_flow.flow_name}'")
                
        except Exception as e:
            logger.error(f"Error registering MCP tools as flows: {e}")
    
    async def call_mcp_tool(
        self, 
        client_name: str, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Any:
        """Call an MCP tool directly."""
        if client_name not in self._mcp_clients:
            raise ValueError(f"MCP client '{client_name}' not found")
        
        client = self._mcp_clients[client_name]
        return await client.call_tool(tool_name, arguments)
    
    async def read_mcp_resource(
        self, 
        client_name: str, 
        resource_uri: str
    ) -> Any:
        """Read an MCP resource directly."""
        if client_name not in self._mcp_clients:
            raise ValueError(f"MCP client '{client_name}' not found")
        
        client = self._mcp_clients[client_name]
        return await client.read_resource(resource_uri)
    
    async def list_mcp_capabilities(self) -> Dict[str, Any]:
        """List all MCP capabilities available to the agent."""
        capabilities = {
            "clients": {},
            "servers": {},
            "total_tools": 0,
            "total_resources": 0
        }
        
        # Client capabilities
        for client_name, client in self._mcp_clients.items():
            if client.is_connected():
                info = client.get_server_info()
                capabilities["clients"][client_name] = info
                if "tools_count" in info:
                    capabilities["total_tools"] += info["tools_count"]
                if "resources_count" in info:
                    capabilities["total_resources"] += info["resources_count"]
        
        # Server capabilities
        for server_name, server in self._mcp_servers.items():
            info = server.get_server_info()
            capabilities["servers"][server_name] = info
        
        return capabilities
    
    async def shutdown_mcp_connections(self):
        """Shutdown all MCP connections."""
        # Shutdown clients
        for client_name, client in self._mcp_clients.items():
            try:
                await client.shutdown()
                logger.debug(f"Shutdown MCP client '{client_name}'")
            except Exception as e:
                logger.warning(f"Error shutting down MCP client '{client_name}': {e}")
        
        # Shutdown servers
        for server_name, server in self._mcp_servers.items():
            try:
                await server.shutdown()
                logger.debug(f"Shutdown MCP server '{server_name}'")
            except Exception as e:
                logger.warning(f"Error shutting down MCP server '{server_name}': {e}")
        
        self._mcp_clients.clear()
        self._mcp_servers.clear()
        self._mcp_tools.clear()
    
    async def _shutdown_impl(self):
        """Override shutdown to include MCP cleanup."""
        await self.shutdown_mcp_connections()
        await super()._shutdown_impl()


# Helper functions for easy MCP integration
async def create_mcp_enabled_agent(
    agent_config,
    mcp_clients: List[MCPClientConfig] = None,
    mcp_servers: List[MCPServerConfig] = None
) -> MCPEnabledAgent:
    """Create an agent with MCP integration."""
    
    agent = MCPEnabledAgent(agent_config)
    await agent.initialize()
    
    # Add MCP clients
    if mcp_clients:
        for client_config in mcp_clients:
            await agent.add_mcp_client(client_config)
    
    # Add MCP servers
    if mcp_servers:
        for server_config in mcp_servers:
            await agent.add_mcp_server(server_config)
    
    return agent


# Convenience decorators for MCP-aware flows
def with_mcp_client(client_name: str):
    """Decorator to inject MCP client into flow."""
    def decorator(cls):
        original_init = cls.__init__
        
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._mcp_client_name = client_name
        
        cls.__init__ = __init__
        
        async def get_mcp_client(self):
            """Get the MCP client from the agent."""
            if hasattr(self, 'agent') and hasattr(self.agent, '_mcp_clients'):
                return self.agent._mcp_clients[self._mcp_client_name] if self._mcp_client_name in self.agent._mcp_clients else None
            return None
        
        cls.get_mcp_client = get_mcp_client
        return cls
    
    return decorator


# Example usage
async def example_mcp_integration():
    """Example of how to use MCP integration."""
    
    # Create MCP-enabled agent
    agent = await create_mcp_enabled_agent(
        agent_config={},  # Your agent config
        mcp_clients=[
            MCPClientConfig(
                name="github_client",
                server_uri="stdio://github-mcp-server",
                transport=MCPTransport.STDIO,
                server_command="github-mcp-server",
                auto_register_tools=True
            )
        ],
        mcp_servers=[
            MCPServerConfig(
                name="agent_server",
                port=8080,
                expose_memory=True
            )
        ]
    )
    
    try:
        # List available capabilities
        capabilities = await agent.list_mcp_capabilities()
        print("MCP Capabilities:", capabilities)
        
        # Call an MCP tool directly
        if "github_client" in agent._mcp_clients:
            result = await agent.call_mcp_tool(
                "github_client",
                "create_issue",
                {"title": "Test Issue", "body": "Created via MCP"}
            )
            print("GitHub issue created:", result)
        
        # Use MCP tool as a flow
        if "mcp_create_issue" in agent.flows:
            flow_result = await agent.execute_flow(
                "mcp_create_issue",
                {"title": "Another Test", "body": "Via flow"}
            )
            print("Flow result:", flow_result)
    
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(example_mcp_integration())