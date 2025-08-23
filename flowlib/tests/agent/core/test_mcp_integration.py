"""Comprehensive tests for MCP Integration Module."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional

from flowlib.agent.core.mcp_integration import (
    MCPClientConfig,
    MCPServerConfig,
    MCPToolFlow,
    MCPEnabledAgent,
    create_mcp_enabled_agent,
    with_mcp_client
)
from flowlib.providers.mcp.base import (
    MCPTransport,
    MCPTool,
    MCPResource,
    MCPToolInputSchema,
    MCPError,
    MCPToolNotFoundError,
    MCPResourceNotFoundError,
    MCPConnectionError
)
from flowlib.providers.mcp.client.provider import MCPClientProvider, MCPClientSettings
from flowlib.providers.mcp.server.provider import MCPServerProvider, MCPServerSettings
from flowlib.agent.core.agent import AgentCore
from flowlib.flows.models.results import FlowResult
from flowlib.flows.models.constants import FlowStatus


class MockMCPClientProvider:
    """Mock MCP client provider for testing."""
    
    def __init__(self, name: str, settings: MCPClientSettings):
        self.name = name
        self.settings = settings
        self.initialized = False
        self.connected = True
        self._tools = {}
        self._resources = {}
        self._server_info = {
            "name": "mock_server",
            "version": "1.0.0",
            "tools_count": 0,
            "resources_count": 0
        }
    
    async def initialize(self):
        """Mock initialization."""
        self.initialized = True
    
    async def shutdown(self):
        """Mock shutdown."""
        self.connected = False
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected
    
    async def list_tools(self) -> Dict[str, MCPTool]:
        """Mock list tools."""
        return self._tools.copy()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Mock call tool."""
        if tool_name not in self._tools:
            raise MCPToolNotFoundError(tool_name)
        return f"result_for_{tool_name}"
    
    async def read_resource(self, resource_uri: str) -> Any:
        """Mock read resource."""
        if resource_uri not in self._resources:
            raise MCPResourceNotFoundError(resource_uri)
        return f"content_of_{resource_uri}"
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server info."""
        info = self._server_info.copy()
        info.update({
            "tools_count": len(self._tools),
            "resources_count": len(self._resources)
        })
        return info
    
    def add_mock_tool(self, name: str, description: str = "Mock tool"):
        """Add a mock tool for testing."""
        tool = MCPTool(
            name=name,
            description=description,
            input_schema=MCPToolInputSchema()
        )
        self._tools[name] = tool
        self._server_info["tools_count"] = len(self._tools)
    
    def add_mock_resource(self, uri: str, name: str = "Mock resource"):
        """Add a mock resource for testing."""
        resource = MCPResource(uri=uri, name=name)
        self._resources[uri] = resource
        self._server_info["resources_count"] = len(self._resources)


class MockMCPServerProvider:
    """Mock MCP server provider for testing."""
    
    def __init__(self, name: str, settings: MCPServerSettings):
        self.name = name
        self.settings = settings
        self.initialized = False
        self.running = False
    
    async def initialize(self):
        """Mock initialization."""
        self.initialized = True
        self.running = True
    
    async def shutdown(self):
        """Mock shutdown."""
        self.running = False
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server info."""
        return {
            "name": self.name,
            "host": self.settings.host,
            "port": self.settings.port,
            "running": self.running,
            "exposed_flows": getattr(self.settings, 'exposed_flows', []),
            "expose_memory": getattr(self.settings, 'expose_memory', True)
        }


class MockAgentCore:
    """Mock agent core for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.flows = {}
        self.initialized = False
    
    async def initialize(self):
        """Mock initialization."""
        self.initialized = True
    
    async def shutdown(self):
        """Mock shutdown."""
        pass
    
    async def _shutdown_impl(self):
        """Mock shutdown implementation."""
        pass
    
    def register_flow(self, flow, flow_name: str):
        """Mock flow registration."""
        self.flows[flow_name] = flow
    
    async def execute_flow(self, flow_name: str, inputs: Dict[str, Any]) -> FlowResult:
        """Mock flow execution."""
        if flow_name not in self.flows:
            raise ValueError(f"Flow '{flow_name}' not found")
        
        flow = self.flows[flow_name]
        result = await flow.execute_tool(inputs)
        return FlowResult(
            status=FlowStatus.SUCCESS if result.get("success", True) else FlowStatus.ERROR,
            data=result
        )


class TestMCPClientConfig:
    """Test MCPClientConfig dataclass."""
    
    def test_client_config_creation(self):
        """Test creating MCP client configuration."""
        config = MCPClientConfig(
            name="test_client",
            server_uri="stdio://test-server"
        )
        
        assert config.name == "test_client"
        assert config.server_uri == "stdio://test-server"
        assert config.transport == MCPTransport.STDIO
        assert config.server_command is None
        assert config.server_args is None
        assert config.auth_token is None
        assert config.auto_register_tools is True
    
    def test_client_config_with_all_fields(self):
        """Test creating client config with all fields."""
        config = MCPClientConfig(
            name="full_client",
            server_uri="http://example.com",
            transport=MCPTransport.SSE,
            server_command="test-server",
            server_args=["--arg1", "--arg2"],
            auth_token="secret_token",
            auto_register_tools=False
        )
        
        assert config.name == "full_client"
        assert config.server_uri == "http://example.com"
        assert config.transport == MCPTransport.SSE
        assert config.server_command == "test-server"
        assert config.server_args == ["--arg1", "--arg2"]
        assert config.auth_token == "secret_token"
        assert config.auto_register_tools is False


class TestMCPServerConfig:
    """Test MCPServerConfig dataclass."""
    
    def test_server_config_creation(self):
        """Test creating MCP server configuration."""
        config = MCPServerConfig(name="test_server")
        
        assert config.name == "test_server"
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.transport == MCPTransport.SSE
        assert config.expose_flows is None
        assert config.expose_memory is True
    
    def test_server_config_with_all_fields(self):
        """Test creating server config with all fields."""
        config = MCPServerConfig(
            name="full_server",
            host="0.0.0.0",
            port=9090,
            transport=MCPTransport.WEBSOCKET,
            expose_flows=["flow1", "flow2"],
            expose_memory=False
        )
        
        assert config.name == "full_server"
        assert config.host == "0.0.0.0"
        assert config.port == 9090
        assert config.transport == MCPTransport.WEBSOCKET
        assert config.expose_flows == ["flow1", "flow2"]
        assert config.expose_memory is False


class TestMCPToolFlow:
    """Test MCPToolFlow wrapper."""
    
    def test_tool_flow_creation(self):
        """Test creating MCP tool flow wrapper."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        tool = MCPTool(
            name="test_tool",
            description="Test tool",
            input_schema=MCPToolInputSchema()
        )
        
        tool_flow = MCPToolFlow(client, tool)
        
        assert tool_flow.client == client
        assert tool_flow.tool == tool
        assert tool_flow.flow_name == "mcp_test_tool"
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        client.add_mock_tool("test_tool")
        
        tool = MCPTool(
            name="test_tool",
            description="Test tool",
            input_schema=MCPToolInputSchema()
        )
        
        tool_flow = MCPToolFlow(client, tool)
        
        result = await tool_flow.execute_tool({"param": "value"})
        
        assert result["success"] is True
        assert result["result"] == "result_for_test_tool"
        assert result["tool_name"] == "test_tool"
    
    @pytest.mark.asyncio
    async def test_execute_tool_failure(self):
        """Test tool execution with error."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        # Don't add the tool, so it will raise MCPToolNotFoundError
        
        tool = MCPTool(
            name="nonexistent_tool",
            description="Nonexistent tool",
            input_schema=MCPToolInputSchema()
        )
        
        tool_flow = MCPToolFlow(client, tool)
        
        result = await tool_flow.execute_tool({"param": "value"})
        
        assert result["success"] is False
        assert "error" in result
        assert result["tool_name"] == "nonexistent_tool"


class TestMCPEnabledAgent:
    """Test MCPEnabledAgent functionality."""
    
    def test_agent_initialization(self):
        """Test MCP-enabled agent initialization."""
        with patch('flowlib.agent.core.mcp_integration.AgentCore.__init__', return_value=None):
            agent = MCPEnabledAgent({})
            
            assert agent._mcp_clients == {}
            assert agent._mcp_servers == {}
            assert agent._mcp_tools == {}
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.core.mcp_integration.MCPClientProvider')
    async def test_add_mcp_client_success(self, mock_client_class):
        """Test successfully adding MCP client."""
        # Setup mocks
        mock_client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        mock_client_class.return_value = mock_client
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {}
            agent._mcp_servers = {}
            agent._mcp_tools = {}
            
            # Mock the _register_mcp_tools_as_flows method
            agent._register_mcp_tools_as_flows = AsyncMock()
            
            config = MCPClientConfig(
                name="test_client",
                server_uri="test://server",
                auto_register_tools=True
            )
            
            result = await agent.add_mcp_client(config)
            
            assert result == mock_client
            assert agent._mcp_clients["test_client"] == mock_client
            agent._register_mcp_tools_as_flows.assert_called_once_with(mock_client)
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.core.mcp_integration.MCPClientProvider')
    async def test_add_mcp_client_no_auto_register(self, mock_client_class):
        """Test adding MCP client without auto-registering tools."""
        mock_client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        mock_client_class.return_value = mock_client
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {}
            agent._mcp_servers = {}
            agent._mcp_tools = {}
            
            agent._register_mcp_tools_as_flows = AsyncMock()
            
            config = MCPClientConfig(
                name="test_client",
                server_uri="test://server",
                auto_register_tools=False
            )
            
            await agent.add_mcp_client(config)
            
            agent._register_mcp_tools_as_flows.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.core.mcp_integration.MCPClientProvider')
    async def test_add_mcp_client_failure(self, mock_client_class):
        """Test handling MCP client creation failure."""
        mock_client_class.side_effect = Exception("Connection failed")
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {}
            
            config = MCPClientConfig(
                name="failing_client",
                server_uri="test://server"
            )
            
            with pytest.raises(Exception, match="Connection failed"):
                await agent.add_mcp_client(config)
            
            assert "failing_client" not in agent._mcp_clients
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.core.mcp_integration.MCPServerProvider')
    async def test_add_mcp_server_success(self, mock_server_class):
        """Test successfully adding MCP server."""
        mock_server = MockMCPServerProvider("test_server", MCPServerSettings(server_name="test"))
        mock_server_class.return_value = mock_server
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {}
            agent._mcp_servers = {}
            agent._mcp_tools = {}
            
            config = MCPServerConfig(
                name="test_server",
                host="localhost",
                port=8080
            )
            
            result = await agent.add_mcp_server(config)
            
            assert result == mock_server
            assert agent._mcp_servers["test_server"] == mock_server
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.core.mcp_integration.MCPServerProvider')
    async def test_add_mcp_server_failure(self, mock_server_class):
        """Test handling MCP server creation failure."""
        mock_server_class.side_effect = Exception("Server start failed")
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_servers = {}
            
            config = MCPServerConfig(name="failing_server")
            
            with pytest.raises(Exception, match="Server start failed"):
                await agent.add_mcp_server(config)
            
            assert "failing_server" not in agent._mcp_servers
    
    @pytest.mark.asyncio
    async def test_register_mcp_tools_as_flows(self):
        """Test registering MCP tools as agent flows."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        client.add_mock_tool("tool1", "First tool")
        client.add_mock_tool("tool2", "Second tool")
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {}
            agent._mcp_servers = {}
            agent._mcp_tools = {}
            agent.register_flow = Mock()
            
            await agent._register_mcp_tools_as_flows(client)
            
            assert len(agent._mcp_tools) == 2
            assert "tool1" in agent._mcp_tools
            assert "tool2" in agent._mcp_tools
            
            # Verify flows were registered
            assert agent.register_flow.call_count == 2
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_success(self):
        """Test calling MCP tool directly."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        client.add_mock_tool("test_tool")
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {"test_client": client}
            
            result = await agent.call_mcp_tool("test_client", "test_tool", {"param": "value"})
            
            assert result == "result_for_test_tool"
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_client_not_found(self):
        """Test calling MCP tool with non-existent client."""
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {}
            
            with pytest.raises(ValueError, match="MCP client 'nonexistent' not found"):
                await agent.call_mcp_tool("nonexistent", "tool", {})
    
    @pytest.mark.asyncio
    async def test_read_mcp_resource_success(self):
        """Test reading MCP resource directly."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        client.add_mock_resource("test://resource", "Test Resource")
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {"test_client": client}
            
            result = await agent.read_mcp_resource("test_client", "test://resource")
            
            assert result == "content_of_test://resource"
    
    @pytest.mark.asyncio
    async def test_read_mcp_resource_client_not_found(self):
        """Test reading MCP resource with non-existent client."""
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {}
            
            with pytest.raises(ValueError, match="MCP client 'nonexistent' not found"):
                await agent.read_mcp_resource("nonexistent", "test://resource")
    
    @pytest.mark.asyncio
    async def test_list_mcp_capabilities(self):
        """Test listing MCP capabilities."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        client.add_mock_tool("tool1")
        client.add_mock_tool("tool2")
        client.add_mock_resource("resource1")
        
        server = MockMCPServerProvider("test_server", MCPServerSettings(server_name="test"))
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {"test_client": client}
            agent._mcp_servers = {"test_server": server}
            agent._mcp_tools = {}
            
            capabilities = await agent.list_mcp_capabilities()
            
            assert "clients" in capabilities
            assert "servers" in capabilities
            assert "total_tools" in capabilities
            assert "total_resources" in capabilities
            
            assert "test_client" in capabilities["clients"]
            assert "test_server" in capabilities["servers"]
            assert capabilities["total_tools"] == 2
            assert capabilities["total_resources"] == 1
    
    @pytest.mark.asyncio
    async def test_shutdown_mcp_connections(self):
        """Test shutting down all MCP connections."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        server = MockMCPServerProvider("test_server", MCPServerSettings(server_name="test"))
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {"test_client": client}
            agent._mcp_servers = {"test_server": server}
            agent._mcp_tools = {"tool1": Mock()}
            
            await agent.shutdown_mcp_connections()
            
            assert client.connected is False
            assert server.running is False
            assert agent._mcp_clients == {}
            assert agent._mcp_servers == {}
            assert agent._mcp_tools == {}
    
    @pytest.mark.asyncio
    async def test_shutdown_mcp_connections_with_errors(self):
        """Test shutdown handling errors gracefully."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        client.shutdown = AsyncMock(side_effect=Exception("Shutdown failed"))
        
        server = MockMCPServerProvider("test_server", MCPServerSettings(server_name="test"))
        server.shutdown = AsyncMock(side_effect=Exception("Server shutdown failed"))
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {"test_client": client}
            agent._mcp_servers = {"test_server": server}
            agent._mcp_tools = {}
            
            # Should not raise exception despite errors
            await agent.shutdown_mcp_connections()
            
            assert agent._mcp_clients == {}
            assert agent._mcp_servers == {}
    
    @pytest.mark.asyncio
    async def test_shutdown_impl_calls_mcp_shutdown(self):
        """Test that _shutdown_impl includes MCP cleanup."""
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            with patch.object(MCPEnabledAgent, 'shutdown_mcp_connections', new_callable=AsyncMock) as mock_mcp_shutdown:
                with patch('flowlib.agent.core.agent.AgentCore._shutdown_impl', new_callable=AsyncMock) as mock_super_shutdown:
                    agent = MCPEnabledAgent()
                    
                    await agent._shutdown_impl()
                    
                    mock_mcp_shutdown.assert_called_once()
                    mock_super_shutdown.assert_called_once()


class TestHelperFunctions:
    """Test helper functions."""
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.core.mcp_integration.MCPEnabledAgent')
    async def test_create_mcp_enabled_agent_basic(self, mock_agent_class):
        """Test creating MCP-enabled agent with basic config."""
        mock_agent = MockAgentCore()
        mock_agent_class.return_value = mock_agent
        
        agent_config = {"test": "config"}
        
        result = await create_mcp_enabled_agent(agent_config)
        
        assert result == mock_agent
        mock_agent_class.assert_called_once_with(agent_config)
    
    @pytest.mark.asyncio
    @patch('flowlib.agent.core.mcp_integration.MCPEnabledAgent')
    async def test_create_mcp_enabled_agent_with_clients_and_servers(self, mock_agent_class):
        """Test creating agent with MCP clients and servers."""
        mock_agent = MockAgentCore()
        mock_agent.add_mcp_client = AsyncMock()
        mock_agent.add_mcp_server = AsyncMock()
        mock_agent_class.return_value = mock_agent
        
        client_configs = [
            MCPClientConfig(name="client1", server_uri="uri1"),
            MCPClientConfig(name="client2", server_uri="uri2")
        ]
        
        server_configs = [
            MCPServerConfig(name="server1"),
            MCPServerConfig(name="server2")
        ]
        
        await create_mcp_enabled_agent(
            {},
            mcp_clients=client_configs,
            mcp_servers=server_configs
        )
        
        assert mock_agent.add_mcp_client.call_count == 2
        assert mock_agent.add_mcp_server.call_count == 2


class TestWithMcpClientDecorator:
    """Test with_mcp_client decorator."""
    
    def test_decorator_adds_client_name(self):
        """Test that decorator adds client name to class."""
        @with_mcp_client("test_client")
        class TestFlow:
            def __init__(self):
                pass
        
        flow = TestFlow()
        assert flow._mcp_client_name == "test_client"
    
    def test_decorator_adds_get_mcp_client_method(self):
        """Test that decorator adds get_mcp_client method."""
        @with_mcp_client("test_client")
        class TestFlow:
            def __init__(self):
                pass
        
        flow = TestFlow()
        assert hasattr(flow, 'get_mcp_client')
        assert callable(flow.get_mcp_client)
    
    @pytest.mark.asyncio
    async def test_get_mcp_client_with_agent(self):
        """Test getting MCP client from agent."""
        mock_client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        
        @with_mcp_client("test_client")
        class TestFlow:
            def __init__(self):
                self.agent = Mock()
                self.agent._mcp_clients = {"test_client": mock_client}
        
        flow = TestFlow()
        client = await flow.get_mcp_client()
        
        assert client == mock_client
    
    @pytest.mark.asyncio
    async def test_get_mcp_client_no_agent(self):
        """Test getting MCP client without agent."""
        @with_mcp_client("test_client")
        class TestFlow:
            def __init__(self):
                pass
        
        flow = TestFlow()
        client = await flow.get_mcp_client()
        
        assert client is None
    
    @pytest.mark.asyncio
    async def test_get_mcp_client_agent_no_clients(self):
        """Test getting MCP client when agent has no MCP clients."""
        @with_mcp_client("test_client")
        class TestFlow:
            def __init__(self):
                self.agent = Mock()
                # Agent doesn't have _mcp_clients attribute
                self.agent._mcp_clients = {}
        
        flow = TestFlow()
        client = await flow.get_mcp_client()
        
        assert client is None


class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_mcp_workflow(self):
        """Test complete MCP integration workflow."""
        # Create mock components
        client = MockMCPClientProvider("github_client", MCPClientSettings(server_uri="test://github"))
        client.add_mock_tool("create_issue", "Create GitHub issue")
        client.add_mock_resource("repo://test", "Test repository")
        
        server = MockMCPServerProvider("agent_server", MCPServerSettings(server_name="agent"))
        
        # Create agent with mocks
        with patch('flowlib.agent.core.mcp_integration.MCPClientProvider', return_value=client):
            with patch('flowlib.agent.core.mcp_integration.MCPServerProvider', return_value=server):
                with patch.object(MCPEnabledAgent, '__init__', return_value=None):
                    agent = MCPEnabledAgent()
                    agent._mcp_clients = {}
                    agent._mcp_servers = {}
                    agent._mcp_tools = {}
                    agent.register_flow = Mock()
                    
                    # Add client and server
                    client_config = MCPClientConfig(
                        name="github_client",
                        server_uri="stdio://github-server",
                        auto_register_tools=True
                    )
                    server_config = MCPServerConfig(
                        name="agent_server",
                        port=8080
                    )
                    
                    await agent.add_mcp_client(client_config)
                    await agent.add_mcp_server(server_config)
                    
                    # Verify setup
                    assert "github_client" in agent._mcp_clients
                    assert "agent_server" in agent._mcp_servers
                    assert len(agent._mcp_tools) == 1
                    
                    # Test capabilities
                    capabilities = await agent.list_mcp_capabilities()
                    assert capabilities["total_tools"] == 1
                    assert capabilities["total_resources"] == 1
                    
                    # Test tool call
                    result = await agent.call_mcp_tool(
                        "github_client",
                        "create_issue",
                        {"title": "Test"}
                    )
                    assert result == "result_for_create_issue"
                    
                    # Test resource read
                    content = await agent.read_mcp_resource(
                        "github_client",
                        "repo://test"
                    )
                    assert content == "content_of_repo://test"
                    
                    # Test shutdown
                    await agent.shutdown_mcp_connections()
                    assert len(agent._mcp_clients) == 0
                    assert len(agent._mcp_servers) == 0
    
    @pytest.mark.asyncio
    async def test_mcp_tool_flow_integration(self):
        """Test integration between MCP tools and flow system."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        client.add_mock_tool("test_tool")
        
        tool = MCPTool(
            name="test_tool",
            description="Test tool",
            input_schema=MCPToolInputSchema()
        )
        
        tool_flow = MCPToolFlow(client, tool)
        
        # Test successful execution
        result = await tool_flow.execute_tool({"param": "value"})
        
        assert result["success"] is True
        assert result["tool_name"] == "test_tool"
        assert "result" in result
        
        # Test integration with mock agent
        agent = MockAgentCore()
        agent.register_flow(tool_flow, "mcp_test_tool")
        
        assert "mcp_test_tool" in agent.flows
        
        # Test flow execution through agent
        flow_result = await agent.execute_flow("mcp_test_tool", {"param": "value"})
        assert flow_result.success is True


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_mcp_tool_not_found_error(self):
        """Test handling MCPToolNotFoundError."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        # Don't add any tools
        
        tool = MCPTool(
            name="nonexistent_tool",
            description="Tool that doesn't exist",
            input_schema=MCPToolInputSchema()
        )
        
        tool_flow = MCPToolFlow(client, tool)
        
        result = await tool_flow.execute_tool({"param": "value"})
        
        assert result["success"] is False
        assert "error" in result
        assert "Tool 'nonexistent_tool' not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_mcp_resource_not_found_error(self):
        """Test handling MCPResourceNotFoundError."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        # Don't add any resources
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {"test_client": client}
            
            with pytest.raises(MCPResourceNotFoundError):
                await agent.read_mcp_resource("test_client", "nonexistent://resource")
    
    @pytest.mark.asyncio
    async def test_client_initialization_failure(self):
        """Test handling client initialization failure."""
        with patch('flowlib.agent.core.mcp_integration.MCPClientProvider') as mock_client_class:
            mock_client_class.side_effect = MCPConnectionError("Failed to connect")
            
            with patch.object(MCPEnabledAgent, '__init__', return_value=None):
                agent = MCPEnabledAgent()
                agent._mcp_clients = {}
                
                config = MCPClientConfig(
                    name="failing_client",
                    server_uri="invalid://server"
                )
                
                with pytest.raises(MCPConnectionError):
                    await agent.add_mcp_client(config)
    
    @pytest.mark.asyncio
    async def test_tool_registration_failure(self):
        """Test handling tool registration failure."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        client.list_tools = AsyncMock(side_effect=Exception("Failed to list tools"))
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {}
            agent._mcp_servers = {}
            agent._mcp_tools = {}
            
            # Should not raise exception, but log error
            await agent._register_mcp_tools_as_flows(client)
            
            assert len(agent._mcp_tools) == 0


class TestEdgeCases:
    """Test edge cases and corner scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_tool_list(self):
        """Test handling empty tool list from MCP client."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        # No tools added
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {}
            agent._mcp_servers = {}
            agent._mcp_tools = {}
            agent.register_flow = Mock()
            
            await agent._register_mcp_tools_as_flows(client)
            
            assert len(agent._mcp_tools) == 0
            agent.register_flow.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_disconnected_client_capabilities(self):
        """Test listing capabilities with disconnected client."""
        client = MockMCPClientProvider("test_client", MCPClientSettings(server_uri="test://uri"))
        client.connected = False  # Simulate disconnected client
        
        with patch.object(MCPEnabledAgent, '__init__', return_value=None):
            agent = MCPEnabledAgent()
            agent._mcp_clients = {"test_client": client}
            agent._mcp_servers = {}
            agent._mcp_tools = {}
            
            capabilities = await agent.list_mcp_capabilities()
            
            # Disconnected client should not appear in capabilities
            assert "test_client" not in capabilities["clients"]
            assert capabilities["total_tools"] == 0
            assert capabilities["total_resources"] == 0
    
    @pytest.mark.asyncio
    async def test_config_with_none_server_args(self):
        """Test config handling when server_args is None."""
        config = MCPClientConfig(
            name="test_client",
            server_uri="test://server",
            server_args=None
        )
        
        assert config.server_args is None
        
        # Test that it defaults to empty list in the agent
        with patch('flowlib.agent.core.mcp_integration.MCPClientProvider') as mock_client_class:
            with patch.object(MCPEnabledAgent, '__init__', return_value=None):
                agent = MCPEnabledAgent()
                agent._mcp_clients = {}
                agent._register_mcp_tools_as_flows = AsyncMock()
                
                # This should not raise an error when server_args is None
                # The agent should handle it by converting to empty list
                mock_client_class.return_value = MockMCPClientProvider("test", MCPClientSettings(server_uri="test"))
                
                # This tests the MCPClientSettings creation in add_mcp_client
                await agent.add_mcp_client(config)


if __name__ == "__main__":
    pytest.main([__file__])