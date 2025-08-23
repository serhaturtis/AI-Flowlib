"""Tests for MCP client provider."""
import pytest
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pydantic import ValidationError

from flowlib.providers.mcp.client.provider import (
    MCPClientProvider,
    MCPClientSettings,
)
from flowlib.providers.mcp.base import MCPTransport, MCPTool, MCPError, MCPConnectionError
from flowlib.core.errors.errors import ProviderError


class TestMCPClientSettings:
    """Test MCP client settings."""
    
    def test_default_settings(self):
        """Test default MCP client settings."""
        settings = MCPClientSettings(server_uri="stdio://test-server")
        
        assert settings.server_uri == "stdio://test-server"
        assert settings.transport == MCPTransport.STDIO
        assert settings.timeout == 30.0
        assert settings.retry_attempts == 3
        assert settings.retry_delay == 1.0
        assert settings.server_command is None
        assert settings.server_args == []
        assert settings.auth_token is None
        assert settings.headers == {}
    
    def test_stdio_settings(self):
        """Test STDIO transport settings."""
        settings = MCPClientSettings(
            server_uri="stdio://test-server",
            transport=MCPTransport.STDIO,
            server_command="python",
            server_args=["-m", "my_mcp_server", "--verbose"],
            timeout=60.0,
            retry_attempts=5,
            retry_delay=2.0
        )
        
        assert settings.server_uri == "stdio://test-server"
        assert settings.transport == MCPTransport.STDIO
        assert settings.server_command == "python"
        assert settings.server_args == ["-m", "my_mcp_server", "--verbose"]
        assert settings.timeout == 60.0
        assert settings.retry_attempts == 5
        assert settings.retry_delay == 2.0
    
    def test_sse_settings(self):
        """Test SSE transport settings."""
        headers = {"User-Agent": "FlowLib/1.0", "Custom": "value"}
        
        settings = MCPClientSettings(
            server_uri="https://api.example.com/mcp/sse",
            transport=MCPTransport.SSE,
            auth_token="bearer_token_123",
            headers=headers,
            timeout=45.0
        )
        
        assert settings.server_uri == "https://api.example.com/mcp/sse"
        assert settings.transport == MCPTransport.SSE
        assert settings.auth_token == "bearer_token_123"
        assert settings.headers == headers
        assert settings.timeout == 45.0
    
    def test_websocket_settings(self):
        """Test WebSocket transport settings."""
        settings = MCPClientSettings(
            server_uri="wss://mcp.example.com/ws",
            transport=MCPTransport.WEBSOCKET,
            auth_token="ws_token_456",
            headers={"Origin": "https://app.example.com"},
            timeout=120.0
        )
        
        assert settings.server_uri == "wss://mcp.example.com/ws"
        assert settings.transport == MCPTransport.WEBSOCKET
        assert settings.auth_token == "ws_token_456"
        assert settings.headers["Origin"] == "https://app.example.com"
        assert settings.timeout == 120.0
    
    def test_settings_validation(self):
        """Test settings validation."""
        # Valid settings
        settings = MCPClientSettings(server_uri="stdio://test")
        assert settings.server_uri == "stdio://test"
        
        # Missing required server_uri
        with pytest.raises(ValidationError):
            MCPClientSettings()
    
    def test_settings_inheritance(self):
        """Test that MCPClientSettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        
        settings = MCPClientSettings(server_uri="test://server")
        assert isinstance(settings, ProviderSettings)


class TestMCPClientProvider:
    """Test MCP client provider."""
    
    @pytest.fixture
    def stdio_settings(self):
        """Create STDIO transport settings."""
        return MCPClientSettings(
            server_uri="stdio://test-server",
            transport=MCPTransport.STDIO,
            server_command="python",
            server_args=["-m", "test_server"],
            timeout=30.0
        )
    
    @pytest.fixture
    def sse_settings(self):
        """Create SSE transport settings."""
        return MCPClientSettings(
            server_uri="https://api.example.com/mcp",
            transport=MCPTransport.SSE,
            auth_token="test_token",
            timeout=30.0
        )
    
    @pytest.fixture
    def provider(self, stdio_settings):
        """Create test provider."""
        return MCPClientProvider(name="test_mcp_client", settings=stdio_settings)
    
    @pytest.fixture
    def mock_client(self):
        """Create mock MCP client."""
        mock = AsyncMock()
        mock.initialize = AsyncMock()
        mock.close = AsyncMock()
        mock.list_tools = AsyncMock()
        mock.call_tool = AsyncMock()
        mock.list_resources = AsyncMock()
        mock.read_resource = AsyncMock()
        mock.is_connected = Mock(return_value=True)
        mock.get_available_tools = Mock(return_value={})
        mock.get_available_resources = Mock(return_value={})
        return mock
    
    @pytest.fixture
    def mock_transport(self):
        """Create mock transport."""
        mock = AsyncMock()
        mock.connect = AsyncMock()
        mock.close = AsyncMock()
        mock.send_message = AsyncMock()
        mock.receive_message = AsyncMock()
        mock.is_connected = Mock(return_value=True)
        return mock
    
    def test_provider_initialization(self, stdio_settings):
        """Test provider initialization."""
        provider = MCPClientProvider(name="test_provider", settings=stdio_settings)
        
        assert provider.name == "test_provider"
        assert provider.provider_type == "mcp_client"
        assert provider.settings == stdio_settings
        assert provider._client is None
        assert provider._connection is None
        assert provider._connected is False
    
    def test_provider_inheritance(self, provider):
        """Test that MCPClientProvider inherits from Provider."""
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, Provider)
    
    @patch('flowlib.providers.mcp.client.provider.create_transport')
    async def test_initialize_stdio_success(self, mock_create_transport, provider, mock_transport, mock_client):
        """Test successful initialization with STDIO transport."""
        mock_create_transport.return_value = mock_transport
        
        with patch('flowlib.providers.mcp.client.provider.BaseMCPClient', return_value=mock_client):
            await provider.initialize()
        
        # Verify transport creation
        mock_create_transport.assert_called_once_with(
            transport_type=MCPTransport.STDIO,
            server_uri="stdio://test-server",
            server_command="python",
            server_args=["-m", "test_server"],
            timeout=30.0,
            auth_token=None,
            headers={}
        )
        
        # Verify client initialization
        mock_client.initialize.assert_called_once_with(mock_transport)
        assert provider._client == mock_client
        assert provider._connection == mock_transport
        assert provider.initialized is True
    
    @patch('flowlib.providers.mcp.client.provider.create_transport')
    async def test_initialize_sse_success(self, mock_create_transport, sse_settings, mock_transport, mock_client):
        """Test successful initialization with SSE transport."""
        provider = MCPClientProvider(name="sse_client", settings=sse_settings)
        mock_create_transport.return_value = mock_transport
        
        with patch('flowlib.providers.mcp.client.provider.BaseMCPClient', return_value=mock_client):
            await provider.initialize()
        
        # Verify transport creation with SSE settings
        mock_create_transport.assert_called_once_with(
            transport_type=MCPTransport.SSE,
            server_uri="https://api.example.com/mcp",
            auth_token="test_token",
            timeout=30.0,
            server_command=None,
            server_args=[],
            headers={}
        )
        
        mock_client.initialize.assert_called_once_with(mock_transport)
        assert provider.initialized is True
    
    @patch('flowlib.providers.mcp.client.provider.create_transport')
    async def test_initialize_connection_error(self, mock_create_transport, provider, mock_transport, mock_client):
        """Test initialization with connection error."""
        mock_create_transport.return_value = mock_transport
        mock_client.initialize.side_effect = Exception("Connection failed")
        
        with patch('flowlib.providers.mcp.client.provider.BaseMCPClient', return_value=mock_client):
            with pytest.raises(MCPConnectionError) as exc_info:
                await provider.initialize()
        
        assert "Failed to connect to MCP server" in str(exc_info.value)
        assert provider._client is None
        assert provider.initialized is False
    
    @patch('flowlib.providers.mcp.client.provider.create_transport')
    async def test_initialize_transport_error(self, mock_create_transport, provider):
        """Test initialization with transport creation error."""
        mock_create_transport.side_effect = Exception("Transport creation failed")
        
        with pytest.raises(MCPConnectionError) as exc_info:
            await provider.initialize()
        
        assert "Failed to connect to MCP server" in str(exc_info.value)
    
    async def test_shutdown_success(self, provider, mock_client, mock_transport):
        """Test successful shutdown."""
        # Set up connected state
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connection', mock_transport)
        object.__setattr__(provider, '_connected', True)
        object.__setattr__(provider, '_initialized', True)
        
        await provider.shutdown()
        
        # Verify cleanup
        mock_client.close.assert_called_once()
        mock_transport.close.assert_called_once()
        assert provider._client is None
        assert provider._connection is None
        assert provider._connected is False
        assert provider.initialized is False
    
    async def test_shutdown_not_connected(self, provider):
        """Test shutdown when not connected."""
        # Should not raise error
        await provider.shutdown()
        
        assert provider._client is None
        assert provider._connected is False
    
    async def test_shutdown_client_error(self, provider, mock_client, mock_transport):
        """Test shutdown with client disconnect error."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connection', mock_transport)
        object.__setattr__(provider, '_connected', True)
        mock_client.close.side_effect = Exception("Close failed")
        
        # Should still clean up even with error
        await provider.shutdown()
        
        assert provider._client is None
        assert provider._connection is None
        assert provider._connected is False
    
    async def test_list_tools_success(self, provider, mock_client):
        """Test successful tool listing."""
        # Set up connected state
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connected', True)
        
        # Mock tools response
        mock_tools = {
            "echo": MCPTool(
                name="echo",
                description="Echo input text",
                input_schema={"type": "object", "properties": {"text": {"type": "string"}}}
            ),
            "calculate": MCPTool(
                name="calculate",
                description="Perform calculation",
                input_schema={"type": "object", "properties": {"expression": {"type": "string"}}}
            )
        }
        mock_client.get_available_tools.return_value = mock_tools
        
        tools = await provider.list_tools()
        
        assert len(tools) == 2
        assert "echo" in tools
        assert "calculate" in tools
        assert tools["echo"].name == "echo"
        assert tools["calculate"].name == "calculate"
        mock_client.get_available_tools.assert_called_once()
    
    async def test_list_tools_not_connected(self, provider):
        """Test tool listing when not connected."""
        object.__setattr__(provider, '_connected', False)
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.list_tools()
        
        assert "MCP client not connected" in str(exc_info.value)
    
    async def test_list_tools_client_error(self, provider, mock_client):
        """Test tool listing with client error."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connected', True)
        mock_client.get_available_tools.side_effect = Exception("List tools failed")
        
        # The method should raise the original exception, not wrap it
        with pytest.raises(Exception) as exc_info:
            await provider.list_tools()
        
        assert "List tools failed" in str(exc_info.value)
    
    async def test_call_tool_success(self, provider, mock_client):
        """Test successful tool call."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connected', True)
        
        # Mock tool call response
        mock_response = {
            "content": [
                {"type": "text", "text": "Hello, World!"}
            ]
        }
        mock_client.call_tool.return_value = mock_response
        
        result = await provider.call_tool("echo", {"text": "Hello, World!"})
        
        assert result == mock_response
        mock_client.call_tool.assert_called_once_with("echo", {"text": "Hello, World!"})
    
    async def test_call_tool_not_connected(self, provider):
        """Test tool call when not connected."""
        object.__setattr__(provider, '_connected', False)
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.call_tool("echo", {"text": "test"})
        
        assert "MCP client not connected" in str(exc_info.value)
    
    async def test_call_tool_not_found(self, provider, mock_client):
        """Test tool call with tool not found error."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connected', True)
        
        from flowlib.providers.mcp.base import MCPToolNotFoundError
        mock_client.call_tool.side_effect = MCPToolNotFoundError("Tool 'nonexistent' not found")
        
        with pytest.raises(MCPToolNotFoundError) as exc_info:
            await provider.call_tool("nonexistent", {})
        
        assert "Tool 'nonexistent' not found" in str(exc_info.value)
    
    async def test_call_tool_client_error(self, provider, mock_client):
        """Test tool call with client error."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connected', True)
        mock_client.call_tool.side_effect = Exception("Tool call failed")
        
        with pytest.raises(Exception) as exc_info:
            await provider.call_tool("echo", {"text": "test"})
        
        assert "Tool call failed" in str(exc_info.value)
    
    async def test_list_resources_success(self, provider, mock_client):
        """Test successful resource listing."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connected', True)
        
        # Mock resources response
        mock_resources = {
            "file1": {
                "uri": "file:///path/to/file1.txt",
                "name": "file1.txt",
                "mimeType": "text/plain"
            },
            "file2": {
                "uri": "file:///path/to/file2.json",
                "name": "file2.json", 
                "mimeType": "application/json"
            }
        }
        mock_client.get_available_resources.return_value = mock_resources
        
        resources = await provider.list_resources()
        
        assert len(resources) == 2
        assert "file1" in resources
        assert "file2" in resources
        assert resources["file1"]["name"] == "file1.txt"
        assert resources["file2"]["mimeType"] == "application/json"
        mock_client.get_available_resources.assert_called_once()
    
    async def test_list_resources_not_connected(self, provider):
        """Test resource listing when not connected."""
        object.__setattr__(provider, '_connected', False)
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.list_resources()
        
        assert "MCP client not connected" in str(exc_info.value)
    
    async def test_read_resource_success(self, provider, mock_client):
        """Test successful resource reading."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connected', True)
        
        # Mock resource content
        mock_content = {
            "contents": [
                {
                    "uri": "file:///path/to/file.txt",
                    "mimeType": "text/plain",
                    "text": "File content here"
                }
            ]
        }
        mock_client.read_resource.return_value = mock_content
        
        content = await provider.read_resource("file:///path/to/file.txt")
        
        assert content == mock_content
        mock_client.read_resource.assert_called_once_with("file:///path/to/file.txt")
    
    async def test_read_resource_not_connected(self, provider):
        """Test resource reading when not connected."""
        object.__setattr__(provider, '_connected', False)
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.read_resource("file:///test.txt")
        
        assert "MCP client not connected" in str(exc_info.value)
    
    async def test_read_resource_not_found(self, provider, mock_client):
        """Test resource reading with resource not found."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connected', True)
        
        from flowlib.providers.mcp.base import MCPError
        mock_client.read_resource.side_effect = MCPError(
            code=-32002,
            message="Resource not found"
        )
        
        with pytest.raises(MCPError) as exc_info:
            await provider.read_resource("file:///nonexistent.txt")
        
        assert "Resource not found" in str(exc_info.value)
    
    async def test_check_connection_true(self, provider, mock_client):
        """Test connection check when connected."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connected', True)
        mock_client.is_connected.return_value = True
        
        result = await provider.check_connection()
        
        assert result is True
    
    async def test_check_connection_false(self, provider, mock_client):
        """Test connection check when disconnected."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connected', False)
        mock_client.is_connected.return_value = False
        
        result = await provider.check_connection()
        
        assert result is False
    
    async def test_check_connection_no_client(self, provider):
        """Test connection check when no client."""
        object.__setattr__(provider, '_client', None)
        
        result = await provider.check_connection()
        
        assert result is False
    
    async def test_reconnect_success(self, provider, mock_client, mock_transport):
        """Test successful reconnection."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connection', mock_transport)
        object.__setattr__(provider, '_connected', False)
        
        await provider.reconnect()
        
        mock_client.initialize.assert_called_once_with(mock_transport)
        assert provider._connected is True
    
    async def test_reconnect_failure(self, provider, mock_client, mock_transport):
        """Test reconnection failure."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connection', mock_transport)
        object.__setattr__(provider, '_connected', False)
        mock_client.initialize.side_effect = Exception("Reconnect failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.reconnect()
        
        assert "Failed to reconnect" in str(exc_info.value)
        assert provider._connected is False
    
    async def test_get_server_info(self, provider, mock_client):
        """Test getting server information."""
        object.__setattr__(provider, '_client', mock_client)
        object.__setattr__(provider, '_connected', True)
        
        # Mock tools and resources
        mock_tools = {"echo": Mock(), "calculate": Mock()}
        mock_resources = {"file1": Mock(), "file2": Mock()}
        mock_client.get_available_tools.return_value = mock_tools
        mock_client.get_available_resources.return_value = mock_resources
        
        info = await provider.get_server_info()
        
        assert info["connected"] is True
        assert info["server_uri"] == provider.settings.server_uri
        assert info["transport"] == provider.settings.transport.value
        assert info["tools_count"] == 2
        assert info["resources_count"] == 2
        assert "echo" in info["available_tools"]
        assert "calculate" in info["available_tools"]
    
    def test_provider_registration(self):
        """Test that provider is properly registered."""
        # This would be tested in integration tests
        # Here we just verify basic provider functionality
        settings = MCPClientSettings(server_uri="stdio://test")
        provider = MCPClientProvider(name="test", settings=settings)
        
        assert provider.name == "test"
        assert provider.provider_type == "mcp_client"


@pytest.mark.integration
class TestMCPClientProviderIntegration:
    """Integration tests for MCP client provider.
    
    These tests require a running MCP server or mock server.
    """
    
    @pytest.fixture
    def integration_settings(self):
        """Create integration test settings."""
        return MCPClientSettings(
            server_uri="stdio://test-server",
            transport=MCPTransport.STDIO,
            server_command="python",
            server_args=["-m", "test_mcp_server"],
            timeout=10.0
        )
    
    @pytest.fixture
    async def provider(self, integration_settings):
        """Create and initialize provider for integration tests."""
        provider = MCPClientProvider(name="integration_client", settings=integration_settings)
        
        try:
            # Note: This would require a real MCP server
            # await provider.initialize()
            yield provider
        finally:
            # Cleanup
            try:
                await provider.shutdown()
            except:
                pass
    
    def test_settings_integration(self, integration_settings):
        """Test settings integration with real values."""
        assert integration_settings.server_uri == "stdio://test-server"
        assert integration_settings.transport == MCPTransport.STDIO
        assert integration_settings.server_command == "python"
        assert integration_settings.server_args == ["-m", "test_mcp_server"]
    
    def test_provider_creation_integration(self, integration_settings):
        """Test provider creation with real settings."""
        provider = MCPClientProvider(name="integration_test", settings=integration_settings)
        
        assert provider.name == "integration_test"
        assert provider.settings.server_uri == "stdio://test-server"
        assert not provider.initialized
    
    async def test_tool_call_flow_simulation(self, integration_settings):
        """Test simulated tool call flow."""
        provider = MCPClientProvider(name="test", settings=integration_settings)
        
        # Simulate tool call without actual connection
        # This tests the method signature and basic validation
        with pytest.raises(ProviderError):  # Should fail - not connected
            await provider.call_tool("echo", {"text": "test"})
    
    def test_transport_configuration_integration(self, integration_settings):
        """Test transport configuration for different protocols."""
        # Test STDIO configuration
        stdio_settings = MCPClientSettings(
            server_uri="stdio://server",
            transport=MCPTransport.STDIO,
            server_command="node",
            server_args=["server.js"]
        )
        
        stdio_provider = MCPClientProvider(name="stdio_test", settings=stdio_settings)
        assert stdio_provider.settings.server_command == "node"
        
        # Test SSE configuration
        sse_settings = MCPClientSettings(
            server_uri="https://api.example.com/mcp",
            transport=MCPTransport.SSE,
            auth_token="bearer_token"
        )
        
        sse_provider = MCPClientProvider(name="sse_test", settings=sse_settings)
        assert sse_provider.settings.auth_token == "bearer_token"