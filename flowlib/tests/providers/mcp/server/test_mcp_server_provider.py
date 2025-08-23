"""Tests for MCP server provider."""
import pytest
import asyncio
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pydantic import ValidationError

# Test both with and without aiohttp installed
try:
    from aiohttp import web, WSMsgType
    import aiohttp_cors
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    web = None
    WSMsgType = None
    aiohttp_cors = None

from flowlib.providers.mcp.server.provider import (
    MCPServerProvider,
    MCPServerSettings,
    FlowToolHandler,
)
from flowlib.providers.mcp.base import MCPTransport, MCPTool, MCPMessage, MCPError
from flowlib.core.errors.errors import ProviderError


class TestMCPServerSettings:
    """Test MCP server settings."""
    
    def test_default_settings(self):
        """Test default MCP server settings."""
        settings = MCPServerSettings()
        
        assert settings.server_name == "flowlib-server"
        assert settings.server_version == "1.0.0"
        assert settings.host == "localhost"
        assert settings.port == 8080
        assert settings.transport == MCPTransport.SSE
        assert settings.expose_all_flows is False
        assert settings.exposed_flows == []
        assert settings.exclude_flows == []
        assert settings.expose_memory is True
        assert settings.expose_flow_registry is True
        assert settings.expose_metrics is False
    
    def test_custom_settings(self):
        """Test custom MCP server settings."""
        settings = MCPServerSettings(
            server_name="custom-mcp-server",
            server_version="2.1.0",
            host="0.0.0.0",
            port=9090,
            transport=MCPTransport.WEBSOCKET,
            expose_all_flows=True,
            exposed_flows=["flow1", "flow2"],
            exclude_flows=["internal_flow"],
            expose_memory=False,
            expose_flow_registry=False,
            expose_metrics=True
        )
        
        assert settings.server_name == "custom-mcp-server"
        assert settings.server_version == "2.1.0"
        assert settings.host == "0.0.0.0"
        assert settings.port == 9090
        assert settings.transport == MCPTransport.WEBSOCKET
        assert settings.expose_all_flows is True
        assert settings.exposed_flows == ["flow1", "flow2"]
        assert settings.exclude_flows == ["internal_flow"]
        assert settings.expose_memory is False
        assert settings.expose_flow_registry is False
        assert settings.expose_metrics is True
    
    def test_stdio_settings(self):
        """Test STDIO transport settings."""
        settings = MCPServerSettings(
            transport=MCPTransport.STDIO,
            expose_all_flows=True
        )
        
        assert settings.transport == MCPTransport.STDIO
        assert settings.expose_all_flows is True
    
    def test_settings_inheritance(self):
        """Test that MCPServerSettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        
        settings = MCPServerSettings()
        assert isinstance(settings, ProviderSettings)


class TestFlowToolHandler:
    """Test flow tool handler."""
    
    @pytest.fixture
    def mock_flow(self):
        """Create mock flow."""
        mock = Mock()
        mock.name = "test_flow"
        mock.description = "Test flow description"
        mock.run_pipeline = AsyncMock()
        mock.run_pipeline.__annotations__ = {}
        mock.execute = AsyncMock()
        mock.get_input_schema = Mock()
        return mock
    
    @pytest.fixture
    def handler(self, mock_flow):
        """Create flow tool handler."""
        return FlowToolHandler(mock_flow, "test_flow")
    
    def test_handler_initialization(self, mock_flow):
        """Test handler initialization."""
        handler = FlowToolHandler(mock_flow, "test_flow")
        
        assert handler.flow == mock_flow
        assert handler.flow_name == "test_flow"
    
    def test_create_tool_schema(self, handler, mock_flow):
        """Test tool schema creation."""
        # Mock flow input schema
        mock_schema = {
            "type": "object",
            "properties": {
                "input_text": {"type": "string"},
                "count": {"type": "integer", "minimum": 1}
            },
            "required": ["input_text"]
        }
        mock_flow.get_input_schema.return_value = mock_schema
        
        tool = handler.create_tool()
        
        assert isinstance(tool, MCPTool)
        assert tool.name == "test_flow"
        assert tool.description == "Test flow description"
        assert tool.input_schema.type == "object"
        assert "input_text" in tool.input_schema.properties
        assert "count" in tool.input_schema.properties
        assert "input_text" in tool.input_schema.required
    
    def test_create_tool_no_schema(self, handler, mock_flow):
        """Test tool creation when flow has no schema."""
        mock_flow.get_input_schema.return_value = None
        
        tool = handler.create_tool()
        
        assert tool.input_schema.type == "object"
        assert tool.input_schema.properties == {}
        assert tool.input_schema.required == []
    
    async def test_execute_tool_success(self, handler, mock_flow):
        """Test successful tool execution."""
        # Mock flow execution
        mock_result = {"output": "Test result", "status": "success"}
        mock_flow.run_pipeline.return_value = mock_result
        
        arguments = {"input_text": "test input", "count": 5}
        result = await handler.execute_tool(arguments)
        
        assert result == mock_result
        mock_flow.run_pipeline.assert_called_once_with(arguments)
    
    async def test_execute_tool_flow_error(self, handler, mock_flow):
        """Test tool execution with flow error."""
        mock_flow.run_pipeline.side_effect = Exception("Flow execution failed")
        
        with pytest.raises(Exception) as exc_info:
            await handler.execute_tool({"input": "test"})
        
        assert "Flow execution failed" in str(exc_info.value)
    
    async def test_execute_tool_empty_arguments(self, handler, mock_flow):
        """Test tool execution with empty arguments."""
        mock_result = {"output": "Default result"}
        mock_flow.run_pipeline.return_value = mock_result
        
        result = await handler.execute_tool({})
        
        assert result == mock_result
        mock_flow.run_pipeline.assert_called_once_with({})


class TestMCPServerProvider:
    """Test MCP server provider."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return MCPServerSettings(
            server_name="test-server",
            host="127.0.0.1",
            port=8081,
            transport=MCPTransport.STDIO,
            expose_all_flows=False,
            exposed_flows=["flow1", "flow2"]
        )
    
    @pytest.fixture
    def provider(self, settings):
        """Create test provider."""
        return MCPServerProvider(name="test_mcp_server", settings=settings)
    
    @pytest.fixture
    def mock_server(self):
        """Create mock MCP server."""
        mock = AsyncMock()
        mock.stop = Mock()
        mock.register_tool = Mock()
        mock.register_resource = Mock()
        mock._tools = {}
        mock._resources = {}
        mock._tool_handlers = {}
        mock._resource_handlers = {}
        return mock
    
    @pytest.fixture
    def mock_flow_registry(self):
        """Create mock flow registry."""
        mock = Mock()
        mock.get_flow_instances = Mock()
        mock.get_flow = Mock()
        return mock
    
    @pytest.fixture
    def mock_flows(self):
        """Create mock flows."""
        flow1 = Mock()
        flow1.name = "flow1"
        flow1.description = "First test flow"
        flow1.get_input_schema = Mock(return_value={"type": "object"})
        flow1.run_pipeline = AsyncMock()
        flow1.run_pipeline.__annotations__ = {}
        flow1.execute = AsyncMock()
        
        flow2 = Mock()
        flow2.name = "flow2"
        flow2.description = "Second test flow"
        flow2.get_input_schema = Mock(return_value={"type": "object"})
        flow2.run_pipeline = AsyncMock()
        flow2.run_pipeline.__annotations__ = {}
        flow2.execute = AsyncMock()
        
        return {"flow1": flow1, "flow2": flow2}
    
    def test_provider_initialization(self, settings):
        """Test provider initialization."""
        provider = MCPServerProvider(name="test_provider", settings=settings)
        
        assert provider.name == "test_provider"
        assert provider.provider_type == "mcp_server"
        assert provider.settings == settings
        assert provider._server is None
        assert provider._http_server is None
        assert provider._running is False
    
    def test_provider_inheritance(self, provider):
        """Test that MCPServerProvider inherits from Provider."""
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, Provider)
    
    @patch('flowlib.providers.mcp.server.provider.BaseMCPServer')
    @patch('flowlib.providers.mcp.server.provider.flow_registry')
    async def test_initialize_success(self, mock_registry, mock_server_class, provider, mock_server, mock_flows):
        """Test successful initialization."""
        mock_server_class.return_value = mock_server
        mock_registry.get_flow_instances.return_value = mock_flows
        
        await provider.initialize()
        
        # Verify server creation
        mock_server_class.assert_called_once_with(
            name="test-server",
            version="1.0.0"
        )
        
        # Verify server assignment
        assert provider._server == mock_server
        assert provider.initialized is True
    
    @patch('flowlib.providers.mcp.server.provider.BaseMCPServer')
    async def test_initialize_server_error(self, mock_server_class, provider):
        """Test initialization with server error."""
        # Make the BaseMCPServer constructor fail
        mock_server_class.side_effect = Exception("Server creation failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "Failed to initialize provider" in str(exc_info.value)
        assert provider._server is None
        assert provider.initialized is False
    
    @patch('flowlib.providers.mcp.server.provider.flow_registry')
    async def test_initialize_flow_registration_error(self, mock_registry, provider, mock_server):
        """Test initialization with flow registration error."""
        mock_registry.get_flow_instances.side_effect = Exception("Registry access failed")
        
        with patch('flowlib.providers.mcp.server.provider.BaseMCPServer', return_value=mock_server):
            with pytest.raises(ProviderError) as exc_info:
                await provider.initialize()
        
        assert "Registry access failed" in str(exc_info.value)
    
    async def test_shutdown_success(self, provider, mock_server):
        """Test successful shutdown."""
        # Set up running state
        provider._server = mock_server
        provider._running = True
        provider._initialized = True
        
        await provider.shutdown()
        
        # Verify cleanup
        mock_server.stop.assert_called_once()
        assert provider._server is None
        assert provider._running is False
        assert provider.initialized is False
    
    async def test_shutdown_not_running(self, provider):
        """Test shutdown when not running."""
        # Should not raise error
        await provider.shutdown()
        
        assert provider._server is None
        assert provider._running is False
    
    async def test_shutdown_server_error(self, provider, mock_server):
        """Test shutdown with server stop error."""
        provider._server = mock_server
        provider._running = True
        provider._initialized = True  # Need this for shutdown to execute
        mock_server.stop.side_effect = Exception("Stop failed")
        
        # Should still clean up even with error
        await provider.shutdown()
        
        assert provider._server is None
        assert provider._running is False
    
    @patch('flowlib.providers.mcp.server.provider.flow_registry')
    async def test_register_flows_expose_all(self, mock_registry, provider, mock_server, mock_flows):
        """Test registering all flows."""
        provider._server = mock_server
        # Create a new provider with updated settings since the model is frozen
        updated_settings = provider.settings.model_copy(update={"expose_all_flows": True})
        provider = provider.model_copy(update={"settings": updated_settings})
        provider._server = mock_server  # Re-assign server after model_copy
        mock_registry.get_flow_instances.return_value = mock_flows
        
        await provider._register_flows()
        
        # Should register both flows
        assert mock_server.register_tool.call_count == 2
        # Verify the tools were registered by checking the calls
        call_args_list = mock_server.register_tool.call_args_list
        registered_tools = [call[0][0].name for call in call_args_list]
        assert "flow1" in registered_tools
        assert "flow2" in registered_tools
    
    @patch('flowlib.providers.mcp.server.provider.flow_registry')
    async def test_register_flows_specific_list(self, mock_registry, provider, mock_server, mock_flows):
        """Test registering specific flows."""
        provider._server = mock_server
        # Create a new provider with updated settings since the model is frozen
        updated_settings = provider.settings.model_copy(update={"exposed_flows": ["flow1"]})
        provider = provider.model_copy(update={"settings": updated_settings})
        provider._server = mock_server  # Re-assign server after model_copy
        mock_registry.get_flow_instances.return_value = mock_flows
        
        await provider._register_flows()
        
        # Should register only flow1
        assert mock_server.register_tool.call_count == 1
        # Verify the correct tool was registered
        call_args_list = mock_server.register_tool.call_args_list
        registered_tools = [call[0][0].name for call in call_args_list]
        assert "flow1" in registered_tools
        assert "flow2" not in registered_tools
    
    @patch('flowlib.providers.mcp.server.provider.flow_registry')
    async def test_register_flows_with_exclusions(self, mock_registry, provider, mock_server, mock_flows):
        """Test registering flows with exclusions."""
        provider._server = mock_server
        # Create a new provider with updated settings since the model is frozen
        updated_settings = provider.settings.model_copy(update={"expose_all_flows": True, "exclude_flows": ["flow2"]})
        provider = provider.model_copy(update={"settings": updated_settings})
        provider._server = mock_server  # Re-assign server after model_copy
        mock_registry.get_flow_instances.return_value = mock_flows
        
        await provider._register_flows()
        
        # Should register only flow1 (flow2 excluded)
        assert mock_server.register_tool.call_count == 1
        # Verify the correct tool was registered
        call_args_list = mock_server.register_tool.call_args_list
        registered_tools = [call[0][0].name for call in call_args_list]
        assert "flow1" in registered_tools
        assert "flow2" not in registered_tools
    
    @patch('flowlib.providers.mcp.server.provider.flow_registry')
    async def test_register_flows_not_found(self, mock_registry, provider, mock_server):
        """Test registering non-existent flows."""
        provider._server = mock_server
        # Create a new provider with updated settings since the model is frozen
        updated_settings = provider.settings.model_copy(update={"exposed_flows": ["nonexistent_flow"]})
        provider = provider.model_copy(update={"settings": updated_settings})
        provider._server = mock_server  # Re-assign server after model_copy
        mock_registry.get_flow_instances.return_value = {}  # Empty flows dict
        
        # Should not raise error, just skip missing flows
        await provider._register_flows()
        
        assert mock_server.register_tool.call_count == 0
    
    async def test_register_resources_memory(self, provider, mock_server):
        """Test registering memory resources."""
        provider._server = mock_server
        # Create a new provider with updated settings since the model is frozen
        updated_settings = provider.settings.model_copy(update={"expose_memory": True})
        provider = provider.model_copy(update={"settings": updated_settings})
        provider._server = mock_server  # Re-assign server after model_copy
        
        await provider._register_resources()
        
        # Should register memory resource
        mock_server.register_resource.assert_called()
        # Check all registered resources to find the memory one
        call_args_list = mock_server.register_resource.call_args_list
        registered_uris = [call[0][0].uri for call in call_args_list]
        assert any(uri.startswith("memory://") for uri in registered_uris)
    
    async def test_register_resources_flow_registry(self, provider, mock_server):
        """Test registering flow registry resources."""
        provider._server = mock_server
        # Create a new provider with updated settings since the model is frozen
        updated_settings = provider.settings.model_copy(update={"expose_flow_registry": True})
        provider = provider.model_copy(update={"settings": updated_settings})
        provider._server = mock_server  # Re-assign server after model_copy
        
        await provider._register_resources()
        
        # Should register flow registry resource
        mock_server.register_resource.assert_called()
    
    async def test_register_resources_metrics(self, provider, mock_server):
        """Test registering metrics resources."""
        provider._server = mock_server
        # Create a new provider with updated settings since the model is frozen
        updated_settings = provider.settings.model_copy(update={"expose_metrics": True})
        provider = provider.model_copy(update={"settings": updated_settings})
        provider._server = mock_server  # Re-assign server after model_copy
        
        await provider._register_resources()
        
        # Should register metrics resource
        mock_server.register_resource.assert_called()
    
    async def test_handle_tool_call_success(self, provider, mock_server, mock_flows):
        """Test successful tool call handling."""
        # Set up tool handler
        flow = mock_flows["flow1"]
        flow.run_pipeline = AsyncMock(return_value={"result": "success"})
        handler = AsyncMock()
        handler.handle = AsyncMock(return_value={"result": "success"})
        
        # Set up server with tool
        provider._server = mock_server
        mock_server._tools = {"flow1": {"handler": handler}}
        
        result = await provider._handle_tool_call("flow1", {"input": "test"})
        
        assert result == {"result": "success"}
        handler.handle.assert_called_once_with({"input": "test"})
    
    async def test_handle_tool_call_not_found(self, provider):
        """Test tool call with unknown tool."""
        with pytest.raises(Exception) as exc_info:
            await provider._handle_tool_call("unknown_tool", {})
        
        assert "Tool not found" in str(exc_info.value)
    
    async def test_handle_tool_call_execution_error(self, provider, mock_server, mock_flows):
        """Test tool call with execution error."""
        # Set up handler that will fail
        handler = AsyncMock()
        handler.handle = AsyncMock(side_effect=Exception("Flow failed"))
        
        # Set up server with tool
        provider._server = mock_server
        mock_server._tools = {"flow1": {"handler": handler}}
        
        with pytest.raises(Exception) as exc_info:
            await provider._handle_tool_call("flow1", {"input": "test"})
        
        assert "Tool execution failed" in str(exc_info.value)
    
    async def test_handle_resource_read_memory(self, provider, mock_server):
        """Test reading memory resource."""
        # Set up server with resource handlers
        provider._server = mock_server
        
        # Mock the resource handler
        async def mock_handler():
            return {"content": {"key1": "value1"}, "mimeType": "application/json"}
        
        mock_server._resources = {"memory://working": "mock_resource"}
        mock_server._resource_handlers = {"memory://working": mock_handler}
        
        result = await provider._handle_resource_read("memory://working")
        
        assert result["mimeType"] == "application/json"
        assert "key1" in str(result["content"])
    
    async def test_handle_resource_read_flow_registry(self, provider, mock_server):
        """Test reading flow registry resource."""
        # Set up server with resource handlers
        provider._server = mock_server
        
        # Mock the resource handler
        def mock_handler():
            return {"content": {"flows": ["flow1", "flow2"], "count": 2}, "mimeType": "application/json"}
        
        mock_server._resources = {"flows://registry": "mock_resource"}
        mock_server._resource_handlers = {"flows://registry": mock_handler}
        
        result = await provider._handle_resource_read("flows://registry")
        
        assert result["mimeType"] == "application/json"
        assert "flow1" in str(result["content"])
    
    async def test_handle_resource_read_not_found(self, provider, mock_server):
        """Test reading unknown resource."""
        # Set up server with empty resources
        provider._server = mock_server
        mock_server._resources = {}
        mock_server._resource_handlers = {}
        
        with pytest.raises(MCPError) as exc_info:
            await provider._handle_resource_read("unknown://resource")
        
        assert "Resource not found" in str(exc_info.value)
    
    async def test_is_running_true(self, provider, mock_server):
        """Test is_running check when server is running."""
        provider._server = mock_server
        provider._running = True
        mock_server.is_running.return_value = True
        
        result = await provider.is_running()
        
        assert result is True
    
    async def test_is_running_false(self, provider, mock_server):
        """Test is_running check when server is not running."""
        provider._server = mock_server
        provider._running = False
        mock_server.is_running.return_value = False
        
        result = await provider.is_running()
        
        assert result is False
    
    async def test_is_running_no_server(self, provider):
        """Test is_running check when no server."""
        provider._server = None
        
        result = await provider.is_running()
        
        assert result is False
    
    async def test_get_server_info(self, provider, mock_server):
        """Test getting server information."""
        # Set up provider with mock server
        provider._server = mock_server
        provider._running = True
        
        info = await provider.get_server_info()
        
        assert info["name"] == "test-server"
        assert info["version"] == "1.0.0"
        assert info["transport"] == "stdio"  # Provider uses STDIO transport
        assert info["host"] == "127.0.0.1"
        assert info["port"] == 8081
    
    def test_provider_registration(self):
        """Test that provider is properly registered."""
        # This would be tested in integration tests
        # Here we just verify basic provider functionality
        settings = MCPServerSettings()
        provider = MCPServerProvider(name="test", settings=settings)
        
        assert provider.name == "test"
        assert provider.provider_type == "mcp_server"


@pytest.mark.skipif(not HTTP_AVAILABLE, reason="aiohttp package not available")
@pytest.mark.integration
class TestMCPServerProviderIntegration:
    """Integration tests for MCP server provider.
    
    These tests require aiohttp and test the HTTP server functionality.
    """
    
    @pytest.fixture
    def integration_settings(self):
        """Create integration test settings."""
        return MCPServerSettings(
            server_name="integration-server",
            host="127.0.0.1",
            port=8082,
            transport=MCPTransport.SSE,
            expose_all_flows=True,
            expose_memory=True,
            expose_flow_registry=True
        )
    
    @pytest.fixture
    async def provider(self, integration_settings):
        """Create and initialize provider for integration tests."""
        provider = MCPServerProvider(name="integration_server", settings=integration_settings)
        
        try:
            # Note: This would start a real server
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
        assert integration_settings.server_name == "integration-server"
        assert integration_settings.host == "127.0.0.1"
        assert integration_settings.port == 8082
        assert integration_settings.transport == MCPTransport.SSE
        assert integration_settings.expose_all_flows is True
    
    def test_provider_creation_integration(self, integration_settings):
        """Test provider creation with real settings."""
        provider = MCPServerProvider(name="integration_test", settings=integration_settings)
        
        assert provider.name == "integration_test"
        assert provider.settings.server_name == "integration-server"
        assert not provider.initialized
    
    async def test_flow_exposure_integration(self, integration_settings):
        """Test flow exposure configuration."""
        # Test selective flow exposure
        selective_settings = MCPServerSettings(
            expose_all_flows=False,
            exposed_flows=["flow1", "flow2"],
            exclude_flows=["internal_flow"]
        )
        
        provider = MCPServerProvider(name="selective_test", settings=selective_settings)
        
        assert provider.settings.expose_all_flows is False
        assert "flow1" in provider.settings.exposed_flows
        assert "internal_flow" in provider.settings.exclude_flows
    
    def test_resource_exposure_integration(self, integration_settings):
        """Test resource exposure configuration."""
        # Test different resource exposure combinations
        resource_settings = MCPServerSettings(
            expose_memory=True,
            expose_flow_registry=False,
            expose_metrics=True
        )
        
        provider = MCPServerProvider(name="resource_test", settings=resource_settings)
        
        assert provider.settings.expose_memory is True
        assert provider.settings.expose_flow_registry is False
        assert provider.settings.expose_metrics is True