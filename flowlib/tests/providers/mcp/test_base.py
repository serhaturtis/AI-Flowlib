"""Tests for MCP base classes and models."""
import pytest
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field, ValidationError
from unittest.mock import Mock, patch

from flowlib.providers.mcp.base import (
    MCPTransport,
    MCPMessageType,
    MCPToolInputSchema,
    MCPTool,
    MCPMessage,
    MCPRequest,
    MCPResponse,
    MCPNotification,
    MCPError,
    MCPProvider,
    MCPProviderSettings,
)


class MockPydanticModel(BaseModel):
    """Mock Pydantic model for schema generation."""
    name: str
    age: int = Field(default=25, ge=0, le=150)
    email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
    tags: List[str] = Field(default_factory=list)


class TestMCPEnums:
    """Test MCP enumeration classes."""
    
    def test_mcp_transport_values(self):
        """Test MCP transport protocol values."""
        assert MCPTransport.STDIO == "stdio"
        assert MCPTransport.SSE == "sse"
        assert MCPTransport.WEBSOCKET == "websocket"
    
    def test_mcp_transport_enumeration(self):
        """Test MCP transport enumeration."""
        all_transports = list(MCPTransport)
        assert len(all_transports) == 3
        assert MCPTransport.STDIO in all_transports
        assert MCPTransport.SSE in all_transports
        assert MCPTransport.WEBSOCKET in all_transports
    
    def test_mcp_message_type_values(self):
        """Test MCP message type values."""
        assert MCPMessageType.REQUEST == "request"
        assert MCPMessageType.RESPONSE == "response"
        assert MCPMessageType.NOTIFICATION == "notification"
    
    def test_mcp_message_type_enumeration(self):
        """Test MCP message type enumeration."""
        all_types = list(MCPMessageType)
        assert len(all_types) == 3
        assert MCPMessageType.REQUEST in all_types
        assert MCPMessageType.RESPONSE in all_types
        assert MCPMessageType.NOTIFICATION in all_types


class TestMCPToolInputSchema:
    """Test MCP tool input schema."""
    
    def test_default_schema(self):
        """Test default tool input schema."""
        schema = MCPToolInputSchema()
        
        assert schema.type == "object"
        assert schema.properties == {}
        assert schema.required == []
    
    def test_custom_schema(self):
        """Test custom tool input schema."""
        properties = {
            "name": {"type": "string"},
            "count": {"type": "integer", "minimum": 0}
        }
        required = ["name"]
        
        schema = MCPToolInputSchema(
            type="object",
            properties=properties,
            required=required
        )
        
        assert schema.type == "object"
        assert schema.properties == properties
        assert schema.required == required
    
    def test_from_pydantic_model(self):
        """Test schema generation from Pydantic model."""
        schema = MCPToolInputSchema.from_pydantic_model(MockPydanticModel)
        
        assert schema.type == "object"
        assert "name" in schema.properties
        assert "age" in schema.properties
        assert "email" in schema.properties
        assert "tags" in schema.properties
        
        # Check required fields
        assert "name" in schema.required
        assert "age" not in schema.required  # Has default
        assert "email" in schema.required
        assert "tags" not in schema.required  # Has default
    
    def test_from_pydantic_model_properties(self):
        """Test detailed properties from Pydantic model."""
        schema = MCPToolInputSchema.from_pydantic_model(MockPydanticModel)
        
        # Check name property
        name_prop = schema.properties["name"]
        assert name_prop["type"] == "string"
        
        # Check age property with constraints
        age_prop = schema.properties["age"]
        assert age_prop["type"] == "integer"
        assert age_prop.get("minimum") == 0
        assert age_prop.get("maximum") == 150
        
        # Check email property with pattern
        email_prop = schema.properties["email"]
        assert email_prop["type"] == "string"
        assert "pattern" in email_prop
        
        # Check tags array property
        tags_prop = schema.properties["tags"]
        assert tags_prop["type"] == "array"
        assert tags_prop["items"]["type"] == "string"


class TestMCPTool:
    """Test MCP tool definition."""
    
    def test_simple_tool(self):
        """Test simple tool definition."""
        schema = MCPToolInputSchema(
            properties={"text": {"type": "string"}},
            required=["text"]
        )
        
        tool = MCPTool(
            name="echo",
            description="Echo the input text",
            input_schema=schema
        )
        
        assert tool.name == "echo"
        assert tool.description == "Echo the input text"
        assert tool.input_schema == schema
    
    def test_tool_with_pydantic_schema(self):
        """Test tool with Pydantic-generated schema."""
        schema = MCPToolInputSchema.from_pydantic_model(MockPydanticModel)
        
        tool = MCPTool(
            name="create_user",
            description="Create a new user",
            input_schema=schema
        )
        
        assert tool.name == "create_user"
        assert tool.description == "Create a new user"
        assert "name" in tool.input_schema.properties
        assert "age" in tool.input_schema.properties
    
    def test_tool_serialization(self):
        """Test tool serialization to JSON."""
        schema = MCPToolInputSchema(
            properties={"value": {"type": "number"}},
            required=["value"]
        )
        
        tool = MCPTool(
            name="calculate",
            description="Perform calculation",
            input_schema=schema
        )
        
        # Test dict conversion
        tool_dict = tool.model_dump()
        assert tool_dict["name"] == "calculate"
        assert tool_dict["description"] == "Perform calculation"
        assert tool_dict["input_schema"]["type"] == "object"
        
        # Test JSON serialization
        json_str = tool.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["name"] == "calculate"
    
    def test_tool_validation(self):
        """Test tool validation."""
        # Valid tool
        schema = MCPToolInputSchema()
        tool = MCPTool(
            name="valid_tool",
            description="A valid tool",
            input_schema=schema
        )
        assert tool.name == "valid_tool"
        
        # Missing required fields
        with pytest.raises(ValidationError):
            MCPTool(name="invalid")  # Missing description and schema


class TestMCPMessage:
    """Test MCP message base class."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        message = MCPMessage(
            type=MCPMessageType.NOTIFICATION,
            method="test/notification"
        )
        
        assert message.type == MCPMessageType.NOTIFICATION
        assert message.method == "test/notification"
        assert message.id is None  # Notifications don't have IDs
    
    def test_message_with_id(self):
        """Test message with ID."""
        message = MCPMessage(
            type=MCPMessageType.REQUEST,
            method="test/request",
            id="req-123"
        )
        
        assert message.type == MCPMessageType.REQUEST
        assert message.method == "test/request"
        assert message.id == "req-123"
    
    def test_message_with_params(self):
        """Test message with parameters."""
        params = {"key": "value", "number": 42}
        
        message = MCPMessage(
            type=MCPMessageType.REQUEST,
            method="test/with_params",
            id="req-456",
            params=params
        )
        
        assert message.params == params
    
    def test_message_serialization(self):
        """Test message serialization."""
        message = MCPMessage(
            type=MCPMessageType.REQUEST,
            method="test/serialize",
            id="req-789",
            params={"data": "test"}
        )
        
        # Test dict conversion
        message_dict = message.model_dump(exclude_none=True)
        assert message_dict["type"] == "request"
        assert message_dict["method"] == "test/serialize"
        assert message_dict["id"] == "req-789"
        assert message_dict["params"]["data"] == "test"
        
        # Test JSON serialization
        json_str = message.model_dump_json(exclude_none=True)
        parsed = json.loads(json_str)
        assert parsed["type"] == "request"


class TestMCPRequest:
    """Test MCP request message."""
    
    def test_request_creation(self):
        """Test request message creation."""
        request = MCPRequest(
            method="tools/call",
            id="call-123",
            params={"name": "echo", "arguments": {"text": "hello"}}
        )
        
        assert request.type == MCPMessageType.REQUEST
        assert request.method == "tools/call"
        assert request.id == "call-123"
        assert request.params["name"] == "echo"
    
    def test_request_without_id(self):
        """Test request validation without ID."""
        # Requests should have IDs
        with pytest.raises(ValidationError):
            MCPRequest(method="test/method")
    
    def test_request_inheritance(self):
        """Test that MCPRequest inherits from MCPMessage."""
        request = MCPRequest(method="test", id="123")
        assert isinstance(request, MCPMessage)


class TestMCPResponse:
    """Test MCP response message."""
    
    def test_response_creation(self):
        """Test response message creation."""
        response = MCPResponse(
            id="call-123",
            result={"output": "hello"}
        )
        
        assert response.type == MCPMessageType.RESPONSE
        assert response.id == "call-123"
        assert response.result["output"] == "hello"
        assert response.error is None
    
    def test_response_with_error(self):
        """Test response message with error."""
        error = MCPError(
            code=-1,
            message="Tool not found"
        )
        
        response = MCPResponse(
            id="call-456",
            error=error.to_dict()
        )
        
        assert response.id == "call-456"
        assert response.result is None
        assert response.error["code"] == -1
        assert response.error["message"] == "Tool not found"
    
    def test_response_validation(self):
        """Test response validation."""
        # Valid response with result
        response = MCPResponse(id="123", result={"data": "test"})
        assert response.result is not None
        
        # Valid response with error
        error = MCPError(code=-1, message="Error")
        response = MCPResponse(id="456", error=error.to_dict())
        assert response.error is not None
        
        # Invalid response without ID
        with pytest.raises(ValidationError):
            MCPResponse(result={"data": "test"})


class TestMCPNotification:
    """Test MCP notification message."""
    
    def test_notification_creation(self):
        """Test notification message creation."""
        notification = MCPNotification(
            method="server/initialized",
            params={"version": "1.0"}
        )
        
        assert notification.type == MCPMessageType.NOTIFICATION
        assert notification.method == "server/initialized"
        assert notification.params["version"] == "1.0"
        assert notification.id is None
    
    def test_notification_without_params(self):
        """Test notification without parameters."""
        notification = MCPNotification(method="server/shutdown")
        
        assert notification.method == "server/shutdown"
        assert notification.params is None
    
    def test_notification_inheritance(self):
        """Test that MCPNotification inherits from MCPMessage."""
        notification = MCPNotification(method="test")
        assert isinstance(notification, MCPMessage)


class TestMCPError:
    """Test MCP error model."""
    
    def test_error_creation(self):
        """Test error creation."""
        error = MCPError(
            code=-32601,
            message="Method not found",
            data={"method": "unknown/method"}
        )
        
        assert error.code == -32601
        assert error.message == "Method not found"
        assert error.data["method"] == "unknown/method"
    
    def test_error_without_data(self):
        """Test error without additional data."""
        error = MCPError(
            code=-1,
            message="Generic error"
        )
        
        assert error.code == -1
        assert error.message == "Generic error"
        assert error.data is None
    
    def test_error_validation(self):
        """Test error validation."""
        # Valid error
        error = MCPError(code=-1, message="Error")
        assert error.code == -1
        
        # Missing required fields should raise TypeError
        with pytest.raises(TypeError):
            MCPError(code=-1)  # Missing message


class TestMCPProviderSettings:
    """Test MCP provider settings."""
    
    def test_default_settings(self):
        """Test default MCP provider settings."""
        settings = MCPProviderSettings()
        
        assert settings.transport == MCPTransport.STDIO
        assert settings.command is None
        assert settings.args == []
        assert settings.env == {}
        assert settings.timeout == 30.0
        assert settings.server_url is None
        assert settings.auth_token is None
    
    def test_stdio_settings(self):
        """Test STDIO transport settings."""
        settings = MCPProviderSettings(
            transport=MCPTransport.STDIO,
            command="python",
            args=["-m", "my_mcp_server"],
            env={"PYTHONPATH": "/custom/path"},
            timeout=60.0
        )
        
        assert settings.transport == MCPTransport.STDIO
        assert settings.command == "python"
        assert settings.args == ["-m", "my_mcp_server"]
        assert settings.env["PYTHONPATH"] == "/custom/path"
        assert settings.timeout == 60.0
    
    def test_sse_settings(self):
        """Test SSE transport settings."""
        settings = MCPProviderSettings(
            transport=MCPTransport.SSE,
            server_url="https://api.example.com/mcp",
            auth_token="bearer_token_123",
            timeout=45.0
        )
        
        assert settings.transport == MCPTransport.SSE
        assert settings.server_url == "https://api.example.com/mcp"
        assert settings.auth_token == "bearer_token_123"
        assert settings.timeout == 45.0
    
    def test_websocket_settings(self):
        """Test WebSocket transport settings."""
        settings = MCPProviderSettings(
            transport=MCPTransport.WEBSOCKET,
            server_url="wss://mcp.example.com/ws",
            auth_token="ws_token_456"
        )
        
        assert settings.transport == MCPTransport.WEBSOCKET
        assert settings.server_url == "wss://mcp.example.com/ws"
        assert settings.auth_token == "ws_token_456"


class TestMCPProvider:
    """Test MCP provider base class."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return MCPProviderSettings(
            transport=MCPTransport.STDIO,
            command="python",
            args=["-m", "test_server"],
            timeout=30.0
        )
    
    @pytest.fixture
    def provider(self, settings):
        """Create test provider."""
        return MCPProvider(name="test_mcp", provider_type="mcp", settings=settings)
    
    def test_provider_initialization(self, settings):
        """Test provider initialization."""
        provider = MCPProvider(name="test_provider", provider_type="mcp", settings=settings)
        
        assert provider.name == "test_provider"
        assert provider.provider_type == "mcp"
        assert provider.settings == settings
        assert provider._transport is None
        assert provider._tools == {}
    
    def test_provider_inheritance(self, provider):
        """Test that MCPProvider inherits from Provider."""
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, Provider)
    
    async def test_initialize_abstract_method(self, provider):
        """Test that initialize is abstract in base class."""
        with pytest.raises(NotImplementedError):
            await provider.initialize()
    
    async def test_shutdown_abstract_method(self, provider):
        """Test that shutdown is abstract in base class."""
        with pytest.raises(NotImplementedError):
            await provider.shutdown()
    
    async def test_call_tool_abstract_method(self, provider):
        """Test that call_tool is abstract in base class."""
        with pytest.raises(NotImplementedError):
            await provider.call_tool("test_tool", {"param": "value"})
    
    async def test_list_tools_abstract_method(self, provider):
        """Test that list_tools is abstract in base class."""
        with pytest.raises(NotImplementedError):
            await provider.list_tools()
    
    def test_create_request_message(self, provider):
        """Test request message creation."""
        request = provider._create_request("test/method", "req-123", {"param": "value"})
        
        assert isinstance(request, MCPRequest)
        assert request.method == "test/method"
        assert request.id == "req-123"
        assert request.params["param"] == "value"
    
    def test_create_response_message(self, provider):
        """Test response message creation."""
        # Success response
        response = provider._create_response("req-123", result={"data": "test"})
        
        assert isinstance(response, MCPResponse)
        assert response.id == "req-123"
        assert response.result["data"] == "test"
        assert response.error is None
        
        # Error response
        error = MCPError(code=-1, message="Test error")
        error_response = provider._create_response("req-456", error=error)
        
        assert error_response.id == "req-456"
        assert error_response.result is None
        assert error_response.error["message"] == "Test error"
    
    def test_create_notification_message(self, provider):
        """Test notification message creation."""
        notification = provider._create_notification("test/event", {"data": "value"})
        
        assert isinstance(notification, MCPNotification)
        assert notification.method == "test/event"
        assert notification.params["data"] == "value"
        assert notification.id is None


class TestMCPIntegration:
    """Integration tests for MCP components."""
    
    def test_tool_to_mcp_format(self):
        """Test converting tool to MCP format."""
        schema = MCPToolInputSchema.from_pydantic_model(MockPydanticModel)
        tool = MCPTool(
            name="create_user",
            description="Create a new user profile",
            input_schema=schema
        )
        
        # Convert to dictionary for MCP protocol
        tool_dict = tool.model_dump()
        
        assert tool_dict["name"] == "create_user"
        assert tool_dict["description"] == "Create a new user profile"
        assert tool_dict["input_schema"]["type"] == "object"
        assert "name" in tool_dict["input_schema"]["properties"]
        assert "age" in tool_dict["input_schema"]["properties"]
    
    def test_message_flow_simulation(self):
        """Test simulated MCP message flow."""
        # Client sends request
        request = MCPRequest(
            method="tools/call",
            id="call-1",
            params={
                "name": "create_user",
                "arguments": {
                    "name": "John Doe",
                    "age": 30,
                    "email": "john@example.com"
                }
            }
        )
        
        # Server processes and sends response
        response = MCPResponse(
            id="call-1",
            result={
                "success": True,
                "user_id": "user-123"
            }
        )
        
        # Verify message flow
        assert request.id == response.id
        assert request.method == "tools/call"
        assert response.result["success"] is True
    
    def test_error_handling_flow(self):
        """Test error handling in MCP flow."""
        # Client sends invalid request
        request = MCPRequest(
            method="tools/call",
            id="call-2",
            params={
                "name": "nonexistent_tool",
                "arguments": {}
            }
        )
        
        # Server responds with error
        error = MCPError(
            code=-32601,
            message="Tool not found",
            data={"tool_name": "nonexistent_tool"}
        )
        
        response = MCPResponse(
            id="call-2",
            error=error.to_dict()
        )
        
        # Verify error handling
        assert request.id == response.id
        assert response.result is None
        assert response.error["code"] == -32601
        assert response.error["data"]["tool_name"] == "nonexistent_tool"