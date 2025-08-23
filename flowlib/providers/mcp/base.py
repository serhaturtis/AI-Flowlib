"""Base classes and models for MCP integration."""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Protocol
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from flowlib.providers.core.base import Provider, ProviderSettings

logger = logging.getLogger(__name__)


class MCPTransport(str, Enum):
    """MCP transport protocols."""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


class MCPMessageType(str, Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response" 
    NOTIFICATION = "notification"


class PydanticSchemaData(BaseModel):
    """Pydantic schema data model."""
    model_config = ConfigDict(extra="forbid")
    
    type: str = Field(default="object", description="Schema type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Schema properties")
    required: List[str] = Field(default_factory=list, description="Required fields")


class MCPToolInputSchema(BaseModel):
    """Schema for MCP tool input."""
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    
    @classmethod
    def from_pydantic_model(cls, model_class) -> 'MCPToolInputSchema':
        """Create schema from Pydantic model."""
        schema_raw = model_class.model_json_schema()
        # Filter schema to only include fields expected by PydanticSchemaData
        filtered_schema = {
            "type": schema_raw["type"] if "type" in schema_raw else "object",
            "properties": schema_raw["properties"] if "properties" in schema_raw else {},
            "required": schema_raw["required"] if "required" in schema_raw else []
        }
        schema_data = PydanticSchemaData.model_validate(filtered_schema)
        return cls(
            type=schema_data.type,
            properties=schema_data.properties,
            required=schema_data.required
        )


class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str
    description: str
    input_schema: MCPToolInputSchema
    
    model_config = ConfigDict(extra="allow")


class MCPResource(BaseModel):
    """MCP resource definition."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    
    model_config = ConfigDict(extra="allow")


class MCPCapabilities(BaseModel):
    """MCP server/client capabilities."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    tools: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    experimental: Optional[Dict[str, Any]] = None


class MCPParams(BaseModel):
    """Base class for MCP method parameters."""
    pass


class ToolCallParamsData(BaseModel):
    """Tool call parameters data model."""
    model_config = ConfigDict(extra="forbid")
    
    name: Optional[str] = Field(default=None, description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ResourceReadParamsData(BaseModel):
    """Resource read parameters data model."""
    model_config = ConfigDict(extra="forbid")
    
    uri: Optional[str] = Field(default=None, description="Resource URI")


class MCPInitializeParams(MCPParams):
    """Parameters for initialize method."""
    protocolVersion: str = "2024-11-05"
    capabilities: MCPCapabilities
    clientInfo: Dict[str, str]


class MCPToolCallParams(MCPParams):
    """Parameters for tools/call method."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class MCPResourceReadParams(MCPParams):
    """Parameters for resources/read method."""
    uri: str


class MCPResult(BaseModel):
    """Base class for MCP method results."""
    pass


class MCPInitializeResult(MCPResult):
    """Result from initialize method."""
    protocolVersion: str
    capabilities: MCPCapabilities
    serverInfo: Dict[str, str]


class MCPToolsListResult(MCPResult):
    """Result from tools/list method."""
    tools: List[MCPTool] = Field(default_factory=list)


class MCPResourcesListResult(MCPResult):
    """Result from resources/list method."""
    resources: List[MCPResource] = Field(default_factory=list)


class MCPToolCallResult(MCPResult):
    """Result from tools/call method."""
    content: List[Dict[str, Any]] = Field(default_factory=list)
    isError: Optional[bool] = None


class MCPResourceReadResult(MCPResult):
    """Result from resources/read method."""
    contents: List[Dict[str, Any]] = Field(default_factory=list)


class MCPMessage(BaseModel):
    """Base MCP message."""
    id: Optional[str] = None
    type: MCPMessageType
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    
    model_config = {"arbitrary_types_allowed": True}


class MCPRequest(MCPMessage):
    """MCP request message."""
    type: MCPMessageType = MCPMessageType.REQUEST
    method: str
    id: str  # Required for requests
    params: Optional[Dict[str, Any]] = None


class MCPResponse(MCPMessage):
    """MCP response message."""
    type: MCPMessageType = MCPMessageType.RESPONSE
    id: str  # Required for responses
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPNotification(MCPMessage):
    """MCP notification message."""
    type: MCPMessageType = MCPMessageType.NOTIFICATION
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPConnection(Protocol):
    """Protocol for MCP connections."""
    
    async def send(self, message: MCPMessage) -> None:
        """Send message to remote party."""
        ...
    
    async def receive(self) -> MCPMessage:
        """Receive message from remote party."""
        ...
    
    async def close(self) -> None:
        """Close the connection."""
        ...


class MCPError(Exception):
    """Base MCP error."""
    
    def __init__(self, message: str, code: int = -1, data: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for MCP messages."""
        result = {
            "code": self.code,
            "message": self.message
        }
        if self.data is not None:
            result["data"] = self.data
        return result


class MCPToolNotFoundError(MCPError):
    """Tool not found error."""
    
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' not found", code=-32601)
        self.tool_name = tool_name


class MCPResourceNotFoundError(MCPError):
    """Resource not found error."""
    
    def __init__(self, resource_uri: str):
        super().__init__(f"Resource '{resource_uri}' not found", code=-32601)
        self.resource_uri = resource_uri


class MCPConnectionError(MCPError):
    """Connection error."""
    
    def __init__(self, message: str):
        super().__init__(f"Connection error: {message}", code=-32003)


class BaseMCPClient:
    """Base MCP client implementation."""
    
    def __init__(self, name: str = "flowlib-client"):
        self.name = name
        self.connection: Optional[MCPConnection] = None
        self.capabilities = MCPCapabilities()
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._request_id = 0
        
    async def initialize(self, connection: MCPConnection) -> None:
        """Initialize client with connection."""
        self.connection = connection
        
        # Perform handshake
        await self._handshake()
        
        # List available tools and resources
        await self._discover_capabilities()
        
        logger.info(f"MCP client '{self.name}' initialized with {len(self._tools)} tools and {len(self._resources)} resources")
    
    async def _handshake(self) -> None:
        """Perform MCP handshake."""
        request = MCPMessage(
            id=str(self._get_next_id()),
            type=MCPMessageType.REQUEST,
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": self.capabilities.model_dump(exclude_none=True),
                "clientInfo": {
                    "name": self.name,
                    "version": "1.0.0"
                }
            }
        )
        
        await self.connection.send(request)
        response = await self.connection.receive()
        
        if response.error:
            raise MCPError(f"Handshake failed: {response.error}")
            
        logger.debug("MCP handshake completed")
    
    async def _discover_capabilities(self) -> None:
        """Discover server capabilities."""
        # List tools
        try:
            tools_response = await self._send_request("tools/list")
            if tools_response.result and "tools" in tools_response.result:
                for tool_data in tools_response.result["tools"]:
                    tool = MCPTool(**tool_data)
                    self._tools[tool.name] = tool
        except Exception as e:
            logger.warning(f"Failed to list tools: {e}")
        
        # List resources
        try:
            resources_response = await self._send_request("resources/list")
            if resources_response.result and "resources" in resources_response.result:
                for resource_data in resources_response.result["resources"]:
                    resource = MCPResource(**resource_data)
                    self._resources[resource.uri] = resource
        except Exception as e:
            logger.warning(f"Failed to list resources: {e}")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the server."""
        if name not in self._tools:
            raise MCPToolNotFoundError(name)
        
        response = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        
        if response.error:
            raise MCPError(f"Tool call failed: {response.error}")
            
        return response.result
    
    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the server."""
        if uri not in self._resources:
            raise MCPResourceNotFoundError(uri)
        
        response = await self._send_request("resources/read", {
            "uri": uri
        })
        
        if response.error:
            raise MCPError(f"Resource read failed: {response.error}")
            
        return response.result
    
    async def _send_request(self, method: str, params: Optional[Dict] = None) -> MCPMessage:
        """Send request and wait for response."""
        if not self.connection:
            raise MCPConnectionError("Not connected")
        
        request = MCPMessage(
            id=str(self._get_next_id()),
            type=MCPMessageType.REQUEST,
            method=method,
            params=params
        )
        
        await self.connection.send(request)
        response = await self.connection.receive()
        
        return response
    
    def _get_next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id
    
    def get_available_tools(self) -> Dict[str, MCPTool]:
        """Get available tools."""
        return self._tools.copy()
    
    def get_available_resources(self) -> Dict[str, MCPResource]:
        """Get available resources."""
        return self._resources.copy()
    
    async def close(self) -> None:
        """Close the connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None


class BaseMCPServer:
    """Base MCP server implementation."""
    
    def __init__(self, name: str = "flowlib-server", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.capabilities = MCPCapabilities(
            tools={},
            resources={}
        )
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._tool_handlers: Dict[str, callable] = {}
        self._resource_handlers: Dict[str, callable] = {}
        
    def register_tool(self, tool: MCPTool, handler: callable) -> None:
        """Register a tool with its handler."""
        self._tools[tool.name] = tool
        self._tool_handlers[tool.name] = handler
        logger.debug(f"Registered MCP tool: {tool.name}")
    
    def register_resource(self, resource: MCPResource, handler: callable) -> None:
        """Register a resource with its handler."""
        self._resources[resource.uri] = resource
        self._resource_handlers[resource.uri] = handler
        logger.debug(f"Registered MCP resource: {resource.uri}")
    
    async def handle_request(self, request: MCPMessage) -> MCPMessage:
        """Handle incoming MCP request."""
        try:
            if request.method == "initialize":
                return await self._handle_initialize(request)
            elif request.method == "tools/list":
                return await self._handle_list_tools(request)
            elif request.method == "tools/call":
                return await self._handle_call_tool(request)
            elif request.method == "resources/list":
                return await self._handle_list_resources(request)
            elif request.method == "resources/read":
                return await self._handle_read_resource(request)
            else:
                return MCPMessage(
                    id=request.id,
                    type=MCPMessageType.RESPONSE,
                    error={"code": -32601, "message": f"Method not found: {request.method}"}
                )
        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return MCPMessage(
                id=request.id,
                type=MCPMessageType.RESPONSE,
                error={"code": -32603, "message": f"Internal error: {str(e)}"}
            )
    
    async def _handle_initialize(self, request: MCPMessage) -> MCPMessage:
        """Handle initialize request."""
        return MCPMessage(
            id=request.id,
            type=MCPMessageType.RESPONSE,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": self.capabilities.model_dump(exclude_none=True),
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }
        )
    
    async def _handle_list_tools(self, request: MCPMessage) -> MCPMessage:
        """Handle tools/list request."""
        tools = [tool.model_dump() for tool in self._tools.values()]
        return MCPMessage(
            id=request.id,
            type=MCPMessageType.RESPONSE,
            result={"tools": tools}
        )
    
    async def _handle_call_tool(self, request: MCPMessage) -> MCPMessage:
        """Handle tools/call request."""
        params = request.params or {}
        params_data = ToolCallParamsData.model_validate(params)
        tool_name = params_data.name
        arguments = params_data.arguments
        
        if tool_name not in self._tool_handlers:
            return MCPMessage(
                id=request.id,
                type=MCPMessageType.RESPONSE,
                error={"code": -32601, "message": f"Tool not found: {tool_name}"}
            )
        
        try:
            handler = self._tool_handlers[tool_name]
            result = await handler(arguments)
            
            return MCPMessage(
                id=request.id,
                type=MCPMessageType.RESPONSE,
                result=result
            )
        except Exception as e:
            return MCPMessage(
                id=request.id,
                type=MCPMessageType.RESPONSE,
                error={"code": -32603, "message": f"Tool execution failed: {str(e)}"}
            )
    
    async def _handle_list_resources(self, request: MCPMessage) -> MCPMessage:
        """Handle resources/list request."""
        resources = [resource.model_dump() for resource in self._resources.values()]
        return MCPMessage(
            id=request.id,
            type=MCPMessageType.RESPONSE,
            result={"resources": resources}
        )
    
    async def _handle_read_resource(self, request: MCPMessage) -> MCPMessage:
        """Handle resources/read request."""
        params = request.params or {}
        params_data = ResourceReadParamsData.model_validate(params)
        uri = params_data.uri
        
        if uri not in self._resource_handlers:
            return MCPMessage(
                id=request.id,
                type=MCPMessageType.RESPONSE,
                error={"code": -32601, "message": f"Resource not found: {uri}"}
            )
        
        try:
            handler = self._resource_handlers[uri]
            result = await handler()
            
            return MCPMessage(
                id=request.id,
                type=MCPMessageType.RESPONSE,
                result=result
            )
        except Exception as e:
            return MCPMessage(
                id=request.id,
                type=MCPMessageType.RESPONSE,
                error={"code": -32603, "message": f"Resource read failed: {str(e)}"}
            )
    
    def stop(self) -> None:
        """Stop the MCP server."""
        # Clear all registered tools and resources
        self._tools.clear()
        self._resources.clear()
        self._tool_handlers.clear()
        self._resource_handlers.clear()
        logger.info(f"MCP server '{self.name}' stopped")


class MCPProviderSettings(BaseModel):
    """Settings for MCP provider."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    transport: MCPTransport = MCPTransport.STDIO
    host: Optional[str] = None
    port: Optional[int] = None
    path: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    
    # STDIO transport settings
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    
    # Network transport settings  
    server_url: Optional[str] = None
    auth_token: Optional[str] = None


class MCPProvider(Provider):
    """Base MCP provider class."""
    
    def __init__(self, name: str = "mcp", provider_type: str = "mcp", settings: Optional[MCPProviderSettings] = None):
        # Initialize parent Provider
        settings_obj = settings or MCPProviderSettings()
        super().__init__(name=name, provider_type=provider_type, settings=settings_obj)
        self._transport = None
        self._tools: Dict[str, MCPTool] = {}
        self._client: Optional[BaseMCPClient] = None
        self._server: Optional[BaseMCPServer] = None
        
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        raise NotImplementedError("Subclasses must implement initialize()")
    
    async def shutdown(self) -> None:
        """Shutdown the provider."""
        raise NotImplementedError("Subclasses must implement shutdown()")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool."""
        raise NotImplementedError("Subclasses must implement call_tool()")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        raise NotImplementedError("Subclasses must implement list_tools()")
    
    async def get_client(self) -> BaseMCPClient:
        """Get MCP client instance."""
        if not self._client:
            self._client = BaseMCPClient()
        return self._client
    
    async def get_server(self) -> BaseMCPServer:
        """Get MCP server instance."""
        if not self._server:
            self._server = BaseMCPServer()
        return self._server
    
    def _create_request(self, method: str, id: str, params: Optional[Dict[str, Any]] = None) -> MCPRequest:
        """Create an MCP request message."""
        return MCPRequest(method=method, id=id, params=params)
    
    def _create_response(self, id: str, result: Optional[Dict[str, Any]] = None, error: Optional[MCPError] = None) -> MCPResponse:
        """Create an MCP response message."""
        error_dict = error.to_dict() if error else None
        return MCPResponse(id=id, result=result, error=error_dict)
    
    def _create_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> MCPNotification:
        """Create an MCP notification message."""
        return MCPNotification(method=method, params=params)