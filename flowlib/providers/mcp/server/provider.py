"""MCP Server Provider implementation."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union, cast
from pydantic import Field

# Optional HTTP dependencies
try:
    from aiohttp import web, WSMsgType
    import aiohttp_cors  # type: ignore[import-not-found]
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

from ..base import Provider, ProviderSettings
from flowlib.providers.core.decorators import provider
# Removed ProviderType import - using config-driven provider access
from flowlib.providers.mcp.base import (
    BaseMCPServer, MCPTool, MCPResource, MCPTransport,
    MCPToolInputSchema, MCPMessage, MCPError
)
from flowlib.flows.base import Flow
from flowlib.flows.registry.registry import flow_registry
from flowlib.core.context.context import Context

logger = logging.getLogger(__name__)


def get_memory_content() -> Dict[str, Any]:
    """Get current memory content for resource exposure."""
    # This is a placeholder function that would connect to actual memory system
    return {
        "type": "working_memory",
        "contents": "memory data",
        "timestamp": str(asyncio.get_event_loop().time())
    }


class MCPServerSettings(ProviderSettings):
    """Settings for MCP server provider."""
    server_name: str = Field("flowlib-server", description="Name of the MCP server")
    server_version: str = Field("1.0.0", description="Version of the MCP server")
    host: str = Field("localhost", description="Host to bind server to")
    port: int = Field(8080, description="Port to bind server to")
    transport: MCPTransport = Field(default=MCPTransport.SSE, description="MCP transport type: STDIO or SSE")
    
    # Flow exposure settings
    expose_all_flows: bool = Field(False, description="Expose all flows as tools")
    exposed_flows: List[str] = Field(default_factory=list, description="Specific flows to expose")
    exclude_flows: List[str] = Field(default_factory=list, description="Flows to exclude from exposure")
    
    # Resource exposure settings
    expose_memory: bool = Field(True, description="Expose memory as resources")
    expose_flow_registry: bool = Field(True, description="Expose flow registry as resources")
    expose_metrics: bool = Field(False, description="Expose performance metrics")


class FlowToolHandler:
    """Handler for flow-based MCP tools."""
    
    def __init__(self, flow_instance: Flow, flow_name: Optional[str] = None):
        self.flow = flow_instance
        # Ensure flow_name is always a string
        if flow_name:
            self.flow_name = flow_name
        else:
            # Try to get name from flow instance, fallback to 'unknown_flow'
            instance_name = getattr(flow_instance, '__name__', None)
            self.flow_name = str(instance_name) if instance_name else 'unknown_flow'
        
    def create_tool(self) -> MCPTool:
        """Create an MCP tool from the flow."""
        # Get flow metadata
        description = "No description available"
        input_schema = MCPToolInputSchema()
        
        if hasattr(self.flow, '__flow_metadata__'):
            metadata = self.flow.__flow_metadata__
            if 'description' in metadata:
                description = metadata['description']
        elif hasattr(self.flow, 'get_input_schema'):
            # Try to get description from flow if it has one
            if hasattr(self.flow, 'description'):
                description = self.flow.description
        
        # Try to extract input schema from pipeline method or flow method
        if hasattr(self.flow, 'get_input_schema'):
            # If flow has get_input_schema method, use it
            try:
                schema_dict = self.flow.get_input_schema()
                if schema_dict:
                    input_schema = MCPToolInputSchema(**schema_dict)
            except Exception as e:
                logger.warning(f"Failed to get schema from flow.get_input_schema(): {e}")
        elif hasattr(self.flow, 'run_pipeline'):
            # Try to extract input schema from pipeline method
            pipeline_method = getattr(self.flow, 'run_pipeline')
            if hasattr(pipeline_method, '__annotations__'):
                for param_name, param_type in pipeline_method.__annotations__.items():
                    if param_name != 'return' and hasattr(param_type, 'model_json_schema'):
                        try:
                            input_schema = MCPToolInputSchema.from_pydantic_model(param_type)
                            break
                        except Exception as e:
                            logger.warning(f"Failed to extract schema from {param_type}: {e}")
        
        return MCPTool(
            name=self.flow_name,
            description=description,
            input_schema=input_schema
        )
    
    async def execute_tool(self, arguments: Dict[str, Any]) -> Any:
        """Execute the tool with provided arguments."""
        return await self.__call__(arguments)
        
    async def __call__(self, arguments: Dict[str, Any]) -> Any:
        """Execute the flow with provided arguments."""
        try:
            # Create input model if flow has pipeline
            if hasattr(self.flow, 'run_pipeline'):
                # Get input model from pipeline signature
                pipeline_method = getattr(self.flow, 'run_pipeline')
                if hasattr(pipeline_method, '__annotations__'):
                    input_type = None
                    for param_name, param_type in pipeline_method.__annotations__.items():
                        if param_name != 'return' and hasattr(param_type, 'model_validate'):
                            input_type = param_type
                            break
                    
                    if input_type:
                        input_data = input_type.model_validate(arguments)
                        result = await self.flow.run_pipeline(input_data)
                    else:
                        # Fallback to direct arguments
                        result = await self.flow.run_pipeline(arguments)
                else:
                    result = await self.flow.run_pipeline(arguments)
            else:
                # Try to execute flow directly with proper Context
                context: Context[Any] = Context(data=arguments)
                result = await self.flow.execute(context)
            
            # Convert result to serializable format
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            elif hasattr(result, 'dict'):
                return result.model_dump()
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error executing flow '{self.flow_name}': {e}")
            raise MCPError(f"Flow execution failed: {str(e)}")


class ResourceHandler:
    """Handler for resource-based MCP resources."""
    
    def __init__(self, resource_getter: Callable[[], Any], resource_name: str):
        self.getter = resource_getter
        self.resource_name = resource_name
    
    async def __call__(self) -> Any:
        """Get the resource data."""
        try:
            result = await self.getter() if asyncio.iscoroutinefunction(self.getter) else self.getter()
            
            # Convert to serializable format
            if hasattr(result, 'model_dump'):
                return {"content": result.model_dump(), "mimeType": "application/json"}
            elif hasattr(result, 'dict'):
                return {"content": result.model_dump(), "mimeType": "application/json"}
            else:
                return {"content": result, "mimeType": "application/json"}
                
        except Exception as e:
            logger.error(f"Error getting resource '{self.resource_name}': {e}")
            raise MCPError(f"Resource access failed: {str(e)}")


# Decorator already imported above

@provider(provider_type="mcp_server", name="mcp-server", settings_class=MCPServerSettings)
class MCPServerProvider(Provider[MCPServerSettings]):
    """Provider that exposes flowlib capabilities as an MCP server."""
    
    def __init__(self, name: str, provider_type: str = "mcp_server", settings: Optional[MCPServerSettings] = None):
        super().__init__(
            name=name,
            provider_type=provider_type,
            settings=settings
        )
        self._server: Optional[BaseMCPServer] = None
        self._http_server: Optional[Any] = None  # AppRunner when HTTP_AVAILABLE
        self._running = False
        
    async def _initialize(self) -> None:
        """Initialize MCP server."""
        try:
            # Create server
            self._server = BaseMCPServer(
                name=self.settings.server_name,
                version=self.settings.server_version
            )
            
            # Register flows as tools
            await self._register_flows_as_tools()
            
            # Register resources
            await self._register_resources()
            
            # Start HTTP server if using SSE or WebSocket
            if self.settings.transport in [MCPTransport.SSE, MCPTransport.WEBSOCKET]:
                await self._start_http_server()
            
            self._running = True
            logger.info(f"MCP server '{self.name}' started on {self.settings.host}:{self.settings.port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server '{self.name}': {e}")
            await self._cleanup()
            raise
    
    async def _shutdown(self) -> None:
        """Shutdown MCP server."""
        await self._cleanup()
    
    async def _cleanup(self) -> None:
        """Clean up server resources."""
        self._running = False
        
        if self._http_server:
            try:
                await self._http_server.cleanup()
            except Exception as e:
                logger.warning(f"Error stopping HTTP server: {e}")
            finally:
                self._http_server = None
        
        if self._server:
            try:
                self._server.stop()
            except Exception as e:
                logger.warning(f"Error stopping MCP server: {e}")
            finally:
                self._server = None
    
    async def _register_flows_as_tools(self) -> None:
        """Convert flows to MCP tools."""
        flows_to_expose = self._get_flows_to_expose()
        
        for flow_name, flow_instance in flows_to_expose.items():
            try:
                handler = FlowToolHandler(flow_instance, flow_name)
                tool = handler.create_tool()
                
                if self._server is None:
                    raise RuntimeError("MCP server not initialized")
                self._server.register_tool(tool, handler)
                logger.debug(f"Registered flow '{flow_name}' as MCP tool")
                
            except Exception as e:
                logger.warning(f"Failed to register flow '{flow_name}' as MCP tool: {e}")
    
    def _get_flows_to_expose(self) -> Dict[str, Flow]:
        """Get flows that should be exposed as MCP tools."""
        all_flows = flow_registry.get_flow_instances()
        
        if self.settings.expose_all_flows:
            # Expose all flows except excluded ones
            return {
                name: cast(Flow, flow) for name, flow in all_flows.items()
                if name not in self.settings.exclude_flows
            }
        else:
            # Only expose specifically listed flows
            return {
                name: cast(Flow, all_flows[name]) for name in self.settings.exposed_flows
                if name in all_flows
            }
    
    def _flow_to_mcp_tool(self, flow_name: str, flow_instance: Flow) -> MCPTool:
        """Convert a flow to an MCP tool."""
        # Get flow metadata
        description = "No description available"
        input_schema = MCPToolInputSchema()
        
        if hasattr(flow_instance, '__flow_metadata__'):
            metadata = flow_instance.__flow_metadata__
            if 'description' in metadata:
                description = metadata['description']
        
        # Try to extract input schema from pipeline method
        if hasattr(flow_instance, 'run_pipeline'):
            pipeline_method = getattr(flow_instance, 'run_pipeline')
            if hasattr(pipeline_method, '__annotations__'):
                for param_name, param_type in pipeline_method.__annotations__.items():
                    if param_name != 'return' and hasattr(param_type, 'model_json_schema'):
                        try:
                            input_schema = MCPToolInputSchema.from_pydantic_model(param_type)
                            break
                        except Exception as e:
                            logger.warning(f"Failed to extract schema from {param_type}: {e}")
        
        return MCPTool(
            name=flow_name,
            description=description,
            input_schema=input_schema
        )
    
    async def _register_resources(self) -> None:
        """Register resources that can be accessed via MCP."""
        
        # Memory resources
        if self.settings.expose_memory:
            memory_resource = MCPResource(
                uri="memory://working",
                name="Working Memory",
                description="Access to working memory contents",
                mime_type="application/json"
            )
            
            async def get_memory() -> Dict[str, str]:
                # This would need to be connected to actual memory system
                return {"type": "working_memory", "contents": "memory data"}
            
            if self._server is None:
                raise RuntimeError("MCP server not initialized")
            self._server.register_resource(
                memory_resource,
                ResourceHandler(get_memory, "working_memory")
            )
        
        # Flow registry resources
        if self.settings.expose_flow_registry:
            registry_resource = MCPResource(
                uri="flows://registry",
                name="Flow Registry",
                description="List of available flows",
                mime_type="application/json"
            )
            
            def get_flow_registry() -> Dict[str, Union[List[str], int]]:
                flows = flow_registry.get_flow_instances()
                return {
                    "flows": list(flows.keys()),
                    "count": len(flows)
                }
            
            if self._server is None:
                raise RuntimeError("MCP server not initialized")
            self._server.register_resource(
                registry_resource,
                ResourceHandler(get_flow_registry, "flow_registry")
            )
        
        # Performance metrics
        if self.settings.expose_metrics:
            metrics_resource = MCPResource(
                uri="metrics://performance",
                name="Performance Metrics",
                description="Server performance metrics",
                mime_type="application/json"
            )
            
            def get_metrics() -> Dict[str, Union[bool, int]]:
                if self._server is None:
                    return {
                        "server_running": self._running,
                        "tools_registered": 0,
                        "resources_registered": 0
                    }
                return {
                    "server_running": self._running,
                    "tools_registered": len(self._server._tools),
                    "resources_registered": len(self._server._resources)
                }

            if self._server is None:
                raise RuntimeError("MCP server not initialized")
            self._server.register_resource(
                metrics_resource,
                ResourceHandler(get_metrics, "performance_metrics")
            )
    
    async def _start_http_server(self) -> None:
        """Start HTTP server for SSE/WebSocket transport."""
        if not HTTP_AVAILABLE:
            raise MCPError("aiohttp and aiohttp-cors are required for HTTP transport")
        
        try:
            from aiohttp import web
            app = web.Application()
            
            # Enable CORS
            cors = aiohttp_cors.setup(app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*"
                )
            })
            
            # Add routes
            app.router.add_post('/messages', self._handle_http_message)
            app.router.add_get('/events', self._handle_sse)
            app.router.add_get('/ws', self._handle_websocket)
            app.router.add_get('/health', self._handle_health)
            
            # Add CORS to all routes
            for route in list(app.router.routes()):
                cors.add(route)
            
            # Start server using aiohttp runner
            from aiohttp import web
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, self.settings.host, self.settings.port)
            await site.start()
            self._http_server = runner
            
        except Exception as e:
            raise MCPError(f"Failed to start HTTP server: {str(e)}")
    
    async def _handle_http_message(self, request: Any) -> Any:
        """Handle HTTP message requests."""
        try:
            data = await request.json()
            message = MCPMessage(**data)

            if self._server is None:
                raise RuntimeError("MCP server not initialized")
            response = await self._server.handle_request(message)
            
            return web.json_response(response.model_dump(exclude_none=True))
            
        except Exception as e:
            logger.error(f"Error handling HTTP message: {e}")
            return web.json_response(
                {"error": {"code": -32603, "message": str(e)}},
                status=500
            )
    
    async def _handle_sse(self, request: Any) -> Any:
        """Handle Server-Sent Events."""
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )
        
        await response.prepare(request)
        
        # Keep connection alive
        try:
            while True:
                await asyncio.sleep(30)  # Send keep-alive every 30 seconds
                await response.write(b"data: {\"type\": \"ping\"}\n\n")
        except Exception:
            pass
        
        return response
    
    async def _handle_websocket(self, request: Any) -> Any:
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        message = MCPMessage(**data)

                        if self._server is None:
                            raise RuntimeError("MCP server not initialized")
                        response = await self._server.handle_request(message)
                        
                        await ws.send_str(
                            json.dumps(response.model_dump(exclude_none=True))
                        )
                    except Exception as e:
                        await ws.send_str(
                            json.dumps({"error": {"code": -32603, "message": str(e)}})
                        )
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        
        return ws
    
    async def _handle_health(self, request: Any) -> Any:
        """Handle health check requests."""
        return web.json_response({
            "status": "healthy" if self._running else "unhealthy",
            "server_name": self.settings.server_name,
            "version": self.settings.server_version,
            "tools_count": len(self._server._tools) if self._server else 0,
            "resources_count": len(self._server._resources) if self._server else 0
        })
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        if not self._server:
            return {"running": False}
        
        return {
            "running": self._running,
            "name": self.settings.server_name,
            "version": self.settings.server_version,
            "host": self.settings.host,
            "port": self.settings.port,
            "transport": self.settings.transport.value,
            "tools_count": len(self._server._tools),
            "resources_count": len(self._server._resources),
            "available_tools": list(self._server._tools.keys()),
            "available_resources": list(self._server._resources.keys())
        }
    
    # Aliases for test compatibility
    async def _register_flows(self) -> None:
        """Alias for _register_flows_as_tools for test compatibility."""
        await self._register_flows_as_tools()
    
    async def _handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call requests."""
        if not self._server or tool_name not in self._server._tools:
            raise MCPError(f"Tool not found: {tool_name}")
        
        try:
            # Get the tool object and execute it
            tool = self._server._tools[tool_name]
            # Assume MCPTool has an execute method or similar
            result = await tool.execute(arguments) if hasattr(tool, 'execute') else tool
            # Ensure the result is a dictionary
            if isinstance(result, dict):
                return result
            else:
                return {"result": result}
        except Exception as e:
            raise MCPError(f"Tool execution failed: {str(e)}")
    
    async def _handle_resource_read(self, uri: str) -> Dict[str, Any]:
        """Handle resource read requests."""
        if not self._server:
            raise MCPError("Server not initialized")
        
        # Find resource by URI
        if uri in self._server._resources:
            try:
                handler = self._server._resource_handlers[uri]
                if asyncio.iscoroutinefunction(handler):
                    result = await handler()
                else:
                    result = handler()
                
                # Ensure proper format for response
                if isinstance(result, dict) and "content" in result:
                    return result
                else:
                    return {
                        "content": result,
                        "mimeType": "application/json",
                        "text": json.dumps(result) if not isinstance(result, str) else result
                    }
            except Exception as e:
                raise MCPError(f"Resource read failed: {str(e)}")
        
        raise MCPError(f"Resource not found: {uri}")
    
    async def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running