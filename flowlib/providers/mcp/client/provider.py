"""MCP Client Provider implementation."""

import asyncio
import logging
from typing import Any

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.base import Provider, ProviderSettings
from flowlib.providers.core.decorators import provider
from flowlib.providers.mcp.transport import create_transport

# Removed ProviderType import - using config-driven provider access
from ..base import (
    BaseMCPClient,
    MCPConnection,
    MCPConnectionError,
    MCPResource,
    MCPTool,
    MCPToolNotFoundError,
    MCPTransport,
)

logger = logging.getLogger(__name__)


class MCPClientSettings(ProviderSettings):
    """Settings for MCP client provider."""

    server_uri: str = Field(
        default="stdio://server",
        description="MCP server URI (e.g., 'stdio://server', 'http://localhost:8080')",
    )
    transport: MCPTransport = Field(
        default=MCPTransport.STDIO, description="MCP transport type: STDIO or SSE"
    )
    timeout: float = Field(30.0, description="Connection timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")
    retry_delay: float = Field(1.0, description="Delay between retry attempts")

    # Transport-specific settings
    server_command: str | None = Field(None, description="Command to start server (for stdio)")
    server_args: list[str] = Field(default_factory=list, description="Arguments for server command")
    auth_token: str | None = Field(None, description="Authentication token (for HTTP/SSE)")
    headers: dict[str, str] = Field(default_factory=dict, description="Additional headers")


# Decorator already imported above


@provider(provider_type="mcp_client", name="mcp-client", settings_class=MCPClientSettings)
class MCPClientProvider(Provider[MCPClientSettings]):
    """Provider that acts as MCP client to connect to external servers."""

    def __init__(
        self,
        name: str,
        provider_type: str = "mcp_client",
        settings: MCPClientSettings | None = None,
    ):
        super().__init__(name=name, provider_type=provider_type, settings=settings)
        self._client: BaseMCPClient | None = None
        self._connection: MCPConnection | None = None
        self._connected = False

    async def initialize(self) -> None:
        """Initialize MCP client connection."""
        try:
            # Create client
            self._client = BaseMCPClient(name=f"flowlib-client-{self.name}")

            # Create transport connection
            self._connection = await create_transport(
                transport_type=self.settings.transport,
                server_uri=self.settings.server_uri,
                server_command=self.settings.server_command,
                server_args=self.settings.server_args,
                timeout=self.settings.timeout,
                auth_token=self.settings.auth_token,
                headers=self.settings.headers,
            )

            # Initialize client with connection
            await self._client.initialize(self._connection)
            self._connected = True
            self._initialized = True

            logger.info(f"MCP client '{self.name}' connected to {self.settings.server_uri}")

        except Exception as e:
            logger.error(f"Failed to initialize MCP client '{self.name}': {e}")
            await self._cleanup()
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def shutdown(self) -> None:
        """Shutdown MCP client connection."""
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._connected = False
        self._initialized = False

        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning(f"Error closing MCP client: {e}")
            finally:
                self._client = None

        if self._connection:
            try:
                await self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing MCP connection: {e}")
            finally:
                self._connection = None

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self._connected or not self._client:
            error_context = ErrorContext.create(
                flow_name="mcp_client",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.call_tool",
                component=self.name,
                operation="call_tool",
            )

            provider_context = ProviderErrorContext(
                provider_name=self.name, provider_type="mcp", operation="call_tool", retry_count=0
            )

            raise ProviderError(
                message="MCP client not connected",
                context=error_context,
                provider_context=provider_context,
            )

        try:
            result = await self._client.call_tool(tool_name, arguments)
            logger.debug(f"MCP tool '{tool_name}' called successfully")
            return result

        except MCPToolNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error calling MCP tool '{tool_name}': {e}")
            # Try to reconnect if connection was lost
            if isinstance(e, MCPConnectionError):
                await self._attempt_reconnect()
            raise

    async def read_resource(self, resource_uri: str) -> Any:
        """Read a resource from the MCP server."""
        if not self._connected or not self._client:
            error_context = ErrorContext.create(
                flow_name="mcp_client",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.read_resource",
                component=self.name,
                operation="read_resource",
            )

            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="mcp",
                operation="read_resource",
                retry_count=0,
            )

            raise ProviderError(
                message="MCP client not connected",
                context=error_context,
                provider_context=provider_context,
            )

        try:
            result = await self._client.read_resource(resource_uri)
            logger.debug(f"MCP resource '{resource_uri}' read successfully")
            return result

        except Exception as e:
            logger.error(f"Error reading MCP resource '{resource_uri}': {e}")
            # Try to reconnect if connection was lost
            if isinstance(e, MCPConnectionError):
                await self._attempt_reconnect()
            raise

    async def list_tools(self) -> dict[str, MCPTool]:
        """List available tools from the MCP server."""
        if not self._connected or not self._client:
            error_context = ErrorContext.create(
                flow_name="mcp_client",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.list_tools",
                component=self.name,
                operation="list_tools",
            )

            provider_context = ProviderErrorContext(
                provider_name=self.name, provider_type="mcp", operation="list_tools", retry_count=0
            )

            raise ProviderError(
                message="MCP client not connected",
                context=error_context,
                provider_context=provider_context,
            )

        return self._client.get_available_tools()

    async def list_resources(self) -> dict[str, MCPResource]:
        """List available resources from the MCP server."""
        if not self._connected or not self._client:
            error_context = ErrorContext.create(
                flow_name="mcp_client",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.list_resources",
                component=self.name,
                operation="list_resources",
            )

            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="mcp",
                operation="list_resources",
                retry_count=0,
            )

            raise ProviderError(
                message="MCP client not connected",
                context=error_context,
                provider_context=provider_context,
            )

        return self._client.get_available_resources()

    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to the MCP server."""
        logger.info(f"Attempting to reconnect MCP client '{self.name}'...")

        for attempt in range(self.settings.retry_attempts):
            try:
                await self._cleanup()
                await asyncio.sleep(self.settings.retry_delay * (attempt + 1))
                await self.initialize()
                logger.info(f"MCP client '{self.name}' reconnected successfully")
                return

            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                if attempt == self.settings.retry_attempts - 1:
                    logger.error(
                        f"Failed to reconnect MCP client '{self.name}' after {self.settings.retry_attempts} attempts"
                    )
                    raise MCPConnectionError("Failed to reconnect to MCP server") from e

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._client is not None

    async def check_connection(self) -> bool:
        """Check if client is connected to MCP server."""
        if not self._client:
            return False
        return self._connected

    async def reconnect(self) -> None:
        """Reconnect to the MCP server."""
        if not self._client or not self._connection:
            error_context = ErrorContext.create(
                flow_name="mcp_client",
                error_type="ProviderError",
                error_location="MCPClientProvider.reconnect",
                component="mcp_client_provider",
                operation="reconnect",
            )
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type=self.provider_type,
                operation="reconnect",
                retry_count=0,
            )
            raise ProviderError(
                "Cannot reconnect: No existing client or connection",
                error_context,
                provider_context,
            )

        try:
            await self._client.initialize(self._connection)
            self._connected = True
            logger.info(f"MCP client '{self.name}' reconnected successfully")
        except Exception as e:
            logger.error(f"Failed to reconnect MCP client '{self.name}': {e}")
            error_context = ErrorContext.create(
                flow_name="mcp_client",
                error_type="ProviderError",
                error_location="MCPClientProvider.reconnect",
                component="mcp_client_provider",
                operation="reconnect",
            )
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type=self.provider_type,
                operation="reconnect",
                retry_count=1,
            )
            raise ProviderError(
                f"Failed to reconnect: {e}", error_context, provider_context, cause=e
) from e

    async def get_server_info(self) -> dict[str, Any]:
        """Get information about the connected server."""
        if not self._connected or not self._client:
            return {"connected": False}

        tools = self._client.get_available_tools()
        resources = self._client.get_available_resources()

        return {
            "connected": True,
            "server_uri": self.settings.server_uri,
            "transport": self.settings.transport.value,
            "tools_count": len(tools),
            "resources_count": len(resources),
            "available_tools": list(tools.keys()),
            "available_resources": list(resources.keys()),
        }


class MCPToolWrapper:
    """Wrapper for MCP tools to make them look like regular functions."""

    def __init__(self, client_provider: MCPClientProvider, tool: MCPTool):
        self.client = client_provider
        self.tool = tool
        self.name = tool.name
        self.description = tool.description
        self.input_schema = tool.input_schema

    async def __call__(self, **kwargs: Any) -> Any:
        """Call the MCP tool with keyword arguments."""
        return await self.client.call_tool(self.tool.name, kwargs)

    def __str__(self) -> str:
        return f"MCPTool({self.name}): {self.description}"

    def __repr__(self) -> str:
        return f"MCPToolWrapper(name='{self.name}', description='{self.description}')"


class MCPResourceWrapper:
    """Wrapper for MCP resources to make them look like regular resources."""

    def __init__(self, client_provider: MCPClientProvider, resource: MCPResource):
        self.client = client_provider
        self.resource = resource
        self.uri = resource.uri
        self.name = resource.name
        self.description = resource.description
        self.mime_type = resource.mime_type

    async def read(self) -> Any:
        """Read the MCP resource."""
        return await self.client.read_resource(self.resource.uri)

    def __str__(self) -> str:
        return f"MCPResource({self.name}): {self.description}"

    def __repr__(self) -> str:
        return f"MCPResourceWrapper(uri='{self.uri}', name='{self.name}')"


# Helper functions for easier usage
async def create_mcp_client(
    name: str, server_uri: str, transport: MCPTransport = MCPTransport.STDIO, **kwargs: Any
) -> MCPClientProvider:
    """Create and initialize an MCP client provider."""
    settings = MCPClientSettings(server_uri=server_uri, transport=transport, **kwargs)

    client = MCPClientProvider(name=name, provider_type="mcp_client", settings=settings)
    await client.initialize()
    return client


async def get_mcp_tools(client: MCPClientProvider) -> dict[str, MCPToolWrapper]:
    """Get wrapped MCP tools from a client."""
    tools = await client.list_tools()
    return {name: MCPToolWrapper(client, tool) for name, tool in tools.items()}


async def get_mcp_resources(client: MCPClientProvider) -> dict[str, MCPResourceWrapper]:
    """Get wrapped MCP resources from a client."""
    resources = await client.list_resources()
    return {uri: MCPResourceWrapper(client, resource) for uri, resource in resources.items()}
