"""MCP-aware decorators for flows."""

import logging
from typing import Optional, Dict, Any, Type
from functools import wraps

from .decorators import flow, pipeline
from flowlib.providers.mcp.base import MCPToolInputSchema

logger = logging.getLogger(__name__)


def mcp_tool(
    name: Optional[str] = None,
    description: str = "",
    input_schema: Optional[Dict[str, Any]] = None,
    expose_by_default: bool = True
):
    """Decorator to mark a flow as an MCP tool.
    
    Args:
        name: Optional custom name for the tool (defaults to flow name)
        description: Description of what the tool does
        input_schema: Optional custom input schema
        expose_by_default: Whether to expose this tool by default
    """
    def decorator(cls):
        # Apply standard flow decorator first (mark as infrastructure)
        flow_cls = flow(name=name, description=description, is_infrastructure=True)(cls)
        
        # Add MCP metadata
        if not hasattr(flow_cls, '__mcp_metadata__'):
            flow_cls.__mcp_metadata__ = {}
        
        flow_cls.__mcp_metadata__.update({
            'expose_as_tool': True,
            'tool_name': name,
            'tool_description': description,
            'input_schema': input_schema,
            'expose_by_default': expose_by_default
        })
        
        return flow_cls
    
    return decorator


def mcp_resource(
    uri: str,
    name: str,
    description: str = "",
    mime_type: str = "application/json"
):
    """Decorator to mark a flow as providing an MCP resource.
    
    Args:
        uri: URI for the resource
        name: Human-readable name
        description: Description of the resource
        mime_type: MIME type of the resource content
    """
    def decorator(cls):
        # Apply standard flow decorator first (mark as infrastructure)
        flow_cls = flow(description=description, is_infrastructure=True)(cls)
        
        # Add MCP resource metadata
        if not hasattr(flow_cls, '__mcp_resource_metadata__'):
            flow_cls.__mcp_resource_metadata__ = {}
        
        flow_cls.__mcp_resource_metadata__.update({
            'expose_as_resource': True,
            'resource_uri': uri,
            'resource_name': name,
            'resource_description': description,
            'mime_type': mime_type
        })
        
        return flow_cls
    
    return decorator


def mcp_client_aware(client_name: str):
    """Decorator to make a flow aware of MCP clients.
    
    Args:
        client_name: Name of the MCP client provider to use
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @wraps(original_init)
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._mcp_client_name = client_name
            self._mcp_client = None
        
        cls.__init__ = __init__
        
        # Add method to get MCP client
        async def get_mcp_client(self):
            """Get the MCP client instance."""
            if not self._mcp_client:
                from flowlib.providers.core.registry import provider_registry
                # Removed ProviderType import - using config-driven provider access
                
                try:
                    self._mcp_client = await provider_registry.get_by_config(self._mcp_client_name)
                except Exception as e:
                    logger.warning(f"Failed to get MCP client '{self._mcp_client_name}': {e}")
            
            return self._mcp_client
        
        cls.get_mcp_client = get_mcp_client
        
        # Add method to call MCP tools
        async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]):
            """Call an MCP tool through the client."""
            client = await self.get_mcp_client()
            if client:
                return await client.call_tool(tool_name, arguments)
            else:
                raise RuntimeError(f"MCP client '{self._mcp_client_name}' not available")
        
        cls.call_mcp_tool = call_mcp_tool
        
        # Add method to read MCP resources
        async def read_mcp_resource(self, resource_uri: str):
            """Read an MCP resource through the client."""
            client = await self.get_mcp_client()
            if client:
                return await client.read_resource(resource_uri)
            else:
                raise RuntimeError(f"MCP client '{self._mcp_client_name}' not available")
        
        cls.read_mcp_resource = read_mcp_resource
        
        return cls
    
    return decorator


# Helper functions for flow developers
def get_mcp_metadata(flow_class) -> Dict[str, Any]:
    """Get MCP metadata from a flow class."""
    metadata = {}
    
    if hasattr(flow_class, '__mcp_metadata__'):
        metadata.update(flow_class.__mcp_metadata__)
    
    if hasattr(flow_class, '__mcp_resource_metadata__'):
        metadata.update(flow_class.__mcp_resource_metadata__)
    
    return metadata


def is_mcp_tool(flow_class) -> bool:
    """Check if a flow is marked as an MCP tool."""
    return (
        hasattr(flow_class, '__mcp_metadata__') and
        ('expose_as_tool' in flow_class.__mcp_metadata__ and flow_class.__mcp_metadata__['expose_as_tool'])
    )


def is_mcp_resource(flow_class) -> bool:
    """Check if a flow is marked as an MCP resource."""
    return (
        hasattr(flow_class, '__mcp_resource_metadata__') and
        ('expose_as_resource' in flow_class.__mcp_resource_metadata__ and flow_class.__mcp_resource_metadata__['expose_as_resource'])
    )


# Example usage patterns
if __name__ == "__main__":
    # Example of MCP tool flow
    from pydantic import BaseModel
    
    class AnalysisInput(BaseModel):
        text: str
        sentiment_analysis: bool = True
        entity_extraction: bool = False
    
    class AnalysisOutput(BaseModel):
        sentiment: str
        entities: list = []
        confidence: float
    
    @mcp_tool(
        name="analyze_text",
        description="Analyze text for sentiment and entities",
        expose_by_default=True
    )
    class TextAnalysisFlow:
        
        @pipeline(input_model=AnalysisInput, output_model=AnalysisOutput)
        async def analyze(self, input_data: AnalysisInput) -> AnalysisOutput:
            # Analysis logic here
            return AnalysisOutput(
                sentiment="positive",
                entities=[],
                confidence=0.85
            )
    
    # Example of MCP resource flow
    @mcp_resource(
        uri="memory://recent_conversations",
        name="Recent Conversations",
        description="Access to recent conversation history"
    )
    class ConversationHistoryFlow:
        
        async def get_conversations(self):
            # Return conversation data
            return {"conversations": []}
    
    # Example of MCP client-aware flow
    @mcp_client_aware("github_client")
    class GitHubIntegrationFlow:
        
        async def create_issue(self, title: str, body: str):
            # Use MCP client to create GitHub issue
            result = await self.call_mcp_tool("create_issue", {
                "title": title,
                "body": body
            })
            return result