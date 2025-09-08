"""Tool decorators for automatic registration.

Following flowlib patterns from flows and resources decorators.
"""

import logging
from typing import Type, Optional, Dict, Any, List, Callable
from .models import ToolMetadata, ToolExecutionContext
from .interfaces import ToolInterface, ToolFactory

logger = logging.getLogger(__name__)


def _get_tool_registry():
    """Lazy import of tool registry to avoid circular dependencies."""
    from .registry import tool_registry
    return tool_registry


class SimpleToolFactory(ToolFactory):
    """Simple factory for decorator-registered tools.
    
    Creates a factory wrapper around tool classes registered via @tool.
    """
    
    def __init__(self, tool_class: Type[ToolInterface], metadata: ToolMetadata):
        """Initialize factory with tool class.
        
        Args:
            tool_class: The tool class to instantiate
            metadata: Tool metadata from decorator
        """
        self.tool_class = tool_class
        self.metadata = metadata
        
    def create_tool(self, context: Optional[ToolExecutionContext] = None) -> ToolInterface:
        """Create tool instance."""
        return self.tool_class()
        
    def get_description(self) -> str:
        """Get tool description."""
        return self.metadata.description


def tool(name: str = None, category: str = "general", description: str = None, 
         aliases: Optional[List[str]] = None, tags: Optional[List[str]] = None,
         version: str = "1.0.0", max_execution_time: Optional[int] = None,
         is_safe: bool = True):
    """Register a class as a tool.
    
    This decorator automatically registers tool classes with the global
    tool registry, following flowlib patterns from flows and resources.
    
    Args:
        name: Tool name (defaults to class name)
        category: Tool category for organization
        description: Tool description (defaults to class docstring)
        aliases: Alternative names for the tool
        tags: Tags for tool discovery
        version: Tool version
        max_execution_time: Maximum execution time in seconds
        is_safe: Whether tool is safe to execute automatically
        
    Example:
        @tool(category="filesystem", description="Read file contents")
        class ReadTool(Tool):
            def execute(self, parameters, context):
                # Implementation
                
    Returns:
        The decorated class unchanged
        
    Raises:
        TypeError: If class doesn't inherit from Tool
        RuntimeError: If registry not initialized
    """
    def decorator(cls):
        # Validate tool class implements interface
        if not hasattr(cls, 'execute') or not hasattr(cls, 'get_name') or not hasattr(cls, 'get_description'):
            raise TypeError(f"Tool '{cls.__name__}' must implement ToolInterface protocol")
            
        # Get registry
        registry = _get_tool_registry()
        if registry is None:
            raise RuntimeError("Tool registry not initialized")
            
        # Determine tool name
        tool_name = name or cls.__name__.lower().replace("tool", "")
        
        # Get description from class or decorator
        tool_description = description
        if not tool_description and cls.__doc__:
            # Use first line of docstring
            tool_description = cls.__doc__.strip().split('\n')[0]
        if not tool_description:
            tool_description = f"{tool_name} tool"
            
        # Create tool metadata model
        tool_metadata = ToolMetadata(
            name=tool_name,
            description=tool_description,
            category=category,
            aliases=aliases or [],
            tags=tags or [],
            version=version,
            max_execution_time=max_execution_time,
            is_safe=is_safe
        )
        
        # Store metadata on class
        cls.__tool_name__ = tool_name
        cls.__tool_category__ = category
        cls.__tool_metadata__ = tool_metadata
        
        # Create factory for this tool
        factory = SimpleToolFactory(cls, tool_metadata)
        
        # Register with global registry
        registry.register(
            tool_name,
            factory,
            metadata=tool_metadata
        )
        
        logger.info(f"Registered tool '{tool_name}' via decorator (category: {category})")
        
        # Return class unchanged
        return cls
        
    return decorator