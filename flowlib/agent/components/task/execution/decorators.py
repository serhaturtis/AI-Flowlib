"""Tool decorators for automatic registration.

Following flowlib patterns from flows and resources decorators.
"""

import logging
from typing import Callable, List, Optional, Type, cast

from .interfaces import AgentToolFactory, AgentToolInterface, ToolInterface
from .models import ToolExecutionContext, ToolMetadata
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


def _get_tool_registry() -> 'ToolRegistry':
    """Lazy import of tool registry to avoid circular dependencies."""
    from .registry import tool_registry
    return tool_registry


class SimpleToolFactory(AgentToolFactory):
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

    def __call__(self) -> AgentToolInterface:
        """Create tool instance (AgentToolFactory protocol method)."""
        return cast(AgentToolInterface, self.tool_class())

    def create_tool(self, context: Optional[ToolExecutionContext] = None) -> AgentToolInterface:
        """Create tool instance."""
        return cast(AgentToolInterface, self.tool_class())

    def get_description(self) -> str:
        """Get full tool description."""
        return self.metadata.description

    def get_planning_description(self) -> str:
        """Get concise planning description (for prompts)."""
        return self.metadata.planning_description or self.metadata.description


def tool(name: Optional[str] = None, tool_category: str = "generic", description: Optional[str] = None,
         planning_description: Optional[str] = None,
         aliases: Optional[List[str]] = None, tags: Optional[List[str]] = None,
         version: str = "1.0.0", max_execution_time: Optional[int] = None,
         allowed_roles: Optional[List[str]] = None,
         denied_roles: Optional[List[str]] = None,
         requires_confirmation: bool = False) -> Callable[[type], type]:
    """Register a class as a tool.

    This decorator automatically registers tool classes with the global
    tool registry, following flowlib patterns from flows and resources.

    Args:
        name: Tool name (defaults to class name)
        category: Tool category for organization
        description: Full tool description (defaults to class docstring)
        planning_description: Concise description for planning prompts (optional, defaults to first sentence of description)
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
    def decorator(cls: type) -> type:
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

        # Get or generate planning description (concise version for prompts)
        tool_planning_description = planning_description
        if not tool_planning_description:
            # Auto-generate: use first sentence of description
            sentences = tool_description.replace('? ', '?|').replace('! ', '!|').replace('. ', '.|').split('|')
            tool_planning_description = sentences[0].strip()

        # Create tool metadata model
        tool_metadata = ToolMetadata(
            name=tool_name,
            description=tool_description,
            planning_description=tool_planning_description,
            tool_category=tool_category,
            aliases=aliases or [],
            tags=tags or [],
            version=version,
            max_execution_time=max_execution_time,
            allowed_roles=allowed_roles or [],
            denied_roles=denied_roles or [],
            requires_confirmation=requires_confirmation
        )

        # Store metadata on class
        cls.__tool_name__ = tool_name  # type: ignore[attr-defined]
        cls.__tool_category__ = tool_category  # type: ignore[attr-defined]
        cls.__tool_metadata__ = tool_metadata  # type: ignore[attr-defined]

        # Create factory for this tool
        factory = SimpleToolFactory(cls, tool_metadata)

        # Register with global registry
        registry.register(
            tool_name,
            factory,
            metadata=tool_metadata
        )

        logger.info(f"Registered tool '{tool_name}' via decorator (category: {tool_category})")

        # Return class unchanged
        return cls

    return decorator
