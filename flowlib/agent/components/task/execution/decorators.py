"""Tool decorators for automatic registration.

Following flowlib patterns from flows and resources decorators.
"""

import logging
from collections.abc import Callable
from typing import cast

from .interfaces import AgentToolFactory, AgentToolInterface, ToolInterface
from .models import ToolExecutionContext, ToolMetadata, ToolParameters
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


def _get_tool_registry() -> "ToolRegistry":
    """Lazy import of tool registry to avoid circular dependencies."""
    from .registry import tool_registry

    return tool_registry


class SimpleToolFactory(AgentToolFactory):
    """Simple factory for decorator-registered tools.

    Creates a factory wrapper around tool classes registered via @tool.
    """

    def __init__(self, tool_class: type[ToolInterface], metadata: ToolMetadata):
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

    def create_tool(self, context: ToolExecutionContext | None = None) -> AgentToolInterface:
        """Create tool instance."""
        return cast(AgentToolInterface, self.tool_class())

    def get_description(self) -> str:
        """Get full tool description."""
        return self.metadata.description

    def get_planning_description(self) -> str:
        """Get concise planning description (for prompts)."""
        return self.metadata.planning_description or self.metadata.description


def tool(
    *,
    parameter_type: type,
    name: str | None = None,
    tool_category: str = "generic",
    description: str | None = None,
    planning_description: str | None = None,
    aliases: list[str] | None = None,
    tags: list[str] | None = None,
    version: str = "1.0.0",
    max_execution_time: int | None = None,
    allowed_roles: list[str] | None = None,
    denied_roles: list[str] | None = None,
    requires_confirmation: bool = False,
) -> Callable[[type], type]:
    """Register a class as a tool with strict parameter type enforcement.

    Following flowlib @pipeline pattern - parameter_type is REQUIRED for type safety.
    Tools receive validated parameter instances, not TodoItems with natural language.

    Args:
        name: Tool name (defaults to class name)
        parameter_type: Pydantic ToolParameters subclass (REQUIRED - enforces type safety)
        tool_category: Tool category for organization
        description: Full tool description (defaults to class docstring)
        planning_description: Concise description for planning prompts
        aliases: Alternative names for the tool
        tags: Tags for tool discovery
        version: Tool version
        max_execution_time: Maximum execution time in seconds
        allowed_roles: Allowed agent roles
        denied_roles: Denied agent roles
        requires_confirmation: Whether tool requires user confirmation

    Example:
        class ReadFileParameters(ToolParameters):
            file_path: str = Field(..., description="Path to file")

        @tool(
            parameter_type=ReadFileParameters,  # REQUIRED keyword argument!
            name="read_file",
            tool_category="filesystem",
            description="Read file contents"
        )
        class ReadFileTool:
            async def execute(self, todo: TodoItem, params: ReadFileParameters, context: ToolExecutionContext) -> ToolResult:
                # todo is TodoItem with task description
                # params is validated ReadFileParameters instance!
                with open(params.file_path) as f:
                    return ToolResult(status=ToolStatus.SUCCESS, message=f.read())

    Returns:
        The decorated class unchanged

    Raises:
        ValueError: If parameter_type is not a ToolParameters subclass
        TypeError: If class doesn't implement ToolInterface protocol
        RuntimeError: If registry not initialized
    """

    def decorator(cls: type) -> type:
        # Validate parameter_type is ToolParameters subclass (STRICT - no fallback)
        if not isinstance(parameter_type, type) or not issubclass(parameter_type, ToolParameters):
            raise ValueError(
                f"Tool '{cls.__name__}' parameter_type must be a ToolParameters subclass, got {parameter_type}. "
                f"Following flowlib pattern: parameter_type is REQUIRED for type safety."
            )

        # Validate tool class implements interface
        if (
            not hasattr(cls, "execute")
            or not hasattr(cls, "get_name")
            or not hasattr(cls, "get_description")
        ):
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
            tool_description = cls.__doc__.strip().split("\n")[0]
        if not tool_description:
            tool_description = f"{tool_name} tool"

        # Get planning description (use full description if not provided)
        tool_planning_description = planning_description or tool_description

        # Create tool metadata model
        tool_metadata = ToolMetadata(
            name=tool_name,
            description=tool_description,
            planning_description=tool_planning_description,
            tool_category=tool_category,
            parameter_type=parameter_type,  # REQUIRED - enforces type safety
            aliases=aliases or [],
            tags=tags or [],
            version=version,
            max_execution_time=max_execution_time,
            allowed_roles=allowed_roles or [],
            denied_roles=denied_roles or [],
            requires_confirmation=requires_confirmation,
        )

        # Store metadata on class
        cls.__tool_name__ = tool_name  # type: ignore[attr-defined]
        cls.__tool_category__ = tool_category  # type: ignore[attr-defined]
        cls.__parameter_type__ = parameter_type  # type: ignore[attr-defined]
        cls.__tool_metadata__ = tool_metadata  # type: ignore[attr-defined]

        # Create factory for this tool
        factory = SimpleToolFactory(cls, tool_metadata)

        # Register with global registry
        registry.register(tool_name, factory, metadata=tool_metadata)

        logger.info(f"Registered tool '{tool_name}' via decorator (category: {tool_category})")

        # Return class unchanged
        return cls

    return decorator
