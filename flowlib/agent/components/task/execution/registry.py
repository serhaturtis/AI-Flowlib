"""Tool Registry with proper BaseRegistry inheritance.

This module provides the tool registry following flowlib architectural patterns.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from flowlib.core.registry.registry import BaseRegistry

from ..models import TodoItem
from .interfaces import AgentToolFactory, AgentToolInterface
from .models import ToolExecutionContext, ToolMetadata, ToolResult

logger = logging.getLogger(__name__)


class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found in the registry."""
    pass


class ToolRegistryEntry:
    """Registry entry for a tool.
    
    Simple data class for storing tool factory and metadata.
    Not using StrictBaseModel because it can't validate Protocol instances.
    """

    def __init__(self, name: str, factory: AgentToolFactory, metadata: ToolMetadata,
                 category: str = "general", aliases: Optional[List[str]] = None):
        self.name = name
        self.factory = factory
        self.metadata = metadata
        self.category = category
        self.aliases = aliases or []


class ToolRegistry(BaseRegistry[AgentToolFactory]):
    """Tool registry for agent tool management.
    
    Inherits from BaseRegistry to provide consistent interface with
    other flowlib registries. Manages tool factories for lazy instantiation.
    """

    def __init__(self) -> None:
        """Initialize tool registry."""
        self._tools: Dict[str, ToolRegistryEntry] = {}
        self._aliases: Dict[str, str] = {}  # alias -> canonical name mapping

    def register(self, name: str, obj: AgentToolFactory, metadata: Optional[ToolMetadata] = None, **kwargs: Union[str, int, bool]) -> None:
        """Register a tool factory.

        Args:
            name: Tool name
            obj: Tool factory instance
            metadata: Tool metadata (optional, will create default if not provided)
            **kwargs: Additional registry metadata
        """
        # Validate factory implements protocol
        if not isinstance(obj, AgentToolFactory):
            raise TypeError(f"Object must implement AgentToolFactory protocol, got {type(obj)}")

        # Use provided metadata or create default
        if metadata is None:
            tool_metadata = ToolMetadata(
                name=name,
                description=obj.get_description()
            )
        else:
            tool_metadata = metadata

        # Create registry entry
        entry = ToolRegistryEntry(
            name=name,
            factory=obj,
            metadata=tool_metadata,
            category=tool_metadata.tool_category,
            aliases=tool_metadata.aliases
        )

        # Store in registry
        self._tools[name] = entry

        # Register aliases
        for alias in tool_metadata.aliases:
            self._aliases[alias] = name

        logger.info(f"Registered tool: {name} (category: {tool_metadata.tool_category})")

    def get(self, name: str, expected_type: Optional[Type[Any]] = None) -> AgentToolFactory:
        """Get tool factory by name.
        
        Args:
            name: Tool name or alias
            expected_type: Optional type checking (not used for factories)
            
        Returns:
            Tool factory instance
            
        Raises:
            KeyError: If tool not found
        """
        # Check if it's an alias
        if name in self._aliases:
            name = self._aliases[name]

        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")

        return self._tools[name].factory

    def contains(self, name: str) -> bool:
        """Check if tool exists.
        
        Args:
            name: Tool name or alias
            
        Returns:
            True if tool exists
        """
        return name in self._tools or name in self._aliases

    def list(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """List registered tools.
        
        Args:
            filter_criteria: Optional filtering (e.g., {"category": "filesystem"})
            
        Returns:
            List of tool names
        """
        if not filter_criteria:
            return list(self._tools.keys())

        # Apply filters
        results = []
        for name, entry in self._tools.items():
            match = True

            # Check category filter
            if "category" in filter_criteria:
                if entry.category != filter_criteria["category"]:
                    match = False

            # Add more filters as needed
            if match:
                results.append(name)

        return results

    def clear(self) -> None:
        """Clear all tool registrations."""
        self._tools.clear()
        self._aliases.clear()
        logger.info("Cleared all tool registrations")

    def remove(self, name: str) -> bool:
        """Remove a tool registration.
        
        Args:
            name: Tool name to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self._tools:
            entry = self._tools[name]

            # Remove aliases
            for alias in entry.aliases:
                self._aliases.pop(alias, None)

            # Remove tool
            del self._tools[name]
            logger.info(f"Removed tool: {name}")
            return True

        return False

    def update(self, name: str, obj: AgentToolFactory, metadata: Optional[ToolMetadata] = None, **kwargs: Union[str, int, bool]) -> bool:
        """Update existing tool registration.
        
        Args:
            name: Tool name
            obj: New factory instance
            metadata: Updated metadata
            
        Returns:
            True if updated existing, False if new registration
        """
        exists = name in self._tools

        # Remove old registration if exists
        if exists:
            self.remove(name)

        # Register new/updated
        self.register(name, obj, metadata)

        return exists

    def register_with_aliases(self, canonical_name: str, obj: AgentToolFactory, aliases: Optional[List[str]] = None, **metadata: Any) -> None:
        """Register tool with aliases.

        Args:
            canonical_name: Primary tool name
            obj: Tool factory
            aliases: Alternative names
            **metadata: Additional metadata
        """
        # Create or update metadata with aliases
        tool_metadata_param = metadata.get('metadata')
        if tool_metadata_param is None:
            tool_metadata = ToolMetadata(
                name=canonical_name,
                description=obj.get_description(),
                aliases=aliases or []
            )
        else:
            tool_metadata = tool_metadata_param
            if aliases:
                # Update metadata aliases
                tool_metadata.aliases = aliases

        self.register(canonical_name, obj, metadata=tool_metadata)

    def list_aliases(self, canonical_name: str) -> List[str]:
        """List aliases for a tool.
        
        Args:
            canonical_name: Tool name
            
        Returns:
            List of aliases
        """
        if canonical_name in self._tools:
            return self._tools[canonical_name].aliases.copy()
        return []

    # Tool-specific methods

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata.
        
        Args:
            name: Tool name or alias
            
        Returns:
            Tool metadata or None if not found
        """
        # Resolve alias
        if name in self._aliases:
            name = self._aliases[name]

        if name in self._tools:
            return self._tools[name].metadata
        return None

    def get_categories(self) -> List[str]:
        """Get list of all tool categories.
        
        Returns:
            Unique list of categories
        """
        categories = set()
        for entry in self._tools.values():
            categories.add(entry.category)
        return sorted(list(categories))

    def create_tool(self, name: str, context: Optional[ToolExecutionContext] = None) -> AgentToolInterface:
        """Create tool instance from factory.
        
        Args:
            name: Tool name
            context: Execution context
            
        Returns:
            Tool instance
            
        Raises:
            KeyError: If tool not found
        """
        factory = self.get(name)
        return factory()

    def get_tool_metadata(self, name: str) -> ToolMetadata:
        """Get metadata for a specific tool.
        
        Args:
            name: Tool name or alias
            
        Returns:
            Tool metadata
            
        Raises:
            KeyError: If tool not found
        """
        # Check if it's an alias
        if name in self._aliases:
            name = self._aliases[name]

        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")

        return self._tools[name].metadata

    def get_registry_info(self) -> Dict[str, Any]:
        """Get registry information.
        
        Returns:
            Dictionary with registry statistics and info
        """
        categories = list(set(entry.category for entry in self._tools.values()))

        return {
            "total_tools": len(self._tools),
            "categories": categories,
            "aliases": len(self._aliases),
            "tools_by_category": {
                cat: [name for name, entry in self._tools.items() if entry.category == cat]
                for cat in categories
            }
        }

    async def execute_todo(self, todo: TodoItem, context: ToolExecutionContext) -> ToolResult:
        """Execute a todo item using its assigned tool.

        Args:
            todo: Todo item with assigned_tool and task description
            context: Execution context

        Returns:
            Tool execution result

        Raises:
            KeyError: If assigned tool not found
            ValueError: If todo has no assigned tool
            ToolPermissionError: If agent lacks permission to use the tool
        """
        if not hasattr(todo, 'assigned_tool') or not todo.assigned_tool:
            raise ValueError(f"Todo {getattr(todo, 'id', 'unknown')} has no assigned tool")

        # Import here to avoid circular dependency (tool_role_manager imports tool_registry)
        from .tool_role_manager import ToolPermissionError, tool_role_manager

        # Validate tool access permissions
        try:
            tool_role_manager.validate_tool_access(context.agent_role, todo.assigned_tool)
        except Exception as e:
            if isinstance(e, ToolPermissionError):
                raise
            # Re-raise other exceptions
            raise

        factory = self.get(todo.assigned_tool)
        tool = factory()
        return await tool.execute(todo, context)

    def list_tools(self) -> List[str]:
        """List all registered tool names.
        
        Convenience method that calls list() with no filters.
        
        Returns:
            List of tool names
        """
        return self.list()

    def list_tools_for_role(self, agent_role: Optional[str]) -> List[str]:
        """List tools available for a specific agent role.

        Args:
            agent_role: Agent role string to filter tools for

        Returns:
            List of tool names accessible to the agent role
        """
        # Import here to avoid circular dependency
        from .tool_role_manager import tool_role_manager
        return tool_role_manager.get_allowed_tools(agent_role)


# Global registry instance (following flowlib pattern)
tool_registry = ToolRegistry()
