"""
Task execution interface definitions.

This module defines the protocols and interfaces for tool execution operations.
These are core interfaces, not agent component interfaces, so they should remain.
"""

from typing import Protocol, runtime_checkable

from ..core.todo import TodoItem
from .models import ToolExecutionContext, ToolParameters, ToolResult


@runtime_checkable
class ToolInterface(Protocol):
    """Interface for tool implementations.

    Defines the contract that all tools must follow.
    """

    async def execute(
        self, todo: TodoItem, parameters: ToolParameters, context: ToolExecutionContext
    ) -> ToolResult:
        """Execute the tool with given task and parameters.

        Args:
            todo: TodoItem describing what needs to be done (natural language task)
            parameters: Structured tool parameters (how to do it)
            context: Execution context

        Returns:
            Tool execution result
        """
        ...


@runtime_checkable
class ToolFactory(Protocol):
    """Interface for tool factories.

    Defines the contract for creating tool instances.
    """

    def __call__(self) -> ToolInterface:
        """Create a tool instance.

        Returns:
            Tool instance
        """
        ...

    def get_description(self) -> str:
        """Get description of the tool.

        Returns:
            Tool description
        """
        ...


@runtime_checkable
class AgentToolInterface(Protocol):
    """Interface for agent tools.

    Tools receive both TodoItem (WHAT to do) and ToolParameters (HOW to do it).
    Parameters are extracted from TodoItem context before tool execution.
    """

    async def execute(
        self, todo: TodoItem, parameters: ToolParameters, context: ToolExecutionContext
    ) -> ToolResult:
        """Execute tool with task description and validated parameters.

        Args:
            todo: TodoItem describing what needs to be done (natural language task)
            parameters: Validated ToolParameters subclass instance (how to do it)
            context: Execution context

        Returns:
            Tool result
        """
        ...


@runtime_checkable
class AgentToolFactory(Protocol):
    """Interface for agent tool factories that create AgentToolInterface instances."""

    def __call__(self) -> AgentToolInterface:
        """Create an agent tool instance.

        Returns:
            Agent tool instance
        """
        ...

    def get_description(self) -> str:
        """Get full description of the tool.

        Returns:
            Tool description
        """
        ...

    def get_planning_description(self) -> str:
        """Get planning description for prompts.

        Returns:
            Planning description (defaults to full description if not provided)
        """
        ...


@runtime_checkable
class ParameterFactoryInterface(Protocol):
    """Interface for parameter factories."""

    def create_parameters(self, todo: TodoItem, context: ToolExecutionContext) -> ToolParameters:
        """Create parameters from TODO item.

        Args:
            todo: TODO item
            context: Execution context

        Returns:
            Tool parameters
        """
        ...
