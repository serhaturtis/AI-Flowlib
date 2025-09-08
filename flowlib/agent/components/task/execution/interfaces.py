"""
Task execution interface definitions.

This module defines the protocols and interfaces for tool execution operations.
These are core interfaces, not agent component interfaces, so they should remain.
"""

from typing import Any, Dict, Optional, Protocol, runtime_checkable
from .models import ToolParameters, ToolResult, ToolExecutionContext, ToolMetadata


@runtime_checkable
class ToolInterface(Protocol):
    """Interface for tool implementations.
    
    Defines the contract that all tools must follow.
    """
    
    async def execute(self, parameters: ToolParameters, context: ToolExecutionContext) -> ToolResult:
        """Execute the tool with given parameters.
        
        Args:
            parameters: Tool parameters
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


@runtime_checkable
class AgentToolInterface(Protocol):
    """Interface for agent tools."""
    
    async def execute(self, todo: Any, context: ToolExecutionContext) -> ToolResult:
        """Execute tool with TODO item.
        
        Args:
            todo: TODO item to execute
            context: Execution context
            
        Returns:
            Tool result
        """
        ...


@runtime_checkable
class ParameterFactoryInterface(Protocol):
    """Interface for parameter factories."""
    
    def create_parameters(self, todo: Any, context: ToolExecutionContext) -> ToolParameters:
        """Create parameters from TODO item.
        
        Args:
            todo: TODO item
            context: Execution context
            
        Returns:
            Tool parameters
        """
        ...


