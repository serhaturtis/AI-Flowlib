"""Agent task execution system.

This module provides clean tool execution without decomposition concerns.
Following flowlib's single source of truth and no backward compatibility principles.

Architecture:
- ToolOrchestrator: Single orchestration approach (handles parameter generation)
- Tool Registry: Discovery and instantiation
- Protocol interfaces: Clean contracts
- @tool decorator: Enforces parameter_type for type safety

Usage:
    from flowlib.agent.components.task.execution import tool_registry, ToolExecutionContext
    from flowlib.agent.components.task.core.todo import TodoItem

    # Get tool and metadata
    factory = tool_registry.get("bash")
    metadata = tool_registry.get_metadata("bash")

    # Create TodoItem (what to do)
    todo = TodoItem(content="List files in current directory")

    # Create parameters (how to do it - validated by Pydantic)
    parameters = metadata.parameter_type(command="ls -la")

    # Execute tool with TodoItem, validated parameters, and context
    tool_instance = factory()
    result = await tool_instance.execute(todo, parameters, context)
"""

# Core system
# Import tool implementations to trigger registration
from . import tool_implementations
from .decorators import tool

# Import interfaces
from .interfaces import AgentToolFactory, AgentToolInterface, ToolFactory, ToolInterface
from .models import (
    AgentTaskRequest,
    TaskExecutionResult,
    TodoExecutionContext,
    TodoExecutionResult,
    ToolExecutionContext,
    ToolExecutionError,
    ToolMetadata,
    ToolParameters,
    ToolResult,
    ToolStatus,
)

# Tool orchestration
from .orchestration import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolOrchestrator,
    tool_orchestrator,
)
from .registry import ToolRegistry, tool_registry

__all__ = [
    # Tool implementations (imported for registration)
    "tool_implementations",
    # Core models
    "ToolParameters",
    "ToolResult",
    "ToolExecutionContext",
    "ToolStatus",
    "ToolMetadata",
    "ToolExecutionError",
    "AgentTaskRequest",
    "TaskExecutionResult",
    "TodoExecutionContext",
    "TodoExecutionResult",
    # Core interfaces
    "ToolInterface",
    "ToolFactory",
    "AgentToolInterface",
    "AgentToolFactory",
    # Registry system
    "ToolRegistry",
    "tool_registry",
    # Decorator
    "tool",
    # Orchestration
    "ToolOrchestrator",
    "tool_orchestrator",
    "ToolExecutionRequest",
    "ToolExecutionResponse",
]
